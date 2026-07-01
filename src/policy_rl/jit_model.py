import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch import Tensor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.policy_rl.model import RL_Policy2  # noqa: E402


class Discrete:
    def __init__(self, n: int):
        self.n = n


class ExportPolicyActWrapper(nn.Module):
    """
    Minimal export wrapper that only keeps tensors required for inference.
    This avoids tracing/saving auxiliary methods with Optional Tensor defaults.
    """

    def __init__(self, policy_module: RL_Policy2, deterministic: bool = True):
        super().__init__()
        self.network = policy_module.network
        self.dist = policy_module.dist
        self.deterministic = deterministic

    def forward(self, obs: Tensor):
        value, act_feature = self.network(obs)
        dist = self.dist(act_feature)
        if self.deterministic:
            action = dist.mode().reshape(-1)
        else:
            action = dist.sample()
        action_log_probs = dist.log_probs(action)
        return value, action, action_log_probs, dist.probs
        # fts = self.network.encoder(obs)
        # return fts


# def _load_checkpoint(policy: RL_Policy2, checkpoint_path: str, device: str) -> None:
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
#         checkpoint = checkpoint["state_dict"]

#     try:
#         policy.load_state_dict(checkpoint)
#         return
#     except RuntimeError:
#         if isinstance(checkpoint, dict):
#             stripped = {}
#             for k, v in checkpoint.items():
#                 if k.startswith("module."):
#                     stripped[k[len("module."):]] = v
#                 else:
#                     stripped[k] = v
#             policy.load_state_dict(stripped)
#             return
#         raise


def _load_checkpoint_into_policy(policy, checkpoint_path):
    print("Loading model {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    if isinstance(checkpoint, dict):
        if 'policy_state_dict' in checkpoint:
            policy.load_state_dict(checkpoint['policy_state_dict'])
        else:
            policy.load_state_dict(checkpoint)
    else:
        policy.load_state_dict(checkpoint)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RL policy to TorchScript")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "exps/models/sequence6-6-1/model_best.pth"),
        help="Path to policy checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "deployed_rl_policy_v7_tmp.pt"),
        help="Output path for traced TorchScript model",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Dummy input batch size")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Export device")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Export deterministic action selection (mode) instead of sampling",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    obs_shape = (12, 256, 256, 3)
    action_space = Discrete(12)

    print(f"obs shape: {obs_shape}")

    policy = RL_Policy2(obs_shape, action_space, device=device, use_history=False)
    # _load_checkpoint(policy, args.checkpoint, device)
    _load_checkpoint_into_policy(policy, args.checkpoint)
    policy.eval()

    bs = args.batch_size
    obs = torch.randint(0, 256, (bs, 12, 256, 256, 3), device=device).float()
    dummy_extras = torch.zeros((bs, 1), device=device)
    dummy_deterministic = torch.tensor(False, dtype=torch.bool, device=device)
    dummy_curr_pano_img_feats = torch.randn((bs, 12), device=device)
    dummy_curr_pano_ang_feats = torch.randn((bs, 4), device=device)
    dummy_hist_pano_img_feats = torch.zeros((bs, 10, 12), device=device)
    dummy_hist_pano_ang_feats = torch.zeros((bs, 10, 4), device=device)
    dummy_hist_actions = torch.zeros((bs, 10), dtype=torch.long, device=device)
    dummy_hist_masks = torch.ones((bs, 10), device=device)
    dummy_compute_hist_embed = torch.tensor(False, dtype=torch.bool, device=device)

    value, action, action_log_probs, probs= policy.act(
        obs,
        dummy_extras,
        dummy_deterministic,
        dummy_curr_pano_img_feats,
        dummy_curr_pano_ang_feats,
        dummy_hist_pano_img_feats,
        dummy_hist_pano_ang_feats,
        dummy_hist_actions,
        dummy_hist_masks,
        dummy_compute_hist_embed,
    )
    print(
        "pre-check success: "
        f"value={tuple(value.shape)}, "
        f"action={tuple(action.shape)}, "
        f"action_log_probs={action_log_probs}",
        
    )

    export_wrapper = ExportPolicyActWrapper(policy, deterministic=args.deterministic)
    export_wrapper.eval()
    
    torch.manual_seed(1)  # For reproducibility of the random input    
    obs1 = torch.randint(0, 256, (bs, 12, 256, 256, 3), device=device).float()
    _, _, _, probs = export_wrapper(obs1)  # Run once to trace the graph and catch any issues before saving
    # fts1 = export_wrapper(obs1)
    print(f"pre-trace success:e probs={probs}")
    
    torch.manual_seed(123)  # For reproducibility of the random input    
    obs2 = torch.randint(0, 256, (bs, 12, 256, 256, 3), device=device).float()
    _, _, _, probs = export_wrapper(obs2)  # Run once to trace the graph and catch any issues before saving
    # fts2 = export_wrapper(obs2)
    print(f"pre-trace success 2: probs={probs}")
    
    
    traced_policy_act = torch.jit.trace(export_wrapper, obs, check_trace=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced_policy_act.save(str(output_path))

    print(f"export success: {output_path}")
    
    path = "/home/wpp/Semantic-Curiosity/Semantic-Curiosity//deployed_rl_policy_v7_tmp.pt"
    act_predictor = torch.jit.load(str(path), map_location=device)
    act_predictor.eval()
    
    with torch.no_grad():
        # value, action, action_log_prob = act_predictor(obs)
        value, action, action_log_prob, probs  = act_predictor(obs1)
        # af = act_predictor(obs1)
        print(f"weights success: probs={probs}")
        
        value, action, action_log_prob, probs  = act_predictor(obs2)
        # af = act_predictor(obs2)
        print(f"weights success: probs={probs}")
    


if __name__ == "__main__":
    main()
