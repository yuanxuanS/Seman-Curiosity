from pathlib import Path
from typing import Tuple

import torch


def run_saved_policy(
    model_path: str = "./deployed_rl_policy_v4.pt",
    batch_size: int = 1,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a saved TorchScript policy and run one forward pass.

    Returns:
        value, action, action_log_prob
    """
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"TorchScript model not found: {model_path_obj}")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    act_predictor = torch.jit.load(str(model_path_obj), map_location=device)
    act_predictor.eval()

    torch.manual_seed(1)  # For reproducibility of the random input
    obs = torch.randint(0, 256, (batch_size, 12, 256, 256, 3), device=device).float()

    with torch.no_grad():
        # value, action, action_log_prob = act_predictor(obs)
        value, action, action_log_prob, probs, af = act_predictor(obs)

    return value, action, action_log_prob, probs, af


if __name__ == "__main__":
    value, action, action_log_prob, probs, af = run_saved_policy()
    print(
        "TorchScript policy call success: "
        f"value_shape={tuple(value.shape)}, "
        f"action={action.tolist()}, "
        f"action_log_prob_shape={tuple(action_log_prob.shape)}",
        f"action_log_prob={probs}",
        f"action_feature={af[0, :20]}",
    )