import torch
import math
import gym
from src.policy_rl.utils.storage import GlobalRolloutStorage
import numpy as np
# 测试RL_policy打包模型的加载

bs = 1

path = "/home/wpp/Semantic-Curiosity/Semantic-Curiosity/src/policy_rl/deployed_rl_policy_curi.pt"
policy = torch.jit.load(str(path))
policy.eval()

map_size_cm = 2000
full_pose = torch.zeros(bs, 3).float()
full_pose[:, :2] = map_size_cm / 100.0 / 2.0     # full pose和local pose单位都是m

# Helper: apply discrete action to full_pose (in-place)
def apply_action_to_pose(full_pose_tensor, action_tensor, turn_angle_deg=30.0, step_size_m=0.25):
    """
    full_pose_tensor: Tensor [bs, 3] with columns [x(m), y(m), yaw_deg]
    action_tensor: Tensor [bs] with discrete actions mapping: 0=forward,1=turn_left,2=turn_right
    Updates full_pose_tensor in-place and returns it.
    """
    if not torch.is_tensor(action_tensor):
        action_tensor = torch.tensor(action_tensor)

    action_tensor = action_tensor.to(full_pose_tensor.device)

    # forward mask
    move_mask = (action_tensor == 0)
    left_mask = (action_tensor == 1)
    right_mask = (action_tensor == 2)

    # compute forward displacement only for move steps
    theta_rad = full_pose_tensor[:, 2] * math.pi / 180.0
    dx = step_size_m * torch.cos(theta_rad)
    dy = step_size_m * torch.sin(theta_rad)

    full_pose_tensor[:, 0] = full_pose_tensor[:, 0] + dx * move_mask.to(dtype=full_pose_tensor.dtype)
    full_pose_tensor[:, 1] = full_pose_tensor[:, 1] + dy * move_mask.to(dtype=full_pose_tensor.dtype)

    # update yaw
    full_pose_tensor[:, 2] = full_pose_tensor[:, 2] + turn_angle_deg * left_mask.to(dtype=full_pose_tensor.dtype)
    full_pose_tensor[:, 2] = full_pose_tensor[:, 2] - turn_angle_deg * right_mask.to(dtype=full_pose_tensor.dtype)

    # normalize to [-180, 180)
    full_pose_tensor[:, 2] = torch.fmod(full_pose_tensor[:, 2] + 180.0, 360.0) - 180.0

    return full_pose_tensor

es = 3
observation_space = gym.spaces.Box(0, 255,
                                            (3, 256,
                                            256),
                                            dtype='uint8')
action_space = gym.spaces.Discrete(3)
l_rollouts = GlobalRolloutStorage(100,
                                bs, observation_space.shape,
                                action_space, 256,
                                es)
torch.manual_seed(56)
# use float32 random input to match model expectations (avoid uint8/byte dtype issues)
input = torch.rand(bs, 3, 256, 256, dtype=torch.float32) * 255.0
l_rollouts.obs[0].copy_(input)   

extras = torch.zeros(bs, es)
local_orientation = torch.zeros(bs, 1).long()
locs = full_pose.cpu().numpy()      # 使用全局pose
local_orientation[0] = int((locs[0, 2] + 180.0) / 5.)
local_xy = torch.zeros(bs, 2)
local_xy[0] = torch.from_numpy(locs[0, :2][np.newaxis, :])
extras[:, 2] = local_orientation[:, 0]
extras[:, :2] = local_xy[:]
# extras = torch.randint(low=0, high=10, size=(bs, es))
l_rollouts.extras[0].copy_(extras)
with torch.no_grad():
    value, action, action_log_prob, rec_states  = policy.act(
        l_rollouts.obs[0],
        l_rollouts.rec_states[0],
        l_rollouts.masks[0],
        extras=l_rollouts.extras[0],
        deterministic=torch.tensor(False, dtype=torch.bool)
    )
    print(f"✅ 加载的 JIT 模型 act 函数执行成功，输出如下：",
        f"value: {value}, action: {action}, action_log_prob: {action_log_prob}")


reward = torch.rand(bs)  # placeholder reward
l_masks = torch.ones(bs).float()    
for step in range(500):
    l_step = step % 100
    
    input = torch.rand(bs, 3, 256, 256, dtype=torch.float32) * 255.0
    # --- Apply discrete action to our internal full_pose estimate ---
    try:
        # action may be a tensor of shape [bs]
        full_pose = apply_action_to_pose(full_pose, action)

        # build extras for next time-step: [x, y, orientation_index]
        # orientation index: map yaw_deg in [-180,180) to [0,71] with 5deg bins
        orient_idx = int((full_pose[0, 2].item() + 180.0) / 5.0)
        orient_idx = max(0, min(71, orient_idx))
        extras_tensor = torch.zeros((bs, es), dtype=torch.float32)
        extras_tensor[:, :2] = full_pose[:, :2]
        extras_tensor[:, 2] = orient_idx

        # write into rollout extras for the next observation used by policy.act
        l_rollouts.extras[l_step + 1].copy_(extras_tensor)
    except Exception:
        # non-fatal: keep running even if pose update fails
        pass
    l_rollouts.insert(
            input, rec_states,      # state_t+1
            action, action_log_prob, value,   # action, reward_t
            reward, l_masks, extras
        )
    with torch.no_grad():
        value, action, action_log_prob, rec_states  = policy.act(
            l_rollouts.obs[l_step + 1],
            l_rollouts.rec_states[l_step + 1],
            l_rollouts.masks[l_step + 1],
            extras=l_rollouts.extras[l_step + 1],
            deterministic=torch.tensor(False, dtype=torch.bool)
        )
        print(f"✅ 输出如下：",
            f"value: {value}, action: {action}, action_log_prob: {action_log_prob}")

    
    reward = torch.rand(bs)
    l_masks = torch.ones(bs).float()
    
    if l_step == 100 - 1:
        l_rollouts.after_update() 