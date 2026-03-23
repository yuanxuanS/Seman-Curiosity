
from src.policy_rl.envs import make_vec_envs
from src.policy_rl.arguments import get_args



if __name__ == "__main__":
    args = get_args()
    
    envs = make_vec_envs(args)      
    obs_all, infos = envs.reset()   # obs: rgb +depth + categories 16 TODO: ?

    observation_space = envs.get_obs_space()[0]
    
    # observations
    obs = 
    batch_size = 1      # 环境数
    pred_wp_heatmap(obs, observation_space, batch_size)