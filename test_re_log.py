
import re
import matplotlib.pyplot as plt

# 日志文件路径（替换为实际路径）
note = "rl_activecam"
reward =    "episode mean reward"       # "vsqf reward" #    #  "episode mean reward" #
log_file_path = "/home/wpp/Seman-Curiosity/rl_activecam.log"
# "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/rl_vsqf_v3_1.log"
# "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/logs/expv7.log"
# "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/rl_vsqf_v3_0.log"

# 读取文件内容
with open(log_file_path, 'r') as file:
    log_lines = file.readlines()

# 提取 episode mean reward 的值
rewards = []
for line in log_lines:
    match = re.search(rf'{reward}=(-?\d+\.\d+)', line)
    if match:
        r_ = float(match.group(1))
        # if r_>0:
        rewards.append(r_)

# 检查是否提取到数据
if not rewards:
    
    print(f"未找到 {reward} 数据！")
else:
    print(len(rewards))
    # 绘制曲线
    plt.plot(rewards, label=f'{reward}')
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward')
    plt.title(f'{reward} Over Time')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(f'{note}_{reward}.png')  # 保存图像