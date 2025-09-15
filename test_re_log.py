
import re
import matplotlib.pyplot as plt

# 日志文件路径（替换为实际路径）
log_file_path = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/logs/expv7.log"
# "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/rl_vsqf_v3_0.log"

# 读取文件内容
with open(log_file_path, 'r') as file:
    log_lines = file.readlines()

# 提取 episode mean reward 的值
rewards = []
for line in log_lines:
    match = re.search(r'episode mean reward=(\d+\.\d+)', line)
    if match:
        rewards.append(float(match.group(1)))

# 检查是否提取到数据
if not rewards:
    print("未找到 episode mean reward 数据！")
else:
    # 绘制曲线
    plt.plot(rewards, label='Episode Mean Reward')
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward')
    plt.title('Episode Mean Reward Over Time')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig('episode_mean_reward_curi_expv7.png')  # 保存图像