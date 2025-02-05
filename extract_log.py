import re
import matplotlib.pyplot as plt
# 定义日志文件路径
log_file_path = '/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exp_obns.log'

# 打开日志文件并读取内容
with open(log_file_path, 'r') as file:
    log_content = file.readlines()

# 定义正则表达式模式来匹配mean值
pattern = re.compile(r'\s*per step Potential reward, mean/med/min/max:\s*([\d\.\-]+)/')

# 提取所有mean值
mean_values = []
for line in log_content:
    match = pattern.search(line)
    if match:
        mean_values.append(float(match.group(1)))

print(f"length: {len(mean_values)}")
# 输出所有mean值
plt.plot(mean_values)
plt.savefig("./mean_potential_re.png")