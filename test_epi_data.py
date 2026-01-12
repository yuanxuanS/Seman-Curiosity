import gzip
import json

episodes_file = "/home/wpp/Seman-Curiosity/data/datasets/objectnav/gibson/v1.1/val/content_activecam/Collierville_episodes.json.gz"
with gzip.open(episodes_file, 'r') as f:
    eps_data = json.loads(
        f.read().decode('utf-8'))
eps_data = eps_data['episodes']
print(eps_data[0])
print(eps_data[-1])