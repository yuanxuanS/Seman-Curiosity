import gzip
import json

episodes_file = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/data_scene/datasets/objectnav/gibson/v1.1/val/content/Collierville_episodes.json.gz"
print("Loading episodes from: {}".format(episodes_file))
with gzip.open(episodes_file, 'r') as f:
    eps_data = json.loads(
        f.read().decode('utf-8'))["episodes"]

print(len(eps_data))