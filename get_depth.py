from src.finetune.dataset_utils import get_loader, SampleLoader
import cv2
base_dir = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/exp_obns_v2_eval_best_sample/"
# "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/exp_obns_eval_best_sample/"
# "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/expv7_eval_best2/"

dataset_path = base_dir + "/episodes_data"
env, epi, step = 0,1,5
sampler = SampleLoader(dataset_path)
depth = sampler.get_sample(env, epi, step, "depth")
cv2.imwrite("./t_depth.png", depth.data.squeeze(-1)*255)
print(depth.data.shape)
