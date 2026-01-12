import re
import matplotlib.pyplot as plt 
import numpy as np


def extract_rew_values(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    rew_values = re.findall(r'rew:\s*([0-9]*\.?[0-9]+)', content)
    return [float(value) for value in rew_values]

def extract_policy_loss_values(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    pattern = r'Policy Loss value/action/dist:\s*([0-9]*\.?[0-9]+)/([0-9]*\.?[0-9]+)/([0-9]*\.?[0-9]+)'
    matches = re.findall(pattern, content)
    
    value_loss_values = []
    action_loss_values = []
    dist_loss_values = []
    for match in matches:
        value_loss_values.append(float(match[0]))
        action_loss_values.append(float(match[1]))
        dist_loss_values.append(float(match[2]))
    
    return value_loss_values, action_loss_values, dist_loss_values

def load_npz(file_path, key):
    data = np.load(file_path)
    data = data[key]
    return data

def plot_curve(y, title, xlabel, save_path="t.png"):
        # x = range(len(y))
        plt.plot(y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.savefig(save_path)
        
if __name__ == "__main__":
    dir_path = '/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/models/expv1_500016/'
    file_path = dir_path + 'train.log'
    value_ls, action_ls, dist_ls = extract_policy_loss_values(file_path)
    print(value_ls)
    y_value = value_ls
    y_action = action_ls
    y_dist = dist_ls
    
    dir_path_2 = '/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/models/expv1_5500008/'
    file_path_2 = dir_path_2 + 'train.log'
    value_ls2, action_ls2, dist_ls2 = extract_policy_loss_values(file_path_2)
    y_value.extend(value_ls2)
    y_action.extend(action_ls2)
    y_dist.extend(dist_ls2)    
    
    title="Training policy loss"
    xlabel="steps (*240)"
    # plot_curve(y_value, "Training value loss", xlabel, "value_losses.png")
    # plot_curve(y_action, "Training action loss", xlabel, "action_losses.png")
    # plot_curve(y_dist, "Training dist loss", xlabel, "dist_losses.png")
    
    p = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/expv1_5500008/train_episode_rewards.npz"
    y = load_npz(p, "episode_reward")
    # plot_curve(y, "Training Episode Rewards(second part)", "episodes", "train_episode_rewards.png")
    
    p1 = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/expv1_5500008_eval/val_episode_rewards.npz"
    y = load_npz(p1, "episode_reward").mean()
    print(f"rl mean: {y}")
    
    p2 = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/random/val_episode_rewards.npz"
    y = load_npz(p2, "episode_reward").mean()
    print(f"random mean: {y}")
    
    