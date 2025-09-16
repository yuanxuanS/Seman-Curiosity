import numpy as np

def load_npz(file_path, key):
    data = np.load(file_path)
    data = data[key]
    return data

if __name__ == "__main__":
    pth = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/actions_Forkland.npz"
    data = load_npz(pth, "actions")
    
    per_line = 5
    for i in range(0, len(data), per_line):
        chunk = data[i:i+per_line]
        print(f"{i} : {chunk}")
        