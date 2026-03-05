import pickle

with open("./asample_straight_sampled.pkl", "rb") as f:
    data = pickle.load(f)

print(data)