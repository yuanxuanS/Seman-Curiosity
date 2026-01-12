import pickle
import numpy as np

file = '/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/vis_maps_v2/Allensville_vismap.txt'
with open(file, 'rb') as f:
    data = pickle.load(f)
    
map= data['Allensville'][0][0][1]
print(np.where(map>0))