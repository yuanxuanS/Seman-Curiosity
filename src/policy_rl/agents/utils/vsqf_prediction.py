from vqf import VQFModel
import torch
from PIL import Image
import numpy as np

class Vsqf_pred():
    def __init__(self, device, magnify=False, magnify_num=3.0):
        load_model = True
        model_pth = '/home/wpp/Seman-Curiosity/best_unseen_model_e1.pth'
        # "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/vqf_logs/08-29_17-25-03_/best_unseen_model_e1.pth"
        # if args.sem_gpu_id == -2:
        #     self.device = "cpu"
        # else:
        #     self.device = "cuda:{}".format(args.sem_gpu_id)
        self.device = device
        self.model = VQFModel(self.device)
        if load_model:
            self.model.load_state_dict(torch.load(model_pth, map_location=device))

        self.magnify = magnify
        self.magnify_num = magnify_num
        
    def preprocess(self, x_lst):
        '''
        x_lst: list of Images
        '''
        imgs = []
        for img_ in x_lst:
            # img_ = Image.fromarray(x_lst[i].astype(np.uint8))
            x = self.model.preprocess(img_)
            imgs.append(x.unsqueeze(0))
        imgs = torch.concat(imgs, dim=0)
        return imgs
    def inference(self, x):
        '''
        x: tensor, B*3*w*h
        '''
        x = x.to(self.device)
        with torch.no_grad():
            # x = self.model.preprocess(x).to(self.device)[None, ...]
            vsqf = self.model(x)
            
        if self.magnify:
            vsqf = self._magnify(vsqf)
        return vsqf
    
    def pred_vsqf(self, x):
        '''
        x: tensor, B*3*w*h
        '''
        with torch.no_grad():
            x = self.model.preprocess(x).to(self.device)[None, ...]
            vsqf = self.model(x)
            
        if self.magnify:
            vsqf = self._magnify(vsqf)
        return vsqf
    
    def _magnify(self, vsqf):
        '''
         归一化并扩大其值范围
        '''
        vsqf = (vsqf.squeeze(0) - vsqf.min() + 0.1) * self.magnify_num / (vsqf.max() - vsqf.min())
        return vsqf