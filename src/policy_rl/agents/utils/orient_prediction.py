from test_gt_orient import OriAny_pred
from third_parties.Orient_Anything.inference import get_3angle, get_3angle_infer_aug, inference, preprocess
from third_parties.Orient_Anything.utils import background_preprocess
import torch

class Orient_pred():
    def __init__(self, args):
        if args.sem_gpu_id == -2:
            self.device = "cpu"
        else:
            self.device = "cuda:{}".format(args.sem_gpu_id)

        self.dino, self.val_preprocess = OriAny_pred(self.device)
        self.augment = False
        
    def pred_orient(self, x):
        '''
        x: image of Image type
        '''
        if self.augment:
            rm_bkg_img = background_preprocess(x, True)
            angles = get_3angle_infer_aug(x, rm_bkg_img, self.dino, self.val_preprocess, self.device)
        else:
            angles = get_3angle(x, self.dino, self.val_preprocess, self.device)
        
        azimuth     = float(angles[0])
        polar       = float(angles[1])
        rotation    = float(angles[2])
        confidence  = float(angles[3])
        return azimuth, confidence

    def pred_orient_multi(self, x_lst):
        '''
        x_lst: list of Image
        return:
            ax: azimuths of images
            pl: polar of images
            ro: rotation of images
            confi: confidences of images
        '''
        imgs_inputs = preprocess(x_lst, self.val_preprocess, self.device)
        ax, pl, ro, confi = inference(imgs_inputs, self.dino)
        return ax.type(torch.float32), confi

if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    img = np.ones((256, 256, 3))
    img2 = np.ones((256, 256, 3))
    device = "cuda:1"
    
    img_lst = [Image.fromarray(img.astype(np.uint8)), 
               Image.fromarray(img2.astype(np.uint8)),]
    
    dino, val_preprocess = OriAny_pred(device)
    imgs_inputs = preprocess(img_lst, val_preprocess, device)
    print(imgs_inputs['pixel_values'].shape)
    ax, pl, ro, confi = inference(imgs_inputs, dino)
    print(ax, pl, ro)
    print(confi, confi.shape)
    