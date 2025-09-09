from third_parties.Orient_Anything.vision_tower import DINOv2_MLP
import torch
from transformers import AutoImageProcessor


def OriAny_pred(device):
  # Orient anything
  ckpt_path = "./third_parties/Orient_Anything/models/largeEx2/dino_weight.pt"


  dino = DINOv2_MLP(
                      dino_mode   = 'large',
                      in_dim      = 1024,
                      out_dim     = 360+180+180+2,
                      evaluate    = True,
                      mask_dino   = False,
                      frozen_back = False
                  )
  dino.eval()
  print('model create')
  dino.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
  dino = dino.to(device)
  print('weight loaded')

  val_preprocess   = AutoImageProcessor.from_pretrained("/home/wpp/Seman-Curiosity/third_parties/Orient_Anything/models/dino/", local_files_only=True)
  return dino, val_preprocess