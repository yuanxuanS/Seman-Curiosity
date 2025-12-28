import torch
f = './output_al/retinanet_coco_ppal_5rounds_2percent_to_10percent/round1/model_final.pth'
model = torch.load(f, map_location='cpu')
print(model['model']['class_quality'].shape)
