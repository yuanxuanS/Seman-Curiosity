import argparse
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import cv2

import numpy as np
import json
import torch
from PIL import Image
import pycocotools.mask as mask_util
from detectron2.structures.boxes import BoxMode

# sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
# sys.path.append(os.path.join(os.getcwd(), "segment_anything"))
# print(os.getcwd())

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
import gc

# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def image_from_numpy(img):
    '''
    img: numpy.array, h, w, c
    '''
    image_pil = Image.fromarray(img).convert("RGB")
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label, color="white")


def save_mask_data(output_dir, mask_list, box_list, label_list, env_epi_step):
    env, episode, step = env_epi_step
    img_pth = output_dir
    
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(img_pth, "epi"+str(episode)+"_env"+str(env) + "_step"+str(step)+'_mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(img_pth, "epi"+str(episode)+"_env"+str(env) + "_step"+str(step)+'_mask.json'), 'w') as f:
        json.dump(json_data, f)
    plt.clf()
    plt.close('all')
    gc.collect()
def init_segment(args):
    '''
    args: dict
    rgb : numpy.array, h,w,c
    '''
    # cfg
    config_file = args['config']  # change the path of the model config file
    grounded_checkpoint = args['grounded_checkpoint']  # change the path of the model
    sam_version = args['sam_version']
    sam_checkpoint = args['sam_checkpoint']
    sam_hq_checkpoint = args['sam_hq_checkpoint']
    use_sam_hq = args['use_sam_hq']
    # image_path = args['input_image']
    output_dir = args['output_dir']
    device = args['device']
    bert_base_uncased_path = args['bert_base_uncased_path']


    # make dir
    os.makedirs(output_dir, exist_ok=True)

    # print(image.shape)
    # print(image_pil.size)
    # load model
    model = load_model(config_file, grounded_checkpoint, bert_base_uncased_path, device=device)
    
    # initialize SAM
    if use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    
    return model, predictor

def pred_mask(predictor, boxes, image, device):
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes, image.shape[:2]).to(device)
    # print(size)
    # print(boxes_filt)
    # print(pred_phrases)
    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )
    return masks
def pred_segment(args, rgb, model, predictor, env_epi_step, return_anno=False, return_per=True, save=False):
    '''
    args: dict
    rgb : numpy.array, h,w,c
    '''
    # cfg
    config_file = args['config']  # change the path of the model config file
    grounded_checkpoint = args['grounded_checkpoint']  # change the path of the model
    sam_version = args['sam_version']
    sam_checkpoint = args['sam_checkpoint']
    sam_hq_checkpoint = args['sam_hq_checkpoint']
    use_sam_hq = args['use_sam_hq']
    # image_path = args['input_image']
    text_prompt = args['text_prompt']
    output_dir = args['output_dir']
    box_threshold = args['box_threshold']
    text_threshold = args['text_threshold']
    device = args['device']
    bert_base_uncased_path = args['bert_base_uncased_path']
    
    # make dir
    if save:
        os.makedirs(output_dir, exist_ok=True)
    
    # visualize raw image
    # image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    image_pil, image = image_from_numpy(rgb)
    
    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, device=device
    )

    if len(boxes_filt) == 0:
        if return_anno:
            return []
        elif return_per:
            return [], boxes_filt, pred_phrases
        else:
            return None
    image = rgb #cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size       # 读入图像的大小
    H, W = size[1], size[0]
    # print("before:", boxes_filt)        # x center, y center, width ,height
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2  # 减去width, height的一半
        boxes_filt[i][2:] += boxes_filt[i][:2]  # # 加上width, height的一半
    # print("after:", boxes_filt)
    boxes_filt = boxes_filt.cpu()   # resize为图像大小范围
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
    # print(size)
    # print(boxes_filt)
    # print(pred_phrases)
    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )

    # print(masks.shape)
    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)

    env, episode, step = env_epi_step

    if save:
        plt.axis('off')
        plt.savefig(
            os.path.join(output_dir, "epi"+str(episode)+"_env"+str(env) + "_step"+str(step)+"_gsam_output.jpg"),
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )

        save_mask_data(output_dir, masks, boxes_filt, pred_phrases, env_epi_step)

        plt.clf()
        plt.close('all')
        gc.collect()
    if return_anno:
        annotations = convert_to_ann(masks, boxes_filt, pred_phrases, env_epi_step)
        return annotations
    elif return_per:
        return masks, boxes_filt, pred_phrases
    else:
        return None
def convert_to_ann(masks, boxes, pred_phrases, env_epi_step):
    from src.constants import coco_categories
    class_map = { 
                "chair": 0,
                "couch": 1,
                "plant": 2,
                "bed": 3,
                "toilet": 4,}
    
    env, episode, step = env_epi_step
    annotations = []
    for mask, box, phrase in zip(masks, boxes, pred_phrases):
        # exclude class
        if "table" in phrase or "refrigerator" in phrase or "cupboard" in phrase or "painting" in phrase or "occluded" in phrase:
            continue
        for name, id in class_map.items():
            if name in phrase:
                class_label = id
                break
        
        gt_logits = torch.zeros(len(class_map)+1)   # last one is background
        gt_logits[class_label] = 1.
        
        box_ = box.tolist()
        box_ = [max(0, min(x, 639.9)) for x in box_]
        anno={
                    'bbox': box_,
                    'bbox_mode': BoxMode.XYXY_ABS,
                    'category_id': torch.tensor(class_label),
                    'segmentation': mask_util.encode(
                        np.asfortranarray(mask.squeeze(0).cpu().numpy())
                    ),
                    # TODO introduce 'uncertainties': y[id_instance].gt_uncertainty_masks[0],
                    'iscrowd': 0,
                    'infos': {"env": env, "episode": episode, "step": step, "id_object": 50000},    # 新物体的id_object都标记为50000， 只影响正负样本学习
                    'gt_logits': gt_logits,
                }
        
        annotations.append(anno)
        
    return annotations
def segment_args():
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", default=True, action="store_true", help="using sam-hq for prediction"
    )
    # parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    parser.add_argument("--bert_base_uncased_path", type=str, required=False, help="bert_base_uncased model path, default=False")
    args = parser.parse_args()
    return args
if __name__ == "__main__":

    args = segment_args()
    
    rgb = cv2.imread("/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/exp_obns_v2_eval_best_sample/episodes_data_orig_imgs/epi1_env0_step0.png")
#     print(rgb.shape)
    model, predictor = init_segment(args)
    pred_segment(args, rgb, model, predictor, (0,1,0))

    
