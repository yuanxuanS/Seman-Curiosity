import os
import re
from src.policy_rl.arguments import get_args
from src.policy_rl.agents.utils.semantic_prediction import SemanticPredMaskRCNN as SemanticPredMaskRCNN
import cv2
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import clip
from PIL import Image

def draw_array(obj_str, obj_value_dict, note, norm=False):
    save_dir = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/images/value_map/"+note
    os.makedirs(save_dir, exist_ok=True)
    
    dis = ['2m', '2.5m', '3m', '3.5m', '4m', '4.5m', '5m', '5.5m']
    matrix = []
    matrix_min, matrix_max = 1e5, 0
    for d in dis:
        values = obj_value_dict[d][::10]
        matrix_min = min(values) if min(values) < matrix_min else matrix_min
        matrix_max = max(values) if max(values) > matrix_max else matrix_max
        matrix.append(values)
        
    matrix = np.array(matrix)
    if norm:
        matrix = (matrix - matrix_min) / (matrix_max - matrix_min)
        # 取负值
        matrix = np.ones_like(matrix) - matrix

    
    # 绘制热力图
    plt.figure(figsize=(18, 3))  # 宽度调大以适应360列
    im = plt.imshow(matrix, 
                    cmap='hot',  # 颜色映射
                    aspect='auto',   # 自动调整纵横比防止变形
                    interpolation='none')  # 禁用插值保持清晰边界
    plt.colorbar(im, label='Value')  # 添加颜色条
    plt.title('value Heatmap')
    plt.xlabel('azimuth (0-359)')
    plt.ylabel('distance')
    plt.savefig(save_dir+"/"+obj_str+".png", dpi=300, bbox_inches='tight')  # 保存高清图
    plt.show()

def manual_bin_count(data, bin_size=0.1):
    """
    手动分箱统计（适合小数据集）
    :param bin_size: 区间宽度（默认0.1）
    :return: 字典{区间: 占比}
    """
    bins = {}
    total = len(data)
    
    # 初始化区间计数器
    for i in range(0, 10):
        lower = i * bin_size
        upper = (i+1) * bin_size
        bins[f"{lower:.1f}-{upper:.1f}"] = 0
    
    # 统计分布
    for num in data:
        if num != None:
            idx = min(int(num / bin_size), 9)  # 处理1.0的边界
            key = list(bins.keys())[idx]
            bins[key] += 1
    
    # 计算百分比
    return {k: v  for k, v in bins.items()}

if __name__ == "__main__":

    mode = "read"   #"predict"  #
    predict_mode = "clip"       # "maskrcnn"
    class_name = "refrigerator"
    preds = {}
    values = {}
    cnt_none = {}
    cnt_false = {}
    
    if mode == "predict":
        if predict_mode == "maskrcnn":
            args = get_args()
            sem_pred = SemanticPredMaskRCNN(args)
            pred_instances = {}
        elif predict_mode == "clip":
            device = "cuda:3" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-L/14", device=device)

            text_str = [class_name]
            text = clip.tokenize([f"a photo contains a {i}" for i in text_str]).to(device)


        # 读取文件夹
        dir = '/data2/wpp_data/obj_azimuth/'+class_name

        tmp = 0        
        for obj_dir in os.listdir(dir):
            if predict_mode == "maskrcnn":
                pred_instances[obj_dir] = {}
                
                preds[obj_dir] = {}
                cnt_none[obj_dir] = {}
                cnt_false[obj_dir] = {}
            values[obj_dir] = {}
            
            for distance in os.listdir(dir + "/"+obj_dir):
                values[obj_dir][distance] = []
                if predict_mode == "maskrcnn":
                    pred_instances[obj_dir][distance] = []
                    preds[obj_dir][distance] = []
                    cnt_none[obj_dir][distance] = 0
                    cnt_false[obj_dir][distance] = 0
                for angle in range(360):
                    # match = re.search(r'render_(\d+).png', img_str)
                    # angle = int(match.group(1)) if match else 1000
                    # assert angle < 361, f"match wrong, angle {angle}"
                    
                    img_pth = dir+"/"+obj_dir+"/" +distance+ "/"+f"/render_{angle:03d}.png"
                   
                    # maskrcnn预测
                    if predict_mode == "maskrcnn":
                        rgb = cv2.imread(img_pth)
                        _, _, instance = sem_pred.get_prediction(rgb, return_score=False, return_instance=True)
                        pred_instances[obj_dir][distance].append(instance)
                        if len(instance) == 0:
                            values[obj_dir][distance].append(1)
                            preds[obj_dir][distance].append(None)
                            cnt_none[obj_dir][distance] += 1
                        elif len(instance) == 1:
                            if int(instance.pred_classes[0].cpu().numpy()) == 72: 
                                values[obj_dir][distance].append(1- float(instance.scores.cpu().numpy()))
                                preds[obj_dir][distance].append(float(instance.scores.cpu().numpy()))
                            else:
                                values[obj_dir][distance].append(1)
                                cnt_false[obj_dir][distance] += 1
                        else:
                            values[obj_dir][distance].append(1 - float(instance.scores.mean().cpu().numpy()))
                            preds[obj_dir][distance].append(float(instance.scores.mean().cpu().numpy()))
                    elif predict_mode == "clip":
                        # 预处理图像用于CLIP模型（仍使用PIL预处理）
                        pil_img = Image.open(img_pth)
                        image = preprocess(pil_img).unsqueeze(0).to(device)

                        with torch.no_grad():
                            image_features = model.encode_image(image)
                            text_features = model.encode_text(text)
                            
                            logits_per_image, logits_per_text = model(image, text)
                            score = logits_per_text.item()
                            values[obj_dir][distance].append(score)
            #         tmp +=1
            #         if tmp > 20:
            #             break
            #     if tmp > 20:
            #         break
            # if tmp > 20:
            #     break
                # clip打分
        
        # 存
        if predict_mode == "maskrcnn":
            with open(class_name+".pkl", "wb") as f:
                pickle.dump(pred_instances, f)
        elif predict_mode == "clip":
            with open(class_name+"_clip.pkl", "wb") as f:
                pickle.dump(values, f)
            
    if mode == "read":
        if predict_mode == "maskrcnn":
            with open(class_name+".pkl", "rb") as f:
                pred_instances = pickle.load(f)
        
            for obj, obj_dict in pred_instances.items():
                values[obj] = {}
                preds[obj] = {}
                cnt_none[obj] = {}
                cnt_false[obj] = {}
                for distance, instances_lst in obj_dict.items():
                    values[obj][distance] = []
                    preds[obj][distance] = []
                    cnt_none[obj][distance] = 0
                    cnt_false[obj][distance] = 0
                    
                    for angle, instance in enumerate(instances_lst):
                        
                        if len(instance) == 0:
                            values[obj][distance].append(1)
                            preds[obj][distance].append(None)
                            cnt_none[obj][distance] += 1
                        elif len(instance) == 1:
                            if int(instance.pred_classes[0].cpu().numpy()) == 72: 
                                values[obj][distance].append(1- float(instance.scores.cpu().numpy()))
                                preds[obj][distance].append(float(instance.scores.cpu().numpy()))
                            else:  # 错检
                                values[obj][distance].append(1)
                                cnt_false[obj][distance] += 1
                        else:
                            values[obj][distance].append(1 - float(instance.scores.mean().cpu().numpy()))
                            preds[obj][distance].append(float(instance.scores.mean().cpu().numpy()))
                    
            # 统计
            all_scores = []
            cnt_none_all = 0
            cnt_false_all = 0
            for obj, obj_dict in pred_instances.items():
                draw_array(obj, values[obj], class_name)
                for dis, preds_scores in obj_dict.items():
                    all_scores.extend(preds[obj][dis])
                    cnt_none_all += cnt_none[obj][dis]
                    cnt_false_all += cnt_false[obj][dis]
            
            result = manual_bin_count(all_scores)
            print(f"score分布", result)
            print(f"未检测到数： {cnt_none_all}")
            print(f"检测错误数： {cnt_false_all}")
    
        elif predict_mode == "clip":
            with open(class_name+"_clip.pkl", "rb") as f:
                values = pickle.load(f)
            
            max_, min_ = 0,  1e4
            dis = ['2m', '2.5m', '3m', '3.5m', '4m', '4.5m', '5m', '5.5m']
            for obj, obj_dict in values.items():
                for d in dis:
                    value = values[obj][d]
                    max_ = max(value) if max(value) > max_ else max_
                    min_ = min(value) if min(value) < min_ else min_
            print(f"clip score: max  {max_}, min {min_}")
                # draw_array(obj, values[obj], class_name+"_clip10_norm", norm=True)
        # 所有角度结果拼接
    
    
    
    
