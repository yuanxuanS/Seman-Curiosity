import json
from src.finetune.dataset_utils import get_loader, SampleLoader
import torch
from src.policy_rl.agents.utils.detect_utils import box_iou_calc
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import pickle

def plot_all_trajectories_combined(objs, data, save_name="combined_analysis_map.png"):
    """
    将多个 obj 的轨迹合并到一张图中绘制
    """
    # 1. 合并所有选定 obj 的数据到一个列表
    all_combined_data = []
    for obj_id in objs:
        if obj_id in data:
            all_combined_data.extend(data[obj_id])
    
    if not all_combined_data:
        print("没有找到有效数据")
        return

    # 2. 统一排序：所有 level 2 的数据放在底层，非 2 的数据放在顶层
    # 这样即使是不同 obj，只要 level 不是 2，都会浮在所有 level 2 之上
    all_combined_data.sort(key=lambda x: 1 if x['pred_level'] != 2 else 0)

    # 3. 创建画布
    plt.figure(figsize=(16, 12))
    
    # 4. 颜色映射处理（基于所有数据的 pred_level）
    unique_levels = sorted(list(set(item['pred_level'] for item in all_combined_data)))
    cmap = plt.get_cmap('tab10', len(unique_levels))
    level_to_color = {level: cmap(i) for i, level in enumerate(unique_levels)}

    # 5. 遍历合并后的数据绘图
    for item in all_combined_data:
        pos_3d = item['position'][0]
        x, z = pos_3d[0], pos_3d[2]
        
        quat_obj = item['position'][1]
        try:
            # 兼容 quaternion 库或普通序列
            w, qx, qy, qz = quat_obj.w, quat_obj.x, quat_obj.y, quat_obj.z
        except AttributeError:
            w, qx, qy, qz = quat_obj[0], quat_obj[1], quat_obj[2], quat_obj[3]

        # 转换为 scipy 旋转对象
        r = R.from_quat([qx, qy, qz, w])
        
        # 使用您代码中指定的初始方向 [0, 0, 1]
        dir_3d = r.apply([0, 0, 1])
        dx, dz = dir_3d[0], dir_3d[2]
        
        level = item['pred_level']
        color = level_to_color[level]
        
        # 视觉分层
        alpha = 0.3 if level == 2 else 0.9
        width = 0.003 if level == 2 else 0.006

        plt.quiver(x, z, dx, dz, 
                   color=color,
                   alpha=alpha,
                   scale=30,          # 如果图上点太密，可以增大 scale
                   width=width,       
                   headwidth=3,       
                   pivot='mid')

    # 6. 辅助设置
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='>', color='w', label=f'Level {l}',
                              markerfacecolor=level_to_color[l], markersize=10) 
                       for l in unique_levels]
    plt.legend(handles=legend_elements, title="Pred Level (All Objects)", loc='upper right')

    plt.xlabel('X Position')
    plt.ylabel('Z Position')
    plt.title(f'Combined Trajectories for Objs: {objs}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal') 
    
    plt.savefig(save_name, dpi=300) # 增加 dpi 保证多物体时清晰
    plt.show()
    

def plot_trajectory_xz_plane(obj_id, data_list):
    """
    在 XZ 平面绘制带朝向箭头的 2D 轨迹图
    :param data_list: 包含 file_name, position, pred_level 的列表
    """
    plt.figure(figsize=(12, 8))
    
    # 1. 数据排序：让 pred_level 为 2 的排在前面（先画），其他的排在后面（后画，即在最上层）
    # key 的逻辑：如果是 2 返回 0，否则返回 1。排序后 0 在前，1 在后。
    data_list_sorted = sorted(data_list, key=lambda x: 1 if x['pred_level'] != 2 else 0)

    # 2. 颜色映射处理
    unique_levels = sorted(list(set(item['pred_level'] for item in data_list)))
    cmap = plt.get_cmap('tab10', len(unique_levels)) # 使用更鲜明的色板
    level_to_color = {level: cmap(i) for i, level in enumerate(unique_levels)}
    
    # 2. 遍历数据绘图
    for item in data_list_sorted:
        # 获取位置 [x, y, z]
        pos_3d = item['position'][0]
        x, z = pos_3d[0], pos_3d[2] # 提取 X 和 Z
        
        # 获取四元数并计算方向向量
        quat_obj = item['position'][1]
        # 假设格式为 quaternion(w, x, y, z)
        try:
            w, qx, qy, qz = quat_obj.w, quat_obj.x, quat_obj.y, quat_obj.z
        except AttributeError:
            w, qx, qy, qz = quat_obj[0], quat_obj[1], quat_obj[2], quat_obj[3]

        # 转换为 scipy 旋转对象 (scipy 顺序为 [x, y, z, w])
        r = R.from_quat([qx, qy, qz, w])
        
        # 定义初始朝向（假设初始面向 X 轴正方向 [1, 0, 0]）
        # 旋转后得到 3D 方向向量
        dir_3d = r.apply([0, 0, 1])
        
        # 提取在 XZ 平面上的投影分量
        dx, dz = dir_3d[0], dir_3d[2]
        
        level = item['pred_level']
        color = level_to_color[level]
        
        # 调整透明度：让背景（level 2）稍微淡一点，突出顶层
        alpha = 0.4 if level == 2 else 1.0
        # 调整箭头大小：让非 2 的箭头稍微大一点或粗一点
        width = 0.004 if level == 2 else 0.007

        # 3. 绘制 2D 箭头 (Quiver)
        # plt.quiver(x, y, u, v)
        plt.quiver(x, z, dx, dz, 
                   color=color,
                   alpha=alpha,
                   scale=25,          
                   width=width,       
                   headwidth=3,       
                   pivot='mid')
    # 4. 辅助设置
    # 创建自定义图例
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='>', color='w', label=f'Level {l}',
                              markerfacecolor=level_to_color[l], markersize=10) 
                       for l in unique_levels]
    plt.legend(handles=legend_elements, title="Pred Level")

    plt.xlabel('X Position')
    plt.ylabel('Z Position')
    plt.title('2D Projection on XZ Plane with Orientations')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal') # 保证 X 和 Z 的比例一致，防止箭头变形
    
    plt.savefig(f"./analysis_pred_map_{obj_id}.png")
    

rgb_info = "./third_parties/detectron2/datasets/embodied_Colli/annotations/instances_val.json"
detector_pred_f = "./third_parties/detectron2/output/embodied_Coll/inference/embodied_val/coco_instances_results.json"
data_pth = "data/visibles/Collierville"

DETECTED = 0
WRONG_DETECTED = 1
NO_DETECTED = 2
from src.vqf_constants import clsid_name_maps        # TODO
CLASSES_TO_IDX = {k: i for i, k in enumerate(clsid_name_maps.keys())}
IDX_TO_CLASSES = {k: i for i, k in CLASSES_TO_IDX.items()}

def save(rgb_info, detector_pred_f):
    with open(rgb_info, 'r') as f:
        rgb_infos = json.load(f)

    # 得到所有rgb的info
    rgb_info = {}     # step_id: {json info}
    for img_info in rgb_infos['images']:
        file_name = img_info['file_name']
        image_id = img_info['id']
        rgb_info[image_id] = [file_name, []]
        # print(img_info)

    with open(detector_pred_f, "r") as f:
        prediction = json.load(f)
    
    for pred in prediction:
        image_id = pred['image_id']
        rgb_info[image_id][1].append(pred)
    
    # key=filename
    rgbname_info = {}
    for rgb_, info in rgb_info.items():
        rgbname_info[info[0]] = info[1]

    
    sampler = SampleLoader(data_pth)
    inputs = sampler.get_env_episode_and_steps_dense_list(more_mode=False)  


    object_ids = []
    object_preds = {}
    for env, episode, step in zip(inputs[0], inputs[1], inputs[2]):
        type = ["bbsgt", "rgb", "depth", "position", "semantic", ]

        sample_data = sampler.get_sample_multimodality(
            env, episode, step, type)
        
        name = f"epi{episode}_env{env}_step{step}.png"
        
        gt = sample_data['bbsgt']
        file_name = gt.frame.sense_info.get_path()
        y = gt.get_bbs_as_gt()
        class_labels = y.gt_classes
        if len(y) == 0:
            continue     
        pred = rgbname_info[name]
        
        
        
        # get gt's obejct id
        gt_obj_ids = []
        for i in range(len(class_labels)):
            sem = sample_data['semantic'].data
            id = torch.unique(y.gt_masks[i]*sem)
            id = id[id>0]
            gt_obj_ids.append(int(id))

        # 预测等级
        pred_boxes = np.array([p['bbox'] for p in pred])
        pred_classes = np.array([IDX_TO_CLASSES[p['category_id']] for p in pred])
        gt_boxes = y.gt_boxes.tensor.numpy()
        
        
        gt_pred_level = []
        
        if len(pred_boxes) == 0:
            gt_pred_level = [NO_DETECTED] * len(gt_boxes)
        else:
            matrix = box_iou_calc(gt_boxes, pred_boxes)
            for i in range(len(y.gt_classes)):
                curr_pred = matrix[i] > 0.5
                
                if curr_pred.any():
                    valid_ = pred_classes[curr_pred]
                    if y.gt_classes[i] in valid_:
                        gt_pred_level.append(DETECTED)
                    else:
                        gt_pred_level.append(WRONG_DETECTED)
                else: # 未检出
                    gt_pred_level.append(NO_DETECTED)
                
        for i, obj in enumerate(gt_obj_ids):
            obj_view_info = {
                    "file_name": name,
                    "position": [sample_data['position'].position, sample_data['position'].orientation],
                    "pred_level": gt_pred_level[i]
                                }
            if obj in object_preds:
                object_preds[obj].append(obj_view_info)
            else:
                object_preds[obj] = [obj_view_info]
            
        for id in gt_obj_ids:
            if id not in object_ids:
                object_ids.append(id)
        
        # if len(object_preds[39]) > 5:
        #     break

save_f = "obj_pred_Colli"
# with open(save_f+".pkl", "wb") as f:
#     pickle.dump(object_preds, f)

with open(save_f+".pkl", "rb") as f:
    data = pickle.load(f)
# print(data.keys())

# objs =  [10, 11, 12, 31, 33, 34, 39, 32, 30, 38, 46, 13, 29, 36, 37, 35]
# for id in objs:
#     plot_trajectory_xz_plane(id, data[id])

plot_all_trajectories_combined(list(data.keys()), data)
            
