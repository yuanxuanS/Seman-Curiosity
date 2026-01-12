from src.finetune.dataset_utils import get_loader, SampleLoader
class_map_coco = { 
                "chair": 56,
                "couch": 57,
                "plant": 58,
                "bed": 59,
                "toilet": 61,
                "refrigerator": 72,}

class_range_min = {cls_: 1e4 for cls_ in class_map_coco.keys()}
class_range_max = {cls_: -1e4 for cls_ in class_map_coco.keys()}

# 读取数据集
base_dir = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/vsqf_v1_eval_re/episodes_data"
# "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/data/vsqf_test_val5/"
# "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/data/vsqf_test_val5/data/Wiconisco"
# "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/vsqf_v1_eval"
# "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/data/fix_test640"
# 
# "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/data/multiSens_test_ft_train5_final"
data_pth = base_dir + ""    #"/data/Markleeville" #"/episodes_data"
sampler = SampleLoader(data_pth)
inputs = sampler.get_env_episode_and_steps_dense_list()  

# 统计每个类别的距离范围： depth_min, depth_max
for env, episode, step in zip(inputs[0], inputs[1], inputs[2]):
# env = 1
# episode = 0
# step = 3260
    sample_data = sampler.get_sample_multimodality(env, episode, step, ["bbsgt", "rgb", "depth"])
    rgb = sample_data['rgb'].data
    instance = sample_data['bbsgt'].data
    # print(f" instance id {instance.pred_classes}")
    if 72 in instance.pred_classes:
        print(f"exis refri")
    depth = sample_data['depth'].data

    for i in range(len(instance.pred_classes)):
        cls_id = instance.pred_classes[i]
        
        # semantic如果存在目标类别
        obj_name = None
        for cls_name, cls_int in class_map_coco.items():
            if cls_int == int(cls_id.cpu().numpy()):
                obj_name = cls_name
                break
        if obj_name is None:
            continue
        
        # 加载mask和对应depth，计算depth均值
        # mask_ = (instance.pred_masks)[i][:, :, None]
        # masked_obj_depth = depth * mask_.cpu().numpy()
        
        # depth_mean= masked_obj_depth[mask_].mean()
        # class_range_min[obj_name] = min(depth_mean, class_range_min[obj_name])
        # class_range_max[obj_name] = max(depth_mean, class_range_max[obj_name])

# 返回每个类别的depth范围, dict
# print("dataset, ",base_dir)
# print("max depth: ", class_range_max)
# print("min depth: ", class_range_min)