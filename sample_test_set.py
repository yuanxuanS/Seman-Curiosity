import os
import cv2
import habitat
from habitat import make_dataset
from itertools import compress
import numpy as np
from detectron2.utils.visualizer import ColorMode, Visualizer
from src.finetune.utils import sim_utils
from src import constants
from src.finetune.dataset_utils import save_obs
from src.policy_rl import arguments
from src.policy_rl.envs.habitat import _get_scenes_from_folder
from src import constants
def main() -> None:
    args = arguments.get_args()
    
    habitat_cfg = "/home/users/wpp/Semantic-Curiosity/Semantic-Curiosity/src/policy_rl/envs/habitat/configs/tasks/objectnav_gibson.yaml"
    config = habitat.get_config(habitat_cfg)
    config.defrost()
    config.DATASET.SPLIT = args.split
    config.DATASET.DATA_PATH = \
        config.DATASET.DATA_PATH.replace("v1", args.version)  # same in args
    config.DATASET.EPISODES_DIR = \
        config.DATASET.EPISODES_DIR.replace("v1", args.version)
    
    if "*" in config.DATASET.CONTENT_SCENES:
        content_dir = os.path.join(config.DATASET.EPISODES_DIR.format(
            split=args.split), "content")
        # scenes = _get_scenes_from_folder(content_dir)
        scenes = [sc + ".glb" for sc in constants.scenes[args.split]]
        config.DATASET.CONTENT_SCENES = scenes
    else:
        scenes = config.DATASET.CONTENT_SCENES
    config.freeze()
    
    
    
    dataset = make_dataset(config.DATASET.TYPE, config=config.DATASET)
    
    output_path = "/data1/wpp_data/semantic_curiosity_sample_test256/"
    os.makedirs(output_path, exist_ok=True)
    data_pth = output_path+"/data/"
    os.makedirs(data_pth, exist_ok=True)
    img_pth = output_path+"/imgs/"
    os.makedirs(img_pth, exist_ok=True)

    semantic_filter_scenes = [
        os.path.exists(
            dataset.scene_ids[i].replace("//", "/").replace(".glb", "_semantic.ply")
        )
        for i in range(len(dataset.scene_ids))
    ]
    scenes = dataset.scene_ids

    scenes = list(compress(scenes, semantic_filter_scenes))

    print(f"image size is {args.env_frame_height}, {args.env_frame_width}")
    for scene_count, scene_id in enumerate(scenes):
        # if scene_count < 4:
        #     continue
        
        extractor = sim_utils.FirstPersonImageExtractor(
            scene_filepath=scene_id,
            img_size=(args.env_frame_height, args.env_frame_width),        # TODO
            output=["rgba", "depth", "semantic"],
        )

        N_TOT = 1000    #500 
        count_samples = 0
        indexes = np.random.permutation(len(extractor))
        for idx in indexes:
            x = extractor[idx]


            if count_samples >= N_TOT:
                break

            if len(x['bbsgt']['instances']) == 0:
                continue
            # 视场模块，需要有指定目标在
            if not object_cls_in_instance(x):
                continue
            save_obs(data_pth, scene_count, 0, x, count_samples)

            # save visualized imgs
            img = x['rgb'][:,:,:3]
            v2 = Visualizer(img)
            v2 = v2.draw_instance_predictions(x["bbsgt"]["instances"].to("cpu"))                    # potential map
            img = cv2.cvtColor(v2.get_image(), cv2.COLOR_BGR2RGB)
            cv2.imwrite(img_pth + "scene_"+str(scene_count)+"_step_"+str(idx)+".png", img)

            count_samples += 1

        extractor.close()
        print(f"tot samples {count_samples}")
        print(f"{scene_id} completed")
        
def object_cls_in_instance(x):
    cls_id = [56, 57, 58, 59, 61]
    assert list(constants.coco_categories_mapping.keys()) == cls_id
    for pc in x['bbsgt']['instances'].pred_classes:
        if pc in cls_id:
            return True
    return False


if __name__ == '__main__':
    main()
