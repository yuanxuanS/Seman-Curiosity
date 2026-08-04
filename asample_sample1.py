from src.finetune.dataset_utils import SampleLoader
import clip
import torch
from asample.constants import target_coco_categories
import argparse
import pickle
import random
import time
from pathlib import Path
import numpy as np
from src.policy_rl.sequence_utils import stc_select_samples

parser = argparse.ArgumentParser()
parser.add_argument(
    "--stc_algorithm",
    type=str,
    default="rewrite",
    choices=["rewrite", "legacy"],
    help="STC algorithm variant for sample selection",
)
parser.add_argument(
    "--gpu_id",
    type=int,
    default=3,
    help="GPU id used for CLIP inference when CUDA is available",
)
parser.add_argument(
    "--cpu",
    action="store_true",
    help="Run CLIP inference on CPU even when CUDA is available",
)
parser.add_argument(
    "--group_empty_tracks",
    action="store_true",
    help="Group consecutive frames without detections into tracks in rewrite STC",
)
parser.add_argument(
    "--random_sample_n",
    type=int,
    default=None,
    help=(
        "Randomly sample N frames from each glbstep without running STC/CLIP. "
        "If a glbstep contains fewer than N frames, all of them are selected."
    ),
)
parser.add_argument(
    "--random_seed",
    type=int,
    default=0,
    help="Random seed used with --random_sample_n",
)
parser.add_argument(
    "--frame_detections_source",
    choices=["saved", "realtime"],
    default="saved",
    help="Read saved bbspred detections or run a detector on each RGB frame",
)
parser.add_argument(
    "--realtime_detector",
    choices=["maskrcnn", "yolov8n"],
    default="maskrcnn",
    help="Detector used when --frame_detections_source=realtime",
)
parser.add_argument(
    "--detector_device",
    default=None,
    help="Detector device, e.g. cuda:0 or cpu (default: follow --cpu/--gpu_id)",
)
parser.add_argument(
    "--detector_confidence",
    type=float,
    default=0.5,
    help="Confidence threshold for real-time detection",
)
parser.add_argument(
    "--maskrcnn_config",
    type=Path,
    default=Path(
        "third_parties/detectron2/configs/embodied/"
        "mask_rcnn_R_50_FPN_1x_embodied.yaml"
    ),
)
parser.add_argument(
    "--maskrcnn_weights",
    type=Path,
    default=Path("third_parties/detectron2/models/model_final_a54504.pkl"),
)
parser.add_argument(
    "--yolov8n_weights",
    type=Path,
    default=Path("yolov8n-seg.pt"),
)
parser.add_argument("--yolov8_iou", type=float, default=0.7)
args = parser.parse_args()

if args.random_sample_n is not None and args.random_sample_n < 0:
    parser.error("--random_sample_n must be greater than or equal to 0")
if not 0.0 <= args.detector_confidence <= 1.0:
    parser.error("--detector_confidence must be between 0 and 1")


def _normalise_instances(instances):
    """Return the common frame-detection form consumed by sequence_utils."""
    instances = instances.to("cpu")
    required_fields = ("pred_boxes", "pred_classes", "scores", "pred_masks")
    missing = [name for name in required_fields if not instances.has(name)]
    if missing:
        raise ValueError("detections are missing fields: {}".format(missing))
    return instances


class RealtimeDetector:
    """Unify Mask R-CNN and YOLOv8-seg outputs as Detectron2 Instances."""

    def __init__(self, detector_name, device):
        self.detector_name = detector_name
        self.device = device
        if detector_name == "maskrcnn":
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor

            config_path = args.maskrcnn_config.expanduser().resolve()
            weights_path = args.maskrcnn_weights.expanduser().resolve()
            if not config_path.is_file():
                raise FileNotFoundError("Mask R-CNN config not found: {}".format(config_path))
            if not weights_path.is_file():
                raise FileNotFoundError("Mask R-CNN weights not found: {}".format(weights_path))

            cfg = get_cfg()
            # Custom keys required by this repository's embodied config.
            cfg.VIS = False
            cfg.SAVE_PTH = ""
            cfg.DATASET_NAME = ""
            cfg.MODEL.NUM_CLASSES = 80
            cfg.merge_from_file(str(config_path))
            cfg.MODEL.WEIGHTS = str(weights_path)
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.detector_confidence
            cfg.MODEL.DEVICE = device
            cfg.freeze()
            self.predictor = DefaultPredictor(cfg)
        else:
            from src.detectors.yolov8_seg import YOLOv8SegBackend

            self.predictor = YOLOv8SegBackend(
                weights=args.yolov8n_weights,
                device=device,
                confidence=args.detector_confidence,
                iou=args.yolov8_iou,
            )

    def predict(self, rgb_images):
        # Both DefaultPredictor and Ultralytics interpret numpy images as BGR.
        bgr_images = [np.ascontiguousarray(image[:, :, ::-1]) for image in rgb_images]
        if self.detector_name == "maskrcnn":
            outputs = [self.predictor(image)["instances"] for image in bgr_images]
        else:
            outputs = [
                detection.to_detectron2_instances()
                for detection in self.predictor.predict(bgr_images)
            ]
        return [_normalise_instances(instances) for instances in outputs]

# 加载数据
stage = 2
data_pth = "/home/wpp/Semantic-Curiosity/Semantic-Curiosity/exps/dump/sequencev2_wotrajR_eval/episodes_data"
# "outputs_asample/imgs/test5_env1/rgb_all_data"
sampler = SampleLoader(data_pth, glbstep=True)
inputs = sampler.get_env_episode_and_steps_dense_list(more_mode=False)  


# 遍历每一glbstep的数据
if stage == 1:
    glb_frames = {}

    for env, episode, glbstep, step in zip(inputs[0], inputs[1], inputs[2], inputs[3]):
        if env not in glb_frames:
            glb_frames[env] = {}
        if episode not in glb_frames[env]:
            glb_frames[env][episode] = {}
        if glbstep not in glb_frames[env][episode]:
            glb_frames[env][episode][glbstep] = [step]
        else:
            glb_frames[env][episode][glbstep].append(step)

    print(glb_frames)
    with open("./asample_straight_indices_sequencev2_yolo.pkl", "wb") as f:
        pickle.dump(glb_frames, f)
    raise SystemExit


else:
    glb_frames_tracks = {}
    glb_frames_sampled = []
    sample_budget = 4
    clip_target_threshold = 0.5
    mod = ["rgb"]
    if args.frame_detections_source == "saved":
        mod.append("bbspred")

    with open("./asample_straight_indices_sequencev2_yolo.pkl", "rb") as f:
        glb_frames_indices = pickle.load(f)
        
    random_sampler = random.Random(args.random_seed)
    if args.random_sample_n is None:
        device = "cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu_id}"
        clip_model, preprocess = clip.load("ViT-L/14", device=device)
        text_str = [key for key in target_coco_categories.keys()]
        text = clip.tokenize([f"a photo contains a {i}" for i in text_str]).to(device)
        realtime_detector = None
        if args.frame_detections_source == "realtime":
            detector_device = args.detector_device or device
            realtime_detector = RealtimeDetector(
                args.realtime_detector, detector_device
            )
            print(
                "using real-time {} detections on {}".format(
                    args.realtime_detector, detector_device
                )
            )


    #组合每一track并进行采样
    for env, env_data in glb_frames_indices.items():
        
        if env not in glb_frames_tracks:
            glb_frames_tracks[env] = {}
            
        for episode, epi_data in env_data.items():
            
            if episode not in glb_frames_tracks[env]:
                glb_frames_tracks[env][episode] = {}
                
            for glbstep, frames in epi_data.items():
                sorted_frames = sorted(frames)

                if args.random_sample_n is not None:
                    sample_count = min(args.random_sample_n, len(sorted_frames))
                    sampled_frames = sorted(
                        random_sampler.sample(sorted_frames, sample_count)
                    )
                    # Random mode deliberately does not construct or score tracks.
                    glb_frames_tracks[env][episode][glbstep] = []
                    for sampled_frame in sampled_frames:
                        glb_frames_sampled.append(
                            [env, episode, glbstep, sampled_frame]
                        )
                    print(
                        f"randomly sampled {sample_count}/{len(sorted_frames)} frames "
                        f"of env{env} epi{episode} glbstep{glbstep}"
                    )
                    continue

                glb_datas = []
                for f in sorted_frames:
                    sample_data = sampler.get_sample_multimodality(
                        env, episode, f, mod, glbstep)
                    glb_datas.append(sample_data)
                    
                # 对每一glbstep的数据进行分组；同一位置且统一类别为同一组
                
                frame_rgbs = [data['rgb'].data for data in glb_datas]
                if args.frame_detections_source == "saved":
                    frame_detections = [
                        _normalise_instances(data['bbspred'].data)
                        for data in glb_datas
                    ]
                else:
                    frame_detections = realtime_detector.predict(frame_rgbs)
                
                s = time.time()
                groups, sampled_frames = stc_select_samples(
                    frame_detections,
                    frame_rgbs,
                    budget=sample_budget,
                    clip_model=clip_model,
                    preprocess=preprocess,
                    text=text,
                    device=device,
                    clip_target_threshold=clip_target_threshold,
                    algorithm=args.stc_algorithm,
                    group_empty_tracks=args.group_empty_tracks,
                )
                print(f"stc-{args.stc_algorithm}-{time.time() - s} ")
                
                glb_frames_tracks[env][episode][glbstep] = groups   # 存储打好分的group

                for sf in sampled_frames:
                    glb_frames_sampled.append([env, episode, glbstep, sf])
                
                print(f"add tracks of env{env} epi{episode} glbstep{glbstep}")
            # 初始化

    if args.random_sample_n is not None:
        output_suffix = f"random_n{args.random_sample_n}_seed{args.random_seed}"
    elif args.stc_algorithm == "rewrite":
        rewrite_idx = 3 if args.group_empty_tracks else 2
        output_suffix = f"bg{sample_budget}_r{rewrite_idx}_{args.stc_algorithm}"
    else:
        rewrite_idx = 0
        output_suffix = f"bg{sample_budget}_r{rewrite_idx}_{args.stc_algorithm}"
    if args.frame_detections_source == "realtime":
        output_suffix += f"_realtime_{args.realtime_detector}"
    with open(f"./asample_straight_tracks_sequencev2_yolo_r2_{output_suffix}.pkl", "wb") as f:
        pickle.dump(glb_frames_tracks, f)
        
    with open(f"./asample_straight_sampled_sequencev2_yolo_r2_{output_suffix}.pkl", "wb") as f:
        pickle.dump(glb_frames_sampled, f)
    print(glb_frames_sampled)
