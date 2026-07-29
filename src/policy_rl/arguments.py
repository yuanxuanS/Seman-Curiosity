import argparse
import torch
import math
def get_args():
    parser = argparse.ArgumentParser(
        description='Semantic-Curiosity')

    # General Arguments
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--auto_gpu_config', type=int, default=1)
    parser.add_argument('--total_num_scenes', type=str, default="auto")
    parser.add_argument('-n', '--num_processes', type=int, default=1,
                        help="""how many training processes to use (default:5)
                                Overridden when auto_gpu_config=1
                                and training on gpus""")
    parser.add_argument('--num_processes_per_gpu', type=int, default=6)
    parser.add_argument('--num_processes_on_first_gpu', type=int, default=1)
    parser.add_argument('--eval', type=int, default=0,
                        help='0: Train, 1: Evaluate (default: 0)')
    parser.add_argument('--num_training_frames', type=int, default=10000000,
                        help='total number of training frames')
    parser.add_argument('--num_eval_episodes', type=int, default=200,
                        help="number of test episodes per scene")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument("--sim_gpu_id", type=int, default=1,
                        help="gpu id on which scenes are loaded")
    parser.add_argument("--sem_gpu_id", type=int, default=-1,
                    help="""gpu id for semantic model,
                            -1: same as sim gpu, -2: cpu""")

    # Logging, loading models, visualization
    parser.add_argument('--log_interval', type=int, default=10,
                        help="""log interval, one log per n updates
                                (default: 10) """)
    parser.add_argument('--save_interval', type=int, default=1,
                        help="""save interval""")
    parser.add_argument('-d', '--dump_location', type=str, default="./exps/",
                        help='path to dump models and log (default: ./exps/)')
    parser.add_argument('--exp_name', type=str, default="exp1",
                        help='experiment name (default: exp1)')
    parser.add_argument('--save_periodic', type=int, default=500000,
                        help='Model save frequency in number of updates')
    parser.add_argument('--load', type=str, default="0",
                        help="""model path to load,
                                0 to not reload (default: 0)""")
    parser.add_argument('--load_pretrain', type=str, default="0",
                help="""pretrained panorama_model checkpoint path from train_panorama_model.py,
                    0 to not reload (default: 0)""")
    parser.add_argument('-v', '--visualize', type=int, default=0,
                        help="""1: Render the observation and
                                   the predicted semantic map,
                                2: Render the observation with semantic
                                   predictions and the predicted semantic map
                                (default: 0)""")
    parser.add_argument('--print_images', type=int, default=0,
                        help='1: save visualization as images')
    
    # Environment, dataset and episode specifications
    parser.add_argument('-efw', '--env_frame_width', type=int, default=640,     # sim返回的RGB大小
                        help='Frame width (default:84)')
    parser.add_argument('-efh', '--env_frame_height', type=int, default=640,
                        help='Frame height (default:84)')
    parser.add_argument('-dfw', '--det_frame_width', type=int, default=256,
                        help='Frame height (default:84)')
    parser.add_argument('-dfh', '--det_frame_height', type=int, default=256,
                        help='Frame height (default:84)')
    parser.add_argument('-fw', '--frame_width', type=int, default=256,      # policy输入大小,map更新时obs大小, 在输入前将env_frame_width变为frame_width大小
                        help='Frame width (default:84)')
    parser.add_argument('-fh', '--frame_height', type=int, default=256,
                        help='Frame height (default:84)')
    parser.add_argument('-cfw', '--camera_frame_width', type=int, default=128,      # camera policy输入大小,map更新时obs大小, 在输入前将env_frame_width变为frame_width大小
                        help='Frame width (default')
    parser.add_argument('-cfh', '--camera_frame_height', type=int, default=128,
                        help='Frame height (default)')
    parser.add_argument('-el', '--max_episode_length', type=int, default=500,
                        help="""Maximum episode length, steps in an episode""")
    
    parser.add_argument("--task_config", type=str,
                        default="tasks/objectnav_gibson.yaml",
                        help="path to config yaml containing task information")
    parser.add_argument("--split", type=str, default="train",
                        help="dataset split (train | val | val_mini) ")
    parser.add_argument('--camera_height', type=float, default=0.88,    # TODO: 对比参数
                        help="agent camera height in metres")
    parser.add_argument('--hfov', type=float, default=79.0,     # TODO: 对比参数
                        help="horizontal field of view in degrees")
    parser.add_argument('--turn_angle', type=float, default=30,
                        help="Agent turn angle in degrees")
    parser.add_argument('--min_depth', type=float, default=0.5,
                        help="Minimum depth for depth sensor in meters")
    parser.add_argument('--max_depth', type=float, default=5.0,
                        help="Maximum depth for depth sensor in meters")
    parser.add_argument('--min_d', type=float, default=1.5,
                        help="min distance to goal during training in meters")
    parser.add_argument('--max_d', type=float, default=100.0,
                        help="max distance to goal during training in meters")
    parser.add_argument('--version', type=str, default="v1.1",
                        help="dataset version")
    # for initialize location
    parser.add_argument('--success_dist', type=float, default=1.0,
                    help="success distance threshold in meters")
    parser.add_argument('--num_train_episodes', type=int, default=10000,        # 貌似仅用于判断一开始加载时，是不是场景
                        help="""number of train episodes per scene
                                before loading the next scene""")
    parser.add_argument('--floor_thr', type=int, default=50,
                        help="floor threshold in cm")
    
    # Model Hyperparameters
    parser.add_argument('--env', type=str, default="sem_cur_exp")
    parser.add_argument('--agent', type=str, default="rl",
                        help="agent type (rl | random)")
    parser.add_argument('--lr', type=float, default=2.5e-5,
                        help='learning rate (default: 2.5e-5)')
    parser.add_argument('--local_hidden_size', type=int, default=256,
                        help='local_hidden_size')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RL Optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RL Optimizer alpha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use_gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy_coef', type=float, default=0.001,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value_loss_coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--num_local_steps', type=int, default=100,    # = horizon size?
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo_epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num_mini_batch', type=str, default="auto",   # 需要更大吗
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip_param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--use_recurrent_local', type=int, default=1,
                        help='use a recurrent local policy')
    parser.add_argument('--reward_coeff', type=float, default=1.25e-3,      
                        help="Semantic curiosity reward coefficient")
    parser.add_argument('--reward_coeff_seq', type=float, default=0.1,      
                        help="sequence reward coefficient")
    parser.add_argument('--reward_coeff_ce', type=float, default=0.1,     
                        help="class entropy reward coefficient")
    parser.add_argument('--distance_reward_coeff', type=float, default=0.02,
                        help="distance reduce reward coefficient")
    parser.add_argument('--num_sem_categories', type=float, default=6,
                        help="number of semantic plus 1")
    parser.add_argument('--sem_pred_prob_thr', type=float, default=0.9,
                        help="Semantic prediction confidence threshold") 
    parser.add_argument(
        '--sem_pred_batch_size', type=int, default=3,
        help="batch size for per-env sequence semantic prediction; set 1 to disable batching"
    )
    parser.add_argument(
        "--detector_backend",
        type=str,
        default="mask_rcnn",
        choices=["mask_rcnn", "yolov8"],
        help="instance-segmentation backend used by main_sequence.py",
    )
    parser.add_argument(
        "--yolov8_weights",
        type=str,
        default="yolov8n-seg.pt",
        help="YOLOv8 segmentation checkpoint used when --detector_backend=yolov8",
    )
    parser.add_argument(
        "--yolov8_iou",
        type=float,
        default=0.7,
        help="YOLOv8 non-maximum suppression IoU threshold",
    )
    parser.add_argument(
        "--yolov8_image_size",
        type=int,
        default=640,
        help="YOLOv8 inference image size",
    )
    parser.add_argument(
        "--yolov8_max_detections",
        type=int,
        default=300,
        help="maximum YOLOv8 detections per image",
    )
    
    # Mapping
    parser.add_argument('--global_downscaling', type=int, default=2)    # full_map缩放 downscaling倍数，得到local_map实际大小
    parser.add_argument('--vision_range', type=int, default=100)
    parser.add_argument('--map_resolution', type=int, default=5)        # 每一网格的实际大小
    parser.add_argument('--du_scale', type=int, default=2,
                        help="输入policy的RGB大小 frame_w 构建地图时是否缩小")
    parser.add_argument('--map_size_cm', type=int, default=2400)
    parser.add_argument('--cat_pred_threshold', type=float, default=5.0)
    parser.add_argument('--map_pred_threshold', type=float, default=1.0)
    parser.add_argument('--exp_pred_threshold', type=float, default=1.0)
    parser.add_argument('--collision_threshold', type=float, default=0.20)
    parser.add_argument('--collision_threshold_real', type=float, default=0.10)

    # samples
    parser.add_argument('--save_samples', default=False,
                        help='save observations')
    # sample obj data 
    parser.add_argument('--num_sample_pts', type=int, default=50000)
    parser.add_argument('--sample_pt_distance_interval', type=float, default=0.1)
    parser.add_argument(
        "--stc_algorithm",
        type=str,
        default="rewrite",
        choices=["rewrite", "legacy"],
        help="STC algorithm variant: rewrite is the current implementation; legacy restores the pre-rewrite implementation",
    )
    
    # vsqf
    parser.add_argument('--magnify', default=False,
                        help='magnify vsqf scores')
    parser.add_argument('--magnify_num', type=float, default=3.0)
    parser.add_argument('--vsqf_version', type=str, default="v2")
    
    # explore algor for poni
    parser.add_argument('--explore_algor', type=str, default="frontier",
                        help="explore algorithm in heuristic, poni | frontier")
    parser.add_argument(
        "--pf_model_path",
        type=str,
        default="./data/poni_models/poni_seed123_gibson_pf_model",
        help="path to PF model weights",
    )
    parser.add_argument(
        "--pf_masking_opt",
        type=str,
        default="unexplored",
        choices=["unexplored", "none"],
    )
    parser.add_argument("--add_agent2loc_distance", action="store_true", default=False)
    parser.add_argument(
        "--add_agent2loc_distance_v2", action="store_true", default=False
    )
    parser.add_argument("--mask_nearest_locations", action="store_true", default=False)
    parser.add_argument(
        "--mask_size",
        type=float,
        default=1.0,
        help="mask size (meters) for mask_nearest_locations option",
    )
    parser.add_argument("--area_weight_coef", type=float, default=0.7)
    parser.add_argument("--dist_weight_coef", type=float, default=0.3)
    parser.add_argument('--poni_num_global_steps', type=int, default=1,    # = horizon size?
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--poni_num_sem_categories', type=float, default=16,
                        help="number of semantic plus 1 in poni")
    
    # for diversity reward
    parser.add_argument(
        "--use_diversity_reward", action="store_true", default=False
    )
    parser.add_argument(
        "--use_traj_feature_reward", action="store_true", default=False,
        help="add adaptive same-region trajectory feature similarity penalty"
    )
    parser.add_argument("--reward_coeff_traj", type=float, default=5)
    parser.add_argument("--traj_sim_percentile", type=float, default=90.0)
    parser.add_argument("--traj_min_region_points", type=int, default=5)
    parser.add_argument("--traj_sim_window", type=int, default=200)
    parser.add_argument("--traj_max_points", type=int, default=50)
    parser.add_argument("--traj_region_update_interval", type=int, default=5)
    parser.add_argument("--traj_region_method", type=str, default="watershed",
                        choices=["watershed", "connected"])
    parser.add_argument("--traj_watershed_min_distance", type=int, default=20)
    parser.add_argument(
        "--log_traj_reward_detail", action="store_true", default=False,
        help="print per-step trajectory reward debug info for one environment"
    )
    parser.add_argument("--log_traj_reward_env", type=int, default=0)
    parser.add_argument(
        "--profile_sequence", action="store_true", default=False,
        help="enable all sequence profiling switches"
    )
    parser.add_argument(
        "--profile_main_sequence", action="store_true", default=False,
        help="print per-section timing stats in main_sequence.py"
    )
    parser.add_argument(
        "--profile_env_step", action="store_true", default=False,
        help="print per-env timing stats inside sequence env workers"
    )
    parser.add_argument(
        "--profile_panorama_encoder", action="store_true", default=False,
        help="print timing stats inside NonHistoryPanoramaModel.encoder"
    )
    parser.add_argument(
        "--profile_interval", type=int, default=10,
        help="print profiling stats every N steps when a profiler is enabled"
    )
    parser.add_argument(
        "--diversity_only", action="store_true", default=False
    )
    parser.add_argument("--diver_coeff", type=float, default=0.1)
    parser.add_argument(
        "--with_penalty", action="store_true", default=False
    )
    # reward curriculum
    parser.add_argument("--r1_coeff", type=float, default=1)
    parser.add_argument("--r2_coeff", type=float, default=0.01)
    parser.add_argument(
        "--curriculum", action="store_true", default=False
    )
    # for topo reward
    parser.add_argument(
        "--check_target", action="store_true", default=False
    )
    # sample locs
    parser.add_argument(
        "--sampled_dir",
        type=str,
        default="",
        help="path for sampled data",
    )
    parser.add_argument(
        "--sample_mode", action="store_true", default=False
    )
    # for panorama
    parser.add_argument(
        "--panorama", action="store_true", default=False
    )
    parser.add_argument('-els', '--max_episode_length_straight', type=int, default=25,
                        help="""Maximum episode length, steps in an episode, in straight envs""")
    # for supervised training
    parser.add_argument(
        "--use_supervised", action="store_true", default=False,
        help="Enable supervised training with expert predictions"
    )
    parser.add_argument(
        "--use_semantic_score", type=int, default=0,
        help="Whether to use semantic (CLIP) scores in expert predictor (default: 0)"
    )
    # for history policy
    parser.add_argument(
        "--use_history_policy", action="store_true", default=False,
        help="Use history-based policy (default: True)"
    )
    parser.add_argument(
        "--use_action_history_token", action="store_true", default=False,
        help=(
            "Keep one recurrent token of previously selected panorama views "
            "and jointly encode it with the 12 current view tokens"
        ),
    )
    # parse arguments
    args = parser.parse_args()

    # frame size
    if not args.eval:
        args.env_frame_height = args.det_frame_height
        args.env_frame_width = args.det_frame_width
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        if args.auto_gpu_config:
            num_gpus = torch.cuda.device_count()
            if args.total_num_scenes != "auto":
                args.total_num_scenes = int(args.total_num_scenes)
            elif "objectnav_gibson" in args.task_config and \
                    "train" in args.split:
                args.total_num_scenes = 25
            elif "objectnav_gibson" in args.task_config and \
                    "val" in args.split:
                args.total_num_scenes = 5
            else:
                assert False, "Unknown task config, please specify" + \
                    " total_num_scenes"

            # GPU Memory required for the SemExp model:         # TODO
            #       0.8 + 0.4 * args.total_num_scenes (GB)
            # GPU Memory required per thread: 2.6 (GB) _ > 2.4
            m_per_thread = 2.6
            min_memory_required =max(0.8 + 0.4 * args.total_num_scenes, m_per_thread)
            # max(0.8 + 0.4 * args.total_num_scenes, m_per_thread)   # TODO
            # Automatically configure number of training threads based on
            # number of GPUs available and GPU memory size
            gpu_memory = 1000
            for i in range(num_gpus):
                gpu_memory = min(gpu_memory,
                                 torch.cuda.get_device_properties(
                                     i).total_memory
                                 / 1024 / 1024 / 1024)
                assert gpu_memory > min_memory_required, \
                    """Insufficient GPU memory for GPU {}, gpu memory ({}GB)
                    needs to be greater than {}GB""".format(
                        i, gpu_memory, min_memory_required)

            num_processes_per_gpu = int(gpu_memory / m_per_thread)
            num_processes_on_first_gpu = \
                int((gpu_memory - min_memory_required) / m_per_thread)

            if args.eval:
                max_threads = num_processes_per_gpu * (num_gpus - 1) \
                    + num_processes_on_first_gpu
                assert max_threads >= args.total_num_scenes, \
                    f"""Insufficient GPU memory for evaluation, max threads is {max_threads}"""

            if num_gpus == 1:
                args.num_processes_on_first_gpu = num_processes_on_first_gpu
                args.num_processes_per_gpu = 0
                args.num_processes = num_processes_on_first_gpu
                assert args.num_processes > 0, "Insufficient GPU memory"
            else:
                num_threads = num_processes_per_gpu * (num_gpus - 1) \
                    + num_processes_on_first_gpu
                num_threads = min(num_threads, args.total_num_scenes)
                args.num_processes_per_gpu = num_processes_per_gpu
                args.num_processes_on_first_gpu = max(
                    0,
                    num_threads - args.num_processes_per_gpu * (num_gpus - 1))
                args.num_processes = num_threads
        

            print("Auto GPU config:")
            print("Number of processes: {}".format(args.num_processes))
            print("Number of processes on GPU 0: {}".format(
                args.num_processes_on_first_gpu))
            print("Number of processes per GPU: {}".format(
                args.num_processes_per_gpu))
            
    else:
        args.sem_gpu_id = -2
    
    if args.num_mini_batch == "auto":
        args.num_mini_batch = max(args.num_processes // 2 , 1)
    else:
        args.num_mini_batch = int(args.num_mini_batch)

    return args
