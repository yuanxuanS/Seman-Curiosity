from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
from .utils import visualization as vu
from src.constants import color_palette
import os
import torch
from ..envs.utils import pose as pu
from ..envs.habitat.active_cam_env import Active_cam_Env
from .utils.semantic_prediction import SemanticPredMaskRCNN as SemanticPredMaskRCNN
from src.finetune.dataset_utils import save_obs
import quaternion
from .utils.detect_utils import box_iou_calc
from detectron2.utils.visualizer import ColorMode, Visualizer

class Active_cam_Agent(Active_cam_Env):
    """The Sem_Curiosity environment agent class. A seperate Sem_Curi_Env_Agent class
    object is used for each environment thread.

    """
    def __init__(self, args, rank, config_env, dataset):

        self.args = args
        super().__init__(args, rank, config_env, dataset)
        
        #
        self.visited_vis = None
        self.last_loc = None
        self.curr_loc = None
        
        # initialize transform for RGB observations
        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((args.frame_height, args.frame_width),
                               interpolation=Image.NEAREST)])

        # initialize semantic segmentation prediction model
        if args.sem_gpu_id == -1:
            args.sem_gpu_id = config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID

        self.sem_pred = SemanticPredMaskRCNN(args)

        if args.visualize or args.print_images:
            self.legend = cv2.imread('docs/legend.png')
            self.vis_image = None
            self.rgb_vis = None
            self.goal_name = "No"
    def reset(self):
        args = self.args
        
        obs, info = super().reset()
        obs, info = self._preprocess_obs(obs, info)

        self.obs_shape = obs.shape

        # Episode initializations
        map_shape = (args.map_size_cm // args.map_resolution,
                     args.map_size_cm // args.map_resolution)
        self.visited_vis = np.zeros(map_shape)
        self.curr_loc = [args.map_size_cm / 100.0 / 2.0,
                         args.map_size_cm / 100.0 / 2.0, 0.]
        
        # visualize
        if args.visualize or args.print_images:
            self.vis_image = vu.init_vis_image_only(self.goal_name, self.legend)
        
        return obs, info
    
    def step_and_preprocess(self, action, wait_env):
        """Function responsible for taking the action and
        preprocessing observations

        Returns:
            obs (ndarray): preprocessed observations ((4+C) x H x W) ? 
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep
        """
        if wait_env > 0:
            self.last_action = None
            self.info["sensor_pose"] = [0., 0., 0.]
            self.timestep += 1
            
            return np.zeros(self.obs.shape), 0., False, self.info
        
        # act and step
        # action = action + np.ones_like(action)   # output: 0-2, add to 1-3
        action = {'action': action}
        obs, _, done, info = super().step(action)       # 4,256,256

        
        
        # preprocess obs
        obs, info = self._preprocess_obs(obs, info) 
        self.last_action = action['action']     
        self.obs = obs
        self.info = info

        
            
        return obs, 0., done, info
    
    
    
    def _preprocess_obs(self, obs, info, use_seg=True):
        args = self.args
        obs = obs.transpose(1, 2, 0)
        
        rgb_ = obs[:, :, :3]     # 256,256,3
        depth_ = obs[:, :, 3:4]
        
        
        if args.det_frame_height != args.env_frame_height:
            # print(f"before resize {rgb_.shape}")
            rgb = cv2.resize(rgb_, (args.det_frame_height, args.det_frame_width))   #, rgb_.shape[-1]))
            # print(f"after resize {rgb.shape}")
            depth = cv2.resize(depth_, (args.det_frame_height, args.det_frame_width))[..., None] #, 1))
        else:
            rgb = rgb_
            depth = depth_
        del rgb_
        del depth_

        sem_seg_pred = self._get_sem_pred(
            rgb.astype(np.uint8), use_seg=use_seg)

        depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)

        ds = args.det_frame_width // args.frame_width  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)
        
        state = np.concatenate((rgb, depth, sem_seg_pred),
                               axis=2).transpose(2, 0, 1)
        return state, info
    
    def obj_exist(self, instances_1, instances_2):
        """
        Check if instances from instances_1 exist in instances_2 by comparing their bounding boxes.
        
        Args:
            instances_1: First set of instances (Instances object) to check.
            instances_2: Second set of instances (Instances object) to compare against.
        
        Returns:
            A boolean array indicating whether each instance in instances_1 exists in instances_2 (IoU > 0.5).
        """
        '''
        instances_1: Instances(n), n is the number of instances
        check if every instances_1 exists in instances_2
        return [bool, bool, ...] of length n
        '''
        
        width, height = instances_1.image_size
        pot_mp = np.zeros((height, width, 1))
        v = Visualizer(pot_mp)
        
        boxes_1 = instances_1.pred_boxes
        boxes_1 = v._convert_boxes(boxes_1)
                
        boxes_2 = instances_2.pred_boxes
        boxes_2 = v._convert_boxes(boxes_2)
        iou = box_iou_calc(boxes_1, boxes_2) # 1*num_maskbox
        is_exists = iou > 0.5
        
        return is_exists
    
    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 100.0
        depth = min_d * 100.0 + depth * max_d * 100.0
        return depth
    
    def _get_sem_pred(self, rgb, use_seg=True, return_score=False, return_instance=False):
        if use_seg:
            semantic_pred, self.rgb_vis = self.sem_pred.get_prediction(rgb, 
                                                                            return_instance=return_instance)
            semantic_pred = semantic_pred.astype(np.float32)
        else:
            semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 6))
            self.rgb_vis = rgb[:, :, ::-1]
        if not (return_instance or return_score):
            return semantic_pred
        else:
            return semantic_pred
    
    def visualize(self, vis_info):
        
        if self.args.visualize or self.args.print_images:
            vis_info['timestep']=self.timestep
            vis_info['goal_name'] = self.info['goal_name'] 
            self._visualize(vis_info)
    def _visualize(self, info, mode="full"):
        
        args = self.args
        dump_dir = "{}/dump/{}/".format(args.dump_location,
                                        args.exp_name)
        ep_dir = '{}/episodes/thread_{}/eps_{}/'.format(
            dump_dir, self.rank, self.episode_no)       # TODO, episode_no
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)


        rgb_vis = cv2.resize(self.rgb_vis, (480, 480),
                                 interpolation=cv2.INTER_NEAREST)
        self.vis_image[50:530, 15:495] = rgb_vis
        
        # text
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.3
        color = (20, 20, 20)  # BGR
        thickness = 1

        self.vis_image[:50, :] = np.ones_like(self.vis_image[:50, :]) * 255
        text = "t={}, goal {}, score:init-curr {:.2f}-{:.2f}, action {}, reward {:.2f}, ".format(info['timestep'], 
                                                                                                    info['goal_name'], 
                                                                                                    info['init score'].cpu().numpy().item(), 
                                                                                                    info['score'].cpu().numpy().item(), 
                                                                                                    info['action'].cpu().numpy().item(),
                                                                                                    info['reward'].cpu().numpy().item())
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = (480 - textsize[0]) // 2 + 15
        textY = (50 + textsize[1]) // 2
        self.vis_image = cv2.putText(self.vis_image, text, (textX, textY),
                                font, fontScale, color, thickness,
                                cv2.LINE_AA)

        if args.visualize:
            # Displaying the image
            cv2.imshow("Thread {}".format(self.rank), self.vis_image)
            cv2.waitKey(1)
            pass

        if args.print_images:
            fn = '{}/episodes/thread_{}/eps_{}/{}-{}-Vis-{}.png'.format(
                dump_dir, self.rank, self.episode_no,
                self.rank, self.episode_no, self.timestep)
            cv2.imwrite(fn, self.vis_image)
