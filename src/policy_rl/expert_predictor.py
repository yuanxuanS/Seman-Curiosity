"""
Expert Predictor: 从 panorama observations 预测 expert action probabilities
用于知识蒸馏训练
"""

import torch
import torch.nn.functional as F
import gym
from asample.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
from asample.models.encoders.resnet_encoders import (
    ResnetDepthEncoder,
    CLIPEncoder,
)
from asample.waypoint_pred.utils import nms


class ExpertPredictor:
    """
    Expert predictor based on waypoint prediction model.
    Takes panorama observations and outputs action probabilities (12 directions).
    Supports batch inference for multiple scenes.
    """
    
    def __init__(self, device="cuda:3", checkpoint_path="data_scene/wp_pred/check_cwp_bestdist_hfov90"):
        """
        Initialize the expert predictor.
        
        Args:
            device: torch device for model
            checkpoint_path: path to waypoint predictor checkpoint
        """
        self.device = device
        
        # Constants
        self.NUM_ANGLES = 120    # 360度划分为120个扇区，每个3度
        self.NUM_IMGS = 12        # 输入的12张环视图像
        self.NUM_CLASSES = 12     # 每个扇区预测12个距离等级 (0.25m - 3.0m)
        self.angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        
        # Initialize models
        self._init_models(checkpoint_path)
    
    def _init_models(self, checkpoint_path):
        """Initialize and load waypoint prediction models"""
        # Waypoint predictor
        self.waypoint_predictor = BinaryDistPredictor_TRM(
            device=torch.device(self.device)
        )
        self.waypoint_predictor.load_state_dict(
            torch.load(checkpoint_path, map_location=torch.device('cpu'))['predictor']['state_dict']
        )
        for param in self.waypoint_predictor.parameters():
            param.requires_grad_(False)
        self.waypoint_predictor.to(self.device)
        self.waypoint_predictor.eval()
        
        # Depth encoder
        model_config = {
            "depth_encoder": {
                "output_size": 128,
                "ddppo_checkpoint": "data_scene/ddppo-models/gibson-2plus-resnet50.pth",
                "backbone": "resnet50",
            },
            "spatial_output": False
        }
        dos = gym.spaces.Box(0., 1.0, (256, 256, 1), dtype='float32')
        
        self.depth_encoder = ResnetDepthEncoder(
            {'depth': dos},
            output_size=model_config['depth_encoder']['output_size'],
            checkpoint=model_config['depth_encoder']['ddppo_checkpoint'],
            backbone=model_config['depth_encoder']['backbone'],
            spatial_output=model_config['spatial_output'],
        ).to(self.device)
        
        # RGB encoder
        self.rgb_encoder = CLIPEncoder(self.device)
    
    def predict(self, panorama_obs_list):
        """
        Predict expert action probabilities for multiple scenes (batch inference).
        
        Args:
            panorama_obs_list: list of panorama_obs dicts, each containing:
                - 'rgb', 'rgb_30', 'rgb_60', ..., 'rgb_330'
                - 'depth', 'depth_30', 'depth_60', ..., 'depth_330'
                Each rgb image: H x W x 3
                Each depth image: H x W x 1
        
        Returns:
            expert_probs: tensor of shape [batch_size, 12] (12 direction probabilities)
        """
        with torch.no_grad():
            batch_size = len(panorama_obs_list)
            
            # Preprocess: convert input to batch format
            # Shape: [batch_size, NUM_IMGS, H, W, C]
            rgb_batch = torch.zeros(
                (batch_size, self.NUM_IMGS, 224, 224, 3), 
                dtype=torch.float32
            ).to(self.device)
            depth_batch = torch.zeros(
                (batch_size, self.NUM_IMGS, 256, 256, 1), 
                dtype=torch.float32
            ).to(self.device)
            
            # Process each scene in the batch
            for b, panorama_obs in enumerate(panorama_obs_list):
                # Convert from counter-clockwise to clockwise order
                for i, angle in enumerate(self.angles):
                    rgb_key = 'rgb' if angle == 0 else f'rgb_{angle}'
                    depth_key = 'depth' if angle == 0 else f'depth_{angle}'
                    
                    # Convert counter-clockwise index to clockwise index
                    target_idx = (self.NUM_IMGS - i) % self.NUM_IMGS
                    
                    # Get and process RGB image
                    rgb_img = torch.from_numpy(panorama_obs[rgb_key]).float()
                    if rgb_img.dim() == 3:
                        rgb_img = rgb_img.permute(2, 0, 1)  # HWC -> CHW
                    rgb_tensor = F.interpolate(
                        rgb_img.unsqueeze(0),
                        size=(224, 224),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                    rgb_batch[b, target_idx] = rgb_tensor.permute(1, 2, 0)  # CHW -> HWC
                    
                    # Get and process Depth image
                    depth_img = torch.from_numpy(panorama_obs[depth_key]).float()
                    if depth_img.dim() == 3:
                        depth_img = depth_img.permute(2, 0, 1)
                    depth_tensor = F.interpolate(
                        depth_img.unsqueeze(0),
                        size=(256, 256),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                    depth_batch[b, target_idx] = depth_tensor.permute(1, 2, 0)
            
            # Reshape for encoder: [batch_size * NUM_IMGS, C, H, W]
            batch_size_ = batch_size
            num_imgs = self.NUM_IMGS
            rgb_batch = rgb_batch.view(batch_size_ * num_imgs, 224, 224, 3)
            depth_batch = depth_batch.view(batch_size_ * num_imgs, 256, 256, 1)
            
            # Convert to CHW format for encoder
            # rgb_batch = rgb_batch.permute(0, 3, 1, 2)  # [B*12, 3, 224, 224]
            # depth_batch = depth_batch.permute(0, 3, 1, 2)  # [B*12, 1, 256, 256]
            
            # Prepare observation dict for encoders
            obs_view12 = {
                'depth': depth_batch,
                'rgb': rgb_batch
            }
            
            # Feature extraction
            depth_embedding = self.depth_encoder(obs_view12)
            rgb_embedding = self.rgb_encoder(obs_view12)
            
            # Waypoint heatmap prediction
            waypoint_heatmap_logits = self.waypoint_predictor(
                rgb_embedding, 
                depth_embedding
            )
            
            # Convert logits to probability distribution
            temperature = 1.
            temperature = max(temperature, 1e-6)
            batch_prob_map = torch.softmax(
                waypoint_heatmap_logits.reshape(
                    batch_size_, 
                    self.NUM_ANGLES * self.NUM_CLASSES
                ) / temperature,
                dim=1
            ).reshape(
                batch_size_, 
                self.NUM_ANGLES, 
                self.NUM_CLASSES
            )
            
            # Apply NMS for each scene in batch
            expert_probs_list = []
            for b in range(batch_size):
                scene_prob = batch_prob_map[b:b+1]  # [1, 120, 12]
                
                # Apply NMS
                # batch_x_norm_wrap = torch.cat((
                #     scene_prob[:, -1:, :],
                #     scene_prob,
                #     scene_prob[:, :1, :]
                # ), dim=1)
                # batch_output_map = nms(
                #     batch_x_norm_wrap.unsqueeze(1),
                #     max_predictions=5,
                #     sigma=(7.0, 5.0)
                # )
                # batch_output_map = batch_output_map.squeeze(1)[:, 1:-1, :]
                
                # Flip to counter-clockwise coordinates
                batch_output_map = scene_prob.flip(dims=[1])
                
                # Group angles: 120 -> 12 (each group has 10 angles)
                batch_output_map = batch_output_map.view(1, 12, 10, 12)
                batch_output_map = batch_output_map.sum(dim=3).sum(dim=2)
                
                # Normalize over 12 directions
                
                # scene_probs = torch.softmax(
                #     batch_output_map / temperature, 
                #     dim=1
                # )  # Shape: [1, 12]
                
                expert_probs_list.append(batch_output_map)
            
            expert_probs = torch.cat(expert_probs_list, dim=0)  # [batch_size, 12]
            
            return expert_probs
    
    def to(self, device):
        """Move models to specified device"""
        self.device = device
        self.waypoint_predictor.to(device)
        self.depth_encoder.to(device)
        self.rgb_encoder.to(device)
        return self
