"""
Expert Predictor: 从 panorama observations 预测 expert action probabilities
用于知识蒸馏训练
"""

import torch
import torch.nn.functional as F
import gym
import clip
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from asample.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
from asample.models.encoders.resnet_encoders import (
    ResnetDepthEncoder,
    CLIPEncoder,
)
from asample.waypoint_pred.utils import nms
from src.vqf_constants import target_coco_categories


class ExpertPredictor:
    """
    Expert predictor based on waypoint prediction model.
    Takes panorama observations and outputs action probabilities (12 directions).
    Supports batch inference for multiple scenes.
    """
    
    def __init__(self, device="cuda:3", checkpoint_path="data_scene/wp_pred/check_cwp_bestdist_hfov90", use_semantic_score=True):
        """
        Initialize the expert predictor.
        
        Args:
            device: torch device for model
            checkpoint_path: path to waypoint predictor checkpoint
            use_semantic_score: whether to use CLIP semantic scores for prediction
        """
        self.device = device
        self.use_semantic_score = use_semantic_score
        
        # Constants
        self.NUM_ANGLES = 120    # 360度划分为120个扇区，每个3度
        self.NUM_IMGS = 12        # 输入的12张环视图像
        self.NUM_CLASSES = 12     # 每个扇区预测12个距离等级 (0.25m - 3.0m)
        self.angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        
        # CLIP model for semantic scoring (only when use_semantic_score=True)
        if self.use_semantic_score:
            print("Loading CLIP model...")
            self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=self.device)
            print("CLIP model loaded.")
            self.clip_model.eval()
            
            # Target category names from target_coco_categories
            self.target_category_names = list(target_coco_categories.keys())
            
            # Tokenize text for target categories
            self.clip_text_tokens = self._tokenize_text()
            
            # CLIP preprocess (same as sequence.py)
            self.clip_preprocess = Compose([
                Resize(224, interpolation=Image.BICUBIC),
                CenterCrop(224),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        else:
            self.clip_model = None
            self.clip_preprocess = None
            self.clip_text_tokens = None
            print("CLIP model disabled (use_semantic_score=False).")
        
        # Initialize models
        self._init_models(checkpoint_path)
    
    def _tokenize_text(self):
        """Tokenize text prompts for target categories."""
        text_prompts = [f"a photo contains a {goal}." for goal in self.target_category_names]
        text_tokens = clip.tokenize(text_prompts).to(self.device)
        return text_tokens
    
    def clip_score_panorama(self, observations):
        """
        Compute CLIP semantic scores for 12 panorama images.
        
        For each of the 12 direction images:
        1. Get CLIP scores for all target categories
        2. Normalize scores (softmax)
        3. Take max probability as semantic score
        
        Returns:
            semantic_scores: tensor of shape [12] with semantic scores for each direction
        """
        NUM_IMGS = 12
        angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        
        # Collect all 12 images
        images = []
        for i, angle in enumerate(angles):
            rgb_key = 'rgb' if angle == 0 else f'rgb_{angle}'
            rgb_img = observations[rgb_key]
            
            # Convert numpy array to PIL Image
            if rgb_img.dtype != np.uint8:
                rgb_img = (rgb_img * 255).astype(np.uint8) if rgb_img.max() <= 1.0 else rgb_img.astype(np.uint8)
            
            pil_img = Image.fromarray(rgb_img)
            # Preprocess for CLIP
            clip_img = self.clip_preprocess(pil_img).to(self.device)
            images.append(clip_img)
        
        # Stack all images: [12, 3, 224, 224]
        images = torch.stack(images)
        
        # Compute CLIP scores
        with torch.no_grad():
            # Get image-text similarity logits
            logits_per_image, _ = self.clip_model(images, self.clip_text_tokens)
            
            # Convert logits to probabilities (softmax over text categories)
            probs = torch.softmax(logits_per_image, dim=1)  # [12, num_categories]
            
            # Get max probability for each image (semantic score for that direction)
            semantic_scores = probs.max(dim=1)[0]  # [12]
        
        return semantic_scores  # Shape: [12]
    
    def clip_score_panorama_batch(self, panorama_obs_list):
        """
        Compute CLIP semantic scores for multiple panorama observations in batch.
        
        Args:
            panorama_obs_list: list of panorama_obs dicts (batch of scenes)
        
        Returns:
            semantic_scores_batch: tensor of shape [batch_size, 12] with semantic scores for each direction
                             Order is clockwise to match batch_output_map
        """
        NUM_IMGS = 12
        angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        batch_size = len(panorama_obs_list)
        
        # Collect all images from all scenes: [batch_size * 12, 3, 224, 224]
        # Use clockwise order to match batch_output_map
        all_images = []
        
        for b, panorama_obs in enumerate(panorama_obs_list):
            # Collect images first, then reorder to clockwise order (same as in predict)
            scene_images = [None] * NUM_IMGS
            for i, angle in enumerate(angles):
                rgb_key = 'rgb' if angle == 0 else f'rgb_{angle}'
                rgb_img = panorama_obs[rgb_key]
                
                # Convert counter-clockwise index to clockwise index
                # target_idx = (NUM_IMGS - i) % NUM_IMGS
                target_idx = i
                
                # Convert numpy array to PIL Image
                if rgb_img.dtype != np.uint8:
                    rgb_img = (rgb_img * 255).astype(np.uint8) if rgb_img.max() <= 1.0 else rgb_img.astype(np.uint8)
                
                pil_img = Image.fromarray(rgb_img)
                # Preprocess for CLIP
                clip_img = self.clip_preprocess(pil_img).to(self.device)
                scene_images[target_idx] = clip_img
            
            # Add scene images in clockwise order
            all_images.extend(scene_images)

        
        # Stack all images: [batch_size * 12, 3, 224, 224]
        images = torch.stack(all_images)
        
        # Compute CLIP scores in batch
        with torch.no_grad():
            # Get image-text similarity logits
            logits_per_image, _ = self.clip_model(images, self.clip_text_tokens)
            
            # Convert logits to probabilities (softmax over text categories)
            probs = torch.softmax(logits_per_image, dim=1)  # [batch_size * 12, num_categories]
            
            # Get max probability for each image (semantic score for that direction)
            semantic_scores_all = probs.max(dim=1)[0]  # [batch_size * 12]
        
        # Reshape to [batch_size, 12] to get scores for each scene
        semantic_scores_batch = semantic_scores_all.view(batch_size, NUM_IMGS)
        
        return semantic_scores_batch  # Shape: [batch_size, 12], clockwise order
    
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
                    # target_idx = (self.NUM_IMGS - i) % self.NUM_IMGS
                    target_idx = i
                    
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
            
            # Compute CLIP semantic scores for ALL scenes in batch BEFORE the loop (only if use_semantic_score=True)
            if self.use_semantic_score:
                semantic_scores_batch = self.clip_score_panorama_batch(panorama_obs_list)  # [batch_size, 12]
                semantic_scores_batch = semantic_scores_batch.to(self.device)
            
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
                # batch_output_map = scene_prob.flip(dims=[1])
                
                
                # Group angles: 120 -> 12 (each group has 10 angles)
                batch_output_map = scene_prob.view(1, 12, 10, 12)
                batch_output_map = batch_output_map.sum(dim=3).sum(dim=2)
                
                # Add semantic_scores to batch_output_map before softmax (CLIP score fusion)
                # Only if use_semantic_score is True
                if self.use_semantic_score:
                    # Extract semantic scores for the current scene from the batch
                    semantic_scores = semantic_scores_batch[b]  # [12]
                    
                    # Normalize semantic_scores first (L2 normalization)
                    semantic_scores_normalized = F.normalize(semantic_scores.unsqueeze(0), p=1, dim=1).squeeze(0)
                    semantic_scores_normalized = torch.softmax(semantic_scores_normalized / 0.03, dim=0)
                    # semantic_scores_normalized = semantic_scores_normalized.flip(dims=[0])
                    # semantic_weight controls the influence of CLIP scores
                    semantic_weight = 1.0  # Can be adjusted
                    batch_output_map = batch_output_map + semantic_scores_normalized.unsqueeze(0) * semantic_weight
                
                # Final normalization over 12 directions
                batch_output_map = F.normalize(batch_output_map, p=1, dim=1)
                
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
