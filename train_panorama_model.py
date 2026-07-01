"""
Supervised training script for panorama_model (from src/policy_rl/panorama_model_non.py).

Task: Given a panoramic observation (12 RGB views), predict the next action direction.
Labels: Expert actions computed from agent position trajectories.
        For each step, we compute which of the 12 view directions (30° apart)
        the agent should turn to face the next goal/trajectory point.

Usage:
    python train_panorama_model.py --data_path /path/to/episodes_data
    python train_panorama_model.py --data_path /path/to/episodes_data --epochs 50 --batch_size 32 --lr 1e-4
"""

import os
import sys
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm
import gym

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.policy_rl.panorama_model_non import (
    panorama_model,
    ModelConfig,
    model_config as default_model_config,
    get_all_point_angle_feature,
)
from src.policy_rl.model import RL_Policy2
from src.finetune.dataset_utils import SampleLoader
from src.finetune.sensors_data import RGBSense, AgentPoseSense
import glob
import pickle


# =============================================================================
# Dataset
# =============================================================================

class PanoramaActionDataset(Dataset):
    """
    Dataset that loads panoramic observations and computes expert actions.

        For each (env, episode, step) sample:
                - Load 12 RGB views (panoramic image)
                - Load a precomputed 12-dim expert probability distribution from pkl
                    using the sample index (env, episode, step).
    """

    RGB_TYPES = ['rgb', 'rgb_30', 'rgb_60', 'rgb_90', 'rgb_120', 'rgb_150',
                 'rgb_180', 'rgb_210', 'rgb_240', 'rgb_270', 'rgb_300', 'rgb_330']

    def __init__(self, data_path, config, device, sampler=None, expert_gt_cache=None, max_steps=None,
                 train=True, train_ratio=0.6,
                 normalize_images=True):
        """
        Args:
            data_path: Path to episode data directory containing .npy files
            config: ModelConfig with hidden_size, angle_feat_size, etc.
            device: torch device
            max_steps: Maximum number of samples to load (None = load all)
            train: If True, use first 80% of episodes for training; else use last 20%
            train_ratio: Split ratio for train/val
            normalize_images: Whether to normalize images to [0, 1]
        """
        self.data_path = data_path
        self.config = config
        self.device = device
        self.normalize_images = normalize_images
        self.expert_gt_cache = expert_gt_cache

        # Load sampler (can be shared by train/val to avoid repeated expensive init)
        self.sampler = sampler if sampler is not None else SampleLoader(data_path, glbstep=True)
        self.samples = self._build_sample_list()
        
        # Split train/val
        total = len(self.samples)
        split_idx = int(total * train_ratio)
        if train:
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

        if max_steps is not None:
            self.samples = self.samples[:max_steps]

        print(f"[{'Train' if train else 'Val'}] Dataset loaded: {len(self.samples)} samples from {data_path}")

    def _build_sample_list(self):
        """Build a list of (env, episode, step) tuples."""
        env_list, episode_list, steps_list = \
            self.sampler.get_env_episode_and_steps_dense_list(more_mode=True)
        
        samples = []
        for e, ep, step in zip(env_list, episode_list, steps_list):
            samples.append((e, ep, step))
        return samples

    def _get_panorama_images(self, env, episode, step):
        """
        Load 12 RGB views for the given sample.
        Returns: torch.Tensor of shape (12, H, W, 3), dtype float32
        """
        modalities = list(self.RGB_TYPES)
        sample_data = self.sampler.get_sample_multimodality(env, episode, step, modalities)

        images = []
        for rgb_key in self.RGB_TYPES:
            img_data = sample_data[rgb_key].data  # H x W x 3, uint8
            img = img_data.astype(np.float32)
            if self.normalize_images:
                img = img / 255.0
            images.append(img)

        images = np.stack(images, axis=0)  # (12, H, W, 3)
        return torch.from_numpy(images)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Returns (panorama_images, expert_probs)."""
        env, episode, step = self.samples[idx]
        
        # Load panorama images
        images = self._get_panorama_images(env, episode, step)

        if self.expert_gt_cache is None:
            raise RuntimeError("expert_gt_cache is required to load training targets")

        try:
            expert_probs = self.expert_gt_cache[int(env)][int(episode)][int(step)]
        except KeyError as exc:
            raise KeyError(f"Missing expert_probs for env={env}, episode={episode}, step={step}") from exc

        return images, torch.from_numpy(np.asarray(expert_probs, dtype=np.float32))


# =============================================================================
# Model wrapper with action head
# =============================================================================

class PanoramaModelWithActionHead(nn.Module):
    """
    Uses the full RL_Policy2 model and takes action probabilities from the
    model's direct per-view action logits.
    """

    def __init__(self, device, num_actions=12, use_history=False):
        super().__init__()
        obs_shape = (12, 224, 224, 3)
        action_space = gym.spaces.Discrete(num_actions)
        self.rl_policy = RL_Policy2(
            obs_shape,
            action_space,
            device=device,
            use_history=use_history,
        )
        self.num_actions = num_actions

    def forward(self, obs):
        """
        Args:
            obs: panoramic observation, shape (batch, num_views, H, W, C)
        Returns:
            action_probs: (batch, num_actions)
        """
        _, action_logits = self.rl_policy(obs)
        return torch.softmax(action_logits, dim=-1)


# =============================================================================
# Training utilities
# =============================================================================

def compute_accuracy(preds, targets):
    """Compute top-1 and top-5 accuracy."""
    target_actions = targets.argmax(dim=1)
    top1 = (preds.argmax(dim=1) == target_actions).float().mean()
    top5 = preds.topk(5, dim=1)[1]  # (batch, 5)
    top5_match = (top5 == target_actions.unsqueeze(1)).any(dim=1).float().mean()
    return top1.item(), top5_match.item()


# =============================================================================
# Main training script
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train panorama_model with supervised action labels')
    
    # Data
    parser.add_argument('--data_path', type=str,
                        default="data_scene/pano_data/data",
                        help='Path to episode data directory')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Max number of training samples (None = all)')
    parser.add_argument('--max_val_samples', type=int, default=1000,
                        help='Max number of validation samples')
    
    # Model
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--image_feat_size', type=int, default=768)
    parser.add_argument('--angle_feat_size', type=int, default=2)
    parser.add_argument('--num_attention_heads', type=int, default=12)
    parser.add_argument('--num_actions', type=int, default=12,
                        help='Number of action directions (default: 12 for 30° views)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Device
    parser.add_argument('--device', type=str, default='cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # Logging / Checkpoint
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Print training stats every N batches')
    parser.add_argument('--save_dir', type=str, default='./exps/panorama_supervised/1',
                        help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--expert_gt_path', type=str, required=True,
                        help='Path to precomputed expert ground-truth pkl file')
    
    # Misc
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    return args


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Build model config
    config = ModelConfig(
        hidden_size=args.hidden_size,
        image_feat_size=args.image_feat_size,
        angle_feat_size=args.angle_feat_size,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        num_attention_heads=args.num_attention_heads,
        pred_head_dropout_prob=args.dropout,
        layer_norm_eps=1e-12,
        output_attentions=False,
    )
    
    # Build model wrapper based on RL_Policy2
    print("Loading RL_Policy2...")
    
    model = PanoramaModelWithActionHead(
        device=device,
        num_actions=args.num_actions,
        use_history=False,
    )
    model.to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} | Trainable: {trainable_params:,}")
    print(f"Loading expert ground-truth cache from: {args.expert_gt_path}")
    with open(args.expert_gt_path, 'rb') as f:
        expert_gt_obj = pickle.load(f)
    expert_gt_cache = expert_gt_obj['cache'] if isinstance(expert_gt_obj, dict) and 'cache' in expert_gt_obj else expert_gt_obj
    
    # Build datasets
    print("Initializing SampleLoader (shared by train/val)...")
    shared_sampler = SampleLoader(args.data_path, glbstep=False)

    train_dataset = PanoramaActionDataset(
        args.data_path, config, device,
        sampler=shared_sampler,
        expert_gt_cache=expert_gt_cache,
        max_steps=args.max_train_samples,
        train=True,
        normalize_images=True
    )
    
    val_dataset = PanoramaActionDataset(
        args.data_path, config, device,
        sampler=shared_sampler,
        expert_gt_cache=expert_gt_cache,
        max_steps=args.max_val_samples,
        train=False,
        normalize_images=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Loss: match predicted distribution to expert distribution.
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        ckpt = torch.load(args.checkpoint_path, map_location=device)

        state_dict = ckpt['model_state_dict']
        # Backward compatible loading:
        # 1) New format: RL_Policy2 state_dict directly
        # 2) Old format: wrapped PanoramaModelWithActionHead state_dict
        loaded = False
        try:
            model.rl_policy.load_state_dict(state_dict)
            loaded = True
            print("Loaded RL_Policy2 state_dict from checkpoint.")
        except Exception:
            pass

        if not loaded:
            model.load_state_dict(state_dict)
            print("Loaded wrapped model state_dict from checkpoint.")

        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch}")
    
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # ---- Training ----
        model.train()
        train_loss = 0.0
        train_top1 = 0.0
        train_top5 = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch_idx, (images, expert_probs) in enumerate(pbar):
            images = images.to(device)  # (batch, 12, H, W, 3), float32, normalized
            expert_probs = expert_probs.to(device)  # (batch, 12)
            
            optimizer.zero_grad()
            
            action_probs = model(images)  # (batch, 12)
            loss = criterion(torch.log(action_probs + 1e-8), expert_probs)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            top1, top5 = compute_accuracy(action_probs, expert_probs)
            train_top1 += top1
            train_top5 += top5
            n_batches += 1
            
            if batch_idx % args.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{train_loss / n_batches:.4f}',
                    'top1': f'{train_top1 / n_batches:.4f}',
                    'top5': f'{train_top5 / n_batches:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
        
        avg_train_loss = train_loss / max(n_batches, 1)
        avg_train_top1 = train_top1 / max(n_batches, 1)
        avg_train_top5 = train_top5 / max(n_batches, 1)
        
        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_top1 = 0.0
        val_top5 = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for images, expert_probs in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                images = images.to(device)
                expert_probs = expert_probs.to(device)
                
                action_probs = model(images)
                loss = criterion(torch.log(action_probs + 1e-8), expert_probs)
                
                val_loss += loss.item()
                top1, top5 = compute_accuracy(action_probs, expert_probs)
                val_top1 += top1
                val_top5 += top5
                n_val_batches += 1
        
        avg_val_loss = val_loss / max(n_val_batches, 1)
        avg_val_top1 = val_top1 / max(n_val_batches, 1)
        avg_val_top5 = val_top5 / max(n_val_batches, 1)
        
        # Update scheduler
        scheduler.step()
        
        # Logging
        log_msg = (
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train [loss={avg_train_loss:.4f}, top1={avg_train_top1:.4f}, top5={avg_train_top5:.4f}] | "
            f"Val [loss={avg_val_loss:.4f}, top1={avg_val_top1:.4f}, top5={avg_val_top5:.4f}] | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )
        print(log_msg)
        
        # Save checkpoint
        policy_state_dict = model.rl_policy.state_dict()
        ckpt = {
            'epoch': epoch,
            # Save RL_Policy2 weights directly so main_sequence.py can load
            # with policy.load_state_dict(checkpoint['model_state_dict']).
            'model_state_dict': model.state_dict(),
            'policy_state_dict': policy_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'train_top1': avg_train_top1,
            'val_loss': avg_val_loss,
            'val_top1': avg_val_top1,
            'args': args,
        }
        
        # Save latest
        torch.save(ckpt, os.path.join(args.save_dir, 'latest.pth'))
        
        # Save best
        if avg_val_top1 > best_val_acc:
            best_val_acc = avg_val_top1
            torch.save(ckpt, os.path.join(args.save_dir, 'best.pth'))
            print(f"  → New best model saved! Val top1: {best_val_acc:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(ckpt, os.path.join(args.save_dir, f'epoch_{epoch+1}.pth'))
    
    print(f"\nTraining complete! Best val top1 accuracy: {best_val_acc:.4f}")
    print(f"Checkpoints saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
