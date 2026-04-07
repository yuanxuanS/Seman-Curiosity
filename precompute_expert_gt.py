"""
Precompute expert ground-truth distributions with ExpertPredictor.

The output is saved as a nested dictionary:
    cache[env_id][episode_id][step_id] = expert_probs (np.ndarray shape [12])

Example:
    python precompute_expert_gt.py \
        --data_path data_scene/pano_data/data/Allensville \
        --output_path data_scene/pano_data/expert_gt/Allensville_expert_gt.pkl \
        --batch_size 16
"""

import argparse
import os
import pickle
from collections import defaultdict

import numpy as np
import torch

from src.finetune.dataset_utils import SampleLoader
from src.policy_rl.expert_predictor import ExpertPredictor


RGB_TYPES = [
    'rgb', 'rgb_30', 'rgb_60', 'rgb_90', 'rgb_120', 'rgb_150',
    'rgb_180', 'rgb_210', 'rgb_240', 'rgb_270', 'rgb_300', 'rgb_330'
]


def parse_args():
    parser = argparse.ArgumentParser(description='Precompute ExpertPredictor ground truth cache')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the episode data directory containing .npy files')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save the cached expert probabilities (.pkl)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for ExpertPredictor inference')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint_path', type=str,
                        default='data_scene/wp_pred/check_cwp_bestdist_hfov90',
                        help='Waypoint predictor checkpoint path used by ExpertPredictor')
    parser.add_argument('--use_semantic_score', action='store_true',
                        help='Enable CLIP semantic scoring inside ExpertPredictor')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Optional maximum number of samples to process')
    parser.add_argument('--glbstep', action='store_true',
                        help='Read data with global-step indexing if the dataset uses it')
    return parser.parse_args()


def build_panorama_obs(sample_loader, env_id, episode_id, step_id):
    modalities = []
    for rgb_key in RGB_TYPES:
        angle = '' if rgb_key == 'rgb' else rgb_key[len('rgb_'):]
        depth_key = 'depth' if angle == '' else f'depth_{angle}'
        modalities.append(rgb_key)
        modalities.append(depth_key)

    sample_data = sample_loader.get_sample_multimodality(env_id, episode_id, step_id, modalities)
    panorama_obs = {}
    for key in modalities:
        panorama_obs[key] = sample_data[key].data
    return panorama_obs


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print(f'Loading samples from: {args.data_path}')
    sample_loader = SampleLoader(args.data_path, glbstep=args.glbstep)
    env_list, episode_list, step_list = sample_loader.get_env_episode_and_steps_dense_list(more_mode=True)

    total_samples = len(step_list)
    if args.max_samples is not None:
        total_samples = min(total_samples, args.max_samples)

    print(f'Total unique samples: {total_samples}')
    print('Loading ExpertPredictor...')
    expert_predictor = ExpertPredictor(
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        use_semantic_score=args.use_semantic_score,
    )

    env_episode_step_list = list(zip(env_list[:total_samples], episode_list[:total_samples], step_list[:total_samples]))

    cache = defaultdict(lambda: defaultdict(dict))
    processed = 0

    for start in range(0, total_samples, args.batch_size):
        batch_items = env_episode_step_list[start:start + args.batch_size]
        panorama_obs_list = []
        for env_id, episode_id, step_id in batch_items:
            panorama_obs_list.append(build_panorama_obs(sample_loader, int(env_id), int(episode_id), int(step_id)))

        expert_probs_batch = expert_predictor.predict(panorama_obs_list).detach().cpu().numpy()

        for (env_id, episode_id, step_id), expert_probs in zip(batch_items, expert_probs_batch):
            cache[int(env_id)][int(episode_id)][int(step_id)] = expert_probs.astype(np.float32)
            processed += 1

        print(f'Processed {processed}/{total_samples}')

    save_obj = {
        'meta': {
            'data_path': args.data_path,
            'checkpoint_path': args.checkpoint_path,
            'device': args.device,
            'use_semantic_score': args.use_semantic_score,
            'batch_size': args.batch_size,
            'num_samples': processed,
        },
        'cache': {env_id: dict(episode_dict) for env_id, episode_dict in cache.items()},
    }

    with open(args.output_path, 'wb') as f:
        pickle.dump(save_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'Saved expert ground-truth cache to: {args.output_path}')


if __name__ == '__main__':
    main()