from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import measure
from skimage.segmentation import watershed


class TrajectoryFeatureReward:
    """Reward visually distinct trajectory points and penalize overly similar same-region points."""

    def __init__(
        self,
        num_envs,
        map_resolution,
        device,
        coeff=0.05,
        sim_percentile=90.0,
        min_region_points=10,
        sim_window=200,
        max_points=50,
        region_update_interval=5,
        region_method="watershed",
        watershed_min_distance=20,
        region_vis_dir=None,
    ):
        self.num_envs = int(num_envs)
        self.map_resolution = float(map_resolution)
        self.device = device
        self.coeff = float(coeff)
        self.sim_percentile = float(np.clip(sim_percentile, 0.0, 100.0))
        self.min_region_points = max(1, int(min_region_points))
        self.sim_window = max(self.min_region_points, int(sim_window))
        self.max_points = max(1, int(max_points))
        self.region_update_interval = max(1, int(region_update_interval))
        self.region_method = region_method
        self.watershed_min_distance = max(1, int(watershed_min_distance))
        self.region_vis_dir = Path(region_vis_dir) if region_vis_dir is not None else None

        self.trajs = [deque(maxlen=self.max_points) for _ in range(self.num_envs)]
        self.sim_records = [
            deque(maxlen=self.max_points * self.max_points)
            for _ in range(self.num_envs)
        ]
        self.next_point_ids = [0 for _ in range(self.num_envs)]
        self.region_maps = [None for _ in range(self.num_envs)]
        self.region_update_steps = [-1 for _ in range(self.num_envs)]
        self.last_debug = [None for _ in range(self.num_envs)]
        self.step = 0

    def reset_env(self, env_idx):
        self.trajs[env_idx].clear()
        self.sim_records[env_idx].clear()
        self.next_point_ids[env_idx] = 0
        self.region_maps[env_idx] = None
        self.region_update_steps[env_idx] = -1
        self.last_debug[env_idx] = None

    def get_similarity_stats(self):
        values = [
            record["sim"]
            for env_records in self.sim_records
            for record in env_records
        ]
        if len(values) == 0:
            return None
        return min(values), max(values)

    def _pose_to_cell(self, pose, map_shape):
        row = int(pose[1] * 100.0 / self.map_resolution)
        col = int(pose[0] * 100.0 / self.map_resolution)
        row = int(np.clip(row, 0, map_shape[0] - 1))
        col = int(np.clip(col, 0, map_shape[1] - 1))
        return row, col

    def _get_region_map(self, maps, env_idx):
        if (
            self.region_maps[env_idx] is not None
            and self.step - self.region_update_steps[env_idx] < self.region_update_interval
        ):
            return self.region_maps[env_idx]

        full_map = maps.full_map[env_idx]
        obstacle = full_map[0].detach().cpu().numpy() > 0.5
        explored = full_map[1].detach().cpu().numpy() > 0.5
        free_explored = np.logical_and(~obstacle, explored)
        # Split explored free space into local regions so visual repetition is judged locally.
        if self.region_method == "watershed":
            region_map = self._watershed_regions(free_explored)
        else:
            region_map = measure.label(free_explored, connectivity=1).astype(np.int32)

        self.region_maps[env_idx] = region_map
        self.region_update_steps[env_idx] = self.step
        self._refresh_traj_regions(env_idx, region_map)
        self._save_region_map(env_idx, region_map)
        return region_map

    def _colorize_region_map(self, region_map):
        labels = region_map.astype(np.int64)
        image = np.zeros((*labels.shape, 3), dtype=np.uint8)
        valid = labels > 0
        image[..., 0] = ((labels * 37) % 255).astype(np.uint8)
        image[..., 1] = ((labels * 67 + 41) % 255).astype(np.uint8)
        image[..., 2] = ((labels * 97 + 83) % 255).astype(np.uint8)
        image[~valid] = 0
        return image

    def _save_region_map(self, env_idx, region_map):
        if self.region_vis_dir is None:
            return
        self.region_vis_dir.mkdir(parents=True, exist_ok=True)
        image = self._colorize_region_map(region_map)
        out_path = self.region_vis_dir / f"step_{self.step:06d}_env_{env_idx}_regions.png"
        Image.fromarray(image).save(out_path)

    def _refresh_traj_regions(self, env_idx, region_map):
        # Region labels may change when the map grows; refresh stored trajectory points.
        for point in self.trajs[env_idx]:
            row, col = point["cell"]
            point["region"] = self._region_at_cell(region_map, row, col)

    def _watershed_regions(self, free_explored):
        if not np.any(free_explored):
            return np.zeros_like(free_explored, dtype=np.int32)

        distance = ndi.distance_transform_edt(free_explored)
        coords = peak_local_max(
            distance,
            min_distance=self.watershed_min_distance,
            labels=free_explored,
            exclude_border=False,
        )
        if coords.shape[0] == 0:
            return measure.label(free_explored, connectivity=1).astype(np.int32)

        # Use distance peaks as seeds, then flood outward to split large connected spaces.
        markers = np.zeros_like(distance, dtype=np.int32)
        markers[coords[:, 0], coords[:, 1]] = np.arange(1, coords.shape[0] + 1)
        region_map = watershed(-distance, markers, mask=free_explored)
        return region_map.astype(np.int32)

    def _region_at_cell(self, region_map, row, col):
        region_id = int(region_map[row, col])
        if region_id != 0:
            return region_id

        row_min = max(0, row - 2)
        row_max = min(region_map.shape[0], row + 3)
        col_min = max(0, col - 2)
        col_max = min(region_map.shape[1], col + 3)
        nearby_regions = region_map[row_min:row_max, col_min:col_max]
        valid_regions = nearby_regions[nearby_regions > 0]
        if valid_regions.size == 0:
            return 0
        return int(np.bincount(valid_regions.reshape(-1)).argmax())

    def _same_region_pair_sims(self, env_idx, point_ids):
        # Historical pairwise similarities already computed inside the same current region.
        point_ids = set(point_ids)
        return [
            record["sim"]
            for record in self.sim_records[env_idx]
            if record["a"] in point_ids and record["b"] in point_ids
        ]

    def _record_current_sims(self, env_idx, point_id, candidates, sims):
        # Only same-region similarities are recorded; cross-region pairs are not used.
        active_ids = {p["id"] for p in self.trajs[env_idx]}
        active_ids.add(point_id)
        for candidate, sim in zip(candidates, sims):
            self.sim_records[env_idx].append(
                {
                    "a": candidate["id"],  # 历史点的point id
                    "b": point_id,
                    "sim": float(sim),
                }
            )
        self.sim_records[env_idx] = deque(
            (
                record for record in self.sim_records[env_idx]
                if record["a"] in active_ids and record["b"] in active_ids
            ),
            maxlen=self.max_points * self.max_points,
        )

    def _compute_env_reward(self, env_idx, region_id, point_id, feat):
        # First restrict comparison to the current local region.
        candidates = [p for p in self.trajs[env_idx] if p["region"] == region_id]
        if len(candidates) == 0:
            self.last_debug[env_idx] = {
                "step": self.step,
                "region": region_id,
                "point_id": point_id,
                "num_candidates": 0,
                "num_hist_sims": 0,
                "raw_reward": 0.0,
                "reason": "no_candidates",
            }
            return 0.0

        candidate_ids = [p["id"] for p in candidates]
        same_region_sims = self._same_region_pair_sims(  # 同一区域的历史traj对的相似度
            env_idx, candidate_ids
        )

        # Current point is compared only to same-region history, then cached for future steps.
        hist_feats = torch.stack([p["feat"] for p in candidates], dim=0)
        sims = torch.matmul(hist_feats, feat)
        max_sim = float(torch.max(sims).item())
        self._record_current_sims(env_idx, point_id, candidates, sims.tolist())

        if len(same_region_sims) < self.min_region_points:
            self.last_debug[env_idx] = {
                "step": self.step,
                "region": region_id,
                "point_id": point_id,
                "num_candidates": len(candidates),
                "num_hist_sims": len(same_region_sims),
                "max_sim": max_sim,
                "raw_reward": 0.0,
                "reason": "not_enough_history",
            }
            return 0.0

        sim_values = np.asarray(same_region_sims, dtype=np.float32)
        # Adaptive threshold from this region's historical pairwise similarity distribution.
        threshold = float(np.percentile(sim_values, self.sim_percentile))
        sim_low = float(np.min(sim_values))
        sim_high = float(np.max(sim_values))

        denom = max(sim_high - sim_low, 1e-6)
        if max_sim <= threshold:
            # Positive reward: less-than-usual similarity means this point is visually distinct.
            normalized_gap = threshold - max_sim
            raw_reward = 0.1 * min(normalized_gap / denom, 1.0)
            self.last_debug[env_idx] = {
                "step": self.step,
                "region": region_id,
                "point_id": point_id,
                "num_candidates": len(candidates),
                "num_hist_sims": len(same_region_sims),
                "max_sim": max_sim,
                "threshold": threshold,
                "sim_min": sim_low,
                "sim_max": sim_high,
                "raw_reward": raw_reward,
                "reason": "distinct_bonus",
            }
            return raw_reward

        normalized_excess = max_sim - threshold
        # Negative reward: more-than-usual similarity means stronger repetition penalty.
        raw_reward = -min(normalized_excess / denom, 1.0)
        self.last_debug[env_idx] = {
            "step": self.step,
            "region": region_id,
            "point_id": point_id,
            "num_candidates": len(candidates),
            "num_hist_sims": len(same_region_sims),
            "max_sim": max_sim,
            "threshold": threshold,
            "sim_min": sim_low,
            "sim_max": sim_high,
            "raw_reward": raw_reward,
            "reason": "similarity_penalty",
        }
        return raw_reward

    def update_and_compute(self, maps, actions, img_feats, done=None):
        rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        if img_feats is None:
            return rewards

        actions_np = np.asarray(actions.detach().cpu() if torch.is_tensor(actions) else actions)
        actions_np = actions_np.reshape(-1)
        done_flags = [False] * self.num_envs if done is None else list(done)
        poses = maps.full_pose.detach().cpu().numpy()
        feats = img_feats.detach().cpu()

        for env_idx in range(self.num_envs):
            if done_flags[env_idx]:
                self.reset_env(env_idx)
                continue

            region_map = self._get_region_map(maps, env_idx)
            row, col = self._pose_to_cell(poses[env_idx], region_map.shape)
            region_id = self._region_at_cell(region_map, row, col)
            if region_id == 0:
                self.last_debug[env_idx] = {
                    "step": self.step,
                    "region": 0,
                    "raw_reward": 0.0,
                    "reason": "invalid_region",
                }
                continue

            action_idx = int(actions_np[env_idx])
            action_idx = int(np.clip(action_idx, 0, feats.size(1) - 1))
            # Normalize so dot products below are cosine similarities.
            feat = F.normalize(feats[env_idx, action_idx].float(), dim=0)
            pos_xy = poses[env_idx, :2].astype(np.float32)
            point_id = self.next_point_ids[env_idx]
            self.next_point_ids[env_idx] += 1

            env_reward = self._compute_env_reward(env_idx, region_id, point_id, feat)
            rewards[env_idx] = self.coeff * env_reward
            if self.last_debug[env_idx] is not None:
                self.last_debug[env_idx]["scaled_reward"] = float(rewards[env_idx].item())
                self.last_debug[env_idx]["coeff"] = self.coeff
                self.last_debug[env_idx]["action"] = action_idx
                self.last_debug[env_idx]["cell"] = (row, col)

            self.trajs[env_idx].append(
                {
                    "id": point_id,
                    "step": self.step,
                    "pos": pos_xy,
                    "cell": (row, col),
                    "region": region_id,
                    "action": action_idx,
                    "feat": feat,
                }
            )

        self.step += 1
        return rewards

    def get_last_debug(self, env_idx=0):
        env_idx = int(np.clip(env_idx, 0, self.num_envs - 1))
        return self.last_debug[env_idx]
