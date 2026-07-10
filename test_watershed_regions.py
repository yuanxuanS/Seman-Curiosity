import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage as ndi


def load_reward_class(repo_root):
    spec = importlib.util.spec_from_file_location(
        "trajectory_reward", repo_root / "src/policy_rl/trajectory_reward.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TrajectoryFeatureReward


def room_with_door(size=120):
    free = np.zeros((size, size), dtype=bool)
    free[15:105, 10:55] = True
    free[15:105, 65:110] = True
    free[55:65, 55:65] = True
    return free


def long_corridor(size=120):
    free = np.zeros((size, size), dtype=bool)
    free[50:70, 10:110] = True
    free[20:100, 45:65] = True
    return free


def open_hall_with_pillars(size=120):
    free = np.zeros((size, size), dtype=bool)
    free[10:110, 10:110] = True
    for r, c in [(35, 35), (35, 80), (80, 35), (80, 80)]:
        free[r - 5:r + 6, c - 5:c + 6] = False
    return free


def four_rooms_cross(size=120):
    free = np.zeros((size, size), dtype=bool)
    free[10:50, 10:50] = True
    free[10:50, 70:110] = True
    free[70:110, 10:50] = True
    free[70:110, 70:110] = True
    free[50:70, 55:65] = True
    free[55:65, 50:70] = True
    return free


def irregular_apartment(size=120):
    free = np.zeros((size, size), dtype=bool)
    free[10:55, 8:55] = True
    free[60:110, 12:52] = True
    free[15:105, 58:112] = True
    free[48:68, 45:70] = True
    free[75:92, 70:92] = False
    free[25:40, 80:100] = False
    return free


def connected_regions(free):
    labels, _ = ndi.label(free)
    return labels.astype(np.int32)


def make_cases():
    return {
        "two_rooms_narrow_door": room_with_door(),
        "cross_corridor": long_corridor(),
        "open_hall_pillars": open_hall_with_pillars(),
        "four_rooms_cross": four_rooms_cross(),
        "irregular_apartment": irregular_apartment(),
    }


def plot_case(name, free, connected, watershed_regions, output_path):
    distance = ndi.distance_transform_edt(free)
    num_connected = len(np.unique(connected[connected > 0]))
    num_watershed = len(np.unique(watershed_regions[watershed_regions > 0]))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(free, cmap="gray")
    axes[0].set_title("free map")
    axes[1].imshow(distance, cmap="viridis")
    axes[1].set_title("distance transform")
    axes[2].imshow(connected, cmap="tab20")
    axes[2].set_title(f"connected regions: {num_connected}")
    axes[3].imshow(watershed_regions, cmap="tab20")
    axes[3].set_title(f"watershed regions: {num_watershed}")

    for ax in axes:
        ax.axis("off")
    fig.suptitle(name)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize watershed region segmentation on synthetic maps."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("feature_distance_analysis/watershed_cases"))
    parser.add_argument("--min-distance", type=int, default=12)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    RewardClass = load_reward_class(repo_root)
    reward = RewardClass(
        num_envs=1,
        map_resolution=5,
        device=torch.device("cpu"),
        region_method="watershed",
        watershed_min_distance=args.min_distance,
    )

    for name, free in make_cases().items():
        connected = connected_regions(free)
        watershed_regions = reward._watershed_regions(free)
        plot_case(
            name=name,
            free=free,
            connected=connected,
            watershed_regions=watershed_regions,
            output_path=args.output_dir / f"{name}.png",
        )
        print(
            f"{name}: connected={len(np.unique(connected[connected > 0]))}, "
            f"watershed={len(np.unique(watershed_regions[watershed_regions > 0]))}, "
            f"saved={args.output_dir / (name + '.png')}"
        )


if __name__ == "__main__":
    main()
