import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def select_pairs(distance, top_k):
    n = distance.shape[0]
    if n < 2:
        return [], []

    row_idx, col_idx = np.triu_indices(n, k=1)
    values = distance[row_idx, col_idx]
    finite = np.isfinite(values)
    row_idx = row_idx[finite]
    col_idx = col_idx[finite]
    values = values[finite]

    order_min = np.argsort(values)[:top_k]
    order_max = np.argsort(values)[-top_k:][::-1]

    min_pairs = [(int(row_idx[i]), int(col_idx[i]), float(values[i])) for i in order_min]
    max_pairs = [(int(row_idx[i]), int(col_idx[i]), float(values[i])) for i in order_max]
    return min_pairs, max_pairs


def load_image(data_dir, filename):
    return Image.open(data_dir / str(filename)).convert("RGB")


def draw_pairs(data_dir, files, glbs, steps, min_pairs, max_pairs, output_path, thumb_size):
    pairs = [("min", pair) for pair in min_pairs] + [("max", pair) for pair in max_pairs]
    if len(pairs) == 0:
        raise ValueError("No image pairs to draw.")

    fig_height = max(8, 2.1 * len(pairs))
    fig, axes = plt.subplots(len(pairs), 2, figsize=(12, fig_height), squeeze=False)

    for row, (kind, (i, j, dist)) in enumerate(pairs):
        for col, idx in enumerate((i, j)):
            image = load_image(data_dir, files[idx])
            image.thumbnail((thumb_size, thumb_size))
            axes[row, col].imshow(image)
            axes[row, col].axis("off")
            axes[row, col].set_title(
                f"{kind.upper()} #{row + 1 if kind == 'min' else row + 1 - len(min_pairs)} | "
                f"d={dist:.6f}\nidx={idx} glb={glbs[idx]} step={steps[idx]}\n{files[idx]}",
                fontsize=9,
            )

    fig.suptitle(
        f"Closest {len(min_pairs)} and Farthest {len(max_pairs)} Feature-Distance Image Pairs",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def print_pairs(label, pairs, files, glbs, steps):
    print(f"\n{label}:")
    for rank, (i, j, dist) in enumerate(pairs, start=1):
        print(
            f"{rank:02d}: d={dist:.6f} | "
            f"[{i}] glb={glbs[i]} step={steps[i]} {files[i]} <-> "
            f"[{j}] glb={glbs[j]} step={steps[j]} {files[j]}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize closest and farthest image pairs from a saved trajectory distance matrix."
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("feature_distance_analysis/matrices/env0_epi1_cosine_dist.npz"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("exps/dump/sequence6-6-1_eval/episodes_data_imgs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("feature_distance_analysis/env0_epi1_top_pairs.png"),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--thumb-size", type=int, default=256)
    args = parser.parse_args()

    data = np.load(args.matrix, allow_pickle=False)
    distance = data["distance"]
    glbs = data["glbs"]
    steps = data["steps"]
    files = data["files"].astype(str)

    min_pairs, max_pairs = select_pairs(distance, args.top_k)
    print(f"Loaded matrix: {args.matrix}")
    print(f"Matrix shape: {distance.shape}")
    print_pairs("Closest pairs", min_pairs, files, glbs, steps)
    print_pairs("Farthest pairs", max_pairs, files, glbs, steps)

    draw_pairs(
        data_dir=args.data_dir,
        files=files,
        glbs=glbs,
        steps=steps,
        min_pairs=min_pairs,
        max_pairs=max_pairs,
        output_path=args.output,
        thumb_size=args.thumb_size,
    )
    print(f"\nSaved visualization: {args.output}")


if __name__ == "__main__":
    main()
