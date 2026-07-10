import argparse
import csv
import importlib.util
import re
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


NAME_RE = re.compile(r"epi(?P<epi>\d+)_env(?P<env>\d+)_glb_(?P<glb>\d+)_step(?P<step>\d+)\.png$")


def load_panorama_model_non(repo_root):
    """Load panorama_model_non.py without importing src/__init__.py."""
    src_mod = types.ModuleType("src")
    src_mod.__path__ = [str(repo_root / "src")]
    finetune_mod = types.ModuleType("src.finetune")
    dataset_utils_mod = types.ModuleType("src.finetune.dataset_utils")
    dataset_utils_mod.get_loader = lambda *args, **kwargs: None

    class SampleLoader:
        pass

    dataset_utils_mod.SampleLoader = SampleLoader
    sys.modules.setdefault("src", src_mod)
    sys.modules.setdefault("src.finetune", finetune_mod)
    sys.modules.setdefault("src.finetune.dataset_utils", dataset_utils_mod)

    module_path = repo_root / "src/policy_rl/panorama_model_non.py"
    spec = importlib.util.spec_from_file_location("panorama_model_non_for_analysis", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_images(data_dir):
    glb_candidates = {}
    skipped = 0
    for path in data_dir.glob("*.png"):
        match = NAME_RE.match(path.name)
        if match is None:
            skipped += 1
            continue
        meta = {key: int(value) for key, value in match.groupdict().items()}
        key = (meta["env"], meta["epi"], meta["glb"])
        prev = glb_candidates.get(key)
        if prev is None or meta["step"] < prev[0]:
            glb_candidates[key] = (meta["step"], path)

    groups = defaultdict(list)
    for (env, epi, glb), (step, path) in glb_candidates.items():
        groups[(env, epi)].append((glb, step, path))
    for key in groups:
        groups[key].sort(key=lambda item: item[0])
    return groups, skipped, len(glb_candidates)


def print_groups(groups, limit):
    print("\nParsed groups preview:")
    keys = sorted(groups)
    shown = keys if limit <= 0 else keys[:limit]
    for env, epi in shown:
        items = groups[(env, epi)]
        preview = ", ".join(
            f"glb{glb}:step{step}:{path.name}" for glb, step, path in items[:8]
        )
        if len(items) > 8:
            preview += f", ... ({len(items)} total)"
        print(f"env={env} epi={epi} n={len(items)} -> {preview}")
    if limit > 0 and len(keys) > limit:
        print(f"... {len(keys) - limit} more groups omitted")


def load_rgb_tensor(path):
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def pool_features(features, mode):
    if features.dim() == 2:
        return features
    if features.dim() == 3:
        if mode == "cls":
            return features[:, 0]
        if mode == "mean":
            return features.mean(dim=1)
        if mode == "flatten":
            return features.flatten(start_dim=1)
    raise ValueError(f"Unsupported feature shape {tuple(features.shape)} for token_pool={mode}")


@torch.no_grad()
def encode_images(model, paths, device, batch_size, token_pool):
    feats = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        batch = []
        for path in batch_paths:
            img = load_rgb_tensor(path).to(device)
            img = img.permute(2, 0, 1)
            batch.append(model.img_transforms(img))
        inputs = torch.stack(batch, dim=0)
        out = model.vit_model.forward_features(inputs)
        out = pool_features(out, token_pool)
        feats.append(out.detach().cpu())
    return torch.cat(feats, dim=0)


def distance_matrix(features, metric):
    if metric == "cosine":
        # normed = F.normalize(features.float(), dim=-1)
        # return 1.0 - torch.matmul(normed, normed.t())
        # return torch.matmul(features.float(), features.float().t())
        return torch.cos(features.float(), features.float().t())
    if metric == "euclidean":
        return torch.cdist(features.float(), features.float(), p=2)
    raise ValueError(f"Unsupported metric: {metric}")


def matrix_stats(dist):
    n = dist.size(0)
    if n <= 1:
        return {
            "n": n,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "p05": np.nan,
            "p50": np.nan,
            "p95": np.nan,
        }

    mask = ~torch.eye(n, dtype=torch.bool)
    values = dist[mask].cpu().numpy()
    return {
        "n": n,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def print_table(rows, limit=None):
    header = "env epi n min max mean std p05 p50 p95"
    print(header)
    print("-" * len(header))
    shown = rows if limit is None else rows[:limit]
    for row in shown:
        print(
            f"{row['env']:>3} {row['epi']:>3} {row['n']:>3} "
            f"{row['min']:.6f} {row['max']:.6f} {row['mean']:.6f} {row['std']:.6f} "
            f"{row['p05']:.6f} {row['p50']:.6f} {row['p95']:.6f}"
        )
    if limit is not None and len(rows) > limit:
        print(f"... {len(rows) - limit} more rows omitted")


def summarize_by_env(rows):
    env_rows = defaultdict(list)
    for row in rows:
        if row["n"] > 1:
            env_rows[row["env"]].append(row)

    print("\nBy-env summary over episode distance statistics:")
    print("env episodes min_min max_max mean_mean mean_std")
    print("-----------------------------------------------")
    for env in sorted(env_rows):
        vals = env_rows[env]
        print(
            f"{env:>3} {len(vals):>8} "
            f"{min(v['min'] for v in vals):.6f} "
            f"{max(v['max'] for v in vals):.6f} "
            f"{np.mean([v['mean'] for v in vals]):.6f} "
            f"{np.mean([v['std'] for v in vals]):.6f}"
        )


def write_csv(rows, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["env", "epi", "n", "min", "max", "mean", "std", "p05", "p50", "p95"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def main():
    parser = argparse.ArgumentParser(
        description="Analyze trajectory feature distance ranges for saved sequence images."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("exps/dump/sequence6-6-1_eval/episodes_data_imgs"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("feature_distance_analysis3"))
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--token-pool", choices=["cls", "mean", "flatten"], default="cls")
    parser.add_argument("--max-groups", type=int, default=0, help="debug limit; 0 means all env/episode groups")
    parser.add_argument("--print-groups", type=int, default=20, help="number of parsed env/episode groups to print; 0 means all")
    parser.add_argument("--print-limit", type=int, default=80)
    parser.add_argument("--save-matrices", dest="save_matrices", action="store_true", default=True)
    parser.add_argument("--no-save-matrices", dest="save_matrices", action="store_false")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    groups, skipped, selected_glbs = parse_images(args.data_dir)
    keys = sorted(groups)
    if args.max_groups > 0:
        keys = keys[:args.max_groups]

    print(f"Data dir: {args.data_dir}")
    print(f"Parsed groups: {len(groups)}; selected glb frames: {selected_glbs}; skipped unmatched files: {skipped}")
    print(f"Analyzing groups: {len(keys)}")
    print(f"Metric: {args.metric}; token_pool: {args.token_pool}; device: {args.device}")
    print(f"Save per-env/per-episode distance matrices: {args.save_matrices}")
    print_groups(groups, args.print_groups)

    pmn = load_panorama_model_non(repo_root)
    config = pmn.ModelConfig(**pmn.model_config)
    model = pmn.panorama_model(config, torch.device(args.device))
    model.eval()

    rows = []
    matrices_dir = args.output_dir / "matrices"
    for idx, key in enumerate(keys, start=1):
        env, epi = key
        glb_paths = groups[key]
        glbs = [item[0] for item in glb_paths]
        steps = [item[1] for item in glb_paths]
        paths = [item[2] for item in glb_paths]

        features = encode_images(model, paths, torch.device(args.device), args.batch_size, args.token_pool)
        dist = distance_matrix(features, args.metric)
        stats = matrix_stats(dist)
        row = {"env": env, "epi": epi, **stats}
        rows.append(row)

        if args.save_matrices:
            matrices_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                matrices_dir / f"env{env}_epi{epi}_{args.metric}_dist.npz",
                glbs=np.asarray(glbs, dtype=np.int32),
                steps=np.asarray(steps, dtype=np.int32),
                distance=dist.cpu().numpy(),
                files=np.asarray([p.name for p in paths]),
            )

        if idx % 25 == 0 or idx == len(keys):
            print(
                f"Processed {idx}/{len(keys)} groups; "
                f"env={env} epi={epi} matrix={tuple(dist.shape)}"
            )

    rows.sort(key=lambda row: (row["env"], row["epi"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.metric}_distance_stats.csv"
    write_csv(rows, csv_path)

    print(f"\nSaved stats: {csv_path}")
    print_table(rows, limit=args.print_limit)
    summarize_by_env(rows)


if __name__ == "__main__":
    main()
