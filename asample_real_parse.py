import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple
import torch
from src.policy_rl.sequence_utils import (
    add_uncertainty_or_clip_fallback,
    extract_object_tracks,
    score_boundary_missing_detections,
    score_class_changes_in_tracks,
    select_top_value_samples,
)
import cv2
import clip
from asample.constants import target_coco_categories

MOVE_NAME_RE = re.compile(r"^glbstep_(\d+)_move_(\d+)\.jpg$", re.IGNORECASE)


def collect_move_images(image_dir: Path) -> List[Path]:
    # Include both lowercase and uppercase extension.
    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.JPG"))
    return sorted(images)


def parse_groups(image_paths: List[Path]) -> Tuple[Dict[int, List[Tuple[int, Path]]], List[Path]]:
    grouped: DefaultDict[int, List[Tuple[int, Path]]] = defaultdict(list)
    unmatched: List[Path] = []

    for path in image_paths:
        if "move" not in path.name.lower():
            continue

        match = MOVE_NAME_RE.match(path.name)
        if not match:
            unmatched.append(path)
            continue

        n = int(match.group(1))
        s = int(match.group(2))
        grouped[n].append((s, path))

    # Sort s for each n.
    ordered = {n: sorted(items, key=lambda x: x[0]) for n, items in grouped.items()}
    return ordered, unmatched


def save_summary_json(grouped: Dict[int, List[Tuple[int, Path]]], output_json: Path) -> None:
    data = {
        str(n): {
            "s_list": [s for s, _ in items],
            "images": [p.name for _, p in items],
        }
        for n, items in sorted(grouped.items(), key=lambda x: x[0])
    }
    output_json.write_text(json.dumps(data, indent=2), encoding="utf-8")


def traverse_in_order(grouped: Dict[int, List[Tuple[int, Path]]]) -> None:
    for n in sorted(grouped.keys()):
        items = grouped[n]
        print(f"n={n}, count={len(items)}, s={[s for s, _ in items]}")
        for s, image_path in items:
            # Put your per-image processing code here when needed.
            print(f"  traverse -> n={n}, s={s}, image={image_path.name}")


def resolve_instances_path(image_dir: Path, instances_pth: Path = None) -> Path:
    if instances_pth is not None:
        resolved = instances_pth.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Instances file not found: {resolved}")
        return resolved

    default_candidates = [image_dir / "instances.pth", image_dir / "move_instances.pth"]
    for candidate in default_candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No instances file found under {image_dir}. Expected one of: "
        f"{default_candidates[0].name}, {default_candidates[1].name}"
    )


def extract_instances_by_n(
    grouped: Dict[int, List[Tuple[int, Path]]],
    instances_map: Dict,
    target_n: int,
    target_s: int,
):
    if target_n not in grouped:
        raise KeyError(f"target n not found: {target_n}")

    # Support case-insensitive matching for image names.
    lower_key_to_key = {
        str(k).lower(): k for k in instances_map.keys() if isinstance(k, str)
    }

    for s, image_path in grouped[target_n]:
        if s != target_s:
            continue

        image_name = image_path.name
        key = image_name if image_name in instances_map else lower_key_to_key.get(image_name.lower())
        if key is None:
            raise KeyError(f"instance not found for image: {image_name}")

        return {
            "image_name": image_name,
            "n": target_n,
            "s": target_s,
            "instances": instances_map[key],
        }

    raise KeyError(f"target s not found under n={target_n}: {target_s}")


def main() -> None:
    parser = argparse.ArgumentParser("Parse and traverse move images by glbstep and move index")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("real_data/traj_0528_4"),
        help="Input folder containing images",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional output json path. Default: <image-dir>/move_parse_summary.json",
    )
    parser.add_argument(
        "--instances-pth",
        type=Path,
        default=None,
        help="Path to instances dict pth. Default auto-search in image-dir",
    )
    parser.add_argument(
        "--save-instance-pth",
        type=Path,
        default=None,
        help="Output pth for the extracted single instance. Default: <image-dir>/instance_n{n}_s{s}.pth",
    )
    args = parser.parse_args()

    image_dir = args.image_dir.resolve()
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = collect_move_images(image_dir)
    grouped, unmatched = parse_groups(image_paths)

    if not grouped:
        raise RuntimeError(f"No matched image found with pattern glbstep_{{n}}_move_{{s}}.jpg in {image_dir}")

    print(f"Found {len(grouped)} unique n values")
    for n in sorted(grouped.keys()):
        s_list = [s for s, _ in grouped[n]]
        print(f"n={n}, s_list={s_list}")

    if unmatched:
        print("Unmatched files (contain 'move' but not standard name):")
        for path in unmatched:
            print(f"  {path.name}")

    output_json = args.save_json.resolve() if args.save_json else image_dir / "move_parse_summary.json"
    save_summary_json(grouped, output_json)
    print(f"Summary saved to: {output_json}")

    print("\nTraverse in order:")
    traverse_in_order(grouped)

    # 
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-L/14", device=device)
    text_str = [key for key in target_coco_categories.keys()]
    text = clip.tokenize([f"a photo contains a {i}" for i in text_str]).to(device)


    # 读取对应的instance.pth
    instances_path = resolve_instances_path(image_dir, args.instances_pth)
    instances_map = torch.load(instances_path, map_location="cpu")
    if not isinstance(instances_map, dict):
        raise TypeError(f"Loaded instances object is not dict: {instances_path}")
    print(f"Loaded instances dict with {len(instances_map)} keys from: {instances_path}")

    selected_num = 0
    selected = {}
    for n in sorted(grouped.keys()):
        instances = []
        rgbs = []
        for s, _ in grouped[n]:
            # 提取指定 n, s 对应的 instance
            target_instance = extract_instances_by_n(grouped, instances_map, n, s)
            instances.append(target_instance['instances'])
            
            
            # 读取对应图像
            image_path = image_dir / target_instance['image_name']
            img = cv2.imread(str(image_path))
            rgbs.append(img)
            
            print(
                f"Extracted instance -> image={target_instance['image_name']}, "
                f"n={target_instance['n']}, s={target_instance['s']}"
            )
        groups = extract_object_tracks(instances)
        groups = score_boundary_missing_detections(
            groups, len(instances), sequence_detections=instances
        )
        groups = score_class_changes_in_tracks(groups)
        
        # 从每个track中提取最高分的图像；
        groups = add_uncertainty_or_clip_fallback(
            clip_model, preprocess, text, groups, rgbs, device
        )
        
        sampled_frames = select_top_value_samples(groups)
        
        selected[n] = [sf + 1 for sf in sampled_frames]     # 和名字中的step保持一致，从1开始
        print(f"Selected samples for n={n}: {sampled_frames}")
        selected_num += len(sampled_frames)
    print(f"Total selected samples: {selected_num}")
    # 保存为txt和json
    output_txt = image_dir / "selected_samples.txt"
    output_json = image_dir / "selected_samples.json"
    output_json.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    with output_txt.open("w", encoding="utf-8") as f:
        for n in sorted(selected.keys()):
            s_list = selected[n]
            f.write(f"n={n}, selected s: {s_list}\n")
            
if __name__ == "__main__":
    main()
