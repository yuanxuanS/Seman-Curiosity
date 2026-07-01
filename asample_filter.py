import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple


def parse_image_name(filename: str):
    """
    Parse image filename to extract n and m from zfteglbstep_{n}_move_{m}.jpg format.
    
    Returns:
        Tuple of (n, m) or None if filename doesn't match pattern
    """
    pattern = r'zfte4_zfteglbstep_(\d+)_move_(\d+)\.jpg'
    match = re.search(pattern, filename)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


def load_selected_samples(json_path: Path) -> Set[Tuple[int, int]]:
    """
    Load selected samples from JSON file.
    
    Expected format: {n: [m1, m2, ...], ...}
    Returns set of (n, m) tuples representing selected images.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    selected_set = set()
    for n_str, m_list in data.items():
        n = int(n_str)
        for m in m_list:
            selected_set.add((n, m))
    
    return selected_set


def filter_coco_json(
    json_path: Path,
    selected_samples: Set[Tuple[int, int]],
    output_path: Path,
) -> None:
    """
    Filter COCO JSON file to keep only selected images and their annotations.
    
    Args:
        json_path: Path to input COCO JSON file
        selected_samples: Set of (n, m) tuples for selected images
        output_path: Path to save filtered JSON
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    if 'images' not in coco_data:
        raise ValueError(f"Invalid COCO format: missing 'images' in {json_path}")
    
    # Filter images and build mapping of old image_id to new image_id
    selected_image_ids = {}
    filtered_images = []
    
    for image_entry in coco_data['images']:
        filename = image_entry.get('file_name', '')
        parsed = parse_image_name(filename)
        
        if parsed and parsed in selected_samples:
            old_id = image_entry['id']
            filtered_images.append(image_entry)
            selected_image_ids[old_id] = image_entry['id']
            print(f"Kept image: {filename}")
        else:
            if parsed:
                print(f"Filtered out: {filename}")
            else:
                print(f"Skipped (non-matching format): {filename}")
    
    print(f"\nKept {len(filtered_images)} images out of {len(coco_data['images'])}")
    
    # Filter annotations to keep only those for selected images
    filtered_annotations = []
    if 'annotations' in coco_data:
        for ann in coco_data['annotations']:
            if ann.get('image_id') in selected_image_ids:
                filtered_annotations.append(ann)
    
    print(f"Kept {len(filtered_annotations)} annotations out of {len(coco_data.get('annotations', []))}")
    
    # Build filtered COCO data
    filtered_coco = {
        'images': filtered_images,
        'annotations': filtered_annotations,
    }
    
    # Preserve other fields (categories, info, etc.)
    for key in coco_data:
        if key not in ['images', 'annotations']:
            filtered_coco[key] = coco_data[key]
    
    # Save filtered JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_coco, f, indent=2, ensure_ascii=False)
    
    print(f"\nFiltered JSON saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter COCO JSON to keep only selected images"
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Path to folder containing selected_samples.json and COCO JSON file",
    )
    parser.add_argument(
        "--selected-json",
        type=str,
        default="selected_samples.json",
        help="Name of selected samples JSON file (default: selected_samples.json)",
    )
    parser.add_argument(
        "--coco-json",
        type=str,
        default="traj_0528_4_backup.json",
        help="Name of COCO annotation JSON file (default: traj_0528_4_backup.json)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="traj_0528_4_filtered.json",
        help="Name of output filtered JSON file (default: traj_0528_4_filtered.json)",
    )
    args = parser.parse_args()
    
    image_dir = args.image_dir.resolve()
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")
    
    selected_json_path = image_dir / args.selected_json
    coco_json_path = image_dir / args.coco_json
    output_json_path = image_dir / args.output_json
    
    if not selected_json_path.exists():
        raise FileNotFoundError(f"Selected samples file not found: {selected_json_path}")
    if not coco_json_path.exists():
        raise FileNotFoundError(f"COCO JSON file not found: {coco_json_path}")
    
    print(f"Directory: {image_dir}")
    print(f"Selected samples file: {selected_json_path.name}")
    print(f"COCO JSON file: {coco_json_path.name}")
    print(f"Output file: {output_json_path.name}\n")
    
    # Load selected samples
    print("--- Loading selected samples ---")
    selected_samples = load_selected_samples(selected_json_path)
    print(f"Loaded {len(selected_samples)} selected samples")
    for n, m in sorted(selected_samples)[:10]:
        print(f"  (n={n}, m={m})")
    if len(selected_samples) > 10:
        print(f"  ... and {len(selected_samples) - 10} more")
    
    # Filter COCO JSON
    print("\n--- Filtering COCO JSON ---")
    filter_coco_json(coco_json_path, selected_samples, output_json_path)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
