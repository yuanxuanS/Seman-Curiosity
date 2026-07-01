import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List


def get_image_files(image_dir: Path) -> List[Path]:
    """Get all image files from directory (common formats)."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    images = []
    for ext in image_extensions:
        images.extend(image_dir.glob(f'*{ext}'))
        images.extend(image_dir.glob(f'*{ext.upper()}'))
    return sorted(set(images))


def find_json_file(image_dir: Path) -> Path:
    """Find COCO annotation JSON file in directory (same name as folder)."""
    folder_name = image_dir.name
    json_candidates = [
        image_dir.parent / f"{folder_name}.json",
        image_dir / f"{folder_name}.json",
        image_dir / "instances.json",
        image_dir / "annotations.json",
    ]
    for json_path in json_candidates:
        if json_path.exists():
            return json_path
    raise FileNotFoundError(f"No COCO JSON file found for {image_dir}")


def rename_images(image_dir: Path, prefix: str, dry_run: bool = False) -> Dict[str, str]:
    """
    Rename all images in directory with prefix.
    
    Args:
        image_dir: Directory containing images
        prefix: Prefix to add to image names
        dry_run: If True, only show what would be renamed without actually doing it
    
    Returns:
        Dictionary mapping old names to new names
    """
    images = get_image_files(image_dir)
    rename_map = {}
    
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")
    
    for image_path in images:
        old_name = image_path.name
        new_name = f"{prefix}{old_name}"
        new_path = image_path.parent / new_name
        
        if old_name == new_name:
            # Prefix already exists or is empty
            rename_map[old_name] = new_name
            continue
        
        if not dry_run:
            if new_path.exists():
                print(f"Warning: {new_name} already exists, skipping")
                continue
            image_path.rename(new_path)
            print(f"Renamed: {old_name} -> {new_name}")
        else:
            print(f"Would rename: {old_name} -> {new_name}")
        
        rename_map[old_name] = new_name
    
    return rename_map


def update_coco_json(json_path: Path, rename_map: Dict[str, str], backup: bool = True) -> None:
    """
    Update COCO JSON file with new image names.
    
    Args:
        json_path: Path to COCO JSON file
        rename_map: Dictionary mapping old names to new names
        backup: If True, create backup of original JSON
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    if 'images' not in coco_data:
        raise ValueError(f"Invalid COCO format: missing 'images' field in {json_path}")
    
    # Backup original
    if backup:
        backup_path = json_path.parent / f"{json_path.stem}_backup.json"
        shutil.copy(json_path, backup_path)
        print(f"Backup saved to: {backup_path}")
    
    # Update image file_name fields
    updated_count = 0
    for image_entry in coco_data['images']:
        old_filename = image_entry.get('file_name', '')
        basename = Path(old_filename).name
        
        if basename in rename_map:
            new_filename = str(Path(old_filename).parent / rename_map[basename])
            image_entry['file_name'] = new_filename
            updated_count += 1
            print(f"Updated JSON: {basename} -> {rename_map[basename]}")
    
    # Save updated JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)
    
    print(f"Updated {updated_count} entries in {json_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Add prefix to image filenames and update COCO JSON annotations"
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Path to folder containing images and COCO JSON",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Prefix to add to image filenames",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Path to COCO JSON file (auto-detected if not provided)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be renamed without actually doing it",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup of original JSON",
    )
    args = parser.parse_args()
    
    image_dir = args.image_dir.resolve()
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    # Find JSON file
    json_path = args.json.resolve() if args.json else find_json_file(image_dir)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    print(f"Processing directory: {image_dir}")
    print(f"Using JSON file: {json_path}")
    print(f"Adding prefix: '{args.prefix}'")
    
    if args.dry_run:
        print("\n[DRY RUN MODE - No changes will be made]\n")
    
    # Step 1: Rename images
    print("\n--- Step 1: Renaming images ---")
    rename_map = rename_images(image_dir, args.prefix, dry_run=args.dry_run)
    
    # Step 2: Update JSON (skip if dry-run)
    if not args.dry_run:
        print("\n--- Step 2: Updating JSON annotations ---")
        update_coco_json(json_path, rename_map, backup=not args.no_backup)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
