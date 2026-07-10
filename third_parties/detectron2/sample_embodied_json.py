#!/usr/bin/env python3
import argparse
import copy
import json
import os
import re
from collections import defaultdict


FILENAME_RE = re.compile(r"epi(?P<episode>\d+)_env(?P<env>\d+)_step(?P<step>\d+)")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sample an embodied COCO annotation json while balancing image counts "
            "across environments and episodes."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input COCO json, e.g. datasets/embodied_frontier/annotations/instances_train.json",
    )
    parser.add_argument(
        "--total",
        type=int,
        required=True,
        help="Target number of images to keep.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the sampled json.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output file name. Defaults to <input_stem>_sample_<total>.json.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=None,
        help="JSON indent. Omit by default to keep the file compact.",
    )
    parser.add_argument(
        "--reset-ann-ids",
        action="store_true",
        help="Reset kept annotation ids to 0..N-1.",
    )
    return parser.parse_args()


def fair_allocate(total, keys, capacities):
    """Allocate total items as evenly as possible without exceeding capacities."""
    quotas = {key: 0 for key in keys}
    remaining = total

    while remaining > 0:
        candidates = [key for key in keys if quotas[key] < capacities[key]]
        if not candidates:
            break
        min_quota = min(quotas[key] for key in candidates)
        progressed = False
        for key in candidates:
            if remaining == 0:
                break
            if quotas[key] == min_quota and quotas[key] < capacities[key]:
                quotas[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    return quotas


def get_image_sequence_info(image, image_id_to_ann_info):
    if image["id"] in image_id_to_ann_info:
        env, episode, step = image_id_to_ann_info[image["id"]]
        return int(env), int(episode), int(step)

    match = FILENAME_RE.search(image.get("file_name", ""))
    if match:
        return (
            int(match.group("env")),
            int(match.group("episode")),
            int(match.group("step")),
        )

    raise ValueError(
        "Cannot infer env/episode/step for image id={} file_name={!r}. "
        "Expected annotation field env_episode_step or a file name like "
        "epi1_env0_step0.png.".format(image.get("id"), image.get("file_name"))
    )


def build_groups(data):
    image_id_to_ann_info = {}
    for ann in data.get("annotations", []):
        info = ann.get("env_episode_step")
        if info is None:
            continue
        if len(info) != 3:
            raise ValueError("Invalid env_episode_step in annotation id={}: {}".format(ann.get("id"), info))
        image_id_to_ann_info.setdefault(ann["image_id"], tuple(info))

    groups = defaultdict(lambda: defaultdict(list))
    for image in data.get("images", []):
        env, episode, step = get_image_sequence_info(image, image_id_to_ann_info)
        groups[env][episode].append((step, image))

    for episodes in groups.values():
        for images in episodes.values():
            images.sort(key=lambda item: (item[0], item[1]["id"]))

    return groups


def sample_image_ids(groups, target_total):
    env_keys = sorted(groups)
    env_capacities = {
        env: sum(len(images) for images in groups[env].values())
        for env in env_keys
    }
    available = sum(env_capacities.values())
    if target_total > available:
        raise ValueError(
            "Target total {} is larger than available images {}.".format(target_total, available)
        )

    env_quotas = fair_allocate(target_total, env_keys, env_capacities)
    selected_ids = set()
    stats = {}

    for env in env_keys:
        episode_keys = sorted(groups[env])
        episode_capacities = {
            episode: len(groups[env][episode])
            for episode in episode_keys
        }
        episode_quotas = fair_allocate(env_quotas[env], episode_keys, episode_capacities)
        stats[env] = {}

        for episode in episode_keys:
            keep = episode_quotas[episode]
            selected_items = groups[env][episode][:keep]
            selected_steps = [step for step, _ in selected_items]
            stats[env][episode] = {
                "count": keep,
                "min_step": min(selected_steps) if selected_steps else None,
                "max_step": max(selected_steps) if selected_steps else None,
            }
            for _, image in selected_items:
                selected_ids.add(image["id"])

    return selected_ids, stats


def make_output(data, selected_image_ids, reset_ann_ids):
    output = copy.deepcopy(data)
    output["images"] = [
        image for image in output.get("images", [])
        if image["id"] in selected_image_ids
    ]
    output["annotations"] = [
        ann for ann in output.get("annotations", [])
        if ann["image_id"] in selected_image_ids
    ]

    if reset_ann_ids:
        for new_id, ann in enumerate(output["annotations"]):
            ann["id"] = new_id

    return output


def print_stats(stats):
    print("Kept images by env/episode:")
    for env in sorted(stats):
        env_total = sum(item["count"] for item in stats[env].values())
        episode_text = ", ".join(
            "epi{}={} steps[{}:{}]".format(
                episode,
                stats[env][episode]["count"],
                stats[env][episode]["min_step"],
                stats[env][episode]["max_step"],
            )
            for episode in sorted(stats[env])
        )
        print("  env{} total={} ({})".format(env, env_total, episode_text))


def main():
    args = parse_args()
    if args.total <= 0:
        raise ValueError("--total must be positive.")

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    groups = build_groups(data)
    selected_image_ids, stats = sample_image_ids(groups, args.total)
    output = make_output(data, selected_image_ids, args.reset_ann_ids)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_name is None:
        input_stem = os.path.splitext(os.path.basename(args.input))[0]
        output_name = "{}_sample_{}.json".format(input_stem, len(output["images"]))
    else:
        output_name = args.output_name
    output_path = os.path.join(args.output_dir, output_name)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=args.indent)

    print_stats(stats)
    print("Input images: {}".format(len(data.get("images", []))))
    print("Output images: {}".format(len(output.get("images", []))))
    print("Output annotations: {}".format(len(output.get("annotations", []))))
    print("Saved to: {}".format(output_path))


if __name__ == "__main__":
    main()
