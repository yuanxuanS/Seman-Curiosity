#!/usr/bin/env python3
import argparse
import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

EPISODE_RE = re.compile(
    r"^\s*episode over in\s+(?P<step>\d+)\s+step,\s+(?P<local_step>\d+)\s+local step",
    re.IGNORECASE,
)
TRAJ_DEBUG_RE = re.compile(r"\[traj_reward_debug\].*?\bscaled_reward=(?P<value>{})".format(FLOAT))
VALUE_PATTERNS = {
    "map_reward": re.compile(r"episode map reward\s*=\s*(?P<value>{})".format(FLOAT)),
    "not_map_cumu_reward": re.compile(
        r"episode (?:not map cumu|sequence) reward\s*=\s*(?P<value>{})".format(FLOAT)
    ),
    "all_reward": re.compile(r"episode (?:all|mean) reward\s*=\s*(?P<value>{})".format(FLOAT)),
    "traj_reward": re.compile(r"episode traj reward\s*=\s*(?P<value>{})".format(FLOAT)),
}
REWARD_KEYS = ("map_reward", "not_map_cumu_reward", "all_reward", "traj_reward")
METRIC_LABELS = {
    "map_reward": "episode map reward",
    "not_map_cumu_reward": "episode not map cumu reward",
    "all_reward": "episode all reward",
    "traj_reward": "episode traj reward",
}


def parse_log(log_path, required_keys=REWARD_KEYS):
    rows = []
    current = None
    traj_sum = 0.0
    traj_count = 0

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            traj_match = TRAJ_DEBUG_RE.search(line)
            if traj_match:
                traj_sum += float(traj_match.group("value"))
                traj_count += 1
                continue

            episode_match = EPISODE_RE.search(line)
            if episode_match:
                current = {
                    "episode": len(rows),
                    "step": int(episode_match.group("step")),
                    "local_step": int(episode_match.group("local_step")),
                    "traj_reward": traj_sum if traj_count > 0 else None,
                    "traj_reward_source": "debug_sum" if traj_count > 0 else "missing",
                }
                traj_sum = 0.0
                traj_count = 0
                continue

            if current is None:
                continue

            for key, pattern in VALUE_PATTERNS.items():
                value_match = pattern.search(line)
                if value_match:
                    current[key] = float(value_match.group("value"))
                    if key == "traj_reward":
                        current["traj_reward_source"] = "episode_line"
                    break

            if all(current.get(key) is not None for key in required_keys):
                rows.append(current)
                current = None

    return rows


def write_csv(rows, csv_path):
    fields = [
        "episode",
        "step",
        "local_step",
        "map_reward",
        "not_map_cumu_reward",
        "all_reward",
        "traj_reward",
        "traj_reward_source",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def plot_rows(rows, out_path, x_axis, metrics):
    if x_axis == "step":
        x = [row["step"] for row in rows]
        xlabel = "global step"
    else:
        x = [row["episode"] for row in rows]
        xlabel = "episode"

    series = [(metric, METRIC_LABELS[metric]) for metric in metrics]

    fig_height = max(4, 2.5 * len(series))
    fig, axes = plt.subplots(len(series), 1, figsize=(14, fig_height), sharex=True)
    if len(series) == 1:
        axes = [axes]
    for ax, (key, label) in zip(axes, series):
        ax.plot(x, [row[key] for row in rows], linewidth=1.2)
        ax.set_ylabel("reward")
        ax.set_title(label)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    axes[-1].set_xlabel(xlabel)
    fig.suptitle("Episode reward trends", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Parse episode reward summaries from rl_sequence logs and plot reward trends."
    )
    parser.add_argument("--log", default="rl_sequencev2_test.log", help="Path to the log file.")
    parser.add_argument("--out", default=None, help="Output figure path. Default: <log_stem>_rewards.png")
    parser.add_argument("--csv", default=None, help="Output CSV path. Default: <log_stem>_rewards.csv")
    parser.add_argument("--x", choices=("step", "episode"), default="step", help="X axis for the plot.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=REWARD_KEYS,
        default=list(REWARD_KEYS),
        help="Reward metrics to plot.",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    out_path = Path(args.out) if args.out else log_path.with_name(f"{log_path.stem}_rewards.png")
    csv_path = Path(args.csv) if args.csv else log_path.with_name(f"{log_path.stem}_rewards.csv")

    rows = parse_log(log_path, required_keys=args.metrics)
    if not rows:
        raise SystemExit(f"No complete episode reward records were parsed from {log_path}")

    write_csv(rows, csv_path)
    plot_rows(rows, out_path, args.x, args.metrics)

    print(f"Parsed {len(rows)} episodes from {log_path}")
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote figure: {out_path}")


if __name__ == "__main__":
    main()
