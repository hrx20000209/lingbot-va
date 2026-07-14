#!/usr/bin/env python
"""Watch a training run's checkpoints/ directory and, for each new
checkpoint_step_N, run the existing teacher-forced replay/MAE/video-debug tool
(tools/eval_so101_front_wrist_replay_curve.py, generalized to N cameras) on one
training episode and one validation episode. This does not reimplement replay,
KV-cache handling, or plotting -- it only schedules the existing tool per
checkpoint so overfitting can be judged from val loss + per-joint MAE + video
quality (per the plan) instead of training loss alone.

Run this alongside script/run_va_posttrain_so101.sh, e.g. in a second terminal:
    python script/monitor_so101_checkpoints.py \
        --save-root train_out/so101_three_cubes \
        --train-episode 0 --val-episode 95
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_RE = re.compile(r"^checkpoint_step_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poll for new checkpoints and run replay-curve debug on each.")
    parser.add_argument("--save-root", type=Path, default=REPO_ROOT / "train_out" / "so101_three_cubes")
    parser.add_argument("--config-name", default="so101")
    parser.add_argument("--dataset-path", type=Path, default=Path("/data/rxhuang/three_cubes_1"),
                         help="Raw source dataset (has videos/ + data/); the lingbot-converted "
                              "output_root has no videos/ dir so cannot be used for replay decoding.")
    parser.add_argument("--train-episode", type=int, default=0)
    parser.add_argument("--val-episode", type=int, default=95)
    parser.add_argument("--poll-interval", type=float, default=30.0)
    parser.add_argument("--max-chunks", type=int, default=40)
    parser.add_argument("--once", action="store_true", help="Process all currently-existing checkpoints once, then exit.")
    return parser.parse_args()


def list_checkpoints(checkpoints_dir: Path) -> list[tuple[int, Path]]:
    if not checkpoints_dir.exists():
        return []
    found = []
    for path in checkpoints_dir.iterdir():
        match = CHECKPOINT_RE.match(path.name)
        if match and (path / "transformer").exists():
            found.append((int(match.group(1)), path))
    return sorted(found)


def run_replay(args: argparse.Namespace, checkpoint: Path, episode_id: int, split_name: str, step: int) -> None:
    output_dir = args.save_root / "debug" / f"step_{step:06d}" / split_name
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "eval_so101_front_wrist_replay_curve.py"),
        "--config-name", args.config_name,
        "--checkpoint", str(checkpoint),
        "--dataset_path", str(args.dataset_path),
        "--episode_id", str(episode_id),
        "--output_dir", str(output_dir),
        "--max_chunks", str(args.max_chunks),
    ]
    print(f"[monitor] step={step} split={split_name} episode={episode_id}: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"[monitor] WARNING: replay for step={step} split={split_name} exited with code {result.returncode}")


def main() -> None:
    args = parse_args()
    checkpoints_dir = args.save_root / "checkpoints"
    processed: set[int] = set()

    while True:
        for step, checkpoint in list_checkpoints(checkpoints_dir):
            if step in processed:
                continue
            run_replay(args, checkpoint, args.train_episode, "train_episode", step)
            run_replay(args, checkpoint, args.val_episode, "val_episode", step)
            processed.add(step)
        if args.once:
            break
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
