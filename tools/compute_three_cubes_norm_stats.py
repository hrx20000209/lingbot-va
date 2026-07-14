#!/usr/bin/env python
"""Compute q01/q99 action quantiles for the 3-camera three_cubes_1_lingbot
conversion, restricted to the training split (val episodes excluded).

Unlike tools/compute_so101_front_wrist_norm_stats.py, this writes a standalone
norm_stat.json (no config-file regex patching) so training config, deploy
config, and the SO101 client can all load the same file as a single source of
truth, and its md5 can be compared across train/server/client at run time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


USED_ACTION_CHANNEL_IDS = [0, 1, 2, 3, 4, 28]
ACTION_DIM = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute train-only SO101 three_cubes action quantiles (q01/q99)."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/data/rxhuang/three_cubes_1"),
        help="Raw LeRobot dataset with the original 6-dim per-frame actions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/rxhuang/three_cubes_1_lingbot/norm_stat.json"),
    )
    parser.add_argument(
        "--val-episodes",
        default="95-99",
        help="Inclusive episode range held out from quantile computation, e.g. '95-99'.",
    )
    return parser.parse_args()


def parse_episode_range(spec: str) -> set[int]:
    start, end = spec.split("-")
    return set(range(int(start), int(end) + 1))


def load_train_actions(source_root: Path, val_episode_ids: set[int]) -> np.ndarray:
    paths = sorted((source_root / "data").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {source_root / 'data'}")
    actions = []
    for path in paths:
        frame = pd.read_parquet(path, columns=["episode_index", "action"])
        frame = frame[~frame["episode_index"].isin(val_episode_ids)]
        if len(frame):
            actions.append(np.stack(frame["action"].to_numpy()).astype(np.float64))
    if not actions:
        raise ValueError("No training-split frames found; check --val-episodes range.")
    return np.concatenate(actions, axis=0)


def mapped_stats(actions: np.ndarray) -> dict:
    if actions.ndim != 2 or actions.shape[1] != 6:
        raise ValueError(f"Expected SO101 action shape [N, 6], got {actions.shape}")
    q01_source = np.quantile(actions, 0.01, axis=0)
    q99_source = np.quantile(actions, 0.99, axis=0)
    q01 = np.zeros(ACTION_DIM, dtype=np.float64)
    q99 = np.ones(ACTION_DIM, dtype=np.float64)
    for source_index, model_index in enumerate(USED_ACTION_CHANNEL_IDS):
        q01[model_index] = q01_source[source_index]
        q99[model_index] = q99_source[source_index]
    return {
        "used_action_channel_ids": USED_ACTION_CHANNEL_IDS,
        "q01_source": q01_source.tolist(),
        "q99_source": q99_source.tolist(),
        "q01": q01.tolist(),
        "q99": q99.tolist(),
    }


def main() -> None:
    args = parse_args()
    val_episode_ids = parse_episode_range(args.val_episodes)
    actions = load_train_actions(args.source_root, val_episode_ids)
    stats = mapped_stats(actions)
    stats["train_frame_count"] = int(actions.shape[0])
    stats["val_episode_ids"] = sorted(val_episode_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote {args.output}")
    print(f"  train_frame_count={stats['train_frame_count']} (val episodes excluded: {stats['val_episode_ids']})")
    print(f"  q01_source={stats['q01_source']}")
    print(f"  q99_source={stats['q99_source']}")


if __name__ == "__main__":
    main()
