#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


USED_ACTION_CHANNEL_IDS = [0, 1, 2, 3, 4, 28]


def load_actions(dataset_path: Path) -> np.ndarray:
    paths = sorted((dataset_path / "data").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {dataset_path / 'data'}")
    actions = []
    for path in paths:
        frame = pd.read_parquet(path, columns=["action"])
        actions.append(np.stack(frame["action"].to_numpy()).astype(np.float32))
    return np.concatenate(actions, axis=0)


def mapped_stats(actions: np.ndarray) -> dict:
    if actions.ndim != 2 or actions.shape[1] != 6:
        raise ValueError(f"Expected SO101 action shape [N, 6], got {actions.shape}")
    q01_source = np.quantile(actions, 0.01, axis=0)
    q99_source = np.quantile(actions, 0.99, axis=0)
    q01 = np.zeros(30, dtype=np.float64)
    q99 = np.ones(30, dtype=np.float64)
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


def update_config(config_path: Path, stats: dict) -> None:
    text = config_path.read_text()
    replacement = (
        "va_so101_front_wrist_cfg.norm_stat = {\n"
        f"    \"q01\": {stats['q01']},\n"
        f"    \"q99\": {stats['q99']},\n"
        "}"
    )
    pattern = (
        r"va_so101_front_wrist_cfg\.norm_stat = \{\n"
        r"\s+\"q01\": .+?,\n"
        r"\s+\"q99\": .+?,\n"
        r"\}"
    )
    new_text, count = re.subn(pattern, replacement, text, flags=re.S)
    if count != 1:
        raise RuntimeError(f"Could not find norm_stat block in {config_path}")
    config_path.write_text(new_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute SO101 front+wrist LingBot action quantiles.")
    parser.add_argument("--dataset_path", type=Path, default=Path("/data/rxhuang/three_cubes_1"))
    parser.add_argument("--config_path", type=Path, default=None)
    parser.add_argument("--no_update_config", action="store_true")
    args = parser.parse_args()
    dataset_path = args.dataset_path.expanduser().resolve()
    actions = load_actions(dataset_path)
    stats = mapped_stats(actions)
    output_path = dataset_path / "so101_front_wrist_lingbot_norm_stats.json"
    output_path.write_text(json.dumps(stats, indent=2) + "\n")

    print(f"Loaded actions: {actions.shape}")
    print("va_so101_front_wrist_cfg.norm_stat = {")
    print(f"    \"q01\": {stats['q01']},")
    print(f"    \"q99\": {stats['q99']},")
    print("}")
    print(f"Wrote {output_path}")
    if args.config_path and not args.no_update_config:
        update_config(args.config_path, stats)
        print(f"Updated {args.config_path}")
    elif args.config_path:
        print(f"To update config, rerun without --no_update_config: {args.config_path}")


if __name__ == "__main__":
    main()
