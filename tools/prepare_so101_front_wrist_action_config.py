#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


DEFAULT_PROMPT = "Pick the green cube and place it inside the blue box."


def load_task(dataset_path: Path) -> str:
    tasks_path = dataset_path / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return DEFAULT_PROMPT
    frame = pd.read_parquet(tasks_path)
    if "task" in frame.columns:
        tasks = frame["task"].dropna().astype(str).tolist()
        return tasks[0] if tasks else DEFAULT_PROMPT
    if frame.index.name == "task" or "task" in (frame.index.names or []):
        return str(frame.index[0])
    return DEFAULT_PROMPT


def episode_lengths(dataset_path: Path) -> dict[int, int]:
    paths = sorted((dataset_path / "data").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {dataset_path / 'data'}")
    columns = ["episode_index", "frame_index"]
    data = pd.concat((pd.read_parquet(path, columns=columns) for path in paths), ignore_index=True)
    data = data.sort_values(["episode_index", "frame_index"])
    return {int(ep): int(len(group)) for ep, group in data.groupby("episode_index", sort=True)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Add SO101 action_config entries to LeRobot episodes.jsonl.")
    parser.add_argument("--dataset_path", type=Path, default=Path("/data/rxhuang/three_cubes_1"))
    parser.add_argument("--default_prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()
    dataset_path = args.dataset_path.expanduser().resolve()
    meta_dir = dataset_path / "meta"
    episodes_path = meta_dir / "episodes.jsonl"
    backup_path = meta_dir / "episodes.jsonl.bak"
    task = load_task(dataset_path) or args.default_prompt

    if episodes_path.exists():
        if not backup_path.exists():
            shutil.copy2(episodes_path, backup_path)
        rows = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    else:
        rows = []
        for episode_index, length in episode_lengths(dataset_path).items():
            rows.append(
                {
                    "episode_index": episode_index,
                    "tasks": [task],
                    "length": length,
                }
            )

    for row in rows:
        tasks = row.get("tasks") or [task]
        action_text = tasks[0] if tasks else args.default_prompt
        length = int(row.get("length") or episode_lengths(dataset_path)[int(row["episode_index"])])
        row["action_config"] = [
            {
                "start_frame": 0,
                "end_frame": length,
                "action_text": action_text,
            }
        ]

    episodes_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"Wrote {len(rows)} episodes to {episodes_path}")
    print(f"Task prompt: {task}")
    print("front_camera_key = \"observation.images.front\"")
    print("wrist_camera_key = \"observation.images.wrist\"")


if __name__ == "__main__":
    main()
