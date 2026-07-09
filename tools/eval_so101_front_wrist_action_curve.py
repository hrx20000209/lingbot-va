#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import av
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from PIL import Image

from wan_va.configs import VA_CONFIGS
from wan_va.wan_va_server import VA_Server


ACTION_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def load_episodes(dataset_path: Path) -> list[dict]:
    path = dataset_path / "meta" / "episodes.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run prepare_so101_front_wrist_action_config.py first.")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def partition_video_files(video_paths: list[Path], lengths: dict[int, int]) -> list[tuple[Path, list[int]]]:
    episode_ids = sorted(lengths)
    cursor = 0
    result = []
    for path in video_paths:
        with av.open(str(path)) as container:
            frame_count = int(container.streams.video[0].frames)
        ids = []
        accumulated = 0
        while cursor < len(episode_ids) and accumulated < frame_count:
            episode_id = episode_ids[cursor]
            ids.append(episode_id)
            accumulated += lengths[episode_id]
            cursor += 1
        result.append((path, ids))
    return result


def decode_episode_camera(dataset_path: Path, camera_key: str, episode_id: int, size: int) -> list[np.ndarray]:
    episodes = load_episodes(dataset_path)
    lengths = {int(row["episode_index"]): int(row["length"]) for row in episodes}
    paths = sorted((dataset_path / "videos" / camera_key / "chunk-000").glob("*.mp4"))
    for video_path, episode_ids in partition_video_files(paths, lengths):
        if episode_id not in episode_ids:
            continue
        offset = sum(lengths[index] for index in episode_ids[: episode_ids.index(episode_id)])
        end = offset + lengths[episode_id]
        frames = []
        with av.open(str(video_path)) as container:
            for frame_index, frame in enumerate(container.decode(video=0)):
                if frame_index < offset:
                    continue
                if frame_index >= end:
                    break
                frames.append(frame.reformat(width=size, height=size, format="rgb24").to_ndarray())
        return frames
    raise ValueError(f"Episode {episode_id} not found in {camera_key}")


def load_episode_actions(dataset_path: Path, episode_id: int) -> np.ndarray:
    paths = sorted((dataset_path / "data").glob("chunk-*/*.parquet"))
    rows = []
    for path in paths:
        frame = pd.read_parquet(path, columns=["episode_index", "frame_index", "action"])
        rows.append(frame[frame["episode_index"] == episode_id])
    episode = pd.concat(rows, ignore_index=True).sort_values("frame_index")
    if episode.empty:
        raise ValueError(f"Episode {episode_id} has no action rows")
    return np.stack(episode["action"].to_numpy()).astype(np.float32)


def plot_curves(pred: np.ndarray, gt: np.ndarray, output_path: Path) -> None:
    count = min(len(pred), len(gt))
    time = np.arange(count)
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for index, axis in enumerate(axes.flat):
        axis.plot(time, pred[:count, index], label="pred")
        axis.plot(time, gt[:count, index], label="ground truth", linestyle="--")
        axis.set_title(ACTION_NAMES[index])
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SO101 LingBot predicted action chunk to GT action.")
    parser.add_argument("--config-name", default="so101_front_wrist")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset_path", type=Path, default=Path("/data/rxhuang/three_cubes_1"))
    parser.add_argument("--episode_id", type=int, default=0)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--action_guidance_scale", type=float, default=1.0)
    parser.add_argument("--attn_window", type=int, default=1)
    args = parser.parse_args()

    cfg = copy.deepcopy(VA_CONFIGS[args.config_name])
    checkpoint = args.checkpoint.expanduser().resolve()
    cfg.transformer_path = str(checkpoint / "transformer" if (checkpoint / "transformer").exists() else checkpoint)
    cfg.guidance_scale = args.guidance_scale
    cfg.action_guidance_scale = args.action_guidance_scale
    cfg.attn_window = args.attn_window
    cfg.save_root = str(args.output_dir / "server_debug")
    cfg.rank = cfg.local_rank = 0
    cfg.world_size = 1

    episodes = load_episodes(args.dataset_path)
    task = next((row.get("tasks", [""])[0] for row in episodes if int(row["episode_index"]) == args.episode_id), "")
    if not task:
        task = "Pick the green cube and place it inside the blue box."
    cameras = {
        key: decode_episode_camera(args.dataset_path, key, args.episode_id, cfg.width)
        for key in cfg.obs_cam_keys
    }
    gt = load_episode_actions(args.dataset_path, args.episode_id)
    obs = {key: frames[args.start_frame] for key, frames in cameras.items()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(obs[cfg.obs_cam_keys[0]]).save(args.output_dir / "front_input.png")
    Image.fromarray(obs[cfg.obs_cam_keys[1]]).save(args.output_dir / "wrist_input.png")

    server = VA_Server(cfg)
    server.infer({"reset": True, "prompt": task})
    ret = server.infer({"obs": [obs], "prompt": [task], "reset": False, "compute_kv_cache": False})
    raw_action = ret["action"]
    pred = rearrange(raw_action, "c t f -> (t f) c")
    gt_window = gt[args.start_frame : args.start_frame + len(pred)]
    count = min(len(pred), len(gt_window))
    pred = pred[:count]
    gt_window = gt_window[:count]
    error = pred - gt_window
    abs_error = np.abs(error)
    metrics = {
        "episode_id": args.episode_id,
        "start_frame": args.start_frame,
        "raw_action_shape": list(raw_action.shape),
        "pred_action_shape": list(pred.shape),
        "ground_truth_shape": list(gt_window.shape),
        "overall_mae": float(abs_error.mean()),
        "per_dim_mae": {name: float(abs_error[:, i].mean()) for i, name in enumerate(ACTION_NAMES)},
        "first_8_step_mae": float(abs_error[: min(8, count)].mean()),
        "first_16_step_mae": float(abs_error[: min(16, count)].mean()),
    }

    csv = pd.DataFrame(
        {
            **{f"pred_{name}": pred[:, i] for i, name in enumerate(ACTION_NAMES)},
            **{f"gt_{name}": gt_window[:, i] for i, name in enumerate(ACTION_NAMES)},
        }
    )
    csv.to_csv(args.output_dir / f"action_curve_ep{args.episode_id}.csv", index=False)
    plot_curves(pred, gt_window, args.output_dir / f"action_curve_ep{args.episode_id}.png")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
