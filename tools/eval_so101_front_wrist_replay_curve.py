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

from wan_va.configs import VA_CONFIGS
from wan_va.wan_va_server import VA_Server


ACTION_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def load_episodes(dataset_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (dataset_path / "meta" / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]


def partition_video_files(video_paths: list[Path], lengths: dict[int, int]) -> list[tuple[Path, list[int]]]:
    episode_ids = sorted(lengths)
    cursor = 0
    result = []
    for path in video_paths:
        with av.open(str(path)) as container:
            frame_count = int(container.streams.video[0].frames)
        ids = []
        total = 0
        while cursor < len(episode_ids) and total < frame_count:
            episode_id = episode_ids[cursor]
            ids.append(episode_id)
            total += lengths[episode_id]
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


def load_episode_arrays(dataset_path: Path, episode_id: int) -> tuple[np.ndarray, np.ndarray]:
    frames = []
    for path in sorted((dataset_path / "data").glob("chunk-*/*.parquet")):
        frame = pd.read_parquet(path, columns=["episode_index", "frame_index", "action", "observation.state"])
        frames.append(frame[frame["episode_index"] == episode_id])
    episode = pd.concat(frames, ignore_index=True).sort_values("frame_index")
    if episode.empty:
        raise ValueError(f"Episode {episode_id} not found in parquet data")
    actions = np.stack(episode["action"].to_numpy()).astype(np.float32)
    states = np.stack(episode["observation.state"].to_numpy()).astype(np.float32)
    return actions, states


def make_obs(cameras: dict[str, list[np.ndarray]], camera_keys: list[str], frame_index: int) -> dict:
    return {key: cameras[key][min(frame_index, len(cameras[key]) - 1)] for key in camera_keys}


def plot_replay(pred: np.ndarray, gt: np.ndarray, state: np.ndarray, boundaries: list[int], output: Path) -> None:
    count = min(len(pred), len(gt), len(state))
    time = np.arange(count) / 30.0
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True)
    groups = ([0, 1, 2], [3, 4, 5])
    colors = ["tab:orange", "tab:blue", "tab:green"]
    for axis, indices in zip(axes, groups):
        for color, idx in zip(colors, indices):
            axis.plot(time, pred[:count, idx], color=color, label=f"{ACTION_NAMES[idx]} pred")
            axis.plot(time, gt[:count, idx], color=color, linestyle="--", label=f"{ACTION_NAMES[idx]} GT action")
            axis.plot(time, state[:count, idx], color=color, linestyle=":", alpha=0.65, label=f"{ACTION_NAMES[idx]} observation.state")
        for boundary in boundaries:
            if boundary < count:
                axis.axvline(boundary / 30.0, color="red", alpha=0.16, linewidth=0.8)
        axis.set_title(", ".join(ACTION_NAMES[idx] for idx in indices))
        axis.set_xlabel("time (s)")
        axis.set_ylabel("joint position")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    fig.suptitle("SO101 dataset replay: prediction vs GT action vs observation.state\nred lines = replanning boundaries")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay SO101 episode through LingBot-VA and plot action curves.")
    parser.add_argument("--config-name", default="so101_front_wrist")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset_path", type=Path, default=Path("/data/rxhuang/three_cubes_1"))
    parser.add_argument("--episode_id", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_chunks", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--action_guidance_scale", type=float, default=1.0)
    parser.add_argument("--attn_window", type=int, default=30)
    parser.add_argument("--reset_each_chunk", action="store_true")
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
    task = next(row["tasks"][0] for row in episodes if int(row["episode_index"]) == args.episode_id)
    cameras = {
        key: decode_episode_camera(args.dataset_path, key, args.episode_id, cfg.width)
        for key in cfg.obs_cam_keys
    }
    gt, state = load_episode_arrays(args.dataset_path, args.episode_id)

    server = VA_Server(cfg)
    server.infer({"reset": True, "prompt": task})

    predictions = []
    boundaries = []
    action_cursor = 0
    first = True
    chunk_idx = 0
    current_obs = {"obs": [make_obs(cameras, cfg.obs_cam_keys, 0)], "prompt": [task]}
    while action_cursor < len(gt) and chunk_idx < args.max_chunks:
        if args.reset_each_chunk:
            server.infer({"reset": True, "prompt": task})
        ret = server.infer(current_obs)
        raw_action = ret["action"]
        flat = rearrange(raw_action, "c t f -> (t f) c").astype(np.float32)
        executable = flat[raw_action.shape[2]:] if (first or args.reset_each_chunk) else flat
        executable = executable[: len(gt) - action_cursor]
        if len(executable) == 0:
            break
        boundaries.append(action_cursor)
        predictions.append(executable)

        obs_history = []
        for offset in range(len(executable)):
            frame_idx = min(action_cursor + offset, len(gt) - 1)
            obs_history.append(
                {
                    "front": cameras[cfg.obs_cam_keys[0]][frame_idx],
                    "wrist": cameras[cfg.obs_cam_keys[1]][frame_idx],
                }
            )
        if not args.reset_each_chunk:
            kv_obs = {
                "obs": [
                    {
                        cfg.obs_cam_keys[0]: ob["front"],
                        cfg.obs_cam_keys[1]: ob["wrist"],
                    }
                    for ob in obs_history[::-2][::-1]
                ],
                "prompt": [task],
                "compute_kv_cache": True,
                "imagine": False,
                "state": raw_action,
            }
            server.infer(kv_obs)
        action_cursor += len(executable)
        current_obs = {"obs": [make_obs(cameras, cfg.obs_cam_keys, action_cursor)], "prompt": [task]}
        first = False
        chunk_idx += 1

    pred = np.concatenate(predictions, axis=0)
    count = min(len(pred), len(gt))
    error = pred[:count] - gt[:count]
    metrics = {
        "episode_id": args.episode_id,
        "chunks": chunk_idx,
        "frames": count,
        "overall_mae": float(np.abs(error).mean()),
        "per_dim_mae": {name: float(np.abs(error[:, i]).mean()) for i, name in enumerate(ACTION_NAMES)},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_replay(pred, gt, state, boundaries, args.output_dir / f"replay_action_curve_ep{args.episode_id}.png")
    pd.DataFrame(
        {
            **{f"pred_{name}": pred[:count, i] for i, name in enumerate(ACTION_NAMES)},
            **{f"gt_{name}": gt[:count, i] for i, name in enumerate(ACTION_NAMES)},
            **{f"state_{name}": state[:count, i] for i, name in enumerate(ACTION_NAMES)},
        }
    ).to_csv(args.output_dir / f"replay_action_curve_ep{args.episode_id}.csv", index=False)
    (args.output_dir / "replay_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
