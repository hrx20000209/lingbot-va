#!/usr/bin/env python

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import av
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from wan_va.configs import VA_CONFIGS
from wan_va.dataset.prepare_three_cubes import episode_lengths, partition_video_files
from wan_va.wan_va_server import VA_Server


def decode_episode_camera(source_root, output_root, camera_key, episode_index, size=256):
    lengths = episode_lengths(output_root)
    paths = sorted((source_root / "videos" / camera_key / "chunk-000").glob("*.mp4"))
    for video_path, episode_ids in partition_video_files(paths, lengths):
        if episode_index not in episode_ids:
            continue
        start = sum(lengths[index] for index in episode_ids[: episode_ids.index(episode_index)])
        end = start + lengths[episode_index]
        images = []
        with av.open(str(video_path)) as container:
            for frame_index, frame in enumerate(container.decode(video=0)):
                if frame_index < start:
                    continue
                if frame_index >= end:
                    break
                images.append(frame.reformat(width=size, height=size, format="rgb24").to_ndarray())
        if len(images) != lengths[episode_index]:
            raise RuntimeError(
                f"Decoded {len(images)} frames for episode {episode_index}, expected {lengths[episode_index]}"
            )
        return images
    raise ValueError(f"Episode {episode_index} was not found in {camera_key}")


def flatten_action_chunk(action, drop_condition_block=False):
    if action.ndim != 3:
        raise ValueError(f"Expected action [C,F,N], got {action.shape}")
    if drop_condition_block:
        action = action[:, 1:, :]
    return action.transpose(1, 2, 0).reshape(-1, action.shape[0])


def plot_episode(prediction, ground_truth, state, boundaries, names, fps, output):
    count = min(len(prediction), len(ground_truth), len(state))
    prediction = prediction[:count]
    ground_truth = ground_truth[:count]
    state = state[:count]
    time = np.arange(count) / fps
    colors = ["tab:orange", "tab:blue", "tab:green"]
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True)
    for axis, indices in zip(axes, ([0, 1, 2], [3, 4, 5])):
        for color, index in zip(colors, indices):
            axis.plot(time, prediction[:, index], color=color, label=f"{names[index]} pred")
            axis.plot(
                time,
                ground_truth[:, index],
                color=color,
                linestyle="--",
                label=f"{names[index]} GT action",
            )
            axis.plot(
                time,
                state[:, index],
                color=color,
                linestyle=":",
                alpha=0.65,
                label=f"{names[index]} observation.state",
            )
        for boundary in boundaries:
            if boundary < count:
                axis.axvline(boundary / fps, color="red", alpha=0.18, linewidth=0.8)
        axis.set_title(", ".join(names[index] for index in indices))
        axis.set_xlabel("time (s)")
        axis.set_ylabel("joint position")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    fig.suptitle(
        "LingBot-VA dataset replay: prediction vs GT action vs observation.state\n"
        "red lines = replanning boundaries"
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)


def main():
    parser = argparse.ArgumentParser(description="Replay a Three Cubes episode through LingBot-VA.")
    parser.add_argument("--transformer-path", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=90)
    parser.add_argument("--source-root", type=Path, default=Path("/data/rxhuang/three_cubes_1"))
    parser.add_argument(
        "--prepared-root", type=Path, default=Path("/data/rxhuang/three_cubes_1_lingbot_v21")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/home/rxhuang/Projects/models/lingbot-va-three-cubes/eval")
    )
    parser.add_argument("--max-chunks", type=int, default=None)
    args = parser.parse_args()

    config = copy.deepcopy(VA_CONFIGS["three_cubes"])
    config.transformer_path = str(args.transformer_path.resolve())
    config.save_root = str(args.output_dir / "server_debug")
    config.rank = config.local_rank = 0
    config.world_size = 1

    info = json.loads((args.source_root / "meta" / "info.json").read_text())
    names = info["features"]["action"]["names"]
    episode_path = args.prepared_root / "data/chunk-000" / f"episode_{args.episode:06d}.parquet"
    episode = pd.read_parquet(episode_path)
    ground_truth = np.stack(episode["action"].to_numpy()).astype(np.float32)
    state = np.stack(episode["observation.state"].to_numpy()).astype(np.float32)
    cameras = {
        key: decode_episode_camera(
            args.source_root, args.prepared_root, key, args.episode, size=config.width
        )
        for key in config.obs_cam_keys
    }
    task = json.loads((args.prepared_root / "meta/tasks.jsonl").read_text().splitlines()[0])["task"]

    server = VA_Server(config)
    server.infer({"reset": True, "prompt": task})
    initial_observation = {key: images[0] for key, images in cameras.items()}
    predictions = []
    boundaries = []
    action_cursor = 0
    camera_cursor = 2
    first = True
    chunk_index = 0
    while action_cursor < len(ground_truth):
        if args.max_chunks is not None and chunk_index >= args.max_chunks:
            break
        result = server.infer({"obs": [initial_observation]})
        full_chunk = result["pred_action"]
        executable = flatten_action_chunk(full_chunk, drop_condition_block=first)
        executable = executable[: len(ground_truth) - action_cursor]
        boundaries.append(action_cursor)
        predictions.append(executable)
        action_cursor += len(executable)
        chunk_index += 1
        if action_cursor >= len(ground_truth):
            break

        camera_count = (config.frame_chunk_size - int(first)) * 4
        indices = [min(camera_cursor + 2 * offset, len(ground_truth) - 1) for offset in range(camera_count)]
        key_frames = [{key: images[index] for key, images in cameras.items()} for index in indices]
        camera_cursor += 2 * camera_count
        server.infer(
            {
                "obs": key_frames,
                "compute_kv_cache": True,
                "pred_action": full_chunk,
            }
        )
        first = False

    prediction = np.concatenate(predictions, axis=0)
    count = min(len(prediction), len(ground_truth))
    error = prediction[:count] - ground_truth[:count]
    metrics = {
        "episode": args.episode,
        "frames": count,
        "chunks": len(boundaries),
        "mae": float(np.abs(error).mean()),
        "joint_mae": {name: float(np.abs(error[:, i]).mean()) for i, name in enumerate(names)},
        "prediction_step_delta": float(np.abs(np.diff(prediction[:count], axis=0)).mean()),
        "ground_truth_step_delta": float(np.abs(np.diff(ground_truth[:count], axis=0)).mean()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"episode_{args.episode:06d}"
    (args.output_dir / f"{stem}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(
        args.output_dir / f"{stem}_curves.npz",
        prediction=prediction,
        ground_truth=ground_truth,
        state=state,
        boundaries=np.asarray(boundaries),
    )
    plot_episode(
        prediction,
        ground_truth,
        state,
        boundaries,
        names,
        int(info["fps"]),
        args.output_dir / f"{stem}_action_comparison.png",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
