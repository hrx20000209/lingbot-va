#!/usr/bin/env python

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from tqdm import tqdm


DEFAULT_SOURCE_ROOT = Path("/data/rxhuang/three_cubes_1")
DEFAULT_OUTPUT_ROOT = Path("/data/rxhuang/three_cubes_1_lingbot_v21")
DEFAULT_MODEL_ROOT = Path("/home/rxhuang/Projects/models/lingbot-va-base")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Three_Cubes_1 for LingBot-VA post-training.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--stage", choices=["all", "metadata", "embeddings", "latents"], default="all")
    parser.add_argument(
        "--camera-keys",
        nargs="+",
        default=["observation.images.front", "observation.images.wrist"],
    )
    parser.add_argument("--target-fps", type=int, default=15)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episode-from", type=int, default=0)
    parser.add_argument("--episode-to", type=int, default=None, help="Exclusive episode index.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_source_frames(source_root: Path) -> pd.DataFrame:
    paths = sorted((source_root / "data").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {source_root / 'data'}")
    frame = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    return frame.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)


def load_task(source_root: Path) -> str:
    task_frame = pd.read_parquet(source_root / "meta" / "tasks.parquet")
    if "task" in task_frame.columns:
        return str(task_frame.sort_values("task_index").iloc[0]["task"])
    if task_frame.index.name == "task":
        return str(task_frame.sort_values("task_index").index[0])
    return str(task_frame.index[0])


def feature_stats(values: np.ndarray) -> dict:
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [len(values)],
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def prepare_metadata(source_root: Path, output_root: Path, camera_keys: list[str], height: int, width: int) -> None:
    frame = load_source_frames(source_root)
    task = load_task(source_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "meta").mkdir(exist_ok=True)
    (output_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    episodes = []
    stats_rows = []
    for episode_index, episode in tqdm(frame.groupby("episode_index", sort=True), desc="Writing episodes"):
        episode_index = int(episode_index)
        episode = episode.sort_values("frame_index").reset_index(drop=True)
        episode_path = output_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        episode.to_parquet(episode_path, index=False)
        length = len(episode)
        episodes.append(
            {
                "episode_index": episode_index,
                "tasks": [task],
                "length": length,
                "action_config": [
                    {
                        "start_frame": 0,
                        "end_frame": length,
                        "action_text": task,
                        "skill": "",
                    }
                ],
            }
        )
        stats_rows.append(
            {
                "episode_index": episode_index,
                "stats": {
                    "action": feature_stats(np.stack(episode["action"].to_numpy())),
                    "observation.state": feature_stats(np.stack(episode["observation.state"].to_numpy())),
                },
            }
        )

    info_source = json.loads((source_root / "meta" / "info.json").read_text())
    action_feature = info_source["features"]["action"]
    state_feature = info_source["features"]["observation.state"]
    features = {
        key: {
            "dtype": "video",
            "shape": [3, height, width],
            "names": ["rgb", "height", "width"],
            "info": {
                "video.height": height,
                "video.width": width,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": int(info_source["fps"]),
                "video.channels": 3,
                "has_audio": False,
            },
        }
        for key in camera_keys
    }
    features.update(
        {
            "action": action_feature,
            "observation.state": state_feature,
            "timestamp": info_source["features"]["timestamp"],
            "frame_index": info_source["features"]["frame_index"],
            "episode_index": info_source["features"]["episode_index"],
            "index": info_source["features"]["index"],
            "task_index": info_source["features"]["task_index"],
        }
    )
    info = {
        "codebase_version": "v2.1",
        "robot_type": info_source.get("robot_type", "so101_follower"),
        "total_episodes": len(episodes),
        "total_frames": len(frame),
        "total_tasks": 1,
        "total_videos": len(episodes) * len(camera_keys),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": int(info_source["fps"]),
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    write_json(output_root / "meta" / "info.json", info)
    (output_root / "meta" / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episodes)
    )
    (output_root / "meta" / "episodes_stats.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in stats_rows)
    )
    (output_root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": task}) + "\n"
    )
    write_json(
        output_root / "conversion.json",
        {
            "source_root": str(source_root.resolve()),
            "camera_keys": camera_keys,
            "source_fps": int(info_source["fps"]),
            "height": height,
            "width": width,
            "so101_action_channels": [0, 1, 2, 3, 4, 28],
            "task": task,
        },
    )


@torch.inference_mode()
def encode_text(model_root: Path, output_root: Path, device: str) -> None:
    from wan_va.modules.utils import load_text_encoder, load_tokenizer

    task = json.loads((output_root / "meta" / "tasks.jsonl").read_text().splitlines()[0])["task"]
    tokenizer = load_tokenizer(model_root / "tokenizer")
    encoder = load_text_encoder(model_root / "text_encoder", torch.bfloat16, device)

    def encode(prompt: str) -> torch.Tensor:
        prompt = prompt_clean(prompt)
        inputs = tokenizer(
            [prompt],
            padding="max_length",
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids = inputs.input_ids.to(device)
        mask = inputs.attention_mask.to(device)
        hidden = encoder(ids, mask).last_hidden_state
        length = int(mask[0].sum())
        result = torch.zeros(512, hidden.shape[-1], dtype=torch.bfloat16, device=device)
        result[:length] = hidden[0, :length].to(torch.bfloat16)
        return result.cpu()

    torch.save(encode(task), output_root / "task_emb.pt")
    torch.save(encode(""), output_root / "empty_emb.pt")


def episode_lengths(output_root: Path) -> dict[int, int]:
    return {
        int(row["episode_index"]): int(row["length"])
        for row in (
            json.loads(line)
            for line in (output_root / "meta" / "episodes.jsonl").read_text().splitlines()
            if line.strip()
        )
    }


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
        if accumulated != frame_count:
            raise ValueError(f"Video {path} has {frame_count} frames but episode lengths sum to {accumulated}")
        result.append((path, ids))
    if cursor != len(episode_ids):
        raise ValueError(f"Only mapped {cursor}/{len(episode_ids)} episodes to video files")
    return result


@torch.inference_mode()
def encode_episode_video(
    frames: list[np.ndarray],
    vae,
    device: str,
) -> torch.Tensor:
    video = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).unsqueeze(0)
    video = video.to(device=device, dtype=torch.bfloat16).div_(127.5).sub_(1.0)
    mu = vae.encode(video).latent_dist.mode()
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    inv_std = (1.0 / torch.tensor(vae.config.latents_std, device=device)).view(
        1, -1, 1, 1, 1
    )
    return ((mu.float() - mean) * inv_std).to(torch.bfloat16).cpu()


@torch.inference_mode()
def extract_latents(args: argparse.Namespace) -> None:
    from wan_va.modules.utils import load_vae

    output_root = args.output_root.resolve()
    lengths = episode_lengths(output_root)
    source_fps = int(json.loads((args.source_root / "meta" / "info.json").read_text())["fps"])
    if source_fps % args.target_fps:
        raise ValueError(f"source fps {source_fps} must be divisible by target fps {args.target_fps}")
    stride = source_fps // args.target_fps
    task = json.loads((output_root / "meta" / "tasks.jsonl").read_text().splitlines()[0])["task"]
    text_emb = torch.load(output_root / "task_emb.pt", map_location="cpu", weights_only=False)
    vae = load_vae(args.model_root / "vae", torch.bfloat16, args.device)

    selected_episode_ids = {
        episode_id
        for episode_id in lengths
        if args.episode_from <= episode_id
        and (args.episode_to is None or episode_id < args.episode_to)
    }
    for camera_key in args.camera_keys:
        video_paths = sorted((args.source_root / "videos" / camera_key / "chunk-000").glob("*.mp4"))
        if not video_paths:
            raise FileNotFoundError(f"No source videos found for {camera_key}")
        mappings = partition_video_files(video_paths, lengths)
        output_dir = output_root / "latents" / "chunk-000" / camera_key
        output_dir.mkdir(parents=True, exist_ok=True)
        for video_path, episode_ids in mappings:
            with av.open(str(video_path)) as container:
                stream = container.streams.video[0]
                frame_iter = iter(container.decode(stream))
                for episode_id in episode_ids:
                    length = lengths[episode_id]
                    sampled = []
                    sampled_ids = []
                    for frame_index in range(length):
                        try:
                            frame = next(frame_iter)
                        except StopIteration as exc:
                            raise RuntimeError(
                                f"Video {video_path} ended while decoding episode {episode_id} "
                                f"at frame {frame_index}/{length}"
                            ) from exc
                        if frame_index % stride == 0:
                            sampled.append(
                                frame.reformat(width=args.width, height=args.height, format="rgb24").to_ndarray()
                            )
                            sampled_ids.append(frame_index)
                    if episode_id not in selected_episode_ids:
                        continue
                    valid_count = ((len(sampled) - 1) // 4) * 4 + 1
                    sampled = sampled[:valid_count]
                    sampled_ids = np.asarray(sampled_ids[:valid_count], dtype=np.int64)
                    output_path = output_dir / f"episode_{episode_id:06d}_0_{length}.pth"
                    if output_path.exists() and not args.overwrite:
                        continue
                    latent = encode_episode_video(sampled, vae, args.device)
                    _, channels, latent_frames, latent_height, latent_width = latent.shape
                    expected_latent_frames = (len(sampled) - 1) // 4 + 1
                    if latent_frames != expected_latent_frames:
                        raise RuntimeError(
                            f"VAE produced {latent_frames} latent frames for {len(sampled)} video frames; "
                            f"expected {expected_latent_frames}"
                        )
                    flattened = latent[0].permute(1, 2, 3, 0).reshape(-1, channels).contiguous()
                    torch.save(
                        {
                            "latent": flattened,
                            "latent_num_frames": latent_frames,
                            "latent_height": latent_height,
                            "latent_width": latent_width,
                            "video_num_frames": len(sampled),
                            "video_height": args.height,
                            "video_width": args.width,
                            "text_emb": text_emb,
                            "text": task,
                            "frame_ids": sampled_ids,
                            "start_frame": 0,
                            "end_frame": length,
                            "fps": args.target_fps,
                            "ori_fps": source_fps,
                        },
                        output_path,
                    )
                    del latent, flattened, sampled
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(f"saved {output_path}")


def main() -> None:
    args = parse_args()
    args.source_root = args.source_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.model_root = args.model_root.expanduser().resolve()
    if args.stage in {"all", "metadata"}:
        prepare_metadata(args.source_root, args.output_root, args.camera_keys, args.height, args.width)
    if args.stage in {"all", "embeddings"}:
        encode_text(args.model_root, args.output_root, args.device)
    if args.stage in {"all", "latents"}:
        extract_latents(args)


if __name__ == "__main__":
    main()
