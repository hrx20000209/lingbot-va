#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import av
import numpy as np
import torch
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from PIL import Image

from wan_va.configs import VA_CONFIGS


CAMERA_KEYS = ["observation.images.front", "observation.images.wrist"]


def load_episodes(dataset_path: Path) -> list[dict]:
    path = dataset_path / "meta" / "episodes.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run tools/prepare_so101_front_wrist_action_config.py first."
        )
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def partition_video_files(video_paths: list[Path], lengths: dict[int, int]) -> list[tuple[Path, list[int]]]:
    episode_ids = sorted(lengths)
    cursor = 0
    result = []
    for path in video_paths:
        with av.open(str(path)) as container:
            frame_count = int(container.streams.video[0].frames)
            if frame_count <= 0:
                frame_count = sum(1 for _ in container.decode(video=0))
        ids = []
        accumulated = 0
        while cursor < len(episode_ids) and accumulated < frame_count:
            episode_id = episode_ids[cursor]
            ids.append(episode_id)
            accumulated += lengths[episode_id]
            cursor += 1
        if accumulated != frame_count:
            raise ValueError(f"Video {path} has {frame_count} frames but mapped episode lengths sum to {accumulated}")
        result.append((path, ids))
    if cursor != len(episode_ids):
        raise ValueError(f"Only mapped {cursor}/{len(episode_ids)} episodes to video files")
    return result


@torch.inference_mode()
def encode_text(model_root: Path, prompt: str, device: str) -> torch.Tensor:
    from wan_va.modules.utils import load_text_encoder, load_tokenizer

    tokenizer = load_tokenizer(model_root / "tokenizer")
    encoder = load_text_encoder(model_root / "text_encoder", torch.bfloat16, device)
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
    hidden = encoder(
        inputs.input_ids.to(device),
        inputs.attention_mask.to(device),
    ).last_hidden_state
    length = int(inputs.attention_mask[0].sum())
    result = torch.zeros(512, hidden.shape[-1], dtype=torch.bfloat16, device=device)
    result[:length] = hidden[0, :length].to(torch.bfloat16)
    del encoder
    torch.cuda.empty_cache()
    return result.cpu()


@torch.inference_mode()
def encode_video(frames: list[np.ndarray], vae, device: str) -> torch.Tensor:
    video = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).unsqueeze(0)
    video = video.to(device=device, dtype=torch.bfloat16).div_(127.5).sub_(1.0)
    mu = vae.encode(video).latent_dist.mode()
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    inv_std = (1.0 / torch.tensor(vae.config.latents_std, device=device)).view(1, -1, 1, 1, 1)
    return ((mu.float() - mean) * inv_std).to(torch.bfloat16).cpu()


def save_debug_frame(frames: list[np.ndarray], camera_key: str, episode_id: int) -> None:
    if not frames:
        return
    out_dir = Path("/tmp/lingbot_so101_debug/latent_extraction")
    out_dir.mkdir(parents=True, exist_ok=True)
    name = camera_key.rsplit(".", 1)[-1]
    Image.fromarray(frames[0]).save(out_dir / f"episode_{episode_id:06d}_{name}_source.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Wan2.2 VAE latents for SO101 front+wrist videos.")
    parser.add_argument("--dataset_path", type=Path, default=Path("/data/rxhuang/three_cubes_1"))
    parser.add_argument("--config-name", default="so101_front_wrist_train")
    parser.add_argument("--target_fps", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()

    dataset_path = args.dataset_path.expanduser().resolve()
    cfg = VA_CONFIGS[args.config_name]
    camera_keys = list(cfg.obs_cam_keys)
    if camera_keys != CAMERA_KEYS:
        raise ValueError(f"Expected only {CAMERA_KEYS}, got {camera_keys}")

    info = json.loads((dataset_path / "meta" / "info.json").read_text())
    image_keys = [key for key, feature in info["features"].items() if feature.get("dtype") == "video"]
    print(f"All image keys: {image_keys}")
    print('front_camera_key = "observation.images.front"')
    print('wrist_camera_key = "observation.images.wrist"')
    for key in camera_keys:
        if key not in image_keys:
            raise KeyError(f"Configured camera key {key!r} not found in dataset image keys {image_keys}")

    all_episodes = load_episodes(dataset_path)
    episodes = all_episodes
    if args.max_episodes is not None:
        allowed = {int(row["episode_index"]) for row in episodes[: args.max_episodes]}
        episodes = [row for row in episodes if int(row["episode_index"]) in allowed]
    else:
        allowed = {int(row["episode_index"]) for row in episodes}
    all_lengths = {int(row["episode_index"]): int(row["length"]) for row in all_episodes}
    selected_lengths = {int(row["episode_index"]): int(row["length"]) for row in episodes}
    action_texts = {
        (int(row["episode_index"]), int(seg["start_frame"]), int(seg["end_frame"])): seg["action_text"]
        for row in episodes
        for seg in row.get("action_config", [])
    }

    source_fps = int(info["fps"])
    if source_fps % args.target_fps:
        raise ValueError(f"source fps {source_fps} must be divisible by target fps {args.target_fps}")
    stride = source_fps // args.target_fps

    from wan_va.modules.utils import load_vae

    model_root = Path(cfg.wan22_pretrained_model_name_or_path).expanduser().resolve()
    empty_emb = encode_text(model_root, "", args.device)
    torch.save(empty_emb, dataset_path / "empty_emb.pt")
    text_cache: dict[str, torch.Tensor] = {"": empty_emb}
    vae = load_vae(model_root / "vae", torch.bfloat16, args.device)

    for camera_key in camera_keys:
        video_paths = sorted((dataset_path / "videos" / camera_key / "chunk-000").glob("*.mp4"))
        if not video_paths:
            raise FileNotFoundError(f"No source videos found for {camera_key}")
        mappings = partition_video_files(video_paths, all_lengths)
        output_dir = dataset_path / "latents" / "chunk-000" / camera_key
        output_dir.mkdir(parents=True, exist_ok=True)
        for video_path, episode_ids in mappings:
            if not any(episode_id in selected_lengths for episode_id in episode_ids):
                continue
            with av.open(str(video_path)) as container:
                frame_iter = iter(container.decode(video=0))
                for episode_id in episode_ids:
                    length = all_lengths[episode_id]
                    decoded = []
                    for frame_index in range(length):
                        frame = next(frame_iter)
                        if frame_index % stride == 0:
                            decoded.append(
                                frame.reformat(width=cfg.width, height=cfg.height, format="rgb24").to_ndarray()
                            )
                    if episode_id not in selected_lengths:
                        continue
                    valid_count = ((len(decoded) - 1) // 4) * 4 + 1
                    sampled = decoded[:valid_count]
                    frame_ids = np.arange(0, length, stride, dtype=np.int64)[:valid_count]
                    save_debug_frame(sampled, camera_key, episode_id)
                    for (ep, start_frame, end_frame), text in action_texts.items():
                        if ep != episode_id:
                            continue
                        if start_frame != 0 or end_frame != length:
                            raise NotImplementedError("Initial SO101 extraction supports full-episode segments only.")
                        output_path = output_dir / f"episode_{episode_id:06d}_{start_frame}_{end_frame}.pth"
                        if output_path.exists() and not args.overwrite:
                            continue
                        if text not in text_cache:
                            text_cache[text] = encode_text(model_root, text, args.device)
                        latent = encode_video(sampled, vae, args.device)
                        _, channels, latent_frames, latent_height, latent_width = latent.shape
                        flattened = latent[0].permute(1, 2, 3, 0).reshape(-1, channels).contiguous()
                        torch.save(
                            {
                                "latent": flattened,
                                "latent_num_frames": latent_frames,
                                "latent_height": latent_height,
                                "latent_width": latent_width,
                                "video_num_frames": len(sampled),
                                "video_height": cfg.height,
                                "video_width": cfg.width,
                                "text_emb": text_cache[text],
                                "text": text,
                                "frame_ids": frame_ids,
                                "start_frame": start_frame,
                                "end_frame": end_frame,
                                "fps": args.target_fps,
                                "ori_fps": source_fps,
                            },
                            output_path,
                        )
                        print(f"saved {output_path}")
                        del latent, flattened
                        gc.collect()
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
