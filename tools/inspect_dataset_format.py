#!/usr/bin/env python
"""Self-check tools/convert_three_cubes.py output against two references:

1. Structural diff vs. the official issue #29 example dataset
   (/data/rxhuang/lingbot_issue29_reference) -- confirms our conversion follows
   the same episodes.jsonl / latents / empty_emb.pt conventions.
2. Latent/video alignment spot-check -- decodes VAE latents for a few sampled
   episodes back to pixels (reusing wan_va.modules.utils.load_vae, no VAE
   logic reimplemented) and compares against the original source frames at the
   same frame_ids, plus reports the normalized action range actually produced
   by the training dataset loader's own _action_post_process (not a
   reimplementation -- imports and calls it directly).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch


def load_episodes(root: Path) -> list[dict]:
    path = root / "meta" / "episodes.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def structural_diff(reference_root: Path, output_root: Path) -> None:
    print("=" * 70)
    print("STRUCTURAL DIFF")
    print("=" * 70)

    ref_episodes = load_episodes(reference_root)
    out_episodes = load_episodes(output_root)
    ref_fields = set(ref_episodes[0].keys())
    out_fields = set(out_episodes[0].keys())
    print(f"episodes.jsonl fields: reference={sorted(ref_fields)} output={sorted(out_fields)}")
    if ref_fields != out_fields:
        print(f"  MISMATCH: reference-only={ref_fields - out_fields} output-only={out_fields - ref_fields}")
    else:
        print("  OK: identical field sets")

    ref_action_cfg_fields = set(ref_episodes[0]["action_config"][0].keys())
    out_action_cfg_fields = set(out_episodes[0]["action_config"][0].keys())
    print(f"action_config[0] fields: reference={sorted(ref_action_cfg_fields)} output={sorted(out_action_cfg_fields)}")
    if ref_action_cfg_fields != out_action_cfg_fields:
        print(f"  MISMATCH: reference-only={ref_action_cfg_fields - out_action_cfg_fields} "
              f"output-only={out_action_cfg_fields - ref_action_cfg_fields}")
    else:
        print("  OK: identical action_config field sets")

    ref_cams = sorted(p.name for p in (reference_root / "latents" / "chunk-000").iterdir())
    out_cams = sorted(p.name for p in (output_root / "latents" / "chunk-000").iterdir())
    print(f"latent camera dirs: reference={ref_cams} output={out_cams}")
    print("  (naming differs intentionally: reference uses 'top', ours uses 'front' -- "
          "both are the primary/top-down-role camera slot, just named after the source dataset's key)")

    ref_sample = next((reference_root / "latents" / "chunk-000" / ref_cams[0]).glob("*.pth"))
    out_sample = next((output_root / "latents" / "chunk-000" / out_cams[0]).glob("*.pth"))
    print(f"latent filename convention: reference={ref_sample.name} output={out_sample.name}")

    ref_data = torch.load(ref_sample, map_location="cpu", weights_only=False)
    out_data = torch.load(out_sample, map_location="cpu", weights_only=False)
    ref_keys = set(ref_data.keys())
    out_keys = set(out_data.keys())
    print(f"latent .pth keys: reference={sorted(ref_keys)} output={sorted(out_keys)}")
    if ref_keys != out_keys:
        print(f"  MISMATCH: reference-only={ref_keys - out_keys} output-only={out_keys - ref_keys}")
    else:
        print("  OK: identical latent dict keys")

    print(f"empty_emb.pt present: reference={(reference_root / 'empty_emb.pt').exists()} "
          f"output={(output_root / 'empty_emb.pt').exists()}")


@torch.inference_mode()
def decode_latent_to_frames(latent_path: Path, vae, device: str) -> np.ndarray:
    data = torch.load(latent_path, map_location="cpu", weights_only=False)
    latent = data["latent"].to(device=device, dtype=torch.bfloat16)
    f, h, w = data["latent_num_frames"], data["latent_height"], data["latent_width"]
    latent = latent.reshape(f, h, w, -1).permute(3, 0, 1, 2).unsqueeze(0)
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    latent = latent.float() * std + mean
    pixels = vae.decode(latent.to(torch.bfloat16), return_dict=False)[0]
    pixels = ((pixels.float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    return pixels[0].permute(1, 2, 3, 0).cpu().numpy()  # [F, H, W, 3]


def load_source_frames_at_ids(source_root: Path, camera_key: str, episode_index: int,
                               frame_ids: np.ndarray, height: int, width: int) -> np.ndarray:
    import av
    import pandas as pd

    # Locate which raw video file contains this episode by re-deriving the same
    # partitioning prepare_three_cubes.py uses (sequential episode-length accumulation).
    from wan_va.dataset.prepare_three_cubes import partition_video_files

    frame = pd.concat(
        (pd.read_parquet(p) for p in sorted((source_root / "data").glob("chunk-*/*.parquet"))),
        ignore_index=True,
    )
    lengths = {int(ep): len(g) for ep, g in frame.groupby("episode_index")}
    video_paths = sorted((source_root / "videos" / camera_key / "chunk-000").glob("*.mp4"))
    mappings = partition_video_files(video_paths, lengths)

    for video_path, episode_ids in mappings:
        if episode_index not in episode_ids:
            continue
        offset = sum(lengths[e] for e in episode_ids if e < episode_index)
        target_local_ids = set((offset + frame_ids).tolist())
        collected = {}
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            for local_index, frame_av in enumerate(container.decode(stream)):
                if local_index in target_local_ids:
                    collected[local_index] = frame_av.reformat(
                        width=width, height=height, format="rgb24"
                    ).to_ndarray()
                if len(collected) == len(target_local_ids):
                    break
        return np.stack([collected[offset + fid] for fid in frame_ids])
    raise ValueError(f"episode {episode_index} not found in {camera_key} video mapping")


def alignment_check(source_root: Path, output_root: Path, model_root: Path, camera_key: str,
                     episode_ids: list[int], device: str, out_dir: Path) -> None:
    print("=" * 70)
    print("LATENT / VIDEO ALIGNMENT CHECK")
    print("=" * 70)
    from wan_va.modules.utils import load_vae

    vae = load_vae(model_root / "vae", torch.bfloat16, device)
    out_dir.mkdir(parents=True, exist_ok=True)

    for episode_index in episode_ids:
        episodes = load_episodes(output_root)
        row = next(r for r in episodes if r["episode_index"] == episode_index)
        start, end = row["action_config"][0]["start_frame"], row["action_config"][0]["end_frame"]
        latent_path = output_root / "latents" / "chunk-000" / camera_key / f"episode_{episode_index:06d}_{start}_{end}.pth"
        if not latent_path.exists():
            print(f"episode {episode_index}: SKIP (latent not yet written: {latent_path})")
            continue
        data = torch.load(latent_path, map_location="cpu", weights_only=False)
        frame_ids = data["frame_ids"]

        decoded = decode_latent_to_frames(latent_path, vae, device)
        source = load_source_frames_at_ids(
            source_root, camera_key, episode_index, frame_ids, data["video_height"], data["video_width"]
        )
        n = min(len(decoded), len(source))
        diff = np.abs(decoded[:n].astype(np.float32) - source[:n].astype(np.float32))
        mae = diff.mean()
        psnr = 20 * np.log10(255.0 / (np.sqrt((diff ** 2).mean()) + 1e-8))
        print(f"episode {episode_index} ({camera_key}): decoded_frames={len(decoded)} source_frames={len(source)} "
              f"pixel_MAE={mae:.2f} PSNR={psnr:.1f}dB")

        try:
            import imageio
            mid = n // 2
            side_by_side = np.concatenate([source[mid], decoded[mid]], axis=1)
            out_path = out_dir / f"episode_{episode_index:06d}_{camera_key}_mid_frame_compare.png"
            imageio.imwrite(out_path, side_by_side)
            print(f"  saved {out_path} (left=source, right=VAE-decoded)")
        except ImportError:
            pass


def action_range_check(output_root: Path, config_name: str, episode_ids: list[int]) -> None:
    print("=" * 70)
    print("NORMALIZED ACTION RANGE CHECK (reuses LatentLeRobotDataset._action_post_process)")
    print("=" * 70)
    from wan_va.configs import VA_CONFIGS
    from wan_va.dataset.lerobot_latent_dataset import LatentLeRobotDataset

    config = VA_CONFIGS[config_name]
    dataset = LatentLeRobotDataset(repo_id=str(output_root), config=config, split="all")
    all_actions = []
    for idx, meta in enumerate(dataset.new_metas):
        if meta["episode_index"] not in episode_ids:
            continue
        sample = dataset[idx]
        all_actions.append(sample["actions"])
    if not all_actions:
        print("  no matching episodes found in dataset.new_metas")
        return
    actions = torch.cat([a.flatten(1) for a in all_actions], dim=1)
    used = config.used_action_channel_ids
    print(f"  channels {used} min={actions[used].min().item():.3f} max={actions[used].max().item():.3f} "
          "(clipped to [-1.5, 1.5] by _action_post_process, not a naive [-1, 1] bound)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect/validate the three_cubes_1_lingbot conversion.")
    parser.add_argument("--reference-root", type=Path, default=Path("/data/rxhuang/lingbot_issue29_reference"))
    parser.add_argument("--source-root", type=Path, default=Path("/data/rxhuang/three_cubes_1"))
    parser.add_argument("--output-root", type=Path, default=Path("/data/rxhuang/three_cubes_1_lingbot"))
    parser.add_argument("--model-root", type=Path, default=Path("/home/rxhuang/Projects/models/lingbot-va-base"))
    parser.add_argument("--config-name", default="so101_train")
    parser.add_argument("--camera-key", default="observation.images.front")
    parser.add_argument("--episodes", nargs="+", type=int, default=None, help="Episode ids to spot-check (default: 3 random train episodes)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=Path("/data/rxhuang/three_cubes_1_lingbot/_self_check"))
    parser.add_argument("--skip-structural", action="store_true")
    parser.add_argument("--skip-alignment", action="store_true")
    parser.add_argument("--skip-action-range", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_ids = args.episodes
    if episode_ids is None:
        random.seed(0)
        episode_ids = sorted(random.sample(range(95), 3))
    print(f"spot-checking episodes: {episode_ids}")

    if not args.skip_structural:
        structural_diff(args.reference_root, args.output_root)
    if not args.skip_alignment:
        alignment_check(args.source_root, args.output_root, args.model_root, args.camera_key,
                         episode_ids, args.device, args.out_dir)
    if not args.skip_action_range:
        action_range_check(args.output_root, args.config_name, episode_ids)


if __name__ == "__main__":
    main()
