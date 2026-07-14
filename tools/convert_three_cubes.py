#!/usr/bin/env python
"""Convert /data/rxhuang/three_cubes_1 (LeRobot v2.1) into LingBot-VA post-training
format with three cameras (front/right/wrist), writing to an independent output
root so the existing /data/rxhuang/three_cubes_1 (front_wrist, in-place mutated)
and /data/rxhuang/three_cubes_1_lingbot_v21 (front_wrist, 2-camera) datasets are
left untouched.

This is a thin CLI wrapper around wan_va.dataset.prepare_three_cubes, which
already implements and has validated (issue #29 reference cosine similarity
0.99993) the metadata/embeddings/latents pipeline. No VAE or action-mapping
logic is reimplemented here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from wan_va.dataset.prepare_three_cubes import encode_text, extract_latents, prepare_metadata


DEFAULT_SOURCE_ROOT = Path("/data/rxhuang/three_cubes_1")
DEFAULT_OUTPUT_ROOT = Path("/data/rxhuang/three_cubes_1_lingbot")
DEFAULT_MODEL_ROOT = Path("/home/rxhuang/Projects/models/lingbot-va-base")
DEFAULT_CAMERA_KEYS = [
    "observation.images.front",
    "observation.images.right",
    "observation.images.wrist",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert three_cubes_1 into LingBot-VA format with 3 cameras "
            "(front/right/wrist), writing to an independent output root."
        )
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--stage", choices=["all", "metadata", "embeddings", "latents"], default="all")
    parser.add_argument("--camera-keys", nargs="+", default=DEFAULT_CAMERA_KEYS)
    parser.add_argument("--target-fps", type=int, default=15)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episode-from", type=int, default=0)
    parser.add_argument("--episode-to", type=int, default=None, help="Exclusive episode index.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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
