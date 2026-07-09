#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from wan_va.configs import VA_CONFIGS
from wan_va.dataset import MultiLatentLeRobotDataset


def describe(key: str, value) -> None:
    if torch.is_tensor(value):
        stats = ""
        if value.numel() and value.is_floating_point():
            fv = value.float()
            stats = (
                f" min={fv.min().item():.6g} max={fv.max().item():.6g}"
                f" mean={fv.mean().item():.6g} std={fv.std().item():.6g}"
            )
        print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}{stats}")
    else:
        print(f"{key}: {type(value).__name__} {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug SO101 front+wrist LingBot dataset samples.")
    parser.add_argument("--config-name", default="so101_front_wrist_train")
    args = parser.parse_args()
    cfg = VA_CONFIGS[args.config_name]
    ds = MultiLatentLeRobotDataset(cfg, num_init_worker=1)
    print("len(ds) =", len(ds))
    sample = ds[0]
    for key, value in sample.items():
        describe(key, value)


if __name__ == "__main__":
    main()
