#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def get_value(row: dict, *names: str):
    for name in names:
        if name in row:
            return row[name]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SO101 LingBot training loss curves.")
    parser.add_argument("--metrics_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.metrics_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No metrics found in {args.metrics_path}")
    steps = [row["step"] for row in rows]
    latent = [get_value(row, "latent_loss", "train/video_loss") for row in rows]
    action = [get_value(row, "action_loss", "train/action_loss") for row in rows]
    grad = [get_value(row, "grad_norm", "train/grad_norm") for row in rows]

    fig, axes = plt.subplots(3 if any(v is not None for v in grad) else 2, 1, figsize=(10, 9), sharex=True)
    if not isinstance(axes, (list, tuple)):
        axes = list(axes)
    axes[0].plot(steps, latent)
    axes[0].set_ylabel("latent_loss")
    axes[0].grid(alpha=0.3)
    axes[1].plot(steps, action, color="tab:orange")
    axes[1].set_ylabel("action_loss")
    axes[1].grid(alpha=0.3)
    if len(axes) > 2:
        axes[2].plot(steps, grad, color="tab:green")
        axes[2].set_ylabel("grad_norm")
        axes[2].grid(alpha=0.3)
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=180)
    print(f"Wrote {args.output_path}")


if __name__ == "__main__":
    main()
