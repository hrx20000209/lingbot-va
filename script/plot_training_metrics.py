#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot LingBot-VA train and validation losses.")
    parser.add_argument("metrics", type=Path, help="Path to metrics.jsonl")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.metrics.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No metrics in {args.metrics}")
    output = args.output or args.metrics.with_name("loss_curves.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for axis, loss_name, title in zip(
        axes,
        ("video_loss", "action_loss"),
        ("Video flow loss", "Action flow loss"),
    ):
        train = [(row["step"], row[f"train/{loss_name}"]) for row in rows]
        val = [
            (row["step"], row[f"val/{loss_name}"])
            for row in rows
            if f"val/{loss_name}" in row
        ]
        axis.plot(*zip(*train), label="train", linewidth=1.5)
        if val:
            axis.plot(*zip(*val), label="validation", marker="o", markersize=3)
        axis.set_title(title)
        axis.set_xlabel("optimizer step")
        axis.set_ylabel("loss")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("LingBot-VA Three Cubes post-training")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
