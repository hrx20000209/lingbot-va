# Three Cubes SO101 post-training

This setup post-trains LingBot-VA on `/data/rxhuang/three_cubes_1` with the
`front` and `wrist` cameras. Model assets and checkpoints live under
`/home/rxhuang/Projects/models`, outside this repository.

## Data contract

- Source data: LeRobot v3, 30 FPS, six absolute SO101 joint actions.
- Video input: chronological 15 FPS frames resized to 256 x 256.
- Cameras: `observation.images.front`, `observation.images.wrist`.
- Action channels: source joints 0-4 map to LingBot channels 0-4; the source
  gripper maps to channel 28.
- Eight 30 Hz actions correspond to one latent frame. Four 15 FPS RGB frames
  are encoded for every new latent frame.
- The initial padded action block is masked and remains zero in normalized
  coordinates, matching inference.
- Episodes 0-89 train the model; episodes 90-99 are validation-only.

The conversion follows the format demonstrated by the LingBot-VA authors in
issue #29. It was checked against their episode 0 reference cache: frame IDs
and tensor shapes match exactly, and VAE latent cosine similarity is 0.99993.

## Environment

```bash
conda activate lingbot
pip install pyarrow
pip install lerobot==0.3.3 --no-deps
pip install draccus==0.10.0 'deepdiff>=7,<9' pyserial feetech-servo-sdk
```

The current repository requires PyTorch 2.9 and CUDA 12.6. The `lingbot`
environment on this machine has been updated accordingly.

## Prepare data

Run all stages on one GPU:

```bash
cd /home/rxhuang/Projects/lingbot-va
CUDA_VISIBLE_DEVICES=3 python -m wan_va.dataset.prepare_three_cubes --stage all
```

The stages can be run independently with `--stage metadata`, `embeddings`, or
`latents`. Camera jobs can also be split across GPUs with `--camera-keys`.
Prepared data is written to `/data/rxhuang/three_cubes_1_lingbot_v21`.

## Train

The default configuration updates the action input/output branches, action
timestep embedding, and the final two shared DiT blocks (416 million trainable
parameters). Training randomly crops 24 latent frames to fit two 24 GB GPUs
while preserving an approximately 6.4 second temporal context.

```bash
cd /home/rxhuang/Projects/lingbot-va
PYTORCH_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc-per-node=2 \
  -m wan_va.train \
  --config-name three_cubes_train
```

Useful command-line overrides include `--num-steps`, `--save-interval`,
`--validation-interval`, `--train-last-n-blocks`, `--learning-rate`,
`--gradient-accumulation-steps`, `--video-loss-weight`, and
`--action-loss-weight`.

Metrics are appended to `metrics.jsonl`. Plot them with:

```bash
python script/plot_training_metrics.py \
  /home/rxhuang/Projects/models/lingbot-va-three-cubes/action_last2/metrics.jsonl
```

## Dataset replay

This runs autoregressive inference on one held-out episode, updates the KV
cache with chronological RGB frames and previously predicted actions, and
plots prediction, ground-truth action, and `observation.state` together.

```bash
CUDA_VISIBLE_DEVICES=3 python script/evaluate_three_cubes_episode.py \
  --transformer-path \
  /home/rxhuang/Projects/models/lingbot-va-three-cubes/action_last2/checkpoints/last/transformer \
  --episode 90
```

The output directory contains the plot, raw curves, and per-joint MAE. Red
vertical lines in the plot are replanning boundaries.

## Real robot inference

Start the model server on a GPU:

```bash
bash script/run_three_cubes_server.sh
```

Then connect the SO101 and two cameras from another terminal:

```bash
bash script/run_three_cubes_async_robot.sh
```

The client executes actions at 30 Hz, captures RGB at 15 Hz, and starts the
next inference while actions remain in the current queue. New chunks replace
the overlapping queue after removing actions that elapsed during inference.
The server KV cache receives the actual executed model actions, not the robot's
current joint state. `async_inference.jsonl` records action, camera, inference,
latency, and queue events for timing analysis.
