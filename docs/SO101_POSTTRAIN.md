# SO101 (front/right/wrist) post-training and deployment

This is a 3-camera iteration of `docs/three_cubes_posttraining.md`. Where that
doc's pipeline (2 cameras, last-2-block partial fine-tune, dataset mutated in
place) is still valid and has real checkpoints under
`/data/rxhuang/lingbot_va_so101_front_wrist_runs`, this one is independent: a
fresh, non-destructive dataset conversion at `/data/rxhuang/three_cubes_1_lingbot`,
matching the official LingBot-VA demo config's defaults wherever they don't
conflict with 3 cameras / overfitting monitoring.

Training is a **partial fine-tune** (`train_mode="action_last_n"`,
`train_last_n_blocks=2` -- last 2 transformer blocks + action branches only),
not full fine-tuning. Full fine-tuning of this 5.09B-parameter transformer
with 3 concatenated cameras was measured to OOM 24GB GPUs even at 6 free GPUs
and a training context (`max_latent_frames`) cut to 4 -- over budget by only
~40-88MB, right at the edge. The bottleneck is the fixed per-GPU
optimizer-state/gradient footprint of full-parameter AdamW: FSDP shards it
across GPUs, but per-sample activation memory doesn't shrink, so adding more
GPUs barely helps once that floor is hit. This matches the same machine's
parallel lerobot-native pipeline (`~/Projects/lerobot/scripts/train_lingbo_va.sh`),
which documents `TRAIN_MODE=full` as "untested... expect to need multi-GPU"
and defaults to LoRA instead.

Action channel mapping is `used_action_channel_ids = [0, 1, 2, 3, 4, 28]`
(arm joints -> channels 0-4, gripper -> channel 28), matching the official
`va_demo_cfg.py` template and cross-checked against the maintainers' issue #29
episode-0 reference cache (VAE latent cosine similarity 0.99993) -- **not** a
naive contiguous `range(6)` mapping.

## 1. Convert the dataset

```bash
cd ~/Projects/lingbot-va
conda activate lingbot

# metadata + text embeddings + VAE latents for front/right/wrist, 15fps, 256x256
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python3 tools/convert_three_cubes.py --stage all --device cuda

# q01/q99 quantiles computed ONLY on the training split (episodes 0-94);
# episodes 95-99 are held out validation and never touch these stats.
python3 tools/compute_three_cubes_norm_stats.py --val-episodes 95-99

# structural diff vs. the official issue #29 example dataset + latent/video
# alignment spot-check + normalized action range check
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python3 tools/inspect_dataset_format.py --device cuda
```

Output: `/data/rxhuang/three_cubes_1_lingbot/` (episodes.jsonl, latents/,
task_emb.pt, empty_emb.pt, norm_stat.json, conversion.json). This directory is
independent from `/data/rxhuang/three_cubes_1` (untouched raw source) and
`/data/rxhuang/three_cubes_1_lingbot_v21` (existing 2-camera conversion).

## 2. Train

```bash
cd ~/Projects/lingbot-va
conda activate lingbot

# auto-detects free GPUs, scales gradient_accumulation_steps to match the
# demo's effective batch size (64) on however many GPUs are actually free.
# Partial fine-tune: last 2 transformer blocks + action branches
# (train_mode="action_last_n" in va_so101_train_cfg.py -- full fine-tune OOMs,
# see that config's comment).
GPUS=1,2,3,5,7 script/run_va_posttrain_so101.sh
```

In a second terminal, watch for overfitting per checkpoint (val loss is
already computed in-process by `wan_va/train.py`'s `validate()` every 50
steps via `validation_interval`/`val_episode_ids` in the config -- no core
training code was changed for this):

```bash
python script/monitor_so101_checkpoints.py \
  --save-root train_out/so101_three_cubes \
  --train-episode 0 --val-episode 95
```

Plot train/val loss curves:

```bash
python script/plot_training_metrics.py train_out/so101_three_cubes/train_metrics.jsonl
```

Pick the best checkpoint under `train_out/so101_three_cubes/checkpoints/` by
val loss + per-joint MAE (from `monitor_so101_checkpoints.py`'s debug output
under `train_out/so101_three_cubes/debug/step_NNNNNN/{train_episode,val_episode}/`)
+ video quality -- not training loss. Expect a useful window around
300-800 steps per issue #29 (100 episodes here vs. their 83).

## 3. Deploy on the real robot (on-device, single process)

`deploy/so101_client.py` loads `VA_Server` **in-process** -- there is no
websocket hop and no separate server process; the same Python process that
runs the robot control loop also holds the model on the local GPU. An async
worker thread calls `VA_Server.infer()` directly (KV-cache update, then next
action chunk) while the main thread keeps executing queued actions at
`--action-hz` without blocking on inference. `script/launch_server_so101.sh`
+ a network client is still there as a separate two-machine debug path (e.g.
testing checkpoints from a workstation without the robot attached), but it is
not the deployment path below.

**Copy artifacts to the edge device first** (checkpoint + task/negative
prompt embeddings + norm stats -- none of these are in git; `wan_va/configs/va_so101_cfg.py`
resolves them from `~/Projects/models/lingbot-va-so101-artifacts/` first,
falling back to the training machine's own `/data/rxhuang/three_cubes_1_lingbot`
path):

```bash
# on the training machine
mkdir -p /tmp/so101_artifacts
cp /data/rxhuang/three_cubes_1_lingbot/{norm_stat.json,task_emb.pt,empty_emb.pt} /tmp/so101_artifacts/
rsync -avz /tmp/so101_artifacts/ thor:~/Projects/models/lingbot-va-so101-artifacts/
rsync -avz train_out/so101_three_cubes/checkpoints/last/ thor:~/Projects/lingbot-va/train_out/so101_three_cubes/checkpoints/last/
```

Then, on Thor, dry run first (runs the full perception/inference/KV-cache
loop without sending motor commands):

```bash
cd ~/Projects/lingbot-va
conda activate lingbot
python deploy/so101_client.py \
  --front-camera 4 --right-camera 6 --wrist-camera 2 \
  --dry-run --max-seconds 120
```

Then closed-loop for real:

```bash
python deploy/so101_client.py --front-camera 4 --right-camera 6 --wrist-camera 2
```

Or open-loop (KV cache advanced with the model's own predicted video latent
instead of real camera frames -- never skips the KV-cache update):

```bash
python deploy/so101_client.py --front-camera 4 --right-camera 6 --wrist-camera 2 --open-loop
```

If Thor's GPU memory is tight, add `--enable-offload` to move the VAE/text
encoder to CPU (see `va_so101_cfg.py`'s `enable_offload`; this is the same
flag `tools/eval_so101_front_wrist_replay_curve.py --enable_offload` uses).

`deploy/so101_client.py` prints the md5 of `norm_stat.json` at startup --
compare it against the training machine's own norm_stat.json to confirm the
copied artifacts actually match this checkpoint. Latency is logged to
`--log-path` (default `train_out/so101_three_cubes/deploy_inference.jsonl`);
per-stage timing (`obs_encode_s`/`video_loop_s`/`action_loop_s`/
`kv_update_s`) rides along in each event's `*_server_timing` field.

## Notes

- `wan_va/train.py` and `wan_va/modules/model.py` training semantics are
  unmodified. `wan_va/wan_va_server.py` got two small additive changes: (1)
  `infer()` now also returns the raw predicted video latent as `pred_latent`
  and per-stage `server_timing`, and (2) `_compute_kv_cache` accepts an
  optional `imagine_latent` to bypass real-observation encoding -- both are
  no-ops for existing closed-loop callers that don't pass the new fields.
- `tools/eval_so101_front_wrist_replay_curve.py` was generalized from a
  hardcoded 2-camera (front/wrist) KV-cache-update loop to loop over
  `cfg.obs_cam_keys` generically; behavior for existing 2-camera configs is
  unchanged. It skips the KV-cache-update call on an episode's final chunk
  (the truncated tail otherwise gives the streaming VAE too few frames for
  its causal time-conv, and the update is unused anyway since no further
  chunk follows) and gained `--enable_offload`.
- `deploy/so101_client.py` was switched from a `WebsocketClientPolicy` network
  client to loading `VA_Server` directly in-process (see section 3) for
  on-device/edge deployment where the model runs on the same machine as the
  robot control loop. `wan_va/configs/va_so101_cfg.py`'s norm_stat/prompt-embedding
  paths were made resolvable from multiple candidate directories (was a single
  hardcoded training-machine path) so the same config works unmodified after
  copying those artifacts to an edge device.
