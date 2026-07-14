# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .va_so101_cfg import va_so101_cfg


va_so101_train_cfg = EasyDict(__name__="Config: VA SO101 three cameras train")
va_so101_train_cfg.update(va_so101_cfg)

va_so101_train_cfg.dataset_path = "/data/rxhuang/three_cubes_1_lingbot"
va_so101_train_cfg.empty_emb_path = os.path.join(
    va_so101_train_cfg.dataset_path, "empty_emb.pt"
)

va_so101_train_cfg.enable_wandb = False
va_so101_train_cfg.load_worker = 4
va_so101_train_cfg.num_init_worker = 4
va_so101_train_cfg.dataset_init_workers = 1
# Matches the validated va_three_cubes_train_cfg precedent (2 cameras, 24
# frames on 2x24GB GPUs partial fine-tune). Kept at 8 here (vs. that config's
# 24) because 3 concatenated camera views widen every latent frame further.
va_so101_train_cfg.max_latent_frames = 8
va_so101_train_cfg.gc_interval = 1
va_so101_train_cfg.cfg_prob = 0.1

# Overfitting monitoring (issue #29 lesson: 83 episodes overfit by ~1000
# steps). train.py already wires validation_interval/val_episode_ids into its
# training loop (see Trainer.__init__ val_loader construction and
# Trainer.validate()) -- no core training-loop code changes needed here.
va_so101_train_cfg.val_episode_ids = [95, 96, 97, 98, 99]
va_so101_train_cfg.validation_interval = 50
va_so101_train_cfg.validation_batches = 5

va_so101_train_cfg.save_interval = 100
# Keep every checkpoint across the full 1500-step budget (15 checkpoints) so
# the best step can be selected post-hoc by val loss + MAE + video quality,
# not by training loss.
va_so101_train_cfg.max_checkpoints = 16

# Partial fine-tune (last-N-block + action branches), not full: full
# fine-tuning of this 5.09B-parameter transformer with 3 concatenated cameras
# was measured to OOM the backward pass on 24GB GPUs even at 6 free GPUs and
# max_latent_frames=4 (over budget by only ~40-88MB -- the bottleneck is the
# fixed per-GPU optimizer-state/gradient footprint of full-parameter AdamW,
# which FSDP shards across GPUs but does not shrink per sample, so more GPUs
# barely help). Matches the validated va_three_cubes_train_cfg precedent
# (train_last_n_blocks=2) and the maintainers' own recommendation in this
# machine's parallel lerobot-native pipeline (~/Projects/lerobot/scripts/
# train_lingbo_va.sh), which defaults to LoRA over full fine-tuning for the
# same reason.
va_so101_train_cfg.train_mode = "action_last_n"
va_so101_train_cfg.train_last_n_blocks = 2

va_so101_train_cfg.learning_rate = 1e-4
va_so101_train_cfg.beta1 = 0.9
va_so101_train_cfg.beta2 = 0.95
va_so101_train_cfg.weight_decay = 0.1
va_so101_train_cfg.warmup_steps = 10
va_so101_train_cfg.max_grad_norm = 2.0

va_so101_train_cfg.batch_size = 1
# Default matches demo's assumed 8-GPU cluster (1 x 8 accum x 8 GPUs = 64
# effective batch). script/run_va_posttrain_so101.sh overrides this via
# --gradient-accumulation-steps to keep the same effective batch on however
# many GPUs are actually free.
va_so101_train_cfg.gradient_accumulation_steps = 8
va_so101_train_cfg.num_steps = 1500
