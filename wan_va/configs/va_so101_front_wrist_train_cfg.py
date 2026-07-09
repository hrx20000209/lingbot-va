# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .va_so101_front_wrist_cfg import va_so101_front_wrist_cfg


va_so101_front_wrist_train_cfg = EasyDict(__name__="Config: VA SO101 front+wrist train")
va_so101_front_wrist_train_cfg.update(va_so101_front_wrist_cfg)

va_so101_front_wrist_train_cfg.dataset_path = "/data/rxhuang/three_cubes_1"
va_so101_front_wrist_train_cfg.empty_emb_path = os.path.join(
    va_so101_front_wrist_train_cfg.dataset_path, "empty_emb.pt"
)

va_so101_front_wrist_train_cfg.enable_wandb = False
va_so101_front_wrist_train_cfg.load_worker = 4
va_so101_front_wrist_train_cfg.num_init_worker = 4
va_so101_front_wrist_train_cfg.max_latent_frames = 8
va_so101_front_wrist_train_cfg.save_interval = 50
va_so101_front_wrist_train_cfg.max_checkpoints = 4
va_so101_front_wrist_train_cfg.gc_interval = 1
va_so101_front_wrist_train_cfg.cfg_prob = 0.1

va_so101_front_wrist_train_cfg.learning_rate = 1e-4
va_so101_front_wrist_train_cfg.beta1 = 0.9
va_so101_front_wrist_train_cfg.beta2 = 0.95
va_so101_front_wrist_train_cfg.weight_decay = 0.1
va_so101_front_wrist_train_cfg.warmup_steps = 10

va_so101_front_wrist_train_cfg.batch_size = 1
va_so101_front_wrist_train_cfg.gradient_accumulation_steps = 8
va_so101_front_wrist_train_cfg.num_steps = 2000
