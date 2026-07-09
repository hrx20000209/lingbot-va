# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .va_three_cubes_cfg import va_three_cubes_cfg


va_three_cubes_train_cfg = EasyDict(__name__="Config: VA Three Cubes train")
va_three_cubes_train_cfg.update(va_three_cubes_cfg)

va_three_cubes_train_cfg.dataset_path = "/data/rxhuang/three_cubes_1_lingbot_v21"
va_three_cubes_train_cfg.empty_emb_path = (
    "/data/rxhuang/three_cubes_1_lingbot_v21/empty_emb.pt"
)
va_three_cubes_train_cfg.val_episode_ids = list(range(90, 100))
va_three_cubes_train_cfg.dataset_init_workers = 1
va_three_cubes_train_cfg.load_worker = 2
va_three_cubes_train_cfg.max_latent_frames = 24
va_three_cubes_train_cfg.enable_wandb = False
va_three_cubes_train_cfg.save_root = (
    "/home/rxhuang/Projects/models/lingbot-va-three-cubes/action_last2"
)
va_three_cubes_train_cfg.save_interval = 50
va_three_cubes_train_cfg.max_checkpoints = 3
va_three_cubes_train_cfg.validation_interval = 10
va_three_cubes_train_cfg.validation_batches = 4
va_three_cubes_train_cfg.metrics_interval = 1
va_three_cubes_train_cfg.gc_interval = 1
va_three_cubes_train_cfg.cfg_prob = 0.1

# Full DiT fine-tuning is the FastWAM-style default, but a 5B model's Adam
# states do not fit comfortably on two 24 GB cards. Updating the action branch
# and final shared blocks still lets visual features adapt to the real cameras.
va_three_cubes_train_cfg.train_mode = "action_last_n"
va_three_cubes_train_cfg.train_last_n_blocks = 2
va_three_cubes_train_cfg.learning_rate = 2e-5
va_three_cubes_train_cfg.beta1 = 0.9
va_three_cubes_train_cfg.beta2 = 0.95
va_three_cubes_train_cfg.weight_decay = 1e-2
va_three_cubes_train_cfg.warmup_steps = 10
va_three_cubes_train_cfg.batch_size = 1
va_three_cubes_train_cfg.gradient_accumulation_steps = 4
va_three_cubes_train_cfg.video_loss_weight = 0.1
va_three_cubes_train_cfg.action_loss_weight = 1.0
va_three_cubes_train_cfg.max_grad_norm = 2.0
va_three_cubes_train_cfg.num_steps = 500
