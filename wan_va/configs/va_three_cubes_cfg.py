# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import va_shared_cfg


va_three_cubes_cfg = EasyDict(__name__="Config: VA Three Cubes")
va_three_cubes_cfg.update(va_shared_cfg)

va_three_cubes_cfg.infer_mode = "server"
va_three_cubes_cfg.wan22_pretrained_model_name_or_path = (
    "/home/rxhuang/Projects/models/lingbot-va-base"
)
va_three_cubes_cfg.transformer_path = None
# The fixed task uses precomputed text embeddings, so only the VAE and
# transformer are loaded. Both fit on a 24 GB card and GPU VAE encoding is
# required for real-time camera updates.
va_three_cubes_cfg.enable_offload = False
va_three_cubes_cfg.prompt_emb_path = "/data/rxhuang/three_cubes_1_lingbot_v21/task_emb.pt"
va_three_cubes_cfg.negative_prompt_emb_path = (
    "/data/rxhuang/three_cubes_1_lingbot_v21/empty_emb.pt"
)

va_three_cubes_cfg.attn_window = 30
va_three_cubes_cfg.frame_chunk_size = 4
va_three_cubes_cfg.env_type = "none"
va_three_cubes_cfg.height = 256
va_three_cubes_cfg.width = 256
va_three_cubes_cfg.action_dim = 30
va_three_cubes_cfg.action_per_frame = 8
va_three_cubes_cfg.obs_cam_keys = [
    "observation.images.front",
    "observation.images.wrist",
]

va_three_cubes_cfg.guidance_scale = 1
va_three_cubes_cfg.action_guidance_scale = 1
va_three_cubes_cfg.num_inference_steps = 5
va_three_cubes_cfg.video_exec_step = -1
va_three_cubes_cfg.action_num_inference_steps = 10
va_three_cubes_cfg.snr_shift = 5.0
va_three_cubes_cfg.action_snr_shift = 1.0

# SO101 stores five arm joints followed by the gripper. LingBot-VA reserves
# channel 28 for the single-arm gripper, matching the authors' issue #29 demo.
va_three_cubes_cfg.used_action_channel_ids = [0, 1, 2, 3, 4, 28]
inverse_used_action_channel_ids = [len(va_three_cubes_cfg.used_action_channel_ids)] * 30
for source_index, model_index in enumerate(va_three_cubes_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[model_index] = source_index
va_three_cubes_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_three_cubes_cfg.action_norm_method = "quantiles"
q01_source = [
    -40.31460579864046,
    -104.92853466370016,
    -44.15901804910798,
    59.547760800456196,
    0.5714534416308061,
    8.959658986291599,
]
q99_source = [
    6.381584361833401,
    44.5878321615694,
    91.82374623055884,
    96.6553610284842,
    72.56077409460853,
    35.717574120828345,
]
q01 = [0.0] * 30
q99 = [1.0] * 30
for source_index, model_index in enumerate(va_three_cubes_cfg.used_action_channel_ids):
    q01[model_index] = q01_source[source_index]
    q99[model_index] = q99_source[source_index]
va_three_cubes_cfg.norm_stat = {"q01": q01, "q99": q99}
