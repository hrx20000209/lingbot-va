# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import json
from pathlib import Path

from easydict import EasyDict

from .shared_config import va_shared_cfg


def _resolve_model_path() -> str:
    candidates = [
        Path("/home/hrx/Projects/models/lingbot-va-base"),
        Path("~/Projects/models/lingbot-va-base").expanduser(),
        Path("/data/rxhuang/models/lingbot-va-base"),
        Path("/home/rxhuang/Projects/models/lingbot-va-base"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


# Produced by tools/convert_three_cubes.py + tools/compute_three_cubes_norm_stats.py.
# Single source of truth for q01/q99 so training config, deploy config, and the
# SO101 client all load the same numbers (compared by md5 at deploy time).
NORM_STAT_PATH = Path("/data/rxhuang/three_cubes_1_lingbot/norm_stat.json")


def _load_norm_stat() -> dict:
    if not NORM_STAT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {NORM_STAT_PATH}. Run tools/compute_three_cubes_norm_stats.py "
            "after tools/convert_three_cubes.py --stage metadata."
        )
    stats = json.loads(NORM_STAT_PATH.read_text())
    return {"q01": stats["q01"], "q99": stats["q99"]}


va_so101_cfg = EasyDict(__name__="Config: VA SO101 three cameras (front/right/wrist)")
va_so101_cfg.update(va_shared_cfg)

va_so101_cfg.wan22_pretrained_model_name_or_path = _resolve_model_path()
va_so101_cfg.transformer_path = None
va_so101_cfg.infer_mode = "server"
# Fixed single-task pick&place: precomputed text embeddings let wan_va_server.py
# skip loading the tokenizer/text encoder at inference time (see VA_Server.__init__).
va_so101_cfg.prompt_emb_path = "/data/rxhuang/three_cubes_1_lingbot/task_emb.pt"
va_so101_cfg.negative_prompt_emb_path = "/data/rxhuang/three_cubes_1_lingbot/empty_emb.pt"

va_so101_cfg.attn_window = 30
va_so101_cfg.frame_chunk_size = 4
va_so101_cfg.env_type = "none"
va_so101_cfg.height = 256
va_so101_cfg.width = 256

va_so101_cfg.action_dim = 30
va_so101_cfg.action_per_frame = 8
va_so101_cfg.obs_cam_keys = [
    "observation.images.front",
    "observation.images.right",
    "observation.images.wrist",
]

va_so101_cfg.guidance_scale = 5
va_so101_cfg.action_guidance_scale = 1
va_so101_cfg.num_inference_steps = 5
va_so101_cfg.video_exec_step = -1
va_so101_cfg.action_num_inference_steps = 10
va_so101_cfg.snr_shift = 5.0
va_so101_cfg.action_snr_shift = 1.0

# SO101 6D action: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
# Validated against the official LingBot-VA issue #29 demo canvas: arm joints
# map to channels 0-4, gripper maps to channel 28 (matches va_demo_cfg.py's own
# `range(0,5) + range(28,29)`, and was cross-checked against the maintainers'
# episode-0 reference latent cache at cosine similarity 0.99993). NOT a naive
# contiguous range(6) mapping.
va_so101_cfg.used_action_channel_ids = [0, 1, 2, 3, 4, 28]
inverse_used_action_channel_ids = [
    len(va_so101_cfg.used_action_channel_ids)
] * va_so101_cfg.action_dim
for i, j in enumerate(va_so101_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_so101_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_so101_cfg.action_norm_method = "quantiles"
va_so101_cfg.norm_stat = _load_norm_stat()
