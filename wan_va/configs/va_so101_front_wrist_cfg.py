# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
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


va_so101_front_wrist_cfg = EasyDict(__name__="Config: VA SO101 front+wrist")
va_so101_front_wrist_cfg.update(va_shared_cfg)

va_so101_front_wrist_cfg.wan22_pretrained_model_name_or_path = _resolve_model_path()
va_so101_front_wrist_cfg.transformer_path = None
va_so101_front_wrist_cfg.infer_mode = "server"

va_so101_front_wrist_cfg.attn_window = 30
va_so101_front_wrist_cfg.frame_chunk_size = 4
va_so101_front_wrist_cfg.env_type = "none"
va_so101_front_wrist_cfg.height = 256
va_so101_front_wrist_cfg.width = 256

va_so101_front_wrist_cfg.action_dim = 30
va_so101_front_wrist_cfg.action_per_frame = 8
va_so101_front_wrist_cfg.obs_cam_keys = [
    "observation.images.front",
    "observation.images.wrist",
]

va_so101_front_wrist_cfg.guidance_scale = 5
va_so101_front_wrist_cfg.action_guidance_scale = 1
va_so101_front_wrist_cfg.num_inference_steps = 5
va_so101_front_wrist_cfg.video_exec_step = -1
va_so101_front_wrist_cfg.action_num_inference_steps = 10
va_so101_front_wrist_cfg.snr_shift = 5.0
va_so101_front_wrist_cfg.action_snr_shift = 1.0

# SO101 6D action:
# [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
# maps to the official LingBot-VA SO101 demo canvas: arm joints 0-4 + gripper 28.
va_so101_front_wrist_cfg.used_action_channel_ids = [0, 1, 2, 3, 4, 28]
inverse_used_action_channel_ids = [
    len(va_so101_front_wrist_cfg.used_action_channel_ids)
] * va_so101_front_wrist_cfg.action_dim
for i, j in enumerate(va_so101_front_wrist_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_so101_front_wrist_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_so101_front_wrist_cfg.action_norm_method = "quantiles"
va_so101_front_wrist_cfg.norm_stat = {
    "q01": [-41.758243560791016, -105.0988998413086, -60.74725341796875, 51.42856979370117, -5.142857074737549, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7137192487716675, 0.0],
    "q99": [10.90109920501709, 53.14285659790039, 93.62637329101562, 99.34066009521484, 79.16483306884766, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 49.484535217285156, 1.0],
}
