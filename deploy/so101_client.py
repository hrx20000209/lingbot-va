#!/usr/bin/env python
"""Closed-/open-loop SO101 real-robot inference client for the 3-camera
(front/right/wrist) LingBot-VA post-train.

Edge on-device deployment: this process loads VA_Server directly in-process
(same GPU, same Python process as the robot control loop) -- there is no
websocket hop, no separate `launch_server_so101.sh` process, and no second
machine. `--config-name`/`--checkpoint` build the same VA_Server used by
`script/launch_server_so101.sh` and `tools/eval_so101_front_wrist_replay_curve.py`;
see those for the config/checkpoint conventions.

Forked from script/async_so101_client.py, which already implements the
correct pattern: an async worker thread that (1) updates the KV cache with
the actions just executed plus fresh camera frames, then (2) requests the
next action chunk, while the main loop keeps executing queued actions at
--action-hz without blocking on inference -- unchanged here, just pointed at
a local VA_Server instance instead of a WebsocketClientPolicy. See that file
for the original 2-camera version; this one adds:

  - a third camera (right) and CLI-configurable indices/resolution
  - an explicit, asserted downsample-stride derivation instead of a bare
    "every 2nd action" modulo check
  - client-side clip of the already-denormalized action to the dataset's
    q01/q99 range (norm_stat.json), since wan_va_server.py's
    postprocess_action denormalizes but does not clip
  - --dry-run (skip robot.send_action, otherwise run the full perception/
    inference/KV-cache loop) and --open-loop (advance the KV cache with the
    model's own predicted video latent instead of real camera frames --
    never skips the KV-cache update)
  - norm_stat.json md5 printed at startup so it can be diffed against
    whatever norm_stat.json a training run used

Key trace notes (see comments inline for where these are used):
  * wan_va_server.py's `_infer` only calls `_encode_obs` (i.e. actually looks
    at the `obs` argument) when frame_st_id == 0. For every later chunk, the
    server relies entirely on the KV cache that compute_kv_cache calls have
    already built up, so the `obs` passed alongside a plain generate call is
    vestigial after the first chunk -- this is why real observations only
    need to flow through the separate compute_kv_cache request.
  * The first generated chunk's leading frame slot is a zero condition
    placeholder (wan_va_server.py `_infer`: `actions[:, :, 0:1] = action_cond`
    when frame_st_id == 0), not a real action, so it must be dropped from the
    very first chunk only (`drop_condition_block=True` iff `initial`). Later
    chunks have frame_st_id != 0, so action_cond is None and every frame in
    the chunk is a real predicted action -- nothing to drop.
"""

import argparse
import copy
import hashlib
import json
import logging
import queue
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from wan_va.configs import VA_CONFIGS
from wan_va.configs.va_so101_cfg import NORM_STAT_PATH as DEFAULT_NORM_STAT_PATH
from wan_va.wan_va_server import VA_Server


JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
CAMERA_MAP = {
    "front": "observation.images.front",
    "right": "observation.images.right",
    "wrist": "observation.images.wrist",
}


def downsample_stride(dataset_fps: int, target_fps: int) -> int:
    """Ratio between the action-execution rate (dataset_fps, e.g. 30Hz robot
    control) and the training-time keyframe sampling rate (target_fps, e.g.
    15Hz VAE latent extraction -- see tools/convert_three_cubes.py
    --target-fps). Every `stride`-th executed action is the point at which a
    fresh camera observation is captured, matching how training data was
    subsampled (see wan_va/dataset/prepare_three_cubes.py extract_latents:
    `stride = source_fps // target_fps`).
    """
    if dataset_fps % target_fps:
        raise ValueError(f"dataset_fps={dataset_fps} must be divisible by target_fps={target_fps}")
    return dataset_fps // target_fps


def load_norm_stat(path: Path) -> dict:
    stats = json.loads(path.read_text())
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    logging.info(f"norm_stat.json md5={md5} path={path}")
    return stats


def flatten_action_chunk(action, drop_condition_block=False):
    action = np.asarray(action)
    if action.ndim != 3:
        raise ValueError(f"Expected [C,F,N] action chunk, got {action.shape}")
    if drop_condition_block:
        # See module docstring: the first chunk's leading frame slot is a
        # zero placeholder, not a real predicted action.
        action = action[:, 1:, :]
    return action.transpose(1, 2, 0).reshape(-1, action.shape[0])


def pack_action_context(actions, include_initial_condition):
    actions = np.asarray(actions, dtype=np.float32)
    if len(actions) % 8:
        raise ValueError(f"Action context must contain complete 8-action blocks, got {len(actions)}")
    context = actions.reshape(-1, 8, 6).transpose(2, 0, 1)
    if include_initial_condition:
        context = np.concatenate([np.zeros((6, 1, 8), dtype=np.float32), context], axis=1)
    return context


def clip_to_norm_stat(action: np.ndarray, norm_stat: dict) -> np.ndarray:
    """Clip the server's already-denormalized 6-dim action to the dataset's
    q01/q99 range (norm_stat['q01_source']/['q99_source'], one value per real
    joint channel -- see tools/compute_three_cubes_norm_stats.py). The server
    denormalizes (wan_va_server.py postprocess_action) but does not clip, so
    diffusion outputs that fall outside the trained range are not otherwise
    bounded before being sent to hardware.
    """
    q01 = np.asarray(norm_stat["q01_source"], dtype=np.float32)
    q99 = np.asarray(norm_stat["q99_source"], dtype=np.float32)
    return np.clip(action, q01, q99)


@dataclass
class InferenceRequest:
    start_step: int
    observations: list[dict[str, np.ndarray]]
    action_context: np.ndarray | None
    imagine_latent: np.ndarray | None = None
    initial: bool = False


class InferenceWorker(threading.Thread):
    """Runs VA_Server.infer() in-process on a background thread so the main
    loop can keep dequeuing/executing actions at --action-hz without blocking
    on inference. `server` is a single VA_Server instance built once in
    main() and touched only from this thread -- the main thread never calls
    server.infer(), so there is no need for a lock around it.
    """

    def __init__(self, server, prompt, initial_observation, request_queue, result_queue, open_loop):
        super().__init__(daemon=True)
        self.server = server
        self.prompt = prompt
        self.initial_observation = initial_observation
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.open_loop = open_loop

    def run(self):
        try:
            self.server.infer({"reset": True, "prompt": self.prompt})
            while True:
                request = self.request_queue.get()
                if request is None:
                    return
                started = time.monotonic()
                kv_timing = {}
                if not request.initial:
                    if self.open_loop:
                        # Advance the KV cache with this model's own predicted
                        # video latent (from the previous infer() call) rather
                        # than skipping the update or requiring fresh camera
                        # frames. See wan_va_server.py _compute_kv_cache's
                        # imagine_latent branch.
                        kv_ret = self.server.infer(
                            {
                                "imagine_latent": request.imagine_latent,
                                "compute_kv_cache": True,
                                "pred_action": request.action_context,
                            }
                        )
                    else:
                        kv_ret = self.server.infer(
                            {
                                "obs": request.observations,
                                "compute_kv_cache": True,
                                "pred_action": request.action_context,
                            }
                        )
                    kv_timing = kv_ret.get("server_timing", {})
                kv_done = time.monotonic()
                # obs is vestigial here after the first chunk (see module
                # docstring), kept only because VA_Server.infer expects it.
                result = self.server.infer({"obs": [self.initial_observation]})
                self.result_queue.put(
                    {
                        "start_step": request.start_step,
                        "action": result["action"],
                        "pred_latent": result.get("pred_latent"),
                        "initial": request.initial,
                        "kv_cache_latency_s": kv_done - started,
                        "kv_cache_server_timing": kv_timing,
                        "infer_latency_s": time.monotonic() - kv_done,
                        "infer_server_timing": result.get("server_timing", {}),
                        "total_latency_s": time.monotonic() - started,
                    }
                )
        except Exception as exc:
            self.result_queue.put({"error": repr(exc)})


def parse_args():
    parser = argparse.ArgumentParser(
        description="SO101 (front/right/wrist) real-robot inference client for LingBot-VA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-name", default="so101")
    parser.add_argument("--checkpoint", type=Path, default=Path("train_out/so101_three_cubes/checkpoints/last"))
    parser.add_argument("--guidance-scale", type=float, default=None, help="Override the config's default if set.")
    parser.add_argument("--action-guidance-scale", type=float, default=None, help="Override the config's default if set.")
    parser.add_argument("--attn-window", type=int, default=None, help="Override the config's default if set.")
    parser.add_argument("--enable-offload", action="store_true",
                         help="Offload VAE/text encoder to CPU to save VRAM (see va_so101_cfg.py).")
    parser.add_argument("--save-root", type=Path, default=Path("train_out/so101_three_cubes/deploy_debug"))
    parser.add_argument("--robot-port", default="/dev/ttyACM1")
    parser.add_argument("--robot-id", default="follower_arm")
    parser.add_argument("--front-camera", type=int, default=4)
    parser.add_argument("--right-camera", type=int, default=6)
    parser.add_argument("--wrist-camera", type=int, default=2)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--model-width", type=int, default=256, help="Resize to training resolution before running inference.")
    parser.add_argument("--model-height", type=int, default=256)
    parser.add_argument(
        "--task",
        default="go to red cube. take the red cube. go to box. put the red cube in box.",
    )
    parser.add_argument("--dataset-fps", type=int, default=30, help="Robot control / dataset native fps.")
    parser.add_argument("--target-fps", type=int, default=15, help="Training-time keyframe sampling fps (must match tools/convert_three_cubes.py --target-fps).")
    parser.add_argument("--action-hz", type=float, default=30.0)
    parser.add_argument("--replan-remaining-actions", type=int, default=16)
    parser.add_argument("--max-relative-target", type=float, default=5.0,
                         help="Max per-step joint delta in degrees; enforced by LeRobot's SO101Follower.")
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument(
        "--norm-stat", type=Path, default=None,
        help="Defaults to NORM_STAT_PATH resolved by wan_va/configs/va_so101_cfg.py "
        "(same artifacts-dir candidates the loaded config itself uses).",
    )
    parser.add_argument("--dry-run", action="store_true",
                         help="Run perception/inference/KV-cache loop but do not send actions to the robot.")
    parser.add_argument("--open-loop", action="store_true",
                         help="Advance the KV cache with the model's own predicted video latent instead of "
                              "real camera frames (still calls compute_kv_cache every replan -- never skipped).")
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("train_out/so101_three_cubes/deploy_inference.jsonl"),
    )
    return parser.parse_args()


def make_robot(args):
    cameras = {
        "front": OpenCVCameraConfig(index_or_path=args.front_camera, fps=30, width=args.camera_width, height=args.camera_height),
        "right": OpenCVCameraConfig(index_or_path=args.right_camera, fps=30, width=args.camera_width, height=args.camera_height),
        "wrist": OpenCVCameraConfig(index_or_path=args.wrist_camera, fps=30, width=args.camera_width, height=args.camera_height),
    }
    config = SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras=cameras,
        use_degrees=True,
        max_relative_target=args.max_relative_target,
    )
    return SO101Follower(config)


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    import cv2

    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def camera_observation(robot_observation, model_width, model_height):
    return {
        model_key: resize_frame(robot_observation[robot_key], model_width, model_height)
        for robot_key, model_key in CAMERA_MAP.items()
    }


def write_event(path, event, start_time):
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"time_s": time.monotonic() - start_time, **event}
    with open(path, "a") as file:
        file.write(json.dumps(event) + "\n")


def build_server(args) -> VA_Server:
    """Load VA_Server in-process on the local GPU. Same config/checkpoint
    conventions as script/launch_server_so101.sh and
    tools/eval_so101_front_wrist_replay_curve.py, minus the network layer --
    there is no separate server process to launch on Thor.
    """
    cfg = copy.deepcopy(VA_CONFIGS[args.config_name])
    checkpoint = args.checkpoint.expanduser().resolve()
    cfg.transformer_path = str(checkpoint / "transformer" if (checkpoint / "transformer").exists() else checkpoint)
    if args.guidance_scale is not None:
        cfg.guidance_scale = args.guidance_scale
    if args.action_guidance_scale is not None:
        cfg.action_guidance_scale = args.action_guidance_scale
    if args.attn_window is not None:
        cfg.attn_window = args.attn_window
    if args.enable_offload:
        cfg.enable_offload = True
    cfg.save_root = str(args.save_root)
    cfg.rank = cfg.local_rank = 0
    cfg.world_size = 1
    return VA_Server(cfg)


def main() -> None:
    args = parse_args()
    if args.replan_remaining_actions % 8:
        raise ValueError("--replan-remaining-actions must be a multiple of 8")
    stride = downsample_stride(args.dataset_fps, args.target_fps)
    assert stride >= 1
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info(f"downsample stride={stride} (dataset_fps={args.dataset_fps} / target_fps={args.target_fps})")
    norm_stat = load_norm_stat(args.norm_stat or DEFAULT_NORM_STAT_PATH)

    logging.info(f"Loading VA_Server in-process: config={args.config_name} checkpoint={args.checkpoint}")
    server = build_server(args)

    robot = make_robot(args)
    robot.connect()
    start_time = time.monotonic()
    stopping = False

    def stop_handler(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    request_queue = queue.Queue()
    result_queue = queue.Queue()
    try:
        first_robot_observation = robot.get_observation()
        first_camera_observation = camera_observation(first_robot_observation, args.model_width, args.model_height)
        worker = InferenceWorker(
            server,
            args.task,
            first_camera_observation,
            request_queue,
            result_queue,
            args.open_loop,
        )
        worker.start()
        # reset=True only once, at episode start (n_view=1 equivalent: the
        # server's _reset call takes no observation, just the prompt).
        request_queue.put(InferenceRequest(0, [], None, initial=True))

        action_queue = deque()
        pending_actions = []
        pending_observations = []
        last_pred_latent = None
        inference_pending = True
        first_result = True
        global_step = 0
        next_action_time = time.monotonic()

        while not stopping and time.monotonic() - start_time < args.max_seconds:
            try:
                while True:
                    result = result_queue.get_nowait()
                    if "error" in result:
                        raise RuntimeError(result["error"])
                    flat = flatten_action_chunk(result["action"], drop_condition_block=result["initial"])
                    flat = np.stack([clip_to_norm_stat(row, norm_stat) for row in flat])
                    elapsed = max(0, global_step - result["start_step"])
                    if elapsed < len(flat):
                        action_queue = deque(flat[elapsed:])
                    else:
                        action_queue.clear()
                    last_pred_latent = result["pred_latent"]
                    inference_pending = False
                    first_result = False
                    write_event(
                        args.log_path,
                        {
                            "event": "inference_end",
                            "start_step": result["start_step"],
                            "current_step": global_step,
                            "kv_cache_latency_s": result["kv_cache_latency_s"],
                            "kv_cache_server_timing": result["kv_cache_server_timing"],
                            "infer_latency_s": result["infer_latency_s"],
                            "infer_server_timing": result["infer_server_timing"],
                            "total_latency_s": result["total_latency_s"],
                            "queue_size": len(action_queue),
                        },
                        start_time,
                    )
            except queue.Empty:
                pass

            if not action_queue:
                if inference_pending:
                    time.sleep(0.002)
                    continue
                if first_result:
                    continue
                logging.warning("Action queue underrun; waiting for the next inference result")
                time.sleep(0.002)
                continue

            now = time.monotonic()
            if now < next_action_time:
                time.sleep(next_action_time - now)
            action = np.asarray(action_queue.popleft(), dtype=np.float32)
            if args.dry_run:
                logging.info(f"[dry-run] step={global_step} action={action.tolist()}")
            else:
                robot.send_action(dict(zip(JOINT_NAMES, action.tolist())))
            pending_actions.append(action)
            global_step += 1
            write_event(
                args.log_path,
                {"event": "action", "step": global_step, "queue_size": len(action_queue), "dry_run": args.dry_run},
                start_time,
            )
            next_action_time += 1.0 / args.action_hz

            if len(pending_actions) % stride == 0:
                observation = camera_observation(robot.get_observation(), args.model_width, args.model_height)
                pending_observations.append(observation)
                write_event(
                    args.log_path,
                    {"event": "camera", "step": global_step},
                    start_time,
                )

            can_replan = (
                not inference_pending
                and len(action_queue) <= args.replan_remaining_actions
                and len(pending_actions) >= 8
                and len(pending_actions) % 8 == 0
            )
            if can_replan:
                include_condition = global_step == len(pending_actions)
                action_context = pack_action_context(pending_actions, include_condition)
                if not args.open_loop:
                    # The last sampled observation is always the frame captured
                    # right after the most recently executed action (assertion
                    # below), so KV-cache context never lags behind actuation.
                    expected_observations = (action_context.shape[1] - int(include_condition)) * (8 // stride)
                    if len(pending_observations) != expected_observations:
                        raise RuntimeError(
                            f"Camera/action context mismatch: {len(pending_observations)} images for "
                            f"{action_context.shape[1]} action blocks (stride={stride})"
                        )
                    assert pending_observations, "closed-loop replan requires at least one sampled observation"
                request_queue.put(
                    InferenceRequest(
                        start_step=global_step,
                        observations=pending_observations,
                        action_context=action_context,
                        imagine_latent=last_pred_latent,
                    )
                )
                write_event(
                    args.log_path,
                    {
                        "event": "inference_start",
                        "start_step": global_step,
                        "queue_size": len(action_queue),
                        "context_blocks": action_context.shape[1],
                        "open_loop": args.open_loop,
                    },
                    start_time,
                )
                pending_actions = []
                pending_observations = []
                inference_pending = True
    finally:
        request_queue.put(None)
        robot.disconnect()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
