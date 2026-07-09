#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wan_va" / "utils" / "Simple_Remote_Infer"))

import numpy as np
from einops import rearrange
from PIL import Image

from deploy.websocket_client_policy import WebsocketClientPolicy
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig


PROMPT = "Pick the green cube and place it inside the blue box"
FRONT_CAMERA_KEY = "observation.images.front"
WRIST_CAMERA_KEY = "observation.images.wrist"
JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def reset(policy: WebsocketClientPolicy, prompt: str) -> None:
    policy.infer({"reset": True, "n_view": 2, "prompt": prompt})


def infer(policy: WebsocketClientPolicy, obs: dict):
    obs["reset"] = False
    obs["compute_kv_cache"] = False
    ret = policy.infer(obs)
    raw_action = ret["action"]
    parsed_actions = rearrange(raw_action, "c t f -> (t f) c")
    return np.asarray(parsed_actions, dtype=np.float32), np.asarray(raw_action, dtype=np.float32)


def compute_kv_cache(policy: WebsocketClientPolicy, obs: dict, raw_action: np.ndarray) -> None:
    obs["reset"] = False
    obs["compute_kv_cache"] = True
    obs["imagine"] = False
    obs["state"] = raw_action
    policy.infer(obs)


def make_robot(args):
    cameras = {
        "front": OpenCVCameraConfig(index_or_path=args.front_camera_index, fps=30, width=640, height=480),
        "wrist": OpenCVCameraConfig(index_or_path=args.wrist_camera_index, fps=30, width=640, height=480),
    }
    config = SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras=cameras,
        use_degrees=True,
    )
    return SO101Follower(config)


def read_state(observation: dict) -> np.ndarray:
    return np.asarray([float(observation[key]) for key in JOINT_KEYS], dtype=np.float32)


def read_camera_obs(observation: dict) -> dict:
    return {
        FRONT_CAMERA_KEY: np.asarray(observation["front"]),
        WRIST_CAMERA_KEY: np.asarray(observation["wrist"]),
    }


def build_obs(history: list[dict], prompt: str) -> dict:
    return {
        "obs": [
            {
                FRONT_CAMERA_KEY: ob["front"],
                WRIST_CAMERA_KEY: ob["wrist"],
            }
            for ob in history[::-2][::-1]
        ],
        "prompt": [prompt],
        "reset": False,
        "compute_kv_cache": False,
    }


def clip_actions(actions: np.ndarray, current_state: np.ndarray) -> np.ndarray:
    out = np.asarray(actions, dtype=np.float32).copy()
    previous = current_state.astype(np.float32).copy()
    max_from_current = np.array([8, 8, 8, 8, 8, 8], dtype=np.float32)
    max_step = np.array([4, 4, 4, 4, 4, 5], dtype=np.float32)
    lower = current_state - max_from_current
    upper = current_state + max_from_current
    for i in range(len(out)):
        out[i] = np.clip(out[i], lower, upper)
        out[i] = np.clip(out[i], previous - max_step, previous + max_step)
        previous = out[i]
    return out


def parsed_to_raw(actions: np.ndarray, raw_shape: tuple[int, int, int]) -> np.ndarray:
    channels, frames, action_per_frame = raw_shape
    required = frames * action_per_frame
    padded = np.zeros((required, channels), dtype=np.float32)
    padded[: min(required, len(actions))] = actions[:required]
    return padded.reshape(frames, action_per_frame, channels).transpose(2, 0, 1)


def send_action(robot, action: np.ndarray) -> None:
    robot.send_action({key: float(value) for key, value in zip(JOINT_KEYS, action)})


def main() -> None:
    parser = argparse.ArgumentParser(description="SO101 front+wrist LingBot-VA websocket client.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29536)
    parser.add_argument("--robot_port", default="/dev/ttyACM1")
    parser.add_argument("--robot_id", default="follower_arm")
    parser.add_argument("--front_camera_index", type=int, default=2)
    parser.add_argument("--wrist_camera_index", type=int, default=4)
    parser.add_argument("--num_steps_to_execute", type=int, default=5)
    parser.add_argument("--execute", type=str2bool, default=False)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--debug_dir", type=Path, default=Path("/tmp/lingbot_so101_real_debug"))
    args = parser.parse_args()

    args.debug_dir.mkdir(parents=True, exist_ok=True)
    robot = make_robot(args)
    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    robot.connect()
    events = []
    try:
        first = robot.get_observation()
        camera_obs = read_camera_obs(first)
        Image.fromarray(camera_obs[FRONT_CAMERA_KEY]).save(args.debug_dir / "front_initial.png")
        Image.fromarray(camera_obs[WRIST_CAMERA_KEY]).save(args.debug_dir / "wrist_initial.png")
        state = read_state(first)

        reset(policy, args.prompt)
        parsed, raw = infer(policy, {"obs": [camera_obs], "prompt": [args.prompt]})
        clipped = clip_actions(parsed, state)
        steps = min(args.num_steps_to_execute, len(clipped))
        print(f"raw action shape: {raw.shape}")
        print(f"parsed action shape: {parsed.shape}")
        print(f"clipped actions first {steps}:\n{clipped[:steps]}")

        real_obs_history = []
        for action in clipped[:steps]:
            if args.execute:
                send_action(robot, action)
            time.sleep(1 / 30)
            ob = robot.get_observation()
            real_obs_history.append(
                {
                    "front": np.asarray(ob["front"]),
                    "wrist": np.asarray(ob["wrist"]),
                }
            )
            events.append({"action": action.tolist(), "executed": args.execute})

        executed_raw = parsed_to_raw(clipped, raw.shape)
        kv_obs = build_obs(real_obs_history, args.prompt)
        compute_kv_cache(policy, kv_obs, executed_raw)
        (args.debug_dir / "events.json").write_text(json.dumps(events, indent=2) + "\n")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
