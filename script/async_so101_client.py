#!/usr/bin/env python

import argparse
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

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import WebsocketClientPolicy


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
    "wrist": "observation.images.wrist",
}


def flatten_action_chunk(action, drop_condition_block=False):
    action = np.asarray(action)
    if action.ndim != 3:
        raise ValueError(f"Expected [C,F,N] action chunk, got {action.shape}")
    if drop_condition_block:
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


@dataclass
class InferenceRequest:
    start_step: int
    observations: list[dict[str, np.ndarray]]
    action_context: np.ndarray | None
    initial: bool = False


class InferenceWorker(threading.Thread):
    def __init__(self, host, port, prompt, initial_observation, request_queue, result_queue):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.prompt = prompt
        self.initial_observation = initial_observation
        self.request_queue = request_queue
        self.result_queue = result_queue

    def run(self):
        try:
            policy = WebsocketClientPolicy(host=self.host, port=self.port)
            policy.infer({"reset": True, "prompt": self.prompt})
            while True:
                request = self.request_queue.get()
                if request is None:
                    return
                started = time.monotonic()
                if not request.initial:
                    policy.infer(
                        {
                            "obs": request.observations,
                            "compute_kv_cache": True,
                            "pred_action": request.action_context,
                        }
                    )
                result = policy.infer({"obs": [self.initial_observation]})
                self.result_queue.put(
                    {
                        "start_step": request.start_step,
                        "action": result["pred_action"],
                        "initial": request.initial,
                        "latency_s": time.monotonic() - started,
                        "server_timing": result.get("server_timing", {}),
                    }
                )
        except Exception as exc:
            self.result_queue.put({"error": repr(exc)})


def parse_args():
    parser = argparse.ArgumentParser(description="Asynchronous LingBot-VA control for an SO101 follower.")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=29536)
    parser.add_argument("--robot-port", default="/dev/ttyACM1")
    parser.add_argument("--robot-id", default="follower_arm")
    parser.add_argument("--front-camera", type=int, default=4)
    parser.add_argument("--wrist-camera", type=int, default=2)
    parser.add_argument(
        "--task",
        default="go to red cube. take the red cube. go to box. put the red cube in box.",
    )
    parser.add_argument("--action-hz", type=float, default=30.0)
    parser.add_argument("--replan-remaining-actions", type=int, default=16)
    parser.add_argument("--max-relative-target", type=float, default=12.0)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("/home/rxhuang/Projects/models/lingbot-va-three-cubes/async_inference.jsonl"),
    )
    return parser.parse_args()


def make_robot(args):
    cameras = {
        "front": OpenCVCameraConfig(index_or_path=args.front_camera, fps=30, width=640, height=480),
        "wrist": OpenCVCameraConfig(index_or_path=args.wrist_camera, fps=30, width=640, height=480),
    }
    config = SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras=cameras,
        use_degrees=True,
        max_relative_target=args.max_relative_target,
    )
    return SO101Follower(config)


def camera_observation(robot_observation):
    return {model_key: robot_observation[robot_key] for robot_key, model_key in CAMERA_MAP.items()}


def write_event(path, event, start_time):
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"time_s": time.monotonic() - start_time, **event}
    with open(path, "a") as file:
        file.write(json.dumps(event) + "\n")


def main():
    args = parse_args()
    if args.replan_remaining_actions % 8:
        raise ValueError("--replan-remaining-actions must be a multiple of 8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
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
        first_camera_observation = camera_observation(first_robot_observation)
        worker = InferenceWorker(
            args.server_host,
            args.server_port,
            args.task,
            first_camera_observation,
            request_queue,
            result_queue,
        )
        worker.start()
        request_queue.put(InferenceRequest(0, [], None, initial=True))

        action_queue = deque()
        pending_actions = []
        pending_observations = []
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
                    elapsed = max(0, global_step - result["start_step"])
                    if elapsed < len(flat):
                        action_queue = deque(flat[elapsed:])
                    else:
                        action_queue.clear()
                    inference_pending = False
                    first_result = False
                    write_event(
                        args.log_path,
                        {
                            "event": "inference_end",
                            "start_step": result["start_step"],
                            "current_step": global_step,
                            "latency_s": result["latency_s"],
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
            robot.send_action(dict(zip(JOINT_NAMES, action.tolist())))
            pending_actions.append(action)
            global_step += 1
            write_event(
                args.log_path,
                {"event": "action", "step": global_step, "queue_size": len(action_queue)},
                start_time,
            )
            next_action_time += 1.0 / args.action_hz

            if len(pending_actions) % 2 == 0:
                observation = camera_observation(robot.get_observation())
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
                expected_observations = (action_context.shape[1] - int(include_condition)) * 4
                if len(pending_observations) != expected_observations:
                    raise RuntimeError(
                        f"Camera/action context mismatch: {len(pending_observations)} images for "
                        f"{action_context.shape[1]} action blocks"
                    )
                request_queue.put(
                    InferenceRequest(
                        start_step=global_step,
                        observations=pending_observations,
                        action_context=action_context,
                    )
                )
                write_event(
                    args.log_path,
                    {
                        "event": "inference_start",
                        "start_step": global_step,
                        "queue_size": len(action_queue),
                        "context_blocks": action_context.shape[1],
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
    main()
