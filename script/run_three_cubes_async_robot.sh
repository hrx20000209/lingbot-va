#!/usr/bin/env bash
set -euo pipefail

cd /home/rxhuang/Projects/lingbot-va
/home/rxhuang/anaconda3/envs/lingbot/bin/python script/async_so101_client.py \
  --server-host 127.0.0.1 \
  --server-port 29536 \
  --robot-port /dev/ttyACM1 \
  --robot-id follower_arm \
  --front-camera 4 \
  --wrist-camera 2 \
  --action-hz 30 \
  --replan-remaining-actions 16 \
  --task "go to red cube. take the red cube. go to box. put the red cube in box."
