# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import json
import numpy as np
import pandas as pd
from pathlib import Path
import os
import logging
from multiprocessing import Pool
from functools import partial
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R

def recursive_find_file(directory, filename='info.json'):
    result = []
    try:
        for root, dirs, files in os.walk(directory):
            if filename in files:
                full_path = os.path.join(root, filename)
                result.append(full_path)
    except PermissionError:
        print(f"Error: can not access {directory}")
    except Exception as e:
        print(f"Error: {e}")
    return result

def construct_lerobot(
    repo_id,
    config,
    split,
):
    return LatentLeRobotDataset(
        repo_id=repo_id,
        config=config,
        split=split,
    )

def construct_lerobot_multi_processor(config, 
                                      split="train",
                                      num_init_worker=8,
                                      ):
    construct_func = partial(
        construct_lerobot,
        config=config,
        split=split,
    )
    repo_list = recursive_find_file(config.dataset_path, 'info.json')
    repo_list = [v.split('/meta/info.json')[0] for v in repo_list]
    print(f"[MultiLatentLeRobotDataset] dataset_path={config.dataset_path}")
    print(f"[MultiLatentLeRobotDataset] found repos={repo_list}")
    if not repo_list:
        return []
    worker_count = min(num_init_worker, len(repo_list))
    if worker_count <= 1:
        return [construct_func(repo_id) for repo_id in repo_list]
    with Pool(worker_count) as pool:
        return pool.map(construct_func, repo_list)

def get_relative_pose(pose):
    if torch.is_tensor(pose):
        pose = pose.detach().cpu().numpy()
    
    rot = R.from_quat(pose[:, 3:7])
    first_rot = R.from_quat(np.tile(pose[:1, 3:7], (pose.shape[0], 1)))
    trans = pose[:, :3]
    relative_trans = trans - trans[0:1]

    relative_rot = first_rot.inv() * rot
    relative_quat = relative_rot.as_quat()

    relative_pose = np.concatenate([relative_trans, relative_quat], axis=1)
    return torch.from_numpy(relative_pose)

class MultiLatentLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        split="train",
        num_init_worker=None,
    ):
        if num_init_worker is None:
            num_init_worker = getattr(config, "dataset_init_workers", 1)
        self._datasets = construct_lerobot_multi_processor(config,
                                                           split,
                                                           num_init_worker, 
                                                           )
        self._datasets = [dataset for dataset in self._datasets if len(dataset)]
        if not self._datasets:
            raise ValueError(f"No LingBot-VA samples found for split={split!r} in {config.dataset_path}")
        self.item_id_to_dataset_id, self.acc_dset_num = (
            self._get_item_id_to_dataset_id()
        )

    def __len__(
        self,
    ):
        return sum(len(v) for v in self._datasets)

    def _get_item_id_to_dataset_id(self):
        item_id_to_dataset_id = {}
        acc_dset_num = {}
        acc_nums = [0]
        id = 0
        for dset_id, dset in enumerate(self._datasets):
            acc_nums.append(acc_nums[-1] + len(dset))
            for _ in range(len(dset)):
                item_id_to_dataset_id[id] = dset_id
                id += 1
        for did in range(len(self._datasets)):
            acc_dset_num[did] = acc_nums[did]
        return item_id_to_dataset_id, acc_dset_num

    def __getitem__(self, idx) -> dict:
        assert idx < len(self)
        cur_dset = self._datasets[self.item_id_to_dataset_id[idx]]
        local_idx = idx - self.acc_dset_num[self.item_id_to_dataset_id[idx]]
        return cur_dset[local_idx]

class LatentLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id,
        config=None,
        split="train",
    ):
        self.root = Path(repo_id).expanduser().resolve()
        self.config = config
        self.split = split
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        if self.info.get("codebase_version") not in {"v2.1", "v3.0"}:
            raise ValueError(
                f"LingBot-VA expects LeRobot v2.1/v3.0 metadata, got "
                f"{self.info.get('codebase_version')!r} at {self.root}. "
                "Check dataset_path and metadata."
            )
        self.chunks_size = int(self.info.get("chunks_size", 1000))
        self.data_path_template = self.info["data_path"]
        episodes_path = self.root / "meta" / "episodes.jsonl"
        if not episodes_path.exists():
            raise FileNotFoundError(
                f"Missing {episodes_path}. Run tools/prepare_so101_front_wrist_action_config.py "
                "to create action_config entries before training."
            )
        self.episode_rows = [
            json.loads(line)
            for line in episodes_path.read_text().splitlines()
            if line.strip()
        ]
        self.episode_rows = self._select_split(self.episode_rows)
        self._episode_action_cache = {}
        self._all_actions_by_episode = None
        self._debug_sample_printed = False
        self.latent_path = self.root / 'latents'
        self.empty_emb = torch.load(config.empty_emb_path, weights_only=False)
        self.cfg_prob = config.cfg_prob
        self.used_video_keys = config.obs_cam_keys
        self.q01 = np.array(config.norm_stat['q01'], dtype='float')[None]
        self.q99 = np.array(config.norm_stat['q99'], dtype='float')[None]
        image_keys = [
            key for key, feature in self.info.get("features", {}).items()
            if feature.get("dtype") == "video" or key.startswith("observation.images.")
        ]
        print(f"[LatentLeRobotDataset] dataset_path={self.root}")
        print(f"[LatentLeRobotDataset] image keys={image_keys}")
        print(f"[LatentLeRobotDataset] resolved obs camera keys={self.used_video_keys}")
        print(f"[LatentLeRobotDataset] episodes={len(self.episode_rows)}")
        print(f"[LatentLeRobotDataset] used_action_channel_ids={config.used_action_channel_ids}")
        print(
            "[LatentLeRobotDataset] q01/q99 selected="
            f"{self.q01[0, config.used_action_channel_ids].tolist()} / "
            f"{self.q99[0, config.used_action_channel_ids].tolist()}"
        )
        self.parse_meta()

    def _select_split(self, episodes):
        val_episode_ids = set(getattr(self.config, "val_episode_ids", []))
        if self.split == "train":
            return [row for row in episodes if row["episode_index"] not in val_episode_ids]
        if self.split == "val":
            return [row for row in episodes if row["episode_index"] in val_episode_ids]
        if self.split == "all":
            return episodes
        raise ValueError(f"Unknown dataset split: {self.split}")

    def parse_meta(self):
        out = []
        missing = []
        total_segments = 0
        for value in self.episode_rows:
            episode_index = value["episode_index"]
            tasks = value["tasks"]
            action_config = value.get("action_config", [])
            total_segments += len(action_config)
            for acfg in action_config:
                cur_meta = {
                    "episode_index": episode_index,
                    "tasks": tasks,
                }
                cur_meta.update(acfg)

                check_statu = self._check_meta(
                    cur_meta["start_frame"],
                    cur_meta["end_frame"],
                    cur_meta["episode_index"],
                )

                if check_statu:
                    out.append(cur_meta)
                elif len(missing) < 10:
                    missing.append(
                        {
                            "episode_index": episode_index,
                            "start_frame": cur_meta["start_frame"],
                            "end_frame": cur_meta["end_frame"],
                        }
                    )
        self.new_metas = out
        print(f"[LatentLeRobotDataset] action_config segments={total_segments}")
        print(f"[LatentLeRobotDataset] valid segments after latent-file check={len(self.new_metas)}")
        if missing:
            print(f"[LatentLeRobotDataset] missing latent examples={missing}")
        if len(self.new_metas) == 0:
            raise ValueError(
                f"No valid latent segments found in {self.root}. Likely causes: missing action_config "
                "in meta/episodes.jsonl, missing latent files under latents/chunk-xxx/<camera_key>/, "
                f"camera key mismatch (configured {self.used_video_keys}), wrong dataset_path, or wrong "
                "latent filename convention episode_000000_0_450.pth."
            )

    def _check_meta(self, start_frame, end_frame, episode_index):
        episode_chunk = episode_index // self.chunks_size
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            if not os.path.exists(latent_file):
                return False
        return True

    def _get_episode_actions(self, episode_index):
        if episode_index not in self._episode_action_cache:
            if "{episode_index" in self.data_path_template:
                episode_chunk = episode_index // self.chunks_size
                relative_path = self.data_path_template.format(
                    episode_chunk=episode_chunk,
                    episode_index=episode_index,
                )
                frame = pd.read_parquet(self.root / relative_path, columns=["action"])
                actions = np.stack(frame["action"].to_numpy()).astype(np.float32)
            else:
                if self._all_actions_by_episode is None:
                    paths = sorted((self.root / "data").glob("chunk-*/*.parquet"))
                    if not paths:
                        raise FileNotFoundError(f"No parquet files found under {self.root / 'data'}")
                    columns = ["episode_index", "frame_index", "action"]
                    data = pd.concat((pd.read_parquet(path, columns=columns) for path in paths), ignore_index=True)
                    data = data.sort_values(["episode_index", "frame_index"])
                    self._all_actions_by_episode = {
                        int(ep): np.stack(group["action"].to_numpy()).astype(np.float32)
                        for ep, group in data.groupby("episode_index", sort=True)
                    }
                actions = self._all_actions_by_episode[int(episode_index)]
            if not self._debug_sample_printed:
                print(f"[LatentLeRobotDataset] raw action shape episode {episode_index}: {actions.shape}")
            self._episode_action_cache[episode_index] = actions
        return self._episode_action_cache[episode_index]

    def _get_range_hf_data(self, start_frame, end_frame, episode_index):
        return {"action": self._get_episode_actions(episode_index)[start_frame:end_frame]}

    def _flatten_latent_dict(self, latent_dict):
        out = {}
        for key, value in latent_dict.items():
            for inner_key, inner_value in value.items():
                new_key = f"{key}.{inner_key}"
                out[new_key] = inner_value
        return out

    def _get_range_latent_data(self, start_frame, end_frame, episode_index):
        episode_chunk = episode_index // self.chunks_size
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        out = {}
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            assert os.path.exists(latent_file)
            latent_data = torch.load(latent_file, weights_only=False)
            out[key] = latent_data
        
        return self._flatten_latent_dict(out)
    
        
    def _cat_video_latents(self,
                           data_dict
                           ):
        latent_lst = []
        for key in self.used_video_keys:
            latent= data_dict[f"{key}.latent"]
            latent_num_frames = data_dict[f"{key}.latent_num_frames"]
            latent_height = data_dict[f"{key}.latent_height"]
            latent_width = data_dict[f"{key}.latent_width"]
            latent = rearrange(latent, 
                                 '(f h w) c -> f h w c', 
                                 f=latent_num_frames, 
                                 h=latent_height, 
                                 w=latent_width)
            latent_lst.append(latent)
        if self.config.env_type == 'robotwin_tshape':
            wrist_latent = torch.cat(latent_lst[1:], dim=2)
            cat_latent = torch.cat([wrist_latent, latent_lst[0]], dim=1)
        else:
            cat_latent = torch.cat(latent_lst, dim=2)

        text_emb = data_dict[f"{self.used_video_keys[0]}.text_emb"]
        if torch.rand(1).item() < self.cfg_prob:
            text_emb = self.empty_emb

        out_dict = dict(
            latents = cat_latent,
            text_emb = text_emb,
        )
        return out_dict
    
    def _action_post_process(self, local_start_frame, local_end_frame, latent_frame_ids, action):
        act_shift = int(latent_frame_ids[0] - local_start_frame)
        frame_stride = latent_frame_ids[1] - latent_frame_ids[0]
        action = action[act_shift:]
        if self.config.env_type == 'robotwin_tshape': ## TODO support get_relative_pose for other dataset, currently only support robotwin 
            left_action = get_relative_pose(action[:, :7])
            right_action = get_relative_pose(action[:, 8:15])
            action = np.concatenate([left_action, action[:, 7:8], right_action, action[:, 15:16]], axis=1)
        action_mask = np.ones_like(action, dtype='bool')
        if not self._debug_sample_printed:
            print(f"[LatentLeRobotDataset] mapped action source shape={action.shape}")
        action = np.pad(
            action,
            pad_width=((frame_stride * 4, 0), (0, 0)),
            mode='constant',
            constant_values=0,
        )
        action_mask = np.pad(
            action_mask,
            pad_width=((frame_stride * 4, 0), (0, 0)),
            mode='constant',
            constant_values=False,
        )

        latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
        required_action_num = latent_frame_num * frame_stride * 4

        action = action[:required_action_num]
        action_mask = action_mask[:required_action_num]
        assert action.shape[0] == required_action_num


        action_paded = np.pad(action, ((0, 0), (0, 1)), mode='constant', constant_values=0)
        action_mask_padded = np.pad(action_mask, ((0, 0), (0, 1)), mode='constant', constant_values=0)

        action_aligned = action_paded[:, self.config.inverse_used_action_channel_ids]
        action_mask_aligned = action_mask_padded[:, self.config.inverse_used_action_channel_ids]
        action_aligned = (action_aligned - self.q01) / (
                self.q99 - self.q01 + 1e-6) * 2. - 1.
        action_aligned = np.clip(action_aligned, -1.5, 1.5)
        action_aligned = rearrange(action_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_mask_aligned = rearrange(action_mask_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_aligned *= action_mask_aligned
        if not self._debug_sample_printed:
            print(f"[LatentLeRobotDataset] final action tensor shape={action_aligned.shape}")
        return torch.from_numpy(action_aligned).float(), torch.from_numpy(action_mask_aligned).bool()

    def _temporal_crop(self, out_dict):
        max_frames = getattr(self.config, "max_latent_frames", None)
        frame_count = out_dict["latents"].shape[0]
        if not max_frames or frame_count <= max_frames:
            return out_dict
        max_start = frame_count - max_frames
        if self.split == "train":
            start = int(torch.randint(0, max_start + 1, (1,)).item())
        else:
            start = max_start // 2
        end = start + max_frames
        out_dict["latents"] = out_dict["latents"][start:end]
        out_dict["actions"] = out_dict["actions"][:, start:end]
        out_dict["actions_mask"] = out_dict["actions_mask"][:, start:end]
        return out_dict

    def __getitem__(self, idx) -> dict:
        idx = idx % len(self.new_metas)
        cur_meta = self.new_metas[idx]
        episode_index = cur_meta["episode_index"]
        start_frame = cur_meta["start_frame"]
        end_frame = cur_meta["end_frame"]
        local_start_frame = start_frame
        local_end_frame = end_frame

        ori_data_dict = self._get_range_latent_data(start_frame, end_frame, episode_index)

        latent_frame_ids = ori_data_dict[f"{self.used_video_keys[0]}.frame_ids"]
        hf_data_frames = self._get_range_hf_data(start_frame, end_frame, episode_index)
        ori_data_dict.update(hf_data_frames)
        out_dict = self._cat_video_latents(ori_data_dict)
        if not self._debug_sample_printed:
            print(f"[LatentLeRobotDataset] latents shape before permute={out_dict['latents'].shape}")
            print(f"[LatentLeRobotDataset] text_emb shape={out_dict['text_emb'].shape}")

        out_dict['actions'], out_dict['actions_mask'] = self._action_post_process(local_start_frame, local_end_frame, latent_frame_ids, ori_data_dict['action'])

        out_dict = self._temporal_crop(out_dict)
        out_dict['latents'] = out_dict['latents'].permute(3, 0, 1, 2)
        self._debug_sample_printed = True
        return out_dict

    def __len__(self):
        return len(self.new_metas)

if __name__ == '__main__':
    from wan_va.configs import VA_CONFIGS
    from tqdm import tqdm
    dset = MultiLatentLeRobotDataset(
        VA_CONFIGS['demo_train']
    )
    for key, value in dset[0].items():
        if isinstance(value, torch.Tensor):
            print(f'{key}: {value.shape} tensor')
        elif isinstance(value, np.ndarray):
            print(f'{key}: {value.shape} np')
        else:
            print(f'{key}: {value}')
    print(len(dset))
    dloader = DataLoader(
            dset,
            batch_size=1,
            shuffle=True,
            num_workers=32,
        )
    max_l = 0
    action_list = []
    for data in tqdm(dloader):
        _, _, F, H, W = data['latents'].shape
        max_l = max(max_l, F*H*W)
        action_list.append(data['actions'].flatten(2).permute(0, 2, 1).flatten(0, 1))
    action_all = torch.cat(action_list, dim=0)
    print(max_l)
    print(action_all.shape, action_all.mean(dim=0), action_all.min(dim=0)[0], action_all.max(dim=0)[0])
    
