#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    EPISODES_PATH,
    EPISODES_STATS_PATH,
    INFO_PATH,
    STATS_PATH,
    TASKS_PATH,
    append_jsonlines,
    write_info,
    write_json,
)


def convert_stats_to_lists(stats):
    """Recursively converts numpy arrays in a stats dictionary to lists."""
    if isinstance(stats, dict):
        return {k: convert_stats_to_lists(v) for k, v in stats.items()}
    if isinstance(stats, np.ndarray):
        return stats.tolist()
    return stats


def merge_datasets(repo_id1, repo_id2, new_repo_id):
    """
    Merges two LeRobotDatasets and uploads the result to the Hugging Face Hub.
    """
    print(f"Merging '{repo_id1}' and '{repo_id2}' into '{new_repo_id}'.")

    # 1. Load metadata of source datasets
    print("Loading metadata from source datasets...")
    meta1 = LeRobotDatasetMetadata(repo_id1, force_cache_sync=True)
    meta2 = LeRobotDatasetMetadata(repo_id2, force_cache_sync=True)
    print("Metadata loaded.")

    # 2. Compatibility checks
    if meta1.fps != meta2.fps:
        raise ValueError(f"FPS differs between datasets: {meta1.fps} vs {meta2.fps}")
    if meta1.robot_type != meta2.robot_type:
        print(
            f"Warning: Robot type differs: '{meta1.robot_type}' vs '{meta2.robot_type}'. "
            f"Using '{meta1.robot_type}'."
        )
    if meta1.features != meta2.features:
        raise ValueError("Features of the two datasets are not identical.")

    # 3. Setup new dataset directory
    new_ds_path = HF_LEROBOT_HOME / new_repo_id
    if new_ds_path.exists():
        print(f"Removing existing directory: {new_ds_path}")
        shutil.rmtree(new_ds_path)
    new_ds_path.mkdir(parents=True)
    (new_ds_path / "meta").mkdir(exist_ok=True)
    print(f"Created new dataset directory: {new_ds_path}")

    # 4. Merge metadata
    # Merge info
    new_info = meta1.info.copy()
    new_info["total_episodes"] = meta1.total_episodes + meta2.total_episodes
    new_info["total_frames"] = meta1.total_frames + meta2.total_frames

    # Merge tasks
    new_tasks = meta1.tasks.copy()  # dict[task_idx, task_str]
    new_task_to_task_index = meta1.task_to_task_index.copy()

    next_task_idx = meta1.total_tasks
    for task_str in meta2.tasks.values():
        if task_str not in new_task_to_task_index:
            new_tasks[next_task_idx] = task_str
            new_task_to_task_index[task_str] = next_task_idx
            next_task_idx += 1

    new_info["total_tasks"] = len(new_tasks)

    new_episodes = {}
    new_episode_stats = {}

    # 5. Process and copy data
    # Process dataset 1
    print(f"Processing dataset 1: {repo_id1}")
    for old_ep_idx in tqdm(range(meta1.total_episodes)):
        new_ep_idx = old_ep_idx

        # Update episode entry
        ep_info = meta1.episodes[old_ep_idx].copy()
        ep_info["episode_index"] = new_ep_idx
        new_episodes[new_ep_idx] = ep_info

        # Update episode stats entry, normalizing the structure
        ep_stat_from_meta = meta1.episodes_stats[old_ep_idx].copy()
        if "stats" not in ep_stat_from_meta:
            # It's a raw stats dict (v2.1+), wrap it
            ep_stat = {"stats": ep_stat_from_meta}
        else:
            # It already has a 'stats' key (pre-v2.1)
            ep_stat = ep_stat_from_meta
        ep_stat["episode_index"] = new_ep_idx
        new_episode_stats[new_ep_idx] = ep_stat

        # Copy parquet file
        old_parquet_path = meta1.root / meta1.get_data_file_path(old_ep_idx)
        new_chunk = new_ep_idx // new_info["chunks_size"]
        new_data_dir = (
            new_ds_path / Path(new_info["data_path"].format(episode_chunk=new_chunk, episode_index=0)).parent
        )
        new_data_dir.mkdir(parents=True, exist_ok=True)
        new_parquet_path = new_data_dir / f"episode_{new_ep_idx:06d}.parquet"
        shutil.copy(old_parquet_path, new_parquet_path)

        # Copy video files
        for vid_key in meta1.video_keys:
            old_video_path = meta1.root / meta1.get_video_file_path(old_ep_idx, vid_key)
            new_video_dir = (
                new_ds_path
                / Path(
                    new_info["video_path"].format(
                        episode_chunk=new_chunk, video_key=vid_key, episode_index=0
                    )
                ).parent
            )
            new_video_dir.mkdir(parents=True, exist_ok=True)
            new_video_path = new_video_dir / f"episode_{new_ep_idx:06d}.mp4"
            if old_video_path.exists():
                shutil.copy(old_video_path, new_video_path)

    # Process dataset 2
    print(f"Processing dataset 2: {repo_id2}")
    for i in tqdm(range(meta2.total_episodes)):
        old_ep_idx = i
        new_ep_idx = meta1.total_episodes + i

        # Update episode entry
        ep_info = meta2.episodes[old_ep_idx].copy()
        ep_info["episode_index"] = new_ep_idx
        new_episodes[new_ep_idx] = ep_info

        # Update episode stats entry, normalizing the structure
        ep_stat_from_meta = meta2.episodes_stats[old_ep_idx].copy()
        if "stats" not in ep_stat_from_meta:
            # It's a raw stats dict (v2.1+), wrap it
            ep_stat = {"stats": ep_stat_from_meta}
        else:
            # It already has a 'stats' key (pre-v2.1)
            ep_stat = ep_stat_from_meta
        ep_stat["episode_index"] = new_ep_idx
        new_episode_stats[new_ep_idx] = ep_stat

        # Modify and copy parquet file
        old_parquet_path = meta2.root / meta2.get_data_file_path(old_ep_idx)
        df = pd.read_parquet(old_parquet_path)

        df["episode_index"] = new_ep_idx

        task_remapping = {
            old_task_idx: new_task_to_task_index[task_str]
            for old_task_idx, task_str in meta2.tasks.items()
        }
        df["task_index"] = df["task_index"].map(task_remapping)

        new_chunk = new_ep_idx // new_info["chunks_size"]
        new_data_dir = (
            new_ds_path / Path(new_info["data_path"].format(episode_chunk=new_chunk, episode_index=0)).parent
        )
        new_data_dir.mkdir(parents=True, exist_ok=True)
        new_parquet_path = new_data_dir / f"episode_{new_ep_idx:06d}.parquet"
        df.to_parquet(new_parquet_path)

        # Copy video files
        for vid_key in meta2.video_keys:
            old_video_path = meta2.root / meta2.get_video_file_path(old_ep_idx, vid_key)
            new_video_dir = (
                new_ds_path
                / Path(
                    new_info["video_path"].format(
                        episode_chunk=new_chunk, video_key=vid_key, episode_index=0
                    )
                ).parent
            )
            new_video_dir.mkdir(parents=True, exist_ok=True)
            new_video_path = new_video_dir / f"episode_{new_ep_idx:06d}.mp4"
            if old_video_path.exists():
                shutil.copy(old_video_path, new_video_path)

    # 6. Finalize and write metadata
    print("Finalizing metadata...")
    from lerobot.datasets.compute_stats import aggregate_stats

    # Aggregate stats
    new_stats = aggregate_stats([ep_stat["stats"] for ep_stat in new_episode_stats.values()])

    # Finalize info
    new_info["total_chunks"] = (
        new_info["total_episodes"] + new_info["chunks_size"] - 1
    ) // new_info["chunks_size"]
    new_info["splits"] = {"train": f"0:{new_info['total_episodes']}"}
    if meta1.video_keys or meta2.video_keys:
        total_videos1 = meta1.info.get("total_videos", meta1.total_episodes * len(meta1.video_keys))
        total_videos2 = meta2.info.get("total_videos", meta2.total_episodes * len(meta2.video_keys))
        new_info["total_videos"] = total_videos1 + total_videos2

    # Write metadata files
    write_info(new_info, new_ds_path)

    new_stats_serializable = convert_stats_to_lists(new_stats)
    write_json(new_stats_serializable, new_ds_path / STATS_PATH)

    for ep_stat in new_episode_stats.values():
        ep_stat_serializable = convert_stats_to_lists(ep_stat)
        append_jsonlines(ep_stat_serializable, new_ds_path / EPISODES_STATS_PATH)

    for task_idx, task_str in new_tasks.items():
        append_jsonlines({"task_index": task_idx, "task": task_str}, new_ds_path / TASKS_PATH)

    for ep_info in new_episodes.values():
        append_jsonlines(ep_info, new_ds_path / EPISODES_PATH)
    print("Metadata written.")

    # 7. Load and push to hub
    print("Loading new dataset and pushing to hub...")
    new_dataset = LeRobotDataset(repo_id=new_repo_id, root=new_ds_path)
    new_dataset.push_to_hub()
    print("Done!")


if __name__ == "__main__":
    repo_id1 = "Thytu/so101-object-in-box_v0.1"
    repo_id2 = "Thytu/so101-object-in-box_v0.1_multi-shot"
    new_repo_id = "Thytu/so101-object-in-box_v0.2-b"
    merge_datasets(repo_id1, repo_id2, new_repo_id)
