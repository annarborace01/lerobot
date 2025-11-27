#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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

import logging
import shutil
from pathlib import Path

import pandas as pd
import tqdm

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_PATH,
    get_file_size_in_mb,
    get_parquet_file_size_in_mb,
    to_parquet_with_hf_images,
    update_chunk_file_indices,
    write_info,
    write_stats,
    write_tasks,
)
from lerobot.datasets.video_utils import concatenate_video_files, get_video_duration_in_s


def validate_all_metadata(all_metadata: list[LeRobotDatasetMetadata]):
    """Validates that all dataset metadata have consistent properties.

    Ensures all datasets have the same fps, robot_type, and features to guarantee
    compatibility when aggregating them into a single dataset.

    Args:
        all_metadata: List of LeRobotDatasetMetadata objects to validate.

    Returns:
        tuple: A tuple containing (fps, robot_type, features) from the first metadata.

    Raises:
        ValueError: If any metadata has different fps, robot_type, or features
                   than the first metadata in the list.
    """

    fps = all_metadata[0].fps
    robot_type = all_metadata[0].robot_type
    features = all_metadata[0].features

    for meta in tqdm.tqdm(all_metadata, desc="Validate all meta data"):
        if fps != meta.fps:
            raise ValueError(f"Same fps is expected, but got fps={meta.fps} instead of {fps}.")
        if robot_type != meta.robot_type:
            raise ValueError(
                f"Same robot_type is expected, but got robot_type={meta.robot_type} instead of {robot_type}."
            )
        if features != meta.features:
            raise ValueError(
                f"Same features is expected, but got features={meta.features} instead of {features}."
            )

    return fps, robot_type, features


def update_data_df(df, src_meta, dst_meta):
    """Updates a data DataFrame with new indices and task mappings for aggregation.

    Adjusts episode indices, frame indices, and task indices to account for
    previously aggregated data in the destination dataset.

    Args:
        df: DataFrame containing the data to be updated.
        src_meta: Source dataset metadata.
        dst_meta: Destination dataset metadata.

    Returns:
        pd.DataFrame: Updated DataFrame with adjusted indices.
    """

    df["episode_index"] = df["episode_index"] + dst_meta.info["total_episodes"]
    df["index"] = df["index"] + dst_meta.info["total_frames"]

    src_task_names = src_meta.tasks.index.take(df["task_index"].to_numpy())
    df["task_index"] = dst_meta.tasks.loc[src_task_names, "task_index"].to_numpy()

    return df


def update_meta_data(
    df,
    dst_meta,
    meta_idx,
    data_idx,
    videos_idx,
    episode_to_file_mapping=None,
    video_src_to_dst_mapping=None,
):
    """Updates metadata DataFrame with new chunk, file, and timestamp indices.

    Adjusts all indices and timestamps to account for previously aggregated
    data and videos in the destination dataset.

    Args:
        df: DataFrame containing the metadata to be updated.
        dst_meta: Destination dataset metadata.
        meta_idx: Dictionary containing current metadata chunk and file indices.
        data_idx: Dictionary containing current data chunk and file indices.
        videos_idx: Dictionary containing current video indices and timestamps.
        episode_to_file_mapping: Optional dict mapping episode indices to (chunk, file) tuples.
        video_src_to_dst_mapping: Optional dict mapping (key, src_chunk, src_file) to (dst_chunk, dst_file).

    Returns:
        pd.DataFrame: Updated DataFrame with adjusted indices and timestamps.
    """

    df["meta/episodes/chunk_index"] = df["meta/episodes/chunk_index"] + meta_idx["chunk"]
    df["meta/episodes/file_index"] = df["meta/episodes/file_index"] + meta_idx["file"]
    
    # Update data indices: use episode mapping if available, otherwise use old logic
    if episode_to_file_mapping is not None:
        # Update each episode's data file location based on actual destination
        # Need to compute the adjusted episode index for lookup
        episode_offset = dst_meta.info["total_episodes"]
        for idx in df.index:
            ep_idx = df.at[idx, "episode_index"] + episode_offset
            if ep_idx in episode_to_file_mapping:
                chunk, file = episode_to_file_mapping[ep_idx]
                df.at[idx, "data/chunk_index"] = chunk
                df.at[idx, "data/file_index"] = file
            else:
                # Fallback to old logic if episode not found
                df.at[idx, "data/chunk_index"] = df.at[idx, "data/chunk_index"] + data_idx["chunk"]
                df.at[idx, "data/file_index"] = df.at[idx, "data/file_index"] + data_idx["file"]
    else:
        # Old logic for backward compatibility
        df["data/chunk_index"] = df["data/chunk_index"] + data_idx["chunk"]
        df["data/file_index"] = df["data/file_index"] + data_idx["file"]
    for key, video_idx in videos_idx.items():
        # Store original video file indices before updating
        orig_chunk_col = f"videos/{key}/chunk_index"
        orig_file_col = f"videos/{key}/file_index"
        df["_orig_chunk"] = df[orig_chunk_col].copy()
        df["_orig_file"] = df[orig_file_col].copy()

        # Update chunk and file indices using video_src_to_dst_mapping if available
        if video_src_to_dst_mapping is not None:
            for idx in df.index:
                src_chunk = df.at[idx, "_orig_chunk"]
                src_file = df.at[idx, "_orig_file"]
                mapping_key = (key, src_chunk, src_file)
                
                if mapping_key in video_src_to_dst_mapping:
                    dst_chunk, dst_file = video_src_to_dst_mapping[mapping_key]
                    df.at[idx, orig_chunk_col] = dst_chunk
                    df.at[idx, orig_file_col] = dst_file
                else:
                    # Fallback if mapping not found
                    df.at[idx, orig_chunk_col] = video_idx["chunk"]
                    df.at[idx, orig_file_col] = video_idx["file"]
        else:
            # Old logic: set all to same destination file
            df[orig_chunk_col] = video_idx["chunk"]
            df[orig_file_col] = video_idx["file"]

        # Apply per-source-file timestamp offsets
        src_to_offset = video_idx.get("src_to_offset", {})
        if src_to_offset:
            # Apply offset based on original source file
            for idx in df.index:
                src_key = (df.at[idx, "_orig_chunk"], df.at[idx, "_orig_file"])
                offset = src_to_offset.get(src_key, 0)
                df.at[idx, f"videos/{key}/from_timestamp"] += offset
                df.at[idx, f"videos/{key}/to_timestamp"] += offset
        else:
            # Fallback to simple offset (for backward compatibility)
            df[f"videos/{key}/from_timestamp"] = (
                df[f"videos/{key}/from_timestamp"] + video_idx["latest_duration"]
            )
            df[f"videos/{key}/to_timestamp"] = df[f"videos/{key}/to_timestamp"] + video_idx["latest_duration"]

        # Clean up temporary columns
        df = df.drop(columns=["_orig_chunk", "_orig_file"])

    df["dataset_from_index"] = df["dataset_from_index"] + dst_meta.info["total_frames"]
    df["dataset_to_index"] = df["dataset_to_index"] + dst_meta.info["total_frames"]
    df["episode_index"] = df["episode_index"] + dst_meta.info["total_episodes"]

    return df


def fix_metadata_episode_references(aggr_root: Path, episode_meta_mapping: dict):
    """Fixes meta/episodes/chunk_index and file_index references after aggregation.
    
    This function updates ONLY the meta/episodes columns without touching video timestamps
    or other metadata that was carefully calculated during aggregation.
    
    Args:
        aggr_root: Root path of the aggregated dataset
        episode_meta_mapping: Dict mapping episode_index to (chunk, file) tuples
    """
    dst_meta_dir = aggr_root / "meta" / "episodes"
    
    for chunk_dir in sorted(dst_meta_dir.glob("chunk-*")):
        for meta_file in sorted(chunk_dir.glob("file-*.parquet")):
            df = pd.read_parquet(meta_file)
            
            # Update ONLY meta/episodes indices, leave everything else untouched
            for idx in df.index:
                ep_idx = df.at[idx, "episode_index"]
                if ep_idx in episode_meta_mapping:
                    chunk, file = episode_meta_mapping[ep_idx]
                    df.at[idx, "meta/episodes/chunk_index"] = chunk
                    df.at[idx, "meta/episodes/file_index"] = file
            
            # Write back with only meta/episodes columns updated
            df.to_parquet(meta_file)


def aggregate_datasets(
    repo_ids: list[str],
    aggr_repo_id: str,
    roots: list[Path] | None = None,
    aggr_root: Path | None = None,
    data_files_size_in_mb: float | None = None,
    video_files_size_in_mb: float | None = None,
    chunk_size: int | None = None,
):
    """Aggregates multiple LeRobot datasets into a single unified dataset.

    This is the main function that orchestrates the aggregation process by:
    1. Loading and validating all source dataset metadata
    2. Creating a new destination dataset with unified tasks
    3. Aggregating videos, data, and metadata from all source datasets
    4. Finalizing the aggregated dataset with proper statistics

    Args:
        repo_ids: List of repository IDs for the datasets to aggregate.
        aggr_repo_id: Repository ID for the aggregated output dataset.
        roots: Optional list of root paths for the source datasets.
        aggr_root: Optional root path for the aggregated dataset.
        data_files_size_in_mb: Maximum size for data files in MB (defaults to DEFAULT_DATA_FILE_SIZE_IN_MB)
        video_files_size_in_mb: Maximum size for video files in MB (defaults to DEFAULT_VIDEO_FILE_SIZE_IN_MB)
        chunk_size: Maximum number of files per chunk (defaults to DEFAULT_CHUNK_SIZE)
    """
    logging.info("Start aggregate_datasets")

    if data_files_size_in_mb is None:
        data_files_size_in_mb = DEFAULT_DATA_FILE_SIZE_IN_MB
    if video_files_size_in_mb is None:
        video_files_size_in_mb = DEFAULT_VIDEO_FILE_SIZE_IN_MB
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE

    all_metadata = (
        [LeRobotDatasetMetadata(repo_id) for repo_id in repo_ids]
        if roots is None
        else [
            LeRobotDatasetMetadata(repo_id, root=root) for repo_id, root in zip(repo_ids, roots, strict=False)
        ]
    )
    fps, robot_type, features = validate_all_metadata(all_metadata)
    video_keys = [key for key in features if features[key]["dtype"] == "video"]

    dst_meta = LeRobotDatasetMetadata.create(
        repo_id=aggr_repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        root=aggr_root,
        use_videos=len(video_keys) > 0,
        chunks_size=chunk_size,
        data_files_size_in_mb=data_files_size_in_mb,
        video_files_size_in_mb=video_files_size_in_mb,
    )

    logging.info("Find all tasks")
    unique_tasks = pd.concat([m.tasks for m in all_metadata]).index.unique()
    dst_meta.tasks = pd.DataFrame({"task_index": range(len(unique_tasks))}, index=unique_tasks)

    meta_idx = {"chunk": 0, "file": 0}
    data_idx = {"chunk": 0, "file": 0}
    videos_idx = {
        key: {"chunk": 0, "file": 0, "latest_duration": 0, "episode_duration": 0} for key in video_keys
    }

    dst_meta.episodes = {}
    
    # Collect all episode-to-metadata-file mappings
    all_episode_meta_mappings = {}

    for src_meta in tqdm.tqdm(all_metadata, desc="Copy data and videos"):
        videos_idx, video_src_to_dst_mapping = aggregate_videos(src_meta, dst_meta, videos_idx, video_files_size_in_mb, chunk_size)
        data_idx, episode_to_file_mapping = aggregate_data(src_meta, dst_meta, data_idx, data_files_size_in_mb, chunk_size)

        meta_idx, episode_meta_mapping = aggregate_metadata(src_meta, dst_meta, meta_idx, data_idx, videos_idx, episode_to_file_mapping, video_src_to_dst_mapping)
        
        # Collect the mappings
        all_episode_meta_mappings.update(episode_meta_mapping)

        dst_meta.info["total_episodes"] += src_meta.total_episodes
        dst_meta.info["total_frames"] += src_meta.total_frames

    # Fix meta/episodes references AFTER all video timestamps have been set
    fix_metadata_episode_references(dst_meta.root, all_episode_meta_mappings)
    
    finalize_aggregation(dst_meta, all_metadata)
    logging.info("Aggregation complete.")


def aggregate_videos(src_meta, dst_meta, videos_idx, video_files_size_in_mb, chunk_size):
    """Aggregates video chunks from a source dataset into the destination dataset.

    Handles video file concatenation and rotation based on file size limits.
    Creates new video files when size limits are exceeded.

    Args:
        src_meta: Source dataset metadata.
        dst_meta: Destination dataset metadata.
        videos_idx: Dictionary tracking video chunk and file indices.
        video_files_size_in_mb: Maximum size for video files in MB (defaults to DEFAULT_VIDEO_FILE_SIZE_IN_MB)
        chunk_size: Maximum number of files per chunk (defaults to DEFAULT_CHUNK_SIZE)

    Returns:
        tuple: (Updated videos_idx dict, video_src_to_dst_mapping dict)
            video_src_to_dst_mapping maps (video_key, src_chunk, src_file) to (dst_chunk, dst_file)
    """
    for key in videos_idx:
        videos_idx[key]["episode_duration"] = 0
        # Track offset for each source (chunk, file) pair
        videos_idx[key]["src_to_offset"] = {}
    
    # Track which destination file each source video file maps to
    video_src_to_dst_mapping = {}

    for key, video_idx in videos_idx.items():
        unique_chunk_file_pairs = {
            (chunk, file)
            for chunk, file in zip(
                src_meta.episodes[f"videos/{key}/chunk_index"],
                src_meta.episodes[f"videos/{key}/file_index"],
                strict=False,
            )
        }
        unique_chunk_file_pairs = sorted(unique_chunk_file_pairs)

        chunk_idx = video_idx["chunk"]
        file_idx = video_idx["file"]
        current_offset = video_idx["latest_duration"]

        for src_chunk_idx, src_file_idx in unique_chunk_file_pairs:
            src_path = src_meta.root / DEFAULT_VIDEO_PATH.format(
                video_key=key,
                chunk_index=src_chunk_idx,
                file_index=src_file_idx,
            )

            dst_path = dst_meta.root / DEFAULT_VIDEO_PATH.format(
                video_key=key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )

            src_duration = get_video_duration_in_s(src_path)

            if not dst_path.exists():
                # Store offset and mapping
                videos_idx[key]["src_to_offset"][(src_chunk_idx, src_file_idx)] = current_offset
                video_src_to_dst_mapping[(key, src_chunk_idx, src_file_idx)] = (chunk_idx, file_idx)
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(src_path), str(dst_path))
                videos_idx[key]["episode_duration"] += src_duration
                current_offset += src_duration
                continue

            # Check file sizes before appending
            src_size = get_file_size_in_mb(src_path)
            dst_size = get_file_size_in_mb(dst_path)

            if dst_size + src_size >= video_files_size_in_mb:
                # Rotate to a new file, this source becomes start of new destination
                # So its offset should be 0
                videos_idx[key]["src_to_offset"][(src_chunk_idx, src_file_idx)] = 0
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, chunk_size)
                video_src_to_dst_mapping[(key, src_chunk_idx, src_file_idx)] = (chunk_idx, file_idx)
                dst_path = dst_meta.root / DEFAULT_VIDEO_PATH.format(
                    video_key=key,
                    chunk_index=chunk_idx,
                    file_index=file_idx,
                )
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(src_path), str(dst_path))
                # Reset offset for next file
                current_offset = src_duration
            else:
                # Append to existing video file - use current accumulated offset
                videos_idx[key]["src_to_offset"][(src_chunk_idx, src_file_idx)] = current_offset
                video_src_to_dst_mapping[(key, src_chunk_idx, src_file_idx)] = (chunk_idx, file_idx)
                concatenate_video_files(
                    [dst_path, src_path],
                    dst_path,
                )
                current_offset += src_duration

            videos_idx[key]["episode_duration"] += src_duration

        videos_idx[key]["chunk"] = chunk_idx
        videos_idx[key]["file"] = file_idx

    return videos_idx, video_src_to_dst_mapping


def aggregate_data(src_meta, dst_meta, data_idx, data_files_size_in_mb, chunk_size):
    """Aggregates data chunks from a source dataset into the destination dataset.

    Reads source data files, updates indices to match the aggregated dataset,
    and writes them to the destination with proper file rotation.

    Args:
        src_meta: Source dataset metadata.
        dst_meta: Destination dataset metadata.
        data_idx: Dictionary tracking data chunk and file indices.

    Returns:
        tuple: (Updated data_idx dict, episode_to_file_mapping dict)
            episode_to_file_mapping maps episode_index to (chunk_idx, file_idx)
    """
    unique_chunk_file_ids = {
        (c, f)
        for c, f in zip(
            src_meta.episodes["data/chunk_index"], src_meta.episodes["data/file_index"], strict=False
        )
    }

    unique_chunk_file_ids = sorted(unique_chunk_file_ids)
    
    # Track which episodes end up in which destination files
    episode_to_file_mapping = {}

    for src_chunk_idx, src_file_idx in unique_chunk_file_ids:
        src_path = src_meta.root / DEFAULT_DATA_PATH.format(
            chunk_index=src_chunk_idx, file_index=src_file_idx
        )
        df = pd.read_parquet(src_path)
        
        # Get episode indices before updating
        episode_indices = df["episode_index"].unique()
        
        df = update_data_df(df, src_meta, dst_meta)
        
        # Record current destination file indices before potential rotation
        current_chunk = data_idx["chunk"]
        current_file = data_idx["file"]

        data_idx = append_or_create_parquet_file(
            df,
            src_path,
            data_idx,
            data_files_size_in_mb,
            chunk_size,
            DEFAULT_DATA_PATH,
            contains_images=len(dst_meta.image_keys) > 0,
            aggr_root=dst_meta.root,
        )
        
        # Map these episodes to their destination file
        # Use the current indices, as append_or_create may have rotated to next file
        dest_chunk = data_idx["chunk"]
        dest_file = data_idx["file"]
        
        for ep_idx in episode_indices:
            # Episodes go to current file, or if rotated, they're in the current file
            episode_to_file_mapping[ep_idx + dst_meta.info["total_episodes"]] = (dest_chunk, dest_file)

    return data_idx, episode_to_file_mapping


def aggregate_metadata(src_meta, dst_meta, meta_idx, data_idx, videos_idx, episode_to_file_mapping=None, video_src_to_dst_mapping=None):
    """Aggregates metadata from a source dataset into the destination dataset.

    Reads source metadata files, updates all indices and timestamps,
    and writes them to the destination with proper file rotation.

    Args:
        src_meta: Source dataset metadata.
        dst_meta: Destination dataset metadata.
        meta_idx: Dictionary tracking metadata chunk and file indices.
        data_idx: Dictionary tracking data chunk and file indices.
        videos_idx: Dictionary tracking video indices and timestamps.
        episode_to_file_mapping: Optional dict mapping episode indices to (chunk, file) tuples.
        video_src_to_dst_mapping: Optional dict mapping (key, src_chunk, src_file) to (dst_chunk, dst_file).

    Returns:
        tuple: (Updated meta_idx, episode_meta_mapping)
    """
    # Instead of trusting the metadata columns which may be corrupted,
    # scan for actual metadata files that exist on disk
    meta_episodes_dir = src_meta.root / "meta" / "episodes"
    actual_meta_files = []
    
    for chunk_dir in sorted(meta_episodes_dir.glob("chunk-*")):
        chunk_idx = int(chunk_dir.name.split("-")[1])
        for meta_file in sorted(chunk_dir.glob("file-*.parquet")):
            file_idx = int(meta_file.stem.split("-")[1])
            actual_meta_files.append((chunk_idx, file_idx))
    
    # Track which episodes' metadata ends up in which destination files
    episode_meta_mapping = {}
    
    for chunk_idx, file_idx in actual_meta_files:
        src_path = src_meta.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        df = pd.read_parquet(src_path)
        
        # Get episode indices before updating
        episode_indices = df["episode_index"].values
        
        df = update_meta_data(
            df,
            dst_meta,
            meta_idx,
            data_idx,
            videos_idx,
            episode_to_file_mapping,
            video_src_to_dst_mapping,
        )
        
        # Record current destination indices before potential rotation
        current_meta_chunk = meta_idx["chunk"]
        current_meta_file = meta_idx["file"]

        meta_idx = append_or_create_parquet_file(
            df,
            src_path,
            meta_idx,
            DEFAULT_DATA_FILE_SIZE_IN_MB,
            DEFAULT_CHUNK_SIZE,
            DEFAULT_EPISODES_PATH,
            contains_images=False,
            aggr_root=dst_meta.root,
        )
        
        # Map these episodes' metadata to their destination file
        dest_meta_chunk = meta_idx["chunk"]
        dest_meta_file = meta_idx["file"]
        
        for ep_idx in episode_indices:
            # Episodes go to current file, accounting for episode offset
            adjusted_ep_idx = ep_idx + dst_meta.info["total_episodes"]
            episode_meta_mapping[adjusted_ep_idx] = (dest_meta_chunk, dest_meta_file)
    
    # Increment latest_duration by the total duration added from this source dataset
    for k in videos_idx:
        videos_idx[k]["latest_duration"] += videos_idx[k]["episode_duration"]

    return meta_idx, episode_meta_mapping


def append_or_create_parquet_file(
    df: pd.DataFrame,
    src_path: Path,
    idx: dict[str, int],
    max_mb: float,
    chunk_size: int,
    default_path: str,
    contains_images: bool = False,
    aggr_root: Path = None,
):
    """Appends data to an existing parquet file or creates a new one based on size constraints.

    Manages file rotation when size limits are exceeded to prevent individual files
    from becoming too large. Handles both regular parquet files and those containing images.

    Args:
        df: DataFrame to write to the parquet file.
        src_path: Path to the source file (used for size estimation).
        idx: Dictionary containing current 'chunk' and 'file' indices.
        max_mb: Maximum allowed file size in MB before rotation.
        chunk_size: Maximum number of files per chunk before incrementing chunk index.
        default_path: Format string for generating file paths.
        contains_images: Whether the data contains images requiring special handling.
        aggr_root: Root path for the aggregated dataset.

    Returns:
        dict: Updated index dictionary with current chunk and file indices.
    """
    dst_path = aggr_root / default_path.format(chunk_index=idx["chunk"], file_index=idx["file"])

    if not dst_path.exists():
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if contains_images:
            to_parquet_with_hf_images(df, dst_path)
        else:
            df.to_parquet(dst_path)
        return idx

    src_size = get_parquet_file_size_in_mb(src_path)
    dst_size = get_parquet_file_size_in_mb(dst_path)

    if dst_size + src_size >= max_mb:
        idx["chunk"], idx["file"] = update_chunk_file_indices(idx["chunk"], idx["file"], chunk_size)
        new_path = aggr_root / default_path.format(chunk_index=idx["chunk"], file_index=idx["file"])
        new_path.parent.mkdir(parents=True, exist_ok=True)
        final_df = df
        target_path = new_path
    else:
        existing_df = pd.read_parquet(dst_path)
        final_df = pd.concat([existing_df, df], ignore_index=True)
        target_path = dst_path

    if contains_images:
        to_parquet_with_hf_images(final_df, target_path)
    else:
        final_df.to_parquet(target_path)

    return idx


def finalize_aggregation(aggr_meta, all_metadata):
    """Finalizes the dataset aggregation by writing summary files and statistics.

    Writes the tasks file, info file with total counts and splits, and
    aggregated statistics from all source datasets.

    Args:
        aggr_meta: Aggregated dataset metadata.
        all_metadata: List of all source dataset metadata objects.
    """
    logging.info("write tasks")
    write_tasks(aggr_meta.tasks, aggr_meta.root)

    logging.info("write info")
    aggr_meta.info.update(
        {
            "total_tasks": len(aggr_meta.tasks),
            "total_episodes": sum(m.total_episodes for m in all_metadata),
            "total_frames": sum(m.total_frames for m in all_metadata),
            "splits": {"train": f"0:{sum(m.total_episodes for m in all_metadata)}"},
        }
    )
    write_info(aggr_meta.info, aggr_meta.root)

    logging.info("write stats")
    aggr_meta.stats = aggregate_stats([m.stats for m in all_metadata])
    write_stats(aggr_meta.stats, aggr_meta.root)
