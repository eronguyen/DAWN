from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from fvcore.common.timer import Timer
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
        self,
        data_path,
        split="training",
        image_size=256,
        num_frames=2,
        num_actions=10,
        min_skip=10,
        max_skip=30,
        observation_type: List[str] = ["rgb_static", "rgb_gripper"],
        observation_from: Optional[List[str]] = None,
        action_type: str = "actions",
        data_percent: float = 100.0,
        data_subset_seed: int = 42,
        **kwargs,
    ):
        timer = Timer()

        # Allow subclasses to predefine self.data_path before super().__init__.
        self.data_path = getattr(self, "data_path", data_path)
        self.split = split
        self.image_size = image_size
        self.num_frames = num_frames
        self.num_actions = num_actions
        self.min_skip = min_skip
        self.max_skip = max_skip
        self.action_type = action_type
        self.data_percent = float(data_percent)
        self.data_subset_seed = int(data_subset_seed)

        self.observation_type = list(observation_type)
        self.observation_from = list(observation_from) if observation_from is not None else list(observation_type)
        if len(self.observation_type) != len(self.observation_from):
            raise ValueError(
                "observation_type and observation_from must have the same length. "
                f"Got {len(self.observation_type)} vs {len(self.observation_from)}."
            )

        if not os.path.exists(self.data_path):
            raise ValueError(f"Data path {self.data_path} does not exist.")

        self.episodes = self._load_episodes()
        self.transform = self._get_transform()

        logger.info(f"Dataset initialized with image_size={image_size}, num_frames={num_frames}, num_actions={num_actions}, min_skip={min_skip}, max_skip={max_skip}, observation_type={observation_type}, action_type={action_type}.")
        logger.info("Loading dataset from %s took %.2f seconds.", self.data_path, timer.seconds())

    def _get_transform(self):
        if self.split == "training":
            return A.Compose(
                [
                    A.Resize(self.image_size, self.image_size),
                    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
                    ToTensorV2(),
                ]
            )
        return A.Compose([A.Resize(self.image_size, self.image_size), ToTensorV2()])

    def _build_episode_metadata(self, episode_path: str, obs_files: Dict[str, List[str]]) -> Dict[str, Any]:
        metadata_path = os.path.join(episode_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        primary_obs = self.observation_from[0]
        frame_names = [os.path.basename(x) for x in obs_files.get(primary_obs, [])]
        if "frames" not in metadata:
            metadata["frames"] = frame_names
        metadata["length"] = len(metadata["frames"])
        return metadata

    def _load_single_episode(self, episode_file: str) -> Dict[str, Any]:
        episode_path = os.path.join(self.data_path, episode_file)
        obs_files = {}
        for obs_from in set(self.observation_from):
            obs_dir = os.path.join(episode_path, obs_from)
            if os.path.isdir(obs_dir):
                obs_files[obs_from] = sorted([os.path.join(obs_dir, x) for x in os.listdir(obs_dir)])
            else:
                obs_files[obs_from] = []
        metadata = self._build_episode_metadata(episode_path, obs_files)

        return {
            "idx": episode_file,
            "path": episode_path,
            "obs_files": obs_files,
            "metadata": metadata,
        }

    def _sample_episode_files(self, episode_files: List[str]) -> List[str]:
        """
        Apply fixed random subset on episode folder names based on `data_percent`.

        - data_percent >= 100: use full dataset
        - 0 < data_percent < 100: keep that percentage of episodes
        """
        percent = self.data_percent * 100.0 if 0.0 < self.data_percent <= 1.0 else self.data_percent

        if percent >= 100.0:
            logger.info(
                "Using full dataset: %d/%d episodes (data_percent=%.2f).",
                len(episode_files),
                len(episode_files),
                percent,
            )
            return episode_files
        if percent <= 0.0:
            raise ValueError(f"data_percent must be > 0, got {self.data_percent}.")

        total = len(episode_files)
        keep = max(1, int(round(total * (percent / 100.0))))
        rng = random.Random(self.data_subset_seed)
        indices = list(range(total))
        rng.shuffle(indices)
        selected = sorted(indices[:keep])
        subset = [episode_files[i] for i in selected]
        logger.info(
            "Using dataset subset: %d/%d episodes (data_percent=%.2f, seed=%d).",
            len(subset),
            total,
            percent,
            self.data_subset_seed,
        )
        return subset

    def _load_episodes(self):
        episode_files = [
            f for f in sorted(os.listdir(self.data_path)) if os.path.isdir(os.path.join(self.data_path, f))
        ]
        selected_episode_files = self._sample_episode_files(episode_files)
        episodes = [self._load_single_episode(f) for f in tqdm(selected_episode_files, desc="Loading episodes")]
        return episodes

    def __len__(self):
        return len(self.episodes)

    def get_frame_indices(self, episode_metadata, first_frame=None):
        if first_frame is not None:
            frames = [first_frame]
        else:
            frames = [random.choice(range(episode_metadata["length"] - 1))]
        skips = []
        while len(frames) < self.num_frames:
            skip = random.randint(self.min_skip, self.max_skip)
            next_idx = min(episode_metadata["length"] - 1, frames[-1] + skip)
            frames.append(next_idx)
            skips.append(skip)
        return frames, skips

    def get_action(self, episode_metadata, frame_idx):
        offset = 1 if self.action_type == "rel_actions" else 0
        return torch.tensor(episode_metadata[self.action_type][offset + frame_idx : offset + frame_idx + self.num_actions])

    def get_extra_data(self, episode_metadata, frame_idx) -> Dict[str, Any]:
        # Extension hook for subclasses (e.g., robot state in CALVIN).
        return {}

    def _sample_language(self, metadata: Dict[str, Any]) -> str:
        try:
            if hasattr(self, "r_map") and hasattr(self, "annos"):
                if "task" not in metadata:
                    metadata["task"] = self.r_map[metadata["language"]]
                return random.choice(self.annos[metadata["task"]])
        except Exception:
            pass

        if isinstance(metadata.get("language"), list):
            return random.choice(metadata["language"])
        return metadata.get("language", "")

    @staticmethod
    def _pad_sequence(x: torch.Tensor, target_len: int) -> torch.Tensor:
        if len(x) >= target_len:
            return x
        if len(x) == 0:
            raise ValueError("Cannot pad an empty tensor sequence.")
        pad_len = target_len - len(x)
        return torch.cat([x, x[-1:].repeat(pad_len, *([1] * (x.ndim - 1)))], dim=0)

    def read_image(self, path):
        return np.array(Image.open(path).convert("RGB"))

    def _read_observation_frames(self, episode: Dict[str, Any], metadata: Dict[str, Any], obs_from: str, frames: List[int]):
        obs_files = episode["obs_files"].get(obs_from, [])
        if not obs_files and "frames" in metadata:
            obs_files = [os.path.join(episode["path"], obs_from, x) for x in metadata["frames"]]
        return np.stack([self.read_image(obs_files[i]) for i in frames], axis=0)

    def __getitem__(self, idx):
        episode = self.episodes[idx]
        metadata = episode["metadata"]

        language = self._sample_language(metadata)
        frames, skips = self.get_frame_indices(metadata, first_frame=None)
        frame_idx = frames[-2] if len(frames) >= 2 else frames[-1]

        data = {
            "idx": episode["idx"],
            "language": language,
            "frame_idx": torch.tensor(frames, dtype=torch.int64),
            "skip_frame": torch.tensor(skips[-1] if skips else 0, dtype=torch.int64),
            "image": {}
        }

        action = self.get_action(metadata, frame_idx)
        data["action"] = self._pad_sequence(action, self.num_actions)

        extra = self.get_extra_data(metadata, frame_idx)
        if extra:
            data.update(extra)

        for obs_type, obs_from in zip(self.observation_type, self.observation_from):
            images = self._read_observation_frames(episode, metadata, obs_from, frames)
            data["image"][obs_type] = self.transform(images=images)["images"] / 255.0

        return data
