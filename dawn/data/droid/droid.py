from typing import List
import logging
import glob
import os
import torch
import datetime
import json
import imageio
import random
from fvcore.common.timer import Timer
import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf  import OmegaConf
import numpy as np

from torch.utils.data import Dataset
from dawn.data.base_dataset import BaseDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

class DroidDataset(BaseDataset):
    def __init__(self, 
        data_path, 
        split="training", 
        image_size=256,
        num_frames=2,
        num_actions=10,
        min_skip=10,
        max_skip=30,
        observation_type: List[str] = ["rgb_static", "rgb_gripper"],  # Default observation types,
        observation_from: List[str] = ["exterior_image_1_left", "wrist_image_left"],
        action_type: str = "actions",
        data_percent: float = 100.0,
        data_subset_seed: int = 42,
        **kwargs
    ):
        split_root = os.path.join(data_path, split, "episodes")
        self.data_path = split_root if os.path.isdir(split_root) else os.path.join(data_path)
        super().__init__(self.data_path, split, image_size, num_frames, num_actions, min_skip, max_skip, observation_type, observation_from, action_type, data_percent, data_subset_seed)

    def _get_transform(self):
        if self.split == "training":
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                # A.ColorJitter(brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2), saturation_range=(0.8, 1.2), hue_range=(-0.1, 0.1), p=0.5),
                ToTensorV2(),
            ])
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            ToTensorV2(),
        ])

    def get_action(self, metadata, frame_idx):
        cartesian = metadata["action_dict"]["cartesian_velocity"][frame_idx: frame_idx + self.num_actions]
        gripper = metadata["action_dict"]["gripper_velocity"][frame_idx: frame_idx + self.num_actions]
        action = np.concatenate([cartesian, gripper], axis=-1)
        action = torch.tensor(action, dtype=torch.float32)
        return action
