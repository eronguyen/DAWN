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
        cache_metadata=True,
        observation_type: List[str] = ["rgb_static", "rgb_gripper"],  # Default observation types,
        observation_from: List[str] = ["exterior_image_1_left", "wrist_image_left"],
        **kwargs
    ):
        self.data_path = os.path.join(data_path)
        super().__init__(data_path, split, image_size, num_frames, num_actions, min_skip, max_skip, cache_metadata, observation_type, observation_from)

    def _get_transform(self):
        if self.split == "training":
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
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
