from __future__ import annotations

import logging
import os
from typing import Dict, List

import torch
from omegaconf import OmegaConf

from dawn.data.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CalvinDataset(BaseDataset):
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
        observation_from: List[str] | None = None,
        action_type="rel_actions",
        use_robot_obs=False,
        **kwargs,
    ):
        self.data_path = os.path.join(data_path, split, "episodes")
        self.use_robot_obs = use_robot_obs

        # Load task-language annotations if available.
        self.annos = {}
        self.r_map = {}
        try:
            self.annos = OmegaConf.load(os.path.join(data_path, "annotations.yaml"))
            for task, phrases in self.annos.items():
                for phrase in phrases:
                    self.r_map[phrase] = task
                self.annos[task].append(task.replace("_", " "))
        except Exception:
            logger.warning("No annotations found, using default language from metadata.")

        super().__init__(
            data_path=data_path,
            split=split,
            image_size=image_size,
            num_frames=num_frames,
            num_actions=num_actions,
            min_skip=min_skip,
            max_skip=max_skip,
            observation_type=observation_type,
            observation_from=observation_from,
            action_type=action_type,
            **kwargs,
        )

        if self.use_robot_obs:
            self.robot_obs_min, self.robot_obs_max, self.robot_obs_range = self._normalize_robot_state()

    def _normalize_robot_state(self):
        # Normalize robot state with dataset-level min/max.
        robot_obs = []
        for episode in self.episodes:
            metadata = episode["metadata"]
            if "robot_obs" in metadata:
                robot_obs.extend(metadata["robot_obs"])
        if len(robot_obs) == 0:
            raise ValueError("use_robot_obs=True but no `robot_obs` found in metadata.")

        robot_obs = torch.tensor(robot_obs, dtype=torch.float32)
        robot_obs_min = robot_obs.min(dim=0)[0]
        robot_obs_max = robot_obs.max(dim=0)[0]
        robot_obs_range = robot_obs_max - robot_obs_min + 1e-8
        logger.info("Robot state normalization ready. dim=%d", robot_obs.shape[-1])
        return robot_obs_min, robot_obs_max, robot_obs_range

    def get_robot_state(self, episode_metadata, frame_idx):
        robot_obs = torch.tensor(
            episode_metadata["robot_obs"][frame_idx : frame_idx + self.num_actions],
            dtype=torch.float32,
        )
        # Keep arm + gripper subset from original implementation.
        robot_obs = (robot_obs[:, 6:-1] - self.robot_obs_min[6:-1]) / self.robot_obs_range[6:-1]
        if robot_obs.ndim == 1:
            robot_obs = robot_obs.unsqueeze(0)
        return robot_obs

    def get_extra_data(self, episode_metadata, frame_idx) -> Dict[str, torch.Tensor]:
        if not self.use_robot_obs:
            return {}
        robot_obs = self.get_robot_state(episode_metadata, frame_idx)
        robot_obs = self._pad_sequence(robot_obs, self.num_actions)
        return {"robot_obs": robot_obs}
