from __future__ import annotations

import sys
sys.path.append(".")  # Ensure project root is in path for imports.
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from dawn.data.calvin.calvin import CalvinDataset


def _shape_or_type(x: Any) -> str:
    if torch.is_tensor(x):
        return f"tensor{tuple(x.shape)}"
    if isinstance(x, list):
        return f"list(len={len(x)})"
    if isinstance(x, dict):
        return f"dict(keys={list(x.keys())})"
    return type(x).__name__


def _resolve_data_path(project_root: Path, config_data_path: str) -> Path:
    p = Path(config_data_path)
    if p.is_absolute():
        return p
    return project_root / p


def run_calvin_dataset_dataloader_smoke_test(
    config_data_path: str = "data/dataset_opt_local/task_ABC_D",
    split: str = "validation",
    num_frames: int = 2,
    batch_size: int = 2,
    num_workers: int = 0,
    max_batches: int = 2,
    data_percent: float = 100.0,
):
    project_root = Path(__file__).resolve().parents[1]
    data_path = _resolve_data_path(project_root, config_data_path)

    print("=== CalvinDataset Smoke Test ===")
    print("project_root:", project_root)
    print("data_path:", data_path)
    print("split:", split)
    print("num_frames:", num_frames)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset path not found: {data_path}\n"
            f"Expected from config: {config_data_path}"
        )

    dataset = CalvinDataset(
        data_path=str(data_path),
        split=split,
        num_frames=num_frames,
        observation_type=["rgb_static", "rgb_gripper"],
        action_type="rel_actions",
        use_robot_obs=False,
        data_percent=data_percent,
    )

    print("dataset length:", len(dataset))

    sample = dataset[0]
    print("\n[Single Sample]")
    for k, v in sample.items():
        print(f"- {k}: {_shape_or_type(v)}")

    required_keys = ["idx", "language", "action", "image", "frame_idx", "skip_frame"]
    missing = [k for k in required_keys if k not in sample]
    if missing:
        raise AssertionError(f"Missing required keys in sample: {missing}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    print("\n[Dataloader Batches]")
    for i, batch in enumerate(loader):
        print(f"batch {i}:")
        for k, v in batch.items():
            print(f"  - {k}: {_shape_or_type(v)}")
            if k == "image":
                for view_name, view_tensor in v.items():
                    print(f"    - {view_name}: {_shape_or_type(view_tensor)}")
            if k == "language":
                print(f"    - sample languages: {v}")
        if i + 1 >= max_batches:
            break

    print("\nCalvin dataset+dataloader smoke test passed.")


if __name__ == "__main__":
    try:
        run_calvin_dataset_dataloader_smoke_test()
    except Exception as e:
        print("Smoke test full failed:", e)
        sys.exit(1)

    try:
        run_calvin_dataset_dataloader_smoke_test(data_percent=10)
    except Exception as e:
        print("Smoke test 10% failed:", e)
        sys.exit(1)
