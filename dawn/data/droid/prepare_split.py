#!/usr/bin/env python3
"""Split flat DROID episodes into Calvin-style train/validation layout.

Source layout (flat):
    {source_dir}/{episode_id}/
        exterior_image_1_left/
        exterior_image_2_left/
        wrist_image_left/
        metadata.json

Output layout (matches calvin-dawn task_ABC_D):
    {output_dir}/
        training/episodes/{0..N-1}/...
        validation/episodes/{0..511}/...
        split_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, List, Sequence, Tuple

from tqdm import tqdm


def list_episode_dirs(source_dir: str) -> List[str]:
    episodes = []
    for name in os.listdir(source_dir):
        path = os.path.join(source_dir, name)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "metadata.json")):
            episodes.append(name)
    episodes.sort()
    return episodes


def split_episodes(
    episodes: Sequence[str],
    num_validation: int,
    seed: int,
) -> Tuple[List[str], List[str]]:
    if num_validation <= 0:
        raise ValueError(f"num_validation must be > 0, got {num_validation}.")
    if num_validation >= len(episodes):
        raise ValueError(
            f"num_validation ({num_validation}) must be smaller than total episodes ({len(episodes)})."
        )

    rng = random.Random(seed)
    shuffled = list(episodes)
    rng.shuffle(shuffled)
    val_episodes = shuffled[:num_validation]
    train_episodes = shuffled[num_validation:]
    return train_episodes, val_episodes


def link_or_copy_episode(
    source_episode_dir: str,
    dest_episode_dir: str,
    mode: str,
) -> None:
    if os.path.lexists(dest_episode_dir):
        if os.path.islink(dest_episode_dir) or os.path.isdir(dest_episode_dir):
            if mode == "symlink":
                os.remove(dest_episode_dir)
            else:
                shutil.rmtree(dest_episode_dir)
        else:
            os.remove(dest_episode_dir)

    if mode == "symlink":
        os.symlink(source_episode_dir, dest_episode_dir, target_is_directory=True)
        return

    shutil.copytree(source_episode_dir, dest_episode_dir, dirs_exist_ok=False)


def materialize_split(
    source_dir: str,
    output_dir: str,
    split_name: str,
    episode_ids: Iterable[str],
    mode: str,
    num_workers: int,
) -> List[dict]:
    split_root = os.path.join(output_dir, split_name, "episodes")
    os.makedirs(split_root, exist_ok=True)

    episode_ids = list(episode_ids)
    tasks = []
    for new_idx, episode_id in enumerate(episode_ids):
        src = os.path.join(source_dir, episode_id)
        dst = os.path.join(split_root, str(new_idx))
        tasks.append((new_idx, episode_id, src, dst))

    manifest = [None] * len(tasks)

    def _run(task: Tuple[int, str, str, str]) -> Tuple[int, dict]:
        new_idx, episode_id, src, dst = task
        link_or_copy_episode(src, dst, mode)
        return new_idx, {"new_idx": new_idx, "source_id": episode_id}

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(_run, task) for task in tasks]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"{split_name}",
        ):
            new_idx, entry = future.result()
            manifest[new_idx] = entry

    return manifest


def write_manifest(
    output_dir: str,
    seed: int,
    num_validation: int,
    train_manifest: List[dict],
    val_manifest: List[dict],
) -> str:
    manifest = {
        "seed": seed,
        "num_validation": num_validation,
        "num_training": len(train_manifest),
        "training": train_manifest,
        "validation": val_manifest,
    }
    manifest_path = os.path.join(output_dir, "split_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly split flat DROID episodes into Calvin-style train/val layout."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="/home/colligo/Codes/localssd/droid",
        help="Flat DROID directory containing one folder per episode.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/colligo/Codes/localssd/droid-dawn",
        help="Output root matching calvin-dawn task_ABC_D structure.",
    )
    parser.add_argument(
        "--num_validation",
        type=int,
        default=512,
        help="Number of episodes reserved for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/validation split.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["symlink", "copy"],
        default="symlink",
        help="Use symlinks (fast) or copy episode folders.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Parallel workers for linking/copying episodes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    episodes = list_episode_dirs(args.source_dir)
    print(f"Found {len(episodes)} episodes in {args.source_dir}")

    train_ids, val_ids = split_episodes(episodes, args.num_validation, args.seed)
    print(
        f"Split with seed={args.seed}: "
        f"{len(train_ids)} training, {len(val_ids)} validation"
    )

    os.makedirs(args.output_dir, exist_ok=True)

    train_manifest = materialize_split(
        args.source_dir,
        args.output_dir,
        "training",
        train_ids,
        args.mode,
        args.num_workers,
    )
    val_manifest = materialize_split(
        args.source_dir,
        args.output_dir,
        "validation",
        val_ids,
        args.mode,
        args.num_workers,
    )

    manifest_path = write_manifest(
        args.output_dir,
        args.seed,
        args.num_validation,
        train_manifest,
        val_manifest,
    )

    print(f"Done. Output written to {args.output_dir}")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
