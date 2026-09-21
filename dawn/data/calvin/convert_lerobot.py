"""Convert a fywang/calvin-task-*-lerobot HuggingFace dataset into our episode
directory structure (see dawn/data/calvin/preprocess.py for the equivalent
converter for raw CALVIN npz dumps).

Source schema (lerobot v2.1, no video encoding, images embedded as PNG bytes):
  observation.images.top    -> rgb_static  (200x200)
  observation.images.wrist  -> rgb_gripper (84x84)
  observation.state         -> robot_obs   (15-dim: tcp_pos(3), tcp_orn(3),
                                             gripper_width(1), joint_pos(7),
                                             gripper_action(1))
  action                    -> rel_actions (7-dim delta action, as recorded)
  task_index -> meta/tasks.jsonl "<task_name>: <sentence>"

`actions` (absolute pose + gripper) isn't provided by the source dataset, so it
is reconstructed the same way CALVIN itself derives it: actions[i] equals the
*next* frame's robot_obs[:6] + gripper (verified byte-identical against the
raw CALVIN task_ABC_D conversion), falling back to the current frame's own
pose for the last step of an episode.
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing
import os
import shutil

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm


def load_task_lookup(meta_dir: str) -> dict[int, str]:
    lookup = {}
    with open(os.path.join(meta_dir, "tasks.jsonl")) as f:
        for line in f:
            entry = json.loads(line)
            _, sentence = entry["task"].split(": ", 1)
            lookup[entry["task_index"]] = sentence
    return lookup


def process_episode(args_tuple):
    parquet_path, episode_index, output_dir, task_lookup = args_tuple
    ep_dir = os.path.join(output_dir, "episodes", str(episode_index))
    static_dir = os.path.join(ep_dir, "rgb_static")
    gripper_dir = os.path.join(ep_dir, "rgb_gripper")
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(gripper_dir, exist_ok=True)

    rows = pq.read_table(parquet_path).to_pylist()
    length = len(rows)

    rel_actions = []
    robot_obs = []
    for i, row in enumerate(rows):
        Image.open(io.BytesIO(row["observation.images.top"]["bytes"])).convert("RGB").save(
            os.path.join(static_dir, f"{i:04d}.jpg"), quality=95
        )
        Image.open(io.BytesIO(row["observation.images.wrist"]["bytes"])).convert("RGB").save(
            os.path.join(gripper_dir, f"{i:04d}.jpg"), quality=95
        )
        rel_actions.append(list(row["action"]))
        robot_obs.append(list(row["observation.state"]))

    actions = []
    for i in range(length):
        nxt = robot_obs[i + 1] if i + 1 < length else robot_obs[i]
        actions.append(nxt[:6] + [nxt[-1]])

    metadata = {
        "language": task_lookup[rows[0]["task_index"]],
        "length": length,
        "actions": actions,
        "rel_actions": rel_actions,
        "robot_obs": robot_obs,
    }
    with open(os.path.join(ep_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)


def main():
    parser = argparse.ArgumentParser(description="Convert a calvin-task-*-lerobot HF dataset")
    parser.add_argument("--repo_id", type=str, default="fywang/calvin-task-D-D-lerobot")
    parser.add_argument("--output_path", type=str, required=True, help="e.g. data/dataset_opt_local/task_D_D")
    parser.add_argument("--annotations_yaml", type=str, default=None, help="annotations.yaml to copy alongside the split (same templates used across CALVIN scenes)")
    parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--download_workers", type=int, default=8, help="Concurrency for the HF Hub download; keep low to avoid the Xet-token rate limit (429s).")
    args = parser.parse_args()

    print(f"Downloading {args.repo_id} ...")
    snapshot_dir = snapshot_download(
        args.repo_id, repo_type="dataset", allow_patterns=["data/**", "meta/**"], max_workers=args.download_workers
    )

    task_lookup = load_task_lookup(os.path.join(snapshot_dir, "meta"))

    with open(os.path.join(snapshot_dir, "meta", "info.json")) as f:
        info = json.load(f)
    splits = info["splits"]
    assert len(splits) == 1, f"Unexpected multi-split source: {splits}"
    (split_name,) = splits.keys()
    output_split = "training" if split_name == "train" else split_name

    output_dir = os.path.join(args.output_path, output_split)
    os.makedirs(os.path.join(output_dir, "episodes"), exist_ok=True)

    with open(os.path.join(snapshot_dir, "meta", "episodes.jsonl")) as f:
        num_episodes = sum(1 for _ in f)

    data_path_template = info["data_path"]
    tasks = []
    for episode_index in range(num_episodes):
        episode_chunk = episode_index // info["chunks_size"]
        parquet_path = os.path.join(
            snapshot_dir, data_path_template.format(episode_chunk=episode_chunk, episode_index=episode_index)
        )
        tasks.append((parquet_path, episode_index, output_dir, task_lookup))

    print(f"Converting {len(tasks)} episodes into {output_dir} with {args.num_workers} workers ...")
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_episode, tasks), total=len(tasks)))

    if args.annotations_yaml:
        shutil.copyfile(args.annotations_yaml, os.path.join(args.output_path, "annotations.yaml"))

    print("Conversion complete.")


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()
