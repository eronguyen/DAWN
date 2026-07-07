import tensorflow_datasets as tfds
import numpy as np
# tqdm is no longer needed, rich.progress will be used instead
# from tqdm.auto import tqdm
import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
import tensorflow as tf
from collections import defaultdict
import json
import time

# Import necessary components from the 'rich' library
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

OFFSET=0
# This line is commented out as in the original code
# tf.config.set_visible_devices([], "GPU")
def prepare_tasks(dataset, output_dir):
    """
    A generator function that prepares tasks one by one.
    This function fetches an episode, converts its 'steps' to a NumPy list,
    and then 'yields' the complete, serializable task.
    (This function remains unchanged)
    """
    for idx, episode in enumerate(dataset):
        # Convert the nested 'steps' dataset into a list of NumPy dicts
        episode['steps'] = list(episode['steps'].as_numpy_iterator())
        # Yield the fully prepared task
        yield (idx + OFFSET, episode, output_dir)

def process_episode(args):
    """
    This function remains the same. It runs in a worker process.
    (This function remains unchanged)
    """
    idx, episode_data, output_dir = args
    metadata = defaultdict(list)
    metadata['idx'] = idx
    metadata['language'] = []
    metadata['frames'] = []
    action_dict = defaultdict(list)
    action_keys = ['cartesian_velocity', 'gripper_velocity']

    for i, step in enumerate(episode_data['steps']):
        # Collect language
        if i == 0:
            for j in range(3):
                lang_key = f'language_instruction'
                if j: lang_key += f'_{j + 1}'
                lang = step[lang_key].decode('utf-8')
                if len(lang) > 0:
                    metadata['language'].append(lang)
        if len(metadata['language']) == 0:
            return f"Empty language instruction in episode {idx}."

        # Collect actions
        diff = 0
        for key in action_keys:
            diff += np.sum(step['action_dict'][key] ** 2)
        if diff == 0:
            # return f"Zero action in episode {idx}, step {i}."
            continue

        for key in action_keys:
            action_dict[key].append(step['action_dict'][key].tolist())

        frame_name = f"{i:04d}.jpg"
        metadata['frames'].append(frame_name)
        # Collect observations
        for k, v in step["observation"].items():
            if "image" in k:
                image = v # This is already a NumPy array
                cur_out = os.path.join(output_dir, f"{idx:05d}", k, frame_name)
                os.makedirs(os.path.dirname(cur_out), exist_ok=True)
                Image.fromarray(image).save(cur_out)

    # metadata['length'] = len(episode_data['steps'])
    metadata['length'] = len(metadata['frames'])
    metadata['action_dict'] = action_dict
    with open(os.path.join(output_dir, f"{idx:05d}", "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    return f"Processed episode {idx}"

if __name__ == "__main__":
    start_time = time.time()
    ds = tfds.load("droid", data_dir="gs://gresearch/robotics", split=f"train[{OFFSET}:]")

    # output_dir = '/nfs/bigcornea/add_disk2/nero/datasets/robotics/droid/opt/validation'
    output_dir = '/home/colligo/Codes/localssd/droid/'
    # Get the total number of episodes for the progress bar
    num_episodes = len(ds)
    print(f"Total episodes to process: {num_episodes}")

    # Define the rich progress bar with custom columns for a clean look
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )

    with progress:
        # Add two separate tasks to the progress bar
        preparing_task_id = progress.add_task("[cyan]Preparing tasks...", total=num_episodes)
        processing_task_id = progress.add_task("[green]Processing episodes...", total=num_episodes)

        # Cap in-flight tasks to bound RAM (each slot holds one episode's numpy arrays).
        MAX_INFLIGHT = 256
        sem = Semaphore(MAX_INFLIGHT)

        with ThreadPoolExecutor(max_workers=128) as executor:
            # 1. Prepare tasks and submit them to the executor
            #    The list of future objects is created here.
            futures = []
            for task in prepare_tasks(ds, output_dir):
                sem.acquire()
                future = executor.submit(process_episode, task)
                future.add_done_callback(lambda _: sem.release())
                futures.append(future)
                # Update the "Preparing" progress bar as each task is submitted
                progress.update(preparing_task_id, advance=1)
            
            # Mark the preparation task as complete once all tasks are submitted
            progress.update(preparing_task_id, description="[bold cyan]Preparation complete ✔")

            # 2. Process results as they are completed
            num_processed = num_success = 0
            for future in as_completed(futures):
                try:
                    result = future.result()
                    num_processed += 1
                    if result and result.startswith("Processed episode"):
                        num_success += 1
                except Exception as e:
                    progress.console.print(f"A task generated an exception: {e}")
                    num_processed += 1
                finally:
                    progress.update(processing_task_id, advance=1)

    print("\n--- Processing Complete ---")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    print(f"Total episodes processed: {num_processed}")
    print(f"Successfully processed episodes: {num_success}")