import tensorflow_datasets as tfds
import numpy as np
from argparse import ArgumentParser

# tqdm is no longer needed, rich.progress will be used instead
# from tqdm.auto import tqdm
import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
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
        yield (idx, episode, output_dir)

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
    frames = []
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
    json.dump(
        metadata,
        open(os.path.join(output_dir, f"{idx:05d}", "metadata.json"), "w"),
        indent=4
    )
    return f"Processed episode {idx}"


if __name__ == "__main__":
    parser = ArgumentParser(description="Preprocess the droid dataset.")
    parser.add_argument("--input_dir", type=str, default="/nfs/bigcornea/add_disk1/robotics/open-x/droid_100/1.0.0", help="Directory containing the droid dataset.")
    parser.add_argument("--output_dir", type=str, default='/home/nero/Robotics/DAWN/data/droid_100_3', help="Directory to save the preprocessed data.")
    args = parser.parse_args()
    
    start_time = time.time()
    builder = tfds.builder_from_directory(
        builder_dir=args.input_dir,
    )
    ds = builder.as_dataset(split='train')
    output_dir = args.output_dir
    # Get the total number of episodes for the progress bar
    num_episodes = builder.info.splits['train'].num_examples
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

        with ProcessPoolExecutor(32) as executor:
            # 1. Prepare tasks and submit them to the executor
            #    The list of future objects is created here.
            futures = []
            for task in prepare_tasks(ds, output_dir):
                futures.append(executor.submit(process_episode, task))
                # Update the "Preparing" progress bar as each task is submitted
                progress.update(preparing_task_id, advance=1)
            
            # Mark the preparation task as complete once all tasks are submitted
            progress.update(preparing_task_id, description="[bold cyan]Preparation complete ✔")

            # 2. Process results as they are completed
            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    # Using progress.console to print errors without breaking the bar
                    progress.console.print(f"A task generated an exception: {e}")
                finally:
                    # Update the "Processing" progress bar as each future completes
                    progress.update(processing_task_id, advance=1)

    print("\n--- Processing Complete ---")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    num_processed = len(results)
    num_success = sum(1 for r in results if "Processed episode" in r)
    print(f"Total episodes processed: {num_processed}")
    print(f"Successfully processed episodes: {num_success}")
    # Uncomment the following lines to print the results if needed
    # for result in results:
    #     print(result)