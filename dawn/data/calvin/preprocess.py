import argparse
import numpy as np
import imageio
import os
from tqdm import tqdm
import multiprocessing
from collections import defaultdict
import json
# import tensorflow_hub as hub


def process_episode(args_tuple):
    """
    Processes a single episode: loads data and saves image frames.
    """
    i, start_idx, end_idx, path, output_path, keys, meta_keys, lang, embedding = args_tuple
    
    data = defaultdict(list)

    data["language"] = lang
    data['length'] = int(end_idx - start_idx + 1)
    # data['language_embedding'] = embedding

    for cnt, idx in enumerate(range(start_idx, end_idx + 1)):
        try:
            cur = np.load(f"{path}/episode_{idx:07d}.npz", allow_pickle=True)

            for key in meta_keys:
                data[key].append(cur[key].tolist())
        
            for key in keys:
                ext = "png" if key.startswith("depth") else "jpg"
                img_path = f"{output_path}/episodes/{i}/{key}/{cnt:04d}.{ext}"
                if cnt == 0:
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)
                
                # Ensure the image data is in a savable format (e.g., uint8)
                img_data = cur[key]
                if ext == "png":
                    img_data = (img_data - 3.5) / (6.5 - 3.5) * 255.0
                    # MIN = min(MIN, np.min(img_data))
                    # MAX = max(MAX, np.max(img_data))
                    # print(MIN, MAX)
                    # print(key, np.mean(img_data), np.min(img_data), np.max(img_data), img_data.shape, img_path)
                    
                if img_data.dtype != np.uint8:
                    img_data = img_data.astype(np.uint8)
                
                # print(img_path)
                imageio.imwrite(img_path, img_data)
                
        except FileNotFoundError:
            # Optional: handle cases where an episode file might be missing
            print(f"Warning: File not found for episode {idx:07d}. Skipping.")
            continue
    
    # print(data)
    
    json.dump(data, open(f"{output_path}/episodes/{i}/metadata.json", "w"), indent=4)

if __name__ == "__main__":
    # Set start method for multiprocessing compatibility, especially on macOS/Windows
    multiprocessing.set_start_method('fork', force=True)

    parser = argparse.ArgumentParser(description="Preprocess Calvin dataset for training")
    parser.add_argument("--data_path", type=str, help="Path to calvin data")
    parser.add_argument("--output_path", type=str, help="Path to save preprocessed data")
    # Optional: allow user to specify number of CPU cores to use
    parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count(), help="Number of CPU cores to use")
    args = parser.parse_args()

    splits = ["training","validation"]
    
    # model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    for s in splits:
        print(f"Processing split: {s}")
        path = f"{args.data_path}/{s}"
        output_path = f"{args.output_path}/{s}"
        os.makedirs(f"{output_path}/episodes", exist_ok=True)
        lang_data = np.load(f"{path}/lang_annotations/auto_lang_ann.npy", allow_pickle=True).item()
        
        ep_start_end_ids = lang_data["info"]["indx"]
        lang_ann = lang_data["language"]["ann"]
        
        keys = ["rgb_static", "rgb_gripper"] #["depth_static"]
        meta_keys = ["actions", "rel_actions", "robot_obs"]
        # Prepare a list of arguments for each task

        set_lang = set(lang_ann)
        print(f"Unique languages found: {len(set_lang)}")
        # lang_emb = {lang: model([lang]).numpy().tolist() for lang in set_lang}
        lang_emb = None
        tasks = []
        from tqdm import tqdm
        for i, ((start_idx, end_idx), lang) in enumerate(tqdm(zip(ep_start_end_ids, lang_ann))):
            # embedding = lang_emb[lang]
            embedding = None
            tasks.append((i, start_idx, end_idx, path, output_path, keys, meta_keys, lang, embedding))

        # Create a pool of worker processes
        # The 'with' statement ensures the pool is properly closed
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            # Use tqdm to display a progress bar
            list(tqdm(pool.imap_unordered(process_episode, tasks), total=len(tasks)))

    print("Preprocessing complete.")