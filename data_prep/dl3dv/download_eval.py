#!/usr/bin/env python3
# Based on https://huggingface.co/datasets/DL3DV/DL3DV-Benchmark/blob/main/download.py
# which is licensed under Creative Commons Attribution-NonCommercial 4.0
# International (CC BY-NC 4.0).
# Modifications are Copyright (c) Meta Platforms, Inc. and affiliates,
# licensed under the license found in the LICENSE file in the root
# directory of this source tree.

"""
Download script for DL3DV evaluation data.

This script downloads the DL3DV scenes required for evaluation based on the
test view indices JSON file. It automatically determines which repository
to use for each scene:
- Scenes with XK prefix (e.g., "2K/...", "3K/...") are downloaded from DL3DV/DL3DV-ALL-960P
- Scenes from "benchmark_missing_from_other_folders" are downloaded from DL3DV/DL3DV-10K-Benchmark


Usage:
    python download_eval.py --output_dir /path/to/dl3dv --view_indices_path assets/dl3dv_2v.json

The expected output structure is:
    $OUTPUT_DIR/
    ├── <batch>/<hash>/
    │   ├── images_4/
    │   │   ├── frame_00001.png
    │   │   └── ...
    │   └── transforms.json
    ├── full_list_train.txt
    └── full_list_test.txt
"""

import argparse
import json
import os
import pickle
import shutil
import traceback
import zipfile
from os.path import join

from huggingface_hub import HfApi
from tqdm import tqdm


# Repository configurations
REPO_960P = "DL3DV/DL3DV-ALL-960P"
REPO_BENCHMARK = "DL3DV/DL3DV-10K-Benchmark"


def hf_download_path(
    api: HfApi, repo: str, rel_path: str, odir: str, max_try: int = 5
) -> bool:
    """Download a file from HuggingFace with retry logic.

    Args:
        api: HuggingFace API instance
        repo: Repository ID (e.g., "DL3DV/DL3DV-ALL-960P")
        rel_path: Relative path to file in repository
        odir: Output directory
        max_try: Maximum number of retry attempts

    Returns:
        True if download succeeded, False otherwise
    """
    counter = 0
    while True:
        if counter >= max_try:
            print(f"ERROR: Download {repo}/{rel_path} failed after {max_try} attempts.")
            return False
        try:
            api.hf_hub_download(
                repo_id=repo,
                filename=rel_path,
                repo_type="dataset",
                local_dir=odir,
                cache_dir=join(odir, ".cache"),
            )
            return True

        except KeyboardInterrupt:
            print("Keyboard Interrupt. Exit.")
            exit(1)
        except Exception:
            traceback.print_exc()
            counter += 1
            print(f"Retry {counter}/{max_try}")


def clean_huggingface_cache(output_dir: str) -> None:
    """Clean HuggingFace cache directory to save space.

    Args:
        output_dir: The output directory containing .cache folder
    """
    cache_dir = join(output_dir, ".cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)


def parse_scene_key(scene_key: str) -> tuple[str, str]:
    """Parse a scene key from the view indices JSON.

    Args:
        scene_key: Scene identifier like "2K/hash..." or "benchmark_missing.../hash..."

    Returns:
        Tuple of (batch_or_prefix, hash)
    """
    parts = scene_key.split("/")
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError(f"Unexpected scene key format: {scene_key}")


def is_960p_scene(batch: str) -> bool:
    """Check if a scene should be downloaded from the 960P repository.

    Args:
        batch: The batch/prefix part of the scene key

    Returns:
        True if scene is from 960P repository (batch ends with 'K')
    """
    return batch.endswith("K") and batch != "benchmark_missing_from_other_folders"


def download_960p_scenes(
    api: HfApi,
    scenes: list[tuple[str, str]],
    output_dir: str,
    clean_cache: bool,
) -> bool:
    """Download scenes from the DL3DV-ALL-960P repository.

    Args:
        api: HuggingFace API instance
        scenes: List of (batch, hash) tuples
        output_dir: Output directory
        clean_cache: Whether to clean cache after each download

    Returns:
        True if all downloads succeeded
    """
    print(f"\nDownloading {len(scenes)} scenes from {REPO_960P}...")

    for batch, scene_hash in tqdm(scenes, desc="960P scenes"):
        rel_path = f"{batch}/{scene_hash}.zip"
        output_path = join(output_dir, batch, scene_hash)

        # Skip if already exists
        if os.path.exists(output_path):
            continue

        if not hf_download_path(api, REPO_960P, rel_path, output_dir):
            print(f"Failed to download {rel_path}")
            return False

        # Extract zip file
        zip_file = join(output_dir, rel_path)
        if os.path.exists(zip_file):
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                members = zip_ref.namelist()
                has_images_root = any(m.startswith("images") for m in members)
                if has_images_root:
                    zip_ref.extractall(output_path)
                else:
                    zip_ref.extractall(join(output_dir, batch))
            os.remove(zip_file)

        if clean_cache:
            clean_huggingface_cache(output_dir)

    return True


def download_benchmark_scenes(
    api: HfApi,
    scenes: list[tuple[str, str]],
    output_dir: str,
    clean_cache: bool,
) -> bool:
    """Download scenes from the DL3DV-10K-Benchmark repository.

    The benchmark repo has files structured as <hash>/nerfstudio/images_4/...
    but we need to restructure them to <original_prefix>/<hash>/images_4/...
    to match the view indices keys.

    Args:
        api: HuggingFace API instance
        scenes: List of (original_prefix, hash) tuples
        output_dir: Output directory
        clean_cache: Whether to clean cache after each download

    Returns:
        True if all downloads succeeded
    """
    if not scenes:
        return True

    print(f"\nDownloading {len(scenes)} scenes from {REPO_BENCHMARK}...")

    # First, download the benchmark metadata
    meta_path = "benchmark-meta.csv"
    cache_path = ".cache/filelist.bin"

    if not hf_download_path(api, REPO_BENCHMARK, meta_path, output_dir):
        print("ERROR: Failed to download benchmark-meta.csv")
        return False

    if not hf_download_path(api, REPO_BENCHMARK, cache_path, output_dir):
        print("ERROR: Failed to download .cache/filelist.bin")
        return False

    # Load the file path dictionary
    filepath_dict = pickle.load(open(join(output_dir, ".cache/filelist.bin"), "rb"))

    # Download each scene
    for original_prefix, scene_hash in tqdm(scenes, desc="Benchmark scenes"):
        # Target directory structure: <original_prefix>/<hash>/images_4/...
        target_scene_dir = join(output_dir, original_prefix, scene_hash)

        # Skip if already exists
        if os.path.exists(target_scene_dir):
            continue

        # The benchmark has a specific directory structure
        # We need to find files for this hash in the filepath_dict
        if scene_hash not in filepath_dict:
            print(f"WARNING: Hash {scene_hash} not found in benchmark metadata")
            continue

        all_files = filepath_dict[scene_hash]

        # Only download images_4 level data (960P equivalent)
        download_files = []
        for f in all_files:
            subdirname = os.path.basename(os.path.dirname(f))
            nerfstudio_or_gs = os.path.basename(os.path.dirname(os.path.dirname(f)))

            # Skip gaussian splatting files
            if nerfstudio_or_gs == "gaussian_splat" or subdirname == "gaussian_splat":
                continue
            # Only keep images_4 and transforms.json
            if subdirname == "images_4" or os.path.basename(f) == "transforms.json":
                download_files.append(f)

        for rel_path in download_files:
            if not hf_download_path(api, REPO_BENCHMARK, rel_path, output_dir):
                print(f"Failed to download {rel_path}")
                return False

        # Restructure: move from <hash>/nerfstudio/... to <original_prefix>/<hash>/...
        # The downloaded structure is: output_dir/<hash>/nerfstudio/images_4/...
        # We want: output_dir/<original_prefix>/<hash>/images_4/...
        downloaded_scene_dir = join(output_dir, scene_hash, "nerfstudio")
        if os.path.exists(downloaded_scene_dir):
            os.makedirs(join(output_dir, original_prefix), exist_ok=True)
            shutil.move(downloaded_scene_dir, target_scene_dir)
            # Clean up the empty hash directory
            empty_hash_dir = join(output_dir, scene_hash)
            if os.path.exists(empty_hash_dir) and not os.listdir(empty_hash_dir):
                os.rmdir(empty_hash_dir)

        if clean_cache:
            clean_huggingface_cache(output_dir)

    return True


def create_full_list_files(
    scene_keys: list[str], output_dir: str, split: str = "test"
) -> None:
    """Create full_list_train.txt and full_list_test.txt files.

    The dataset expects these files to list available sequences.

    Args:
        scene_keys: List of scene keys from the view indices JSON
        output_dir: Output directory
        split: Which split to create ("test" or "train")
    """
    list_path = join(output_dir, f"full_list_{split}.txt")

    # Convert scene keys to the expected format
    lines = []
    for key in scene_keys:
        batch, scene_hash = parse_scene_key(key)

        # For benchmark scenes, we store them with their original prefix
        # but the dataset code expects just the hash paths
        lines.append(f"{batch}/{scene_hash}\n")

    with open(list_path, "w") as f:
        f.writelines(lines)

    print(f"Created {list_path} with {len(lines)} sequences")


def download_eval_dataset(args: argparse.Namespace) -> bool:
    """Main function to download the evaluation dataset.

    Args:
        args: Command line arguments

    Returns:
        True if download succeeded
    """
    output_dir = args.output_dir
    view_indices_path = args.view_indices_path
    clean_cache = not args.no_clean_cache

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize HuggingFace API
    api = HfApi()

    # Load view indices JSON to determine which scenes to download
    print(f"Loading view indices from {view_indices_path}...")
    with open(view_indices_path, "r") as f:
        view_indices = json.load(f)

    scene_keys = list(view_indices.keys())

    # Limit scenes if max_scenes is specified (for testing)
    if args.max_scenes is not None:
        scene_keys = scene_keys[: args.max_scenes]
        print(f"Limiting to {len(scene_keys)} scenes (--max_scenes={args.max_scenes})")

    print(f"Found {len(scene_keys)} scenes to download")

    # Separate scenes by repository
    scenes_960p = []
    scenes_benchmark = []

    for key in scene_keys:
        batch, scene_hash = parse_scene_key(key)
        if is_960p_scene(batch):
            scenes_960p.append((batch, scene_hash))
        else:
            scenes_benchmark.append((batch, scene_hash))

    print(f"  - {len(scenes_960p)} scenes from DL3DV-ALL-960P")
    print(f"  - {len(scenes_benchmark)} scenes from DL3DV-10K-Benchmark")

    # Download 960P scenes
    if scenes_960p:
        if not download_960p_scenes(api, scenes_960p, output_dir, clean_cache):
            return False

    # Download benchmark scenes
    if scenes_benchmark:
        if not download_benchmark_scenes(
            api, scenes_benchmark, output_dir, clean_cache
        ):
            return False

    # Create full_list files
    create_full_list_files(scene_keys, output_dir, split="test")

    # Also create an empty train file for completeness
    train_path = join(output_dir, "full_list_train.txt")
    if not os.path.exists(train_path):
        with open(train_path, "w") as f:
            pass
        print(f"Created empty {train_path}")

    print("\nDownload complete!")
    print(f"Data saved to: {output_dir}")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download DL3DV evaluation dataset based on view indices JSON"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--view_indices_path",
        type=str,
        required=True,
        help="Path to the view indices JSON file (e.g., assets/dl3dv_2v.json)",
    )
    parser.add_argument(
        "--no_clean_cache",
        action="store_true",
        help="If set, will NOT clean the HuggingFace cache after downloads (default: clean cache)",
    )
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=None,
        help="Maximum number of scenes to download (for testing). If not set, downloads all scenes.",
    )

    args = parser.parse_args()

    if download_eval_dataset(args):
        print("Download Done. Refer to", args.output_dir)
    else:
        print(f"Download to {args.output_dir} failed. See error message above.")
        exit(1)


if __name__ == "__main__":
    main()
