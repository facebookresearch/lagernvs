#!/usr/bin/env python3
"""
Download script for DL3DV training data.

This script downloads DL3DV scenes from the HuggingFace dataset for training.
It downloads from the 960P resolution repository (DL3DV/DL3DV-ALL-960P).


Usage (external):
    python download_train.py --output_dir /path/to/dl3dv --batch 2K --max_scenes 100

The expected output structure is:
    $OUTPUT_DIR/
    ├── <batch>/<hash>/
    │   ├── images_4/
    │   │   ├── frame_00001.png
    │   │   └── ...
    │   └── transforms.json
    ├── full_list_train.txt
    └── full_list_test.txt

Requirements:
    - huggingface_hub
    - pandas
    - tqdm

You need access to the DL3DV dataset on HuggingFace:
    1. Go to https://huggingface.co/datasets/DL3DV/DL3DV-ALL-960P
    2. Request access to the dataset
    3. Run `huggingface-cli login` to authenticate
"""

import argparse
import os
import shutil
import traceback
import zipfile
from os.path import join

import pandas as pd


from huggingface_hub import HfApi


# Repository for 960P resolution data
REPO_960P = "DL3DV/DL3DV-ALL-960P"

# Valid batch names in DL3DV
VALID_BATCHES = ["1K", "2K", "3K", "4K", "5K", "6K", "7K", "8K", "9K", "10K"]


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


def get_batch_scenes(batch: str, csv_path: str = None) -> list[str]:
    """Get list of scene hashes for a given batch.

    Args:
        batch: Batch name (e.g., "2K", "3K")
        csv_path: Optional path to DL3DV-valid.csv

    Returns:
        List of scene hash names
    """
    # Download or use provided CSV
    if csv_path is None:
        # Try to download from DL3DV repo if not provided
        csv_url = "https://raw.githubusercontent.com/DL3DV-10K/Dataset/main/cache/DL3DV-valid.csv"
        print(f"Note: CSV file not provided. Download DL3DV-valid.csv from {csv_url}")
        print("Or provide --csv_path argument.")
        return []

    df = pd.read_csv(csv_path)
    batch_df = df[df["batch"] == batch]
    return batch_df["hash"].tolist()


def download_scenes(
    api: HfApi,
    batch: str,
    scene_hashes: list[str],
    output_dir: str,
    clean_cache: bool,
) -> tuple[int, int]:
    """Download scenes from the DL3DV-ALL-960P repository.

    Args:
        api: HuggingFace API instance
        batch: Batch name (e.g., "2K")
        scene_hashes: List of scene hash names to download
        output_dir: Output directory
        clean_cache: Whether to clean cache after each download

    Returns:
        Tuple of (success_count, total_count)
    """
    print(f"\nDownloading {len(scene_hashes)} scenes from {REPO_960P} batch {batch}...")

    success_count = 0
    for scene_hash in tqdm(scene_hashes, desc=f"Batch {batch}"):
        rel_path = f"{batch}/{scene_hash}.zip"
        output_path = join(output_dir, batch, scene_hash)

        # Skip if already exists
        if os.path.exists(output_path):
            success_count += 1
            continue

        if not hf_download_path(api, REPO_960P, rel_path, output_dir):
            print(f"Failed to download {rel_path}")
            continue

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
            success_count += 1

        if clean_cache:
            clean_huggingface_cache(output_dir)

    return success_count, len(scene_hashes)


def create_full_list_files(
    scene_entries: list[str], output_dir: str, split: str = "train"
) -> None:
    """Create full_list_train.txt or full_list_test.txt file.

    Args:
        scene_entries: List of scene entries in format "batch/hash"
        output_dir: Output directory
        split: Which split to create ("test" or "train")
    """
    list_path = join(output_dir, f"full_list_{split}.txt")

    lines = [f"{entry}\n" for entry in scene_entries]

    with open(list_path, "w") as f:
        f.writelines(lines)

    print(f"Created {list_path} with {len(lines)} sequences")


def download_train_dataset(args: argparse.Namespace) -> bool:
    """Main function to download the training dataset.

    Args:
        args: Command line arguments

    Returns:
        True if download succeeded
    """
    output_dir = args.output_dir
    batch = args.batch
    max_scenes = args.max_scenes
    clean_cache = not args.no_clean_cache

    # Validate batch
    if batch not in VALID_BATCHES:
        print(f"ERROR: Invalid batch '{batch}'. Valid options: {VALID_BATCHES}")
        return False

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize HuggingFace API
    api = HfApi()

    # Get scene list for this batch
    scene_hashes = get_batch_scenes(batch, args.csv_path)

    if not scene_hashes:
        print(f"No scenes found for batch {batch}")
        return False

    # Limit scenes if max_scenes is specified
    if max_scenes is not None:
        scene_hashes = scene_hashes[:max_scenes]
        print(f"Limiting to {len(scene_hashes)} scenes (--max_scenes={max_scenes})")

    print(f"Found {len(scene_hashes)} scenes to download for batch {batch}")

    # Download scenes
    success_count, total_count = download_scenes(
        api, batch, scene_hashes, output_dir, clean_cache
    )

    print(f"\nDownloaded {success_count}/{total_count} scenes successfully")

    # Create full_list files
    scene_entries = [f"{batch}/{h}" for h in scene_hashes[:success_count]]
    create_full_list_files(scene_entries, output_dir, split="train")

    # Create empty test file if it doesn't exist
    test_path = join(output_dir, "full_list_test.txt")
    if not os.path.exists(test_path):
        open(test_path, "w").close()
        print(f"Created empty {test_path}")

    print("\nDownload complete!")
    print(f"Data saved to: {output_dir}")

    return success_count > 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download DL3DV training data from HuggingFace"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default="2K",
        choices=VALID_BATCHES,
        help="Which batch to download (default: 2K)",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Path to DL3DV-valid.csv file (download from https://github.com/DL3DV-10K/Dataset)",
    )
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=None,
        help="Maximum number of scenes to download (for testing). If not set, downloads all scenes in batch.",
    )
    parser.add_argument(
        "--no_clean_cache",
        action="store_true",
        help="If set, will NOT clean the HuggingFace cache after downloads (default: clean cache)",
    )

    args = parser.parse_args()

    if download_train_dataset(args):
        print("Download Done. Refer to", args.output_dir)
    else:
        print(f"Download to {args.output_dir} failed. See error message above.")
        exit(1)


if __name__ == "__main__":
    main()
