# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

import numpy as np
import torch
from data import camera_utils
from data.sources.base_dataset import BaseDataset


class Re10kDataset(BaseDataset):
    ROOT_PATH = os.environ.get("LAGERNVS_DATA_ROOT", "./data") + "/re10k"

    def __init__(
        self,
        view_selector,
        im_size_hw,
        split="train",
        video_length=0,
        num_cond_views=2,
        zero_out_cam_cond_p=0.0,
        video_path_type="linear_interp",
    ):
        super().__init__(
            view_selector=view_selector,
            root_path=self.ROOT_PATH,
            split=split,
            im_size_hw=im_size_hw,
            num_cond_views=num_cond_views,
            video_length=video_length,
            zero_out_cam_cond_p=zero_out_cam_cond_p,
            video_path_type=video_path_type,
        )

    def _initialize_sequences(self):
        """Initialize sequences - RE10K specific implementation"""
        if hasattr(self.view_selector, "view_indices"):
            self.sequences = list(self.view_selector.view_indices.keys())

            valid_view_indices = {}
            for sequence in self.sequences:
                has_valid_indices = (
                    self.view_selector.view_indices[sequence] is not None
                )
                if not has_valid_indices:
                    continue
                valid_view_indices[sequence] = self.view_selector.view_indices[sequence]
            self.sequences = list(valid_view_indices.keys())
        else:

            list_path = os.path.join(self.root_path, self.split, "full_list.txt")
            with open(list_path, "r") as f:
                all_sequences = [
                    line.strip().split("/")[-1].split(".")[0] for line in f.readlines()
                ]

            # Filter out sequences with too few images for training.
            # This scan is expensive, so we cache the valid sequences list.
            # The cache is stored under the dataset root and only needs to be
            # generated once per split - subsequent runs will load from cache.
            min_images = 25  # view_sampler_range starts at 25
            cache_path = os.path.join(
                self.root_path,
                f".valid_sequences_cache_{self.split}_min{min_images}.json",
            )

            # Try to load from cache first
            if os.path.exists(cache_path):
                print(f"Loading valid sequences from cache: {cache_path}")
                with open(cache_path, "r") as f:
                    cache_data = json.load(f)
                    valid_sequences = cache_data["valid_sequences"]
                    print(f"Loaded {len(valid_sequences)} valid sequences from cache")
            else:
                # Cache doesn't exist - scan all sequences (only done once)
                print(
                    f"Scanning sequences to filter by image count (min={min_images})..."
                )
                print("This scan is only done once - results will be cached.")
                valid_sequences = []
                invalid_sequences = []
                for seq in all_sequences:
                    seq_path = os.path.join(self.root_path, self.split, "images", seq)
                    try:
                        num_images = len(
                            [f for f in os.listdir(seq_path) if f.endswith(".png")]
                        )
                        if num_images >= min_images:
                            valid_sequences.append(seq)
                        else:
                            invalid_sequences.append(
                                {"seq": seq, "num_images": num_images}
                            )
                            print(
                                f"Skipping sequence {seq}: only {num_images} images (need {min_images})"
                            )
                    except Exception as e:
                        invalid_sequences.append({"seq": seq, "error": str(e)})
                        print(f"Skipping sequence {seq}: could not list images")

                # Save cache for future runs
                cache_data = {
                    "min_images": min_images,
                    "valid_sequences": valid_sequences,
                    "invalid_sequences": invalid_sequences,
                }
                try:
                    with open(cache_path, "w") as f:
                        json.dump(cache_data, f, indent=2)
                    print(
                        f"Cached {len(valid_sequences)} valid sequences to: {cache_path}"
                    )
                except Exception as e:
                    print(f"Warning: could not save cache to {cache_path}: {e}")

            self.sequences = valid_sequences
        print(f"Found {len(self.sequences)} sequences")

    def load_cameras(self, seq_name, frame_indices, im_hw_orig, tgt_hw):
        """Load specific frames and their cameras from a sequence"""

        camera_path = os.path.join(
            self.root_path,
            self.split,
            "metadata",
            f"{seq_name}.json",
        )

        with open(camera_path, "r") as f:
            cameras_all = json.load(f)["frames"]
            cameras = [cameras_all[frame_idx] for frame_idx in frame_indices]
            # Skip first line and get only needed frames

        intrinsics = []
        c2w_poses = []
        crop_hw_in_orig = camera_utils.get_full_res_crop_dims_constant_ar(
            im_hw_orig, tgt_hw
        )
        for camera in cameras:
            fx, fy, cx, cy = camera_utils.adjust_intrinsics_for_crop_and_resize(
                camera["fxfycxcy"], im_hw_orig, crop_hw_in_orig, tgt_hw
            )
            intrinsics.append([fx, fy, cx, cy])

            # Convert world-to-camera poses to camera-to-world poses
            w2c_mat_src = np.array(camera["w2c"]).astype(np.float32)
            # Invert to get camera-to-world pose
            c2w_mat_src = np.linalg.inv(w2c_mat_src)
            c2w_poses.append(c2w_mat_src)

        return (torch.tensor(np.array(intrinsics)), torch.tensor(np.array(c2w_poses)))

    def get_image_paths_and_frame_indices_for_seq(
        self,
        seq_name,
        num_views,
        num_cond_views,
    ):

        seq_path = os.path.join(self.root_path, self.split, "images", seq_name)
        image_paths = sorted(
            [
                os.path.join(seq_path, f)
                for f in os.listdir(seq_path)
                if f.endswith(".png")
            ]
        )

        seq_name = os.path.basename(seq_path)

        frame_indices = self.view_selector.sample_views(
            num_views,
            num_cond_views,
            seq_name,
            len(image_paths),
        )

        if frame_indices is None:
            selected_timesteps = None
        else:
            selected_timesteps = torch.zeros(len(frame_indices), dtype=torch.float32)

        return image_paths, frame_indices, selected_timesteps
