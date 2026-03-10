import json
import os

import numpy as np
import torch
from data import camera_utils
from data.normalization import NormalizationMode
from data.sources.base_dataset import BaseDataset


class Dl3dvDataset(BaseDataset):
    ROOT_PATH = os.environ.get("LAGERNVS_DATA_ROOT", "./data") + "/dl3dv"

    def __init__(
        self,
        view_selector,
        im_size_hw,
        split="train",
        num_cond_views=2,
        normalization_mode=NormalizationMode.CAMERAS,
        video_length=0,
        zero_out_cam_cond_p=False,
        video_path_type="linear_interp",
    ):
        super().__init__(
            view_selector=view_selector,
            root_path=self.ROOT_PATH,
            split=split,
            im_size_hw=im_size_hw,
            num_cond_views=num_cond_views,
            normalization_mode=normalization_mode,
            video_length=video_length,
            zero_out_cam_cond_p=zero_out_cam_cond_p,
            video_path_type=video_path_type,
        )

    def _initialize_sequences(self):
        """Initialize sequences - DL3DV specific implementation"""
        list_path = os.path.join(self.root_path, f"full_list_{self.split}.txt")
        with open(list_path, "r") as f:
            full_sequence_list = []
            seq_id_to_folder_map = {}
            for line in f.readlines():
                folder_name = line.strip().split("/")[-2]
                sequence_id = line.strip().split("/")[-1]
                full_sequence_list.append(
                    os.path.join(
                        line.strip().split("/")[-2], line.strip().split("/")[-1]
                    )
                )
                seq_id_to_folder_map[sequence_id] = folder_name

        if hasattr(self.view_selector, "view_indices"):
            self.sequences = list(self.view_selector.view_indices.keys())

            for seq_name in self.sequences:
                if seq_name not in full_sequence_list:
                    print(
                        f"Warning! seq {seq_name} had been removed by prefiltering, it's likely a bad sequence"
                    )
        else:
            self.sequences = full_sequence_list
            
        print(f"Found {len(self.sequences)} sequences")

    def load_cameras(self, seq_name, frame_indices, im_hw_orig, tgt_hw):
        """Load specific frames and their cameras from a sequence"""
        try:
            camera_path = os.path.join(self.root_path, seq_name, "transforms.json")

            # Depthsplat is stored as blender provided by the original dataset
            # our convention is opencv cameras, y down and z backward
            blender2opencv_c2w = np.array(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
            ).astype(np.float32)

            
            with open(camera_path, "r") as f:
                cameras_all = json.load(f)
                w_orig, h_orig, fx_orig, fy_orig, cx_orig, cy_orig = (
                    cameras_all["w"],
                    cameras_all["h"],
                    cameras_all["fl_x"],
                    cameras_all["fl_y"],
                    cameras_all["cx"],
                    cameras_all["cy"],
                )
                cameras = [
                    cameras_all["frames"][frame_idx] for frame_idx in frame_indices
                ]
                # Skip first line and get only needed frames
            im_hw_orig = (h_orig, w_orig)

            intrinsics = []
            c2w_poses = []
            crop_hw_in_orig = camera_utils.get_full_res_crop_dims_constant_ar(
                im_hw_orig, tgt_hw
            )
            for camera in cameras:
                fx, fy, cx, cy = camera_utils.adjust_intrinsics_for_crop_and_resize(
                    (fx_orig, fy_orig, cx_orig, cy_orig),
                    im_hw_orig,
                    crop_hw_in_orig,
                    tgt_hw,
                )
                intrinsics.append([fx, fy, cx, cy])

                # Cameras are stored as blender c2w cameras.
                # Convert to opencv c2w cameras.
                c2w_mat_src = (
                    np.array(camera["transform_matrix"]).astype(np.float32)
                    @ blender2opencv_c2w
                )
                c2w_poses.append(c2w_mat_src)

        except IndexError:
            print(
                f"Sequence {seq_name} tried to sample {len(frame_indices)} images but some are out of range"
            )
            raise IndexError

        return (torch.tensor(np.array(intrinsics)), torch.tensor(np.array(c2w_poses)))

    def get_image_name_list(self, seq_name):
        camera_path = os.path.join(self.root_path, seq_name, "transforms.json")
        try:
            with open(camera_path, "r") as f:
                cameras_all = json.load(f)
        except FileNotFoundError:
            print("Transforms file does not exist")
            return []
        fnames = [
            os.path.basename(camera["file_path"]) for camera in cameras_all["frames"]
        ]
        return fnames

    def get_image_paths_and_frame_indices_for_seq(
        self,
        seq_name,
        num_views,
        num_cond_views,
    ):
        
        seq_path = os.path.join(self.root_path, seq_name, "images_4")

        # in DL3DV not all images had been registered by COLMAP
        # read images from transforms json
        image_name_list = self.get_image_name_list(seq_name)
        image_paths = [
            os.path.join(seq_path, image_name) for image_name in image_name_list
        ]
        # some folders are corrupted and folder is empty
        avail_image_paths = sorted(
            [
                os.path.join(seq_path, f)
                for f in os.listdir(seq_path)
                if f.endswith(".png")
            ]
        )
        if len(avail_image_paths) == 0:
            print(f"Warning! seq {seq_name} does not have images")

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
