# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import os

import numpy as np
import torch
import vis
from data import camera_utils, normalization
from PIL import Image
from torchvision import transforms as T


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        view_selector,
        root_path,
        split="train",
        im_size_hw=(256, 256),
        num_cond_views=None,
        video_length=0,
        zero_out_cam_cond_p=False,
        video_path_type="bspline_interp_eased",
        load_depths=False,
    ):
        self.view_selector = view_selector
        self.root_path = root_path
        self.split = split
        self.im_size_hw = im_size_hw
        self.num_cond_views = num_cond_views
        self.reference_h = im_size_hw[0]
        self.max_hw = max(im_size_hw)
        self.aspect_ratio = im_size_hw[1] / im_size_hw[0]
        self.video_length = video_length
        self.zero_out_cam_cond_p = zero_out_cam_cond_p
        self.video_path_type = video_path_type
        self.load_depths = load_depths

        self.to_tensor = T.ToTensor()

        # Initialize sequences (to be implemented by child classes)
        self.sequences = []
        self._initialize_sequences()

        # Create mapping from sequence name to index
        self.seq_name_to_idx = {
            seq_name: i for i, seq_name in enumerate(self.sequences)
        }

    def __len__(self):
        return len(self.sequences)

    def _initialize_sequences(self):
        """To be implemented by child classes"""
        raise NotImplementedError

    def get_image_paths_and_frame_indices_for_seq(self, seq_name):
        """To be implemented by child classes"""
        raise NotImplementedError

    def load_cameras(self, seq_name, frame_indices):
        """To be implemented by child classes"""
        raise NotImplementedError

    def pad_images_and_concat_poses(
        self,
        images_data,
        c2w_poses_data,
        intrinsics_fxfycxcy_data,
        timesteps_data,
        c2w_poses_traj,
        intrinsics_traj,
        num_cond_views,
        tgt_hw,
    ):
        c2w_poses = torch.concat(
            [c2w_poses_data[:num_cond_views], c2w_poses_traj[: self.video_length]],
            dim=0,
        )
        intrinsics_fxfycxcy_px = torch.concat(
            [
                intrinsics_fxfycxcy_data[:num_cond_views],
                intrinsics_traj[: self.video_length],
            ],
            dim=0,
        )
        images = torch.concat(
            [
                images_data[:num_cond_views],
                torch.zeros((self.video_length, 3, tgt_hw[0], tgt_hw[1])),
            ]
        )
        first_timestamp = timesteps_data[0]
        new_timesteps = np.full((self.video_length + num_cond_views,), first_timestamp)
        new_timesteps[:num_cond_views] = timesteps_data[:num_cond_views]
        return images, c2w_poses, intrinsics_fxfycxcy_px, new_timesteps

    def get_video_data(
        self,
        images,
        c2w_poses,
        intrinsics_fxfycxcy_px,
        selected_timesteps,
        num_cond_views,
        tgt_hw,
    ):
        K_empty = torch.eye(3).float()[None]
        K = K_empty.repeat(intrinsics_fxfycxcy_px.shape[0], 1, 1)
        K[:, 0, 0] = intrinsics_fxfycxcy_px[:, 0]
        K[:, 1, 1] = intrinsics_fxfycxcy_px[:, 1]
        K[:, 0, 2] = intrinsics_fxfycxcy_px[:, 2]
        K[:, 1, 2] = intrinsics_fxfycxcy_px[:, 3]

        if self.video_path_type == "loop_360":
            _, traj_c2w_poses, traj_fxfycxcy = (
                vis.create_360_camera_trajectory_from_c2w_and_intrinsics(
                    c2w_poses=c2w_poses[None, ...],
                    intrinsics=K[None, ...],
                    num_frames_traj=self.video_length,
                    num_cond=num_cond_views,
                )
            )
        elif self.video_path_type == "bspline_interp":
            _, traj_c2w_poses, traj_fxfycxcy = vis.create_bspline_interp(
                c2w_poses=c2w_poses[None, ...],
                intrinsics=K[None, ...],
                num_frames_traj=self.video_length,
                num_cond=num_cond_views,
            )
        elif self.video_path_type == "bspline_interp_eased":
            _, traj_c2w_poses, traj_fxfycxcy = vis.create_bspline_interp(
                c2w_poses=c2w_poses[None, ...],
                intrinsics=K[None, ...],
                num_frames_traj=self.video_length // 2,
                num_cond=num_cond_views,
                ease_in_out=True,
                double_to_repeat=True,
            )
        else:
            raise NotImplementedError

        # vis assumes batched input, so squeeze
        traj_c2w_poses = traj_c2w_poses.squeeze(0)
        traj_fxfycxcy = traj_fxfycxcy.squeeze(0)
        images, c2w_poses, intrinsics_fxfycxcy_px, new_timesteps = (
            self.pad_images_and_concat_poses(
                images,
                c2w_poses,
                intrinsics_fxfycxcy_px,
                selected_timesteps,
                traj_c2w_poses,
                traj_fxfycxcy,
                num_cond_views,
                tgt_hw,
            )
        )
        return images, c2w_poses, intrinsics_fxfycxcy_px, new_timesteps

    def get_image_ids(self, seq_name, image_paths, frame_indices):
        image_names = [
            os.path.basename(image_paths[frame_idx]).split(".jpg")[0]
            for frame_idx in frame_indices
        ]
        image_ids = [seq_name + "_" + image_name for image_name in image_names]
        return image_ids

    def load_images_and_dims(self, image_paths, frame_indices):
        images = []
        for frame_idx in frame_indices:
            with open(image_paths[frame_idx], "rb") as f:
                img = Image.open(f)
                img.load()  # Load image data before file is closed
                img_tensor = self.to_tensor(img)
                if img_tensor.shape[0] == 4:
                    img_tensor = img_tensor[:3]
                images.append(img_tensor)

        orig_hw = (images[0].shape[1], images[0].shape[2])

        for image in images:
            assert (
                image.shape[1] == orig_hw[0] and image.shape[2] == orig_hw[1]
            ), f"shape {image.shape}"

        return images, orig_hw

    def crop_and_resize_data_arrays(
        self,
        data_arrays,
        orig_hw,
        tgt_hw,
        interpolation=T.InterpolationMode.BILINEAR,
        clip_bounds=True,
    ):
        center_crop_dims = camera_utils.get_full_res_crop_dims_constant_ar(
            orig_hw, tgt_hw
        )
        center_crop = T.CenterCrop(center_crop_dims)
        data_arrays = [center_crop(data_array) for data_array in data_arrays]
        resize_fn = T.Resize(tgt_hw, interpolation=interpolation)
        data_arrays = [resize_fn(data_array) for data_array in data_arrays]
        data_arrays = torch.stack(data_arrays)
        # resizing can result in values outside 0-1
        if clip_bounds:
            data_arrays = torch.clip(data_arrays, 0.0, 1.0)

        return data_arrays

    def __getitem__(self, idx_possibly_tuple):
        if type(idx_possibly_tuple) is int:
            assert self.num_cond_views is not None
            return self.get_item(
                idx_possibly_tuple, None, self.num_cond_views, self.aspect_ratio
            )
        elif type(idx_possibly_tuple) is tuple:
            assert self.num_cond_views is None
            seq_index, num_views, num_cond_views, aspect_ratio = idx_possibly_tuple
            return self.get_item(seq_index, num_views, num_cond_views, aspect_ratio)

    def get_item(self, index, num_views, num_cond_views, aspect_ratio):
        seq_name = self.sequences[index]
        scene_scale_ratio = 1e6
        tried_n_times = 0

        # compute target width depending on reference height and aspect ratio
        # aspect ratio is typically expressed as w:h (e.g. 16:9), we
        # express it like that as a float
        tgt_h_ref = self.reference_h
        tgt_w_ref = tgt_h_ref * aspect_ratio
        if tgt_w_ref > self.max_hw:
            tgt_w = self.max_hw
            tgt_h = int(self.max_hw / aspect_ratio) // 8 * 8
        elif tgt_h_ref > self.max_hw:
            tgt_w = int(self.max_hw * aspect_ratio) // 8 * 8
            tgt_h = self.max_hw
        else:
            tgt_w = int(tgt_w_ref) // 8 * 8
            tgt_h = int(tgt_h_ref) // 8 * 8
        tgt_hw = (tgt_h, tgt_w)

        # sometimes the input cameras are very close together.
        # In that case, scaling cameras based on their position
        # will scale them to a huge distance. While loop
        # samples images until the source cameras are far enough apart.
        while scene_scale_ratio > 1e4 or torch.isnan(scene_scale_ratio):
            image_paths, frame_indices, selected_timesteps = (
                self.get_image_paths_and_frame_indices_for_seq(
                    seq_name, num_views, num_cond_views
                )
            )

            if frame_indices is None or len(image_paths) == 0:
                print(
                    f"seq {seq_name} returned frame_indices {frame_indices} - has only {len(image_paths)} images"
                )
                if self.num_cond_views is not None:
                    return self.__getitem__(np.random.randint(0, len(self)))
                else:
                    return self.__getitem__(
                        (
                            np.random.randint(0, len(self)),
                            num_views,
                            num_cond_views,
                            aspect_ratio,
                        )
                    )

            images, orig_hw = self.load_images_and_dims(image_paths, frame_indices)
            images = self.crop_and_resize_data_arrays(images, orig_hw, tgt_hw)

            # read camera poses, adjusts for cropping dimensions
            intrinsics_fxfycxcy_px_post_crop, c2w_poses = self.load_cameras(
                seq_name, frame_indices, orig_hw, tgt_hw
            )

            if self.video_length > 0:
                (
                    images,
                    c2w_poses,
                    intrinsics_fxfycxcy_px_post_crop,
                    selected_timesteps,
                ) = self.get_video_data(
                    images,
                    c2w_poses,
                    intrinsics_fxfycxcy_px_post_crop,
                    selected_timesteps,
                    num_cond_views,
                    tgt_hw,
                )

            (
                c2w_poses,
                camera_scale,
                scene_scale_ratio,
            ) = normalization.normalize_extrinsics(
                c2w_poses,
                num_cond_views=num_cond_views,
            )
            tried_n_times += 1

        # Camera scale == 0 means that there was only one conditioning view
        # world_points is None means that this dataset was not normalized
        # based on points. Both of them mean that scale normalization wasn't valid
        # and we shouldn't train on this example.
        is_valid = torch.tensor(
            float(camera_scale) > 1e-3,
        )

        # hw is needed for fov computation
        cam_enc, cam_cond_token = normalization.build_cam_cond(
            c2w_poses,
            intrinsics_fxfycxcy_px_post_crop,
            num_cond_views=num_cond_views,
            tgt_hw=tgt_hw,
            camera_scale=camera_scale,
            zero_out_cam_cond_p=self.zero_out_cam_cond_p,
            split=self.split,
        )

        image_ids = self.get_image_ids(seq_name, image_paths, frame_indices)

        return (
            images,
            cam_enc,
            image_ids,
            cam_cond_token,
            is_valid,
        )
