# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Scene normalization and camera conditioning token construction."""

from enum import Enum

import numpy as np
import torch
from data import camera_utils
from vggt.utils import pose_enc


CAMERA_SCALE_MULTIPLIER = 1.35


def normalize_extrinsics(
    in_c2ws: torch.Tensor,
    num_cond_views: int,
):
    """Normalize camera extrinsics: re-center on first camera then rescale.

    Uses camera-based normalization: scales by CAMERA_SCALE_MULTIPLIER * max(conditioning camera norms).
    For single-view inference, scene_scale is set to 1.0.

    Args:
        in_c2ws: [V, 4, 4] camera-to-world matrices.
        num_cond_views: Number of conditioning views.

    Returns:
        (in_c2ws, camera_scale, scene_scale_ratio)
    """
    # Use the first camera as the reference frame
    rel_transform_matrix = torch.linalg.inv(in_c2ws[0].clone())[None, ...]
    in_c2ws = rel_transform_matrix @ in_c2ws

    # Rescale the scene based on the conditioning cameras
    if num_cond_views != 1:
        scene_scale = torch.max(torch.norm(in_c2ws[:num_cond_views, :3, 3], dim=-1))
        scene_scale = CAMERA_SCALE_MULTIPLIER * scene_scale
    else:
        # When there is only one camera, the first camera is at the origin
        # We set scene scale to 1.0 to avoid division by zero
        scene_scale = 1.0

    scene_scale_ratio = torch.max(torch.norm(in_c2ws[:, :3, 3], dim=-1)) / scene_scale
    in_c2ws[:, :3, 3] /= scene_scale
    camera_scale = torch.max(torch.norm(in_c2ws[:num_cond_views, :3, 3], dim=-1))

    return (
        in_c2ws,
        camera_scale,
        scene_scale_ratio,
    )


def build_cam_cond(
    c2w_poses,
    intrinsics_fxfycxcy_px,
    num_cond_views,
    tgt_hw,
    camera_scale,
    zero_out_cam_cond_p,
    split,
):
    """Compute camera encoding (Plucker rays) and camera conditioning tokens.

    Args:
        c2w_poses: [V, 4, 4] camera-to-world matrices.
        intrinsics_fxfycxcy_px: [V, 4] intrinsics as (fx, fy, cx, cy).
        num_cond_views: Number of conditioning views.
        tgt_hw: (H, W) target image size.
        camera_scale: Scalar camera scale from normalization.
        zero_out_cam_cond_p: Probability of zeroing out conditioning camera info.
        split: "train" or "test".

    Returns:
        (cam_enc, cam_cond_token): Plucker rays [V, 6, H, W] and tokens [V, 13].
    """
    cam_cond_token = pose_enc.extri_intri_to_pose_encoding(
        c2w_poses.unsqueeze(0),
        intrinsics_fxfycxcy_px.unsqueeze(0),
        image_size_hw=tgt_hw,
    ).squeeze(0)

    Ks = camera_utils.get_K_matrices(intrinsics_fxfycxcy_px)
    cam_enc = camera_utils.compute_plucker_rays(c2w_poses, Ks, tgt_hw)

    zero_out_this_instance = np.random.uniform() <= zero_out_cam_cond_p
    if zero_out_this_instance:
        cam_enc[:num_cond_views] *= 0.0
        cam_cond_token[:num_cond_views] *= 0.0

    # For single-view inference at test time, we set camera_scale to 0
    # and world_points_scale to 1 to indicate no camera-based scaling
    if num_cond_views == 1 and split == "test":
        camera_scale = 0.0
        world_points_scale = 1.0
    else:
        # Camera-based normalization: use camera scale, no points scale
        world_points_scale = 0.0

    scene_scale_tokens = (
        torch.tensor([camera_scale, world_points_scale])
        .unsqueeze(0)
        .expand(cam_cond_token.shape[0], -1)
    )
    cam_cond_token = torch.cat([cam_cond_token, scene_scale_tokens], dim=-1)

    return cam_enc, cam_cond_token
