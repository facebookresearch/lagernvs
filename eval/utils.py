# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random

import einops
import numpy as np
import torch
import torchvision


def set_seed(seed):
    """Set random seed for reproducibility across random, torch, and numpy."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def mask_target_views(images, num_cond_views):
    """Zero out target views for conditioning.

    Args:
        images: Image tensor of shape (B, V, C, H, W)
        num_cond_views: Number of conditioning views to keep

    Returns:
        Tensor with conditioning views preserved and target views zeroed
    """
    return torch.cat(
        [
            images[:, :num_cond_views],
            torch.zeros_like(images[:, num_cond_views:]),
        ],
        dim=1,
    )


def image_tensor_to_grid_numpy(batch):
    """Convert a batch of image tensors to a grid as a numpy array.

    Args:
        batch: Tensor of shape (B, V, C, H, W)

    Returns:
        Numpy array of shape (H', W', C) representing the grid
    """
    grid = einops.rearrange(batch, "b v c h w -> (b v) c h w")
    grid = torchvision.utils.make_grid(grid[:64], nrow=8)
    bigger_dimension = max(grid.shape[1], grid.shape[2])
    resize_factor = 512 / bigger_dimension
    out_shape = (int(grid.shape[1] * resize_factor), int(grid.shape[2] * resize_factor))
    grid = torchvision.transforms.functional.resize(
        grid,
        size=out_shape,
        interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
        antialias=True,
    )
    grid = grid.permute(1, 2, 0).numpy()
    grid = np.clip(grid, 0, 1)
    return grid
