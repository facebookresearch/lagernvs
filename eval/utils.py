# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

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


def analyze_worst_scenes(psnr_scores, scene_ids, percentile=10):
    """Identify and output the scene IDs with the lowest PSNR scores.

    Args:
        psnr_scores: Tensor containing all PSNR scores gathered from all devices
        scene_ids: List of scene ID strings
        percentile: Percentile threshold for worst scenes (default: 10)

    Returns:
        List of (scene_id, score) tuples for the worst performing scenes
    """
    psnr_np = psnr_scores.numpy()
    threshold = np.percentile(psnr_np, percentile)
    worst_indices = np.where(psnr_np <= threshold)[0]
    worst_sorted = sorted(worst_indices, key=lambda idx: psnr_np[idx])

    worst_scenes_and_scores = []
    for worst_idx in worst_sorted:
        scene_id = scene_ids[worst_idx]
        score = psnr_np[worst_idx]
        worst_scenes_and_scores.append((scene_id, score))

    print("\n===== WORST PERFORMING SCENES (LOWEST 10% PSNR) =====")
    print(f"PSNR threshold: {threshold:.4f}")
    print(f"Number of scenes below threshold: {len(worst_scenes_and_scores)}")
    print("\nScene IDs with lowest PSNR scores (in increasing PSNR order):")
    print("Scene ID\tPSNR Score")
    print("-" * 30)
    for scene_id, score in worst_scenes_and_scores:
        print(f"{scene_id}\t\t{score:.4f}")
    print("=" * 50)
    return worst_scenes_and_scores


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
