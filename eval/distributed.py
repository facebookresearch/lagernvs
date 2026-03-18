# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils import misc


def gather_metrics_distributed(
    all_metrics: dict, device, world_size: int
) -> tuple[dict, dict]:
    """Gather evaluation metrics from all distributed processes.

    Args:
        all_metrics: Dict with keys "psnr", "ssim", "lpips", "scene_ids"
        device: torch device
        world_size: number of distributed processes

    Returns:
        gathered_metrics: Dict of averaged metrics
        all_scores: Dict of all individual scores
    """
    all_scores = {}
    gathered_metrics = {}

    torch.distributed.barrier()
    print("Gathering scores from all processes...")

    max_count = 0

    for metric_name in ["psnr", "ssim", "lpips"]:
        local_tensor = torch.tensor(all_metrics[metric_name], device=device)
        local_count = torch.tensor([len(all_metrics[metric_name])], device=device)

        all_counts = [torch.zeros_like(local_count) for _ in range(world_size)]
        torch.distributed.all_gather(all_counts, local_count)
        counts = [count.item() for count in all_counts]

        gathered_scores = []
        max_count = max(counts) if counts else 0

        print(
            f"Padding local tensor of length {local_tensor.shape[0]} to max count: {max_count} ",
            force=True,
        )
        if local_tensor.shape[0] < max_count:
            padding = torch.full(
                (max_count - local_tensor.shape[0],), float("nan"), device=device
            )
            local_tensor = torch.cat([local_tensor, padding])

        gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_tensors, local_tensor)

        for _, (tensor, count) in enumerate(zip(gathered_tensors, counts)):
            gathered_scores.append(tensor[:count].cpu())

        all_scores[metric_name] = torch.cat(gathered_scores)
        gathered_metrics[metric_name] = all_scores[metric_name].mean().item()

    gathered_scene_ids = _gather_scene_ids(
        all_metrics["scene_ids"], max_count, world_size, counts
    )
    all_scores["scene_ids"] = gathered_scene_ids

    return gathered_metrics, all_scores


def _gather_scene_ids(
    local_scene_ids: list, max_count: int, world_size: int, counts: list
) -> list:
    """Gather scene ID strings from all processes.

    Args:
        local_scene_ids: List of scene IDs from this process
        max_count: Maximum count across all processes
        world_size: Number of distributed processes
        counts: List of counts per process

    Returns:
        Flat list of all scene IDs from all processes
    """
    print("Gathering scene names from all processes...")
    gathered_scene_ids = []

    print(
        f"Padding local tensor of length {len(local_scene_ids)} on rank {misc.get_rank()} to max count: {max_count} ",
        force=True,
    )
    if len(local_scene_ids) < max_count:
        padding = [None] * (max_count - len(local_scene_ids))
        local_scene_ids = local_scene_ids + padding

    gathered_lists = [[None] * max_count for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered_lists, local_scene_ids)

    for _, (sids, count) in enumerate(zip(gathered_lists, counts)):
        gathered_scene_ids.append(sids[:count])

    flat_gathered_scene_ids = []
    for l_sids in gathered_scene_ids:
        flat_gathered_scene_ids.extend(l_sids)

    return flat_gathered_scene_ids
