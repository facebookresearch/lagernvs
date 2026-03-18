# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from eval.distributed import gather_metrics_distributed
from eval.export import save_scene_images
from eval.metrics import MetricsComputer
from eval.utils import mask_target_views
from vis import render_chunked


@torch.no_grad()
def run_cond_eval(
    generator,
    device,
    num_cond_views,
    dataloader,
    rank,
    world_size,
    save_path=None,
    eval_resolution=None,
):
    """Run conditional evaluation on a dataset.

    Args:
        generator: The model to evaluate
        device: torch device
        num_cond_views: Number of conditioning views
        dataloader: DataLoader for evaluation data
        rank: Current process rank
        world_size: Total number of distributed processes
        save_path: Optional path to save images
        eval_resolution: Optional (H, W) tuple to resize images for metrics

    Returns:
        gathered_metrics: Dict of averaged metrics
        all_scores: Dict of all individual scores
    """
    all_metrics = {"psnr": [], "ssim": [], "lpips": [], "scene_ids": []}
    metrics_computer = MetricsComputer(device)

    for images, rays, image_ids, cam_token, _ in dataloader:
        images = images.to(device)
        rays = rays.to(device)
        cam_token = cam_token.to(device)
        scene_ids = [image_id.split(".")[0] for image_id in image_ids[0]]

        cond_image_input = mask_target_views(images, num_cond_views)
        chunk_size = max(1, 48 // num_cond_views)
        pred_image = render_chunked(
            generator,
            (cond_image_input, rays, cam_token),
            view_chunk_size=chunk_size,
            num_cond_views=num_cond_views,
        )

        target_gt_image = images[:, num_cond_views:]

        if save_path is not None:
            for scene_idx in range(pred_image.shape[0]):
                scene_id_txt = scene_ids[scene_idx].replace("/", "_")
                scene_path = f"{save_path}/{scene_id_txt}"
                save_scene_images(
                    scene_path,
                    gt_views=[
                        target_gt_image[scene_idx, v]
                        for v in range(pred_image.shape[1])
                    ],
                    pred_views=[
                        pred_image[scene_idx, v] for v in range(pred_image.shape[1])
                    ],
                    input_images=images[scene_idx],
                    num_cond_views=num_cond_views,
                )

        for scene_idx in range(pred_image.shape[0]):
            psnr_views, ssim_views, lpips_views = [], [], []

            for pred_img, gt_img in zip(
                pred_image[scene_idx], target_gt_image[scene_idx]
            ):
                if eval_resolution is not None:
                    eval_h, eval_w = eval_resolution
                    pred_img, gt_img = [
                        F.interpolate(
                            im.unsqueeze(0),
                            size=(eval_h, eval_w),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)
                        for im in [pred_img, gt_img]
                    ]
                metrics = metrics_computer.compute_all(pred_img, gt_img)
                psnr_views.append(metrics["psnr"])
                ssim_views.append(metrics["ssim"])
                lpips_views.append(metrics["lpips"])

            all_metrics["psnr"].append(sum(psnr_views) / len(psnr_views))
            all_metrics["ssim"].append(sum(ssim_views) / len(ssim_views))
            all_metrics["lpips"].append(sum(lpips_views) / len(lpips_views))
            all_metrics["scene_ids"].append(scene_ids[scene_idx])

    gathered_metrics, all_scores = gather_metrics_distributed(
        all_metrics, device, world_size
    )

    if rank == 0:
        print("Evaluation metrics:")
        for metric_name, metric_value in gathered_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")

    return gathered_metrics, all_scores
