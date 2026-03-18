# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import os

import av
import einops
import numpy as np
import torch
from torchvision.utils import save_image
from utils import misc


def save_scene_images(
    scene_path: str,
    gt_views: list,
    pred_views: list,
    input_images: torch.Tensor,
    num_cond_views: int,
):
    """Save GT, prediction, and input images for a scene.

    Args:
        scene_path: Path to save scene images
        gt_views: List of ground truth view tensors
        pred_views: List of predicted view tensors
        input_images: Tensor of input conditioning images
        num_cond_views: Number of conditioning views
    """
    os.makedirs(scene_path, exist_ok=True)

    for v_idx in range(len(gt_views)):
        for img, fname in zip([pred_views[v_idx], gt_views[v_idx]], ["pred", "gt"]):
            save_filename = f"{scene_path}/{v_idx:06d}_{fname}.png"
            with io.BytesIO() as buffer:
                save_image(img, buffer, format="PNG")
                buffer.seek(0)
                with open(save_filename, "wb") as f_out:
                    f_out.write(buffer.getvalue())

    for input_idx in range(num_cond_views):
        save_filename = f"{scene_path}/input_{input_idx:06d}.png"
        with io.BytesIO() as buffer:
            save_image(input_images[input_idx], buffer, format="PNG")
            buffer.seek(0)
            with open(save_filename, "wb") as f_out:
                f_out.write(buffer.getvalue())


def save_video(
    video_tensor,
    output_path,
    fps=25,
):
    """Save a single video tensor to mp4.

    Args:
        video_tensor: Video tensor of shape (V, C, H, W), float in [0, 1]
        output_path: Path to save .mp4 file
        fps: Frames per second
    """
    video = einops.rearrange(video_tensor, "v c h w -> v h w c")
    video = video.detach().cpu().numpy()
    video = np.clip(video, 0, 1)
    video = (video * 255).astype(np.uint8)

    with io.BytesIO() as buffer:
        with av.open(buffer, mode="w", format="mp4") as container:
            stream = container.add_stream("libx264", rate=fps)
            stream.height, stream.width = video.shape[1], video.shape[2]
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18"}
            for frame_np in video:
                frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        buffer.seek(0)
        with open(output_path, "wb") as f_out:
            f_out.write(buffer.getvalue())

    print(f"Saved video of length {len(video)} to {output_path}")


def save_video_batch_dist(
    video_out,
    dir_out,
    image_names,
    suffix=None,
):
    """Save video from all batches and all devices.

    Args:
        video_out: Video tensor of shape (B, V, C, H, W)
        dir_out: Output directory for videos
        image_names: List of image name tuples for scene identification
        suffix: Optional suffix for output filename
    """
    for b_idx in range(video_out.shape[0]):
        scene_name = "_".join(image_names[0][b_idx].split("/"))
        if "/" in scene_name:
            scene_name = scene_name.split("/")[0] + "_" + scene_name.split("/")[1]
        img_1 = image_names[0][b_idx].split("_")[-1].split(".")[0]
        img_2 = image_names[1][b_idx].split("_")[-1].split(".")[0]
        path_out = os.path.join(dir_out, f"{scene_name}_{img_1}_{img_2}.mp4")
        if suffix is not None:
            path_out = os.path.join(
                dir_out, f"{scene_name}_{img_1}_{img_2}_{suffix}.mp4"
            )

        save_video(video_out[b_idx], path_out)


def save_eval_scores(
    log_dir,
    dataset_name_log,
    start_iter,
    scores,
    all_scores,
    eval_resolution=None,
    suffix=None,
):
    """Save evaluation scores to a text file organized by dataset name.

    Args:
        log_dir: Base log directory
        dataset_name_log: Dataset name for subdirectory
        start_iter: Iteration number for filename
        scores: Dictionary of aggregate metric scores
        all_scores: Dictionary containing per-scene scores
        eval_resolution: Optional resolution tuple for filename
        suffix: Optional suffix for filename
    """
    scores_dir = os.path.join(log_dir, dataset_name_log)
    if misc.is_main_process():
        os.makedirs(scores_dir, exist_ok=True)

        res_suffix = (
            f"_{eval_resolution[0]}x{eval_resolution[1]}"
            if eval_resolution is not None
            else ""
        )
        suffix = f"_{suffix}" if suffix is not None else ""
        scores_path = os.path.join(
            scores_dir, f"scores_iter_{start_iter:06d}{res_suffix}{suffix}.txt"
        )
        with open(scores_path, "w") as f:
            f.write("=== Evaluation Scores ===\n\n")

            f.write("Per-Scene Scores:\n")
            for i, scene_id in enumerate(all_scores["scene_ids"]):
                psnr = all_scores["psnr"][i].item()
                ssim = all_scores["ssim"][i].item() if "ssim" in all_scores else 0.0
                lpips = all_scores["lpips"][i].item() if "lpips" in all_scores else 0.0
                f.write(
                    f"  {scene_id:20s}: psnr={psnr:7.4f}, ssim={ssim:7.4f}, lpips={lpips:7.4f}\n"
                )

            f.write("\nAggregate Scores:\n")
            for metric, value in scores.items():
                f.write(f"  {metric:10s}: {value:.4f}\n")

        print(f"Saved eval scores to {scores_path}")
    torch.distributed.barrier()
