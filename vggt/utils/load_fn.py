# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the VGGT license found at
# https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt

import torch
from PIL import Image
from torchvision import transforms as TF


def load_and_preprocess_images(
    image_path_list, mode="square_crop", target_size=512, patch_size=8
):
    """
    Load and preprocess images for model input.

    Args:
        image_path_list (list): List of paths to image files
        mode (str): Preprocessing mode.
            - "square_crop": Center-crops to the largest inscribed square at original
              resolution, then resizes to target_size x target_size.
            - "resize": Resizes maintaining aspect ratio so that the longer side
              equals target_size. The shorter side is rounded to the nearest multiple
              of patch_size. Raises ValueError if the shorter side would be less
              than 0.5 * target_size.
        target_size (int): Target size in pixels (default: 512)
        patch_size (int): Patch size for dimension rounding in "resize" mode (default: 8)

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty, mode is invalid, or aspect ratio
            is too extreme in "resize" mode.
    """
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    if mode not in ["square_crop", "resize"]:
        raise ValueError("Mode must be 'square_crop' or 'resize'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()

    for image_path in image_path_list:
        with open(image_path, "rb") as f:
            img = Image.open(f)
            img.load()

        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "square_crop":
            short_side = min(width, height)
            left = (width - short_side) // 2
            top = (height - short_side) // 2
            img = img.crop((left, top, left + short_side, top + short_side))
            img = img.resize((target_size, target_size), Image.Resampling.BICUBIC)
            img = to_tensor(img)
        else:  # mode == "resize"
            if width >= height:
                new_width = target_size
                new_height = (
                    round(height * (target_size / width) / patch_size) * patch_size
                )
            else:
                new_height = target_size
                new_width = (
                    round(width * (target_size / height) / patch_size) * patch_size
                )

            shorter_side = min(new_width, new_height)
            if shorter_side < 0.5 * target_size:
                raise ValueError(
                    f"Image aspect ratio too extreme: shorter side ({shorter_side}px) "
                    f"is less than 0.5 * target_size ({0.5 * target_size:.0f}px). "
                    f"Original size: {width}x{height}. "
                    f"Consider using mode='square_crop' instead."
                )

            img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
            img = to_tensor(img)

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    if len(shapes) > 1:
        raise ValueError(
            f"Input images have different shapes after preprocessing: {shapes}. "
            f"All images must have the same resolution. Please crop or resize "
            f"your input images so they share approximately the same intrinsic "
            f"parameters (resolution and field of view)."
        )

    images = torch.stack(images)

    if len(image_path_list) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images
