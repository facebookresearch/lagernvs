import os
#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Minimal inference script for LagerNVS.

This script demonstrates a complete inference pipeline using the example images
included with the release. Simply run:

    python minimal_inference.py

The output video will be saved to output_video.mp4.
"""

import torch
from eval.export import save_video
from models.encoder_decoder import EncDec_VitB8
from vggt.utils.load_fn import load_and_preprocess_images
from vis import create_target_camera_path, render_chunked
from huggingface_hub import hf_hub_download


def main():
    # Device and dtype setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )

    # Example images included with the release
    image_names = [
        "images/frame_000000.png",
        "images/frame_000003.png",
        "images/frame_000006.png",
    ]
    num_cond_views = len(image_names)
    video_length = 100

    print(f"Loading {num_cond_views} input images...")

    # Load images (width=512, height scaled proportionally, dims divisible by 8)
    images = (
        load_and_preprocess_images(
            image_names, mode="crop", target_size=512, patch_size=8
        )
        .to(device)
        .unsqueeze(0)
    )
    image_size_hw = (images.shape[-2], images.shape[-1])
    print(f"Image size: {image_size_hw}")

    # Create target camera path (uses VGGT internally for pose estimation)
    print("Creating target camera path...")
    rays, cam_tokens = create_target_camera_path(
        image_names, video_length, num_cond_views, image_size_hw, device, dtype
    )

    # Load model
    # Use attention_to_features_type="full_attention" for Re10k model
    # Use attention_to_features_type="bidirectional_cross_attention" for General/DL3DV
    print("Loading model...")
    model = EncDec_VitB8(
        pretrained_vggt=False,
        attention_to_features_type="bidirectional_cross_attention",
    )
    # Support local checkpoint via env var for offline use
    ckpt_path = os.environ.get("LAGERNVS_CKPT_PATH")
    if ckpt_path is None:
        ckpt_path = hf_hub_download("facebook/lagernvs_general_512", filename="model.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Handle both checkpoint formats: {"model": state_dict} or state_dict directly
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Render
    print("Rendering...")
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            video_out = render_chunked(
                model, (images, rays, cam_tokens), num_cond_views=num_cond_views, device=device
            )

    # Save output video
    output_path = "output_video.mp4"
    save_video(video_out[0], output_path)
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    main()
