# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: Stan Szymanowicz (https://x.com/StanSzymanowicz)
#
# Browser-based interactive LagerNVS viewer via WebSocket.
# Headless alternative to the Open3D viewer in run_interactive.py — no display
# server required. Scene preparation and model rendering are imported from
# run_interactive.py; this file adds the WebSocket transport and serves the
# HTML/JS client (interactive_viewer.html).
#
# Setup:
#   - Same prerequisites as run_interactive.py (conda env, HF_TOKEN, test_data/).
#   - Additional dependency: pip install websockets
#
# Usage (run on the GPU server):
#   python run_interactive_server.py                        # all scenes in test_data/
#   python run_interactive_server.py --scenes my_scene      # specific scene(s)
#   python run_interactive_server.py --square               # 512x512 (higher quality, slower)
#   python run_interactive_server.py --jpeg_quality 95      # higher quality JPEG (larger frames)
#
# Access (from your local machine):
#   ssh -L 8765:localhost:8765 <gpu-server>
#   Open http://localhost:8765 in your browser.
#
# Controls:
#   Mouse drag       rotate camera (grab-and-drag)
#   Arrow keys       look left/right/up/down
#   W/A/S/D          move forward/left/backward/right
#   Q/E              move up/down
#   [ / ]            decrease/increase movement speed
#   Shift            hold for faster movement
#   R                reset camera to initial viewpoint
#   1-9              switch between loaded scenes
#   Scroll wheel     zoom forward/backward
#
# Known limitations:
#   - Viewer FPS is lower than GPU render FPS due to JPEG encoding and network
#     round-trip latency (especially over SSH tunnels). Both values are shown in
#     the status bar.
#   - High-frequency regions (grass, tree leaves) render with visible artifacts.
#   - Areas not observed by any source image will contain errors.
#   - See run_interactive.py and README.md for additional model limitations.

import asyncio
import io
import json
import logging
import os
import time
import warnings

import numpy as np
import websockets
from PIL import Image

warnings.filterwarnings("ignore", message=".*riton.*")
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["XFORMERS_DISABLE_TRITON"] = "1"
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
logging.getLogger("xformers").setLevel(logging.ERROR)

import torch

torch.backends.cudnn.benchmark = True

from models.layers.attention import Attention

from run_interactive import (
    find_scene_dirs,
    load_model,
    load_vggt,
    make_plucker_rays,
    prepare_scene,
    render_single_view,
    RES_SQUARE,
    RES_WIDE,
    SCRIPT_DIR,
    setup_device,
)


def frame_to_jpeg(frame_tensor, quality=85):
    img_np = frame_tensor.clamp(0, 1).permute(1, 2, 0).mul(255).byte().cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(img_np).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def main(args):
    device, dtype = setup_device()
    res = RES_SQUARE if args.square else RES_WIDE
    H, W = res
    test_data_dir = os.path.join(SCRIPT_DIR, "test_data")

    if args.scenes is not None:
        scene_dirs = [os.path.join(test_data_dir, s) for s in args.scenes]
        for d in scene_dirs:
            assert os.path.isdir(d), f"Scene '{d}' not found"
    else:
        scene_dirs = find_scene_dirs(test_data_dir)

    print(f"Using dtype: {dtype}")

    print("Loading LagerNVS model...")
    model = load_model(args.model_repo, device=device)
    attn_module = next(m for m in model.modules() if isinstance(m, Attention))
    major, minor = torch.cuda.get_device_capability()
    fa3 = "flash3" in str(attn_module.flash_attn_ops)
    if fa3:
        print(f"  Using Flash Attention 3 (SM {major}.{minor})")
    elif major >= 9:
        print(f"  Flash Attention 3 not available in this xformers build, "
              f"falling back to Flash Attention 2 (SM {major}.{minor})")
    else:
        print(f"  Using Flash Attention 2 (SM {major}.{minor}, "
              f"Flash Attention 3 requires SM >= 9.0)")
    print(f"  LagerNVS loaded")

    print("Loading VGGT model...")
    vggt_model = load_vggt(device)
    print("  VGGT loaded")

    scenes = [
        prepare_scene(
            os.path.relpath(d, test_data_dir), d, model, vggt_model, device, dtype, res
        )
        for d in scene_dirs
    ]
    del vggt_model
    torch.cuda.empty_cache()

    scene_names = [sc["name"] for sc in scenes]
    print(f"\nScenes ready: {scene_names}")

    active_scene_idx = 0

    async def handle_client(websocket):
        nonlocal active_scene_idx

        await websocket.send(
            json.dumps(
                {
                    "type": "init",
                    "scenes": scene_names,
                    "up_vectors": [sc["up_vector"] for sc in scenes],
                    "width": W,
                    "height": H,
                }
            )
        )
        print("Client connected")

        frame_count = 0
        frame_time = 0.0
        logged_dtypes = False

        async for message in websocket:
            msg = json.loads(message)

            if msg["type"] == "scene":
                idx = msg["index"]
                if 0 <= idx < len(scenes):
                    active_scene_idx = idx
                    await websocket.send(
                        json.dumps({"type": "scene_ack", "index": idx})
                    )
                continue

            if msg["type"] == "render":
                w2c = np.array(msg["w2c"], dtype=np.float64).reshape(4, 4)
                sc = scenes[active_scene_idx]

                rays = make_plucker_rays(w2c, sc["K_np"], res, device)
                torch.cuda.synchronize()
                t0 = time.time()
                frame = render_single_view(model, sc["rec_tokens"], rays, dtype)
                torch.cuda.synchronize()
                render_time = time.time() - t0

                if not logged_dtypes:
                    print(f"  rec_tokens dtype: {sc['rec_tokens'].dtype}")
                    print(f"  rays dtype: {rays.dtype}")
                    print(f"  output frame dtype: {frame.dtype}")
                    logged_dtypes = True

                jpeg_bytes = frame_to_jpeg(frame, quality=args.jpeg_quality)

                frame_time += render_time
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30 / frame_time
                    print(
                        f"  [{sc['name']}] {fps:.1f} render-fps, "
                        f"{len(jpeg_bytes)/1024:.0f}KB/frame"
                    )
                    frame_time = 0.0

                render_fps = 1.0 / max(render_time, 1e-7)
                header = json.dumps({"render_fps": round(render_fps, 1)}).encode()
                length_prefix = len(header).to_bytes(2, "big")
                await websocket.send(length_prefix + header + jpeg_bytes)

    html_path = os.path.join(SCRIPT_DIR, "interactive_viewer.html")
    with open(html_path, "r") as f:
        html_content = f.read()

    async def process_request(connection, request):
        if request.headers.get("Upgrade") is None:
            response = connection.respond(200, html_content)
            response.headers["Content-Type"] = "text/html; charset=utf-8"
            return response

    async def serve():
        async with websockets.serve(
            handle_client,
            "0.0.0.0",
            args.port,
            max_size=10 * 1024 * 1024,
            process_request=process_request,
        ):
            print(f"Open http://localhost:{args.port} in your browser")
            await asyncio.Future()

    asyncio.run(serve())


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--square", action="store_true")
    p.add_argument("--jpeg_quality", type=int, default=85)
    p.add_argument(
        "--model_repo",
        default="facebook/lagernvs_general_512",
        help="HuggingFace repo ID for the checkpoint",
    )
    main(p.parse_args())
