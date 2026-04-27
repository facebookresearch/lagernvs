# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: Jonathon Luiten (https://x.com/JonathonLuiten)
#
# Interactive LagerNVS viewer — estimates camera poses with VGGT, encodes scenes with LagerNVS,
# then renders novel views in real-time with interaction via Open3D.
#
# Setup:
#   - Download the model checkpoint from facebook/lagernvs_general_512 (see README)
#     and place it at ./checkpoints/model.pt (or pass --checkpoint path)
#   - Place scene data in ./test_data/ (supports nested subfolders)
#   - Each leaf folder containing images is treated as a scene
#   - Each scene folder should contain 2-10 input images (.png, .jpg, or .jpeg)
#   - Images can be any resolution/aspect ratio; they are center-cropped and resized
#   - All images within a scene should depict the same static subject/environment
#
# Usage:
#   python run_interactive.py                        # all scenes in test_data/
#   python run_interactive.py --scenes chair dog     # specific scenes
#   python run_interactive.py --view_scale 2         # smaller window
#   python run_interactive.py --square               # 512x512 (higher quality, slower)
#
# Controls:
#   1234567890-=][poiuytr   select scene 1-20
#   W/S          move forward/backward
#   A/D          rotate left/right
#   Mouse drag   rotate camera
#   Scroll       zoom
#   Ctrl+drag    pan
#   Esc          quit

import argparse, logging, os, sys, time, warnings
warnings.filterwarnings("ignore", message=".*riton.*")
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["XFORMERS_DISABLE_TRITON"] = "1"
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
logging.getLogger("xformers").setLevel(logging.ERROR)

import einops, numpy as np, open3d as o3d, torch, torch.nn.functional as F
from tqdm import tqdm
torch.backends.cudnn.benchmark = True

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Replace xformers attention with PyTorch native SDPA for broader GPU compatibility
from models.layers.attention import Attention

def _sdpa_forward(self, q, kv=None):
    if kv is None: kv = q
    q, k, v = self.q_proj(q), self.k_proj(kv), self.v_proj(kv)
    q, k, v = (einops.rearrange(t, "b l (nh dh) -> b nh l dh", dh=self.head_dim) for t in (q, k, v))
    if self.use_qk_norm: q, k = self.q_norm(q), self.k_norm(k)
    x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout if self.training else 0.0)
    return self.attn_fc_dropout(self.proj(einops.rearrange(x, "b nh l dh -> b l (nh dh)")))

Attention.forward = _sdpa_forward

from data.camera_utils import compute_plucker_rays, get_K_matrices
from models.encoder_decoder import EncDec_VitB8
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri

CAMERA_SCALE_MULTIPLIER = 1.35
RES_SQUARE = (512, 512)
RES_WIDE = (288, 512)
SCENE_KEYS = [ord(c) for c in "1234567890-=][poiuytr"]
MOVE_STEP = 0.06
ROT_STEP = 0.12
active_scene = 0
reset_view = False
key_state = {}

def setup_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    return device, dtype

def crop_and_resize(img, tgt_h, tgt_w):
    from PIL import Image
    w, h = img.size
    src_ar, tgt_ar = w / h, tgt_w / tgt_h
    if src_ar > tgt_ar:
        crop_w = int(h * tgt_ar)
        left = (w - crop_w) // 2
        img = img.crop((left, 0, left + crop_w, h))
    else:
        crop_h = int(w / tgt_ar)
        top = (h - crop_h) // 2
        img = img.crop((0, top, w, top + crop_h))
    return img.resize((tgt_w, tgt_h), Image.BICUBIC)

def load_images(image_dir, res, device="cuda"):
    from PIL import Image
    from torchvision import transforms as TF
    image_paths = sorted(os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    to_tensor = TF.ToTensor()
    images = torch.stack([to_tensor(crop_and_resize(Image.open(p).convert("RGB"), res[0], res[1])) for p in image_paths])
    return images.to(device).unsqueeze(0), image_paths

def normalize_scene(all_c2w, cond_indices):
    all_c2w_norm = torch.linalg.inv(all_c2w[cond_indices[0]]).unsqueeze(0) @ all_c2w
    scene_scale = torch.clamp(CAMERA_SCALE_MULTIPLIER * torch.max(torch.norm(all_c2w_norm[cond_indices, :3, 3], dim=-1)), min=1e-6)
    all_c2w_norm[:, :3, 3] /= scene_scale
    return all_c2w_norm, torch.max(torch.norm(all_c2w_norm[cond_indices, :3, 3], dim=-1)).item()

def load_model(checkpoint_path, device="cuda"):
    model = EncDec_VitB8(pretrained_vggt=False, attention_to_features_type="bidirectional_cross_attention")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model"])
    return model.to(device).eval()

def load_vggt(device):
    from vggt.models.vggt import VGGT as VGGTModel
    vggt = VGGTModel(pred_cameras=True)
    vggt.load_state_dict(torch.hub.load_state_dict_from_url("https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt", map_location="cpu"), strict=False)
    return vggt.to(device).eval()

def estimate_poses_vggt(images, vggt_model, device, dtype, res):
    H, W = res
    vggt_h = (int(518 * H / W) // 14) * 14
    vggt_images = F.interpolate(images[0], size=(vggt_h, 518), mode="bilinear", antialias=True)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=dtype):
        pose_enc = vggt_model(vggt_images)
    if pose_enc.dim() == 2: pose_enc = pose_enc.unsqueeze(0)
    extrinsics_w2c, intrinsics_3x3 = pose_encoding_to_extri_intri(pose_enc, image_size_hw=res)
    S = extrinsics_w2c.shape[1]
    R_c2w = extrinsics_w2c[:, :, :3, :3].transpose(-1, -2)
    t_c2w = -R_c2w @ extrinsics_w2c[:, :, :3, 3:]
    all_c2w = torch.zeros(S, 4, 4)
    all_c2w[:, :3, :3], all_c2w[:, :3, 3:], all_c2w[:, 3, 3] = R_c2w[0], t_c2w[0], 1.0
    c2w_norm, camera_scale = normalize_scene(all_c2w, list(range(S)))
    avg_f = (intrinsics_3x3[:, :, 0, 0].mean().item() + intrinsics_3x3[:, :, 1, 1].mean().item()) / 2.0
    K = torch.zeros(3, 3)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[2, 2] = avg_f, avg_f, W / 2.0, H / 2.0, 1.0
    return c2w_norm, K, camera_scale, list(range(S))

def build_posed_tokens(c2w_norm, fxfycxcy, cond_indices, camera_scale, res):
    pose_enc = extri_intri_to_pose_encoding(c2w_norm[cond_indices][:, :3, :4].unsqueeze(0), fxfycxcy[cond_indices].unsqueeze(0), image_size_hw=res)
    return torch.cat([pose_enc, torch.tensor([[camera_scale, 0.0]]).expand(1, len(cond_indices), 2)], dim=-1)

def precompute_encoder(model, images, cam_tokens, device, dtype):
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=dtype):
        return einops.rearrange(model.reconstructor(images, cam_tokens.to(device)), "b v p c -> b (v p) c")

def render_single_view(model, rec_tokens, target_rays, dtype):
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=dtype):
        return model.renderer(einops.repeat(rec_tokens, "b np d -> (b v) np d", v=target_rays.shape[1]), target_rays)[0, 0]

def w2c_to_c2w(w2c):
    c2w = torch.eye(4, device=w2c.device, dtype=w2c.dtype)
    c2w[:3, :3], c2w[:3, 3] = w2c[:3, :3].T, -w2c[:3, :3].T @ w2c[:3, 3]
    return c2w

def depth2pts(depth, c2w, K_inv):
    H, W = depth.shape
    y, x = torch.meshgrid(torch.arange(H, device=depth.device, dtype=torch.float32), torch.arange(W, device=depth.device, dtype=torch.float32), indexing="ij")
    uv_hom = torch.stack([x + 0.5, y + 0.5, torch.ones_like(x)], dim=-1)
    dirs = (K_inv @ uv_hom.reshape(-1, 3).T).T.reshape(H, W, 3)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    pts_cam_hom = torch.cat([dirs * depth.unsqueeze(-1), torch.ones(H, W, 1, device=depth.device)], dim=-1)
    return (c2w @ pts_cam_hom.reshape(-1, 4).T).T[:, :3].reshape(H, W, 3)

def make_plucker_rays(w2c, K, res, device):
    w2c = torch.tensor(w2c, dtype=torch.float32) if isinstance(w2c, np.ndarray) else w2c.cpu().float()
    K = torch.tensor(K, dtype=torch.float32) if isinstance(K, np.ndarray) else K.cpu().float()
    fxfycxcy = torch.tensor([[K[0, 0].item(), K[1, 1].item(), K[0, 2].item(), K[1, 2].item()]])
    return compute_plucker_rays(w2c_to_c2w(w2c).unsqueeze(0), get_K_matrices(fxfycxcy), res).unsqueeze(0).to(device)

def prepare_scene(scene_name, scene_dir, model, vggt_model, device, dtype, res):
    print(f"\nPreparing: {scene_name}")
    images, image_paths = load_images(scene_dir, res, device=device)
    t0 = time.time()
    c2w_norm, K, camera_scale, cond_indices = estimate_poses_vggt(images, vggt_model, device, dtype, res)
    print(f"  VGGT Pose Estimation: f={K[0,0]:.1f} scale={camera_scale:.4f} ({time.time()-t0:.2f}s)")
    fxfycxcy = torch.zeros(len(cond_indices), 4)
    fxfycxcy[:, 0], fxfycxcy[:, 1], fxfycxcy[:, 2], fxfycxcy[:, 3] = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    t0 = time.time()
    rec_tokens = precompute_encoder(model, images, build_posed_tokens(c2w_norm, fxfycxcy, cond_indices, camera_scale, res), device, dtype)
    print(f"  LagerNVS Encoded ({time.time()-t0:.2f}s)")
    return {"name": scene_name, "K_np": K.numpy(), "rec_tokens": rec_tokens}

def find_scene_dirs(root):
    scenes = []
    for dirpath, dirnames, filenames in os.walk(root):
        if any(f.lower().endswith((".png", ".jpg", ".jpeg")) for f in filenames):
            scenes.append(dirpath)
    return sorted(scenes)

def main(args):
    global active_scene, reset_view
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

    if len(scene_dirs) > len(SCENE_KEYS):
        print(f"Warning: {len(scene_dirs)} scenes found, only first {len(SCENE_KEYS)} will be loaded (keyboard limit)")
        scene_dirs = scene_dirs[:len(SCENE_KEYS)]

    print("Loading LagerNVS model...")
    t0 = time.time()
    model = load_model(args.checkpoint, device=device)
    print(f"  LagerNVS loaded ({time.time()-t0:.1f}s)")
    print("Loading VGGT model...")
    t0 = time.time()
    vggt_model = load_vggt(device)
    print(f"  VGGT loaded ({time.time()-t0:.1f}s)")
    scenes = [prepare_scene(os.path.relpath(d, test_data_dir), d, model, vggt_model, device, dtype, res) for d in scene_dirs]
    del vggt_model
    torch.cuda.empty_cache()

    print(f"\nScenes ready:")
    key_labels = "1234567890-=][poiuytr"[:len(scenes)]
    for i, sc in enumerate(scenes):
        print(f"  [{key_labels[i]}] {sc['name']} (f={sc['K_np'][0,0]:.1f})")

    K_np = np.array([[W, 0, W/2], [0, W, H/2], [0, 0, 1]], dtype=np.float64)
    K_inv = torch.linalg.inv(torch.tensor(K_np, dtype=torch.float32)).to(device)
    y, x = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32), torch.arange(W, device=device, dtype=torch.float32), indexing="ij")
    uv_hom = torch.stack([x + 0.5, y + 0.5, torch.ones_like(x)], dim=-1)
    cached_dirs = (K_inv @ uv_hom.reshape(-1, 3).T).T.reshape(H, W, 3)
    cached_dirs = cached_dirs / cached_dirs.norm(dim=-1, keepdim=True)
    cached_dirs_hom = torch.cat([cached_dirs, torch.ones(H, W, 1, device=device)], dim=-1).reshape(-1, 4)
    init_w2c_np = np.eye(4, dtype=np.float64)
    win_h, win_w = int(H * args.view_scale), int(W * args.view_scale)

    active_scene = 0
    sc = scenes[0]
    init_im = render_single_view(model, sc["rec_tokens"], make_plucker_rays(init_w2c_np, sc["K_np"], res, device), dtype)
    init_pts = depth2pts(torch.ones(H, W, device=device), torch.eye(4, device=device, dtype=torch.float32), K_inv)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(init_pts.reshape(-1, 3).cpu().numpy().astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(init_im.permute(1, 2, 0).reshape(-1, 3).float().cpu().numpy().astype(np.float64))

    vis = o3d.visualization.VisualizerWithKeyCallback()
    def set_scene(vis, action, mods, idx):
        global active_scene, reset_view
        if action == 1: active_scene, reset_view = idx, True
    for i in range(len(scenes)):
        vis.register_key_action_callback(SCENE_KEYS[i], lambda v, a, m, idx=i: set_scene(v, a, m, idx))
    def on_key(vis, action, mods, key):
        if action != 0: key_state[key] = True
    for k in [ord('W'), ord('S'), ord('A'), ord('D')]:
        vis.register_key_action_callback(k, lambda v, a, m, key=k: on_key(v, a, m, key))
        vis.register_key_action_callback(k + 32, lambda v, a, m, key=k: on_key(v, a, m, key))

    vis.create_window(width=win_w, height=win_h, visible=True)
    vis.add_geometry(pcd)
    view_control = vis.get_view_control()

    def set_camera_params(vc, w2c):
        vk = K_np.copy() * args.view_scale
        vk[2, 2] = 1
        cp = o3d.camera.PinholeCameraParameters()
        cp.extrinsic, cp.intrinsic.intrinsic_matrix, cp.intrinsic.height, cp.intrinsic.width = w2c, vk, win_h, win_w
        vc.convert_from_pinhole_camera_parameters(cp, allow_arbitrary=True)

    set_camera_params(view_control, init_w2c_np)
    view_control.set_constant_z_near(0.001)
    view_control.set_constant_z_far(1000.0)
    render_opts = vis.get_render_option()
    render_opts.point_size, render_opts.light_on, render_opts.background_color = args.view_scale, False, [0, 0, 0]

    frame_count, frame_time, report_every = 0, 0.0, 10
    pbar = tqdm(total=0, bar_format="{desc}")

    while True:
        s_idx = active_scene
        sc = scenes[s_idx]
        if reset_view:
            set_camera_params(view_control, init_w2c_np)
            reset_view = False

        current_w2c = view_control.convert_to_pinhole_camera_parameters().extrinsic
        if any(key_state.get(k) for k in [ord('W'), ord('S'), ord('A'), ord('D')]):
            c2w = np.linalg.inv(current_w2c)
            forward = c2w[:3, 2]
            if key_state.get(ord('W')): c2w[:3, 3] += forward * MOVE_STEP
            if key_state.get(ord('S')): c2w[:3, 3] -= forward * MOVE_STEP
            if key_state.get(ord('A')):
                R = np.array([[np.cos(-ROT_STEP), 0, np.sin(-ROT_STEP)], [0, 1, 0], [-np.sin(-ROT_STEP), 0, np.cos(-ROT_STEP)]])
                c2w[:3, :3] = c2w[:3, :3] @ R
            if key_state.get(ord('D')):
                R = np.array([[np.cos(ROT_STEP), 0, np.sin(ROT_STEP)], [0, 1, 0], [-np.sin(ROT_STEP), 0, np.cos(ROT_STEP)]])
                c2w[:3, :3] = c2w[:3, :3] @ R
            current_w2c = np.linalg.inv(c2w)
            set_camera_params(view_control, current_w2c)
            key_state.clear()
        current_c2w = w2c_to_c2w(torch.tensor(current_w2c, dtype=torch.float32, device=device))

        t_frame = time.time()
        im = render_single_view(model, sc["rec_tokens"], make_plucker_rays(current_w2c, sc["K_np"], res, device), dtype)
        torch.cuda.synchronize()
        frame_time += max(time.time() - t_frame, 1e-7)
        frame_count += 1

        if frame_count == report_every:
            pbar.set_description(f"FPS: {report_every / frame_time:.1f} | [{key_labels[s_idx]}] {sc['name']}")
            frame_time, frame_count = 0.0, 0

        pts = (current_c2w @ cached_dirs_hom.T).T[:, :3].reshape(H, W, 3)
        pcd.points = o3d.utility.Vector3dVector(pts.reshape(-1, 3).cpu().numpy().astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(np.clip(im.permute(1, 2, 0).reshape(-1, 3).float().cpu().numpy(), 0, 1).astype(np.float64))
        vis.update_geometry(pcd)
        if not vis.poll_events(): break
        vis.update_renderer()

    vis.destroy_window()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--view_scale", type=int, default=4)
    p.add_argument("--square", action="store_true", help="Use 512x512 square instead of 288x512 widescreen")
    p.add_argument("--checkpoint", default=os.path.join(SCRIPT_DIR, "checkpoints", "model.pt"))
    main(p.parse_args())
