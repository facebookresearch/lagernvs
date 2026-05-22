# LagerNVS Model Card

## Model Overview

LagerNVS is a feed-forward model for novel view synthesis (NVS) that generates new views from arbitrary camera viewpoints in a single forward pass. All models use the `EncDecVitB/8` architecture (VGGT encoder + 12-layer cross-attention renderer, patch size 8).

## Available Checkpoints

| Checkpoint | HuggingFace Repo | Training Data | Resolution | Cond. Views | Attention Type |
|-----------|-----------------|---------------|------------|-------------|----------------|
| General | `facebook/lagernvs_general_512` | 15 datasets | 512 (longer side) | 1-10 | Bidirectional Cross-Attention |
| Re10k | `facebook/lagernvs_re10k_2v_256` | Re10k only | 256x256 | 2 | Full Attention |
| DL3DV | `facebook/lagernvs_dl3dv_2-6_v_256` | DL3DV only | 256x256 | 2-6 | Bidirectional Cross-Attention |

## Evaluation Results

Below results are for the **General model v2** (`lagernvs_general_512`) at 512×512 resolution.

### Re10k Dataset (v2)

| Views | Posed | Split | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|-------|-------|-------|--------|--------|---------|
| 2 | ✓ | PixelSplat | 28.99 | 0.900 | 0.149 |
| 2 | ✗ | PixelSplat | 27.88 | 0.875 | 0.161 |
| 2 | ✓ | FLARE | 26.36 | 0.866 | 0.190 |
| 2 | ✗ | FLARE | 25.11 | 0.833 | 0.210 |

### DL3DV Dataset (v2)

| Views | Posed | Split | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|-------|-------|-------|--------|--------|---------|
| 2 | ✓ | DepthSplat | 21.66 | 0.688 | 0.290 |
| 2 | ✗ | DepthSplat | 21.27 | 0.666 | 0.303 |
| 4 | ✓ | DepthSplat | 24.90 | 0.778 | 0.189 |
| 4 | ✗ | DepthSplat | 23.89 | 0.738 | 0.208 |
| 6 | ✓ | DepthSplat | 26.09 | 0.806 | 0.161 |
| 6 | ✗ | DepthSplat | 24.86 | 0.761 | 0.181 |
| 16 | ✓ | Rayzer | 25.20 | 0.776 | 0.174 |
| 16 | ✗ | Rayzer | 23.31 | 0.713 | 0.214 |

### CO3D Dataset (v2)

| Views | Posed | Split | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|-------|-------|-------|--------|--------|---------|
| 3 | ✓ | ReconFusion | 21.18 | 0.688 | 0.393 |
| 3 | ✗ | ReconFusion | 19.88 | 0.660 | 0.445 |
| 6 | ✓ | ReconFusion | 23.41 | 0.728 | 0.326 |
| 6 | ✗ | ReconFusion | 21.37 | 0.680 | 0.388 |
| 9 | ✓ | ReconFusion | 24.40 | 0.740 | 0.302 |
| 9 | ✗ | ReconFusion | 22.05 | 0.689 | 0.365 |

### MipNeRF-360 Dataset (v2)

| Views | Posed | Split | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|-------|-------|-------|--------|--------|---------|
| 3 | ✓ | ReconFusion | 17.74 | 0.428 | 0.505 |
| 3 | ✗ | ReconFusion | 17.29 | 0.409 | 0.535 |
| 6 | ✓ | ReconFusion | 19.17 | 0.466 | 0.444 |
| 6 | ✗ | ReconFusion | 18.81 | 0.446 | 0.472 |
| 9 | ✓ | ReconFusion | 20.19 | 0.487 | 0.412 |
| 9 | ✗ | ReconFusion | 19.69 | 0.461 | 0.440 |

### Results for the previous checkpoint (`revision="v1"`)

| Dataset | Views | Posed | Split | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---------|-------|-------|-------|--------|--------|---------|
| Re10k | 2 | ✓ | PixelSplat | 29.05 | 0.901 | 0.147 |
| Re10k | 2 | ✗ | PixelSplat | 28.28 | 0.885 | 0.155 |
| Re10k | 2 | ✓ | FLARE | 26.40 | 0.867 | 0.188 |
| Re10k | 2 | ✗ | FLARE | 25.64 | 0.848 | 0.201 |
| DL3DV | 2 | ✓ | DepthSplat | 21.77 | 0.692 | 0.287 |
| DL3DV | 2 | ✗ | DepthSplat | 21.33 | 0.670 | 0.301 |
| DL3DV | 4 | ✓ | DepthSplat | 24.94 | 0.780 | 0.188 |
| DL3DV | 4 | ✗ | DepthSplat | 23.99 | 0.744 | 0.206 |
| DL3DV | 6 | ✓ | DepthSplat | 26.14 | 0.808 | 0.159 |
| DL3DV | 6 | ✗ | DepthSplat | 24.97 | 0.769 | 0.178 |
| DL3DV | 16 | ✓ | Rayzer | 25.42 | 0.782 | 0.171 |
| DL3DV | 16 | ✗ | Rayzer | 23.49 | 0.719 | 0.211 |
| CO3D | 3 | ✓ | ReconFusion | 21.31 | 0.691 | 0.386 |
| CO3D | 3 | ✗ | ReconFusion | 20.22 | 0.667 | 0.431 |
| CO3D | 6 | ✓ | ReconFusion | 23.65 | 0.733 | 0.317 |
| CO3D | 6 | ✗ | ReconFusion | 21.65 | 0.684 | 0.377 |
| CO3D | 9 | ✓ | ReconFusion | 24.74 | 0.747 | 0.292 |
| CO3D | 9 | ✗ | ReconFusion | 22.37 | 0.697 | 0.352 |
| Mip360 | 3 | ✓ | ReconFusion | 18.08 | 0.434 | 0.497 |
| Mip360 | 3 | ✗ | ReconFusion | 17.45 | 0.413 | 0.531 |
| Mip360 | 6 | ✓ | ReconFusion | 19.39 | 0.469 | 0.436 |
| Mip360 | 6 | ✗ | ReconFusion | 18.97 | 0.447 | 0.466 |
| Mip360 | 9 | ✓ | ReconFusion | 20.39 | 0.493 | 0.402 |
| Mip360 | 9 | ✗ | ReconFusion | 19.68 | 0.462 | 0.438 |

## Notes

- **Posed** (✓): Ground-truth camera poses are provided as input
- **Unposed** (✗): Camera poses are estimated automatically using VGGT
- **Split**: The evaluation split/protocol follows the cited paper's methodology
- All metrics computed at 512×512 resolution for the general model
