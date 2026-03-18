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

All results are for the **General model** (`lagernvs_general_512`) at 512×512 resolution.

### Re10k Dataset

| Views | Posed | Split | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|-------|-------|-------|--------|--------|---------|
| 2 | ✓ | PixelSplat | 29.05 | 0.901 | 0.147 |
| 2 | ✗ | PixelSplat | 28.28 | 0.885 | 0.155 |
| 2 | ✓ | FLARE | 26.40 | 0.867 | 0.188 |
| 2 | ✗ | FLARE | 25.64 | 0.848 | 0.201 |

### DL3DV Dataset

| Views | Posed | Split | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|-------|-------|-------|--------|--------|---------|
| 2 | ✓ | DepthSplat | 21.77 | 0.692 | 0.287 |
| 2 | ✗ | DepthSplat | 21.33 | 0.670 | 0.301 |
| 4 | ✓ | DepthSplat | 24.94 | 0.780 | 0.188 |
| 4 | ✗ | DepthSplat | 23.99 | 0.744 | 0.206 |
| 6 | ✓ | DepthSplat | 26.14 | 0.808 | 0.159 |
| 6 | ✗ | DepthSplat | 24.97 | 0.769 | 0.178 |
| 16 | ✓ | Rayzer | 25.42 | 0.782 | 0.171 |
| 16 | ✗ | Rayzer | 23.49 | 0.719 | 0.211 |

### CO3D Dataset

| Views | Posed | Split | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|-------|-------|-------|--------|--------|---------|
| 3 | ✓ | ReconFusion | 21.31 | 0.691 | 0.386 |
| 3 | ✗ | ReconFusion | 20.22 | 0.667 | 0.431 |
| 6 | ✓ | ReconFusion | 23.65 | 0.733 | 0.317 |
| 6 | ✗ | ReconFusion | 21.65 | 0.684 | 0.377 |
| 9 | ✓ | ReconFusion | 24.74 | 0.747 | 0.292 |
| 9 | ✗ | ReconFusion | 22.37 | 0.697 | 0.352 |

### MipNeRF-360 Dataset

| Views | Posed | Split | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|-------|-------|-------|--------|--------|---------|
| 3 | ✓ | ReconFusion | 18.08 | 0.434 | 0.497 |
| 3 | ✗ | ReconFusion | 17.45 | 0.413 | 0.531 |
| 6 | ✓ | ReconFusion | 19.39 | 0.469 | 0.436 |
| 6 | ✗ | ReconFusion | 18.97 | 0.447 | 0.466 |
| 9 | ✓ | ReconFusion | 20.39 | 0.493 | 0.402 |
| 9 | ✗ | ReconFusion | 19.68 | 0.462 | 0.438 |

## Notes

- **Posed** (✓): Ground-truth camera poses are provided as input
- **Unposed** (✗): Camera poses are estimated automatically using VGGT
- **Split**: The evaluation split/protocol follows the cited paper's methodology
- All metrics computed at 512×512 resolution for the general model
