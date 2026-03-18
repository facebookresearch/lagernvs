# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from math import exp

import lpips as lpips_lib
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class MetricsComputer:
    """Unified metrics computation with lazy LPIPS initialization."""

    def __init__(self, device):
        self.device = device
        self._lpips_model = None

    @property
    def lpips_model(self):
        if self._lpips_model is None:
            self._lpips_model = lpips_lib.LPIPS(net="vgg").to(self.device)
            self._lpips_model.eval()
        return self._lpips_model

    @torch.no_grad()
    def compute_all(self, pred_img, gt_img):
        """Compute PSNR, SSIM, and LPIPS for a single image pair."""
        return {
            "psnr": compute_psnr(pred_img, gt_img),
            "ssim": compute_ssim(pred_img, gt_img),
            "lpips": compute_lpips(pred_img, gt_img, self.lpips_model),
        }


@torch.no_grad()
def compute_psnr(gen_image, gt_image):
    """Compute Peak Signal-to-Noise Ratio between two images."""
    return -10 * torch.log10(torch.mean((gen_image - gt_image) ** 2)).item()


@torch.no_grad()
def compute_ssim(img1, img2, window_size=11, size_average=True):
    """Compute Structural Similarity Index between two images."""
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average).item()


@torch.no_grad()
def compute_lpips(gen_image, gt_image, lpips_model):
    """Compute LPIPS perceptual similarity between two images.

    Args:
        gen_image: Generated image tensor in [0, 1] range
        gt_image: Ground truth image tensor in [0, 1] range
        lpips_model: Pre-loaded LPIPS model

    Returns:
        LPIPS distance as a scalar
    """
    lpips = lpips_model(
        gen_image.unsqueeze(0) * 2 - 1, gt_image.unsqueeze(0) * 2 - 1
    ).item()
    return lpips


def gaussian(window_size, sigma):
    """Create a 1D Gaussian kernel."""
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """Create a 2D Gaussian window for SSIM computation."""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """Internal SSIM computation function."""
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
