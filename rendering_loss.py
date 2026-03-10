import einops
import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from easydict import EasyDict as edict


class PerceptualLoss(nn.Module):
    # ImageNet normalization mean values (RGB order)
    IMAGENET_MEAN: tuple[float, ...] = (123.6800 / 255, 116.7790 / 255, 103.9390 / 255)

    # Normalization divisors for each feature level (e1 through e5)
    FEATURE_SCALE_DIVISORS: tuple[float, ...] = (2.6, 4.8, 3.7, 5.6, 1.5 / 10)

    def __init__(self, device: str = "cpu", post=True) -> None:
        super().__init__()
        # Layer indices for multi-scale feature extraction (after conv layers)
        # Maps to: relu1_2, relu2_2, relu3_2, relu4_2, relu5_2
        feature_layer_indices_post: tuple[int, ...] = (3, 8, 13, 22, 31)
        # Maps to: conv1_2, conv2_2, conv3_2, conv4_2, conv5_2
        feature_layer_indices_pre: tuple[int, ...] = (2, 7, 12, 21, 30)
        if post:
            self.FEATURE_LAYER_INDICES = feature_layer_indices_post
        else:
            self.FEATURE_LAYER_INDICES = feature_layer_indices_pre
        self.device = device
        self._init_backbone()
        self._init_normalization_params()
        self._init_feature_scales()

    def _init_backbone(self) -> None:
        """Initialize and configure VGG19 backbone for feature extraction."""
        version = torchvision.__version__
        if "+" in version:
            version = version.split("+")[0]
        self.vgg = torchvision.models.vgg19(weights="IMAGENET1K_V1")
        self._prune_unused_layers()

    def _prune_unused_layers(self) -> None:
        """Remove VGG layers beyond the last feature extraction point."""
        final_layer_idx = max(self.FEATURE_LAYER_INDICES)
        if final_layer_idx < len(self.vgg.features) - 1:
            for layer in self.vgg.features[final_layer_idx + 1 :]:
                del layer

    def _init_normalization_params(self) -> None:
        """Register ImageNet mean as buffer for input normalization."""
        mean_tensor = torch.FloatTensor(self.IMAGENET_MEAN).reshape(1, 3, 1, 1)
        self.register_buffer("_normalization_mean", mean_tensor)

    def _init_feature_scales(self) -> None:
        """Register feature scaling factors for perceptual loss computation."""
        scale_tensor = torch.FloatTensor(self.FEATURE_SCALE_DIVISORS)
        self.register_buffer("_feature_scales", scale_tensor)

    def _get_multiscale_features(
        self, normalized_input: torch.Tensor
    ) -> list[torch.Tensor]:
        """
        Extract feature maps at multiple VGG layers.

        Args:
            normalized_input: Float[Tensor, "B C H W"], preprocessed for VGG

        Returns:
            List of feature tensors from specified layer indices
        """
        collected_features = []
        activation = normalized_input
        for layer_idx, layer in enumerate(self.vgg.features):
            activation = layer(activation)
            if layer_idx in self.FEATURE_LAYER_INDICES:
                collected_features.append(activation)
        return collected_features

    def _normalize_input(self, rgb_image: torch.Tensor) -> torch.Tensor:
        """Convert [0,1] RGB image to VGG input format."""
        return (rgb_image - self._normalization_mean) * 255.0

    def _l1_error_with_optional_mask(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute L1 error."""
        return torch.mean(torch.abs(prediction - target), dim=[1, 2, 3])

    def forward(
        self,
        pred_img: torch.Tensor,
        target_img: torch.Tensor,
    ) -> torch.Tensor:
        """Compute perceptual loss between prediction and target."""
        # Normalize inputs for VGG
        target_normalized = self._normalize_input(target_img)
        pred_normalized = self._normalize_input(pred_img)

        # Extract multi-scale feature representations
        target_feats = self._get_multiscale_features(target_normalized)
        pred_feats = self._get_multiscale_features(pred_normalized)

        # Pixel-level error
        e0 = self._l1_error_with_optional_mask(target_normalized, pred_normalized)

        # Feature-level errors with scaling factors from registered buffer
        feature_errors = [
            self._l1_error_with_optional_mask(target_feats[i], pred_feats[i])
            / self._feature_scales[i]
            for i in range(len(target_feats))
        ]

        # Combine all errors and normalize
        total_loss = (e0 + sum(feature_errors)) / 255.0

        return total_loss


class RenderingLossModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if self.cfg.opt.perceptual_loss_weight > 0.0:
            self.perceptual_loss_module = self._freeze_and_set_eval(
                PerceptualLoss(post=self.cfg.opt.get("perceptual_loss_post", True))
            )

    def _freeze_and_set_eval(self, module: nn.Module) -> nn.Module:
        """Freeze module parameters and set to evaluation mode."""
        for p in module.parameters():
            p.requires_grad_(False)
        return module.eval()

    def _load_lpips_distributed(self) -> nn.Module:
        """Load LPIPS model with distributed synchronization to prevent duplicate downloads."""
        is_main = (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        )
        model = None
        if is_main:
            model = lpips.LPIPS(net="vgg")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if not is_main:
            model = lpips.LPIPS(net="vgg")
        return self._freeze_and_set_eval(model)

    def forward(self, model_output, target, is_valid: torch.Tensor):
        loss_edict = self.forward_nvs_loss(model_output, target, is_valid)
        return loss_edict

    def _compute_mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        batch_size: int,
        num_views: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute MSE loss with per-example breakdown."""
        if self.cfg.opt.l2_loss_weight <= 0.0:
            zeros = torch.zeros(batch_size, device=pred.device, dtype=pred.dtype)
            return torch.tensor(1e-8, device=pred.device), zeros
        raw = F.mse_loss(pred, target, reduction="none")
        per_view = raw.mean(dim=[1, 2, 3]).reshape(batch_size, num_views)
        return per_view.mean(), per_view.mean(dim=1)

    def _compute_perceptual(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        batch_size: int,
        num_views: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute perceptual loss with per-example breakdown."""
        if self.cfg.opt.perceptual_loss_weight <= 0.0:
            zeros = torch.zeros(batch_size, device=pred.device, dtype=pred.dtype)
            return torch.tensor(0.0, device=pred.device), zeros
        per_view = self.perceptual_loss_module(pred, target).reshape(
            batch_size, num_views
        )
        return per_view.mean(), per_view.mean(dim=1)

    def _compute_psnr(self, mse: torch.Tensor) -> torch.Tensor:
        """Convert MSE to PSNR in dB."""
        return -10.0 * torch.log10(mse.detach())

    def _aggregate_weighted_loss(
        self,
        l2: torch.Tensor,
        perceptual: torch.Tensor,
    ) -> torch.Tensor:
        """Combine individual losses with configured weights."""
        return (
            self.cfg.opt.l2_loss_weight * l2
            + self.cfg.opt.perceptual_loss_weight * perceptual
        )

    def forward_nvs_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        is_valid: torch.Tensor,
    ) -> edict:
        """
        Compute image reconstruction losses.

        Args:
            pred: Predicted images [B, V, 3, H, W] in range [0, 1]
            target: Ground truth images [B, V, 3, H, W] in range [0, 1]
            is_valid: Boolean mask [B] indicating valid examples

        Returns:
            Dictionary containing loss values and metrics
        """
        b, v, c, h, w = pred.size()
        assert c == 3, f"Expected 3 channels, got {c}"

        pred_flat = einops.rearrange(pred, "b v c h w -> (b v) c h w")
        target_flat = einops.rearrange(target, "b v c h w -> (b v) c h w")

        l2, l2_per_ex = self._compute_mse(pred_flat, target_flat, b, v)
        perc, perc_per_ex = self._compute_perceptual(pred_flat, target_flat, b, v)

        # Apply validity mask
        mask = is_valid.float()  # (B,)
        l2_per_ex = l2_per_ex * mask
        perc_per_ex = perc_per_ex * mask
        # Recompute means with proper normalization
        num_valid = mask.sum().clamp(min=1)
        l2 = l2_per_ex.sum() / num_valid
        perc = perc_per_ex.sum() / num_valid

        total_loss = self._aggregate_weighted_loss(l2, perc)
        total_per_ex = self._aggregate_weighted_loss(l2_per_ex, perc_per_ex)

        return edict(
            loss=total_loss,
            l2_loss=l2,
            psnr=self._compute_psnr(l2),
            loss_per_example=total_per_ex,
            perceptual_loss=perc,
            norm_perceptual_loss=perc / l2,
        )
