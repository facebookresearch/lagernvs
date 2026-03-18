# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from data import dataset_factory, view_selector
from eval.quantitative import run_cond_eval
from rendering_loss import RenderingLossModule
from utils import misc
from utils.distributed_sampler import NoDropDistributedSampler


def get_loss_fn(cfg, device):
    loss_fn = RenderingLossModule(cfg).to(device)
    return loss_fn


def warmup_constant_lambda(iter_idx, warmup_steps):
    if iter_idx < warmup_steps:
        return float(iter_idx) / float(max(1, warmup_steps))
    return 1.0


def get_per_dataset_loss(loss_per_example, image_ids):
    # assume that loss_per_example has already been detached

    # Gather loss_per_example and image_ids from all devices for distributed training
    if dist.is_initialized() and dist.get_world_size() > 1:
        world_size = dist.get_world_size()

        # Use all_gather_object for both losses and image_ids to handle variable batch sizes
        # This is simpler and sufficient for logging purposes
        gathered_losses = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_losses, loss_per_example.cpu().tolist())

        # Convert back to tensor and concatenate
        all_losses_list = []
        for batch_losses in gathered_losses:
            all_losses_list.extend(batch_losses)
        all_losses = torch.tensor(all_losses_list, device=loss_per_example.device)

        # Gather image_ids from all devices
        # image_ids is a list where image_ids[0] contains image_ids of the first view
        gathered_image_ids = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_image_ids, image_ids[0])

        # Flatten the gathered image_ids
        all_image_ids = []
        for batch_image_ids in gathered_image_ids:
            all_image_ids.extend(batch_image_ids)
    else:
        # Single device case
        all_losses = loss_per_example
        all_image_ids = image_ids[0]

    # Extract dataset names from image IDs
    dataset_names = [n.split("_")[0] for n in all_image_ids]
    dataset_losses = {}
    dataset_counts = {}

    for ex_loss, dataset_name in zip(all_losses, dataset_names):
        try:
            dataset_losses[dataset_name] += ex_loss
        except KeyError:
            dataset_losses[dataset_name] = ex_loss
            dataset_counts[dataset_name] = 0
        dataset_counts[dataset_name] += 1

    for dataset_name in dataset_losses.keys():
        dataset_losses[dataset_name] /= dataset_counts[dataset_name]

    return dataset_losses


def warmup_cosine_lambda(
    iter_idx,
    warmup_steps,
    hold_steps,
    num_iter_total,
    cosine_max_range=1.0,
    cosine_min_range=0.0,
):
    if iter_idx < warmup_steps:
        return float(iter_idx) / float(max(1, warmup_steps))
    elif iter_idx <= warmup_steps + hold_steps:
        return 1.0
    else:
        decay_iters = num_iter_total - warmup_steps - hold_steps
        decay_iter_idx = iter_idx - warmup_steps - hold_steps
        # cosine factor between 0 and 1
        cosine_factor_0_to_1 = 0.5 * np.cos(np.pi * decay_iter_idx / decay_iters) + 0.5
        cosine_factor = cosine_min_range + cosine_factor_0_to_1 * (
            cosine_max_range - cosine_min_range
        )
        return cosine_factor


def warmup_step_lambda(iter_idx, warmup_steps, hold_steps, step_factor=0.1):
    if iter_idx < warmup_steps:
        return float(iter_idx) / float(max(1, warmup_steps))
    elif iter_idx <= warmup_steps + hold_steps:
        return 1.0
    else:
        return step_factor


def get_lr_lambda(cfg):
    if cfg.opt.lr_scheduler == "constant":
        return partial(warmup_constant_lambda, warmup_steps=cfg.opt.warmup_steps)
    elif cfg.opt.lr_scheduler == "cosine":
        return partial(
            warmup_cosine_lambda,
            warmup_steps=cfg.opt.warmup_steps,
            hold_steps=cfg.opt.hold_steps,
            num_iter_total=cfg.opt.lr_scheduler_iter,
            cosine_min_range=cfg.opt.cosine_min_range,
        )
    elif cfg.opt.lr_scheduler == "step":
        return partial(
            warmup_step_lambda,
            warmup_steps=cfg.opt.warmup_steps,
            hold_steps=cfg.opt.hold_steps,
        )


def create_optimizer(
    model,
    weight_decay,
    learning_rate,
    betas,
    freeze_vggt=True,
):
    """Create optimizer with weight decay applied selectively.

    Args:
        model: PyTorch model
        weight_decay: Weight decay for parameters (not applied to 1D params and scene_tokens)
        learning_rate: Learning rate for all parameters
        betas: Adam betas
        freeze_vggt: If True, don't optimize vggt parameters

    Returns:
        optimizer, optimized_param_dict, all_param_dict
    """
    # start with all of the candidate parameters
    all_param_dict = {name: param for name, param in model.named_parameters()}
    # filter out those that do not require grad
    optimized_param_dict = {}
    for name, param in all_param_dict.items():
        if param.requires_grad:
            if freeze_vggt:
                if "vggt" not in name:
                    optimized_param_dict[name] = param
                else:
                    # if vggt is supposed to be frozen, we do not optimize it
                    continue
            else:
                optimized_param_dict[name] = param

    # Split parameters into 2 groups based on weight decay
    decay_params, nodecay_params = [], []
    decay_names, nodecay_names = [], []

    for name, param in optimized_param_dict.items():
        should_decay = not (param.dim() == 1 or "scene_tokens" in name)

        if should_decay:
            decay_params.append(param)
            decay_names.append(name)
        else:
            nodecay_params.append(param)
            nodecay_names.append(name)

    optim_groups = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": nodecay_params,
            "weight_decay": 0.0,
        },
    ]

    # use fused AdamW optimizer by default.
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas, fused=True
    )

    # Print Model Information
    if dist.get_rank() == 0:

        def get_module_name(name):
            parts = name.split(".")
            if len(parts) > 2 and parts[0] == "module":
                return parts[1] + "." + parts[2]
            return parts[0]  # Fallback to first part if no 'module.' prefix

        print(
            f"Optimizer: AdamW, learning_rate: {learning_rate}, "
            f"weight decay: {weight_decay}, betas: {betas}"
        )

        # Number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in optimized_param_dict.values())
        optim_module_names = sorted(
            set(get_module_name(name) for name in optimized_param_dict.keys())
        )
        frozen_module_names = sorted(
            set(
                get_module_name(name)
                for name in set(all_param_dict.keys())
                - set(optimized_param_dict.keys())
            )
        )

        print(
            f"Total parameters: {format_number(total_params)}, Trainable parameters: {format_number(trainable_params)}"
        )
        print(f"Optimized parameters: {optim_module_names}")
        print(f"Frozen parameters: {frozen_module_names}")

        print(f"Parameters with weight decay: {decay_names}")
        print(f"Parameters without weight decay: {nodecay_names}")

    return optimizer, optimized_param_dict, all_param_dict


def format_number(num):
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    return str(num)


def process_gradients(
    loss,
    optimized_param_dict,
    optim_param_list,
    grad_clip_norm,
    allowed_gradnorm_factor=5,
):
    """Handle gradient processing: sanitization, clipping, and skip determination.

    Args:
        loss: Loss tensor to validate for finite values.
        optimized_param_dict: Dict of {name: param} for gradient sanitization.
        optim_param_list: List of parameters for norm computation and clipping.
        grad_clip_norm: Maximum gradient norm (no clipping if <= 0).
        allowed_gradnorm_factor: Factor for computing the skip threshold.

    Returns:
        bool: True if optimizer step should be skipped, False otherwise.
    """
    # Exit early on invalid loss
    if not torch.isfinite(loss):
        print("NaN or Inf loss detected, skipping this iteration")
        loss.data.zero_()
        return True

    # Sanitize non-finite gradient values
    _fix_nonfinite_gradients(optimized_param_dict)

    # Compute norm and clip if enabled
    if grad_clip_norm > 0:
        norm_tensor = torch.nn.utils.clip_grad_norm_(
            optim_param_list, max_norm=grad_clip_norm
        )
        # Skip if gradient norm exceeds threshold (comparison stays on GPU)
        if norm_tensor > grad_clip_norm * allowed_gradnorm_factor:
            print("WARNING: grad norm too large, skipping optimizer step")
            return True

    return False


def _fix_nonfinite_gradients(param_dict):
    """Replace NaN/Inf gradient values with safe defaults in-place."""
    for param in param_dict.values():
        if param.requires_grad and param.grad is not None:
            param.grad.nan_to_num_(nan=0.0, posinf=1e-6, neginf=-1e-6)


def log_training_metrics(
    log_writer,
    loss_dict,
    image_ids,
    scheduler,
    iter_idx,
):
    """Log training metrics to tensorboard.

    Args:
        log_writer: Tensorboard SummaryWriter (can be None on non-main processes).
        loss_dict: Loss dictionary containing loss, psnr, l2_loss, loss_per_example.
        image_ids: Image IDs for per-dataset loss computation.
        scheduler: Learning rate scheduler.
        iter_idx: Current iteration index.
    """
    print(f"Iter {iter_idx} loss {loss_dict.loss.item()}")

    # get_per_dataset_loss uses dist.all_gather_object which is a collective
    # operation requiring ALL ranks to participate. Must be called before early return.
    per_dataset_loss = get_per_dataset_loss(
        loss_dict.loss_per_example.detach(), image_ids
    )

    if not misc.is_main_process():
        return

    log_writer.add_scalar("train/loss", loss_dict.loss.item(), iter_idx)
    log_writer.add_scalar("train/psnr", loss_dict.psnr.detach().item(), iter_idx)
    log_writer.add_scalar("train/l2_loss", loss_dict.l2_loss.detach().item(), iter_idx)
    log_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], iter_idx)

    for dataset_name, dataset_loss in per_dataset_loss.items():
        log_writer.add_scalar(
            f"train/dataset_{dataset_name}/loss", dataset_loss.item(), iter_idx
        )


def get_next_batch(data_iter, dataset, epoch_idx, device):
    """Handle epoch transitions and move batch to device."""
    try:
        data = next(data_iter)
    except StopIteration:
        print("New dataloader...")
        epoch_idx += 1
        dataloader = dataset.get_loader(epoch_idx)
        data_iter = iter(dataloader)
        data = next(data_iter)

    images, rays, image_ids, cam_token, is_valid, num_cond_views = data

    return (
        (
            images.to(device),
            rays.to(device),
            image_ids,
            cam_token.to(device),
            is_valid.to(device),
            num_cond_views,
        ),
        data_iter,
        epoch_idx,
    )


def run_quantitative_evaluation(
    cfg,
    model,
    device,
    iter_idx,
    writer,
    batch_size_per_device,
):
    """Run quantitative evaluation on all configured test datasets and log metrics."""
    print(
        f"Running conditional generation for eval on rank {misc.get_rank()}", force=True
    )
    dist.barrier()
    model.eval()

    for (
        dataset_name,
        dataset_name_log,
        zero_out_cam_cond_p,
        test_view_indices_path,
        test_num_cond_views,
    ) in zip(
        cfg.test_data.dataset_names,
        cfg.test_data.dataset_names_log,
        cfg.test_data.zero_out_cam_cond_p,
        cfg.test_data.test_view_indices_paths,
        cfg.test_data.num_cond_views,
    ):
        print(
            f"Evaluating {dataset_name} named {dataset_name_log} "
            f"with zero-out probability {zero_out_cam_cond_p} "
            f"from test path {test_view_indices_path}"
        )

        quant_test_view_selector = view_selector.FixedViewSelector(
            test_view_indices_path
        )
        quant_test_dataset = dataset_factory.available_datasets[dataset_name](
            view_selector=quant_test_view_selector,
            split="test",
            im_size_hw=cfg.data.im_size_hw,
            num_cond_views=test_num_cond_views,
            zero_out_cam_cond_p=zero_out_cam_cond_p,
        )
        quant_test_sampler = NoDropDistributedSampler(
            quant_test_dataset, shuffle=False, drop_last=False, seed=cfg.seed
        )
        quant_test_dataloader = torch.utils.data.DataLoader(
            quant_test_dataset,
            batch_size=batch_size_per_device,
            sampler=quant_test_sampler,
            num_workers=8,
            pin_memory=True,
        )

        gathered_eval_metrics, _ = run_cond_eval(
            model,
            device,
            test_num_cond_views,
            quant_test_dataloader,
            rank=misc.get_rank(),
            world_size=misc.get_world_size(),
        )

        if misc.is_main_process():
            writer.add_scalar(
                f"eval/{dataset_name_log}/psnr", gathered_eval_metrics["psnr"], iter_idx
            )
            writer.add_scalar(
                f"eval/{dataset_name_log}/ssim", gathered_eval_metrics["ssim"], iter_idx
            )
            writer.add_scalar(
                f"eval/{dataset_name_log}/lpips",
                gathered_eval_metrics["lpips"],
                iter_idx,
            )

        print(
            f"Finished conditional generation for eval on rank {misc.get_rank()}",
            force=True,
        )

    model.train()
