# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
import random

import numpy as np
import torch
from data.dynamic_dataloader import DynamicTorchDataset
from models.encoder_decoder import EncDec_VitB8
from train_utils import (
    create_optimizer,
    get_loss_fn,
    get_lr_lambda,
    get_next_batch,
    log_training_metrics,
    process_gradients,
    run_qualitative_visualizations,
    run_quantitative_evaluation,
)
from utils import misc
from utils.checkpoint import load_checkpoint, save_checkpoint
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def _train_step(
    model,
    optimizer,
    scheduler,
    loss_fn,
    train_batch,
    optimized_param_dict,
    optim_param_list,
    grad_clip_norm,
):
    """Single training iteration: forward, backward, gradient processing, optimizer step."""
    images, rays, image_ids_train, cam_token, is_valid, num_cond_views = train_batch

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        output_rgb = model(
            images,
            rays,
            cam_token,
            num_cond_views=num_cond_views[0],
        )
        loss_dict = loss_fn(
            output_rgb[:, num_cond_views[0] :, :, :, :],
            images[:, num_cond_views[0] :, :, :, :],
            is_valid,
        )

    loss_dict.loss.backward()

    skip_optimizer_step = process_gradients(
        loss_dict.loss,
        optimized_param_dict,
        optim_param_list,
        grad_clip_norm,
    )

    if not skip_optimizer_step:
        optimizer.step()

    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    return loss_dict, image_ids_train


def main(cfg) -> None:
    cfg.log_dir = os.path.join(cfg.log_dir, cfg.exp_name)
    # Set up ddp
    print(
        f"Running setup on rank {os.environ['RANK']} with world size {os.environ['WORLD_SIZE']}"
    )
    misc.init_distributed_mode(cfg)
    # Ensure proper device setting
    device = torch.device(cfg.gpu)
    torch.cuda.set_device(device)

    # fix the seed for reproducibility
    seed = cfg.seed + misc.get_rank()
    set_seed(seed)

    dict_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    print(dict_cfg)

    if misc.is_main_process():
        misc.makedirs(cfg.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=cfg.log_dir)
    else:
        log_writer = None

    # Initialize LPIPS model on all ranks for distributed evaluation
    loss_fn = get_loss_fn(cfg, device)

    model = EncDec_VitB8(
        freeze_vggt=cfg.opt.freeze_vggt,
        pretrained_vggt=cfg.model.pretrained_vggt,
        attention_to_features_type=cfg.model.attention_to_features_type,
        pretrained_patch_embed=cfg.model.get("pretrained_patch_embed", False),
    ).to(device)

    if cfg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.gpu], find_unused_parameters=cfg.opt.freeze_vggt
        )

    # Creates an optimizer that uses weight decay on all
    # layers apart from the normalization layer
    optimizer, optimized_param_dict, _ = create_optimizer(
        model,
        cfg.opt.weight_decay,
        cfg.opt.lr,
        cfg.opt.betas,
        cfg.opt.freeze_vggt,
    )
    optim_param_list = list(optimized_param_dict.values())

    lr_lambda = get_lr_lambda(cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    optimizer.zero_grad()
    model.train()

    # Create checkpoint directory
    if misc.is_main_process():
        checkpoint_dir = os.path.join(cfg.log_dir, "checkpoints")
        misc.makedirs(checkpoint_dir, exist_ok=True)

    # cfg.opt.batch_size sets the global batch size
    epoch_idx = 0
    batch_size_per_device = max(1, cfg.opt.batch_size // misc.get_world_size())
    dataset = DynamicTorchDataset(
        cfg,
        max_bs_for_2_cond=batch_size_per_device,
        num_workers=12,
        shuffle=True,
        pin_memory=True,
        split="train",
        seed=cfg.seed,
    )
    dataloader = dataset.get_loader(epoch_idx)
    data_iter = iter(dataloader)

    # Distributed test dataloader for evaluation
    batch_size_per_device_test = max(
        1, cfg.opt.batch_size_test // misc.get_world_size()
    )
    dist_test_dataset = DynamicTorchDataset(
        cfg,
        max_bs_for_2_cond=batch_size_per_device_test,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
        split="test",
        seed=cfg.seed,
    )
    dist_test_dataloader = dist_test_dataset.get_loader(epoch_idx)
    test_data_iter = iter(dist_test_dataloader)
    test_epoch_idx = 0

    # Get fixed test batch for consistent visualization
    fixed_batch, test_data_iter, test_epoch_idx = get_next_batch(
        test_data_iter, dist_test_dataset, test_epoch_idx, device
    )
    images_vis, rays_vis, _, cam_token_vis, _, num_cond_views_vis = fixed_batch
    fixed_eval_inputs = (
        images_vis,
        rays_vis,
        cam_token_vis,
        num_cond_views_vis,
    )

    # Load checkpoint if available in the log directory. Used for preemption.
    # Also handles loading pretrained checkpoint for fine-tuning if specified in config.
    start_iter = load_checkpoint(cfg, model, optimizer, scheduler)

    print(f"Training model on rank {cfg.rank}", force=True)
    for iter_idx in range(start_iter, cfg.opt.num_iter_total):
        train_batch, data_iter, epoch_idx = get_next_batch(
            data_iter, dataset, epoch_idx, device
        )

        loss_dict, image_ids_train = _train_step(
            model,
            optimizer,
            scheduler,
            loss_fn,
            train_batch,
            optimized_param_dict,
            optim_param_list,
            cfg.opt.grad_clip_norm,
        )

        if iter_idx % cfg.eval.log_iter == 0 or iter_idx < 201:
            log_training_metrics(
                log_writer,
                loss_dict,
                image_ids_train,
                scheduler,
                iter_idx,
            )

        if iter_idx % cfg.eval.vis_iter == 0:
            print("Running visualization iteration")
            test_data_iter, test_epoch_idx = run_qualitative_visualizations(
                log_writer,
                model,
                device,
                iter_idx,
                train_batch,
                fixed_eval_inputs,
                test_data_iter,
                dist_test_dataset,
                test_epoch_idx,
            )

        if iter_idx % cfg.eval.eval_iter == 0 and iter_idx > 0:
            if cfg.eval.measure_time:
                measure_time_b1(log_writer, model, fixed_eval_inputs, iter_idx)
            run_quantitative_evaluation(
                cfg,
                model,
                device,
                iter_idx,
                log_writer,
                batch_size_per_device_test,
            )

        # Save checkpoint at regular intervals
        if iter_idx % cfg.eval.ckpt_iter == 0 and iter_idx > 0:
            save_checkpoint(cfg, model, optimizer, scheduler, iter_idx)

        if iter_idx % 1000 == 0 and iter_idx > 0:
            # Save checkpoint at regular intervals
            save_checkpoint(
                cfg, model, optimizer, scheduler, iter_idx, only_latest=True
            )

    # Save final checkpoint
    save_checkpoint(cfg, model, optimizer, scheduler, iter_idx)

    # When done logging
    if misc.is_main_process():
        log_writer.close()


@torch.no_grad()
def measure_time_b1(writer, model, cond_inputs, iter_idx):
    print("Running inference speed evaluation")
    (
        cond_image_inputs,
        rays,
        cam_token,
        num_cond_views,
    ) = cond_inputs
    # run inference for one batch, one view to measure speed
    im_sliced = cond_image_inputs[:1, : num_cond_views[0] + 1, ...]
    cam_sliced = rays[:1, : num_cond_views[0] + 1, ...]
    cam_token_sliced = cam_token[:1, : num_cond_views[0] + 1, ...]
    # temp_timesteps_sliced = temp_timesteps[:1, : num_cond_views[0] + 1]

    times = []
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for i in range(110):
            _, t_i = model(
                im_sliced,
                cam_sliced,
                cam_token_sliced,
                num_cond_views=num_cond_views[0],
                timeit=True,
            )
            if i < 10:
                continue
            times.append(t_i)
    t_avg = sum(times) / len(times)
    if misc.is_main_process():
        writer.add_scalar(
            f"eval/timeit_{num_cond_views[0]}", t_avg, global_step=iter_idx
        )

if __name__ == "__main__":
    import argparse
    from utils.io import load_config

    parser = argparse.ArgumentParser(description="Train LagerNVS model")
    parser.add_argument(
        "-e",
        "--exp-name",
        required=True,
        help="experiment name",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        required=True,
        help="path to config file",
    )
    args, _ = parser.parse_known_args()

    import os
    root_path = os.path.dirname(__file__)
    config = load_config(args.config_file, base_config_path=None, root_path=root_path)
    config.exp_name = args.exp_name

    print(config)
    main(config)
