# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os

import torch
from data import dataset_factory, view_selector
from data.normalization import NormalizationMode
from eval.export import save_eval_scores, save_video_batch_dist
from eval.quantitative import run_cond_eval
from eval.utils import set_seed
from models.encoder_decoder import EncDec_VitB8
from utils import misc
from utils.checkpoint import load_checkpoint
from utils.distributed_sampler import NoDropDistributedSampler
from vis import render_chunked
from omegaconf import OmegaConf


def run_quantitative_eval(
    cfg,
    model: torch.nn.Module,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    num_cond_views: int,
    dataset_name_log: str,
    iteration_idx: int,
) -> None:
    eval_image_out_dir = os.path.join(
        cfg.log_dir, dataset_name_log, f"images_iter_{iteration_idx:06d}"
    )
    if misc.is_main_process():
        os.makedirs(eval_image_out_dir, exist_ok=True)
    torch.distributed.barrier()

    scores, all_scores = run_cond_eval(
        model,
        device,
        num_cond_views,
        dataloader,
        rank=misc.get_rank(),
        world_size=misc.get_world_size(),
        save_path=eval_image_out_dir,
        eval_resolution=cfg.eval_data.get("eval_resolution", None),
    )

    save_eval_scores(
        cfg.log_dir,
        dataset_name_log,
        iteration_idx,
        scores,
        all_scores,
        eval_resolution=cfg.eval_data.get("eval_resolution", None),
        suffix=cfg.eval_data.get("suffix", None),
    )


def run_video_eval(
    cfg,
    model: torch.nn.Module,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    num_cond_views: int,
    dataset_name_log: str,
    iteration_idx: int,
) -> None:
    """Render and save evaluation videos."""
    video_out_dir = os.path.join(
        cfg.log_dir, dataset_name_log, f"videos_iter_{iteration_idx:06d}"
    )
    if misc.is_main_process():
        os.makedirs(video_out_dir, exist_ok=True)
    torch.distributed.barrier()

    print(
        f"Starting video saving and {cfg.eval_data.video_path_type} camera video rendering"
    )

    for images, rays, image_names, cam_token, _is_valid in dataloader:
        with torch.no_grad():
            images = images.to(device)
            rays = rays.to(device)
            cam_token = cam_token.to(device)

            video_out = render_chunked(
                model,
                (images, rays, cam_token),
                num_cond_views=num_cond_views,
            )

            save_video_batch_dist(
                video_out,
                video_out_dir,
                image_names,
            )

    print("Finished video saving and camera loop rendering")


def create_eval_dataloader(cfg, is_video):
    quant_test_view_selector = view_selector.FixedViewSelector(
        cfg.eval_data.test_view_indices_path
    )

    if is_video:
        video_length = cfg.data.eval_video_length
        video_path_type = cfg.eval_data.video_path_type
        # batch size 1 per device for video rendering
        batch_size_per_device = 1
    else:
        batch_size_per_device = cfg.opt.batch_size // misc.get_world_size()
        # dummy video parameters, video length 0 will not create videos
        video_length = 0
        video_path_type = "linear_interp"

    test_dataset = dataset_factory.available_datasets[cfg.eval_data.dataset_name](
        view_selector=quant_test_view_selector,
        split="test",
        im_size_hw=cfg.data.im_size_hw,
        num_cond_views=cfg.eval_data.num_cond_views,
        zero_out_cam_cond_p=cfg.eval_data.zero_out_cam_cond_p,
        video_length=video_length,
        video_path_type=video_path_type,
        normalization_mode=NormalizationMode(cfg.eval_data.normalization_mode),
    )

    quant_test_sampler = NoDropDistributedSampler(
        test_dataset, shuffle=False, drop_last=False, seed=cfg.seed
    )
    dist_test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size_per_device,
        sampler=quant_test_sampler,
        num_workers=8,
        pin_memory=True,
    )
    return dist_test_dataloader


def main(cfg) -> None:
    cfg.log_dir = os.path.join(cfg.log_dir, cfg.exp_name)
    print("Running eval")
    print(
        f"Running setup on rank {os.environ['RANK']} with world size {os.environ['WORLD_SIZE']}"
    )
    misc.init_distributed_mode(cfg)
    device = torch.device(cfg.gpu)
    torch.cuda.set_device(device)

    seed = cfg.seed + misc.get_rank()
    set_seed(seed)

    dict_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    print(dict_cfg)

    model = EncDec_VitB8(
        pretrained_vggt=False,  # Pre-trained models are meant for training only
        pretrained_patch_embed=False,  # at eval we load the saved checkpoint
        attention_to_features_type=cfg.model.attention_to_features_type,
    ).to(device)

    if cfg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.gpu], find_unused_parameters=True
        )

    iteration_idx = load_checkpoint(
        cfg,
        model,
        None,
        None,
        test_only=True,
        strict=True,
    )
    print("Loaded checkpoint")
    model.eval()

    quant_dataloader = create_eval_dataloader(cfg, is_video=False)

    run_quantitative_eval(
        cfg,
        model,
        device,
        quant_dataloader,
        cfg.eval_data.num_cond_views,
        cfg.eval_data.dataset_name_log,
        iteration_idx,
    )
    # comment out if video rendering not required, can be slow
    # when executed for the whole dataset
    video_dataloader = create_eval_dataloader(cfg, is_video=True)

    run_video_eval(
        cfg,
        model,
        device,
        video_dataloader,
        cfg.eval_data.num_cond_views,
        cfg.eval_data.dataset_name_log,
        iteration_idx,
    )


if __name__ == "__main__":
    import argparse
    from utils.io import load_config

    parser = argparse.ArgumentParser(description="Evaluate LagerNVS model")
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
