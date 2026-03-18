# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Callable, Optional

import numpy as np
from data import joint_dataset
from data.worker_fn import get_worker_init_fn
from torch.utils.data import DataLoader, DistributedSampler, Sampler


class DynamicTorchDataset:
    def __init__(
        self,
        cfg,
        max_bs_for_2_cond: int,
        num_workers: int,
        shuffle: bool,
        pin_memory: bool,
        split: str,
        drop_last: bool = True,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
        persistent_workers: bool = False,
        seed: int = 42,
    ) -> None:
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        self.persistent_workers = persistent_workers
        self.seed = seed

        # Instantiate the dataset
        self.dataset = joint_dataset.JointDataset(
            subdataset_list=cfg.data.subdataset_list,
            split=split,
            im_size_hw=cfg.data.im_size_hw,
            num_cond_views=None,
            zero_out_cam_cond_p=cfg.data.zero_out_cam_cond_p,
            target_has_input_p=cfg.opt.target_has_input_p,
        )

        # Extract aspect ratio and image number ranges from the configuration
        self.aspect_ratio_range = cfg.data.aspect_ratio_range
        self.num_cond_views_range = cfg.data.num_cond_views_aug

        # Validate the aspect ratio and image number ranges
        if (
            len(self.aspect_ratio_range) != 2
            or self.aspect_ratio_range[0] > self.aspect_ratio_range[1]
        ):
            raise ValueError(
                f"aspect_ratio_range must be [min, max] with min <= max, got {self.aspect_ratio_range}"
            )
        if (
            len(self.num_cond_views_range) != 2
            or self.num_cond_views_range[0] < 1
            or self.num_cond_views_range[0] > self.num_cond_views_range[1]
        ):
            raise ValueError(
                f"num_cond_views_range must be [min, max] with 1 <= min <= max, got {self.num_cond_views_range}"
            )

        # Create samplers
        self.sampler = DynamicDistributedSampler(
            self.dataset, seed=seed, shuffle=shuffle
        )
        view_to_sample_prob = {
            int(v.split(",")[0]): float(v.split(",")[1])
            for v in cfg.data.view_to_sample_prob
        }
        self.batch_sampler = DynamicBatchSampler(
            self.sampler,
            self.aspect_ratio_range,
            self.num_cond_views_range,
            num_tgt_views=cfg.data.num_tgt_views,
            view_to_sample_prob=view_to_sample_prob,
            seed=seed,
            max_bs_for_2_cond=max_bs_for_2_cond,
            max_tgt_views_for_2_cond=cfg.data.num_views - 2,
        )

    def get_loader(self, epoch):
        print("Building dynamic dataloader with seed:", self.seed)

        # Set the epoch for the sampler
        self.sampler.set_epoch(epoch)
        if hasattr(self.dataset, "epoch"):
            self.dataset.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

        # Create and return the dataloader
        return DataLoader(
            self.dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_sampler=self.batch_sampler,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            timeout=600 if self.num_workers > 0 else 0,  # 10 min timeout for workers
            worker_init_fn=get_worker_init_fn(
                seed=self.seed,
                num_workers=self.num_workers,
                epoch=epoch,
                worker_init_fn=self.worker_init_fn,
            ),
        )


class DynamicBatchSampler(Sampler):
    """
    A custom batch sampler that dynamically adjusts batch size, aspect ratio, and image number
    for each sample. Batches within a sample share the same aspect ratio and image number.
    """

    def __init__(
        self,
        sampler,
        aspect_ratio_range,
        num_cond_views_range,
        num_tgt_views,
        view_to_sample_prob,
        epoch=0,
        seed=42,
        max_bs_for_2_cond=48,
        max_tgt_views_for_2_cond=8,
    ):
        """
        Initializes the dynamic batch sampler.

        Args:
            sampler: Instance of DynamicDistributedSampler.
            aspect_ratio_range: List containing [min_aspect_ratio, max_aspect_ratio].
            num_cond_views_range: List containing [min_cond_views, max_cond_views] per sample.
            num_tgt_views: Number of target views to use.
            epoch: Current epoch number.
            seed: Random seed for reproducibility.
            max_bs_for_2_cond: Maximum batch size for 2 conditioning views.
        """
        self.sampler = sampler
        self.aspect_ratio_range = aspect_ratio_range
        self.num_cond_views_range = num_cond_views_range
        self.max_tgt_views_for_2_cond = max_tgt_views_for_2_cond
        self.num_tgt_views = num_tgt_views
        self.view_to_sample_prob = view_to_sample_prob
        self.rng = random.Random()
        # Maximum image number per GPU
        self.max_bs_for_2_cond = max_bs_for_2_cond

        # Uniformly sample from the range of possible image numbers
        # For any image number, the weight is 1.0 (uniform sampling). You can set any different weights here.
        if num_cond_views_range[1] <= 4:
            self.image_num_weights = {
                num_images: 1.0
                for num_images in range(
                    num_cond_views_range[0], num_cond_views_range[1] + 1
                )
            }
        else:
            image_num_weights = view_to_sample_prob
            self.image_num_weights = {}
            for num_images in range(
                num_cond_views_range[0], num_cond_views_range[1] + 1
            ):
                self.image_num_weights[num_images] = image_num_weights[num_images]

        for random_num_cond_views in range(
            num_cond_views_range[0], num_cond_views_range[1] + 1
        ):
            max_batch_size = max_bs_for_2_cond * 2 * self.max_tgt_views_for_2_cond
            batch_size = (max_batch_size // self.num_tgt_views) / max(
                random_num_cond_views, 2
            )
            print(
                "For num_cond_views = ",
                random_num_cond_views,
                " batch size = ",
                batch_size,
                " per GPU",
            )

        # Possible image numbers, e.g., [2, 3, 4, ..., 24]
        self.possible_nums = np.array(
            [
                n
                for n in self.image_num_weights.keys()
                if self.num_cond_views_range[0] <= n <= self.num_cond_views_range[1]
            ]
        )

        # Normalize weights for sampling
        weights = [self.image_num_weights[n] for n in self.possible_nums]
        self.normalized_weights = np.array(weights) / sum(weights)

        # Set the epoch for the sampler
        self.set_epoch(epoch + seed)

    def set_epoch(self, epoch):
        """
        Sets the epoch for this sampler, affecting the random sequence.

        Args:
            epoch: The epoch number.
        """
        self.sampler.set_epoch(epoch)
        self.epoch = epoch
        self.rng.seed(epoch * 100)

    def __iter__(self):
        """
        Yields batches of samples with synchronized dynamic parameters.

        Returns:
            Iterator yielding batches of indices with associated parameters.
        """
        sampler_iterator = iter(self.sampler)

        while True:
            try:
                # Sample random image number and aspect ratio
                random_num_cond_views = int(
                    np.random.choice(self.possible_nums, p=self.normalized_weights)
                )
                random_aspect_ratio = round(
                    # sample on a logarithm and exponentiate to uniformly sample the
                    # mutliplicative variable
                    np.exp(
                        self.rng.uniform(
                            np.log(self.aspect_ratio_range[0]),
                            np.log(self.aspect_ratio_range[1]),
                        )
                    ),
                    2,
                )
                num_views = random_num_cond_views + self.num_tgt_views
                # Update sampler parameters
                self.sampler.update_parameters(
                    aspect_ratio=random_aspect_ratio,
                    num_cond_views=random_num_cond_views,
                    num_views=num_views,
                )

                # Calculate batch size based on max images per GPU and current image number
                max_batch_size = (
                    self.max_bs_for_2_cond * 2 * self.max_tgt_views_for_2_cond
                )
                batch_size = (max_batch_size // self.num_tgt_views) / max(
                    random_num_cond_views, 2
                )
                # batch_size = self.max_bs_for_2_cond * 2 / random_num_cond_views
                batch_size = np.floor(batch_size).astype(int)
                batch_size = max(1, batch_size)  # Ensure batch size is at least 1

                # Collect samples for the current batch
                current_batch = []
                for _ in range(batch_size):
                    try:
                        item = next(
                            sampler_iterator
                        )  # item is (idx, aspect_ratio, num_cond_views)
                        current_batch.append(item)
                    except StopIteration:
                        break  # No more samples

                if not current_batch:
                    break  # No more data to yield

                yield current_batch

            except StopIteration:
                break  # End of sampler's iterator

    def __len__(self):
        # Return a large dummy length
        return 1000000


class DynamicDistributedSampler(DistributedSampler):
    """
    Extends PyTorch's DistributedSampler to include dynamic aspect_ratio and image_num
    parameters, which can be passed into the dataset's __getitem__ method.
    """

    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.aspect_ratio = None
        self.num_views = None
        self.num_cond_views = None

    def __iter__(self):
        """
        Yields a sequence of (index, image_num, aspect_ratio).
        Relies on the parent class's logic for shuffling/distributing
        the indices across replicas, then attaches extra parameters.
        """
        indices_iter = super().__iter__()

        for idx in indices_iter:
            yield (
                idx,
                self.num_views,
                self.num_cond_views,
                self.aspect_ratio,
            )

    def update_parameters(self, aspect_ratio, num_cond_views, num_views):
        """
        Updates dynamic parameters for each new epoch or iteration.

        Args:
            aspect_ratio: The aspect ratio to set.
            image_num: The number of images to set.
        """
        self.aspect_ratio = aspect_ratio
        self.num_views = num_views
        self.num_cond_views = num_cond_views
