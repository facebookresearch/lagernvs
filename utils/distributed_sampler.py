# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import math
from typing import Iterator, Optional, TypeVar

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler

T_co = TypeVar("T_co", covariant=True)


class NoDropDistributedSampler(Sampler[T_co]):
    """
    Custom distributed sampler that ensures no test samples are dropped
    and no additional indices are added.

    Unlike the standard DistributedSampler, this sampler:
    1. Does not add padding indices when drop_last=False
    2. Ensures all samples are processed exactly once across all processes
    3. Handles uneven distribution gracefully by giving some processes fewer samples

    This is particularly useful for evaluation where you want to process
    every sample exactly once without duplicates or synthetic padding.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

        if self.drop_last:
            # Standard behavior: drop samples to make even distribution
            self.num_samples = math.floor(self.total_size / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(self.dataset)
            # Custom behavior: distribute samples as evenly as possible
            # Some processes may get one fewer sample than others
            base_samples = self.total_size // self.num_replicas
            extra_samples = self.total_size % self.num_replicas

            # First 'extra_samples' processes get one additional sample
            if self.rank < extra_samples:
                self.num_samples = base_samples + 1
            else:
                self.num_samples = base_samples

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.total_size, generator=g).tolist()
        else:
            indices = list(range(self.total_size))

        # No special treatment for drop_last. Total size has been decreased
        # if drop_last=True. If drop_last=False, some processes will get 1 sample
        # extra, this should come from the slicing as implemented below.
        # Corectness of the number of indices received by each process will
        # be tested with the assert below.
        indices = indices[self.rank : self.total_size : self.num_replicas]

        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
