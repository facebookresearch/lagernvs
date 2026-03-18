# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from data.sources import dl3dv_dataset, re10k_dataset

# View selector types used by JointDataset for training.
SEQUENTIAL = "sequential"
UNORDERED = "unordered"
RANDOM = "random"
DENSE_VIDEO = "dense_video"

# Registry: name -> (dataset_class, selector_type)
_dataset_registry = {
    "re10k": (re10k_dataset.Re10kDataset, SEQUENTIAL),
    "dl3dv": (dl3dv_dataset.Dl3dvDataset, SEQUENTIAL),
}

# Backward-compatible dict: name -> dataset_class
available_datasets = {name: entry[0] for name, entry in _dataset_registry.items()}


def get_selector_type(dataset_name):
    """Look up the view selector type for a dataset name.

    Returns one of SEQUENTIAL, UNORDERED, RANDOM, DENSE_VIDEO, or None.
    Raises KeyError if the dataset name is not registered.
    """
    return _dataset_registry[dataset_name][1]
