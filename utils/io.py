# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os

import omegaconf


def recursive_merge_configs(
    config, default_key="__default__", base_config_path=None, root_path=None
):
    if default_key in config:
        if root_path is not None:
            config_path = os.path.join(root_path, config[default_key])
        else:
            config_path = config[default_key]
        default_config = omegaconf.OmegaConf.load(config_path)
        del config[default_key]
        merged_config = omegaconf.OmegaConf.merge(default_config, config)
        return recursive_merge_configs(
            merged_config, default_key, base_config_path, root_path
        )
    else:
        if base_config_path is not None:
            if root_path is not None:
                base_config_path = os.path.join(root_path, base_config_path)
            default_config = omegaconf.OmegaConf.load(base_config_path)
            return omegaconf.OmegaConf.merge(default_config, config)
        else:
            return config


def load_config(path, default_key="__default__", base_config_path=None, root_path=None):
    """
    Load an OmegaConf file and recursively merge other configs when __default__ is a key
    Args:
        path (str): Path to the main OmegaConf file.
        default_key (str): Key to recursively load default config form
        base_config_path (str): Path to the base OmegaConf file.
        root_path (str): Absolute path to the root of the config folder.
    Returns:
        omegaconf.DictConfig: The merged configuration.
    """
    if root_path is not None:
        path = os.path.join(root_path, path)
    config = omegaconf.OmegaConf.load(path)
    config_cli = omegaconf.OmegaConf.from_cli()

    return omegaconf.OmegaConf.merge(
        recursive_merge_configs(config, default_key, base_config_path, root_path),
        config_cli,
    )
