import logging

import torch.utils.data as data
from data import view_selector
from data.dataset_factory import available_datasets, get_selector_type, SEQUENTIAL
from data.normalization import NormalizationMode

logger = logging.getLogger(__name__)


def _create_view_selector(
    selector_type,
    view_sampler_range,
    target_has_input_p,
    expansion_factor,
):
    """Create a view selector instance based on the selector type."""
    if selector_type != SEQUENTIAL:
        raise ValueError(f"Only SEQUENTIAL selector type is supported, got {selector_type!r}")
    return view_selector.ExpandedLinearViewSelector(
        view_sampler_range[0],
        view_sampler_range[1],
        target_has_input_p,
        expansion_factor,
    )


class JointDataset(data.Dataset):
    def __init__(
        self,
        subdataset_list,
        split,
        im_size_hw,
        num_cond_views,
        zero_out_cam_cond_p,
        target_has_input_p,
    ):
        super().__init__()
        self.dataset_name = split
        self.subdataset_names = [
            subdataset_info.name for subdataset_info in subdataset_list
        ]
        self.view_sampler_ranges = [
            subdataset_info.view_sampler_range for subdataset_info in subdataset_list
        ]
        self.expansion_ranges = [
            subdataset_info.expansion_factor for subdataset_info in subdataset_list
        ]
        self.equalization_length = [
            subdataset_info.equalization_length for subdataset_info in subdataset_list
        ]
        self.normalization_mode = [
            subdataset_info.normalization_mode for subdataset_info in subdataset_list
        ]
        self.subdatasets = []
        self.non_repeated_lengths = []
        for (
            normalization_mode,
            subdataset_name,
            view_sampler_range,
            expansion_factor,
        ) in zip(
            self.normalization_mode,
            self.subdataset_names,
            self.view_sampler_ranges,
            self.expansion_ranges,
        ):
            selector_type = get_selector_type(subdataset_name)
            view_selector_instance = _create_view_selector(
                selector_type,
                view_sampler_range,
                target_has_input_p,
                expansion_factor,
            )

            self.subdatasets.append(
                available_datasets[subdataset_name](
                    view_selector=view_selector_instance,
                    split=split,
                    im_size_hw=im_size_hw,
                    num_cond_views=num_cond_views,
                    zero_out_cam_cond_p=zero_out_cam_cond_p,
                    normalization_mode=NormalizationMode(normalization_mode),
                )
            )
            print(
                f"[INIT OK] Loaded dataset {subdataset_name} with length {len(self.subdatasets[-1])}",
                flush=True,
            )
            self.non_repeated_lengths.append(len(self.subdatasets[-1]))

        self.subdataset_start_idx = [0]
        print("Approximately equalizing dataset lengths...")
        # make list 1 longer than number of datasets
        # last number is the start index of a virtual dataset
        self.repeat_factors = []
        for subdataset_idx, non_repeated_length in enumerate(self.non_repeated_lengths):
            print(f"Dataset {subdataset_idx} has length {non_repeated_length}")
            repeat_factor = (
                self.equalization_length[subdataset_idx] // non_repeated_length
            )
            self.repeat_factors.append(repeat_factor)
            self.subdataset_start_idx.append(
                self.subdataset_start_idx[subdataset_idx]
                + non_repeated_length * repeat_factor
            )

        print(f"Total length: {self.__len__()}, including the following datasets:")

        for subdataset_name, non_repeated_length, repeat_factor in zip(
            self.subdataset_names, self.non_repeated_lengths, self.repeat_factors
        ):
            print(
                f"{subdataset_name}: originally {non_repeated_length}, repeated {repeat_factor} times to reach {non_repeated_length * repeat_factor} examples"
            )

    def get_subdataset_idx_and_ex(self, example_idx_global):
        # find which dataset to use for a given example
        dataset_to_use_idx = 0
        while example_idx_global > self.subdataset_start_idx[dataset_to_use_idx + 1]:
            dataset_to_use_idx += 1
        example_idx_local = (
            example_idx_global - self.subdataset_start_idx[dataset_to_use_idx]
        ) % self.non_repeated_lengths[dataset_to_use_idx]
        return dataset_to_use_idx, example_idx_local

    def __len__(self):
        return self.subdataset_start_idx[-1]

    def __getitem__(self, tuple_idx):
        l_idx_global, num_views, num_cond_views, aspect_ratio = tuple_idx
        d_idx, l_idx = self.get_subdataset_idx_and_ex(l_idx_global)

        images, cam_enc, image_ids, cam_cond_token, is_valid = self.subdatasets[
            d_idx
        ].__getitem__((l_idx, num_views, num_cond_views, aspect_ratio))
        image_ids_prepend = []
        for im_id in image_ids:
            image_ids_prepend.append(f"{self.subdataset_names[d_idx]}_{im_id}")
        return (
            images,
            cam_enc,
            image_ids_prepend,
            cam_cond_token,
            is_valid,
            num_cond_views,
        )
