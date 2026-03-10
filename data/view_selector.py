import json
import logging
import random

import numpy as np

logger = logging.getLogger(__name__)


class ViewSelector:
    def __init__(self):
        pass

    def sample_views(self):
        return


def get_delta_t_and_start_idx(
    num_frames_available, num_cond, view_range_min, view_range_max
):
    # Handle single conditioning view case - no spacing between views needed
    if num_cond == 1:
        delta_t = min(
            random.randint(view_range_min, view_range_max), num_frames_available - 1
        )
        if delta_t == num_frames_available - 1:
            start_idx = 0
        else:
            start_idx = random.randint(0, num_frames_available - 1 - delta_t)
        return delta_t, start_idx

    max_possible_delta_t = min(num_frames_available // (num_cond - 1), view_range_max)

    delta_t = random.randint(
        min(max_possible_delta_t, view_range_min), max_possible_delta_t
    )
    if num_frames_available - delta_t * (num_cond - 1) < 0:
        logger.error(
            "num_frames_available - delta_t * (num_cond - 1) < 0: num_frames_available=%d, delta_t=%d, num_cond=%d, view_range_min=%d, view_range_max=%d",
            num_frames_available,
            delta_t,
            num_cond,
            view_range_min,
            view_range_max,
        )
    if delta_t <= 0:
        logger.error(
            "delta_t < 0: num_frames_available=%d, delta_t=%d, num_cond=%d, view_range_min=%d, view_range_max=%d",
            num_frames_available,
            delta_t,
            num_cond,
            view_range_min,
            view_range_max,
        )
    max_start_idx = max(0, num_frames_available - delta_t * (num_cond - 1))

    start_idx = random.randint(0, max_start_idx)

    return delta_t, start_idx


class ExpandedLinearViewSelector(ViewSelector):
    # Sample views from a bounded range of views such that
    # target views can be sampled from +- 10% of the sampled range compared
    # to the conditioning views
    def __init__(
        self,
        view_range_min,
        view_range_max,
        target_has_input_p,
        expansion_factor=0.2,
    ):
        super().__init__()
        self.view_range_min = view_range_min
        self.view_range_max = view_range_max
        self.target_has_input_p = target_has_input_p
        self.temp_jitter_prop = 0.1
        self.expansion_factor = expansion_factor

    def sample_views(self, num_views_to_sample, num_cond, seq_name, num_frames):
        if self.view_range_min > num_frames or num_views_to_sample > num_frames:
            logger.warning(
                "view_range_min > num_frames or num_views_to_sample > num_frames: view_range_min=%d, num_frames=%d, num_views_to_sample=%d",
                self.view_range_min,
                num_frames,
                num_views_to_sample,
            )
            return None

        try:
            # max delta t that will still fit within the sequence. Upper bounded
            # by value from config
            delta_t, start_idx = get_delta_t_and_start_idx(
                num_frames, num_cond, self.view_range_min, self.view_range_max
            )

            temp_jitter = int(delta_t * self.temp_jitter_prop)
            per_view_temp_jitter = (
                np.random.randint(max(1, (2 * temp_jitter)), size=num_cond)
                - temp_jitter
            )
            cond_timestamps = [
                start_idx + c_idx * delta_t + per_view_temp_jitter[c_idx]
                for c_idx in range(num_cond)
            ]
            cond_timestamps = sorted(
                min(max(0, cond_t), num_frames - 1) for cond_t in cond_timestamps
            )
            sampling_end_idx = min(
                num_frames - 1,
                cond_timestamps[-1] + int(delta_t * self.expansion_factor),
            )
            # random shuffle
            cond_timestamps = np.random.choice(
                cond_timestamps, (num_cond,), replace=False
            )

            sampling_start_idx = max(
                0, start_idx - int(delta_t * self.expansion_factor)
            )

            tgt_timesteps_options = [
                t
                for t in np.arange(sampling_start_idx, sampling_end_idx)
                if t not in cond_timestamps
            ]

            if len(tgt_timesteps_options) == 0:
                logger.error(
                    "No target timestep options available for seq=%s: sampling_range=[%d, %d), cond_timestamps=%s, "
                    "start_idx=%d, delta_t=%d, temp_jitter=%d, per_view_temp_jitter=%s, num_frames=%d",
                    seq_name,
                    sampling_start_idx,
                    sampling_end_idx,
                    cond_timestamps,
                    start_idx,
                    delta_t,
                    temp_jitter,
                    per_view_temp_jitter,
                    num_frames,
                )
                tgt_timesteps_options = np.arange(sampling_start_idx, sampling_end_idx)

            num_targets_needed = num_views_to_sample - num_cond
            if num_targets_needed > len(tgt_timesteps_options):
                logger.debug(
                    "Sampling with replacement: num_targets_needed=%d > available_options=%d for seq=%s",
                    num_targets_needed,
                    len(tgt_timesteps_options),
                    seq_name,
                )

            tgt_timestamps = np.random.choice(
                tgt_timesteps_options,
                (num_targets_needed,),
                replace=num_targets_needed > len(tgt_timesteps_options),
            )

            all_frames_indices = np.concatenate([cond_timestamps, tgt_timestamps])

            if np.random.uniform() < self.target_has_input_p:
                new_tgt_timestamps = np.random.choice(
                    all_frames_indices,
                    len(tgt_timestamps),
                    replace=False,
                )
                all_frames_indices[num_cond : num_cond + len(tgt_timestamps)] = (
                    new_tgt_timestamps
                )
            if self.target_has_input_p < 0.0:
                min_num_cond_tgt = min(num_cond, len(tgt_timestamps))
                all_frames_indices[num_cond : num_cond + min_num_cond_tgt] = (
                    cond_timestamps[:min_num_cond_tgt]
                )

            return all_frames_indices

        except Exception as e:
            logger.error(
                "ExpandedLinearViewSelector failed to sample views for seq=%s: start_idx=%d, delta_t=%d, temp_jitter=%d, num_frames=%d, error=%s",
                seq_name,
                start_idx,
                delta_t,
                temp_jitter,
                num_frames,
                str(e),
            )
            raise e


class FixedViewSelector(ViewSelector):
    def __init__(self, test_indices_path):
        super().__init__()
        with open(test_indices_path, "r") as f:
            self.view_indices = json.load(f)

    def sample_views(self, num_views_to_sample, num_cond, seq_name, num_frames):
        assert num_views_to_sample is None, (
            f"num_views must be None for fixed view selector, got {num_views_to_sample}"
        )
        views = np.concatenate(
            [
                self.view_indices[seq_name]["context"],
                self.view_indices[seq_name]["target"],
            ]
        )
        return views


