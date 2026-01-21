# Copyright 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The TransferQueue Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

from transfer_queue.sampler import BaseSampler


class RankAwareSampler(BaseSampler):
    """Rank-aware sampler for distributed training with TransferQueue.

    This sampler is designed for distributed data parallel training scenarios
    where each rank retrieves data independently.

    This sampler guarantees that all ranks within the same DP group receive
    the same sample indices.

    The sampler maintains per-DP-group state to coordinate sampling across ranks:

    - First rank in a DP group to call :meth:`sample` performs actual sampling from
      ``ready_indexes`` and caches the result
    - Subsequent ranks in the same DP group retrieve the cached indices
    - Once all ranks in the DP group have fetched their samples, the cached state is
      cleaned up.


    Please refer to our roadmap for more details:
    [Roadmap] StreamingDataLoader for task-separated RL post-training
    https://github.com/Ascend/TransferQueue/issues/1
    """

    def __init__(self):
        """Initialize the RankAwareSampler.

        The sampler maintains internal state to coordinate sampling across ranks
        within the same DP group. This state tracks which samples have been sampled
        and how many times they have been fetched.
        """
        super().__init__()

    def sample(
        self,
        ready_indexes: list[int],
        batch_size: int,
        dp_group: int,
        dp_world_size: int,
        world_size: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[int], list[int]]:
        """Sample indices for the current rank, coordinating with other DP ranks.

        This method implements coordinated sampling for distributed training.
        The first rank in each DP group to call this method performs actual sampling
        from ``ready_indexes`` and caches the result. Subsequent ranks in the same
        DP group receive the cached indices directly.

        Args:
            ready_indexes: List of global indices for which all required fields of the
                corresponding samples have been produced, and the samples are not labeled
                as consumed in the corresponding task.
            batch_size: Number of samples to select. If larger than available
                ready samples, all available samples will be returned.
            dp_group: The group id of current data parallel group. Used to
                identify which DP group this rank belongs to.
            dp_world_size: Number of ranks in the data parallelism group. Used to
                determine when all ranks have fetched their samples.
            world_size: Total number of ranks across all parallelism dimensions.
                Used to determine when all ranks have fetched their samples.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            List of sampled global indices. Typically, has length `batch_size`,
            or returns an empty list if samples are insufficient.

            List of global indices that should be labeled as consumed
            (will never be retrieved by other dp_groups in the future).

        Raises:
            RuntimeError: If ``world_size`` is not divisible by ``dp_world_size``.
        """

        # Check if this DP group already has sampled data cached
        data_for_dp_group = self._states.get(dp_group, None)

        # Calculate how many times this batch should be fetched across all ranks
        if dp_world_size <= 0 or world_size % dp_world_size != 0:
            raise RuntimeError(f"world_size ({world_size}) is not divisible by dp_world_size ({dp_world_size})")

        fetches_per_batch = world_size // dp_world_size

        if data_for_dp_group is None:
            # Select first batch_size indices from ready_indexes
            sampled_indexes = ready_indexes[:batch_size]

            if len(sampled_indexes) < batch_size:
                return [], []

            # Initialize state for this DP group
            self._states[dp_group] = {}
            consumed_indexes = sampled_indexes

            # Cache the sampled indices for other ranks in this DP group
            self._states[dp_group]["index"] = sampled_indexes
            self._states[dp_group]["fetch_count"] = 1

        else:
            # Return the cached indices (identical to what first rank received)
            sampled_indexes = self._states[dp_group]["index"]
            consumed_indexes = self._states[dp_group]["index"]

            # Increment fetch count to track progress
            self._states[dp_group]["fetch_count"] += 1

        # Check if this was the last rank in the DP group to fetch
        if self._states[dp_group]["fetch_count"] >= fetches_per_batch:
            del self._states[dp_group]

        return sampled_indexes, consumed_indexes
