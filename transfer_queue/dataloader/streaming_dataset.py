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

import logging
import os
import time
import uuid
from typing import Any, Iterator

from torch.utils.data import IterableDataset

from transfer_queue import (
    TransferQueueClient,
    ZMQServerInfo,
)

TQ_STREAMING_DATASET_EMPTY_BATCH_SLEEP_INTERVAL = float(
    os.environ.get("TQ_STREAMING_DATASET_EMPTY_BATCH_SLEEP_INTERVAL", 1)
)  # in seconds

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

# Ensure logger has a handler
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(handler)


class StreamingDataset(IterableDataset):
    """Streaming dataset for distributed training with TransferQueue.

    This dataset is designed to work with RankAwareSampler for distributed training
    scenarios where each rank independently retrieves data through TransferQueue.

    Each rank calls the sampler independently, passing its own rank information.
    The RankAwareSampler guarantees that all ranks within the same DP group receive
    the same sample indices, ensuring no data is duplicated or missed.

    Usage Example:
        >>> # On each rank:
        >>> dataset = StreamingDataset(
        ...     config=config,
        ...     micro_batch_size=4,
        ...     required_fields=["input_ids", "attention_mask"],
        ...     partition_id="train",
        ...     task_name="update_actor",
        ...     dp_group=rank // dp_world_size,  # DP group ID for this rank
        ...     dp_world_size=dp_world_size,
        ...     world_size=total_world_size,
        ...     rank=rank,
        ... )
        >>> dataloader = StreamingDataLoader(
        ...     dataset,
        ...     batch_size=micro_batch_size,
        ...     num_workers=0,
        ... )
        >>> for batch in dataloader:
        ...     # batch is a TensorDict with the requested fields
        ...     pass
    """

    def __init__(
        self,
        config: dict[str, Any],
        micro_batch_size: int,
        required_fields: list[str],
        partition_id: str,
        task_name: str,
        data_replica_group: int,
        data_replica_rank: int,
        data_replica_world_size: int,
    ):
        """Initialize the StreamingDataset.

        Args:
            config: Configuration dictionary containing:
                - controller_info: ZMQServerInfo for the TransferQueueController
                - storage_backend: Storage backend type (e.g., "SimpleAsyncStorageManager")
            micro_batch_size: Number of samples per micro-batch.
            required_fields: List of field names to retrieve from storage.
            partition_id: Partition ID for data versioning.
            task_name: Unique identifier for the training task.
            data_replica_group: The group id of current data replica group. Used to
                identify which data replica group this rank belongs to.
            data_replica_rank: Local rank inside this data_replica_group.
            data_replica_world_size: Total number of ranks in this data_replica_group.
        """

        self.config = config
        self.micro_batch_size = micro_batch_size
        self.required_fields = required_fields
        self.partition_id = partition_id
        self.task_name = task_name
        self.data_replica_group = data_replica_group
        self.data_replica_rank = data_replica_rank
        self.data_replica_world_size = data_replica_world_size

        if data_replica_world_size < 1:
            raise ValueError(f"data_replica_world_size {data_replica_world_size} must >= 1")

        if data_replica_rank >= data_replica_world_size or data_replica_rank < 0:
            raise ValueError(
                f"data_replica_rank {data_replica_rank} must be greater than or equal to 0 and less than "
                f"data_replica_world_size {data_replica_world_size}"
            )

        # Build sampling config for controller
        self.sampling_config = {
            "data_replica_group": self.data_replica_group,
            "data_replica_rank": self.data_replica_rank,
            "data_replica_world_size": self.data_replica_world_size,
            "task_name": self.task_name,
            "partition_id": self.partition_id,
        }

        self._tq_client = None

        super().__init__()

    def _create_client(self):
        client_id = uuid.uuid4().hex[:8]
        controller_info = self.config.get("controller_info", None)
        if not controller_info or not isinstance(controller_info, ZMQServerInfo):
            raise ValueError("Invalid or missing controller_info in config")

        storage_backend = self.config.get("storage_backend", None)
        if not storage_backend:
            raise ValueError("Missing storage_backend in config")

        self._tq_client = TransferQueueClient(client_id, self.config[controller_info])
        self._tq_client.initialize_storage_manager(manager_type=self.config.storage_backend, config=self.config)

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the dataset, yielding batches of data.

        Yields:
            Tuple[TensorDict, BatchMeta]: A tuple containing:
                - TensorDict: Batch of data with the requested fields.
                - BatchMeta: Corresponding metadata to interact with TransferQueue.
        Note:
            This iterator runs indefinitely until the data source is exhausted.
            The caller should handle StopIteration when appropriate (e.g., when
            all data has been consumed and no more data will be produced).
        """
        if self._tq_client is None:
            self._create_client()

        while not self._tq_client.check_consumption_status(self.task_name, self.partition_id):
            try:
                # Get metadata from controller
                batch_meta = self._tq_client.get_meta(
                    data_fields=self.required_fields,
                    batch_size=self.micro_batch_size,
                    partition_id=self.partition_id,
                    task_name=self.task_name,
                    sampling_config=self.sampling_config,
                )

                # Check if we got valid data
                if batch_meta.size == 0:
                    logger.debug(
                        f"[StreamingDataset]: Received empty batch, waiting for more data... "
                        f"Required batch_size={self.micro_batch_size}, data_fields={self.required_fields},"
                        f"partition_id={self.partition_id}, task_name={self.task_name}."
                    )

                    time.sleep(TQ_STREAMING_DATASET_EMPTY_BATCH_SLEEP_INTERVAL)
                else:
                    batch = self._tq_client.get_data(batch_meta)
                    yield (batch, batch_meta)

            except Exception as e:
                logger.error(f"[StreamingDataset]: Error in data iteration: {e}")
                raise
