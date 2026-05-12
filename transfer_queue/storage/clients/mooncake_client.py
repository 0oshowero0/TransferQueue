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

import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import torch
from torch import Tensor

from transfer_queue.storage.clients.base import StorageClientFactory, StorageKVClient
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.tensor_utils import allocate_empty_tensors, get_nbytes, merge_contiguous_memory

logger = get_logger(__name__)

MOONCAKE_STORE_IMPORTED: bool = True
try:
    from mooncake.store import MooncakeDistributedStore, ReplicateConfig

except ImportError:
    MOONCAKE_STORE_IMPORTED = False

BATCH_SIZE_LIMIT: int = 400
MAX_WORKER_THREADS = 4
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.0


@StorageClientFactory.register("MooncakeStoreClient")
class MooncakeStoreClient(StorageKVClient):
    """
    Storage client for MooncakeStore.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        if not MOONCAKE_STORE_IMPORTED:
            raise ImportError("Mooncake Store not installed. Please install via: pip install mooncake-transfer-engine")

        # Required: Address of local host
        self.local_hostname = config.get("local_hostname", "")
        # Required: Address of the HTTP metadata server (e.g., "localhost:8080")
        self.metadata_server = config.get("metadata_server", None)
        # Required: Address of the master server RPC endpoint (e.g., "localhost:8081")
        self.master_server_address = config.get("master_server_address")

        self.global_segment_size = int(config.get("global_segment_size", 4096 * 1024 * 1024))
        self.local_buffer_size = int(config.get("local_buffer_size", 1024 * 1024 * 1024))
        self.protocol = config.get("protocol", "tcp")
        self.device_name = config.get("device_name", "")
        if self.device_name is None:
            self.device_name = ""

        if self.local_hostname is None or self.local_hostname == "":
            from transfer_queue.utils.zmq_utils import get_node_ip_address

            ip = get_node_ip_address()
            logger.info(f"Try to use Ray IP ({ip}) as local hostname for MooncakeStore.")
            self.local_hostname = ip

        if self.metadata_server is None or not isinstance(self.metadata_server, str):
            raise ValueError("Missing or invalid 'metadata_server' in config")
        if self.master_server_address is None or not isinstance(self.master_server_address, str):
            raise ValueError("Missing or invalid 'master_server_address' in config")

        if not self.metadata_server.startswith("http://") and not self.metadata_server.startswith("etcd://"):
            self.metadata_server = f"http://{self.metadata_server}"
        if not self.metadata_server.startswith("etcd://") and not self.metadata_server.endswith("/metadata"):
            self.metadata_server = self.metadata_server + "/metadata"

        self.replica_config = ReplicateConfig()
        self.replica_config.with_hard_pin = True

        self._store = MooncakeDistributedStore()
        ret = self._store.setup(
            self.local_hostname,
            self.metadata_server,
            self.global_segment_size,
            self.local_buffer_size,
            self.protocol,
            self.device_name,
            self.master_server_address,
        )
        if ret != 0:
            raise RuntimeError(f"Mooncake store setup failed with error code: {ret}")

    def put(self, keys: list[str], values: list[Any]) -> None:
        """Stores multiple key-value pairs to MooncakeStore.

        Args:
            keys (List[str]): List of unique string identifiers.
            values (List[Any]): List of values to store (tensors, scalars, dicts, etc.).
        """

        if not isinstance(keys, list) or not isinstance(values, list):
            raise ValueError("keys and values must be lists")
        if len(keys) != len(values):
            raise ValueError("Number of keys must match number of values")

        tensor_keys = []
        tensor_values = []
        non_tensor_keys = []
        non_tensor_values = []

        for key, value in zip(keys, values, strict=True):
            if isinstance(value, torch.Tensor):
                tensor_keys.append(key)
                tensor_values.append(value)
            else:
                non_tensor_keys.append(key)
                non_tensor_values.append(value)

        futures = []
        with ThreadPoolExecutor(max_workers=MAX_WORKER_THREADS) as executor:
            for i in range(0, len(tensor_keys), BATCH_SIZE_LIMIT):
                batch_keys = tensor_keys[i : i + BATCH_SIZE_LIMIT]
                batch_tensors = tensor_values[i : i + BATCH_SIZE_LIMIT]
                futures.append(executor.submit(self._put_tensors_thread_worker, batch_keys, batch_tensors))

            for i in range(0, len(non_tensor_keys), BATCH_SIZE_LIMIT):
                batch_keys = non_tensor_keys[i : i + BATCH_SIZE_LIMIT]
                batch_values = non_tensor_values[i : i + BATCH_SIZE_LIMIT]
                futures.append(executor.submit(self._put_bytes_thread_worker, batch_keys, batch_values))

            for future in as_completed(futures):
                future.result()

        return None

    def _put_tensors_thread_worker(self, batch_keys: list[str], batch_tensors: list[Tensor]) -> None:
        """Worker thread for putting batch of tensors to MooncakeStore."""

        batch_ptrs, batch_sizes, _contiguous_tensors = self._preprocess_tensors_for_put(batch_tensors)
        batch_ptr_reduced, batch_sizes_reduced = merge_contiguous_memory(batch_ptrs, batch_sizes)
        self._register_all_buffers(batch_ptr_reduced, batch_sizes_reduced)

        try:
            results = self._store.batch_upsert_from(batch_keys, batch_ptrs, batch_sizes, config=self.replica_config)
            if len(results) != len(batch_keys):
                raise RuntimeError(f"batch_upsert_from returned {len(results)} results, expected {len(batch_keys)}")

            failed_indices = [j for j, r in enumerate(results) if r != 0]
            if not failed_indices:
                return

            current_failed_keys = [batch_keys[i] for i in failed_indices]
            current_failed_codes = [results[i] for i in failed_indices]
            current_failed_indices = failed_indices

            logger.error(
                f"batch_upsert_from failed for keys {current_failed_keys} with error codes {current_failed_codes}. "
                f"Retrying up to {MAX_RETRIES} times..."
            )

            for attempt in range(1, MAX_RETRIES + 1):
                retry_ptrs = [batch_ptrs[i] for i in current_failed_indices]
                retry_sizes = [batch_sizes[i] for i in current_failed_indices]

                retry_results = self._store.batch_upsert_from(
                    current_failed_keys, retry_ptrs, retry_sizes, config=self.replica_config
                )

                next_failed_indices = []
                next_failed_keys = []
                next_failed_codes = []

                for i, ret in enumerate(retry_results):
                    if ret != 0:
                        next_failed_indices.append(current_failed_indices[i])
                        next_failed_keys.append(current_failed_keys[i])
                        next_failed_codes.append(ret)

                if not next_failed_indices:
                    logger.info("batch_upsert_from succeeded after retransmission.")
                    break  # All retries in this attempt succeeded.

                logger.error(
                    f"batch_upsert_from retry {attempt}/{MAX_RETRIES} failed for {len(next_failed_keys)} keys "
                    f"with error codes {next_failed_codes}."
                )

                current_failed_indices = next_failed_indices
                current_failed_keys = next_failed_keys
                current_failed_codes = next_failed_codes

                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SECONDS)
            else:
                raise RuntimeError(
                    f"batch_upsert_from failed for keys {current_failed_keys} with error codes "
                    f"{current_failed_codes} after retrying {MAX_RETRIES} times."
                )

        finally:
            self._unregister_all_buffers(batch_ptr_reduced)

    def _put_bytes_thread_worker(self, batch_keys: list[str], batch_values: list[Any]):
        """Worker thread for putting batch of non-tensors to MooncakeStore."""

        serialized_values = [pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL) for v in batch_values]

        # FIXME: When MooncakeStore supports per-key status codes for upsert_batch and get_batch,
        #        switch the bytes write/read paths from whole-batch retry to per-key selective retry,
        #        matching the tensor-path behaviour.
        ret = self._store.upsert_batch(batch_keys, serialized_values, self.replica_config)
        if ret == 0:
            return

        logger.error(
            f"upsert_batch failed for {len(batch_keys)} keys with error code: {ret}. "
            f"Retrying up to {MAX_RETRIES} times..."
        )

        for attempt in range(1, MAX_RETRIES + 1):
            ret = self._store.upsert_batch(batch_keys, serialized_values, self.replica_config)
            if ret == 0:
                logger.info("upsert_batch succeeded after retransmission.")
                return

            logger.error(
                f"upsert_batch retry {attempt}/{MAX_RETRIES} failed for {len(batch_keys)} keys with error code: {ret}."
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)

        raise RuntimeError(
            f"upsert_batch failed for {len(batch_keys)} keys with error code: {ret} after retrying {MAX_RETRIES} times."
        )

    def get(
        self,
        keys: list[str],
        shapes: list[Any] | None = None,
        dtypes: list[Any] | None = None,
        custom_backend_meta: list[str] | None = None,
    ) -> list[Any]:
        """Get multiple key-value pairs from MooncakeStore.

        Args:
            keys: Keys to fetch.
            shapes: Expected tensor shapes (use [] for scalars).
            dtypes: Expected dtypes; use None for non-tensor data.
            custom_backend_meta: Optional custom backend metadata.

        Returns:
            Retrieved values in the same order as input keys.
        """

        if shapes is None or dtypes is None:
            raise ValueError("MooncakeStoreClient needs shapes and dtypes for zero-copy transfer.")
        if not (len(keys) == len(shapes) == len(dtypes)):
            raise ValueError("Lengths of keys, shapes, dtypes must match")

        tensor_indices = []
        non_tensor_indices = []

        for i, dtype in enumerate(dtypes):
            if dtype is not None:
                tensor_indices.append(i)
            else:
                non_tensor_indices.append(i)

        results = [None] * len(keys)

        futures = []
        with ThreadPoolExecutor(max_workers=MAX_WORKER_THREADS) as executor:
            for i in range(0, len(tensor_indices), BATCH_SIZE_LIMIT):
                batch_indexes = tensor_indices[i : i + BATCH_SIZE_LIMIT]
                batch_keys = [keys[i] for i in batch_indexes]
                batch_shapes = [shapes[i] for i in batch_indexes]
                batch_dtypes = [dtypes[i] for i in batch_indexes]
                futures.append(
                    executor.submit(
                        self._get_tensors_thread_worker, batch_keys, batch_shapes, batch_dtypes, batch_indexes
                    )
                )

            for i in range(0, len(non_tensor_indices), BATCH_SIZE_LIMIT):
                batch_indexes = non_tensor_indices[i : i + BATCH_SIZE_LIMIT]
                batch_keys = [keys[i] for i in batch_indexes]
                futures.append(executor.submit(self._get_bytes_thread_worker, batch_keys, batch_indexes))

            for future in as_completed(futures):
                retrieved_values, batch_indexes = future.result()
                for idx, val in zip(batch_indexes, retrieved_values, strict=True):
                    results[idx] = val

        return results

    def _get_tensors_thread_worker(
        self, batch_keys: list[str], batch_shapes: list[tuple], batch_dtypes: list[torch.dtype], indexes: list[int]
    ) -> tuple[list[Tensor], list[int]]:
        batch_nbytes = get_nbytes(batch_dtypes, batch_shapes)
        batch_buffer_tensors, batch_buffer_ptrs, region_ptrs, region_sizes = allocate_empty_tensors(
            batch_dtypes, batch_shapes
        )

        self._register_all_buffers(region_ptrs, region_sizes)
        try:
            ret_codes = self._store.batch_get_into(batch_keys, batch_buffer_ptrs, batch_nbytes)
            if len(ret_codes) != len(batch_keys):
                raise RuntimeError(f"batch_get_into returned {len(ret_codes)} results, expected {len(batch_keys)}")

            failed_indices = [i for i, ret in enumerate(ret_codes) if ret < 0]
            if not failed_indices:
                return batch_buffer_tensors, indexes

            # error handling
            current_failed_keys = [batch_keys[i] for i in failed_indices]
            current_failed_codes = [ret_codes[i] for i in failed_indices]
            current_failed_indices = failed_indices

            logger.error(
                f"batch_get_into failed for keys {current_failed_keys} with error codes {current_failed_codes}. "
                f"Retrying up to {MAX_RETRIES} times..."
            )

            for attempt in range(1, MAX_RETRIES + 1):
                # Reuse the originally allocated pointers; no need to allocate/register new buffers.
                retry_ptrs = [batch_buffer_ptrs[i] for i in current_failed_indices]
                retry_nbytes = [batch_nbytes[i] for i in current_failed_indices]

                retry_codes = self._store.batch_get_into(current_failed_keys, retry_ptrs, retry_nbytes)

                next_failed_indices = []
                next_failed_keys = []
                next_failed_codes = []

                for i, ret in enumerate(retry_codes):
                    if ret < 0:
                        next_failed_indices.append(current_failed_indices[i])
                        next_failed_keys.append(current_failed_keys[i])
                        next_failed_codes.append(ret)

                if not next_failed_indices:
                    logger.info("batch_get_into succeeded after retransmission.")
                    break  # All retries in this attempt succeeded.

                logger.error(
                    f"batch_get_into retry {attempt}/{MAX_RETRIES} failed for {len(next_failed_keys)} keys "
                    f"with error codes {next_failed_codes}."
                )

                # Narrow down to still-failed items for the next retry attempt.
                current_failed_indices = next_failed_indices
                current_failed_keys = next_failed_keys
                current_failed_codes = next_failed_codes

                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SECONDS)
            else:
                # All retries exhausted.
                raise RuntimeError(
                    f"batch_get_into failed for keys {current_failed_keys} with error codes "
                    f"{current_failed_codes} after retrying {MAX_RETRIES} times."
                )

        finally:
            self._unregister_all_buffers(region_ptrs)

        return batch_buffer_tensors, indexes

    def _get_bytes_thread_worker(self, batch_keys: list[str], indexes: list[int]) -> tuple[list[Any], list[int]]:
        raw_results = self._store.get_batch(batch_keys)
        if len(raw_results) != len(batch_keys):
            raise RuntimeError(f"get_batch returned {len(raw_results)} items, expected {len(batch_keys)}")

        # FIXME: Use MooncakeStore provided ret codes to detect transmission failures when supported
        # Currently we rely on empty bytes (b'') to detect transmission failures because
        # MooncakeStore does not currently return a separate status code per key.
        failed_indices = [i for i, result in enumerate(raw_results) if result == b""]
        if failed_indices:
            current_failed_keys = [batch_keys[i] for i in failed_indices]
            current_failed_indices = failed_indices

            logger.error(f"get_batch failed for keys {current_failed_keys}. Retrying up to {MAX_RETRIES} times...")

            for attempt in range(1, MAX_RETRIES + 1):
                retry_results = self._store.get_batch(current_failed_keys)

                next_failed_keys = []
                next_failed_indices = []

                for i, result in enumerate(retry_results):
                    original_idx = current_failed_indices[i]
                    if result == b"":
                        next_failed_keys.append(current_failed_keys[i])
                        next_failed_indices.append(original_idx)
                    else:
                        # Write the successfully retried value back to its original slot immediately.
                        raw_results[original_idx] = result

                if not next_failed_indices:
                    logger.info("get_batch succeeded after retransmission.")
                    break  # All retries in this attempt succeeded.

                logger.error(f"get_batch retry {attempt}/{MAX_RETRIES} failed for {len(next_failed_keys)} keys.")

                # Narrow down to still-failed items for the next retry attempt.
                current_failed_keys = next_failed_keys
                current_failed_indices = next_failed_indices

                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SECONDS)
            else:
                # All retries exhausted.
                raise RuntimeError(
                    f"get_batch failed for keys {current_failed_keys} after retrying {MAX_RETRIES} times."
                )

        deserialized_results = [pickle.loads(result) if result != b"" else None for result in raw_results]
        return deserialized_results, indexes

    def clear(self, keys: list[str], custom_backend_meta: list[Any] | None = None) -> None:
        """Deletes multiple keys from MooncakeStore.

        Args:
            keys (List[str]): List of keys to remove.
            custom_backend_meta (List[Any], optional): ...
        """
        ret_codes = self._store.batch_remove(keys, force=True)
        for i, ret in enumerate(ret_codes):
            if not (ret == 0 or ret == -704):
                logger.error(f"remove failed for key `{keys[i]}` with error code: {ret}")

    def close(self):
        """Closes MooncakeStore."""
        if self._store:
            self._store.close()
            self._store = None

    @staticmethod
    def _preprocess_tensors_for_put(values: list[Tensor]) -> tuple[list[int], list[int], list[Tensor]]:
        ptr_list: list[int] = []
        size_list: list[int] = []
        tensor_list: list[Tensor] = []  # hold reference for the contiguous tensor
        for t in values:
            # TODO: support gpu direct rdma and use different data paths.
            #       For GPU, it's more reasonable to perform data copy since
            #       The register overhead is much higher than CPU
            if t.device.type == "cuda":
                t = t.cpu()
            t = t.contiguous()
            tensor_list.append(t)
            ptr_list.append(t.data_ptr())
            size_list.append(t.nbytes)
        return ptr_list, size_list, tensor_list

    def _register_all_buffers(self, ptrs, sizes):
        for ptr, size in zip(ptrs, sizes, strict=True):
            self._store.register_buffer(ptr, size)

    def _unregister_all_buffers(self, ptrs):
        for ptr in ptrs:
            self._store.unregister_buffer(ptr)
