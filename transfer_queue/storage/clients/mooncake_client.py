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
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

from transfer_queue.storage.clients.base import TransferQueueStorageKVClient
from transfer_queue.storage.clients.factory import StorageClientFactory
from transfer_queue.utils.tensor_utils import get_nbytes

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(handler)

MOONCAKE_STORE_IMPORTED: bool = True
try:
    from mooncake.store import MooncakeDistributedStore, ReplicateConfig
except ImportError:
    MOONCAKE_STORE_IMPORTED = False

BATCH_SIZE_LIMIT: int = 200
MAX_WORKER_THREADS = 4

# Mapping from torch dtype to numpy dtype for buffer reinterpretation.
_TORCH_TO_NP_DTYPE = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float16: np.float16,
    torch.uint8: np.uint8,
    torch.bool: np.bool_,
    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
}


@StorageClientFactory.register("MooncakeStoreClient")
class MooncakeStoreClient(TransferQueueStorageKVClient):
    """
    Storage client for MooncakeStore.

    Uses a pre-allocated per-thread communication buffer. All tensors are
    copied into the buffer before put / after get, eliminating per-tensor
    register/unregister overhead.
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
            from transfer_queue.utils.zmq_utils import get_node_ip_address_raw

            ip = get_node_ip_address_raw()
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

        # Thread-local reusable transfer buffer.
        self._thread_local = threading.local()
        self._buffers_lock = threading.Lock()
        self._registered_buffers: list[tuple[int, int]] = []
        # Cap per-thread buffer to avoid excessive memory usage.
        self._transfer_buffer_size = min(self.local_buffer_size, 256 * 1024 * 1024)

    def _get_thread_buffer(self) -> torch.Tensor:
        """Get or create a thread-local transfer buffer and register it with Mooncake."""
        buf = getattr(self._thread_local, "buffer", None)
        if buf is None:
            buf = torch.empty(self._transfer_buffer_size, dtype=torch.uint8)
            self._store.register_buffer(buf.data_ptr(), buf.nbytes)
            self._thread_local.buffer = buf
            with self._buffers_lock:
                self._registered_buffers.append((buf.data_ptr(), buf.nbytes))
        return buf

    @staticmethod
    def _copy_tensor_to_buffer(buffer: torch.Tensor, offset: int, t: torch.Tensor) -> int:
        """Copy tensor bytes into buffer at given offset. Returns nbytes copied."""
        nbytes = t.nbytes
        # Reinterpret source tensor as byte tensor and copy.
        # unsqueeze(0) handles 0-dim (scalar) tensors safely.
        src = t.contiguous().unsqueeze(0).view(torch.uint8).reshape(-1)
        dest = buffer[offset : offset + nbytes]
        dest.copy_(src)
        return nbytes

    @staticmethod
    def _tensor_from_buffer(
        buffer: torch.Tensor, offset: int, shape: tuple, dtype: torch.dtype, nbytes: int
    ) -> torch.Tensor:
        """Create a tensor view from buffer bytes without copying."""
        buf_np = buffer[offset : offset + nbytes].numpy()

        if dtype == torch.bfloat16:
            arr = buf_np.view(np.uint16)
            t = torch.from_numpy(arr).view(torch.bfloat16)
        else:
            np_dtype = _TORCH_TO_NP_DTYPE.get(dtype)
            if np_dtype is None:
                raise ValueError(f"Unsupported dtype for buffer view: {dtype}")
            arr = buf_np.view(np_dtype)
            t = torch.from_numpy(arr)

        return t.reshape(shape)

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

    def _put_tensors_thread_worker(self, batch_keys: list[str], batch_tensors: list[Tensor]):
        """Worker thread for putting batch of tensors to MooncakeStore."""

        buffer = self._get_thread_buffer()
        offset = 0
        ptrs = []
        sizes = []
        total_copied = 0

        for t in batch_tensors:
            nbytes = t.nbytes
            if offset + nbytes > buffer.nbytes:
                raise RuntimeError(
                    f"Buffer overflow in put: need {offset + nbytes} bytes, "
                    f"buffer size is {buffer.nbytes}. Consider increasing local_buffer_size."
                )
            self._copy_tensor_to_buffer(buffer, offset, t)
            ptrs.append(buffer.data_ptr() + offset)
            sizes.append(nbytes)
            offset += nbytes
            total_copied += nbytes

        logger.info(
            f"[MooncakeStoreClient] Buffer put: tensors={len(batch_tensors)}, "
            f"total_bytes={total_copied}, buffer_size={buffer.nbytes}"
        )

        results = self._store.batch_upsert_from(batch_keys, ptrs, sizes, config=self.replica_config)
        if not all(r == 0 for r in results):
            failed_indices = [j for j, r in enumerate(results) if r != 0]
            error_codes = [results[j] for j in failed_indices]
            raise RuntimeError(f"batch_put_tensor failed for indices {failed_indices} with error codes: {error_codes}")

    def _put_bytes_thread_worker(self, batch_keys: list[str], batch_values: list[Any]):
        """Worker thread for putting batch of non-tensors to MooncakeStore."""

        batch_values = [pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL) for v in batch_values]

        ret = self._store.upsert_batch(batch_keys, batch_values, self.replica_config)
        if ret != 0:
            raise RuntimeError(f"put_batch failed with error code: {ret}")

    def get(
        self,
        keys: list[str],
        shapes: Optional[list[Any]] = None,
        dtypes: Optional[list[Any]] = None,
        custom_backend_meta: Optional[list[str]] = None,
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
        buffer = self._get_thread_buffer()
        batch_nbytes = get_nbytes(batch_dtypes, batch_shapes)
        total_needed = sum(batch_nbytes)
        if total_needed > buffer.nbytes:
            raise RuntimeError(
                f"Buffer overflow in get: need {total_needed} bytes, "
                f"buffer size is {buffer.nbytes}. Consider increasing local_buffer_size."
            )

        offset = 0
        ptrs = []
        tensors = []

        for shape, dtype, nbytes in zip(batch_shapes, batch_dtypes, batch_nbytes, strict=False):
            t = self._tensor_from_buffer(buffer, offset, shape, dtype, nbytes)
            tensors.append(t)
            ptrs.append(buffer.data_ptr() + offset)
            offset += nbytes

        ret_codes = self._store.batch_get_into(batch_keys, ptrs, batch_nbytes)
        if len(ret_codes) != len(batch_keys):
            raise RuntimeError(f"batch_get_into returned {len(ret_codes)} results, expected {len(batch_keys)}")
        for i, ret in enumerate(ret_codes):
            if ret < 0:
                raise RuntimeError(f"batch_get_into failed for key `{batch_keys[i]}` with error code: {ret}")

        # Clone to free the reusable buffer for subsequent requests.
        return [t.clone() for t in tensors], indexes

    def _get_bytes_thread_worker(self, batch_keys: list[str], indexes: list[int]) -> tuple[list[Any], list[int]]:
        results = []

        batch_results = self._store.get_batch(batch_keys)
        if len(batch_results) != len(batch_keys):
            raise RuntimeError(f"get_batch returned {len(batch_results)} items, expected {len(batch_keys)}")

        batch_results = [pickle.loads(result) for result in batch_results]
        results.extend(batch_results)

        return results, indexes

    def clear(self, keys: list[str], custom_backend_meta=None):
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
        """Closes MooncakeStore and unregisters all thread-local buffers."""
        if self._store:
            for ptr, size in self._registered_buffers:
                self._store.unregister_buffer(ptr)
            self._registered_buffers.clear()
            self._store.close()
            self._store = None
