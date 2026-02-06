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

"""Unit tests for kv interface in transfer_queue.interface."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from tensordict import TensorDict

# Setup path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from transfer_queue.metadata import BatchMeta, FieldMeta, SampleMeta  # noqa: E402
from transfer_queue.utils.enum_utils import ProductionStatus  # noqa: E402


def create_batch_meta(global_indexes, fields_data):
    """Helper to create BatchMeta for testing."""
    samples = []
    for sample_id, global_idx in enumerate(global_indexes):
        fields_dict = {}
        for field_name, tensor in fields_data.items():
            field_meta = FieldMeta(
                name=field_name,
                dtype=tensor.dtype,
                shape=tensor.shape,
                production_status=ProductionStatus.READY_FOR_CONSUME,
            )
            fields_dict[field_name] = field_meta
        sample = SampleMeta(
            partition_id="test_partition",
            global_index=global_idx,
            fields=fields_dict,
        )
        samples.append(sample)
    return BatchMeta(samples=samples)


class TestKVPut:
    """Tests for kv_put function."""

    def test_kv_put_with_fields(self):
        """Test kv_put with fields parameter."""
        mock_client = MagicMock()
        mock_batch_meta = MagicMock()
        mock_client.kv_retrieve_keys.return_value = mock_batch_meta

        tensor_data = TensorDict(
            {"text": torch.tensor([[1, 2, 3]]), "label": torch.tensor([0])},
            batch_size=[1],
        )

        with patch("transfer_queue.interface._maybe_create_transferqueue_client", return_value=mock_client):
            from transfer_queue.interface import kv_put

            kv_put(key="test_key", partition_id="partition_1", fields=tensor_data, tag={"type": "test"})

        # Verify kv_retrieve_keys was called
        mock_client.kv_retrieve_keys.assert_called_once_with(keys=["test_key"], partition_id="partition_1", create=True)

        # Verify update_custom_meta was called
        mock_batch_meta.update_custom_meta.assert_called_once()

        # Verify put was called
        mock_client.put.assert_called_once()

    def test_kv_put_with_dict_fields(self):
        """Test kv_put converts dict to TensorDict correctly."""
        mock_client = MagicMock()
        mock_batch_meta = MagicMock()
        mock_client.kv_retrieve_keys.return_value = mock_batch_meta

        with patch("transfer_queue.interface._maybe_create_transferqueue_client", return_value=mock_client):
            from transfer_queue.interface import kv_put

            # Test with dict containing tensor
            kv_put(
                key="test_key",
                partition_id="partition_1",
                fields={"text": torch.tensor([1, 2, 3])},
                tag=None,
            )

        # Verify put was called
        mock_client.put.assert_called_once()
        call_args = mock_client.put.call_args
        fields_arg = call_args[0][0]
        assert "text" in fields_arg
        # The dict should be converted to TensorDict
        assert isinstance(fields_arg, TensorDict)

    def test_kv_put_with_tag_only(self):
        """Test kv_put with only tag parameter (no fields)."""
        mock_client = MagicMock()
        mock_batch_meta = MagicMock()
        mock_client.kv_retrieve_keys.return_value = mock_batch_meta

        with patch("transfer_queue.interface._maybe_create_transferqueue_client", return_value=mock_client):
            from transfer_queue.interface import kv_put

            kv_put(key="test_key", partition_id="partition_1", fields=None, tag={"score": 0.9})

        # Verify put was NOT called (only set_custom_meta)
        mock_client.put.assert_not_called()
        mock_client.set_custom_meta.assert_called_once_with(mock_batch_meta)

    def test_kv_put_raises_error_without_fields_and_tag(self):
        """Test kv_put raises ValueError when neither fields nor tag provided."""
        mock_client = MagicMock()

        with patch("transfer_queue.interface._maybe_create_transferqueue_client", return_value=mock_client):
            from transfer_queue.interface import kv_put

            with pytest.raises(ValueError, match="Please provide at least one parameter"):
                kv_put(key="test_key", partition_id="partition_1", fields=None, tag=None)


class TestKVBatchPut:
    """Tests for kv_batch_put function."""

    def test_kv_batch_put_success(self):
        """Test kv_batch_put with valid inputs."""
        mock_client = MagicMock()
        mock_batch_meta = MagicMock()
        mock_client.kv_retrieve_keys.return_value = mock_batch_meta

        batch_data = TensorDict(
            {
                "text": torch.tensor([[1, 2], [3, 4], [5, 6]]),
                "label": torch.tensor([0, 1, 2]),
            },
            batch_size=[3],
        )

        with patch("transfer_queue.interface._maybe_create_transferqueue_client", return_value=mock_client):
            from transfer_queue.interface import kv_batch_put

            keys = ["key1", "key2", "key3"]
            tags = [{"tag": "v1"}, {"tag": "v2"}, {"tag": "v3"}]

            kv_batch_put(keys=keys, partition_id="partition_1", fields=batch_data, tags=tags)

        mock_client.kv_retrieve_keys.assert_called_once_with(keys=keys, partition_id="partition_1", create=True)
        mock_batch_meta.update_custom_meta.assert_called_once_with(tags)
        mock_client.put.assert_called_once()

    def test_kv_batch_put_tags_length_mismatch(self):
        """Test kv_batch_put raises error when tags length doesn't match keys."""
        mock_client = MagicMock()

        batch_data = TensorDict(
            {
                "text": torch.tensor([[1, 2], [3, 4], [5, 6]]),
                "label": torch.tensor([0, 1, 2]),
            },
            batch_size=[3],
        )

        with patch("transfer_queue.interface._maybe_create_transferqueue_client", return_value=mock_client):
            from transfer_queue.interface import kv_batch_put

            keys = ["key1", "key2", "key3"]
            tags = [{"tag": "v1"}, {"tag": "v2"}]  # Only 2 tags for 3 keys

            with pytest.raises(ValueError, match="does not match length of tags"):
                kv_batch_put(keys=keys, partition_id="partition_1", fields=batch_data, tags=tags)


class TestKVGet:
    """Tests for kv_get function."""

    def test_kv_get_single_key(self):
        """Test kv_get with single key."""
        mock_client = MagicMock()
        mock_batch_meta = MagicMock()
        mock_client.kv_retrieve_keys.return_value = mock_batch_meta
        mock_client.get_data.return_value = TensorDict({"data": torch.tensor([1, 2])})

        with patch("transfer_queue.interface._maybe_create_transferqueue_client", return_value=mock_client):
            from transfer_queue.interface import kv_get

            kv_get(keys="test_key", partition_id="partition_1")

        # keys is passed directly (not wrapped in list) for single key
        mock_client.kv_retrieve_keys.assert_called_once_with(keys="test_key", partition_id="partition_1", create=False)
        mock_client.get_data.assert_called_once_with(mock_batch_meta)

    def test_kv_get_multiple_keys(self):
        """Test kv_get with multiple keys."""
        mock_client = MagicMock()
        mock_batch_meta = MagicMock()
        mock_client.kv_retrieve_keys.return_value = mock_batch_meta
        mock_client.get_data.return_value = TensorDict({"data": torch.tensor([1, 2])})

        with patch("transfer_queue.interface._maybe_create_transferqueue_client", return_value=mock_client):
            from transfer_queue.interface import kv_get

            keys = ["key1", "key2", "key3"]
            kv_get(keys=keys, partition_id="partition_1")

        mock_client.kv_retrieve_keys.assert_called_once_with(keys=keys, partition_id="partition_1", create=False)

    def test_kv_get_with_fields(self):
        """Test kv_get with specific fields."""
        mock_client = MagicMock()
        mock_batch_meta = MagicMock()
        mock_batch_meta.select_fields = MagicMock(return_value=mock_batch_meta)
        mock_client.kv_retrieve_keys.return_value = mock_batch_meta
        mock_client.get_data.return_value = TensorDict({"text": torch.tensor([1, 2])})

        with patch("transfer_queue.interface._maybe_create_transferqueue_client", return_value=mock_client):
            from transfer_queue.interface import kv_get

            kv_get(keys="test_key", partition_id="partition_1", fields="text")

        mock_batch_meta.select_fields.assert_called_once_with(["text"])


class TestKVList:
    """Tests for kv_list function."""

    def test_kv_list_with_keys(self):
        """Test kv_list returns keys and custom_meta."""
        mock_client = MagicMock()
        mock_client.kv_list.return_value = ["key1", "key2", "key3"]
        mock_batch_meta = MagicMock()
        mock_batch_meta.global_indexes = [0, 1, 2]
        mock_batch_meta.get_all_custom_meta = MagicMock(return_value={0: {}, 1: {}, 2: {}})
        mock_client.kv_retrieve_keys.return_value = mock_batch_meta

        with patch("transfer_queue.interface._maybe_create_transferqueue_client", return_value=mock_client):
            from transfer_queue.interface import kv_list

            keys, custom_meta = kv_list(partition_id="partition_1")

        assert keys == ["key1", "key2", "key3"]
        assert len(custom_meta) == 3

    def test_kv_list_empty_partition(self):
        """Test kv_list returns None when partition is empty."""
        mock_client = MagicMock()
        mock_client.kv_list.return_value = [], []

        with patch("transfer_queue.interface._maybe_create_transferqueue_client", return_value=mock_client):
            from transfer_queue.interface import kv_list

            keys, custom_meta = kv_list(partition_id="empty_partition")

        assert keys == []
        assert custom_meta == []


class TestKVClear:
    """Tests for kv_clear function."""

    def test_kv_clear_single_key(self):
        """Test kv_clear with single key."""
        mock_client = MagicMock()
        mock_batch_meta = MagicMock()
        mock_batch_meta.size = 1
        mock_client.kv_retrieve_keys.return_value = mock_batch_meta

        with patch("transfer_queue.interface._maybe_create_transferqueue_client", return_value=mock_client):
            from transfer_queue.interface import kv_clear

            kv_clear(keys="test_key", partition_id="partition_1")

        mock_client.kv_retrieve_keys.assert_called_once_with(
            keys=["test_key"], partition_id="partition_1", create=False
        )
        mock_client.clear_samples.assert_called_once_with(mock_batch_meta)

    def test_kv_clear_multiple_keys(self):
        """Test kv_clear with multiple keys."""
        mock_client = MagicMock()
        mock_batch_meta = MagicMock()
        mock_batch_meta.size = 3
        mock_client.kv_retrieve_keys.return_value = mock_batch_meta

        with patch("transfer_queue.interface._maybe_create_transferqueue_client", return_value=mock_client):
            from transfer_queue.interface import kv_clear

            kv_clear(keys=["key1", "key2", "key3"], partition_id="partition_1")

        mock_client.kv_retrieve_keys.assert_called_once_with(
            keys=["key1", "key2", "key3"], partition_id="partition_1", create=False
        )
        mock_client.clear_samples.assert_called_once()


class TestAsyncKVPut:
    """Tests for async_kv_put function."""

    def test_async_kv_put_with_fields(self):
        """Test async_kv_put with fields parameter."""
        mock_client = MagicMock()
        mock_batch_meta = MagicMock()
        mock_client.async_kv_retrieve_keys = AsyncMock(return_value=mock_batch_meta)
        mock_client.async_put = AsyncMock()
        mock_client.async_set_custom_meta = AsyncMock()

        tensor_data = TensorDict(
            {"text": torch.tensor([[1, 2, 3]]), "label": torch.tensor([0])},
            batch_size=[1],
        )

        with patch("transfer_queue.interface._maybe_create_transferqueue_client", return_value=mock_client):
            from transfer_queue.interface import async_kv_put

            asyncio.run(
                async_kv_put(key="test_key", partition_id="partition_1", fields=tensor_data, tag={"type": "test"})
            )

        mock_client.async_kv_retrieve_keys.assert_called_once_with(
            keys=["test_key"], partition_id="partition_1", create=True
        )
        mock_batch_meta.update_custom_meta.assert_called_once()
        mock_client.async_put.assert_called_once()

    def test_async_kv_put_with_tag_only(self):
        """Test async_kv_put with only tag (no fields)."""
        mock_client = MagicMock()
        mock_batch_meta = MagicMock()
        mock_client.async_kv_retrieve_keys = AsyncMock(return_value=mock_batch_meta)
        mock_client.async_put = AsyncMock()
        mock_client.async_set_custom_meta = AsyncMock()

        with patch("transfer_queue.interface._maybe_create_transferqueue_client", return_value=mock_client):
            from transfer_queue.interface import async_kv_put

            asyncio.run(async_kv_put(key="test_key", partition_id="partition_1", fields=None, tag={"score": 0.9}))

        mock_client.async_put.assert_not_called()
        mock_client.async_set_custom_meta.assert_called_once_with(mock_batch_meta)


class TestAsyncKVBatchPut:
    """Tests for async_kv_batch_put function."""

    def test_async_kv_batch_put_success(self):
        """Test async_kv_batch_put with valid inputs."""
        mock_client = MagicMock()
        mock_batch_meta = MagicMock()
        mock_client.async_kv_retrieve_keys = AsyncMock(return_value=mock_batch_meta)
        mock_client.async_put = AsyncMock()
        mock_client.async_set_custom_meta = AsyncMock()

        batch_data = TensorDict(
            {
                "text": torch.tensor([[1, 2], [3, 4], [5, 6]]),
                "label": torch.tensor([0, 1, 2]),
            },
            batch_size=[3],
        )

        with patch("transfer_queue.interface._maybe_create_transferqueue_client", return_value=mock_client):
            from transfer_queue.interface import async_kv_batch_put

            keys = ["key1", "key2", "key3"]
            tags = [{"tag": "v1"}, {"tag": "v2"}, {"tag": "v3"}]

            asyncio.run(async_kv_batch_put(keys=keys, partition_id="partition_1", fields=batch_data, tags=tags))

        mock_client.async_kv_retrieve_keys.assert_called_once_with(keys=keys, partition_id="partition_1", create=True)
        mock_batch_meta.update_custom_meta.assert_called_once_with(tags)
        mock_client.async_put.assert_called_once()


class TestAsyncKVGet:
    """Tests for async_kv_get function."""

    def test_async_kv_get_single_key(self):
        """Test async_kv_get with single key."""
        mock_client = MagicMock()
        mock_batch_meta = MagicMock()
        mock_client.async_kv_retrieve_keys = AsyncMock(return_value=mock_batch_meta)
        mock_client.async_get_data = AsyncMock(return_value=TensorDict({"data": torch.tensor([1, 2])}))

        with patch("transfer_queue.interface._maybe_create_transferqueue_client", return_value=mock_client):
            from transfer_queue.interface import async_kv_get

            asyncio.run(async_kv_get(keys="test_key", partition_id="partition_1"))

        # keys is passed directly (not wrapped in list) for single key
        mock_client.async_kv_retrieve_keys.assert_called_once_with(
            keys="test_key", partition_id="partition_1", create=False
        )
        mock_client.async_get_data.assert_called_once_with(mock_batch_meta)


class TestAsyncKVList:
    """Tests for async_kv_list function."""

    def test_async_kv_list_with_keys(self):
        """Test async_kv_list returns keys and custom_meta."""
        mock_client = MagicMock()
        mock_client.async_kv_list = AsyncMock(return_value=["key1", "key2", "key3"])
        mock_batch_meta = MagicMock()
        mock_batch_meta.global_indexes = [0, 1, 2]
        mock_batch_meta.get_all_custom_meta = MagicMock(return_value={0: {}, 1: {}, 2: {}})
        mock_client.async_kv_retrieve_keys = AsyncMock(return_value=mock_batch_meta)

        with patch("transfer_queue.interface._maybe_create_transferqueue_client", return_value=mock_client):
            from transfer_queue.interface import async_kv_list

            keys, custom_meta = asyncio.run(async_kv_list(partition_id="partition_1"))

        assert keys == ["key1", "key2", "key3"]
        assert len(custom_meta) == 3

    def test_async_kv_list_empty_partition(self):
        """Test async_kv_list returns None when partition is empty."""
        mock_client = MagicMock()
        mock_client.async_kv_list = AsyncMock(return_value=[])

        with patch("transfer_queue.interface._maybe_create_transferqueue_client", return_value=mock_client):
            from transfer_queue.interface import async_kv_list

            keys, custom_meta = asyncio.run(async_kv_list(partition_id="empty_partition"))

        assert keys is None
        assert custom_meta is None


class TestAsyncKVClear:
    """Tests for async_kv_clear function."""

    def test_async_kv_clear_single_key(self):
        """Test async_kv_clear with single key."""
        mock_client = MagicMock()
        mock_batch_meta = MagicMock()
        mock_batch_meta.size = 1
        mock_client.async_kv_retrieve_keys = AsyncMock(return_value=mock_batch_meta)
        mock_client.async_clear_samples = AsyncMock()

        with patch("transfer_queue.interface._maybe_create_transferqueue_client", return_value=mock_client):
            from transfer_queue.interface import async_kv_clear

            asyncio.run(async_kv_clear(keys="test_key", partition_id="partition_1"))

        mock_client.async_kv_retrieve_keys.assert_called_once_with(
            keys=["test_key"], partition_id="partition_1", create=False
        )
        mock_client.async_clear_samples.assert_called_once_with(mock_batch_meta)

    def test_async_kv_clear_multiple_keys(self):
        """Test async_kv_clear with multiple keys."""
        mock_client = MagicMock()
        mock_batch_meta = MagicMock()
        mock_batch_meta.size = 3
        mock_client.async_kv_retrieve_keys = AsyncMock(return_value=mock_batch_meta)
        mock_client.async_clear_samples = AsyncMock()

        with patch("transfer_queue.interface._maybe_create_transferqueue_client", return_value=mock_client):
            from transfer_queue.interface import async_kv_clear

            asyncio.run(async_kv_clear(keys=["key1", "key2", "key3"], partition_id="partition_1"))

        mock_client.async_kv_retrieve_keys.assert_called_once()
        mock_client.async_clear_samples.assert_called_once()


class TestKVInterfaceDictConversion:
    """Tests for dict to TensorDict conversion in kv_put."""

    def test_kv_put_with_nontensor_value(self):
        """Test kv_put converts non-tensor values using NonTensorStack."""
        mock_client = MagicMock()
        mock_batch_meta = MagicMock()
        mock_client.kv_retrieve_keys.return_value = mock_batch_meta

        with patch("transfer_queue.interface._maybe_create_transferqueue_client", return_value=mock_client):
            from transfer_queue.interface import kv_put

            # Test with non-tensor value (like a string or list)
            kv_put(
                key="test_key",
                partition_id="partition_1",
                fields={"meta": {"key": "value"}},
                tag=None,
            )

        # Verify put was called
        mock_client.put.assert_called_once()
        call_args = mock_client.put.call_args
        fields_arg = call_args[0][0]
        # The dict should be converted to TensorDict
        assert isinstance(fields_arg, TensorDict)
        assert "meta" in fields_arg

    def test_kv_put_rejects_nested_tensor(self):
        """Test kv_put raises ValueError for nested tensors (requires batch_put)."""
        mock_client = MagicMock()

        with patch("transfer_queue.interface._maybe_create_transferqueue_client", return_value=mock_client):
            from transfer_queue.interface import kv_put

            nested_tensor = torch.nested.nested_tensor([[1, 2], [3, 4]])

            with pytest.raises(ValueError, match="Please use.*kv_batch_put"):
                kv_put(
                    key="test_key",
                    partition_id="partition_1",
                    fields={"nested": nested_tensor},
                    tag=None,
                )

    def test_kv_put_invalid_fields_type(self):
        """Test kv_put raises ValueError for invalid fields type."""
        mock_client = MagicMock()

        with patch("transfer_queue.interface._maybe_create_transferqueue_client", return_value=mock_client):
            from transfer_queue.interface import kv_put

            with pytest.raises(ValueError, match="field can only be dict or TensorDict"):
                kv_put(
                    key="test_key",
                    partition_id="partition_1",
                    fields="invalid_string",
                    tag=None,
                )
