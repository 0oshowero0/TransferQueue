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

"""Unit tests for the high-level KV interface in transfer_queue.interface.

This module tests the kv_batch_get and async_kv_batch_get functions, specifically
the polling and timeout behavior when fields are not immediately available.
"""

import sys
import threading
import time
from pathlib import Path
from threading import Thread
from unittest.mock import patch

import pytest
import torch
import zmq
from tensordict import TensorDict

# Add parent directory to path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from transfer_queue import TransferQueueClient  # noqa: E402
from transfer_queue.metadata import (  # noqa: E402
    BatchMeta,
    FieldMeta,
    SampleMeta,
)
from transfer_queue.utils.enum_utils import ProductionStatus, TransferQueueRole  # noqa: E402
from transfer_queue.utils.zmq_utils import (  # noqa: E402
    ZMQMessage,
    ZMQRequestType,
    ZMQServerInfo,
)

# =============================================================================
# Mock Controllers for Testing Polling/Timeout Behavior
# =============================================================================


class MockControllerWithFieldDelay:
    """Mock controller that simulates fields becoming available over time.

    This mock is used to test the polling behavior of kv_batch_get when
    fields are not immediately available (simulating async writes).
    """

    def __init__(self, controller_id="controller_delay"):
        self.controller_id = controller_id
        self.context = zmq.Context()

        # Socket for data requests
        self.request_socket = self.context.socket(zmq.ROUTER)
        self.request_port = self._bind_to_random_port(self.request_socket)

        self.zmq_server_info = ZMQServerInfo(
            role=TransferQueueRole.CONTROLLER,
            id=controller_id,
            ip="127.0.0.1",
            ports={
                "request_handle_socket": self.request_port,
            },
        )

        self.running = True
        self.request_thread = Thread(target=self._handle_requests, daemon=True)
        self.request_thread.start()

        # Track call counts to simulate delayed field availability
        self.kv_retrieve_call_count = {}
        self._lock = threading.Lock()

    def _bind_to_random_port(self, socket):
        port = socket.bind_to_random_port("tcp://127.0.0.1")
        return port

    def _handle_requests(self):
        poller = zmq.Poller()
        poller.register(self.request_socket, zmq.POLLIN)

        while self.running:
            try:
                socks = dict(poller.poll(100))
                if self.request_socket in socks:
                    messages = self.request_socket.recv_multipart()
                    identity = messages.pop(0)
                    serialized_msg = messages
                    request_msg = ZMQMessage.deserialize(serialized_msg)

                    if request_msg.request_type == ZMQRequestType.KV_RETRIEVE_KEYS:
                        response_body = self._mock_kv_retrieve_keys_delayed(request_msg.body)
                        response_type = ZMQRequestType.KV_RETRIEVE_KEYS_RESPONSE
                    else:
                        response_body = {"error": f"Unknown request type: {request_msg.request_type}"}
                        response_type = ZMQRequestType.CLEAR_META_RESPONSE

                    response_msg = ZMQMessage.create(
                        request_type=response_type,
                        sender_id=self.controller_id,
                        receiver_id=request_msg.sender_id,
                        body=response_body,
                    )
                    self.request_socket.send_multipart([identity, *response_msg.serialize()])
            except zmq.Again:
                continue
            except Exception as e:
                if self.running:
                    print(f"MockControllerWithFieldDelay exception: {e}")
                else:
                    print(f"MockControllerWithFieldDelay ERROR: {e}")
                    raise

    def _mock_kv_retrieve_keys_delayed(self, request_body):
        """Mock KV retrieve that simulates fields becoming available over time."""
        keys = request_body.get("keys", [])
        partition_id = request_body.get("partition_id", "")

        # Create a unique key for tracking call count
        call_key = f"{partition_id}:{','.join(keys) if isinstance(keys, list) else keys}"

        with self._lock:
            if call_key not in self.kv_retrieve_call_count:
                self.kv_retrieve_call_count[call_key] = 0
            self.kv_retrieve_call_count[call_key] += 1
            call_number = self.kv_retrieve_call_count[call_key]

        # Simulate: first 2 calls return only "input_ids", after that return all fields
        if call_number <= 2:
            # Only input_ids available initially
            field_names = ["input_ids"]
        else:
            # All fields available
            field_names = ["input_ids", "attention_mask", "response"]

        # Generate global indexes
        if not hasattr(self, "_kv_index_map"):
            self._kv_index_map = {}
        if partition_id not in self._kv_index_map:
            self._kv_index_map[partition_id] = 0
        start_index = self._kv_index_map.get(partition_id, 0)
        global_indexes = list(range(start_index, start_index + len(keys)))
        self._kv_index_map[partition_id] = global_indexes[-1] + 1

        # Create metadata for each key
        samples = []
        for i, key in enumerate(keys):
            fields = {}
            for field_name in field_names:
                field_meta = FieldMeta(
                    name=field_name,
                    dtype=torch.int64 if field_name == "input_ids" else torch.float32,
                    shape=torch.Size([1, 10]) if field_name == "input_ids" else torch.Size([1, 5]),
                    production_status=ProductionStatus.READY_FOR_CONSUME,
                )
                fields[field_name] = field_meta
            sample = SampleMeta(
                partition_id=partition_id,
                global_index=global_indexes[i],
                fields=fields,
            )
            samples.append(sample)

        metadata = BatchMeta(samples=samples)
        return {"metadata": metadata}

    def reset_call_counts(self):
        """Reset the call count tracking for testing."""
        with self._lock:
            self.kv_retrieve_call_count = {}

    def stop(self):
        self.running = False
        time.sleep(0.2)
        self.request_socket.close()
        self.context.term()


class MockControllerForTimeout:
    """Mock controller that never provides certain fields (for timeout testing)."""

    def __init__(self, controller_id="controller_timeout"):
        self.controller_id = controller_id
        self.context = zmq.Context()

        self.request_socket = self.context.socket(zmq.ROUTER)
        self.request_port = self._bind_to_random_port(self.request_socket)

        self.zmq_server_info = ZMQServerInfo(
            role=TransferQueueRole.CONTROLLER,
            id=controller_id,
            ip="127.0.0.1",
            ports={
                "request_handle_socket": self.request_port,
            },
        )

        self.running = True
        self.request_thread = Thread(target=self._handle_requests, daemon=True)
        self.request_thread.start()

    def _bind_to_random_port(self, socket):
        return socket.bind_to_random_port("tcp://127.0.0.1")

    def _handle_requests(self):
        poller = zmq.Poller()
        poller.register(self.request_socket, zmq.POLLIN)

        while self.running:
            try:
                socks = dict(poller.poll(100))
                if self.request_socket in socks:
                    messages = self.request_socket.recv_multipart()
                    identity = messages.pop(0)
                    serialized_msg = messages
                    request_msg = ZMQMessage.deserialize(serialized_msg)

                    if request_msg.request_type == ZMQRequestType.KV_RETRIEVE_KEYS:
                        response_body = self._mock_kv_retrieve_keys_never_available(request_msg.body)
                        response_type = ZMQRequestType.KV_RETRIEVE_KEYS_RESPONSE
                    else:
                        response_body = {"error": f"Unknown request type: {request_msg.request_type}"}
                        response_type = ZMQRequestType.CLEAR_META_RESPONSE

                    response_msg = ZMQMessage.create(
                        request_type=response_type,
                        sender_id=self.controller_id,
                        receiver_id=request_msg.sender_id,
                        body=response_body,
                    )
                    self.request_socket.send_multipart([identity, *response_msg.serialize()])
            except zmq.Again:
                continue
            except Exception as e:
                if self.running:
                    print(f"MockControllerForTimeout exception: {e}")
                else:
                    print(f"MockControllerForTimeout ERROR: {e}")
                    raise

    def _mock_kv_retrieve_keys_never_available(self, request_body):
        """Mock KV retrieve that never provides certain fields."""
        keys = request_body.get("keys", [])
        partition_id = request_body.get("partition_id", "")

        # Only provide "input_ids" - "attention_mask" and "response" will never be available
        field_names = ["input_ids"]

        if not hasattr(self, "_kv_index_map"):
            self._kv_index_map = {}
        if partition_id not in self._kv_index_map:
            self._kv_index_map[partition_id] = 0
        start_index = self._kv_index_map.get(partition_id, 0)
        global_indexes = list(range(start_index, start_index + len(keys)))
        self._kv_index_map[partition_id] = global_indexes[-1] + 1

        samples = []
        for i, key in enumerate(keys):
            fields = {}
            for field_name in field_names:
                field_meta = FieldMeta(
                    name=field_name,
                    dtype=torch.int64,
                    shape=torch.Size([1, 10]),
                    production_status=ProductionStatus.READY_FOR_CONSUME,
                )
                fields[field_name] = field_meta
            sample = SampleMeta(
                partition_id=partition_id,
                global_index=global_indexes[i],
                fields=fields,
            )
            samples.append(sample)

        metadata = BatchMeta(samples=samples)
        return {"metadata": metadata}

    def stop(self):
        self.running = False
        time.sleep(0.2)
        self.request_socket.close()
        self.context.term()


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_controller_delay():
    """Fixture providing a mock controller with delayed field availability."""
    controller = MockControllerWithFieldDelay()
    yield controller
    controller.stop()


@pytest.fixture
def mock_controller_timeout():
    """Fixture providing a mock controller that never provides certain fields."""
    controller = MockControllerForTimeout()
    yield controller
    controller.stop()


def create_mock_client(mock_controller):
    """Create a TransferQueueClient connected to the given mock controller.

    Note: Storage methods are mocked at high level, so no actual storage is needed.
    """
    client = TransferQueueClient(
        client_id="client_test",
        controller_info=mock_controller.zmq_server_info,
    )

    with patch(
        "transfer_queue.storage.managers.simple_backend_manager.AsyncSimpleStorageManager._connect_to_controller"
    ):
        # Create a dummy zmq_server_info for storage (not actually used since we mock storage methods)
        storage_info = ZMQServerInfo(
            role=TransferQueueRole.STORAGE,
            id="dummy_storage",
            ip="127.0.0.1",
            ports={"put_get_socket": 9999},
        )

        config = {
            "controller_info": mock_controller.zmq_server_info,
            "zmq_info": {"dummy_storage": storage_info},
        }
        client.initialize_storage_manager(manager_type="SimpleStorage", config=config)

        # Mock storage methods at high level
        async def mock_put_data(data, metadata):
            pass

        async def mock_get_data(metadata):
            # Return test data matching the expected fields
            return TensorDict(
                {
                    "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
                    "response": torch.tensor([[100, 101, 102, 103, 104]]),
                },
                batch_size=1,
            )

        async def mock_clear_data(metadata):
            pass

        client.storage_manager.put_data = mock_put_data
        client.storage_manager.get_data = mock_get_data
        client.storage_manager.clear_data = mock_clear_data

    return client


# =============================================================================
# Sync KV Interface Tests
# =============================================================================


class TestKVMixedFieldPolling:
    """Tests for kv_batch_get polling behavior when fields become available."""

    def test_kv_batch_get_polls_until_fields_available(self, mock_controller_delay):
        """Test that kv_batch_get polls and waits for fields to become available.

        This test simulates the scenario where:
        1. Initial kv_retrieve_keys call returns only "input_ids"
        2. Subsequent calls (after polling) return all fields including "response"
        3. kv_batch_get should eventually succeed after polling
        """
        import transfer_queue.interface as interface

        client = create_mock_client(mock_controller_delay)

        # Patch the client creation to use our mock
        original_client = interface._TRANSFER_QUEUE_CLIENT
        try:
            interface._TRANSFER_QUEUE_CLIENT = client

            # This should poll until all fields are available and succeed
            result = interface.kv_batch_get(
                keys="test_key", partition_id="test_partition", fields=["input_ids", "attention_mask", "response"]
            )

            # Verify we got all requested fields
            assert "input_ids" in result.keys()
            assert "attention_mask" in result.keys()
            assert "response" in result.keys()

        finally:
            interface._TRANSFER_QUEUE_CLIENT = original_client

    def test_kv_batch_get_timeout_on_missing_fields(self, mock_controller_timeout):
        """Test that kv_batch_get raises timeout error when fields never become available.

        This test simulates the scenario where:
        1. kv_retrieve_keys only returns "input_ids"
        2. We request "attention_mask" and "response" which never become available
        3. kv_batch_get should raise RuntimeError after timeout
        """
        import transfer_queue.interface as interface
        from transfer_queue.interface import TQ_KV_POLLING_METADATA_TIMEOUT

        # Temporarily reduce timeout for faster test
        original_timeout = TQ_KV_POLLING_METADATA_TIMEOUT
        interface.TQ_KV_POLLING_METADATA_TIMEOUT = 1  # 1 second for testing

        client = create_mock_client(mock_controller_timeout)

        original_client = interface._TRANSFER_QUEUE_CLIENT
        try:
            interface._TRANSFER_QUEUE_CLIENT = client

            with pytest.raises(RuntimeError, match="Timeout for kv_batch_get"):
                interface.kv_batch_get(
                    keys="test_key", partition_id="test_partition", fields=["input_ids", "attention_mask", "response"]
                )

        finally:
            interface._TRANSFER_QUEUE_CLIENT = original_client
            interface.TQ_KV_POLLING_METADATA_TIMEOUT = original_timeout


# =============================================================================
# Async KV Interface Tests
# =============================================================================


@pytest.mark.asyncio
class TestAsyncKVMixedFieldPolling:
    """Tests for async_kv_batch_get polling behavior."""

    async def test_async_kv_batch_get_polls_until_fields_available(self, mock_controller_delay):
        """Test that async_kv_batch_get polls and waits for fields to become available."""
        import transfer_queue.interface as interface

        client = create_mock_client(mock_controller_delay)

        original_client = interface._TRANSFER_QUEUE_CLIENT
        try:
            interface._TRANSFER_QUEUE_CLIENT = client

            # This should poll until all fields are available and succeed
            result = await interface.async_kv_batch_get(
                keys="test_key", partition_id="test_partition", fields=["input_ids", "attention_mask", "response"]
            )

            # Verify we got all requested fields
            assert "input_ids" in result.keys()
            assert "attention_mask" in result.keys()
            assert "response" in result.keys()

        finally:
            interface._TRANSFER_QUEUE_CLIENT = original_client

    async def test_async_kv_batch_get_timeout_on_missing_fields(self, mock_controller_timeout):
        """Test that async_kv_batch_get raises timeout error when fields never become available."""
        import transfer_queue.interface as interface
        from transfer_queue.interface import TQ_KV_POLLING_METADATA_TIMEOUT

        # Temporarily reduce timeout for faster test
        original_timeout = TQ_KV_POLLING_METADATA_TIMEOUT
        interface.TQ_KV_POLLING_METADATA_TIMEOUT = 1  # 1 second for testing

        client = create_mock_client(mock_controller_timeout)

        original_client = interface._TRANSFER_QUEUE_CLIENT
        try:
            interface._TRANSFER_QUEUE_CLIENT = client

            with pytest.raises(RuntimeError, match="Timeout for async_kv_batch_get"):
                await interface.async_kv_batch_get(
                    keys="test_key", partition_id="test_partition", fields=["input_ids", "attention_mask", "response"]
                )

        finally:
            interface._TRANSFER_QUEUE_CLIENT = original_client
            interface.TQ_KV_POLLING_METADATA_TIMEOUT = original_timeout


# =============================================================================
# Run Tests
# =============================================================================


def run_tests():
    """Run all tests manually for debugging."""
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    run_tests()
