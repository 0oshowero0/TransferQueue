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

import os
import sys
import textwrap
import warnings
from pathlib import Path

warnings.filterwarnings(
    action="ignore",
    message=r"The PyTorch API of nested tensors is in prototype stage*",
    category=UserWarning,
    module=r"torch\.nested",
)

warnings.filterwarnings(
    action="ignore",
    message=r"Tip: In future versions of Ray, Ray will no longer override accelerator visible "
    r"devices env var if num_gpus=0 or num_gpus=None.*",
    category=FutureWarning,
    module=r"ray\._private\.worker",
)


import ray  # noqa: E402
import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402

# Add the parent directory to the path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import transfer_queue as tq  # noqa: E402

# Configure Ray
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_DEBUG"] = "1"

if not ray.is_initialized():
    ray.init(namespace="TransferQueueTutorial")


def demonstrate_kv_api():
    """
    Demonstrate the Key-Value (KV) semantic API:
    kv_put & kv_batch_put -> kv_get -> kv_list -> kv_clear
    """
    print("=" * 80)
    print("Key-Value Semantic API Demo: kv_put → kv_get → kv_list → kv_clear")
    print("=" * 80)

    # Step 1: Put a single key-value pair with kv_put
    print("[Step 1] Putting a single sample with kv_put...")

    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.ones(input_ids.size())

    single_sample = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        batch_size=input_ids.size(0),
    )

    partition_id = "Train"
    key = "0_0"  # User-defined key: "{uid}_{session_id}"
    tag = {"global_steps": 0, "status": "running", "model_version": 0}

    print(f"  Created single sample with key: {key}, fields: {list(single_sample.keys())}, and tag: {tag}")
    print("  Note: kv_put accepts a user-defined string key instead of auto-generated index")

    tq.kv_put(key=key, partition_id=partition_id, fields=single_sample, tag=tag)
    print(f"  ✓ kv_put: key='{key}', tag={tag}")

    # Step 2: Put multiple key-value pairs with kv_batch_put
    print("\n[Step 2] Putting batch data with kv_batch_put...")

    batch_input_ids = torch.tensor(
        [
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
        ]
    )
    batch_attention_mask = torch.ones_like(batch_input_ids)

    data_batch = TensorDict(
        {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
        },
        batch_size=batch_input_ids.size(0),
    )

    keys = ["1_0", "1_1", "1_2", "2_0"]  # 4 keys for 4 samples
    tags = [{"global_steps": 1, "status": "running", "model_version": 1} for _ in range(len(keys))]

    print(f"  Created batch with {data_batch.batch_size[0]} samples")
    print(f"  Batch keys: {keys}")
    tq.kv_batch_put(keys=keys, partition_id=partition_id, fields=data_batch, tags=tags)
    print(f"  ✓ kv_batch_put: {len(keys)} samples written to partition '{partition_id}'")

    # Step 3: Append additional fields to existing samples
    print("\n[Step 3] Appending new fields to existing samples...")

    batch_response = torch.tensor(
        [
            [4, 5, 6],
            [7, 8, 9],
        ]
    )
    response_batch = TensorDict(
        {
            "response": batch_response,
        },
        batch_size=batch_response.size(0),
    )

    append_keys = ["1_1", "2_0"]  # Appending to existing samples
    append_tags = [{"global_steps": 1, "status": "finish", "model_version": 1} for _ in range(len(append_keys))]
    print(f"  Adding 'response' field to keys: {append_keys}")
    tq.kv_batch_put(keys=append_keys, partition_id=partition_id, fields=response_batch, tags=append_tags)
    print("  ✓ Field appended successfully (sample now has input_ids, attention_mask, and response)")

    # Step 4: List all keys and tags in a partition
    print("\n[Step 4] Listing all keys and tags in partition...")
    all_keys, all_tags = tq.kv_list(partition_id=partition_id)
    print(f"  Found {len(all_keys)} keys in partition '{partition_id}':")
    for k, t in zip(all_keys, all_tags, strict=False):
        print(f"    - key='{k}', tag={t}")

    # Step 5: Retrieve specific fields using kv_get
    print("\n[Step 5] Retrieving specific fields with kv_get...")
    retrieved_input_ids = tq.kv_get(keys=all_keys, partition_id=partition_id, fields="input_ids")
    print(f"  Retrieved 'input_ids' field for all {len(all_keys)} samples:")
    print(f"    Shape: {retrieved_input_ids.batch_size}")
    print(f"    Values: {retrieved_input_ids['input_ids']}")

    # TODO: this will fail because only single sample has an extra fields...
    # need to add additional check during kv_get to make sure other samples are correctly tackled
    # # Step 6: Retrieve all fields using kv_get
    # print("\n[Step 6] Retrieving all fields with kv_get...")
    # retrieved_all = tq.kv_get(keys=all_keys, partition_id=partition_id)
    # print(f"  Retrieved all fields for {len(all_keys)} samples:")
    # print(f"    Fields: {list(retrieved_all.keys())}")

    # Step 7: Clear specific keys
    print("\n[Step 7] Clearing keys from partition...")
    keys_to_clear = all_keys[:2]  # Clear first 2 keys
    tq.kv_clear(keys=keys_to_clear, partition_id=partition_id)
    print(f"  ✓ Cleared keys: {keys_to_clear}")

    remaining_keys, _ = tq.kv_list(partition_id=partition_id)
    print(f"  Remaining keys in partition: {remaining_keys}")


def main():
    print("=" * 80)
    print(
        textwrap.dedent(
            """
        TransferQueue Tutorial 2: Key-Value (KV) Semantic API

        This tutorial demonstrates the KV semantic API, which provides a simpler
        interface for data storage and retrieval using user-defined string keys
        instead of auto-generated numeric indexes.

        Key Methods:
        1. kv_put          - Put a single key-value pair with optional metadata tag
        2. kv_batch_put    - Put multiple key-value pairs efficiently in batch
        3. kv_get          - Retrieve data by key(s), optionally specifying fields
        4. kv_list        - List all keys and their metadata tags in a partition
        5. kv_clear        - Remove key-value pairs from storage

        Key Features:
        ✓ User-defined keys      - Use meaningful string keys instead of numeric indexes
        ✓ Fine-grained access    - Get/put individual fields within a sample
        ✓ Partition management   - Each partition maintains its own key-value mapping
        ✓ Metadata tags          - Attach custom metadata (status, scores, etc.) to samples

        Use Cases:
        - Storing per-model-checkpoint states
        - Managing evaluation results by sample ID
        - Caching intermediate computation results
        - Fine-grained data access without full BatchMeta management

        Limitations (vs Full API):
        - No built-in production/consumption tracking (manage via tags)
        - No Sampler-based sampling (implement sampling logic externally)
        - Controller doesn't control streaming (manual key management required)
        """
        )
    )
    print("=" * 80)

    try:
        print("Setting up TransferQueue...")
        tq.init()

        print("\nDemonstrating the KV semantic API...")
        demonstrate_kv_api()

        print("\n" + "=" * 80)
        print("Tutorial Complete!")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("  1. KV API simplifies data access with user-defined string keys")
        print("  2. kv_batch_put is more efficient for bulk operations")
        print("  3. Use 'fields' parameter to get/put specific fields only")
        print("  4. Tags enable custom metadata for production status, scores, etc.")
        print("  5. Use kv_list to inspect partition contents")

        # Cleanup
        tq.close()
        ray.shutdown()
        print("\nCleanup complete")

    except Exception as e:
        print(f"Error during tutorial: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
