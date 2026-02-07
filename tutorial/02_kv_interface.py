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
    Demonstrate xxxxx
    """
    print("=" * 80)
    print("Data xxxx")
    print("=" * 80)

    # Step 1: Put single data sample
    print("[Step 1] Putting single data into TransferQueue...")

    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.ones(input_ids.size())

    single_sample = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        batch_size=input_ids.size(0),
    )

    print(f"  Created single sample with multiple fields:{single_sample.keys()}.")
    print("  Leveraging TransferQueue, we can provide fine-grained access into each single field of a sample (key).")

    partition_id = "Train"
    key = [f"{uid}_{session_id}" for uid in [0] for session_id in [0]]
    tag = [{"global_steps": 0, "status": "running", "model_version": 0}]
    tq.kv_put(key, partition_id=partition_id, fields=single_sample, tag=tag)
    print(f"  ✓ Data put to partition: {partition_id}")

    # Step 2: Put data batch
    print("[Step 2] Putting multiple data into TransferQueue...")

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

    partition_id = "Train"
    keys = [f"{uid}_{session_id}" for uid in [1, 1, 1, 2] for session_id in [0, 1, 2, 0]]
    tags = [{"global_steps": 1, "status": "running", "model_version": 1} for _ in range(len(keys))]

    print(f"  Created {data_batch.batch_size[0]} samples, assigning keys: {keys}, tags: {tags}")
    tq.kv_batch_put(keys, partition_id=partition_id, fields=data_batch, tags=tags)
    print(f"  ✓ Data put to partition: {partition_id}")

    # Step 3: Append new data fields to existing samples
    print("[Step 3] Putting multiple data into TransferQueue...")
    batch_response = torch.tensor(
        [
            [4, 5, 6],
            [7, 8, 9],
        ]
    )
    data_batch = TensorDict(
        {
            "response": batch_response,
        },
        batch_size=batch_response.size(0),
    )

    keys = [f"{uid}_{session_id}" for uid in [1, 2] for session_id in [1, 0]]
    tags = [{"global_steps": 1, "status": "finish", "model_version": 1} for _ in range(len(keys))]

    tq.kv_batch_put(keys, partition_id=partition_id, fields=data_batch, tags=tags)

    # Step 4: Query all keys and tags
    print("[Step 4] Query all the keys and tags from TransferQueue...")
    all_keys, all_tags = tq.kv_list(partition_id=partition_id)
    print(f"  ✓ Got keys: {keys}")
    print(f"    Got tags: {tags}")

    # Step 5: Get specific fields of values
    print("[Step 5] ...")
    retrieved_input_ids_data = tq.kv_get(all_keys, fields="input_ids")
    print("  ✓ Data retrieved successfully")
    print(f"    {retrieved_input_ids_data}")

    # Step 6: Get all fields of values
    print("[Step 5] ...")
    retrieved_all_data = tq.kv_get(all_keys)
    print("  ✓ Data retrieved successfully")
    print(f"    {retrieved_all_data}")

    # Step 5: Clear
    print("[Step 5] Clearing partition...")
    tq.kv_clear(all_keys, partition_id=partition_id)
    print("  ✓ Keys are cleared")


def main():
    print("=" * 80)
    print(
        textwrap.dedent(
            """
        TransferQueue Tutorial 2: Key-Value Semantic API
    
        This script demonstrate the key-value semantic API of TransferQueue:
        1. kv_put & kv_batch_put - Put key-value pairs and custom tags into TransferQueue
        2. kv_get - Get values from TransferQueue according to user-specified keys
        3. kv_list - Get all the keys and tags from TransferQueue
        4. kv_clear - Delete the value and tags of given keys
    
        Supported Features:
        1. Fine-grained access - user can put/get partial fields inside a data sample (key)
        2. Partition management - each logical partition manages their own key-value mapping
        
        Unsupported Features:
        1. Production & consumption management (user have to manually management through tags)
        2. User-defined sampler in TransferQueue controller (user need to do sampling by themselves through tags)
        3. Fully streamed data pipeline (TQ controller cannot determine which sample to dispatch to the consumers)
        
        """
        )
    )
    print("=" * 80)

    try:
        print("Setting up TransferQueue...")
        tq.init()

        print("Demonstrating the key-value semantic API...")
        demonstrate_kv_api()

        print("=" * 80)
        print("Tutorial Complete!")
        print("=" * 80)
        print("Key Takeaways:")
        print("1. ")

        # Cleanup
        tq.close()
        ray.shutdown()
        print("\n✓ Cleanup complete")

    except Exception as e:
        print(f"Error during tutorial: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
