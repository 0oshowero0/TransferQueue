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

import sys

from pathlib import Path

from tensordict import TensorDict, NonTensorStack
import torch
import numpy as np

# Setup path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from transfer_queue.utils.common import dict_to_tensordict  # noqa: E402

def test_dict_to_tensordict():
    print("Test 1: 普通 Tensor 字典")
    try:
        d1 = {"a": torch.randn(2, 4)}
        print(dict_to_tensordict(d1))
    except Exception as e:
        print(f"FAILED: 普通 Tensor 竟然报错: {e}")

    print("\nTest 2: Numpy 数组")
    d2 = {"a": np.array([1, 2]), "b": "tag"}
    td2 = dict_to_tensordict(d2)
    if "a" not in td2.keys():
        print("FAILED: Numpy 数组丢失了")

    print("\nTest 3: 标量/非迭代对象")
    try:
        d3 = {"a": 1, "b": "tag"}
        dict_to_tensordict(d3)
    except Exception as e:
        print(f"FAILED: 传入整数 1 崩溃: {e}")

    print("\nTest 4: 嵌套 Tensor")
    nt = torch.nested.nested_tensor([torch.randn(2), torch.randn(3)])
    d4 = {"a": nt}
    try:
        print(f"Nested success: {dict_to_tensordict(d4)}")
    except Exception as e:
        print(f"FAILED: NestedTensor 报错: {e}")

if __name__ == "__main__":
    test_dict_to_tensordict()