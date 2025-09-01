# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Optional, Tuple
import torch
from torch import nn
from torch.overrides import handle_torch_function, has_torch_function_unary



def apply_binary_mask(input_: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if has_torch_function_unary(input_):
        return handle_torch_function(
            apply_binary_mask, (input_,), input_, mask        )

    return input_ * mask

from nncf.torch.layer_utils import StatefulModuleInterface

class BinaryMask(nn.Module, StatefulModuleInterface):
    """
    A module that applies sparsity to its input tensor.
    """
    binary_mask: torch.Tensor

    def __init__(self, shape: Tuple[int, ...], device: Optional[torch.device] = None):
        super(self).__init__()
        self.register_buffer("binary_mask", torch.ones(shape, dtype=torch.bool, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_binary_mask(x, self.binary_mask)




s = BinaryMaskModule([3,3])

t = torch.rand([3,3])


print(s(t))

s.binary_mask = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=torch.bool)
print(s(t))
