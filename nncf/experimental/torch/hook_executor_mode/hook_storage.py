# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
from typing import MutableMapping

import torch
from torch import nn

# storage[HOOK_TYPE][GROUP][OP_NAME][PORT_ID]
TYPE_STORAGE = MutableMapping[str, MutableMapping[str, MutableMapping[str, MutableMapping[str, nn.Module]]]]


class HookType(Enum):
    """
    Enumeration for defining types of hooks.
    """

    PRE_HOOK = "pre_hook"
    POST_HOOK = "post_hook"

    def __str__(self) -> str:
        """
        Return the string representation of the HookType.
        """
        return self.value


class HookStorage(nn.Module):
    """
    A module for storing and executing hooks.

    The storage structure is defined as:
    storage[HOOK_TYPE][GROUP][OP_NAME][PORT_ID]

    Attributes:
        storage (nn.ModuleDict): A nested dictionary structure for storing hooks.
    """

    def __init__(self) -> None:
        """
        Initialize an empty HookStorage.
        """
        super().__init__()
        self.storage: TYPE_STORAGE = nn.ModuleDict()

    def insert_hook(self, hook_type: HookType, group_name: str, op_name: str, port_id: int, module: nn.Module) -> None:
        """
        Insert a hook module into the storage.

        :param hook_type (HookType): The type of the hook (pre-hook or post-hook).
        :param group_name (str): The group name under which the hook is categorized.
        :param op_name (str): The operation name the hook is associated with.
        :param port_id (int): The port ID the hook is associated with.
        :param module (nn.Module): The hook module to be stored.
        """
        if hook_type.value not in self.storage:
            self.storage[hook_type.value] = nn.ModuleDict()

        storage = self.storage[hook_type.value]
        if group_name not in storage:
            storage[group_name] = nn.ModuleDict()
        if op_name not in storage[group_name]:
            storage[group_name][op_name] = nn.ModuleDict()
        str_port_id = str(port_id)
        if str_port_id in storage[group_name][op_name]:
            raise RuntimeError(f"Hook already set for {hook_type=} {group_name=} {op_name=} {port_id=}")
        storage[group_name][op_name][str_port_id] = module

    def execute_hook(self, hook_type: HookType, op_name: str, port_id: int, value: torch.Tensor) -> torch.Tensor:
        """
        Execute hooks stored by group_name for a specific operation and port.

        :param hook_type (HookType): The type of the hook to execute (pre-hook or post-hook).
        :param op_name (str): The operation name the hook is associated with.
        :param port_id (int): The port ID the hook is associated with.
        :param value (torch.Tensor): The tensor value to be processed by the hook.

        Returns:
            torch.Tensor: The processed tensor value after all applicable hooks have been applied.
        """
        if hook_type.value not in self.storage:
            return value

        storage = self.storage[hook_type.value]
        str_port_id = str(port_id)
        for group_name in sorted(storage):
            if op_name in storage[group_name] and str_port_id in storage[group_name][op_name]:
                value = storage[group_name][op_name][str_port_id](value)
        return value

    def remove_group(self, group_name: str) -> None:
        """
        Remove all hooks associated with a specific group name.

        Args:
            group_name (str): The group name to remove from the storage.
        """
        if HookType.PRE_HOOK.value in self.storage:
            self.storage[HookType.PRE_HOOK.value].pop(group_name)
        if HookType.POST_HOOK.value in self.storage:
            self.storage[HookType.POST_HOOK.value].pop(group_name)
