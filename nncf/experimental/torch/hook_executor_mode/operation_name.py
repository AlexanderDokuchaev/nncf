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

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

SEP_CHAR = "/"


@dataclass
class OperationName:
    """
    Represents the name of an operation consisting of a module name, function name, and an optional call ID.

    :param module_name: The name of the module.
    :param fn_name: The name of the function.
    :param call_id: The optional call ID associated with the operation.
    """

    module_name: str
    fn_name: str
    call_id: Optional[int] = None

    def __str__(self) -> str:
        """
        Returns a string representation of the OperationName instance.
        Dots will be replaced to two dots, because ModuleDict restricts names with dots.

        Returns: The string representation in the format "module_name/fn_name" or "module_name/fn_name/call_id".
        """
        m_name = ":".join(self.module_name.split("."))  # ModuleDict restricts names with dots
        m_name = "-".join(m_name.split(SEP_CHAR))
        if self.call_id is None:
            return f"{m_name}{SEP_CHAR}{self.fn_name}"
        return f"{m_name}{SEP_CHAR}{self.fn_name}{SEP_CHAR}{self.call_id}"

    def to_str(self) -> str:
        return str(self)
