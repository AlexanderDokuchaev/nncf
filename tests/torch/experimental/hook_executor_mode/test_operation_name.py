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


from dataclasses import dataclass
from typing import Any, Optional

import pytest

from nncf.experimental.torch.hook_executor_mode.operation_name import OperationName


@dataclass
class OperationNameTestCase:
    model_name: str
    fn_name: str
    call_id: Optional[str]
    ref: str


def idfn(value: Any):
    if isinstance(value, OperationNameTestCase):
        return value.ref
    return None


@pytest.mark.parametrize(
    "param",
    (
        OperationNameTestCase("module", "foo", None, "module/foo"),
        OperationNameTestCase("module", "foo", 0, "module/foo/0"),
        OperationNameTestCase("module", "foo", 1, "module/foo/1"),
        OperationNameTestCase("module.module", "foo", None, "module:module/foo"),
        OperationNameTestCase("module.module", "foo", 0, "module:module/foo/0"),
        OperationNameTestCase("hook.module:module/foo", "foo", None, "hook:module:module-foo/foo"),
    ),
    ids=idfn,
)
def test_operation_name(param: OperationNameTestCase):
    op_name = OperationName(module_name=param.model_name, fn_name=param.fn_name, call_id=param.call_id)
    assert str(op_name) == param.ref
    assert op_name.to_str() == param.ref
