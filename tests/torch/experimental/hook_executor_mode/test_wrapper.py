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

from copy import deepcopy

import onnxruntime as ort
import pytest
import torch

from nncf.experimental.torch.hook_executor_mode.hook_storage import HookType
from nncf.experimental.torch.hook_executor_mode.wrapper import ATR_HOOK_STORAGE
from nncf.experimental.torch.hook_executor_mode.wrapper import insert_hook
from nncf.experimental.torch.hook_executor_mode.wrapper import is_wrapped
from nncf.experimental.torch.hook_executor_mode.wrapper import wrap_model
from tests.torch.experimental.hook_executor_mode import helpers


def test_wrapper():
    example_input = helpers.ConvModel.get_example_inputs()
    model = helpers.ConvModel()
    model.eval()
    ret = model(example_input)
    wrapped = wrap_model(model)
    wrapped_ret = wrapped(example_input)
    torch.testing.assert_close(ret, wrapped_ret)


@pytest.mark.parametrize("hook_type", HookType)
def test_insert_hook(hook_type):
    example_input = helpers.ConvModel.get_example_inputs()
    model = helpers.ConvModel()
    wrapped = wrap_model(model)
    import inspect

    print(inspect.signature(wrapped.forward))
    assert is_wrapped(wrapped)

    hook = helpers.CallCount()
    insert_hook(wrapped, hook_type, "hook_group", "/relu/0", 0, hook)
    wrapped(example_input)

    assert hook.call_count == 1


@pytest.mark.parametrize("hook_type", HookType)
def test_insert_hook_twice_raise(hook_type):
    model = helpers.ConvModel()
    wrapped = wrap_model(model)

    hook = helpers.CallCount()
    insert_hook(wrapped, hook_type, "hook_group", "/relu/0", 0, hook)
    with pytest.raises(RuntimeError, match="Hook already set for.*"):
        insert_hook(wrapped, hook_type, "hook_group", "/relu/0", 0, hook)


@pytest.mark.parametrize("hook_type", HookType)
def test_insert_nested_hook(hook_type: HookType):
    example_input = helpers.ConvModel.get_example_inputs()
    model = helpers.ConvModel()
    wrapped = wrap_model(model)

    hook = helpers.CallCount()
    insert_hook(wrapped, hook_type, "hook_group", "/relu/0", 0, helpers.AddModule(2.0))
    insert_hook(
        wrapped,
        hook_type,
        "hook_group",
        f"{ATR_HOOK_STORAGE}:storage:{hook_type.value}:hook_group:-relu-0:0/add/0",
        0,
        hook,
    )
    wrapped(example_input)

    assert hook.call_count == 1


def test_export_strict_false():
    example_input = helpers.ConvModel.get_example_inputs()

    model = helpers.ConvModel()
    return_origin = model(example_input)

    wrapped = wrap_model(model)
    insert_hook(wrapped, HookType.POST_HOOK, "hook_group", "/relu/0", 0, helpers.AddModule(2.0))
    reference = wrapped(example_input)

    m_traced = torch.export.export(wrapped, args=(example_input,), strict=False)
    actual = m_traced.module()(example_input)

    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(actual, return_origin + 2.0)


def test_jit_trace():
    example_input = helpers.ConvModel.get_example_inputs()

    model = helpers.ConvModel()
    return_origin = model(example_input)

    wrapped = wrap_model(model)
    insert_hook(wrapped, HookType.POST_HOOK, "hook_group", "/relu/0", 0, helpers.AddModule(2.0))
    reference = wrapped(example_input)

    m_traced = torch.jit.trace(wrapped, example_inputs=(example_input,), strict=False)
    actual = m_traced(example_input)

    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(actual, return_origin + 2.0)


def test_compile_via_trace():
    example_input = helpers.ConvModel.get_example_inputs()

    model = helpers.ConvModel()
    return_origin = model(example_input)
    wrapped = wrap_model(model)
    insert_hook(wrapped, HookType.POST_HOOK, "hook_group", "/relu/0", 0, helpers.AddModule(2.0))
    reference = wrapped(example_input)
    m_traced = torch.jit.trace(wrapped, example_inputs=(example_input,), strict=False)
    m_compiled = torch.compile(m_traced)
    actual = m_compiled(example_input)

    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(actual, return_origin + 2.0)


def test_export_onnx(tmp_path):
    example_input = helpers.ConvModel.get_example_inputs()

    model = helpers.ConvModel()
    return_origin = model(example_input)

    wrapped = wrap_model(model)
    insert_hook(wrapped, HookType.POST_HOOK, "hook_group", "/relu/0", 0, helpers.AddModule(2.0))
    reference = wrapped(example_input)

    onnx_file = tmp_path / "model.onnx"
    torch.onnx.export(wrapped, (example_input,), onnx_file.as_posix())
    session = ort.InferenceSession(onnx_file)

    actual = session.run(None, {"input": example_input.numpy()})[0]
    torch.testing.assert_close(torch.tensor(actual), reference)
    torch.testing.assert_close(torch.tensor(actual), return_origin + 2.0)


def test_deepcopy():
    example_input = helpers.ConvModel.get_example_inputs()
    model = helpers.get_wrapped_simple_model_with_hook()
    ref = model(example_input)
    copy = deepcopy(model)
    act = copy(example_input)
    torch.testing.assert_close(act, ref)


def test_pickle(tmp_path):
    example_input = helpers.ConvModel.get_example_inputs()
    model = helpers.get_wrapped_simple_model_with_hook()
    ref = model(example_input)

    path = tmp_path / "model.pt"
    torch.save(model, path)
    loaded = torch.load(path, weights_only=False)
    act = loaded(example_input)
    torch.testing.assert_close(act, ref)
