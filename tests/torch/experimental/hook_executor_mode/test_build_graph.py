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
from typing import Any, Dict, List, Tuple

import pytest
import torch
from torch import nn

from nncf.experimental.torch.hook_executor_mode.build_graph_mode import GraphBuilderMode
from nncf.experimental.torch.hook_executor_mode.build_graph_mode import TensorInfo
from nncf.experimental.torch.hook_executor_mode.graph_utils import ConstMeta
from nncf.experimental.torch.hook_executor_mode.graph_utils import EdgeMeta
from nncf.experimental.torch.hook_executor_mode.graph_utils import FunctionMeta
from nncf.experimental.torch.hook_executor_mode.graph_utils import InOutMeta
from nncf.experimental.torch.hook_executor_mode.graph_utils import NodeType
from nncf.experimental.torch.hook_executor_mode.graph_utils import TensorDesc
from nncf.experimental.torch.hook_executor_mode.graph_utils import TensorMeta
from nncf.experimental.torch.hook_executor_mode.hook_executor_mode import OpMeta


@dataclass
class ParamArgKwargInput:
    name: str
    args: Tuple[Any]
    kwargs: Dict[str, Any]
    ref: List[Any]

    @staticmethod
    def idfn(val):
        if isinstance(val, ParamArgKwargInput):
            return val.name


@pytest.mark.parametrize(
    "param",
    (
        ParamArgKwargInput(
            name="arg_1_tensor",
            args=(torch.ones((1, 1), dtype=torch.int8),),
            kwargs={},
            ref=[
                {
                    "type": NodeType.input,
                    "meta": InOutMeta(shape=(1, 1), dtype=torch.int8, name="arg_0"),
                }
            ],
        ),
        ParamArgKwargInput(
            name="arg_2_tensor",
            args=(torch.ones((1, 1), dtype=torch.float32), torch.ones((1), dtype=torch.int8)),
            kwargs={},
            ref=[
                {
                    "type": NodeType.input,
                    "meta": InOutMeta(shape=(1, 1), dtype=torch.float32, name="arg_0"),
                },
                {
                    "type": NodeType.input,
                    "meta": InOutMeta(shape=(1,), dtype=torch.int8, name="arg_1"),
                },
            ],
        ),
        ParamArgKwargInput(
            name="arg_tuple_tensor",
            args=((torch.ones((1, 1), dtype=torch.float32), torch.ones((1), dtype=torch.int8)),),
            kwargs={},
            ref=[
                {
                    "type": NodeType.input,
                    "meta": InOutMeta(shape=(1, 1), dtype=torch.float32, name="arg_0_0"),
                },
                {
                    "type": NodeType.input,
                    "meta": InOutMeta(shape=(1,), dtype=torch.int8, name="arg_0_1"),
                },
            ],
        ),
        ParamArgKwargInput(
            name="kwargs_tensor",
            args=(),
            kwargs={"x": torch.ones((1, 1), dtype=torch.float32), "y": torch.ones((1), dtype=torch.int8)},
            ref=[
                {
                    "type": NodeType.input,
                    "meta": InOutMeta(shape=(1, 1), dtype=torch.float32, name="kwarg_x"),
                },
                {
                    "type": NodeType.input,
                    "meta": InOutMeta(shape=(1,), dtype=torch.int8, name="kwarg_y"),
                },
            ],
        ),
    ),
    ids=ParamArgKwargInput.idfn,
)
def test_register_model_inputs(param: ParamArgKwargInput):
    mode = GraphBuilderMode(nn.Identity(), nn.Identity())
    mode.register_model_inputs(*param.args, **param.kwargs)
    for act, r in zip(mode.graph.nodes(data=True), param.ref):
        assert act[1] == r


class TensorInfoMock:
    def __init__(self, source_node_id=0, source_type=TensorDesc.input):
        self.data = []
        self.source_node_id = source_node_id
        self.source_type = source_type

    def __get__(self, tensor: torch.Tensor):
        if not isinstance(tensor, torch.Tensor):
            return None
        return TensorInfo(
            source_type=self.source_type,
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            source_node_id=self.source_node_id,
            output_port_id=0,
            source_name="_model.weight",
        )

    def get(self, tensor: torch.Tensor, default=None):
        return self.__get__(tensor)


@dataclass
class ParamModelOutput:
    name: str
    outputs: Any
    ref: List[Any]

    @staticmethod
    def idfn(val):
        if isinstance(val, ParamModelOutput):
            return val.name


@pytest.mark.parametrize(
    "param",
    (
        ParamModelOutput(
            name="tensor",
            outputs=torch.ones((1, 1), dtype=torch.int8),
            ref={
                "nodes": [
                    {
                        "type": NodeType.output,
                        "meta": InOutMeta(shape=(1, 1), dtype=torch.int8, name="output"),
                    }
                ],
                "edges": [{"meta": EdgeMeta(dtype=torch.int8, shape=(1, 1), input_port=0, output_port=0)}],
            },
        ),
        ParamModelOutput(
            name="tuple_of_tensor",
            outputs=(torch.ones((1, 1), dtype=torch.int32), torch.ones((1, 1), dtype=torch.int8)),
            ref={
                "nodes": [
                    {
                        "type": NodeType.output,
                        "meta": InOutMeta(shape=(1, 1), dtype=torch.int32, name="output_0"),
                    },
                    {
                        "type": NodeType.output,
                        "meta": InOutMeta(shape=(1, 1), dtype=torch.int8, name="output_1"),
                    },
                ],
                "edges": [
                    {"meta": EdgeMeta(dtype=torch.int32, shape=(1, 1), input_port=0, output_port=0)},
                    {"meta": EdgeMeta(dtype=torch.int8, shape=(1, 1), input_port=0, output_port=0)},
                ],
            },
        ),
        ParamModelOutput(
            name="dict_of_tensor",
            outputs={"x": torch.ones((1, 1), dtype=torch.int32), "y": torch.ones((1, 1), dtype=torch.int8)},
            ref={
                "nodes": [
                    {
                        "type": NodeType.output,
                        "meta": InOutMeta(shape=(1, 1), dtype=torch.int32, name="output_x"),
                    },
                    {
                        "type": NodeType.output,
                        "meta": InOutMeta(shape=(1, 1), dtype=torch.int8, name="output_y"),
                    },
                ],
                "edges": [
                    {"meta": EdgeMeta(dtype=torch.int32, shape=(1, 1), input_port=0, output_port=0)},
                    {"meta": EdgeMeta(dtype=torch.int8, shape=(1, 1), input_port=0, output_port=0)},
                ],
            },
        ),
        ParamModelOutput(
            name="tuple_tensor_dict",
            outputs=(torch.ones((1, 1), dtype=torch.int32), {"y": torch.ones((1,), dtype=torch.int8)}),
            ref={
                "nodes": [
                    {
                        "type": NodeType.output,
                        "meta": InOutMeta(shape=(1, 1), dtype=torch.int32, name="output_0"),
                    },
                    {
                        "type": NodeType.output,
                        "meta": InOutMeta(shape=(1,), dtype=torch.int8, name="output_1_y"),
                    },
                ],
                "edges": [
                    {"meta": EdgeMeta(dtype=torch.int32, shape=(1, 1), input_port=0, output_port=0)},
                    {"meta": EdgeMeta(dtype=torch.int8, shape=(1,), input_port=0, output_port=0)},
                ],
            },
        ),
    ),
    ids=ParamModelOutput.idfn,
)
def test_register_model_outputs(param: ParamModelOutput):
    mode = GraphBuilderMode(nn.Identity(), nn.Identity())
    mode.tensor_info = TensorInfoMock()
    mode.register_model_outputs(param.outputs)
    for act, r in zip(mode.graph.nodes(data=True), param.ref["nodes"]):
        assert act[1] == r
    for act, r in zip(mode.graph.edges(data=True), param.ref["edges"]):
        assert act[2] == r


@pytest.mark.parametrize(
    "param",
    (
        ParamArgKwargInput(
            name="arg_1_tensor",
            args=(torch.ones((1, 1), dtype=torch.int8),),
            kwargs={},
            ref={
                "nodes": [
                    {
                        "type": NodeType.fn_call,
                        "meta": FunctionMeta(
                            op_name="_model.foo",
                            fn_name="add",
                            args=[TensorMeta(dtype=torch.int8, shape=(1, 1))],
                            kwargs={},
                        ),
                    }
                ],
                "edges": [{"meta": EdgeMeta(dtype=torch.int8, shape=(1, 1), input_port=0, output_port=0)}],
            },
        ),
        ParamArgKwargInput(
            name="arg_2_tensor_int",
            args=(torch.ones((1, 1), dtype=torch.int8), 2),
            kwargs={},
            ref={
                "nodes": [
                    {
                        "type": NodeType.fn_call,
                        "meta": FunctionMeta(
                            op_name="_model.foo",
                            fn_name="add",
                            args=[
                                TensorMeta(dtype=torch.int8, shape=(1, 1)),
                                2,
                            ],
                            kwargs={},
                        ),
                    }
                ],
                "edges": [
                    {"meta": EdgeMeta(dtype=torch.int8, shape=(1, 1), input_port=0, output_port=0)},
                    {"meta": EdgeMeta(dtype=torch.float32, shape=(1,), input_port=1, output_port=0)},
                ],
            },
        ),
        ParamArgKwargInput(
            name="arg_tensor_kwarg_tensor",
            args=(torch.ones((1, 1), dtype=torch.int8),),
            kwargs={"x": torch.ones((1,), dtype=torch.float32), "y": 2},
            ref={
                "nodes": [
                    {
                        "type": NodeType.fn_call,
                        "meta": FunctionMeta(
                            op_name="_model.foo",
                            fn_name="add",
                            args=[TensorMeta(dtype=torch.int8, shape=(1, 1))],
                            kwargs={
                                "x": TensorMeta(dtype=torch.float32, shape=(1,)),
                                "y": 2,
                            },
                        ),
                    }
                ],
                "edges": [
                    {"meta": EdgeMeta(dtype=torch.float32, shape=(1,), input_port=1, output_port=0)},
                    {"meta": EdgeMeta(dtype=torch.int8, shape=(1, 1), input_port=0, output_port=0)},
                ],
            },
        ),
    ),
    ids=ParamArgKwargInput.idfn,
)
def test_process_op_inputs(param: ParamArgKwargInput):
    mode = GraphBuilderMode(nn.Identity(), nn.Identity())
    mode.tensor_info = TensorInfoMock()
    op_meta = OpMeta(op_name="_model.foo", func=torch.add)
    mode.process_op_inputs(param.args, param.kwargs, op_meta)

    for act, r in zip(mode.graph.nodes(data=True), param.ref["nodes"]):
        assert act[1] == r
    for act, r in zip(mode.graph.edges(data=True), param.ref["edges"]):
        assert act[2] == r


@pytest.mark.parametrize(
    "tensor, ref",
    (
        (
            torch.ones((1, 1), dtype=torch.int8),
            {
                "type": NodeType.const,
                "meta": ConstMeta(dtype=torch.int8, shape=(1, 1), name_in_model="_model.weight"),
            },
        ),
    ),
)
@pytest.mark.parametrize("tensor_desc", (TensorDesc.parameter, TensorDesc.buffer))
def test_maybe_add_node_for_parameters(tensor, ref, tensor_desc: TensorDesc):
    mode = GraphBuilderMode(nn.Identity(), nn.Identity())
    mode.tensor_info = TensorInfoMock(source_node_id=None, source_type=tensor_desc)
    mode._maybe_add_node_for_parameters(tensor)

    for act, r in zip(mode.graph.nodes(data=True), [ref]):
        assert act[1] == r
    assert len(mode.graph.edges()) == 0
