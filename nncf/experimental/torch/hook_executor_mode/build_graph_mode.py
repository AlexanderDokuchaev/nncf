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

from itertools import chain
from typing import Any, Callable, Dict, MutableMapping, Tuple
from weakref import ref

import networkx as nx
import torch
from torch import nn

import nncf
from nncf.common.logging import nncf_logger as logger
from nncf.experimental.torch.hook_executor_mode.graph_utils import ConstMeta
from nncf.experimental.torch.hook_executor_mode.graph_utils import EdgeMeta
from nncf.experimental.torch.hook_executor_mode.graph_utils import FunctionMeta
from nncf.experimental.torch.hook_executor_mode.graph_utils import InOutMeta
from nncf.experimental.torch.hook_executor_mode.graph_utils import NodeType
from nncf.experimental.torch.hook_executor_mode.graph_utils import TensorDesc
from nncf.experimental.torch.hook_executor_mode.graph_utils import TensorInfo
from nncf.experimental.torch.hook_executor_mode.graph_utils import TensorMeta
from nncf.experimental.torch.hook_executor_mode.hook_executor_mode import HookExecutorMode
from nncf.experimental.torch.hook_executor_mode.hook_executor_mode import OpMeta
from nncf.experimental.torch.hook_executor_mode.hook_storage import HookStorage
from nncf.experimental.torch.hook_executor_mode.weak_map import WeakUnhashableKeyMap
from nncf.experimental.torch.hook_executor_mode.wrapper import get_hook_storage
from nncf.experimental.torch.hook_executor_mode.wrapper import is_wrapped


class GraphBuilderMode(HookExecutorMode):

    def __init__(self, model: nn.Module, hook_storage: HookStorage):
        """
        Initialize the GraphBuilderMode.

        :param model: The PyTorch model to which the hooks will be applied.
        :param hook_storage: Storage for hooks to be executed.
        """
        super().__init__(model=model, hook_storage=hook_storage)
        self.cur_node_id = 0
        self.graph = nx.DiGraph()
        self.tensor_info: MutableMapping[ref[torch.Tensor], TensorInfo] = WeakUnhashableKeyMap()

        for name, parameter in self.model.named_parameters():
            self.tensor_info[parameter] = TensorInfo(
                source_type=TensorDesc.parameter,
                shape=tuple(parameter.shape),
                dtype=parameter.dtype,
                source_node_id=None,
                output_port_id=0,
                source_name=name,
            )
        for name, parameter in self.model.named_buffers():
            self.tensor_info[parameter] = TensorInfo(
                source_type=TensorDesc.buffer,
                shape=tuple(parameter.shape),
                dtype=parameter.dtype,
                source_node_id=None,
                output_port_id=0,
                source_name=name,
            )

    def _get_new_node_id(self) -> int:
        """Return unique id for new node"""
        key = self.cur_node_id
        self.cur_node_id += 1
        return key

    def execute_pre_hooks(
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any], op_meta: OpMeta
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        _args, _kwargs = super().execute_pre_hooks(args, kwargs, op_meta)
        self.process_op_inputs(_args, _kwargs, op_meta)
        return _args, _kwargs

    def execute_post_hooks(self, outputs: Any, op_meta: OpMeta) -> Any:
        self._process_magic_function(outputs, op_meta)
        self.process_op_outputs(outputs, op_meta)
        outputs = super().execute_post_hooks(outputs, op_meta)
        return outputs

    def __enter__(self) -> GraphBuilderMode:
        """Overload __enter__ to get correct return type hint"""
        super().__enter__()
        return self

    def _register_input_tensor(self, tensor: torch.Tensor, op_node_id: int, port_id: int, op_meta: OpMeta) -> None:
        """
        Registers an input tensor in the graph.

        :param tensor: The input tensor to register.
        :param op_node_id: The ID of the operation node.
        :param port_id: The port ID for the tensor input.
        """
        if not isinstance(tensor, torch.Tensor):
            return
        tensor_info = self.tensor_info.get(tensor, None)
        if tensor_info is None:
            logger.debug(f"No found tensor info for {op_meta.op_name} {op_meta.func.__name__}")

            # import traceback

            # print(f"------------: {op_meta.op_name} {op_meta.func.__name__}")
            # for line in traceback.format_stack()[-8:-3]:
            #     print(line.strip())
            return
        self.graph.add_edge(
            tensor_info.source_node_id,
            op_node_id,
            meta=EdgeMeta.from_tensor(tensor, input_port=port_id, output_port=tensor_info.output_port_id),
        )

    def _maybe_add_node_for_parameters(self, tensor: torch.Tensor) -> None:
        """
        Adds a node for parameters or buffers used for the first time.

        :param tensor: The tensor to add as a node.
        """
        if not isinstance(tensor, torch.Tensor):
            return
        tensor_info = self.tensor_info.get(tensor)
        if tensor_info is not None and tensor_info.source_node_id is None:
            # Add not for parameter or buffer that used in first time
            node_id = self._get_new_node_id()
            self.graph.add_node(
                node_id,
                type=NodeType.const,
                meta=ConstMeta.from_tensor(tensor=tensor, name_in_model=tensor_info.source_name),
            )
            tensor_info.source_node_id = node_id
            logger.debug(f"GraphBuilderMode._maybe_add_node_for_parameters: {node_id=} {tensor_info.source_name=}")

    def process_op_inputs(self, args: Tuple[Any], kwargs: Dict[str, Any], op_meta: OpMeta) -> None:
        """
        Processes the inputs of an operation and updates the graph.

        :param args: The positional arguments for the operation.
        :param kwargs: The keyword arguments for the operation.
        :param op_meta: Metadata about the operation.
        """
        for arg in chain(args, kwargs.values()):
            self._maybe_add_node_for_parameters(arg)

        node_id = self._get_new_node_id()
        op_meta.extra_info["node_id"] = node_id
        op_name = str(op_meta.op_name)

        op_attrs = []
        op_kwargs = {}

        for port_id, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                self._register_input_tensor(arg, node_id, port_id, op_meta)
                op_attrs.append(TensorMeta.from_tensor(arg))
            elif isinstance(arg, (list, tuple, set)) and any(isinstance(x, torch.Tensor) for x in arg):
                op_attr = []
                for x in arg:
                    if isinstance(x, torch.Tensor):
                        self._register_input_tensor(x, node_id, port_id, op_meta)
                        op_attr.append(TensorMeta.from_tensor(x))
                    else:
                        op_attr.append(x)
                op_attrs.append(op_attr)
            else:
                op_attrs.append(arg)

        for port_id, (name, value) in enumerate(kwargs.items(), start=len(args)):
            if isinstance(value, torch.Tensor):
                self._register_input_tensor(value, node_id, port_id, op_meta)
                op_kwargs[name] = TensorMeta.from_tensor(value)
            else:
                op_kwargs[name] = value

        self.graph.add_node(
            node_id,
            type=NodeType.fn_call,
            meta=FunctionMeta(op_name=op_name, fn_name=op_meta.func.__name__, args=op_attrs, kwargs=op_kwargs),
        )

        logger.debug(f"GraphBuilderMode.process_op_inputs: {node_id=} {op_name=} {op_attrs=} {op_kwargs=}")

    def _process_magic_function(self, output: torch.Tensor, op_meta: OpMeta) -> None:
        """
        Processes special functions like attribute access on tensors using grad_fn.

        :param output: The output tensor.
        :param op_meta: Metadata about the operation.
        """
        if op_meta.func.__name__ == "__get__" and isinstance(output, torch.Tensor) and output.grad_fn is not None:
            # To detect .T, .mT, .H, .mH attributes of tensor
            fn_name = None
            fn_args = None
            if output.grad_fn.name() == "TransposeBackward0":
                fn_name = "transpose"
                fn_args = {
                    "dim0": -(2**64 - output.grad_fn._saved_dim0),
                    "dim1": -(2**64 - output.grad_fn._saved_dim1),
                }
            if output.grad_fn.name() == "PermuteBackward0":
                fn_name = "permute"
                fn_args = {"dims": output.grad_fn._saved_dims}

            if fn_name is not None:
                self.graph.nodes[op_meta.extra_info["node_id"]]["op_name"] = fn_name
            if fn_args is not None:
                self.graph.nodes[op_meta.extra_info["node_id"]]["op_attrs"] = fn_args

    def process_op_outputs(self, outputs: Any, op_meta: OpMeta) -> None:
        """
        Processes the outputs of an operation and updates the graph.

        :param outputs: The outputs of the operation.
        :param op_meta: Metadata about the operation.
        """
        if isinstance(outputs, torch.Tensor):
            logger.debug("GraphBuilderMode.process_op_outputs: Tensor")

            self.tensor_info[outputs] = TensorInfo(
                source_type=TensorDesc.tensor,
                shape=tuple(outputs.shape),
                dtype=outputs.dtype,
                source_node_id=op_meta.extra_info["node_id"],
                output_port_id=0,
                source_name=None,
            )
        elif isinstance(outputs, (tuple, list)) and any(isinstance(t, torch.Tensor) for t in outputs):
            for idx, t in enumerate(outputs):
                logger.debug(f"GraphBuilderMode.process_op_outputs: {type(t)}")

                if isinstance(t, torch.Tensor):
                    self.tensor_info[t] = TensorInfo(
                        source_type=TensorDesc.tensor,
                        shape=tuple(t.shape),
                        dtype=t.dtype,
                        source_node_id=op_meta.extra_info["node_id"],
                        output_port_id=idx,
                        source_name=None,
                    )
        else:
            logger.debug(f"GraphBuilderMode.process_op_outputs: {type(outputs)}")
            if op_meta.func.__name__ == "__get__":
                self.graph.remove_node(op_meta.extra_info["node_id"])

    def register_model_inputs(self, *args: Tuple[Any], **kwargs: Dict[str, Any]) -> None:
        """
        Registers the inputs of the model in the graph.
        """
        for idx, arg in enumerate(args):
            _call_fn_for_each_tensors(arg, f"arg_{idx}", self._register_model_input)
        for in_name, val in kwargs.items():
            _call_fn_for_each_tensors(val, f"kwarg_{in_name}", self._register_model_input)

    def _register_model_input(self, name: str, tensor: torch.Tensor) -> None:
        """
        Registers a single model input tensor in the graph.

        :param name: The name of the input tensor.
        :param tensor: The input tensor to register.
        """
        node_id = self._get_new_node_id()
        self.tensor_info[tensor] = TensorInfo(
            source_type=NodeType.input,
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            source_node_id=node_id,
            output_port_id=0,
            source_name=None,
        )
        self.graph.add_node(node_id, type=NodeType.input, meta=InOutMeta.from_tensor(tensor, name))
        logger.debug(f"GraphBuilderMode._register_model_input: {node_id=} {name=}")

    def register_model_outputs(self, outputs: Any) -> None:
        """
        Registers the outputs of the model in the graph.

        :param  outputs: The model outputs to register.
        """
        _call_fn_for_each_tensors(outputs, "output", self._register_model_output)

    def _register_model_output(self, name: str, output: Any) -> None:
        """
        Registers a single model output tensor in the graph.

        :param name: The name of the output tensor.
        :param output: The output tensor to register.
        """
        tensor_info = self.tensor_info.get(output, None)
        if tensor_info is not None:
            node_id = self._get_new_node_id()
            self.graph.add_node(node_id, type=NodeType.output, meta=InOutMeta.from_tensor(output, name))
            self.graph.add_edge(
                tensor_info.source_node_id,
                node_id,
                meta=EdgeMeta.from_tensor(output, input_port=0, output_port=tensor_info.output_port_id),
            )
            logger.debug(f"GraphBuilderMode._register_model_output: {node_id=} {name=}")
        else:
            logger.debug(f"GraphBuilderMode._register_model_output: {name=} Not tensor info for {output=}")


def _call_fn_for_each_tensors(obj: Any, cur_name: str, fn: Callable[[str, torch.Tensor], None]) -> None:
    """
    Calls a function for each tensor in a nested structure.

    :param obj: The nested structure containing tensors.
    :param cur_name: The current name prefix for tensors.
    :param fn: The function to call for each tensor.
    """
    if isinstance(obj, torch.Tensor):
        fn(cur_name, obj)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            _call_fn_for_each_tensors(value, f"{cur_name}_{key}", fn)
    elif isinstance(obj, (list, tuple, set)):
        for idx, value in enumerate(obj):
            _call_fn_for_each_tensors(value, f"{cur_name}_{idx}", fn)


def build_graph(model: nn.Module, *args: Any, **kwargs: Any) -> nx.DiGraph:
    """
    Constructs a computational graph of the given model.

    This function builds a directed graph `nx.DiGraph` representing the operations
    and data flow within the model by leveraging hooks by using GraphBuilderMode.

    :param model: The PyTorch model for which the computational graph will be built.
    :return: A nx.DiGraph where nodes represent operations of model.
    """
    if not is_wrapped(model):
        raise nncf.InstallationError("Model is not wrapped")

    with torch.enable_grad():
        # Gradient use to get information about __get__ functions to detect tensor.(T, mT, H, mH) attributes
        with GraphBuilderMode(model=model, hook_storage=get_hook_storage(model)) as ctx:
            ctx.register_model_inputs(*args, **kwargs)
            outputs = model.forward.orig_forward(*args, **kwargs)
            ctx.register_model_outputs(outputs)
    return ctx.graph
