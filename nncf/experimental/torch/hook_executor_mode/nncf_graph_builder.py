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


from typing import Dict, Optional, Union
import nncf
import torch
import networkx as nx
from torch import nn

from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.layer_attributes import Dtype
from nncf.torch.dynamic_graph.layer_attributes_handlers import get_layer_attributes_from_args_and_kwargs
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.experimental.torch.hook_executor_mode.graph_utils import ConstMeta
from nncf.experimental.torch.hook_executor_mode.graph_utils import EdgeMeta
from nncf.experimental.torch.hook_executor_mode.graph_utils import FunctionMeta
from nncf.experimental.torch.hook_executor_mode.graph_utils import InOutMeta
from nncf.experimental.torch.hook_executor_mode.graph_utils import NodeType
from nncf.experimental.torch.hook_executor_mode.graph_utils import TensorDesc
from nncf.experimental.torch.hook_executor_mode.graph_utils import TensorInfo
from nncf.experimental.torch.hook_executor_mode.graph_utils import TensorMeta
# from nncf.experimental.torch.hook_executor_mode import ModuleWithHooks
import nncf.torch.graph.operator_metatypes as om
from nncf.experimental.torch.hook_executor_mode.wrapper import get_hook_storage
from nncf.experimental.torch.hook_executor_mode.build_graph_mode import build_graph

class GraphConverter:
    """
    Builds the NNCFGraph from an nx.DiGraph instance.
    """

    @staticmethod
    def convert_node_type(type: NodeType, meta: Union[ConstMeta, FunctionMeta, InOutMeta]):
        if type == NodeType.input:
            return "nncf_model_input"
        if type == NodeType.output:
            return "nncf_model_output"
        if type == NodeType.const:
            return "nncf_model_const"
        return meta.fn_name

    @staticmethod
    def convert_node_meta_type(node_type: str, meta: Union[ConstMeta, FunctionMeta, InOutMeta]) -> om.PTOperatorMetatype:
        node_metatype: om.PTOperatorMetatype = om.PT_OPERATOR_METATYPES.get_operator_metatype_by_op_name(node_type)
        node_sub_meta_type: Optional[om.PTSubMetatype] = None
        if node_metatype.get_subtypes() and isinstance(meta, FunctionMeta):
            # TODO: change match metatype to use args and kwargs
            layer_attr = get_layer_attributes_from_args_and_kwargs(node_type, meta.args, meta.kwargs)
            node_sub_meta_type = node_metatype.determine_subtype(layer_attr)

        return node_sub_meta_type or node_metatype

    @staticmethod
    def get_name_of_node(meta: Union[ConstMeta, FunctionMeta, InOutMeta]):
        if isinstance(meta, ConstMeta):
            return meta.name_in_model
        if isinstance(meta, FunctionMeta):
            return meta.op_name
        if isinstance(meta, InOutMeta):
            return meta.name

    @staticmethod
    def convert_dtype(dtype: torch.dtype) -> Dtype:
        if dtype in [torch.float, torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            return Dtype.FLOAT
        return Dtype.INTEGER

    @classmethod
    def convert_to_nncf_graph(cls, nx_graph: nx.DiGraph) -> PTNNCFGraph:
        nncf_graph = PTNNCFGraph()

        map_nx_node_to_nncf_node: Dict[int, NNCFNode] = {}
        for node, data in nx_graph.nodes(data=True):
            meta: Union[ConstMeta, FunctionMeta, InOutMeta] = data["meta"]
            node_type = cls.convert_node_type(data["type"], meta)
            node_metatype = cls.convert_node_meta_type(node_type, meta)
            nncf_node = nncf_graph.add_nncf_node(
                node_name=cls.get_name_of_node(meta),
                node_type=node_type,
                node_metatype=node_metatype,
            )
            map_nx_node_to_nncf_node[node] = nncf_node

        for s_node, t_node, data in nx_graph.edges(data=True):
            meta: Union[EdgeMeta] = data["meta"]
            source_node = map_nx_node_to_nncf_node[s_node]
            target_node = map_nx_node_to_nncf_node[t_node]
            nncf_graph.add_edge_between_nncf_nodes(
                source_node.node_id,
                target_node.node_id,
                tensor_shape=meta.shape,
                input_port_id=meta.input_port,
                output_port_id=0, #meta.output_port,
                dtype=cls.convert_dtype(meta.dtype),
            )
        return nncf_graph

    @classmethod
    def build_nncf_graph(cls, model: nn.Module) -> PTNNCFGraph:
        if isinstance(model.example_input, dict):
            nx_graph = build_graph(model,**model.example_input)
        elif isinstance(model.example_input, tuple):
            nx_graph = build_graph(model,*model.example_input)
        else:
            nx_graph = build_graph(model, model.example_input)
        # nx_graph = model.build_graph(model.example_input)
        return cls.convert_to_nncf_graph(nx_graph)
