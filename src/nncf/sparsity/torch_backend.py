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


import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.sparsity.backend import SparsityBackend
from nncf.torch.function_hook.nncf_graph.layer_attributes import PT2OpLayerAttributes
from nncf.torch.graph.operator_metatypes import CONVOLUTION_METATYPES
from nncf.torch.graph.operator_metatypes import OPERATORS_WITH_WEIGHTS_METATYPES

OPERATORS_WITH_WEIGHTS_METATYPES = CONVOLUTION_METATYPES


class PTSparsityBackend(SparsityBackend):
    @staticmethod
    def is_node_with_weights(graph: NNCFGraph, node: NNCFNode) -> bool:
        if node.metatype not in OPERATORS_WITH_WEIGHTS_METATYPES:
            return None

        layer_attributes = node.layer_attributes
        if not isinstance(layer_attributes, PT2OpLayerAttributes):
            raise nncf.InternalError("The node does not have PT2OpLayerAttributes.")

        return bool(layer_attributes.constant_port_ids)
