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

from typing import List, Optional, Type

import nncf
from nncf.api.compression import TModel
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.parameters import SparsityMode
from nncf.scopes import IgnoredScope
from nncf.scopes import get_ignored_node_names_from_ignored_scope
from nncf.sparsity.backend import SparsityBackend


class Sparsity:
    def __init__(self, mode: SparsityMode, ratio: float, ignored_scope: Optional[IgnoredScope] = None) -> None:
        self.mode = mode
        self.ignored_scope = ignored_scope
        self.init_ratio = ratio
        self.backend_entity: Optional[Type[BackendType]] = None
        if ratio <= 0 or ratio >= 1:
            msg = "Sparsity ratio must be in the range (0, 1)."
            raise nncf.ParameterNotSupportedError(msg)

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
    ) -> TModel:
        self._set_backend_entity(model)

        nodes_to_sparsity = self.get_list_of_nodes()
        for node in nodes_to_sparsity:
            weight = self.backend.get_weight_node(model, node)

        return model

    def get_list_of_nodes(self, graph: NNCFGraph) -> list[NNCFNode]:
        ignored_names = get_ignored_node_names_from_ignored_scope(
            self.ignored_scope, graph, strict=self.ignored_scope.validate
        )

        ret: List[NNCFNode] = []

        for node in graph.get_all_nodes():
            if not self.backend.is_node_with_weights(graph, node):
                continue
            if node.node_name in ignored_names:
                continue
            ret.append(node)

        return ret

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm.

        :param model: Backend-specific input model.
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.TORCH:
            from nncf.sparsity.torch_backend import PTSparsityBackend

            self.backend_entity = PTSparsityBackend
        else:
            msg = f"Cannot return backend-specific entity because {model_backend.value} is not supported!"
            raise nncf.UnsupportedBackendError(msg)

    @property
    def backend(self) -> Type[SparsityBackend]:
        if self.backend_entity is None:
            msg = "The backend-specific entity is not initialized."
            raise nncf.InternalError(msg)
        return self.backend_entity
