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

from typing import Any, Optional

from nncf.api.compression import TModel
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.parameters import SparsityMode
from nncf.scopes import IgnoredScope
from nncf.sparsity.algorithm import Sparsity


def sparsity(
    model: TModel,
    mode: SparsityMode,
    ratio: float,
    ignored_scope: Optional[IgnoredScope] = None,
    examples_inputs: Optional[Any] = None,
) -> TModel:
    """
    Applies post-training quantization to the provided model.

    :param model: A model to be sparsify.
    :param mode: Special sparsity mode that specify different ways of the optimization.
    :param ratio: The target sparsity ratio (0, 1).
    :param ignored_scope: An ignored scope that defined the list of model control
        flow graph nodes to be ignored during quantization.
    :param example_input: An example of model input. Used to trace model operations (required only for PT).
    :return: The sparsified model.
    """
    backend = get_backend(model)
    if backend == BackendType.TORCH:
        pass
    from nncf.common.factory import NNCFGraphFactory

    algorithm = Sparsity(mode=mode, ratio=ratio, ignored_scope=ignored_scope)

    graph = NNCFGraphFactory.create(model, examples_inputs)

    model = algorithm.apply(model, graph)

    return model


# def sparsity_update_ratio(
#     model: TModel,
#     mode: SparsityMode,
#     ratio: float,
# ) -> TModel:
#     if ratio <= 0 or ratio >= 1:
#         msg = "Sparsity ratio must be in the range (0, 1)."
#         raise nncf.errors.ParameterNotSupportedError(msg)

#     backend = get_backend(model)
#     if backend == BackendType.TORCH:
#         from nncf.torch.function_hook.sparsity.sparsity_model import update_sparsity_ratio_impl

#         return update_sparsity_ratio_impl(  # type: ignore[no-any-return]
#             model=model,
#             mode=mode,
#             ratio=ratio,
#         )

#     msg = f"Unsupported type of backend: {backend}"
#     raise nncf.UnsupportedBackendError(msg)
