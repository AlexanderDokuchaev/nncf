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

from typing import Any, Callable, Dict, Tuple

from torch import nn
import torch
import nncf
from nncf.experimental.torch.hook_executor_mode.build_graph_mode import GraphBuilderMode
from nncf.experimental.torch.hook_executor_mode.hook_executor_mode import HookExecutorMode
from nncf.experimental.torch.hook_executor_mode.hook_storage import HookStorage
from nncf.experimental.torch.hook_executor_mode.hook_storage import HookType
import networkx as nx

class ModuleWithHooks(nn.Module):
    """
    A wrapper module that integrates hook functionality into a model.

    This class is designed to wrap a given model and manage hooks using the HookExecutorMode class.

    Attributes:
        _model (nn.Module): The model being wrapped.
        _hooks (HookStorage): The storage for hooks associated with this model.
    """

    def __init__(self, model: nn.Module) -> None:
        """
        Initialize the GraphXModule.

        Args:
            model (nn.Module): The model to be wrapped.

        Raises:
            RuntimeError: If the model is already wrapped by GraphXModule.
        """
        super().__init__()
        if isinstance(model, ModuleWithHooks):
            raise RuntimeError("Model already wrapped")
        self._model = model

        self._hooks = HookStorage()

    @property
    def model(self) -> nn.Module:
        """
        Get the wrapped model.

        Returns:
            nn.Module: The wrapped model.
        """
        return self._model

    @property
    def hooks(self) -> HookStorage:
        """
        Get the hook storage associated with the model.

        Returns:
            HookStorage: The storage for hooks.
        """
        return self._hooks

    def forward(self, *args: Any, **kwargs: Dict[str, Any]) -> Any:
        """
        Forward pass of the wrapped model with hook execution.

        Args:
            *args (Any): Positional arguments to be passed to the model's forward method.
            **kwargs (Any): Keyword arguments to be passed to the model's forward method.

        Returns:
            Any: The output of the model's forward method.
        """
        kwargs = kwargs or {}
        with HookExecutorMode(model=self, hook_storage=self.hooks) as ctx:
            args, kwargs = ctx.execute_input_hooks(args, kwargs)
            outputs = self.model(*args, **kwargs)
            outputs = ctx.execute_output_hooks(outputs)
            return outputs

    def build_graph(self, *args: Any, **kwargs: Any) -> nx.DiGraph:
        with torch.enable_grad():
            # Gradient use to get information about __get__ functions to detect tensor.(T, mT, H, mH) attributes
            with GraphBuilderMode(model=self, hook_storage=self.hooks) as ctx:
                ctx.register_model_inputs(*args, **kwargs)
                args, kwargs = ctx.execute_input_hooks(args, kwargs)
                outputs = self.model(*args, **kwargs)
                outputs = ctx.execute_output_hooks(outputs)
                ctx.register_model_outputs(outputs)
        return ctx.graph



def is_wrapped(model: nn.Module) -> bool:
    """
    Checks if a given model has been wrapped by the `wrap_model` function.

    :param model: The nn.Module to check.
    :return: `True` if the model's `forward` method is an instance of `ForwardWithHooks`,
        indicating that the model has been wrapped; `False` otherwise.
    """
    return isinstance(model, ModuleWithHooks)
