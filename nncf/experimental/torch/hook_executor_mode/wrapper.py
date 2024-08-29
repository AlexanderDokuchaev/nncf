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

import nncf
from nncf.experimental.torch.hook_executor_mode.hook_executor_mode import HookExecutorMode
from nncf.experimental.torch.hook_executor_mode.hook_storage import HookStorage
from nncf.experimental.torch.hook_executor_mode.hook_storage import HookType

ATR_HOOK_STORAGE = "__hooks"


class ForwardWithHooks:
    """Class to wrap forward function of nn.Module, to forward function of the model with enabled HookExecutorMode"""

    __slots__ = "orig_forward", "__dict__", "__weakref__"

    def __new__(cls, orig_forward: Callable):
        if not callable(orig_forward):
            raise TypeError("the first argument must be callable")

        if isinstance(orig_forward, ForwardWithHooks):
            raise TypeError("Func already wrapped")

        self = super(ForwardWithHooks, cls).__new__(cls)

        self.orig_forward = orig_forward
        return self

    def __call__(self, *args: Tuple[Any], **keywords: Dict[str, Any]):
        model = self.orig_forward.__self__
        with HookExecutorMode(model=model, hook_storage=get_hook_storage(model)):
            return self.orig_forward(*args, **keywords)

    def __repr__(self):
        return f"ForwardWithHooks.{repr(self.orig_forward)}"

    def __reduce__(self):
        return type(self), (self.orig_forward,), (self.orig_forward, self.__dict__ or None)

    def __setstate__(self, state):
        if not isinstance(state, tuple):
            raise TypeError("argument to __setstate__ must be a tuple")
        if len(state) != 2:
            raise TypeError(f"expected 2 items in state, got {len(state)}")
        func, namespace = state
        if not callable(func) or (namespace is not None and not isinstance(namespace, dict)):
            raise TypeError("invalid partial state")

        if namespace is None:
            namespace = {}

        self.__dict__ = namespace
        self.orig_forward = func

    @property
    def __code__(self):
        return self.orig_forward.__code__


class ReplicateForDataParallel:
    """
    Class to wrap _replicate_for_data_parallel function of nn.Module,
    to correctly wrap forward with enabled HookExecutorMode.
    """

    __slots__ = "func", "__dict__", "__weakref__"

    def __new__(cls, func: Callable):
        if not callable(func):
            raise TypeError("the first argument must be callable")

        if isinstance(func, ReplicateForDataParallel):
            raise TypeError("Func already wrapped")

        self = super(ReplicateForDataParallel, cls).__new__(cls)

        self.func = func
        return self

    def __call__(self, *args: Tuple[Any], **keywords: Dict[str, Any]):
        module: nn.Module = self.func.__self__
        saved_wrapped_forward = module.forward
        module.__dict__.pop("forward")

        replica: nn.Module = self.func(*args, **keywords)

        replica.forward = ForwardWithHooks(replica.forward)
        module.forward = saved_wrapped_forward

        return replica

    def __repr__(self):
        return f"ReplicateForDataParallel.{repr(self.func)}"

    def __reduce__(self):
        return type(self), (self.func,), (self.func, self.__dict__ or None)

    def __setstate__(self, state):
        if not isinstance(state, tuple):
            raise TypeError("argument to __setstate__ must be a tuple")
        if len(state) != 2:
            raise TypeError(f"expected 2 items in state, got {len(state)}")
        func, namespace = state
        if not callable(func) or (namespace is not None and not isinstance(namespace, dict)):
            raise TypeError("invalid partial state")

        if namespace is None:
            namespace = {}

        self.__dict__ = namespace
        self.func = func


def wrap_model(model: nn.Module) -> nn.Module:
    """
    Wraps a nn.Module to inject custom behavior into the forward pass and replication process.

    This function modifies the given model by:
    1. Replacing the model's `forward` method with a wrapped version (`ForwardWithHooks`) that allows
       additional hooks to be executed during the forward pass by using HookExecutorMode.
    2. Wrapping the model's `_replicate_for_data_parallel` method with `ReplicateForDataParallel`,
       which allows custom behavior when the model is replicated across multiple devices (e.g., for
       data parallelism).
    3. Adding a new module, `HookStorage`, to the model under the attribute `ATR_HOOK_STORAGE`.

    :param model: The nn.Module to be wrapped.
    :return: The modified model with the custom behavior injected.
    """

    if "forward" in model.__dict__:
        raise nncf.InternalError("Wrapper does not supported models with overrided forward function")
    model.forward = ForwardWithHooks(model.forward)
    model._replicate_for_data_parallel = ReplicateForDataParallel(model._replicate_for_data_parallel)
    model.add_module(ATR_HOOK_STORAGE, HookStorage())
    return model


def is_wrapped(model: nn.Module) -> bool:
    """
    Checks if a given model has been wrapped by the `wrap_model` function.

    :param model: The nn.Module to check.
    :return: `True` if the model's `forward` method is an instance of `ForwardWithHooks`,
        indicating that the model has been wrapped; `False` otherwise.
    """
    return isinstance(model.forward, ForwardWithHooks)


def get_hook_storage(model: nn.Module) -> HookStorage:
    """
    Retrieves the `HookStorage` module from the given model.

    This function accesses the model's attribute defined by `ATR_HOOK_STORAGE`
    and returns the `HookStorage` module associated with it.


    :param model: The PyTorch model from which to retrieve the `HookStorage`.
    :return: The `HookStorage` module associated with the model.
    """
    if not is_wrapped(model):
        raise nncf.InstallationError("Model is not wrapped")
    return getattr(model, ATR_HOOK_STORAGE)


def insert_hook(
    model: nn.Module, hook_type: HookType, group_name: str, op_name: str, port_id: int, hook_module: nn.Module
):
    """
    Inserts a hook into the model's `HookStorage` for a specified operation.
    """
    storage = get_hook_storage(model)
    if storage is None:
        raise nncf.InternalError("Insertion hook to not wrapped model")
    storage.insert_hook(hook_type, group_name, op_name, port_id, hook_module)


def remove_group(model: nn.Module, group_name: str):
    """
    Removes all hooks associated with a specific group from the model's `HookStorage`.
    """
    storage = get_hook_storage(model)
    if storage is None:
        raise nncf.InternalError("Insertion hook to not wrapped model")
    storage.remove_group(group_name)
