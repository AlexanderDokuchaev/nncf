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

import inspect
import types
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from types import MethodType
from types import TracebackType
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from weakref import ReferenceType
from weakref import ref

from torch import nn
from torch.overrides import TorchFunctionMode

from nncf.common.logging import nncf_logger as logger
from nncf.experimental.torch.hook_executor_mode.hook_storage import HookStorage
from nncf.experimental.torch.hook_executor_mode.hook_storage import HookType
from nncf.experimental.torch.hook_executor_mode.operation_name import OperationName

IGNORED_FN_NAMES = [
    "__repr__",
    "_assert",
    "dim",
    "size",
    "is_floating_point",
]


@dataclass
class OpMeta:
    """
    Metadata for an operation to be executed, including its name, the callable function,
    and any additional information.


    :param op_name: The name of the operation.
    :param func: The function to be executed for the operation.
    :param extra_info: A dictionary for storing any additional information about the operation.
    """

    op_name: str
    func: Callable[..., Any]
    extra_info: Dict[str, Any] = field(default_factory=lambda: dict())


def _get_full_fn_name(fn: Callable[..., Any]) -> str:
    """
    Get the full name of a function, including its module if applicable.

    :param fn: The function for which to get the full name.
    :returns: The full name of the function.
    """
    if inspect.ismethoddescriptor(fn) or inspect.ismethod(fn):
        return fn.__qualname__
    if inspect.isbuiltin(fn) or inspect.isfunction(fn):
        return f"{fn.__module__}.{fn.__name__}"
    return f"{fn.__name__}"


ORIG_CALL_IMPL_STORAGE_NAME = "_origin_call_impl"


class HookExecutorMode(TorchFunctionMode):
    """
    Executes pre- and post-hooks for PyTorch functions within a model's execution.

    This mode wraps the function calls in the model to allow custom hooks to be executed before
    and after the actual function calls.


    :param model: The PyTorch model to which the hooks will be applied.
    :param hook_storage: Storage for hooks to be executed.
    :param module_call_stack: A stack tracking the modules being called.
    :param module_name_map: A map from module references to their names.
    :param nested_enter_count: A counter to track nested context manager entries.
    :param op_calls: A dictionary to track operation calls.
    """

    def __init__(self, model: nn.Module, hook_storage: HookStorage) -> None:
        """
        Initialize the HookExecutorMode.

        :param model: The PyTorch model to which the hooks will be applied.
        :param hook_storage: Storage for hooks to be executed.
        """
        super().__init__()
        self.hook_storage = hook_storage
        self.model = model
        self.module_call_stack: List[nn.Module] = []
        self.module_name_map: Dict[ReferenceType[nn.Module], str] = {ref(m): n for n, m in self.model.named_modules()}
        self.module_name_map[ref(model)] = ""
        self.nested_enter_count = 0
        self.op_calls: Dict[str, Dict[str, int]] = dict()

    def _get_wrapped_call(self, fn_call: MethodType) -> Callable[..., Any]:
        """
        Wrap a function call to include pushing to and popping from the module call stack.

        :param fn_call: The original function call to wrap.
        :returns: The wrapped function call.
        """

        def wrapped_call(self_: nn.Module, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Any:
            self.push_module_call_stack(self_)
            retval = fn_call.__func__(self_, *args, **kwargs)
            self.pop_module_call_stack()
            return retval

        setattr(wrapped_call, ORIG_CALL_IMPL_STORAGE_NAME, fn_call)
        return wrapped_call

    def __enter__(self) -> HookExecutorMode:
        """
        Enter the context manager.
        Wrapping the _call_impl function of each module on first nested enter.

        :returns: The instance of HookExecutorMode.
        """
        super().__enter__()
        if self.nested_enter_count == 0:
            # Wrap _call_impl function of instance each module.
            # Note: __call__ can`t not be overrided for instance, the function can be override only in class namespace.
            # TODO: register_forward_pre_hook and register_forward_hook does not works with DataParallel
            #       KeyError: return self.module_name_map[ref(self.module_call_stack[-1])]
            logger.debug("HookExecutorMode.__enter__: wrap _call_impl function")
            for _, module in self.model.named_modules():
                module._call_impl = types.MethodType(self._get_wrapped_call(module._call_impl), module)
            self.push_module_call_stack(self.model)
        self.nested_enter_count += 1
        return self

    def __exit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        """
        Exit the context manager, unwrapping the _call_impl function of each module.

        :param exc_type (Optional[Type[BaseException]]): Exception type.
        :param exc_val (Optional[BaseException]): Exception value.
        :param exc_tb (Optional[TracebackType]): Traceback.
        """
        self.nested_enter_count -= 1
        if self.nested_enter_count == 0:
            # Unwrap _call_impl functions
            logger.debug("HookExecutorMode.__exit__: unwrap _call_impl function")
            for _, module in self.model.named_modules():
                module._call_impl = getattr(module._call_impl, ORIG_CALL_IMPL_STORAGE_NAME)
            self.pop_module_call_stack()
        super().__exit__(exc_type, exc_val, exc_tb)

    def __torch_function__(
        self,
        func: Callable[..., Any],
        types: List[Type[Any]],
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Override the __torch_function__ method to add pre- and post-hook execution.

        :param func: The function being called.
        :param types: List of types.
        :param args: The arguments to the function.
        :param kwargs: The keyword arguments to the function.

        :returns: The output of the function call after hooks have been executed.
        """
        kwargs = kwargs or {}

        if not self.module_call_stack:
            # Ignore all function before call source module
            return func(*args, **kwargs)

        fn_name = func.__name__
        if fn_name in IGNORED_FN_NAMES:
            return func(*args, **kwargs)
        op_name = self.get_current_executed_op_name(fn_name)

        full_fn_name = _get_full_fn_name(func)
        logger.debug(f"HookExecutorMode.__torch_function__: {full_fn_name=} {op_name=}")

        self.register_op(fn_name)
        op_meta = OpMeta(op_name=op_name, func=func)
        args, kwargs = self.execute_pre_hooks(args, kwargs, op_meta)
        output = func(*args, **kwargs)
        output = self.execute_post_hooks(output, op_meta)
        return output

    def push_module_call_stack(self, module: nn.Module) -> None:
        """
        Push a module to the call stack and initialize its operation calls.

        :param module: The module to push onto the call stack.
        """
        self.module_call_stack.append(module)
        module_name = self.get_current_module_name()
        self.op_calls[module_name] = defaultdict(int)
        logger.debug(f"HookExecutorMode.push_module_call_stack: {module_name=}")

    def pop_module_call_stack(self) -> None:
        """
        Pop a module from the call stack and remove its operation calls.
        """
        module_name = self.get_current_module_name()
        del self.op_calls[module_name]
        self.module_call_stack.pop()
        logger.debug(f"HookExecutorMode.pop_module_call_stack: {module_name=}")

    def get_current_module_name(self) -> str:
        """
        Get the name of the current module being executed.

        :returns: The name of the current module.
        """
        return self.module_name_map[ref(self.module_call_stack[-1])]

    def get_current_executed_op_name(self, fn_name: str) -> str:
        """
        Get the name of the current operation being executed.

        :param fn_name: The function name of the operation.
        :returns: The name of the operation.
        """
        module_name = self.get_current_module_name()
        op_name = OperationName(module_name, fn_name)
        op_name.call_id = self.op_calls[module_name][str(op_name)]
        return str(op_name)

    def register_op(self, fn_name: str) -> None:
        """
        Register an operation call for the current module and increment call counter.

        :param fn_name: The function name of the operation.
        """
        module_name = self.get_current_module_name()
        op_name = str(OperationName(module_name, fn_name))
        self.op_calls[module_name][op_name] += 1

    def execute_pre_hooks(
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any], op_meta: OpMeta
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """
        Execute pre-hooks for the operation.

        :param args: The arguments to the function.
        :param kwargs: The keyword arguments to the function.
        :param op_meta: Metadata for the operation.
        :returns: The modified arguments and keyword arguments after pre-hooks.
        """
        _args = list(args)

        with self:
            for idx, value in enumerate(_args):
                _args[idx] = self.hook_storage.execute_hook(HookType.PRE_HOOK, op_meta.op_name, idx, value)

            for port_id, kw_name in enumerate(kwargs, start=len(_args)):
                kwargs[kw_name] = self.hook_storage.execute_hook(
                    HookType.PRE_HOOK, op_meta.op_name, port_id, kwargs[kw_name]
                )
        return tuple(_args), kwargs

    def execute_post_hooks(self, output: Any, op_meta: OpMeta) -> Any:
        """
        Execute post-hooks for the operation.

        :param output: The output of the function.
        :param op_meta: Metadata for the operation.
        :returns: The modified output after post-hooks.
        """
        with self:
            is_tuple = False
            if isinstance(output, tuple):
                # Need to return named tuples like torch.return_types.max
                cls_tuple = type(output)
                output = list(output)
                is_tuple = True

            if isinstance(output, list):
                for idx, value in enumerate(output):
                    output[idx] = self.hook_storage.execute_hook(HookType.POST_HOOK, op_meta.op_name, idx, value)
                if is_tuple:
                    output = cls_tuple(output)
            else:
                output = self.hook_storage.execute_hook(HookType.POST_HOOK, op_meta.op_name, 0, output)
        return output
