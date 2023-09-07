# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from typing import Callable, List, Optional, Tuple, TypeVar, Union

from nncf.experimental.tensor import Tensor
from nncf.experimental.tensor import unwrap_tensor_data
from nncf.experimental.tensor.enums import TensorDataType
from nncf.experimental.tensor.enums import TensorDeviceType

TTensor = TypeVar("TTensor")


def _tensor_guard(func: callable):
    """
    A decorator that ensures that the first argument to the decorated function is a Tensor.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if isinstance(args[0], Tensor):
            return func(*args, **kwargs)
        raise NotImplementedError(f"Function `{func.__name__}` is not implemented for {type(args[0])}")

    return wrapper


@functools.singledispatch
@_tensor_guard
def device(a: TTensor) -> TensorDeviceType:
    """
    Return the device of the tensor.

    :param a: The input tensor.
    :return: The device of the tensor.
    """
    return device(a.data)


@functools.singledispatch
@_tensor_guard
def squeeze(a: TTensor, axis: Optional[Union[int, Tuple[int]]] = None) -> TTensor:
    """
    Remove axes of length one from a.

    :param a: The input tensor.
    :param axis: Selects a subset of the entries of length one in the shape.
    :return: The input array, but with all or a subset of the dimensions of length 1 removed.
      This is always a itself or a view into a. Note that if all axes are squeezed,
      the result is a 0d array and not a scalar.
    """
    return Tensor(squeeze(a.data, axis=axis))


@functools.singledispatch
@_tensor_guard
def flatten(a: TTensor) -> TTensor:
    """
    Return a copy of the tensor collapsed into one dimension.

    :param a: The input tensor.
    :return: A copy of the input tensor, flattened to one dimension.
    """
    return Tensor(flatten(a.data))


@functools.singledispatch
@_tensor_guard
def max(a: TTensor, axis: Optional[Union[int, Tuple[int]]] = None) -> TTensor:  # pylint: disable=redefined-builtin
    """
    Return the maximum of an array or maximum along an axis.

    :param a: The input tensor.
    :param axis: Axis or axes along which to operate. By default, flattened input is used.
    :return: Maximum of a.
    """
    return Tensor(max(a.data, axis))


@functools.singledispatch
@_tensor_guard
def min(a: TTensor, axis: Optional[Union[int, Tuple[int]]] = None) -> TTensor:  # pylint: disable=redefined-builtin
    """
    Return the minimum of an array or minimum along an axis.

    :param a: The input tensor.
    :param axis: Axis or axes along which to operate. By default, flattened input is used.
    :return: Minimum of a.
    """
    return Tensor(min(a.data, axis))


@functools.singledispatch
@_tensor_guard
def abs(a: TTensor) -> Tensor:  # pylint: disable=redefined-builtin
    """
    Calculate the absolute value element-wise.

    :param a: The input tensor.
    :return: A tensor containing the absolute value of each element in x.
    """
    return Tensor(abs(a.data))


@functools.singledispatch
@_tensor_guard
def astype(a: TTensor, data_type: TensorDataType) -> TTensor:
    """
    Copy of the tensor, cast to a specified type.

    :param a: The input tensor.
    :param dtype: Type code or data type to which the tensor is cast.

    :return: Copy of the tensor in specified type.
    """
    return Tensor(astype(a.data, data_type))


@functools.singledispatch
@_tensor_guard
def dtype(a: TTensor) -> TensorDataType:
    """
    Return data type of the tensor.

    :param a: The input tensor.
    :return: The data type of the tensor.
    """
    return dtype(a.data)


@functools.singledispatch
@_tensor_guard
def reshape(a: TTensor, shape: List[int]) -> TTensor:
    """
    Gives a new shape to a tensor without changing its data.

    :param a: Tensor to be reshaped.
    :param shape: The new shape should be compatible with the original shape.
    :return: Reshaped tensor.
    """
    return Tensor(reshape(a.data, shape))


@functools.singledispatch
@_tensor_guard
def all(a: TTensor, axis: Optional[Union[int, Tuple[int]]] = None) -> TTensor:  # pylint: disable=redefined-builtin
    """
    Test whether all tensor elements along a given axis evaluate to True.

    :param a: The input tensor.
    :param axis: Axis or axes along which a logical AND reduction is performed.
    :return: A new boolean or tensor.
    """
    return Tensor(all(a.data, axis=axis))


@functools.singledispatch
@_tensor_guard
def allclose(a: TTensor, b: TTensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) -> bool:
    """
    Returns True if two arrays are element-wise equal within a tolerance.

    :param a: The first input tensor.
    :param b: The second input tensor.
    :param rtol: The relative tolerance parameter, defaults to 1e-05.
    :param atol: The absolute tolerance parameter, defaults to 1e-08.
    :param equal_nan: Whether to compare NaN`s as equal. If True,
      NaN`s in a will be considered equal to NaN`s in b in the output array.
      Defaults to False.
    :return: True if the two arrays are equal within the given tolerance, otherwise False.
    """
    return allclose(
        a.data,
        unwrap_tensor_data(b),
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    )


@functools.singledispatch
@_tensor_guard
def any(a: TTensor, axis: Optional[Union[int, Tuple[int]]] = None) -> TTensor:  # pylint: disable=redefined-builtin
    """
    Test whether any tensor elements along a given axis evaluate to True.

    :param a: The input tensor.
    :param axis: Axis or axes along which a logical OR reduction is performed.
    :return: A new boolean or tensor.
    """
    return Tensor(any(a.data, axis))


@functools.singledispatch
@_tensor_guard
def count_nonzero(a: TTensor, axis: Optional[Union[int, Tuple[int]]] = None) -> TTensor:
    """
    Counts the number of non-zero values in the tensor input.

    :param a: The tensor for which to count non-zeros.
    :param axis: Axis or tuple of axes along which to count non-zeros.
    :return: Number of non-zero values in the tensor along a given axis.
      Otherwise, the total number of non-zero values in the tensor is returned.
    """
    return Tensor(count_nonzero(a.data, axis))


@functools.singledispatch
@_tensor_guard
def isempty(a: TTensor) -> bool:
    """
    Return True if input tensor is empty.

    :param a: The input tensor.
    :return: True if tensor is empty, otherwise False.
    """
    return isempty(a.data)


@functools.singledispatch
@_tensor_guard
def isclose(a: TTensor, b: TTensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) -> TTensor:
    """
    Returns a boolean array where two arrays are element-wise equal within a tolerance.

    :param a: The first input tensor.
    :param b: The second input tensor.
    :param rtol: The relative tolerance parameter, defaults to 1e-05.
    :param atol: The absolute tolerance parameter, defaults to 1e-08.
    :param equal_nan: Whether to compare NaN`s as equal. If True,
      NaN`s in a will be considered equal to NaN`s in b in the output array.
      Defaults to False.
    :return: Returns a boolean tensor of where a and b are equal within the given tolerance.
    """
    return Tensor(
        isclose(
            a.data,
            unwrap_tensor_data(b),
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )
    )


@functools.singledispatch
@_tensor_guard
def maximum(x1: TTensor, x2: TTensor) -> TTensor:
    """
    Element-wise maximum of tensor elements.

    :param x1: The first input tensor.
    :param x2: The second input tensor.
    :return: Output tensor.
    """
    return Tensor(maximum(x1.data, unwrap_tensor_data(x2)))


@functools.singledispatch
@_tensor_guard
def minimum(x1: TTensor, x2: TTensor) -> TTensor:
    """
    Element-wise minimum of tensor elements.

    :param x1: The first input tensor.
    :param x2: The second input tensor.
    :return: Output tensor.
    """
    return Tensor(minimum(x1.data, unwrap_tensor_data(x2)))


@functools.singledispatch
@_tensor_guard
def ones_like(a: TTensor) -> TTensor:
    """
    Return a tensor of ones with the same shape and type as a given tensor.

    :param a: The shape and data-type of a define these same attributes of the returned tensor.
    :return: Tensor of ones with the same shape and type as a.
    """
    return Tensor(ones_like(a.data))


@functools.singledispatch
@_tensor_guard
def where(condition: TTensor, x: TTensor, y: TTensor) -> TTensor:
    """
    Return elements chosen from x or y depending on condition.

    :param condition: Where True, yield x, otherwise yield y.
    :param x: Value at indices where condition is True.
    :param y: Value at indices where condition is False.
    :return: A tensor with elements from x where condition is True, and elements from y elsewhere.
    """
    return Tensor(
        where(
            condition.data,
            unwrap_tensor_data(x),
            unwrap_tensor_data(y),
        )
    )


@functools.singledispatch
@_tensor_guard
def zeros_like(a: TTensor) -> TTensor:
    """
    Return an tensor of zeros with the same shape and type as a given tensor.

    :param input: The shape and data-type of a define these same attributes of the returned tensor.
    :return: tensor of zeros with the same shape and type as a.
    """
    return Tensor(zeros_like(a.data))


@functools.singledispatch
def stack(x: List[TTensor], axis: int = 0) -> TTensor:
    """
    Stacks a list or deque of Tensors rank-R tensors into one Tensor rank-(R+1) tensor.

    :param x: List or deque of Tensors.
    :param axis: The axis to stack along.
    :return: Stacked Tensor.
    """
    if isinstance(x, List):
        unwrapped_x = [i.data for i in x]
        # singledispatch cannot dispatch function by element in a list
        res = stack.dispatch(type(unwrapped_x[0]))(unwrapped_x, axis=axis)
        return Tensor(res)
    raise NotImplementedError(f"Function `stack` is not implemented for {type(x)}")


@functools.singledispatch
@_tensor_guard
def unstack(a: TTensor, axis: int = 0) -> List[Tensor]:
    """
    Unstack a Tensor into list.

    :param a: Tensor to unstack.
    :param axis: The axis to unstack along.
    :return: List of Tensor.
    """
    res = unstack(a.data, axis=axis)
    return [Tensor(i) for i in res]


@functools.singledispatch
@_tensor_guard
def moveaxis(a: TTensor, source: Union[int, List[int]], destination: Union[int, List[int]]) -> TTensor:
    """
    Move axes of an array to new positions.

    :param a: The array whose axes should be reordered.
    :param source: Original positions of the axes to move. These must be unique.
    :param destination: Destination positions for each of the original axes. These must also be unique.
    :return: Array with moved axes.
    """
    return Tensor(moveaxis(a.data, source, destination))


@functools.singledispatch
@_tensor_guard
def mean(a: TTensor, axis: Union[int, List[int]] = None, keepdims: bool = False) -> TTensor:
    """
    Compute the arithmetic mean along the specified axis.

    :param a: Array containing numbers whose mean is desired.
    :param axis: Axis or axes along which the means are computed.
    :param keepdims: Destination positions for each of the original axes. These must also be unique.
    :return: Array with moved axes.
    """
    return Tensor(mean(a.data, axis, keepdims))


@functools.singledispatch
@_tensor_guard
def round(a: TTensor, decimals=0) -> TTensor:  # pylint: disable=redefined-builtin
    """
    Evenly round to the given number of decimals.

    :param a: Input data.
    :param decimals: Number of decimal places to round to (default: 0). If decimals is negative,
      it specifies the number of positions to the left of the decimal point.
    :return: An array of the same type as a, containing the rounded values.
    """
    return Tensor(round(a.data, decimals))


@functools.singledispatch
@_tensor_guard
def binary_operator(a: TTensor, b: TTensor, operator_fn: Callable) -> TTensor:
    """
    Applies a binary operation to two tensors with disable warnings.

    :param a: The first tensor.
    :param b: The second tensor.
    :param operator_fn: The binary operation function.
    :return: The result of the binary operation.
    """
    return Tensor(binary_operator(a.data, unwrap_tensor_data(b), operator_fn))


@functools.singledispatch
@_tensor_guard
def binary_reverse_operator(a: TTensor, b: TTensor, operator_fn: Callable) -> TTensor:
    """
    Applies a binary reverse operation to two tensors with disable warnings.

    :param a: The first tensor.
    :param b: The second tensor.
    :param operator_fn: The binary operation function.
    :return: The result of the binary operation.
    """
    return Tensor(binary_reverse_operator(a.data, unwrap_tensor_data(b), operator_fn))


def _initialize_backends():
    # pylint: disable=unused-import
    import nncf.experimental.tensor.numpy_functions

    try:
        import nncf.experimental.tensor.torch_functions
    except ImportError:
        pass


_initialize_backends()
