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

from typing import List, Optional, Tuple, Union

import numpy as np

from nncf.experimental.tensor import functions
from nncf.experimental.tensor.enums import TensorDataType
from nncf.experimental.tensor.enums import TensorDeviceType

DTYPE_MAP = {
    TensorDataType.float16: np.dtype(np.float16),
    TensorDataType.float32: np.dtype(np.float32),
    TensorDataType.float64: np.dtype(np.float64),
    TensorDataType.int8: np.dtype(np.int8),
    TensorDataType.uint8: np.dtype(np.uint8),
}

DTYPE_MAP_REV = {v: k for k, v in DTYPE_MAP.items()}


@functions.device.register(np.bool_)
@functions.device.register(np.ndarray)
@functions.device.register(np.number)
def _(a: Union[np.ndarray, np.number]) -> TensorDeviceType:
    return TensorDeviceType.CPU


@functions.squeeze.register(np.bool_)
@functions.squeeze.register(np.ndarray)
@functions.squeeze.register(np.number)
def _(a: Union[np.ndarray, np.number], axis: Optional[Union[int, Tuple[int]]] = None) -> np.ndarray:
    return np.squeeze(a, axis=axis)


@functions.flatten.register(np.bool_)
@functions.flatten.register(np.ndarray)
@functions.flatten.register(np.number)
def _(a: Union[np.ndarray, np.number]) -> np.ndarray:
    return a.flatten()


@functions.max.register(np.bool_)
@functions.max.register(np.ndarray)
@functions.max.register(np.number)
def _(a: Union[np.ndarray, np.number], axis: Optional[Union[int, Tuple[int]]] = None) -> np.ndarray:
    return np.max(a, axis=axis)


@functions.amax.register(np.bool_)
@functions.amax.register(np.ndarray)
@functions.amax.register(np.number)
def _(a: Union[np.ndarray, np.number], axis: Optional[Union[int, Tuple[int]]] = None) -> np.ndarray:
    return np.amax(a, axis=axis)


@functions.min.register(np.bool_)
@functions.min.register(np.ndarray)
@functions.min.register(np.number)
def _(a: Union[np.ndarray, np.number], axis: Optional[Union[int, Tuple[int]]] = None) -> np.ndarray:
    return np.min(a, axis=axis)


@functions.amin.register(np.bool_)
@functions.amin.register(np.ndarray)
@functions.amin.register(np.number)
def _(a: Union[np.ndarray, np.number], axis: Optional[Union[int, Tuple[int]]] = None) -> np.ndarray:
    return np.amin(a, axis=axis)


@functions.abs.register(np.bool_)
@functions.abs.register(np.ndarray)
@functions.abs.register(np.number)
def _(a: Union[np.ndarray, np.number]) -> np.ndarray:
    return np.absolute(a)


@functions.astype.register(np.bool_)
@functions.astype.register(np.ndarray)
@functions.astype.register(np.number)
def _(a: Union[np.ndarray, np.number], dtype: TensorDataType) -> np.ndarray:
    return a.astype(DTYPE_MAP[dtype])


@functions.dtype.register(np.bool_)
@functions.dtype.register(np.ndarray)
@functions.dtype.register(np.number)
def _(a: Union[np.ndarray, np.number]) -> TensorDataType:
    return DTYPE_MAP_REV[np.dtype(a.dtype)]


@functions.reshape.register(np.bool_)
@functions.reshape.register(np.ndarray)
@functions.reshape.register(np.number)
def _(a: Union[np.ndarray, np.number], shape: Union[int, Tuple[int]]) -> np.ndarray:
    return a.reshape(shape)


@functions.all.register(np.bool_)
@functions.all.register(np.ndarray)
@functions.all.register(np.number)
def _(a: Union[np.ndarray, np.number], axis: Optional[Union[int, Tuple[int]]] = None) -> Union[np.ndarray, bool]:
    return np.all(a, axis=axis)


@functions.allclose.register(np.bool_)
@functions.allclose.register(np.ndarray)
@functions.allclose.register(np.number)
def _(
    a: Union[np.ndarray, np.number],
    b: Union[np.ndarray, np.number],
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


@functions.any.register(np.bool_)
@functions.any.register(np.ndarray)
@functions.any.register(np.number)
def _(a: Union[np.ndarray, np.number], axis: Optional[Union[int, Tuple[int]]] = None) -> Union[np.ndarray, bool]:
    return np.any(a, axis=axis)


@functions.count_nonzero.register(np.bool_)
@functions.count_nonzero.register(np.ndarray)
@functions.count_nonzero.register(np.number)
def _(a: Union[np.ndarray, np.number], axis: Optional[Union[int, Tuple[int]]] = None) -> np.ndarray:
    return np.count_nonzero(a, axis=axis)


@functions.isempty.register(np.bool_)
@functions.isempty.register(np.ndarray)
@functions.isempty.register(np.number)
def _(a: Union[np.ndarray, np.number]) -> bool:
    return a.size == 0


@functions.isclose.register(np.bool_)
@functions.isclose.register(np.ndarray)
@functions.isclose.register(np.number)
def _(
    a: Union[np.ndarray, np.number],
    b: np.ndarray,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
):
    return np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


@functions.maximum.register(np.bool_)
@functions.maximum.register(np.ndarray)
@functions.maximum.register(np.number)
def _(x1: Union[np.ndarray, np.number], x2: np.ndarray) -> np.ndarray:
    return np.maximum(x1, x2)


@functions.minimum.register(np.bool_)
@functions.minimum.register(np.ndarray)
@functions.minimum.register(np.number)
def _(x1: Union[np.ndarray, np.number], x2: np.ndarray) -> np.ndarray:
    return np.minimum(x1, x2)


@functions.ones_like.register(np.bool_)
@functions.ones_like.register(np.ndarray)
@functions.ones_like.register(np.number)
def _(a: Union[np.ndarray, np.number]) -> np.ndarray:
    return np.ones_like(a)


@functions.where.register(np.bool_)
@functions.where.register(np.ndarray)
@functions.where.register(np.number)
def _(
    condition: Union[np.ndarray, np.number],
    x: Union[np.ndarray, np.number, float, bool],
    y: Union[np.ndarray, float, bool],
) -> np.ndarray:
    return np.where(condition, x, y)


@functions.zeros_like.register(np.bool_)
@functions.zeros_like.register(np.ndarray)
@functions.zeros_like.register(np.number)
def _(a: Union[np.ndarray, np.number]) -> np.ndarray:
    return np.zeros_like(a)


@functions.stack.register(np.bool_)
@functions.stack.register(np.ndarray)
@functions.stack.register(np.number)
def _(x: Union[np.ndarray, np.number], axis: int = 0) -> List[np.ndarray]:
    return np.stack(x, axis=axis)


@functions.unstack.register(np.bool_)
@functions.unstack.register(np.ndarray)
@functions.unstack.register(np.number)
def _(x: Union[np.ndarray, np.number], axis: int = 0) -> List[np.ndarray]:
    return [np.squeeze(e, axis) for e in np.split(x, x.shape[axis], axis=axis)]


@functions.moveaxis.register(np.ndarray)
def _(a: np.ndarray, source: Union[int, List[int]], destination: Union[int, List[int]]) -> np.ndarray:
    return np.moveaxis(a, source, destination)


@functions.mean.register(np.bool_)
@functions.mean.register(np.ndarray)
@functions.mean.register(np.number)
def _(a: Union[np.ndarray, np.number], axis: Union[int, List[int]] = None, keepdims: bool = False) -> np.ndarray:
    return np.mean(a, axis=axis, keepdims=keepdims)


@functions.round.register(np.bool_)
@functions.round.register(np.ndarray)
@functions.round.register(np.number)
def _(a: Union[np.ndarray, np.number], decimals: int = 0) -> np.ndarray:
    return np.round(a, decimals=decimals)
