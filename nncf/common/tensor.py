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
import abc
from abc import abstractmethod
from enum import IntEnum
from typing import Callable
from typing import Generic, List, Tuple, Type, TypeVar, Union, Any
from typing import Iterator
from typing import Optional

import numpy as np

TensorType = TypeVar("TensorType")
DeviceType = TypeVar("DeviceType")
TensorElementsType = TypeVar("TensorElementsType")


class TensorDtype(IntEnum):
    FLOAT32 = 0
    INT64 = 1


class NNCFTensor(Generic[TensorType], abc.ABC):
    """
    An interface of framework specific tensors for common NNCF algorithms.
    """

    @property
    @abstractmethod
    def backend(self) -> Type["NNCFTensorBackend"]:
        pass

    def __init__(self, tensor: TensorType):
        assert not isinstance(tensor, NNCFTensor)
        self._tensor: TensorType = tensor

    def __eq__(self, other: Any) -> "NNCFTensor":
        # Assuming every backend implements this basic semantic
        if isinstance(other, NNCFTensor):
            return self.__class__(self._tensor == other.tensor)
        return self._tensor > other

    def __lt__(self, other: Any) -> "NNCFTensor":
        if isinstance(other, NNCFTensor):
            return self.__class__(self._tensor < other.tensor)
        return self._tensor < other

    def __gt__(self, other: Any) -> "NNCFTensor":
        if isinstance(other, NNCFTensor):
            return self.__class__(self._tensor > other.tensor)
        return self._tensor > other

    def __pow__(self, other) -> "NNCFTensor":
        return self.__class__(self._tensor ** other)

    def __invert__(self) -> "NNCFTensor":
        return self.__class__(~self._tensor)

    def __add__(self, other: Any) -> 'NNCFTensor':
        return self.__class__(self._tensor + other.tensor)

    def __sub__(self, other: Any) -> 'NNCFTensor':
        return self.__class__(self._tensor - other.tensor)

    def __mul__(self, other: Any) -> 'NNCFTensor':
        return self.__class__(self._tensor * other.tensor)

    def __truediv__(self, other: Any) -> 'NNCFTensor':
        return self.__class__(self._tensor / other)

    def __len__(self) -> int:
        return len(self._tensor)

    def __getitem__(self, item) -> 'NNCFTensor':
        return self.__class__(self._tensor[item])

    def __setitem__(self, key, value):
        self._tensor[key] = value

    @property
    def tensor(self):
        return self._tensor

    @property
    @abstractmethod
    def ndim(self) -> int:
        pass

    @property
    @abstractmethod
    def shape(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def device(self) -> DeviceType:
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        pass

    @abstractmethod
    def mean(self, axis: int, keepdims: bool = None) -> "NNCFTensor":
        pass

    @abstractmethod
    def median(self, axis: int, keepdims: bool = False) -> "NNCFTensor":
        pass


    @abstractmethod
    def reshape(self, *shape: int) -> "NNCFTensor":
        pass

    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator:
        pass

    @abstractmethod
    def dot(self, other: "NNCFTensor") -> "NNCFTensor":
        pass

    @abstractmethod
    def astype(self, dtype: TensorDtype) -> "NNCFTensor":
        pass

    @property
    @abstractmethod
    def dtype(self) -> TensorDtype:
        pass

    @abstractmethod
    def any(self) -> bool:
        pass

    def min(self) -> float:
        pass

    def max(self) -> float:
        pass



class NNCFTensorBackend(abc.ABC):
    inf = None  # TODO(vshampor): IMPLEMENT ME

    @staticmethod
    @abstractmethod
    def moveaxis(x: NNCFTensor, src: int, dst: int) -> NNCFTensor:
        pass

    @staticmethod
    @abstractmethod
    def mean(x: NNCFTensor, axis: Union[int, Tuple[int, ...]]) -> NNCFTensor:
        pass

    @staticmethod
    @abstractmethod
    def mean_of_list(tensor_list: List[NNCFTensor], axis: int) -> NNCFTensor:
        pass

    @staticmethod
    @abstractmethod
    def isclose_all(tensor1: NNCFTensor, tensor2: NNCFTensor, rtol=1e-05, atol=1e-08) -> bool:
        pass

    @staticmethod
    @abstractmethod
    def stack(tensor_list: List[NNCFTensor]) -> NNCFTensor:
        pass

    @staticmethod
    @abstractmethod
    def count_nonzero(mask: NNCFTensor) -> NNCFTensor:
        pass

    @staticmethod
    @abstractmethod
    def abs(tensor: NNCFTensor) -> NNCFTensor:
        pass

    @staticmethod
    @abstractmethod
    def min(tensor: NNCFTensor, axis: int = None) -> NNCFTensor:
        pass

    @staticmethod
    @abstractmethod
    def max(tensor: NNCFTensor, axis: int = None) -> NNCFTensor:
        pass

    @staticmethod
    @abstractmethod
    def expand_dims(tensor: NNCFTensor, axes: List[int]) -> NNCFTensor:
        pass

    @staticmethod
    @abstractmethod
    def sum(tensor: NNCFTensor, axes: List[int]) -> NNCFTensor:
        pass

    @staticmethod
    @abstractmethod
    def transpose(tensor: NNCFTensor, axes: List[int]) -> NNCFTensor:
        pass

    @staticmethod
    @abstractmethod
    def eps(dtype: TensorDtype) -> float:
        pass

    @staticmethod
    @abstractmethod
    def median(tensor: NNCFTensor) -> NNCFTensor:
        pass

    @staticmethod
    @abstractmethod
    def clip(tensor: NNCFTensor, min_val: float, max_val: Optional[float] = None) -> NNCFTensor:
        pass

    @staticmethod
    @abstractmethod
    def ones(shape: Union[int, List[int]], dtype: TensorDtype) -> NNCFTensor:
        pass

    @staticmethod
    @abstractmethod
    def squeeze(tensor: NNCFTensor) -> NNCFTensor:
        pass

    @staticmethod
    @abstractmethod
    def power(tensor: NNCFTensor, pwr: float) -> NNCFTensor:
        pass

    @staticmethod
    @abstractmethod
    def quantile(tensor: NNCFTensor, quantile: Union[float, List[float]], axis: Union[int, List[int]] = None) -> Union[float, List[float]]:
        pass

    @staticmethod
    @abstractmethod
    def logical_or(tensor1: NNCFTensor, tensor2: NNCFTensor) -> NNCFTensor:
        pass

    @staticmethod
    @abstractmethod
    def masked_mean(tensor: NNCFTensor, mask: NNCFTensor, axis: int = None, keepdims: bool = False) -> NNCFTensor:
        pass

    @staticmethod
    @abstractmethod
    def masked_median(tensor: NNCFTensor, mask: NNCFTensor, axis: int = None, keepdims: bool = False) -> NNCFTensor:
        pass