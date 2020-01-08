from numpy import ndarray
from typing import Union, Iterable

from probability.distributions.mixins.rv_continuous_1d_mixin import RVContinuous1dMixin
from probability.distributions.mixins.rv_discrete_1d_mixin import RVDiscrete1dMixin


RVMixin = Union[RVContinuous1dMixin, RVDiscrete1dMixin]

Array1d = Union[Iterable[int], Iterable[float], ndarray]
FloatArray1d = Union[Iterable[float], ndarray]
FloatOrFloatArray1d = Union[float, Iterable[float], ndarray]

Array2d = Union[Iterable[Iterable[int]], Iterable[Iterable[float]], ndarray]
FloatArray2d = Union[Iterable[Iterable[float]], ndarray]
FloatOrFloatArray2d = Union[float, Iterable[Iterable[float]], ndarray]
