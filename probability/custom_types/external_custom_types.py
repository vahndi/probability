from typing import Union, Iterable, Sized, Mapping, Any

from numpy import ndarray
from pandas import Series

from probability.calculations.mixins import ProbabilityCalculationMixin

Array1d = Union[Iterable[int], Iterable[float], ndarray]
FloatArray1d = Union[Iterable[float], ndarray, Series, Sized]
FloatOrFloatArray1d = Union[float, Iterable[float], ndarray]

Array2d = Union[Iterable[Iterable[int]], Iterable[Iterable[float]], ndarray]
FloatArray2d = Union[Iterable[Iterable[float]], ndarray]
FloatOrFloatArray2d = Union[float, Iterable[Iterable[float]], ndarray]

AnyFloatMap = Mapping[Any, float]
IntFloatMap = Mapping[int, float]
AnyCalcMap = Mapping[Any, ProbabilityCalculationMixin]
