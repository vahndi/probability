from typing import (
    Any, Hashable, List, Optional, Protocol, TypeVar, runtime_checkable
)

from pandas import Index, Series

SelfT = TypeVar('SelfT')


@runtime_checkable
class SupportsRVS1d(Protocol):
    """
    Protocol for 1D distributions that support random variate sampling.
    """
    def rvs(
            self,
            num_samples: int,
            random_state: Optional[int] = None
    ) -> Series: ...


@runtime_checkable
class SupportsBounds1d(Protocol):
    """
    Protocol for 1D distributions with lower and upper plotting/integration
    bounds.
    """
    @property
    def lower_bound(self) -> float: ...

    @property
    def upper_bound(self) -> float: ...


@runtime_checkable
class SupportsDataDistribution(Protocol):
    """
    Protocol for empirical data distributions wrapping a pandas Series.
    """
    _data: Series

    @property
    def data(self) -> Series: ...

    @property
    def name(self) -> Any: ...

    def rename(self: SelfT, name: str) -> SelfT: ...

    def filter_to(self: SelfT, other: Any) -> SelfT: ...

    def __len__(self) -> int: ...


@runtime_checkable
class CategoricalDistribution(SupportsDataDistribution, Protocol):
    """
    Protocol for categorical empirical distributions (Boolean, Nominal, Ordinal,
    Count) that expose discrete categories and filtering.
    """
    _categories: List[Any]

    @property
    def categories(self) -> list: ...

    @property
    def num_categories(self) -> int: ...

    def keep(self: SelfT, categories: Any) -> SelfT: ...

    def drop(self: SelfT, categories: Any) -> SelfT: ...

    def counts(self) -> Series: ...

    def pmf(self) -> Series: ...


@runtime_checkable
class NumericDataDistribution(SupportsDataDistribution, SupportsRVS1d, Protocol):
    """
    Protocol for numeric empirical data distributions with descriptive stats.
    """
    def mean(self) -> float: ...

    def var(self) -> float: ...

    def std(self) -> float: ...


@runtime_checkable
class SupportsDataSeries(Protocol):
    """
    Protocol for Series-of-distributions containers (DataSeriesMixin).
    """
    _data: Series

    @property
    def index(self) -> Index: ...

    @property
    def data(self) -> Series: ...

    @property
    def name(self) -> Hashable: ...

    def keys(self) -> List[str]: ...

    def lens(self) -> Series: ...


class SupportsTrendSeries(SupportsDataSeries, Protocol):
    """
    Protocol for ordered DataSeries classes supporting trend permutation tests.
    """
    def unsplit(self) -> Any: ...

    def _sample_series(
            self,
            unsplit_dist: Any,
            n_x: Series
    ) -> 'SupportsTrendSeries': ...

    def increase_probs(self) -> Series: ...

    def _p_trend(self, increasing: bool, n_iter: int = 1_000) -> float: ...
