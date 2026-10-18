from typing import (
    Any, Hashable, List, Optional, Protocol, Tuple, TypeVar, Union,
    runtime_checkable
)

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame, Index, Series

SelfT = TypeVar('SelfT')

NUM_SAMPLES_COMPARISON = 100_000


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
class SupportsRVSNd(Protocol):
    """
    Protocol for N-dimensional distributions that support random variate
    sampling into a DataFrame.
    """
    _names: List[str]

    def rvs(
            self,
            num_samples: int,
            random_state: Optional[int] = None,
            full_name: bool = False
    ) -> DataFrame: ...


SupportsRVS = Union[SupportsRVS1d, SupportsRVSNd]


def is_rvs(arg: Any) -> bool:
    """
    Return True if the arg can be sampled as a random variable.
    """
    return isinstance(arg, (SupportsRVS1d, SupportsRVSNd))



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
class SupportsMoments1d(Protocol):
    """
    Protocol for 1D distributions providing descriptive moments.
    """
    def mean(self) -> float: ...

    def median(self) -> float: ...

    def std(self) -> float: ...

    def var(self) -> float: ...


@runtime_checkable
class SupportsPDF1d(Protocol):
    """
    Protocol for 1D continuous distributions exposing PDF and log-PDF functions.
    """
    def pdf(self) -> Any: ...

    def log_pdf(self) -> Any: ...


@runtime_checkable
class SupportsPMF1d(Protocol):
    """
    Protocol for 1D discrete distributions exposing PMF and log-PMF functions.
    """
    def pmf(self) -> Any: ...

    def log_pmf(self) -> Any: ...


@runtime_checkable
class SupportsCDF1d(Protocol):
    """
    Protocol for 1D distributions exposing CDF and log-CDF functions.
    """
    def cdf(self) -> Any: ...

    def log_cdf(self) -> Any: ...


@runtime_checkable
class SupportsSF1d(Protocol):
    """
    Protocol for 1D distributions exposing survival and log-survival functions.
    """
    def sf(self) -> Any: ...

    def log_sf(self) -> Any: ...


@runtime_checkable
class SupportsPPF1d(Protocol):
    """
    Protocol for 1D distributions exposing the percent point function (inverse
    CDF).
    """
    def ppf(self) -> Any: ...


@runtime_checkable
class SupportsHDI1d(SupportsPPF1d, Protocol):
    """
    Protocol for 1D continuous distributions supporting Highest Density Interval
    calculation.
    """
    def hdi(self, credible_mass: float) -> Tuple[float, float]: ...


@runtime_checkable
class SupportsRV1d(
    SupportsRVS1d, SupportsMoments1d, SupportsCDF1d, SupportsSF1d,
    SupportsPPF1d, Protocol
):
    """
    Protocol for 1D parametric random variable distributions (continuous or
    discrete).
    """
    def interval(self, percent: float) -> Tuple[float, float]: ...

    def plot(self, *args: Any, **kwargs: Any) -> Axes: ...


@runtime_checkable
class SupportsContinuousRV1d(
    SupportsRV1d, SupportsBounds1d, SupportsPDF1d, SupportsHDI1d, Protocol
):
    """
    Protocol for 1D continuous parametric distributions.
    """
    pass


@runtime_checkable
class SupportsDiscreteRV1d(
    SupportsRV1d, SupportsPMF1d, Protocol
):
    """
    Protocol for 1D discrete parametric distributions.
    """
    pass


@runtime_checkable
class SupportsConjugate(Protocol):
    """
    Protocol for Bayesian conjugate models providing prior, likelihood, and
    posterior distributions.
    """
    def prior(self, **kwargs: Any) -> SupportsRV1d: ...

    def likelihood(self, **kwargs: Any) -> SupportsRV1d: ...

    def posterior(self, **kwargs: Any) -> SupportsRV1d: ...

    def plot(self, **kwargs: Any) -> Figure: ...


@runtime_checkable
class SupportsPredictive(Protocol):
    """
    Protocol for Bayesian conjugate models that expose prior and posterior
    predictive distributions.
    """
    def prior_predictive(self, **kwargs: Any) -> SupportsRV1d: ...

    def posterior_predictive(self, **kwargs: Any) -> SupportsRV1d: ...


@runtime_checkable
class SupportsResetDistribution(Protocol):
    """
    Protocol for distribution classes that rebuild their underlying scipy
    distribution when parameters are mutated.
    """
    def _reset_distribution(self) -> None: ...


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
class NumericDataDistribution(
    SupportsDataDistribution, SupportsRVS1d, Protocol
):
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
