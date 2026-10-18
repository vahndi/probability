from typing import Union, List, Tuple, TYPE_CHECKING

from pandas import Series

if TYPE_CHECKING:
    from probability.distributions.continuous.normal import Normal
    from probability.distributions.continuous.students_t import StudentsT


class DataMinMixin(object):

    _data: Series

    def min(self) -> Union[float, int, str]:
        """
        Return the smallest value in the data.
        """
        return self._data.min()


class DataMaxMixin(object):

    _data: Series

    def max(self) -> Union[float, int, str]:
        """
        Return the largest value in the data.
        """
        return self._data.max()


class DataMeanMixin(object):

    _data: Series

    def mean(self) -> float:
        """
        Return the mean value of the data.
        """
        return self._data.mean()


class DataMedianMixin(object):

    _data: Series

    def median(self) -> Union[int, float]:
        """
        Return the median value of the data.
        """
        return self._data.median()


class DataStdMixin(object):

    _data: Series

    def std(self) -> float:
        """
        Return the standard deviation of the data.
        """
        return self._data.std()


class DataVarMixin(object):

    _data: Series

    def var(self) -> float:
        """
        Return the variance of the data.
        """
        return self._data.var()


class DataSemMixin(object):

    _data: Series

    def se(self) -> float:
        """
        Return the standard error of the mean of the data (s / sqrt(n)).
        """
        return float(self._data.sem())

    def sem(self) -> float:
        """
        Alias for se(): standard error of the mean of the data.
        """
        return self.se()

    def mean_dist(
            self,
            use_t: bool = False
    ) -> Union['Normal', 'StudentsT']:
        """
        Return the sampling distribution of the sample mean.

        :param use_t: If True, return a StudentsT distribution with n - 1
                      degrees of freedom; otherwise return a Normal
                      distribution.
        """
        mu = float(self._data.mean())
        se = self.se()
        if use_t:
            from probability.distributions.continuous.students_t import (
                StudentsT
            )
            return StudentsT(nu=len(self._data) - 1, mu=mu, sigma=se)
        from probability.distributions.continuous.normal import Normal
        return Normal(mu=mu, sigma=se)

    def ci(
            self,
            percent: float = 0.95,
            use_t: bool = False
    ) -> Tuple[float, float]:
        """
        Return the confidence interval around the sample mean.

        :param percent: Confidence level between 0 and 1 (default 0.95).
        :param use_t: If True, use the Student's t-distribution with n - 1
                      degrees of freedom; otherwise use the Normal distribution.
        """
        return self.mean_dist(use_t=use_t).interval(percent)


class DataModeMixin(object):

    _data: Series

    def mode(self) -> Union[int, float, str,
                            List[int], List[float], List[str]]:
        """
        Return the most frequently occurring value(s) in the data.
        """
        mode = self._data.mode()
        if len(mode) > 1:
            return mode.to_list()
        else:
            return mode[0]

