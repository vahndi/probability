from itertools import product
from typing import TypeVar, Dict, Optional, Tuple

from pandas import Series, DataFrame

from probability.distributions.mixins.rv_mixins import NUM_SAMPLES_COMPARISON
from probability.distributions.protocols import (
    CategoricalDistribution, NumericDataDistribution
)

DCDM = TypeVar('DCDM', bound=NumericDataDistribution)
DPCM = TypeVar('DPCM', bound='DataProbComparisonMixin')


class DataProbComparisonMixin(object):

    def _check_can_compare(self: DPCM, other: DPCM):

        if not isinstance(other, type(self)):
            cls_name = type(self).__name__
            article = 'an' if cls_name[0].lower() in 'aeiou' else 'a'
            raise TypeError(
                f'Can only compare {article} {cls_name} with another {cls_name}'
            )

    def _comparison_samples(
            self: DPCM, other: DPCM,
            num_samples: Optional[int] = None
    ) -> Tuple[Series, Series]:
        """
        Method to make it faster and more accurate to implement <= and >=.
        """
        if num_samples is None:
            num_samples = NUM_SAMPLES_COMPARISON
        self_samples = self.rvs(num_samples)
        other_samples = other.rvs(num_samples)
        return self_samples, other_samples

    def prob_equal_to(self: DPCM, other: DPCM,
                      num_samples: int = NUM_SAMPLES_COMPARISON) -> float:
        """
        Return the probability that self = other. based on sampling.
        """
        self._check_can_compare(other)
        self_values, other_values = self._comparison_samples(other, num_samples)
        return (self_values == other_values).mean()

    def __eq__(self: DPCM, other: DPCM) -> float:
        """
        Return the probability that self = other. based on sampling.
        """
        return self.prob_equal_to(other)

    def prob_not_equal_to(self: DPCM, other: DPCM,
                          num_samples: int = NUM_SAMPLES_COMPARISON) -> float:
        """
        Return the probability that self != other. based on sampling.
        """
        return 1 - self.prob_equal_to(other, num_samples)

    def __ne__(self: DPCM, other: DPCM) -> float:
        """
        Return the probability that self != other. based on sampling.
        """
        return self.prob_not_equal_to(other)

    def prob_less_than(self: DPCM, other: DPCM,
                       num_samples: int = NUM_SAMPLES_COMPARISON) -> float:
        """
        Return the probability that self < other. based on sampling.
        """
        self._check_can_compare(other)
        self_values, other_values = self._comparison_samples(other, num_samples)
        return (self_values < other_values).mean()

    def __lt__(self: DPCM, other: DPCM) -> float:
        """
        Return the probability that self < other. based on sampling.
        """
        return self.prob_less_than(other)

    def prob_greater_than(self: DPCM, other: DPCM,
                          num_samples: int = NUM_SAMPLES_COMPARISON) -> float:
        """
        Return the probability that self > other. based on sampling.
        """
        self._check_can_compare(other)
        self_values, other_values = self._comparison_samples(other, num_samples)
        return (self_values > other_values).mean()

    def __gt__(self: DPCM, other: DPCM) -> float:
        """
        Return the probability that self > other. based on sampling.
        """
        return self.prob_greater_than(other)

    def __le__(self: DPCM, other: DPCM) -> float:
        """
        Return the probability that self <= other. based on sampling.
        """
        self._check_can_compare(other)
        self_values, other_values = self._comparison_samples(other)
        return (self_values <= other_values).mean()

    def __ge__(self: DPCM, other: DPCM) -> float:
        """
        Return the probability that self >= other. based on sampling.
        """
        self._check_can_compare(other)
        self_values, other_values = self._comparison_samples(other)
        return (self_values >= other_values).mean()


class DataCohensDMixin(object):

    _data: Series

    def cohens_d(self: DCDM, other: DCDM) -> float:
        """
        Calculate the Cohen's d standardized difference of means between self
        and other.

        https://en.wikipedia.org/wiki/Effect_size#Cohen's_d
        """
        n1, n2 = len(self), len(other)
        v1, v2 = self._data.var(), other._data.var()
        x1, x2 = self._data.mean(), other._data.mean()
        # pooled standard deviation
        s = (
            ((n1 - 1) * v1 + (n2 - 1) * v2) /
            (n1 + n2 - 2)
        ) ** 0.5
        return (x1 - x2) / s

    def conditional_cohens_d(
            self: DCDM,
            categorical: CategoricalDistribution
    ) -> DataFrame:
        """
        Return a matrix of the Cohen's d of the Ratio conditioned on each
        category compared to it conditioned on each other category.
        Index represents control and columns represent treatment.

        :param categorical: Categorical data distribution to condition on.

        https://en.wikipedia.org/wiki/Effect_size#Cohen's_d
        """
        dists: Dict[str, DCDM] = {
            category: self.filter_to(
                categorical.keep(category)
            ).rename(category)
            for category in categorical.categories
        }
        v: Dict[str, float] = {
            category: dist.var()
            for category, dist in dists.items()
        }
        n: Dict[str, int] = {
            category: len(dist)
            for category, dist in dists.items()
        }
        x: Dict[str, float] = {
            category: dist.mean()
            for category, dist in dists.items()
        }
        results = []
        for control, treatment in product(dists.keys(), dists.keys()):
            n1, n2 = n[control], n[treatment]
            v1, v2 = v[control], v[treatment]
            x1, x2 = x[control], x[treatment]
            s = (
                    ((n1 - 1) * v1 + (n2 - 1) * v2) /
                    (n1 + n2 - 2)
            ) ** 0.5
            results.append({
                'control': control,
                'treatment': treatment,
                'd': (x2 - x1) / s
            })
        return DataFrame(results).pivot(
            index='control', columns='treatment', values='d'
        ).reindex(dists.keys(), axis=0).reindex(dists.keys(), axis=1)


class DataHypothesisTestMixin(object):
    """
    Mixin providing two-sample difference-in-means distributions, confidence
    intervals, test statistics, p-values, and power/sample-size calculations.
    """
    _data: Series

    def _welch_df(self: DCDM, other: DCDM) -> float:
        se1_sq = self.se() ** 2
        se2_sq = other.se() ** 2
        n1, n2 = len(self), len(other)
        num = (se1_sq + se2_sq) ** 2
        den = (se1_sq ** 2) / (n1 - 1) + (se2_sq ** 2) / (n2 - 1)
        return float(num / den)

    def mean_diff(
            self: DCDM,
            other: DCDM,
            null_diff: float = 0.0,
            use_t: bool = False
    ):
        """
        Return the sampling distribution of the difference in means
        (self.mean() - other.mean() - null_diff).

        :param other: Control or comparison distribution.
        :param null_diff: Hypothesized difference under H0 (shifts the mean).
        :param use_t: If True, return a Welch-Satterthwaite StudentsT
                      distribution; otherwise return a Normal distribution.
        """
        diff_mu = float(self.mean() - other.mean() - null_diff)
        diff_se = float((self.se() ** 2 + other.se() ** 2) ** 0.5)
        if use_t:
            from probability.distributions.continuous.students_t import (
                StudentsT
            )
            return StudentsT(
                nu=self._welch_df(other),
                mu=diff_mu,
                sigma=diff_se
            )
        from probability.distributions.continuous.normal import Normal
        return Normal(mu=diff_mu, sigma=diff_se)

    def mean_diff_ci(
            self: DCDM,
            other: DCDM,
            percent: float = 0.95,
            null_diff: float = 0.0,
            use_t: bool = False
    ) -> Tuple[float, float]:
        """
        Return the confidence interval for the difference in means
        (self.mean() - other.mean() - null_diff).
        """
        return self.mean_diff(
            other=other,
            null_diff=null_diff,
            use_t=use_t
        ).interval(percent)

    def t_stat(
            self: DCDM,
            other: DCDM,
            null_diff: float = 0.0
    ) -> float:
        """
        Return the two-sample test statistic for the difference in means:
        ((mean(self) - mean(other)) - null_diff) / sqrt(se(self)^2 + se(other)^2).
        """
        diff_mu = float(self.mean() - other.mean() - null_diff)
        diff_se = float((self.se() ** 2 + other.se() ** 2) ** 0.5)
        return diff_mu / diff_se

    def p_value(
            self: DCDM,
            other: DCDM,
            null_diff: float = 0.0,
            two_sided: bool = True,
            use_t: bool = False
    ) -> float:
        """
        Return the p-value for testing H0: mean(self) - mean(other) == null_diff.

        :param other: Comparison distribution.
        :param null_diff: Null hypothesis difference (default 0.0).
        :param two_sided: If True, compute two-sided p-value; otherwise compute
                          upper one-sided p-value for mean(self) - mean(other) > null_diff.
        :param use_t: If True, evaluate against Student's t with Welch df;
                      otherwise evaluate against StandardNormal.
        """
        stat = self.t_stat(other=other, null_diff=null_diff)
        if use_t:
            from probability.distributions.continuous.students_t import (
                StudentsT
            )
            ref_dist = StudentsT(nu=self._welch_df(other), mu=0.0, sigma=1.0)
        else:
            from probability.distributions.continuous.normal import Normal
            ref_dist = Normal.StandardNormal()

        if two_sided:
            return float(2.0 * (1.0 - ref_dist.cdf().at(abs(stat))))
        return float(1.0 - ref_dist.cdf().at(stat))

    def sample_size_for_effect(
            self: DCDM,
            mde: float,
            alpha: float = 0.05,
            power: float = 0.80,
            two_sided: bool = True,
            approx_16: bool = False
    ) -> int:
        """
        Calculate the required sample size per group to detect a minimum
        detectable effect `mde` given baseline standard deviation `self.std()`.

        :param mde: Minimum detectable effect (delta) in the same units as self.
        :param alpha: Significance level (default 0.05).
        :param power: Statistical power 1 - beta (default 0.80).
        :param two_sided: Whether the test is two-sided (default True).
        :param approx_16: If True, use the rule-of-thumb multiplier 16 * (std / mde)^2
                          for alpha=0.05, power=0.80.
        """
        from math import ceil
        sigma = float(self.std())
        if approx_16:
            return int(ceil(16.0 * (sigma / mde) ** 2))
        from probability.distributions.continuous.normal import Normal
        std_norm = Normal.StandardNormal()
        z_alpha = abs(
            std_norm.ppf().at(alpha / 2.0 if two_sided else alpha)
        )
        z_power = std_norm.ppf().at(power)
        multiplier = 2.0 * (z_alpha + z_power) ** 2
        return int(ceil(multiplier * (sigma / mde) ** 2))


