from typing import Tuple, Optional

from matplotlib.axes import Axes
from numpy import linspace, ndarray, tile, tril
from pandas import Series, Index
from scipy.stats import gamma

from probability.plots import new_axes


class GammaExponential(object):
    """
    Class for calculating Bayesian probabilities using the Gamma Exponential distribution.

    Prior Hyper-parameters
    ----------------------
    * `α` and `β` OR `k` and `θ` are the hyper-parameters of the prior.
    * `k` and `α` are the same parameter, i.e. the shape parameter.
    * `θ` is the scale parameter.
    * `β` is the rate or inverse-scale parameter. `β = 1 / θ`
    * `a > 0` and can be interpreted as the number of prior observations.
    * `β > 0` and can be interpreted as the total time for prior observations.
    * `k > 0`
    * `θ > 0`

    Posterior Hyper-parameters
    --------------------------
    * `n` is the number of observations
    * `x̄` is the average time between observations

    Model parameters
    ----------------
    * `λ` is the rate of the exponential distribution `P(x) = λ·exp(-λx)`
    * `0 < λ`

    Links
    -----
    * https://en.wikipedia.org/wiki/Gamma_distribution
    * https://en.wikipedia.org/wiki/Exponential_distribution
    * https://en.wikipedia.org/wiki/Conjugate_prior#When_likelihood_function_is_a_continuous_distribution
    """
    def __init__(self, alpha: Optional[float] = None, beta: Optional[float] = None,
                 k: Optional[float] = None, theta: Optional[float] = None,
                 n: int = 0, x_mean: float = 0.0,
                 lambda_: Optional[ndarray] = None):
        """
        Create a new gamma-exponential model using the parameters of the prior gamma distribution.

        :param alpha: Scale parameter when providing the rate parameter.
        :param beta: Rate parameter.
        :param k: Scale parameter when providing the scale parameter.
        :param theta: Scale parameter. Inverse of rate parameter.
        :param lambda_: Values to define probability distribution at - controls granularity
        """
        self._parametrization = None
        if None not in (alpha, beta) and k is None and theta is None:
            self.alpha = alpha
            self.beta = beta
            self._parametrization = 'ab'
        elif None not in (k, theta) and alpha is None and beta is None:
            self.alpha = k
            self.beta = 1 / theta
            self._parametrization = 'kt'
        elif alpha is None and beta is None and k is None and theta is None:
            self.alpha = 0.001
            self.beta = 0.001
        else:
            raise ValueError('Either provide α and β or k and θ')
        self.n = n
        self.x_mean = x_mean
        self.lambda_ = lambda_ if lambda_ is not None else linspace(0, 10, 10001)

    def prior(self, lambda_: Optional[ndarray] = None) -> Series:
        """
        Return the prior probability of the parameter θ given the priors α, β

        `p(x|α,β)`

        :param lambda_: vector of possible `λ`s
        """
        lambda_ = lambda_ or self.lambda_
        if self._parametrization == 'ab':
            name = 'p(λ|α={},β={})'.format(self.alpha, self.beta)
        else:
            name = 'p(λ|k={},θ={})'.format(self.alpha, 1 / self.beta)
        return Series(
            data=gamma(a=self.alpha, scale=1 / self.beta).pdf(lambda_),
            index=Index(data=lambda_, name='λ'), name=name
        )

    def posterior(self, n: Optional[int] = None, x_mean: Optional[float] = None,
                  lambda_: Optional[ndarray] = None) -> Series:
        """
        Return the posterior probability of the parameters given the data n, x̄ and priors α,β

        `p(λ|n,x̄,α,β)`

        :param n: number of observations
        :param x_mean: average time between observations
        :param lambda_: vector of possible `λ`s
        """
        lambda_ = lambda_ if lambda_ is not None else self.lambda_
        n = n or self.n
        x_mean = x_mean or self.x_mean
        if self._parametrization == 'ab':
            name = 'p(λ|n={},x̄={},α={},β={})'.format(n, x_mean, self.alpha, self.beta)
        else:
            name = 'p(λ|n={},x̄={},k={},θ={})'.format(n, x_mean, self.alpha, 1 / self.beta)
        return Series(
            data=gamma(a=self.alpha + n, scale=1 / (self.beta + n * x_mean)).pdf(lambda_),
            index=Index(data=lambda_, name='λ'), name=name
        )

    def posterior_hpd(self, percent: float = 0.94,
                      n: Optional[int] = None, x_mean: Optional[float] = None) -> Tuple[float, float]:
        """
        Return the bounds of the highest posterior density region of the posterior distribution.

        `p(λ|n,x̄,α,β)`

        :param percent: percentage for the HPD
        :param n: number of observations
        :param x_mean: average time between observations
        """
        n = n or self.n
        x_mean = x_mean or self.x_mean
        dist = gamma(a=self.alpha + n, scale=1 / (self.beta + n * x_mean))
        return dist.interval(percent)

    def posterior_mean(self, n: Optional[int] = None, x_mean: Optional[float] = None) -> float:
        """
        Return the mean of the posterior distribution.

        :param n: number of observations
        :param x_mean: average time between observations
        """
        n = n or self.n
        x_mean = x_mean or self.x_mean
        dist = gamma(a=self.alpha + n, scale=1 / (self.beta + n * x_mean))
        return dist.mean()

    def plot_prior(self, lambda_: Optional[ndarray] = None,
                   color: Optional[str] = None, ax: Optional[Axes] = None) -> Axes:
        """
        Plot the prior probability of the parameter θ given the priors α, β

        `p(λ|α,β)`

        :param lambda_: vector of possible `λ`s
        :param color: Optional color for the series.
        :param ax: Optional matplotlib axes
        """
        ax = ax or new_axes()
        prior = self.prior(lambda_=lambda_)
        prior.plot(kind='line', label='α={}, β={}'.format(self.alpha, self.beta),
                   color=color or 'C0', ax=ax)
        ax.set_xlabel('λ')
        if self._parametrization == 'ab':
            y_label = 'p(λ|α,β)'
        else:
            y_label = 'p(λ|k,θ)'
        ax.set_ylabel(y_label)
        ax.legend()
        return ax

    def plot_posterior(self,
                       n: Optional[int] = None, x_mean: Optional[float] = None,
                       lambda_: Optional[ndarray] = None,
                       ndp: int = 2,
                       hpd_width: float = 0.94, hpd_y: Optional[float] = None, hpd_color: str = 'k',
                       label: Optional[str] = None, color: Optional[str] = None,
                       ax: Optional[Axes] = None) -> Axes:
        """
        Return the posterior probability of the parameters given the data n, m and priors α, β

        `p(λ|n,x̄,α,β)`

        :param lambda_: vector of possible `θ`s
        :param n: number of observations
        :param x_mean: average time between observations
        :param ndp: Number of decimal places to round the labels for the upper and lower bounds of the HPD and the mean.
        :param hpd_width: Width of the Highest Posterior Density region to plot (0 to 1). Defaults to 0.94
        :param hpd_y: Manual override of the y-coordinate for the HPD line. Defaults to posterior max / 10
        :param hpd_color: Color for the HPD line.
        :param label: Optional series label to override the default.
        :param color: Optional color for the series.
        :param ax: Optional matplotlib axes
        """
        ax = ax or new_axes()
        lambda_ = lambda_ if lambda_ is not None else self.lambda_
        n = n or self.n
        x_mean = x_mean or self.x_mean
        posterior = self.posterior(lambda_=lambda_, n=n, x_mean=x_mean)
        # plot distribution
        if self._parametrization == 'ab':
            label = label or 'α={}, β={}, n={}, x̄={}'.format(self.alpha, self.beta, n, x_mean)
        else:
            label = label or 'k={}, θ={}, n={}, x̄={}'.format(self.alpha, 1 / self.beta, n, x_mean)
        ax = posterior.plot(kind='line', label=label, color=color or 'C2', ax=ax)
        # plot posterior_hpd
        hpd_low, hpd_high = self.posterior_hpd(percent=hpd_width, n=n, x_mean=x_mean)
        hpd_y = hpd_y if hpd_y is not None else posterior.max() / 10
        ax.plot((hpd_low, hpd_high), (hpd_y, hpd_y), color=hpd_color)
        ax.text(hpd_low, hpd_y, str(round(hpd_low, ndp)), ha='right', va='top')
        ax.text(hpd_high, hpd_y, str(round(hpd_high, ndp)), ha='left', va='top')
        ax.text((hpd_low + hpd_high) / 2, posterior.max() * 0.5,
                '{:.0f}% HPD'.format(hpd_width * 100), ha='center', va='bottom')
        # plot mean
        mean = self.posterior_mean(n=n, x_mean=x_mean)
        ax.text(mean, posterior.max() * 0.95, 'mean = {}'.format(str(round(mean, ndp))), ha='center')
        # labels
        ax.set_xlabel('λ')
        if self._parametrization == 'ab':
            ax.set_ylabel('p(λ|n,x̄,α,β)')
        else:
            ax.set_ylabel('p(λ|n,x̄,k,θ)')
        ax.legend()
        return ax

    def sample_posterior(self, num_samples: int = 100000) -> ndarray:
        """
        Return random samples from the posterior distribution.

        :param num_samples: Number of samples to draw.
        """
        return gamma(
            a=self.alpha + self.n,
            scale=1 / (self.beta + self.n * self.x_mean)
        ).rvs(num_samples)

    def prob_posterior_greater(self, other: 'GammaExponential',
                               method: str = 'samples', num_samples: int = 100000) -> float:
        """
        Return the approximate probability that a random value of the posterior density is greater
        than that of another distribution.
        N.B. this method only produces and approximate solution and is slow due to the requirement to
        calculate the square matrix. Sampling from each distribution is much faster and more accurate.
        Use a closed form solution wherever possible.

        :param other: The other GammaExponential distribution.
        :param method: One of ['samples', 'approx']
        :param num_samples: Number of samples to draw. Recommend 100k for 'samples' and 10k+1 for 'approx'.
        :rtype: float
        """
        if method == 'samples':
            return (self.sample_posterior(num_samples) > other.sample_posterior(num_samples)).mean()
        elif method == 'approx':
            n_steps = num_samples or 10001
            # get PDFs
            pdf_self = self.posterior(lambda_=self.lambda_)
            pdf_other = other.posterior(lambda_=self.lambda_)
            # tile each pdf in orthogonal directions and multiply probabilities
            n_steps = len(self.lambda_)
            a_self = tile(pdf_self.values, (n_steps, 1)).T
            a_other = tile(pdf_other.values, (n_steps, 1))
            a_prod = a_self * a_other
            # sum p1(θ1) * p2(θ2) in lower triangle (excluding diagonal) i.e. where θ1 > θ2, and normalize
            p_total = tril(a_prod, k=1).sum() / a_prod.sum()
            return p_total
        else:
            raise ValueError("method must be one of ['samples', 'approx']")


if __name__ == '__main__':

    ge = GammaExponential(alpha=1, beta=1, n=50, x_mean=2)
    print(ge.posterior_mean())
