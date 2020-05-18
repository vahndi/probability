import matplotlib.pyplot as plt
from mpl_format.axes.axis_utils import new_axes
from numpy.ma import arange

from probability.distributions import Beta
from probability.distributions.conjugate.beta_binomial import BetaBinomial


k = range(11)
x = arange(0, 1, 0.01)


def plot_wikipedia_pmfs():
    """
    https://en.wikipedia.org/wiki/Beta-binomial_distribution#/media/File:Beta-binomial_distribution_pmf.png
    """
    kwargs = dict(kind='line', marker='o')
    ax = new_axes(width=10, height=10)
    BetaBinomial(alpha=0.2, beta=0.25, n=10, m=1).plot(k=k, color='black', ax=ax, **kwargs)
    BetaBinomial(alpha=0.7, beta=2, n=10, m=1).plot(k=k, color='red', ax=ax, **kwargs)
    BetaBinomial(alpha=2, beta=2, n=10, m=1).plot(k=k, color='green', ax=ax, **kwargs)
    BetaBinomial(alpha=600, beta=400, n=10, m=1).plot(k=k, color='blue', ax=ax, **kwargs)
    ax.legend()
    plt.show()


def plot_wikipedia_cdfs():
    """
    https://en.wikipedia.org/wiki/Beta-binomial_distribution#/media/File:Beta-binomial_cdf.png
    """
    ax = new_axes(width=10, height=10)
    BetaBinomial(alpha=0.2, beta=0.25, n=10, m=1).cdf().plot(k=k, color='black', ax=ax)
    BetaBinomial(alpha=0.7, beta=2, n=10, m=1).cdf().plot(k=k, color='red', ax=ax)
    BetaBinomial(alpha=2, beta=2, n=10, m=1).cdf().plot(k=k, color='green', ax=ax)
    BetaBinomial(alpha=600, beta=400, n=10, m=1).cdf().plot(k=k, color='blue', ax=ax)
    ax.legend()
    plt.show()


def plot_ml_app():
    """
    Machine Learning: A Probabilistic Perspective. Figure 3.6
    """
    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    bb_1 = BetaBinomial(n=20, alpha=2, beta=2, m=1)
    bb_1.prior().plot(x=x, color='red', ax=ax[0])
    Beta(alpha=3, beta=17).plot(x=x, color='black', ax=ax[0])  # using a Beta to plot likelihood on the same scale
    bb_1.posterior(m=3).plot(x=x, color='blue', ax=ax[0])
    ax[0].legend()
    bb_2 = BetaBinomial(n=20, alpha=5, beta=2, m=1)
    bb_2.prior().plot(x=x, color='red', ax=ax[1])
    Beta(alpha=11, beta=13).plot(x=x, color='black', ax=ax[1])  # using a Beta to plot likelihood on the same scale
    bb_2.posterior(m=11).plot(x=x, color='blue', ax=ax[1])
    ax[1].legend()
    plt.show()


if __name__ == '__main__':

    plot_wikipedia_pmfs()
    plot_wikipedia_cdfs()
    plot_ml_app()
