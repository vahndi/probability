import matplotlib.pyplot as plt

from probability.distributions.discrete import Binomial
from probability.plots import new_axes


# https://en.wikipedia.org/wiki/Binomial_distribution

k = list(range(41))


def plot_pmfs():
    """
    https://en.wikipedia.org/wiki/Binomial_distribution#/media/File:Binomial_distribution_pmf.svg
    """
    ax = new_axes(width=10, height=10)
    Binomial(n=20, p=0.5).pmf().plot(k=k, color='blue', ax=ax)
    Binomial(n=20, p=0.7).pmf().plot(k=k, color='lightgreen', ax=ax)
    Binomial(n=40, p=0.5).pmf().plot(k=k, color='red', ax=ax)
    ax.set_ylim(0, 0.25)
    ax.set_title('Probability mass function')
    ax.legend(loc='upper right')
    plt.show()


def plot_cdfs():
    """
    https://en.wikipedia.org/wiki/Binomial_distribution#/media/File:Binomial_distribution_cdf.svg
    """
    ax = new_axes(width=10, height=10)
    Binomial(n=20, p=0.5).cdf().plot(k=k, color='blue', ax=ax)
    Binomial(n=20, p=0.7).cdf().plot(k=k, color='lightgreen', ax=ax)
    Binomial(n=40, p=0.5).cdf().plot(k=k, color='red', ax=ax)
    ax.set_ylim(0, 1.0)
    ax.set_title('Cumulative distribution function')
    ax.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':

    plot_pmfs()
    plot_cdfs()
