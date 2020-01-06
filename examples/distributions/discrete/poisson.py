import matplotlib.pyplot as plt

from probability.distributions.discrete import Binomial
from probability.distributions.discrete.poisson import Poisson
from probability.plots import new_axes


# https://en.wikipedia.org/wiki/Poisson_distribution

k = list(range(21))


def plot_pmfs():
    """
    https://en.wikipedia.org/wiki/Poisson_distribution#/media/File:Poisson_pmf.svg
    """
    ax = new_axes(width=10, height=10)
    Poisson(lambda_=1).plot(k=k, color='orange', ax=ax)
    Poisson(lambda_=4).plot(k=k, color='purple', ax=ax)
    Poisson(lambda_=10).plot(k=k, color='lightblue', ax=ax)
    ax.set_ylim(0, 0.4)
    ax.set_title('Probability mass function')
    ax.legend(loc='upper right')
    plt.show()


def plot_cdfs():
    """
    https://en.wikipedia.org/wiki/Poisson_distribution#/media/File:Poisson_pmf.svg
    """
    ax = new_axes(width=10, height=10)
    Poisson(lambda_=1).cdf().plot(k=k, color='orange', ax=ax)
    Poisson(lambda_=4).cdf().plot(k=k, color='purple', ax=ax)
    Poisson(lambda_=10).cdf().plot(k=k, color='lightblue', ax=ax)
    ax.set_title('Cumulative distribution function')
    ax.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':

    plot_pmfs()
    plot_cdfs()
