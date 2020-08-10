import matplotlib.pyplot as plt

from examples.colors import ML_APP_DARK_BLUE
from probability.distributions.discrete import Binomial
from mpl_format.axes.axis_utils import new_axes


k_wiki = list(range(41))
k_ml_app = list(range(11))


def plot_wikipedia_pmfs():
    """
    https://en.wikipedia.org/wiki/Binomial_distribution#/media/File:Binomial_distribution_pmf.svg
    """
    ax = new_axes()
    Binomial(n=20, p=0.5).plot(
        k=k_wiki, kind='line', color='blue', ax=ax, ls='', marker='d',
        mean=True
    )
    Binomial(n=20, p=0.7).plot(
        k=k_wiki, kind='line', color='lightgreen', ax=ax, ls='', marker='s'
    )
    Binomial(n=40, p=0.5).plot(
        k=k_wiki, kind='line', color='red', ax=ax, ls='', marker='o'
    )
    ax.set_ylim(0, 0.25)
    ax.set_title('Probability mass function')
    ax.legend(loc='upper right')
    plt.show()


def plot_wikipedia_cdfs():
    """
    https://en.wikipedia.org/wiki/Binomial_distribution#/media/File:Binomial_distribution_cdf.svg
    """
    ax = new_axes()
    Binomial(n=20, p=0.5).cdf().plot(
        k=k_wiki, kind='line', color='blue', ax=ax
    )
    Binomial(n=20, p=0.7).cdf().plot(
        k=k_wiki, kind='line', color='lightgreen', ax=ax
    )
    Binomial(n=40, p=0.5).cdf().plot(
        k=k_wiki, kind='line', color='red', ax=ax
    )
    ax.set_ylim(0, 1.0)
    ax.set_title('Cumulative distribution function')
    ax.legend(loc='upper right')
    plt.show()


def plot_ml_app_pmfs():
    """
    Machine Learning: A Probabilistic Perspective. Figure 2.4
    """
    _, ax = plt.subplots(ncols=2, figsize=(16, 9))
    Binomial(n=10, p=0.25).plot(k=k_ml_app, color=ML_APP_DARK_BLUE, ax=ax[0])
    ax[0].legend(loc='upper right')
    Binomial(n=10, p=0.9).plot(k=k_ml_app, color=ML_APP_DARK_BLUE, ax=ax[1])
    ax[1].legend(loc='upper left')
    plt.show()


def plot_scipy_pmf():
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html
    """
    ax = Binomial(n=5, p=0.4).plot(
        k=range(0, 5), kind='line', ls='', marker='o', color='blue', vlines=True
    )
    ax.set_xlim(-0.5, 4.5)
    plt.show()


if __name__ == '__main__':

    plot_wikipedia_pmfs()
    plot_wikipedia_cdfs()
    plot_ml_app_pmfs()
    plot_scipy_pmf()
