import matplotlib.pyplot as plt

from examples.colors import ML_APP_DARK_BLUE
from probability.distributions.discrete.poisson import Poisson
from mpl_format.axes.axis_utils import new_axes


k_wikipedia = list(range(21))
k_ml_app = list(range(30))


def plot_wikipedia_pmfs():
    """
    https://en.wikipedia.org/wiki/Poisson_distribution#/media/File:Poisson_pmf.svg
    """
    ax = new_axes(width=10, height=10)
    Poisson(lambda_=1).plot(k=k_wikipedia, kind='line',
                            color='gray', mfc='orange', marker='o', ax=ax)
    Poisson(lambda_=4).plot(k=k_wikipedia, kind='line',
                            color='gray', mfc='purple', marker='o', ax=ax)
    Poisson(lambda_=10).plot(k=k_wikipedia, kind='line',
                             color='gray', mfc='lightblue', marker='o', ax=ax)
    ax.set_ylim(0, 0.4)
    ax.set_title('Probability mass function')
    ax.legend(loc='upper right')
    plt.show()


def plot_wikipedia_cdfs():
    """
    https://en.wikipedia.org/wiki/Poisson_distribution#/media/File:Poisson_pmf.svg
    """
    ax = new_axes(width=10, height=10)
    Poisson(lambda_=1).cdf().plot(k=k_wikipedia, color='gray',
                                  mfc='orange', marker='o', ax=ax)
    Poisson(lambda_=4).cdf().plot(k=k_wikipedia, color='gray',
                                  mfc='purple', marker='o', ax=ax)
    Poisson(lambda_=10).cdf().plot(k=k_wikipedia, color='gray',
                                   mfc='lightblue', marker='o', ax=ax)
    ax.set_title('Cumulative distribution function')
    ax.legend(loc='lower right')
    plt.show()


def plot_ml_app_pmfs():
    """
    Machine Learning: A Probabilistic Perspective. Figure 2.6
    """
    _, ax = plt.subplots(ncols=2, figsize=(16, 9))
    Poisson(lambda_=1).plot(k=k_ml_app, color=ML_APP_DARK_BLUE, ax=ax[0])
    ax[0].legend(loc='upper right')
    Poisson(lambda_=10).plot(k=k_ml_app, color=ML_APP_DARK_BLUE, ax=ax[1])
    ax[1].legend(loc='upper right')
    plt.show()


if __name__ == '__main__':

    plot_wikipedia_pmfs()
    plot_wikipedia_cdfs()
    plot_ml_app_pmfs()
