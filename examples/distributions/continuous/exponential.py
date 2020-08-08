import matplotlib.pyplot as plt
from numpy import arange

from probability.distributions.continuous.exponential import Exponential
from mpl_format.axes.axis_utils import new_axes


x = arange(0, 5.01, 0.05)


def plot_wikipedia_pdfs():
    """
    https://en.wikipedia.org/wiki/Exponential_distribution#/media/File:Exponential_probability_density.svg
    """
    ax = new_axes()
    Exponential(lambda_=0.5).plot(x=x, color='orange', ax=ax)
    Exponential(lambda_=1).plot(x=x, color='purple', ax=ax)
    Exponential(lambda_=1.5).plot(x=x, color='lightblue', ax=ax)
    ax.legend()
    ax.set_ylim(0, 1.5)
    plt.show()


def plot_wikipedia_cdfs():
    """
    https://en.wikipedia.org/wiki/Exponential_distribution#/media/File:Exponential_cdf.svg
    """
    ax = new_axes()
    Exponential(lambda_=0.5).cdf().plot(x=x, color='orange', ax=ax)
    Exponential(lambda_=1).cdf().plot(x=x, color='purple', ax=ax)
    Exponential(lambda_=1.5).cdf().plot(x=x, color='lightblue', ax=ax)
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.show()


if __name__ == '__main__':

    plot_wikipedia_pdfs()
    plot_wikipedia_cdfs()
