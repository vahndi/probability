import matplotlib.pyplot as plt
from numpy.ma import arange

from probability.distributions.continuous.laplace import Laplace
from mpl_format.axes.axis_utils import new_axes

x = arange(-10, 10.1, 0.1)


def plot_wikipedia_pdfs():
    """
    https://en.wikipedia.org/wiki/Laplace_distribution#/media/File:Laplace_pdf_mod.svg
    """
    ax = new_axes()
    Laplace(mu=0, b=1).plot(x=x, color='red', ax=ax)
    Laplace(mu=0, b=2).plot(x=x, color='black', ax=ax)
    Laplace(mu=0, b=4).plot(x=x, color='blue', ax=ax)
    Laplace(mu=-5, b=4).plot(x=x, color='green', ax=ax)
    ax.legend(loc='upper right')
    plt.show()


def plot_wikipedia_cdfs():
    """
    https://en.wikipedia.org/wiki/Laplace_distribution#/media/File:Laplace_cdf_mod.svg
    """
    ax = new_axes()
    Laplace(mu=0, b=1).cdf().plot(x=x, color='red', ax=ax)
    Laplace(mu=0, b=2).cdf().plot(x=x, color='black', ax=ax)
    Laplace(mu=0, b=4).cdf().plot(x=x, color='blue', ax=ax)
    Laplace(mu=-5, b=4).cdf().plot(x=x, color='green', ax=ax)
    ax.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':

    plot_wikipedia_pdfs()
    plot_wikipedia_cdfs()
