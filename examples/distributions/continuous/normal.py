import matplotlib.pyplot as plt
from numpy.ma import arange

from probability.distributions.continuous.normal import Normal
from mpl_format.axes.axis_utils import new_axes

x = arange(-5, 5.05, 0.05)


def plot_wikipedia_pdfs():
    """
    https://en.wikipedia.org/wiki/Normal_distribution#/media/File:Normal_Distribution_PDF.svg
    """
    ax = new_axes()
    Normal(mu=0, sigma_sq=0.2).plot(x=x, color='blue', ax=ax)
    Normal(mu=0, sigma_sq=1.0).plot(x=x, color='red', ax=ax)
    Normal(mu=0, sigma_sq=5.0).plot(x=x, color='orange', ax=ax)
    Normal(mu=-2, sigma_sq=0.5).plot(x=x, color='green', ax=ax)
    ax.legend(loc='upper right')
    plt.show()


def plot_wikipedia_cdfs():
    """
    https://en.wikipedia.org/wiki/Normal_distribution#/media/File:Normal_Distribution_CDF.svg
    """
    ax = new_axes()
    Normal(mu=0, sigma_sq=0.2).cdf().plot(x=x, color='blue', ax=ax)
    Normal(mu=0, sigma_sq=1.0).cdf().plot(x=x, color='red', ax=ax)
    Normal(mu=0, sigma_sq=5.0).cdf().plot(x=x, color='orange', ax=ax)
    Normal(mu=-2, sigma_sq=0.5).cdf().plot(x=x, color='green', ax=ax)
    ax.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':

    plot_wikipedia_pdfs()
    plot_wikipedia_cdfs()
