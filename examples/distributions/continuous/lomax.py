import matplotlib.pyplot as plt
from numpy import arange

from probability.distributions.continuous.lomax import Lomax
from mpl_format.axes.axis_utils import new_axes


x = arange(0, 6.01, 0.05)


def plot_wikipedia_pdfs():
    """
    https://en.wikipedia.org/wiki/Lomax_distribution#/media/File:LomaxPDF.png
    """
    ax = new_axes()
    Lomax(lambda_=1, alpha=2).plot(x=x, color='blue', ax=ax)
    Lomax(lambda_=2, alpha=2).plot(x=x, color='green', ax=ax)
    Lomax(lambda_=4, alpha=1).plot(x=x, color='red', ax=ax)
    Lomax(lambda_=6, alpha=1).plot(x=x, color='orange', ax=ax)
    ax.legend()
    plt.show()


def plot_wikipedia_cdfs():
    """
    https://en.wikipedia.org/wiki/Lomax_distribution#/media/File:LomaxCDF.png
    """
    ax = new_axes()
    Lomax(lambda_=1, alpha=2).cdf().plot(x=x, color='blue', ax=ax)
    Lomax(lambda_=2, alpha=2).cdf().plot(x=x, color='green', ax=ax)
    Lomax(lambda_=4, alpha=1).cdf().plot(x=x, color='red', ax=ax)
    Lomax(lambda_=6, alpha=1).cdf().plot(x=x, color='orange', ax=ax)
    ax.legend()
    plt.show()


if __name__ == '__main__':

    plot_wikipedia_pdfs()
    plot_wikipedia_cdfs()
