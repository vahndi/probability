import matplotlib.pyplot as plt
from numpy import linspace

from probability.distributions import Beta
from mpl_format.axes.axis_utils import new_axes

# https://en.wikipedia.org/wiki/Beta_distribution

x = linspace(0, 1, 101)


def plot_wikipedia_pdfs():
    """
    https://en.wikipedia.org/wiki/Beta_distribution#/media/File:Beta_distribution_pdf.svg
    """
    ax = new_axes(width=10, height=10)
    Beta(0.5, 0.5).pdf().plot(x=x, color='red', mean=True, std=True, ax=ax)
    Beta(5, 1).pdf().plot(x=x, color='blue', mean=True, ax=ax)
    Beta(1, 3).pdf().plot(x=x, color='green', mean=True, ax=ax)
    Beta(2, 2).pdf().plot(x=x, color='purple', mean=True, ax=ax)
    Beta(2, 5).pdf().plot(x=x, color='orange', mean=True, ax=ax)
    ax.set_ylim(0, 2.5)
    ax.set_title('Probability density function')
    ax.legend(loc='upper center')
    plt.show()


def plot_wikipedia_cdfs():
    """
    https://en.wikipedia.org/wiki/Beta_distribution#/media/File:Beta_distribution_cdf.svg
    """
    ax = new_axes(width=10, height=10)
    Beta(0.5, 0.5).cdf().plot(x=x, color='red', ax=ax)
    Beta(5, 1).cdf().plot(x=x, color='blue', ax=ax)
    Beta(1, 3).cdf().plot(x=x, color='green', ax=ax)
    Beta(2, 2).cdf().plot(x=x, color='purple', ax=ax)
    Beta(2, 5).cdf().plot(x=x, color='orange', ax=ax)
    ax.set_title('Cumulative distribution function')
    ax.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':

    plot_wikipedia_pdfs()
    plot_wikipedia_cdfs()


