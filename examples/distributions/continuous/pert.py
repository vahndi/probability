# https://en.wikipedia.org/wiki/PERT_distribution

import matplotlib.pyplot as plt
from mpl_format.axes.axis_utils import new_axes
from numpy import linspace
from probability.distributions import PERT

x = linspace(0, 100, 1001)


def plot_wikipedia_pdfs():
    """
    https://en.wikipedia.org/wiki/PERT_distribution#/media/File:PERT_pdf_examples.jpg
    """
    ax = new_axes(width=10, height=10)
    PERT(0, 10, 100).pdf().plot(x=x, color='blue', ax=ax)
    PERT(0, 50, 100).pdf().plot(x=x, color='orange', ax=ax)
    PERT(0, 70, 100).pdf().plot(x=x, color='gray', ax=ax)
    ax.set_ylim(0, 0.03)
    ax.set_title('Probability density function')
    ax.legend(loc='upper center')
    plt.show()


def plot_wikipedia_cdfs():
    """
    https://en.wikipedia.org/wiki/PERT_distribution#/media/File:PERT_cdf_examples.jpg
    """
    ax = new_axes(width=10, height=10)
    PERT(0, 10, 100).cdf().plot(x=x, color='blue', ax=ax)
    PERT(0, 50, 100).cdf().plot(x=x, color='orange', ax=ax)
    PERT(0, 70, 100).cdf().plot(x=x, color='gray', ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title('Cumulative density function')
    ax.legend(loc='upper center')
    plt.show()


if __name__ == '__main__':

    plot_wikipedia_pdfs()
    plot_wikipedia_cdfs()

