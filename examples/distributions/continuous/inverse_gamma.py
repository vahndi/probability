import matplotlib.pyplot as plt
from numpy import arange

from probability.distributions.continuous.inverse_gamma import InverseGamma
from mpl_format.axes.axis_utils import new_axes

x = arange(0, 3.01, 0.01)


def plot_wikipedia_pdfs():

    ax = new_axes()
    InverseGamma(alpha=1, beta=1).plot(x=x, color='red', ax=ax)
    InverseGamma(alpha=2, beta=1).plot(x=x, color='green', ax=ax)
    InverseGamma(alpha=3, beta=1).plot(x=x, color='blue', ax=ax)
    InverseGamma(alpha=3, beta=0.5).plot(x=x, color='cyan', ax=ax)
    ax.set_ylim(0, 5)
    ax.legend()
    plt.show()


def plot_wikipedia_cdfs():

    ax = new_axes()
    InverseGamma(alpha=1, beta=1).cdf().plot(x=x, color='red', ax=ax)
    InverseGamma(alpha=2, beta=1).cdf().plot(x=x, color='green', ax=ax)
    InverseGamma(alpha=3, beta=1).cdf().plot(x=x, color='blue', ax=ax)
    InverseGamma(alpha=3, beta=0.5).cdf().plot(x=x, color='cyan', ax=ax)
    ax.legend()
    plt.show()


if __name__ == '__main__':

    plot_wikipedia_pdfs()
    plot_wikipedia_cdfs()
