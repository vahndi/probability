import matplotlib.pyplot as plt
from numpy.ma import arange

from probability.distributions.continuous.gamma import Gamma
from mpl_format.axes.axis_utils import new_axes


# https://en.wikipedia.org/wiki/Gamma_distribution


x = arange(0, 20.1, 0.1)


def plot_wikipedia_pdfs():
    """
    https://en.wikipedia.org/wiki/Gamma_distribution#/media/File:Gamma_distribution_pdf.svg
    """
    ax = new_axes()
    Gamma.from_k_theta(k=1, theta=2).plot(x=x, color='red', ax=ax)
    Gamma.from_k_theta(k=2, theta=2).plot(x=x, color='orange', ax=ax)
    Gamma.from_k_theta(k=3, theta=2).plot(x=x, color='yellow', ax=ax)
    Gamma.from_k_theta(k=5, theta=1).plot(x=x, color='green', ax=ax)
    Gamma.from_k_theta(k=9, theta=0.5).plot(x=x, color='black', ax=ax)
    Gamma.from_k_theta(k=7.5, theta=1).plot(x=x, color='blue', ax=ax)
    Gamma.from_k_theta(k=0.5, theta=1).plot(x=x, color='purple', ax=ax)
    ax.set_ylim(0, 0.5)
    ax.legend(loc='upper right')
    plt.show()


def plot_wikipedia_cdfs():
    """
    https://en.wikipedia.org/wiki/Gamma_distribution#/media/File:Gamma_distribution_cdf.svg
    """
    ax = new_axes()
    Gamma.from_k_theta(k=1, theta=2).cdf().plot(x=x, color='red', ax=ax)
    Gamma.from_k_theta(k=2, theta=2).cdf().plot(x=x, color='orange', ax=ax)
    Gamma.from_k_theta(k=3, theta=2).cdf().plot(x=x, color='yellow', ax=ax)
    Gamma.from_k_theta(k=5, theta=1).cdf().plot(x=x, color='green', ax=ax)
    Gamma.from_k_theta(k=9, theta=0.5).cdf().plot(x=x, color='black', ax=ax)
    Gamma.from_k_theta(k=7.5, theta=1).cdf().plot(x=x, color='blue', ax=ax)
    Gamma.from_k_theta(k=0.5, theta=1).cdf().plot(x=x, color='purple', ax=ax)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')
    plt.show()


def plot_increasing_alpha_beta():

    ax = new_axes()
    x = arange(0, 1.001, 0.001)
    Gamma(alpha=4, beta=10).pdf().plot(x=x, color='red', ax=ax)
    Gamma(alpha=8, beta=20).pdf().plot(x=x, color='red', ax=ax)
    Gamma(alpha=16, beta=40).pdf().plot(x=x, color='orange', ax=ax)
    Gamma(alpha=32, beta=80).pdf().plot(x=x, color='yellow', ax=ax)
    Gamma(alpha=64, beta=160).pdf().plot(x=x, color='green', ax=ax)
    Gamma(alpha=128, beta=320).pdf().plot(x=x, color='black', ax=ax)
    Gamma(alpha=256, beta=640).pdf().plot(x=x, color='blue', ax=ax)
    Gamma(alpha=512, beta=1280).pdf().plot(x=x, color='purple', ax=ax)
    ax.set_xlim(0, 1)
    ax.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':

    # plot_wikipedia_pdfs()
    # plot_wikipedia_cdfs()
    plot_increasing_alpha_beta()
