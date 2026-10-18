import matplotlib.pyplot as plt

from probability.distributions.discrete.geometric import Geometric
from probability.plotting import new_axes

k = list(range(1, 11))


def plot_wikipedia_pmfs():
    """
    https://en.wikipedia.org/wiki/Geometric_distribution#/media/
    File:Geometric_pmf.svg
    """
    ax = new_axes(width=12, height=9)
    Geometric(p=0.2).plot(k=k, kind='line', color='gray',
                          ax=ax, marker='o', mfc='orange')
    Geometric(p=0.5).plot(k=k, kind='line', color='gray',
                          ax=ax, marker='o', mfc='purple')
    Geometric(p=0.8).plot(k=k, kind='line', color='gray',
                          ax=ax, marker='o', mfc='lightblue')
    ax.set_xlabel('x')
    ax.set_ylabel('P(X=x)')
    ax.set_title('Probability mass function')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right')
    plt.show()


def plot_wikipedia_cdfs():
    """
    https://en.wikipedia.org/wiki/Geometric_distribution#/media/
    File:Geometric_cdf.svg
    """
    ax = new_axes(width=12, height=9)
    Geometric(p=0.2).cdf().plot(k=k, kind='line', color='gray',
                                ax=ax, marker='o', mfc='orange')
    Geometric(p=0.5).cdf().plot(k=k, kind='line', color='gray',
                                ax=ax, marker='o', mfc='purple')
    Geometric(p=0.8).cdf().plot(k=k, kind='line', color='gray',
                                ax=ax, marker='o', mfc='lightblue')
    ax.set_xlabel('x')
    ax.set_ylabel('P(X≤x)')
    ax.set_title('Cumulative distribution function')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':

    plot_wikipedia_pmfs()
    plot_wikipedia_cdfs()
