import matplotlib.pyplot as plt

from probability.distributions.discrete.hyper_geometric import HyperGeometric
from probability.plotting import new_axes

k = list(range(1, 61))


def plot_wikipedia_pmfs():
    """
    https://en.wikipedia.org/wiki/Hypergeometric_distribution#/media/
    File:HypergeometricPDF.png
    """
    ax = new_axes(width=12, height=9)
    HyperGeometric(N=500, K=50, n=100).plot(
        k=k, kind='line', color='blue',
        ax=ax, marker='o', mfc='blue'
    )
    HyperGeometric(N=500, K=60, n=200).plot(
        k=k, kind='line', color='green',
        ax=ax, marker='o', mfc='green'
    )
    HyperGeometric(N=500, K=70, n=300).plot(
        k=k, kind='line', color='red',
        ax=ax, marker='o', mfc='red'
    )
    ax.set_xlabel('k')
    ax.set_ylabel('P(X=k)')
    ax.set_title('Probability mass function')
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 0.15)
    ax.legend(loc='upper right')
    plt.show()


def plot_wikipedia_cdfs():
    """
    https://en.wikipedia.org/wiki/Hypergeometric_distribution#/media/
    File:HypergeometricCDF.png
    """
    ax = new_axes(width=12, height=9)
    HyperGeometric(N=500, K=50, n=100).cdf().plot(
        k=k, kind='line', color='blue',
        ax=ax, marker='o', mfc='blue'
    )
    HyperGeometric(N=500, K=60, n=200).cdf().plot(
        k=k, kind='line', color='green',
        ax=ax, marker='o', mfc='green'
    )
    HyperGeometric(N=500, K=70, n=300).cdf().plot(
        k=k, kind='line', color='red',
        ax=ax, marker='o', mfc='red'
    )
    ax.set_xlabel('x')
    ax.set_ylabel('P(X≤x)')
    ax.set_title('Cumulative distribution function')
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':

    plot_wikipedia_pmfs()
    plot_wikipedia_cdfs()
