import matplotlib.pyplot as plt
from mpl_format.axes.axes_formatter import AxesFormatter

from probability.distributions.discrete.hyper_geometric import HyperGeometric

k = list(range(1, 61))


def plot_wikipedia_pmfs():
    """
    https://en.wikipedia.org/wiki/Hypergeometric_distribution#/media/File:HypergeometricPDF.png
    """
    axf = AxesFormatter(width=12, height=9)
    HyperGeometric(N=500, K=50, n=100).plot(
        k=k, kind='line', color='blue',
        ax=axf.axes, marker='o', mfc='blue'
    )
    HyperGeometric(N=500, K=60, n=200).plot(
        k=k, kind='line', color='green',
        ax=axf.axes, marker='o', mfc='green'
    )
    HyperGeometric(N=500, K=70, n=300).plot(
        k=k, kind='line', color='red',
        ax=axf.axes, marker='o', mfc='red'
    )
    axf.set_text(
        x_label='k', y_label='P(X=k)', title='Probability mass function'
    ).set_x_lim(0, 60).set_y_lim(0, 0.15)
    axf.axes.legend(loc='upper right')
    plt.show()


def plot_wikipedia_cdfs():
    """
    https://en.wikipedia.org/wiki/Hypergeometric_distribution#/media/File:HypergeometricCDF.png
    """
    axf = AxesFormatter(width=12, height=9)
    HyperGeometric(N=500, K=50, n=100).cdf().plot(
        k=k, kind='line', color='blue',
        ax=axf.axes, marker='o', mfc='blue'
    )
    HyperGeometric(N=500, K=60, n=200).cdf().plot(
        k=k, kind='line', color='green',
        ax=axf.axes, marker='o', mfc='green'
    )
    HyperGeometric(N=500, K=70, n=300).cdf().plot(
        k=k, kind='line', color='red',
        ax=axf.axes, marker='o', mfc='red'
    )
    axf.set_text(
        x_label='x', y_label='P(X≤x)', title='Cumulative distribution function'
    ).set_x_lim(0, 60).set_y_lim(0, 1)
    axf.axes.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':

    plot_wikipedia_pmfs()
    plot_wikipedia_cdfs()
