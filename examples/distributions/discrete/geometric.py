import matplotlib.pyplot as plt
from mpl_format.axes.axes_formatter import AxesFormatter

from probability.distributions.discrete.geometric import Geometric

k = list(range(1, 11))


def plot_wikipedia_pmfs():
    """
    https://en.wikipedia.org/wiki/Geometric_distribution#/media/File:Geometric_pmf.svg
    """
    axf = AxesFormatter(width=12, height=9)
    Geometric(p=0.2).plot(k=k, kind='line', color='gray',
                          ax=axf.axes, marker='o', mfc='orange')
    Geometric(p=0.5).plot(k=k, kind='line', color='gray',
                          ax=axf.axes, marker='o', mfc='purple')
    Geometric(p=0.8).plot(k=k, kind='line', color='gray',
                          ax=axf.axes, marker='o', mfc='lightblue')
    axf.set_text(
        x_label='x', y_label='P(X=x)', title='Probability mass function'
    ).set_x_lim(0, 10).set_y_lim(0, 1)
    axf.axes.legend(loc='upper right')
    plt.show()


def plot_wikipedia_cdfs():
    """
    https://en.wikipedia.org/wiki/Geometric_distribution#/media/File:Geometric_cdf.svg
    """
    axf = AxesFormatter(width=12, height=9)
    Geometric(p=0.2).cdf().plot(k=k, kind='line', color='gray',
                                ax=axf.axes, marker='o', mfc='orange')
    Geometric(p=0.5).cdf().plot(k=k, kind='line', color='gray',
                                ax=axf.axes, marker='o', mfc='purple')
    Geometric(p=0.8).cdf().plot(k=k, kind='line', color='gray',
                                ax=axf.axes, marker='o', mfc='lightblue')
    axf.set_text(
        x_label='x', y_label='P(X≤x)', title='Cumulative distribution function'
    ).set_x_lim(0, 10).set_y_lim(0, 1)
    axf.axes.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':

    plot_wikipedia_pmfs()
    plot_wikipedia_cdfs()
