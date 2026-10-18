import matplotlib.pyplot as plt

from probability.distributions.discrete.beta_binomial import BetaBinomial
from probability.plotting import new_axes

k_wiki = list(range(11))
alpha_wiki = [0.2, 0.7, 2, 600]
beta_wiki = [0.25, 2, 2, 400]
colors_wiki = ['black', 'red', 'green', 'blue']


def plot_wikipedia_pmfs():
    """
    https://en.wikipedia.org/wiki/Beta-binomial_distribution#/media/
    File:Beta-binomial_distribution_pmf.png
    """
    ax = new_axes(width=12, height=9)
    for alpha, beta, color in zip(alpha_wiki, beta_wiki, colors_wiki):
        BetaBinomial(n=10, alpha=alpha, beta=beta).plot(
            k=k_wiki, color=color, kind='line', ax=ax,
            marker='o'
        )
    ax.legend()
    plt.show()


def plot_wikipedia_cdfs():
    """
    https://en.wikipedia.org/wiki/File:Beta-binomial_cdf.png
    """
    ax = new_axes(width=12, height=9)
    for alpha, beta, color in zip(alpha_wiki, beta_wiki, colors_wiki):
        BetaBinomial(n=10, alpha=alpha, beta=beta).cdf().plot(
            k=k_wiki, color=color, kind='line', ax=ax,
            marker='o'
        )
    ax.legend()
    plt.show()


def plot_scipy_pmf():
    """
    https://docs.scipy.org/doc/scipy/reference/generated/
    scipy.stats.betabinom.html#scipy.stats.betabinom
    """
    bb = BetaBinomial(n=5, alpha=2.3, beta=0.63)
    ax = new_axes()
    bb.pmf().plot(k=range(0, 5), kind='line', ls='',
                  marker='o', color='blue', vlines=True, ax=ax)
    ax.set_xlim(-0.5, 4.5)
    plt.show()


if __name__ == '__main__':

    plot_wikipedia_pmfs()
    plot_wikipedia_cdfs()
    plot_scipy_pmf()
