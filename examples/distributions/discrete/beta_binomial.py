from mpl_format.axes.axes_formatter import AxesFormatter

from probability.distributions.discrete.beta_binomial import BetaBinomial

k_wiki = list(range(11))


def plot_wikipedia_pmfs():
    """
    https://en.wikipedia.org/wiki/Beta-binomial_distribution#/media/File:Beta-binomial_distribution_pmf.png
    """
    alphas = [0.2, 0.7, 2, 600]
    betas = [0.25, 2, 2, 400]
    colors = ['black', 'red', 'green', 'blue']
    axf = AxesFormatter(width=12, height=9)
    for alpha, beta, color in zip(alphas, betas, colors):
        BetaBinomial(n=10, alpha=alpha, beta=beta).plot(
            k=k_wiki, color=color, kind='line', ax=axf.axes,
            marker='o'
        )
    axf.add_legend()
    axf.show()


def plot_scipy_pmf():
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html#scipy.stats.betabinom
    """
    bb = BetaBinomial(n=5, alpha=2.3, beta=0.63)
    axf = AxesFormatter()
    bb.pmf().plot(k=range(0, 5), kind='line', ls='',
                  marker='o', color='blue', vlines=True, ax=axf.axes)
    axf.set_x_lim(-0.5, 4.5)
    axf.show()


if __name__ == '__main__':

    plot_wikipedia_pmfs()
    plot_scipy_pmf()

