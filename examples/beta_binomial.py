import matplotlib.pyplot as plt

from probability.distributions.conjugate.discrete.beta_binomial import BetaBinomial
from probability.plots import new_axes


k = range(11)


def plot_pmfs():

    ax = new_axes(width=10, height=10)
    BetaBinomial(alpha=0.2, beta=0.25, n=10).plot(k=k, color='black', ax=ax)
    BetaBinomial(alpha=0.7, beta=2, n=10).plot(k=k, color='red', ax=ax)
    BetaBinomial(alpha=2, beta=2, n=10).plot(k=k, color='green', ax=ax)
    BetaBinomial(alpha=600, beta=400, n=10).plot(k=k, color='blue', ax=ax)
    ax.legend()
    plt.show()


def plot_cdfs():

    ax = new_axes(width=10, height=10)
    BetaBinomial(alpha=0.2, beta=0.25, n=10).cdf().plot(k=k, color='black', ax=ax)
    BetaBinomial(alpha=0.7, beta=2, n=10).cdf().plot(k=k, color='red', ax=ax)
    BetaBinomial(alpha=2, beta=2, n=10).cdf().plot(k=k, color='green', ax=ax)
    BetaBinomial(alpha=600, beta=400, n=10).cdf().plot(k=k, color='blue', ax=ax)
    ax.legend()
    plt.show()


if __name__ == '__main__':

    plot_pmfs()
    plot_cdfs()

