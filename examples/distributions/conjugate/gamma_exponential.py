import matplotlib.pyplot as plt
from numpy import arange, linspace
from probability.distributions.conjugate.gamma_exponential import GammaExponential
from probability.plots import new_axes


rates = arange(0, 5, 0.01)
durations = linspace(0.1, 10, 1000)


def plot_gamma_exponential():

    ge = GammaExponential(alpha=10, beta=5, n=100, x_mean=0.3)
    ax = new_axes()
    ge.prior().plot(x=rates, color='r', ax=ax)
    ge.posterior().plot(x=rates, color='g', ax=ax)
    ax.legend()
    plt.show()
    ax = new_axes()
    ge.pdf().plot(x=durations, ax=ax)
    ax.legend()
    plt.show()


if __name__ == '__main__':

    plot_gamma_exponential()
