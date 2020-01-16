import matplotlib.pyplot as plt
from numpy import arange
from scipy import randn
from scipy.stats import norm

from probability.distributions.conjugate.inv_gamma_normal import InvGammaNormal
from probability.plots import new_axes

x = arange(0, 20.01, 0.01)
mu, sigma = 2, 3
x_i = mu + sigma * randn(1000)
dist = InvGammaNormal(alpha=1, beta=1, x=x_i, mu=2)


def plot_parameters():

    ax = new_axes()
    dist.prior().plot(x=x, color='r', ax=ax)
    dist.posterior().plot(x=x, color='g', ax=ax)
    ax.legend()
    plt.show()


def plot_predictions():

    ax = new_axes()
    predicted = dist.rvs(100000)
    ax.hist(predicted, bins=100, density=True, label='PPD samples')
    x_actual = arange(predicted.min(), predicted.max(), 0.01)
    actual = norm(loc=mu, scale=sigma).pdf(x_actual)
    ax.plot(x_actual, actual, label='True Distribution')
    ax.legend()
    plt.show()


if __name__ == '__main__':

    plot_parameters()
    plot_predictions()
