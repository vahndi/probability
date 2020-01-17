import matplotlib.pyplot as plt
from numpy import arange
from scipy import randn
from scipy.stats import norm

from probability.distributions.conjugate.normal_normal import NormalNormal
from probability.plots import new_axes

x = arange(0, 20.01, 0.01)
mu, sigma = 2, 3
x_i = mu + sigma * randn(1000)
dist_1 = NormalNormal(mu_0=1.5, sigma_sq_0=8, sigma_sq=9, x=x_i)
dist_2 = NormalNormal(mu_0=1.5, tau_0=1/8, tau=1/9, x=x_i)


def plot_parameters(dist: NormalNormal):

    ax = new_axes()
    dist.prior().plot(x=x, color='r', ax=ax)
    dist.posterior().plot(x=x, color='g', ax=ax)
    ax.legend()
    plt.show()


def plot_predictions(dist: NormalNormal):

    ax = new_axes()
    predicted = dist.rvs(100000)
    ax.hist(predicted, bins=100, density=True, label='PPD samples')
    x_actual = arange(predicted.min(), predicted.max(), 0.01)
    actual = norm(loc=mu, scale=sigma).pdf(x_actual)
    ax.plot(x_actual, actual, label='True Distribution')
    ax.legend()
    plt.show()


if __name__ == '__main__':

    for dist_num in (dist_1, dist_2):
        plot_parameters(dist_num)
        plot_predictions(dist_num)
