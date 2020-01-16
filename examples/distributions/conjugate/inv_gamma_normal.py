import matplotlib.pyplot as plt
from numpy import arange
from scipy import randn

from probability.distributions.conjugate.inv_gamma_normal import InvGammaNormal
from probability.plots import new_axes


def plot_inv_gamma_normal():

    x = arange(0, 20.01, 0.01)
    mu, sigma = 2, 3
    x_i = mu + sigma * randn(100)
    ign = InvGammaNormal(alpha=1, beta=1, x=x_i, mu=2)

    ax = new_axes()
    ign.prior().plot(x=x, color='r', ax=ax)
    ign.posterior().plot(x=x, color='g', ax=ax)
    ax.legend()

    plt.show()

