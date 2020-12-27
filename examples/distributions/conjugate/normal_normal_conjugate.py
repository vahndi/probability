import matplotlib.pyplot as plt
from numpy import arange
from pandas import Series
from scipy import randn

from probability.distributions.conjugate.normal_normal_conjugate import \
    NormalNormalConjugate

x = arange(0, 20.01, 0.01)
mu, sigma = 2, 3
x_i = Series(mu + sigma * randn(1000))
dist_1 = NormalNormalConjugate(
    n=len(x_i), x_sum=x_i.sum(),
    mu_0=1.5, sigma_0_sq=8, sigma_sq=sigma ** 2
)
dist_2 = NormalNormalConjugate(
    n=len(x_i), x_sum=x_i.sum(),
    mu_0=1.5, tau_0=1 / 8, tau=1 / (sigma ** 2)

)


def plot_examples():

    dist_1.plot()
    plt.show()
    dist_2.plot()
    plt.show()


if __name__ == '__main__':

    plot_examples()
