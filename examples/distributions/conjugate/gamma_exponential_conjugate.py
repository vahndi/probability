import matplotlib.pyplot as plt

from probability.distributions.conjugate.gamma_exponential_conjugate import \
    GammaExponentialConjugate


def plot_example():

    ge = GammaExponentialConjugate(x_mean=0.3, n=100, alpha=10, beta=5)
    ge.plot()
    plt.show()


if __name__ == '__main__':

    plot_example()
