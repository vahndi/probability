import matplotlib.pyplot as plt

from probability.distributions.conjugate.beta_geometric_conjugate import \
    BetaGeometricConjugate


def plot_example():

    bg = BetaGeometricConjugate(alpha=5, beta=10, n=5, k=30)
    bg.plot()


if __name__ == '__main__':

    plot_example()
    plt.show()
