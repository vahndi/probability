import matplotlib.pyplot as plt

from probability.distributions.conjugate.gamma_poisson_conjugate import \
    GammaPoissonConjugate


def plot_example():

    gp = GammaPoissonConjugate(alpha=5, beta=10, n=120, k=50)
    gp.plot()
    plt.show()


if __name__ == '__main__':

    plot_example()
