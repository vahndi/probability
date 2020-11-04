import matplotlib.pyplot as plt

from probability.distributions.conjugate.dirichlet_multinomial_conjugate import \
    DirichletMultinomialConjugate


def plot_example():

    dm = DirichletMultinomialConjugate(alpha=[1, 2, 3], x=[20, 15, 10])
    dm.plot()
    plt.show()


if __name__ == '__main__':

    plot_example()
