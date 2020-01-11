import matplotlib.pyplot as plt
from probability.distributions.multivariate.multinomial import Multinomial


def plot_2d():

    mn2d = Multinomial(n=10, p=[0.3, 0.7])
    mn2d.pmf().plot(mn2d.permutations())
    plt.show()


def plot_3d():

    mn3d = Multinomial(n=10, p=[0.3, 0.5, 0.2])
    mn3d.pmf().plot(mn3d.permutations())
    plt.show()


def plot_4d():

    mn4d = Multinomial(n=10, p=[0.1, 0.2, 0.3, 0.4])
    mn4d.pmf().plot(mn4d.permutations())
    plt.show()


if __name__ == '__main__':

    plot_2d()
    plot_3d()
    plot_4d()
