import matplotlib.pyplot as plt
from math import sqrt
from numpy import arange

from examples.colors import ML_APP_DARK_BLUE
from probability.distributions import Normal, Laplace
from probability.distributions.continuous.students_t import StudentsT


x = arange(-4, 4.01, 0.05)


def plot_normal_students_t_laplace():
    """
    Machine Learning: A Probabilistic Perspective. Figure 2.7
    """
    _, axes = plt.subplots(ncols=2, figsize=(16, 9))
    # define distributions
    normal = Normal(mu=0, sigma=1)
    students_t = StudentsT(nu=1)
    laplace = Laplace(mu=0, b=1 / sqrt(2))
    # plot pdfs
    ax = axes[0]
    normal.plot(x=x, ls=':', color='black', ax=ax)
    students_t.plot(x=x, ls='--', color=ML_APP_DARK_BLUE, ax=ax)
    laplace.plot(x=x, ls='-', color='red', ax=ax)
    ax.set_ylim(0, 0.8)
    ax.legend(loc='upper right')
    # plot log-pdfs
    ax = axes[1]
    normal.log_pdf().plot(x=x, ls=':', color='black', ax=ax)
    students_t.log_pdf().plot(x=x, ls='--', color=ML_APP_DARK_BLUE, ax=ax)
    laplace.log_pdf().plot(x=x, ls='-', color='red', ax=ax)
    ax.set_ylim(-9, 0)
    ax.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':

    plot_normal_students_t_laplace()
