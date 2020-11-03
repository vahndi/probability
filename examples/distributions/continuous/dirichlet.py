import matplotlib.pyplot as plt
from mpl_format.axes.axis_utils import new_axes
from numpy import linspace

from probability.distributions import Dirichlet

x = linspace(0, 1, 101)


def plot_pdf_simplex():

    ax = new_axes(width=10, height=10)
    Dirichlet(alpha=[1, 2, 3]).pdf().plot_simplex(ax=ax)
    plt.show()


def plot_pdf_margins():

    ax = new_axes(width=10, height=10)
    Dirichlet(alpha=[1, 2, 3]).pdf().plot(ax=ax)
    plt.show()


if __name__ == '__main__':

    # plot_pdf_simplex()
    plot_pdf_margins()
