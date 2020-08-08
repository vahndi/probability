import matplotlib.pyplot as plt
from numpy import arange, inf

from probability.distributions import Normal
from probability.distributions.continuous.students_t import StudentsT
from mpl_format.axes.axis_utils import new_axes

x_wikipedia_1 = arange(-5, 5.01, 0.05)
x_wikipedia_2 = arange(-4, 4.01, 0.05)


def plot_wikipedia_pdfs():
    """
    https://en.wikipedia.org/wiki/Student%27s_t-distribution#/media/File:Student_t_pdf.svg
    """
    ax = new_axes()
    StudentsT(nu=1).plot(x=x_wikipedia_1, color='orange', ax=ax)
    StudentsT(nu=2).plot(x=x_wikipedia_1, color='purple', ax=ax)
    StudentsT(nu=5).plot(x=x_wikipedia_1, color='lightblue', ax=ax)
    StudentsT(nu=1e9).plot(x=x_wikipedia_1, color='black', ax=ax)
    ax.legend(loc='upper right')
    plt.show()


def plot_wikipedia_cdfs():
    """
    https://en.wikipedia.org/wiki/Student%27s_t-distribution#/media/File:Student_t_cdf.svg
    """
    ax = new_axes()
    StudentsT(nu=1).cdf().plot(x=x_wikipedia_1, color='orange', ax=ax)
    StudentsT(nu=2).cdf().plot(x=x_wikipedia_1, color='purple', ax=ax)
    StudentsT(nu=5).cdf().plot(x=x_wikipedia_1, color='lightblue', ax=ax)
    StudentsT(nu=1e9).cdf().plot(x=x_wikipedia_1, color='black', ax=ax)
    ax.legend(loc='lower right')
    plt.show()


def plot_wikipedia_pdfs_vs_normal():
    """
    https://en.wikipedia.org/wiki/Student%27s_t-distribution#/media/File:T_distribution_1df_enhanced.svg
    https://en.wikipedia.org/wiki/Student%27s_t-distribution#/media/File:T_distribution_2df_enhanced.svg
    https://en.wikipedia.org/wiki/Student%27s_t-distribution#/media/File:T_distribution_3df_enhanced.svg
    https://en.wikipedia.org/wiki/Student%27s_t-distribution#/media/File:T_distribution_5df_enhanced.svg
    https://en.wikipedia.org/wiki/Student%27s_t-distribution#/media/File:T_distribution_10df_enhanced.svg
    https://en.wikipedia.org/wiki/Student%27s_t-distribution#/media/File:T_distribution_30df_enhanced.svg
    """
    _, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 9))
    normal = Normal(mu=0, sigma=1)
    max_degrees = (1, 2, 3, 5, 10, 30)
    for d in range(len(max_degrees)):
        ax = axes.flat[d]
        normal.plot(x=x_wikipedia_2, color='blue', ax=ax)
        for d2 in range(d):
            StudentsT(nu=max_degrees[d2]).plot(x=x_wikipedia_2, color='green',
                                               alpha=0.5 ** (d - d2), ax=ax)
        StudentsT(nu=max_degrees[d]).plot(x=x_wikipedia_2, color='red', ax=ax)
        ax.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':

    plot_wikipedia_pdfs()
    plot_wikipedia_cdfs()
    plot_wikipedia_pdfs_vs_normal()
