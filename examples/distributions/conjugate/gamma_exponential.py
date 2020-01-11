import matplotlib.pyplot as plt

from probability.distributions.conjugate.old.gamma_exponential_old import GammaExponential
from probability.plots import new_axes


def plot_wikipedia():

    ax = new_axes()
    for k, t, c in [
        (1, 2, 'red'),
        (2, 2, 'orange'),
        (3, 2, 'yellow'),
        (5, 1, 'green'),
        (9, 0.5, 'black'),
        (7.5, 1, 'blue'),
        (0.5, 1, 'purple'),
        (0.1, 0.1, 'pink'),
    ]:
        ge = GammaExponential(k=k, theta=t)
        ge.plot_prior(ax=ax, color=c)
    ax.set_ylim(0, 0.5)
    ax.set_title('https://en.wikipedia.org/wiki/Gamma_distribution#/media/File:Gamma_distribution_pdf.svg')
    plt.show()


if __name__ == '__main__':

    plot_wikipedia()
