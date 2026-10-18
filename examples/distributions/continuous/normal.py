import matplotlib.pyplot as plt
from numpy.ma import arange

from probability.distributions.continuous.normal import Normal
from probability.plotting import new_axes

x = arange(-5, 5.05, 0.05)


def plot_wikipedia_pdfs():
    """
    https://en.wikipedia.org/wiki/Normal_distribution#/media/
    File:Normal_Distribution_PDF.svg
    """
    ax = new_axes()
    Normal(mu=0, sigma_sq=0.2).plot(x=x, color='blue', ax=ax)
    Normal(mu=0, sigma_sq=1.0).plot(x=x, color='red', ax=ax)
    Normal(mu=0, sigma_sq=5.0).plot(x=x, color='orange', ax=ax)
    Normal(mu=-2, sigma_sq=0.5).plot(x=x, color='green', ax=ax)
    ax.legend(loc='upper right')
    plt.show()


def plot_wikipedia_cdfs():
    """
    https://en.wikipedia.org/wiki/Normal_distribution#/media/
    File:Normal_Distribution_CDF.svg
    """
    ax = new_axes()
    Normal(mu=0, sigma_sq=0.2).cdf().plot(x=x, color='blue', ax=ax)
    Normal(mu=0, sigma_sq=1.0).cdf().plot(x=x, color='red', ax=ax)
    Normal(mu=0, sigma_sq=5.0).cdf().plot(x=x, color='orange', ax=ax)
    Normal(mu=-2, sigma_sq=0.5).cdf().plot(x=x, color='green', ax=ax)
    ax.legend(loc='upper left')
    plt.show()


def plot_shaded_intervals():
    """
    Demonstrate shaded Confidence Interval and tail interval plotting.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    dist = Normal(mu=100, sigma_sq=225)  # Normal(100, 15^2)

    # 1. Two-sided 95% Confidence Interval
    dist.plot(ci=0.95, color='tab:blue', ax=axes[0])
    axes[0].set_title(f"95% Equal-Tailed Interval\nCI: {dist.interval(0.95)}")
    axes[0].legend()

    # 2. Lower 5% tail (critical region for one-sided test)
    q05 = dist.ppf().at(0.05)
    dist.plot(fill_interval=(None, q05), color='tab:red', ax=axes[1])
    axes[1].set_title(f"Lower 5% Tail Region\nCritical value: z = {q05:.2f}")
    axes[1].legend()

    # 3. Upper 5% tail
    q95 = dist.ppf().at(0.95)
    dist.plot(fill_interval=(q95, None), color='tab:purple', ax=axes[2])
    axes[2].set_title(f"Upper 5% Tail Region\nCritical value: z = {q95:.2f}")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def demonstrate_linear_combinations():
    """
    Demonstrate exact closed-form linear algebra on Normal distributions:
    Z = a * X + b * Y + c ~ Normal(a*mu_X + b*mu_Y + c, a^2*sigma_X^2 + b^2*sigma_Y^2)
    """
    print("=" * 60)
    print("Normal Distribution Linear Combinations:")
    print("=" * 60)

    X = Normal(mu=10.0, sigma=2.0)
    Y = Normal(mu=25.0, sigma=5.0)

    # Addition
    Z_add = X + Y
    print(f"X: {X}")
    print(f"Y: {Y}")
    print(f"X + Y = {Z_add}")

    # Scaling and subtraction
    Z_comb = 2.0 * X - 0.5 * Y + 10.0
    print(f"2*X - 0.5*Y + 10 = {Z_comb}")
    print(f"P(2*X - 0.5*Y + 10 > 15) = {float(1.0 - Z_comb.cdf().at(15.0)):.4f}")


if __name__ == '__main__':

    demonstrate_linear_combinations()
    plot_wikipedia_pdfs()
    plot_wikipedia_cdfs()
    plot_shaded_intervals()
