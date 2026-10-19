"""
PyMC Export Helper for probability distributions.
Provides optional bridge to convert probability distribution objects into PyMC random variables.
PyMC is imported lazily inside `to_pymc` so the core library remains completely dependency-free.
"""
from typing import Any


def to_pymc(distribution: Any, name: str, **kwargs) -> Any:
    """
    Convert a probability distribution instance into the corresponding PyMC random variable.
    Must be called inside an active `with pm.Model():` context.

    :param distribution: A continuous or discrete probability distribution instance.
    :param name: Variable name for the PyMC random variable.
    :param kwargs: Additional keyword arguments forwarded to the PyMC constructor (e.g. `observed=data`, `dims=...`).
    :return: A PyMC random variable instance.
    """
    try:
        import pymc as pm
    except ImportError:
        raise ImportError(
            "PyMC is required to use .to_pymc(). "
            "Please install it via: pip install pymc"
        )

    dist_name = type(distribution).__name__

    # Continuous distributions
    if dist_name == 'Normal':
        return pm.Normal(name, mu=float(distribution.mu), sigma=float(distribution.sigma), **kwargs)

    elif dist_name == 'Beta':
        return pm.Beta(name, alpha=float(distribution.alpha), beta=float(distribution.beta), **kwargs)

    elif dist_name == 'Gamma':
        # In probability, alpha is shape and beta is rate
        return pm.Gamma(name, alpha=float(distribution.alpha), beta=float(distribution.beta), **kwargs)

    elif dist_name == 'InverseGamma':
        return pm.InverseGamma(name, alpha=float(distribution.alpha), beta=float(distribution.beta), **kwargs)

    elif dist_name == 'ContinuousUniform':
        return pm.Uniform(name, lower=float(distribution.a), upper=float(distribution.b), **kwargs)

    elif dist_name == 'LogNormal':
        return pm.LogNormal(name, mu=float(distribution.mu), sigma=float(distribution.sigma), **kwargs)

    elif dist_name == 'Exponential':
        return pm.Exponential(name, lam=float(distribution.lambda_), **kwargs)

    elif dist_name == 'Cauchy':
        return pm.Cauchy(name, alpha=float(distribution.x0), beta=float(distribution.gamma), **kwargs)

    elif dist_name == 'Weibull':
        return pm.Weibull(name, alpha=float(distribution.k), beta=float(distribution.lambda_), **kwargs)

    elif dist_name == 'Pareto':
        return pm.Pareto(name, alpha=float(distribution.alpha), m=float(distribution.x_m), **kwargs)

    elif dist_name in ('StudentsT', 'StudentT'):
        nu = float(getattr(distribution, 'nu', getattr(distribution, 'df', 1.0)))
        mu = float(getattr(distribution, 'mu', getattr(distribution, 'loc', 0.0)))
        sigma = float(getattr(distribution, 'sigma', getattr(distribution, 'scale', 1.0)))
        return pm.StudentT(name, nu=nu, mu=mu, sigma=sigma, **kwargs)

    elif dist_name == 'HalfNormal':
        return pm.HalfNormal(name, sigma=float(distribution.sigma), **kwargs)

    elif dist_name == 'Laplace':
        return pm.Laplace(name, mu=float(distribution.mu), b=float(distribution.b), **kwargs)

    # Discrete distributions
    elif dist_name == 'Binomial':
        return pm.Binomial(name, n=int(distribution.n), p=float(distribution.p), **kwargs)

    elif dist_name == 'Bernoulli':
        return pm.Bernoulli(name, p=float(distribution.p), **kwargs)

    elif dist_name == 'Poisson':
        return pm.Poisson(name, mu=float(distribution.lambda_), **kwargs)

    elif dist_name == 'Geometric':
        return pm.Geometric(name, p=float(distribution.p), **kwargs)

    elif dist_name == 'NegativeBinomial':
        return pm.NegativeBinomial(name, n=float(distribution.r), p=float(distribution.p), **kwargs)

    elif dist_name == 'DiscreteUniform':
        return pm.DiscreteUniform(name, lower=int(distribution.a), upper=int(distribution.b), **kwargs)

    else:
        raise NotImplementedError(
            f"PyMC translation for distribution '{dist_name}' is not currently implemented."
        )
