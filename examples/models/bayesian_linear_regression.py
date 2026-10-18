import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from probability.models import BayesianLinearRegression

# Simulate data: wage ~ 10.0 + 2.5 * education + 1.2 * experience + noise
np.random.seed(42)
n = 150
education = np.random.normal(12, 2, n)
experience = np.random.normal(10, 4, n)
wage = 10.0 + 2.5 * education + 1.2 * experience + np.random.normal(0, 3.0, n)

df = pd.DataFrame({
    'education': education,
    'experience': experience,
    'wage': wage
})

# 1. Fit the Bayesian model
model = BayesianLinearRegression.fit(
    X=df[['education', 'experience']],
    y=df['wage'],
    fit_intercept=True
)


def print_summary():
    print("=" * 60)
    print("Bayesian Linear Regression Summary:")
    print("=" * 60)
    print(model.summary())
    print("\nIndividual Marginal Posteriors:")
    beta_edu = model.coef('education')
    print(f"  beta[education] ~ {beta_edu}")
    print(f"  95% HDI for education: {beta_edu.hdi(0.95)}")
    print(f"  P(beta[education] > 0) = {model.prob_positive('education'):.4f}")
    print(f"  Residual variance sigma_sq ~ {model.sigma_sq}")


def plot_results():
    # 2. Forest plot of coefficients
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    model.plot_coefficients(ax=axes[0], exclude_intercept=True)

    # 3. Fit plot along education dimension
    model.plot_fit(x_var='education', ax=axes[1])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print_summary()
    plot_results()
