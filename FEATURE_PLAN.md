# `probability` — Scope, Boundary & Overarching Feature Plan

## 1. Vision & Architectural Boundary

`probability` is a library for **immediate, interactive, object-oriented probabilistic inference, distribution algebra, causal analysis, and decision modeling** in Python.

### Core Design Principles
1. **First-Class Algebraic Objects**: Distributions, empirical data columns, conditional tables, and conjugate models are self-contained objects supporting Python operators (`+`, `-`, `*`, `/`, `<`, `>`, `|`, `&`, `~`) directly.
2. **Instantaneous Execution**: Inference, comparison, and conditioning execute in milliseconds via exact analytical formulas, conjugate updates, low-dimensional quadrature/grid methods, or direct vectorized/Numba-accelerated evaluation—without model compilation or Markov chain convergence diagnostics ($\hat{R}$, ESS).
3. **Lightweight Scientific Stack**: Built strictly on `numpy`, `scipy`, `pandas`, `numba`, `networkx`, `matplotlib`, and `seaborn`—no symbolic tensor backend (`pytensor`, `jax`, `torch`).

### Boundary Between `probability` and PyMC
| Dimension | `probability` (In Scope) | PyMC / PPLs (Out of Scope) |
| :--- | :--- | :--- |
| **Inference Engine** | Closed-form algebra, conjugacy, Empirical Bayes, 1D/2D grid & Laplace approximations, exact discrete factor graphs, bootstrap/permutation tests | Gradient-based MCMC (HMC / NUTS), Variational Inference (ADVI), SMC on high-dimensional graphs |
| **Model Representation** | Direct Python objects & operators (`Normal(0, 1) + Normal(2, 3)`, `Y \| T`) | Symbolic DAGs inside `with pm.Model():` compiled via PyTensor/JAX |
| **Dimensionality** | 1D–3D parameter spaces, arbitrary-width tabular data, conjugate linear models, discrete graphical models | Arbitrary multi-level non-conjugate GLMs, crossed random effects, $d \gg 3$ latent spaces |
| **Interoperability** | Design/elicitate priors and analyze/compare posterior sample arrays | Sample complex non-conjugate hierarchical posteriors |

> **Rule of Thumb**: If a problem can be solved via exact distribution algebra, conjugacy, empirical Bayes, low-dimensional grid/Laplace approximation, exact discrete graphical models, or direct Monte Carlo/bootstrap on data distributions, it belongs in `probability`. Once a model requires differentiating a log-joint density or running Markov chains over a high-dimensional non-conjugate parameter space, hand off to PyMC.

---

## 2. Overarching Feature Plan

The roadmap is organized into **six phases**, progressing from completing existing foundations to higher-level modeling and interoperability.

---

### Phase 1: Complete the Conjugate & Analytical Distribution Ecosystem
Finish and unify all standard univariate and multivariate conjugate families so every textbook conjugate model works out of the box with `.prior()`, `.likelihood()`, `.posterior()`, `.prior_predictive()`, `.posterior_predictive()`, and `.plot()`.

- [x] **1.1 Normal-Precision & Normal-Variance Conjugates**
  - Finalize and export [`GammaNormalConjugate`](probability/distributions/conjugate/gamma_normal_conjugate.py) (known $\mu$, unknown precision $\tau \sim \text{Gamma}(\alpha, \beta)$).
  - Finalize and export [`InverseGammaNormalConjugate`](probability/distributions/conjugate/inverse_gamma_normal_conjugate.py) (known $\mu$, unknown variance $\sigma^2 \sim \text{InvGamma}(\alpha, \beta)$) with Student-$t$ predictive distributions.
- [x] **1.2 Joint Unknown Mean & Variance (`NormalInverseGammaConjugate`)**
  - Implement the 2-parameter conjugate Normal-Inverse-Gamma model for unknown $(\mu, \sigma^2)$ with:
    - Marginal posterior $\mu \mid D \sim \text{StudentT}(\nu_n, \mu_n, \sigma_n^2)$
    - Marginal posterior $\sigma^2 \mid D \sim \text{InverseGamma}(\alpha_n, \beta_n)$
    - Posterior predictive $\tilde{x} \mid D \sim \text{StudentT}(\nu_n, \mu_n, \sigma_n^2 (1 + 1/n))$
    - `.infer_posterior(data: Series | Ratio)` classmethod for one-line Bayesian inference on continuous data.
- [x] **1.3 Multivariate Normal-Inverse-Wishart (`MVNormalInverseWishartConjugate`)**
  - Conjugate prior/posterior for multivariate Normal data with unknown mean vector $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$, yielding multivariate Student-$t$ marginal and predictive distributions.
- [ ] **1.4 Additional Univariate Parametric Distributions**
  - `LogNormal`, `Weibull`, `Cauchy`, `Pareto`, `Uniform` (discrete & continuous parity), and `NegativeBinomial` (with `BetaNegativeBinomialConjugate`).
  - Moment-matching & MLE `.fit(data)` and `.from_mean_and_std()` constructors across all continuous distributions.

---

### Phase 2: Low-Dimensional Non-Conjugate Bayesian Inference (Grid & Laplace)
Support *Think Bayes* and *Statistical Rethinking* (Chapters 2–4) workflows where the prior and likelihood are not conjugate, but parameter dimension is $d \in \{1, 2\}$.

- [x] **2.1 1D Grid Approximation (`GridDistribution1d` / `BayesianGrid1d`)**
  - Initialize from any continuous/discrete prior or custom prior array over a 1D support grid.
  - `.update(likelihood_fn, data)` for sequential Bayesian updating.
  - Exact `.mean()`, `.std()`, `.median()`, `.mode()`, `.quantile()`, `.hdi()`, `. credible_interval()`, `.sample()`, and `.plot()` (returning `Axes`).
  - Support arithmetic (`+`, `-`, `*`, `/`) and inequality comparisons (`<`, `>`) between two `GridDistribution1d` objects.
- [x] **2.2 2D Grid Approximation (`GridDistribution2d`)**
  - Joint 2D posterior grid (e.g., $(\mu, \sigma)$ or $(\alpha, \beta)$ in logistic/binomial dose-response models).
  - `.marginal_x()` and `.marginal_y()` returning `GridDistribution1d`.
  - `.plot_contour()` and `.plot_heatmap()` on native Matplotlib `Axes`.
- [x] **2.3 Quadratic / Laplace Approximation (`LaplaceApproximation`)**
  - Given a log-posterior function (or prior + likelihood specification), find the MAP estimate $\hat{\theta}$ via `scipy.optimize` and compute the inverse Hessian $\mathbf{H}^{-1}$ via finite differences.
  - Return a `Normal` (1D) or `MultivariateNormal` ($d$-D) posterior object with full access to `.hdi()`, `.summary()`, and `.sample()`.

---

### Phase 3: Empirical Bayes & Hierarchical Shrinkage on Series
Bring partial pooling to `BooleanSeries`, `CountSeries`, and `RatioSeries` without requiring MCMC.

- [x] **3.1 Beta-Binomial Empirical Bayes on `BooleanSeries`**
  - `BooleanSeries.empirical_bayes_prior() -> Beta`: Fit hyperparameter $(\hat{\alpha}, \hat{\beta})$ across categories via method-of-moments or marginal MLE (Beta-Binomial marginal likelihood).
  - `BooleanSeries.shrunk_posteriors() -> BetaSeries`: Return shrunk posterior `Beta` distributions for each category, automatically pulling small-sample categories toward the grand mean.
  - `BooleanSeries.plot_shrinkage(ax=None)`: Visualize raw category means vs. Empirical Bayes shrunk posterior means and HDIs.
- [x] **3.2 Gamma-Poisson Empirical Bayes on `CountSeries`**
  - Fit a shared `Gamma(alpha, beta)` prior across count groups and compute shrunk `Gamma` posteriors and `NegativeBinomial` predictive distributions per group.
- [x] **3.3 Normal-Normal Empirical Bayes (James-Stein / Random Effects) on `RatioSeries`**
  - Estimate between-group variance $\tau_0^2$ and within-group variance $\sigma_i^2$, returning shrunk `Normal` posteriors for each group's mean.

---

### Phase 4: Closed-Form Bayesian & Causal Regression Models
Support *Causal Inference in Python* (Chapters 03+) and applied regression workflows using closed-form linear algebra.

- [x] **4.1 Analytical Bayesian Linear Regression (`BayesianLinearRegression`)**
  - Conjugate Normal-Inverse-Gamma linear model $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$:
    - Closed-form multivariate Student-$t$ / Normal marginal posteriors for each coefficient $\beta_j$ (returned as `StudentsT` or `Normal` distribution objects!).
    - Exact analytical `.posterior_predictive(X_new)` returning `StudentsT` distributions.
    - Zellner's $g$-prior and Ridge ($L_2$) prior presets.
    - `.plot_coefficients(hdi=0.95, ax=None)` forest plot of posterior coefficient distributions.
- [x] **4.2 OLS & Causal Adjustment Estimators on Data Distributions**
  - Frisch-Waugh-Lovell (FWL) partialling-out / residualization helper on `Ratio` (`y.residualize(X) -> Ratio`) so users can inspect adjusted distributions and plot adjusted density bars directly.
  - Heteroskedasticity-robust (HC1/HC2/HC3) and cluster-robust standard errors producing `StudentsT` sampling distributions for treatment effects.
- [x] **4.3 Propensity Score, IPW & Doubly-Robust Estimators**
  - Inverse Probability Weighting (IPW) and Augmented IPW (Doubly Robust ATE/ATT) with bootstrap confidence intervals and balance diagnostics (`love_plot(ax=None)` building on `.normalized_diff()`).

---

### Phase 5: Exact Discrete Graphical Models & Decision Analysis
Expand the *Bayesian Reasoning and Machine Learning* (BRML) and decision-modeling capabilities where exact discrete algebra shines.

- [ ] **5.1 Discrete Bayesian Network (`BayesianNetwork`)**
  - Construct a DAG of discrete `Discrete` and `Conditional` tables.
  - Exact marginal and conditional queries $P(X \mid E = e)$ via variable elimination (building on `prob_sum_product`).
  - D-separation and conditional independence queries (`net.is_independent(X, Y, given=Z)`).
  - `.draw(ax=None)` using `networkx` and `probability.plotting`.
- [ ] **5.2 Markov Chains & Hidden Markov Models (`MarkovChain`, `HMM`)**
  - Stationary distributions, $n$-step transition probabilities, absorption times, forward-backward filtering, and Viterbi decoding.
- [ ] **5.3 Continuous Uncertainty in `DecisionTree` and `ProjectNetwork`**
  - Support distribution-valued payoffs/costs in `DecisionTree` (computing expected utility and risk profiles / Value of Perfect Information EVPI).
  - Critical-path probability and Monte Carlo completion-time distribution (`Ratio`) in `ProjectNetwork`.

---

### Phase 6: Prior Elicitation & PyMC / ArviZ Bridge
Make `probability` the best companion library for users who *do* drop into PyMC for complex hierarchical models.

- [ ] **6.1 Prior Elicitation From Quantiles (`.from_interval()` / `.from_quantiles()`)**
  - Construct any 2-parameter distribution (`Normal`, `Beta`, `Gamma`, `InverseGamma`, `LogNormal`) by specifying desired quantile bounds, e.g. `Gamma.from_interval(lower=2.0, upper=10.0, prob=0.95)`.
- [ ] **6.2 Posterior Sample Wrapping (`EmpiricalPosterior` / `Ratio` Enhancements)**
  - `Ratio.from_samples(samples)` / ArviZ `InferenceData` converter that extracts 1D parameter draws into `Ratio` (or `RatioSeries`), enabling `prob_A > prob_B`, `.hdi(0.95)`, ROPE (Region of Practical Equivalence) probability `.prob_in_rope(low, high)`, and `.plot_kde()` on MCMC traces.
- [ ] **6.3 Optional PyMC Export Helper (`.to_pymc(name)`)**
  - Lazy/optional helper (requiring `pymc` only if called) that converts a `probability` distribution (`Beta(2, 5)`, `Normal(0, 1)`, `Gamma(3, 2)`) into the corresponding `pm.Distribution(name, ...)` inside an active `pm.Model` context.

---

## 3. Recommended Implementation Order

1. **Phase 1.1–1.2**: `GammaNormalConjugate`, `InverseGammaNormalConjugate`, and `NormalInverseGammaConjugate` (completes the conjugate module and unlocks exact Bayesian t-tests/regression).
2. **Phase 4.1–4.2**: `BayesianLinearRegression` and FWL residualization (`Ratio.residualize`) (unlocks *Causal Inference in Python* Chapters 03–05 notebooks).
3. **Phase 3.1–3.3**: Empirical Bayes shrinkage on `BooleanSeries`, `CountSeries`, and `RatioSeries`.
4. **Phase 2.1–2.3**: 1D/2D Grid and Laplace approximations (unlocks *Think Bayes* Chapters 02+ and *Statistical Rethinking* notebooks).
5. **Phase 6.1–6.3**: Quantile-based prior elicitation (`.from_interval()`) and MCMC sample/ROPE helpers.
6. **Phase 5.1–5.3**: `BayesianNetwork` wrapper, `MarkovChain`, and EVPI in `DecisionTree`.
