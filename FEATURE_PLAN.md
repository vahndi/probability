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
- [x] **1.4 Additional Univariate Parametric Distributions**
  - Added [`LogNormal`](probability/distributions/continuous/log_normal.py), [`Weibull`](probability/distributions/continuous/weibull.py), [`Cauchy`](probability/distributions/continuous/cauchy.py), [`Pareto`](probability/distributions/continuous/pareto.py), [`DiscreteUniform`](probability/distributions/discrete/discrete_uniform.py), and [`BetaNegativeBinomialConjugate`](probability/distributions/conjugate/beta_negative_binomial_conjugate.py).
  - Enhanced [`NegativeBinomial`](probability/distributions/discrete/negative_binomial.py) with `.from_mean_and_variance()` and `.fit(data)`.
  - Added `.fit(data)`, `.from_quantiles()`, and `.from_interval()` constructors across continuous distributions.

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

- [x] **5.1 Discrete Bayesian Network (`BayesianNetwork`)**
  - Directed Acyclic Graph (DAG) wrapper in [`probability.discrete.BayesianNetwork`](probability/discrete/bayesian_network.py) constructed from `Discrete` and `Conditional` CPTs.
  - Exact marginal and conditional inference via **Variable Elimination** (`.query(target, evidence=None)`).
  - D-separation and conditional independence testing via `.is_independent(X, Y, given=Z)`.
  - DAG rendering via `.draw(ax=None)` using NetworkX.
  - Full joint distribution calculation via `.joint()`.
  - Comprehensive unit tests in [`tests/test_discrete/test_bayesian_network.py`](tests/test_discrete/test_bayesian_network.py) and BRML Chapter 3 showcase notebook in [`examples/notebooks/brml/chapter_03__belief_networks.ipynb`](examples/notebooks/brml/chapter_03__belief_networks.ipynb) (Alarm Network, Asia / Chest Clinic Network, and explaining away).
- [x] **5.2 Markov Chains & Hidden Markov Models (`MarkovChain`, `HMM`)**
  - Discrete-time Markov Chains in [`probability.discrete.MarkovChain`](probability/discrete/markov_chain.py):
    - Multi-step transition matrices $T^n$ (`.n_step_transition_matrix(n)`).
    - Exact stationary distributions $\pi T = \pi, \sum \pi_i = 1$ (`.stationary_distribution()`).
    - Absorbing chain fundamental matrix $N = (I - Q)^{-1}$, absorption probabilities $B = NR$, and expected times to absorption (`.fundamental_matrix()`, `.absorption_probabilities()`, `.expected_time_to_absorption()`).
    - Trajectory simulation (`.simulate()`) and transition graph plotting (`.draw()`).
    - Temporal Bayesian Network unrolling (`.to_bayesian_network(n_steps)`).
  - Hidden Markov Models in [`probability.discrete.HMM`](probability/discrete/hmm.py):
    - Scaled Forward algorithm for sequence marginal log-likelihood $\ln P(X_{1:T})$ and causal online filtering $P(Z_t \mid x_{1:t})$ (`.log_likelihood()`, `.filter()`).
    - Backward algorithm for exact retrospective offline smoothing $P(Z_t \mid x_{1:T})$ (`.smooth()`).
    - Dynamic programming log-space Viterbi algorithm for global MAP sequence decoding $\arg\max_{z_{1:T}} P(z_{1:T} \mid x_{1:T})$ (`.decode()`).
    - Trajectory sampling (`.sample()`), state probability curves visualization (`.plot_states()`), and temporal unrolling into `BayesianNetwork` (`.to_bayesian_network(n_steps)`).
  - Unit tests in [`tests/test_discrete/test_markov_chain.py`](tests/test_discrete/test_markov_chain.py), [`tests/test_discrete/test_hmm.py`](tests/test_discrete/test_hmm.py), and [`tests/test_examples/test_brml/test_chapter_23.py`](tests/test_examples/test_brml/test_chapter_23.py).
  - Showcase executed notebook in [`examples/notebooks/brml/chapter_23__discrete_state_markov_models.ipynb`](examples/notebooks/brml/chapter_23__discrete_state_markov_models.ipynb) covering Weather Forecasting, Gambler's Ruin absorbing chain, and the Dishonest Casino benchmark.
- [x] **5.3 Continuous Uncertainty in `DecisionTree` and `ProjectNetwork`**
  - Enhanced [`DecisionTree`](probability/models/decision_tree/decision_tree.py), [`ChanceNode`](probability/models/decision_tree/nodes.py), and [`AmountNode`](probability/models/decision_tree/nodes.py):
    - Supported distribution-valued payoffs and costs (e.g., `Normal(50, 5)` or `PERT(1, 2, 3)`) with automatic expected value extraction via `.mean_amount`.
    - Computed exact cumulative risk profiles (lottery over terminal monetary outcomes) via `.risk_profile(choice_name=None, policy='optimal')`.
    - Visualized risk profiles as cumulative step CDFs via `.plot_risk_profile()`.
    - Computed Expected Value of Perfect Information (EVPI) both analytically from payoff matrices (`DecisionTree.calculate_evpi(payoffs, probs)`) and directly from solved tree models (`.evpi()`).
  - Enhanced [`ProjectNetwork`](probability/models/project_network/project_network.py):
    - Monte Carlo simulation of task duration realizations and project paths (`.simulate(n_samples=1000)`).
    - Project completion time distribution wrapped as a first-class [`Ratio`](probability/distributions/data/ratio.py) object (`.completion_time_distribution(n_samples=1000)`).
    - Critical path probabilities for all simple paths (`.critical_path_probabilities()`).
    - Task criticality index $P(\text{Task } T \text{ on critical path})$ (`.task_criticality_index()`).
    - Visualized completion time KDE with mean and 95th-percentile lines (`.plot_completion_time()`).
  - Comprehensive unit tests in [`tests/test_models/test_decision_tree.py`](tests/test_models/test_decision_tree.py) and [`tests/test_models/test_project_network.py`](tests/test_models/test_project_network.py).
  - Runnable example scripts in [`examples/models/decision_tree/risk_profile_and_evpi.py`](examples/models/decision_tree/risk_profile_and_evpi.py) and [`examples/models/project_network.py`](examples/models/project_network.py).

---

### Phase 6: Prior Elicitation & PyMC / ArviZ Bridge
Make `probability` the best companion library for users who *do* drop into PyMC for complex hierarchical models.

- [x] **6.1 Prior Elicitation From Quantiles (`.from_interval()` / `.from_quantiles()`, `.from_mean_and_std()`)**
  - Elicitation module in [`probability.distributions.elicitation`](probability/distributions/elicitation.py) with exact analytical and 1D root-finding solvers for:
    - `elicit_normal(q1, v1, q2, v2)` (closed-form)
    - `elicit_log_normal(q1, v1, q2, v2)` (closed-form on log scale)
    - `elicit_gamma(q1, v1, q2, v2)` (robust monotonic 1D Brent's method on $\alpha$ with closed-form rate $\beta$)
    - `elicit_beta(q1, v1, q2, v2)` (regularized incomplete beta root solver)
    - `elicit_inverse_gamma(q1, v1, q2, v2)` (exact duality to Gamma via $1/Y$)
    - `elicit_continuous_uniform(q1, v1, q2, v2)` (closed-form)
  - Added `.from_interval(lower, upper, prob=0.95)`, `.from_quantiles(q1, v1, q2, v2)`, and `.from_mean_and_std(mean, std)` classmethods to `Normal`, `Beta`, `Gamma`, `InverseGamma`, `LogNormal`, and `ContinuousUniform`.
  - Added [`LogNormal`](probability/distributions/continuous/log_normal.py) continuous distribution with mode, interval, fit, and elicitation methods.
  - Comprehensive unit tests in [`tests/test_distributions/test_continuous/test_prior_elicitation.py`](tests/test_distributions/test_continuous/test_prior_elicitation.py) and runnable example script in [`examples/distributions/continuous/prior_elicitation.py`](examples/distributions/continuous/prior_elicitation.py).
- [x] **6.2 Posterior Sample Wrapping (`Ratio` Enhancements & ROPE Analysis)**
  - Enhanced [`Ratio`](probability/distributions/data/ratio.py) with first-class posterior sample analysis:
    - `.from_samples(samples, name=None)` constructor accepting array-like sequences (lists, NumPy arrays, Pandas Series) as well as duck-typed ArviZ/xarray `InferenceData` objects without hard dependency on `arviz`.
    - Empirical Highest Density Interval `.hdi(prob=0.95)` computing the shortest interval enclosing the specified posterior mass.
    - Region of Practical Equivalence (ROPE) metrics:
      - `.prob_in_rope(low, high)`: posterior probability that the parameter lies within $[low, high]$.
      - `.prob_outside_rope(low, high)`: posterior probability mass strictly outside the ROPE.
      - `.rope_decision(low, high, prob=0.95)`: Kruschke decision rule evaluating whether the $95\%$ HDI is completely inside ROPE ('accept_null'), completely outside ('reject_null'), or overlapping ('undecided').
      - `.plot_rope(low, high, prob=0.95, ax=None)`: KDE distribution plot annotated with the HDI credible band and shaded ROPE region.
  - Comprehensive unit tests in [`tests/test_distributions/test_data/test_ratio_posterior_samples.py`](tests/test_distributions/test_data/test_ratio_posterior_samples.py).
- [x] **6.3 Optional PyMC Export Helper (`.to_pymc(name, **kwargs)`)**
  - Optional bridge in [`probability.distributions.pymc`](probability/distributions/pymc.py) and on [`RVS1dMixin.to_pymc(name, **kwargs)`](probability/distributions/mixins/rv_mixins.py):
    - Completely dependency-free core with lazy import inside `to_pymc()`. Raises an informative `ImportError` if `pymc` is not installed.
    - Seamlessly converts continuous and discrete distributions (`Normal`, `Beta`, `Gamma`, `InverseGamma`, `ContinuousUniform`, `LogNormal`, `Exponential`, `Cauchy`, `Weibull`, `Pareto`, `StudentsT`, `HalfNormal`, `Laplace`, `Binomial`, `Poisson`, `Geometric`, `NegativeBinomial`, `DiscreteUniform`) into corresponding PyMC random variables inside active `with pm.Model():` contexts.
  - Comprehensive unit tests in [`tests/test_distributions/test_pymc_bridge.py`](tests/test_distributions/test_pymc_bridge.py).

---

## 3. Recommended Implementation Order

1. **Phase 1.1–1.2**: `GammaNormalConjugate`, `InverseGammaNormalConjugate`, and `NormalInverseGammaConjugate` (completes the conjugate module and unlocks exact Bayesian t-tests/regression).
2. **Phase 4.1–4.2**: `BayesianLinearRegression` and FWL residualization (`Ratio.residualize`) (unlocks *Causal Inference in Python* Chapters 03–05 notebooks).
3. **Phase 3.1–3.3**: Empirical Bayes shrinkage on `BooleanSeries`, `CountSeries`, and `RatioSeries`.
4. **Phase 2.1–2.3**: 1D/2D Grid and Laplace approximations (unlocks *Think Bayes* Chapters 02+ and *Statistical Rethinking* notebooks).
5. **Phase 6.1–6.3**: Quantile-based prior elicitation (`.from_interval()`) and MCMC sample/ROPE helpers.
6. **Phase 5.1–5.3**: `BayesianNetwork` wrapper, `MarkovChain`, and EVPI in `DecisionTree`.
