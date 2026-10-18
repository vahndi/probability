import os
import sys
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbclient import NotebookClient

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def execute_notebook(nb_path: Path):
    print(f"Executing: {nb_path.name}...")
    nb = nbformat.read(str(nb_path), as_version=4)
    client = NotebookClient(
        nb,
        timeout=180,
        resources={'metadata': {'path': str(nb_path.parent)}}
    )
    client.execute()
    nbformat.write(nb, str(nb_path))
    print(f"  Successfully executed and saved: {nb_path.name}")


def fix_chapter_02_03():
    p = ROOT_DIR / 'examples' / 'notebooks' / 'tb' / 'chapter_02_03__proportions_and_the_euro_problem.ipynb'
    nb = nbformat.read(str(p), as_version=4)
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'conjugate_model =' in cell.source:
            cell.source = (
                "conjugate_model = BetaBinomialConjugate(n=n, k=k, alpha=1.0, beta=1.0)\n"
                "conjugate_post = conjugate_model.posterior()\n\n"
                "print(f\"Conjugate Exact Mean: {conjugate_post.mean():.4f}\")\n"
                "print(f\"Grid Approximation:   {post_uniform.mean():.4f}\")\n"
                "print(f\"Difference:           {abs(conjugate_post.mean() - post_uniform.mean()):.6e}\")"
            )
    nbformat.write(nb, str(p))
    execute_notebook(p)


def build_cip_chapter_04():
    p = ROOT_DIR / 'examples' / 'notebooks' / 'cip' / 'chapter_04__the_unreasonable_effectiveness_of_linear_regression.ipynb'
    nb = new_notebook()
    cells = [
        new_markdown_cell(
            "# Chapter 04 — The Unreasonable Effectiveness of Linear Regression & FWL\n"
            "> *Reference: Paraphrased from Matheus Facure's Causal Inference in Python (Chapter 4).*\n\n"
            "Linear regression is one of the most powerful tools for causal inference when controlling for observed confounders.\n"
            "This notebook demonstrates:\n"
            "1. **Omitted Variable Bias** in estimating the returns to education (`wage.csv`).\n"
            "2. **Multivariate Causal Regression** with heteroskedasticity-robust standard errors (`LinearRegression` / `OLS`).\n"
            "3. **Distribution-First Treatment Effects**: Each coefficient as a first-class `StudentsT` sampling distribution.\n"
            "4. **Frisch-Waugh-Lovell (FWL) Theorem**: Exact partialling-out decomposition via `.fwl_decompose()` and `.plot_fwl()`.\n"
            "5. **Confounder Residualization on Distributions**: `Ratio.residualize()` to create debiased continuous distributions."
        ),
        new_code_cell(
            "import set_paths\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "from make_data import load_wage\n"
            "from probability.distributions import Ratio, StudentsT\n"
            "from probability.models import LinearRegression, OLS"
        ),
        new_markdown_cell(
            "## 4.1 Omitted Variable Bias: Returns to Education\n"
            "Does education cause higher wages, or do high-ability individuals obtain more education AND earn higher wages?\n"
            "We inspect the Young Men National Longitudinal Survey (`wage.csv`)."
        ),
        new_code_cell(
            "wage_df = load_wage().dropna(subset=['wage', 'educ', 'IQ', 'exper', 'tenure', 'age'])\n"
            "print(f\"Sample size: {len(wage_df)} workers\")\n"
            "wage_df[['wage', 'educ', 'IQ', 'exper', 'tenure', 'age']].head()"
        ),
        new_markdown_cell(
            "### Naive Bivariate Model\n"
            "Regressing hourly wage solely on education ignores ability (`IQ`) and experience, yielding a heavily biased slope."
        ),
        new_code_cell(
            "naive_model = LinearRegression.fit(\n"
            "    X=wage_df[['educ']],\n"
            "    y=wage_df['wage'],\n"
            "    cov_type='HC3'\n"
            ")\n"
            "print('--- Naive Bivariate Model (Omitted Variable Bias) ---')\n"
            "naive_model.summary().round(3)"
        ),
        new_markdown_cell(
            "## 4.2 Multivariate Causal Regression with Robust Standard Errors\n"
            "By controlling for `IQ`, `exper`, `tenure`, `age`, and `married`, we partial out ability and maturity bias.\n"
            "We use `cov_type='HC3'` for heteroskedasticity-consistent standard errors."
        ),
        new_code_cell(
            "controls = ['educ', 'IQ', 'exper', 'tenure', 'age', 'married']\n"
            "causal_model = LinearRegression.fit(\n"
            "    X=wage_df[controls],\n"
            "    y=wage_df['wage'],\n"
            "    cov_type='HC3'\n"
            ")\n"
            "print('--- Multivariate Controlled Causal Model (HC3 SE) ---')\n"
            "causal_model.summary().round(3)"
        ),
        new_markdown_cell(
            "## 4.3 Distribution-First Treatment Effect (`StudentsT`)\n"
            "The `probability` library treats regression coefficients as exact Student's t sampling distributions.\n"
            "This enables immediate probabilistic queries like $P(\\beta_{\\text{educ}} > 0)$, exact HDI intervals, and testing."
        ),
        new_code_cell(
            "beta_educ = causal_model.coefficient('educ')\n"
            "print(f\"Education Coefficient: StudentsT(nu={beta_educ.nu:.0f}, mu={beta_educ.mu:.3f}, sigma={beta_educ.sigma:.3f})\")\n"
            "print(f\"P(beta_educ > 0):       {float(beta_educ > 0):.4%}\")\n"
            "print(f\"P(beta_educ > $30/yr):   {float(beta_educ > 30):.2%}\")\n"
            "print(f\"95% HDI Interval:        {beta_educ.interval(0.95)}\")"
        ),
        new_markdown_cell(
            "## 4.4 Frisch-Waugh-Lovell (FWL) Theorem\n"
            "The Frisch-Waugh-Lovell theorem proves that multivariate regression is equivalent to:\n"
            "1. Partialling out all other covariates from the treatment (`educ` $\\to \\tilde{D}$).\n"
            "2. Partialling out all other covariates from the outcome (`wage` $\\to \\tilde{y}$).\n"
            "3. Running a simple bivariate regression of $\\tilde{y}$ on $\\tilde{D}$."
        ),
        new_code_cell(
            "tilde_educ, tilde_wage, fwl_bivariate = causal_model.fwl_decompose('educ')\n"
            "beta_multi = float(causal_model.params['educ'])\n"
            "beta_fwl = float(fwl_bivariate.params['educ_residualized'])\n\n"
            "print(f\"Multivariate Coefficient: {beta_multi:.8f}\")\n"
            "print(f\"FWL Bivariate Coefficient:{beta_fwl:.8f}\")\n"
            "print(f\"Absolute Difference:       {abs(beta_multi - beta_fwl):.2e} (Mathematically Identical!)\")"
        ),
        new_markdown_cell(
            "## 4.5 Residualization on Data Distributions (`Ratio.residualize`)\n"
            "We can directly call `.residualize()` on any `Ratio` distribution to remove confounder variance while preserving the unit scale."
        ),
        new_code_cell(
            "raw_wage = Ratio(wage_df['wage'])\n"
            "adj_wage = raw_wage.residualize(wage_df[['IQ', 'exper', 'tenure', 'age', 'married']], keep_mean=True)\n\n"
            "print(f\"Raw Wage Mean:      ${raw_wage.mean():.2f}, SD: ${raw_wage.std():.2f}\")\n"
            "print(f\"Adjusted Wage Mean: ${adj_wage.mean():.2f}, SD: ${adj_wage.std():.2f} (Confounder variance filtered)\")"
        ),
        new_markdown_cell(
            "## 4.6 Visualizations: Coefficient Forest Plot & FWL Partialled Scatter Plot\n"
            "We inspect the full coefficient forest plot and the partialled-out causal relationship."
        ),
        new_code_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n\n"
            "causal_model.plot_coefficients(\n"
            "    hdi=0.95,\n"
            "    ax=axes[0],\n"
            "    title=\"Controlled OLS Coefficients (95% CI, HC3 SE)\"\n"
            ")\n\n"
            "causal_model.plot_fwl(\n"
            "    treatment_var='educ',\n"
            "    hdi=0.95,\n"
            "    ax=axes[1],\n"
            "    title=\"FWL Partialled Causal Slope: Wage on Education\"\n"
            ")\n\n"
            "plt.tight_layout()"
        )
    ]
    nb.cells = cells
    nbformat.write(nb, str(p))
    execute_notebook(p)


def build_cip_chapter_05_06():
    p = ROOT_DIR / 'examples' / 'notebooks' / 'cip' / 'chapter_05_06__propensity_scores_and_doubly_robust_estimation.ipynb'
    nb = new_notebook()
    cells = [
        new_markdown_cell(
            "# Chapter 05 & 06 — Propensity Scores, IPW & Doubly Robust Estimation\n"
            "> *Reference: Paraphrased from Matheus Facure's Causal Inference in Python (Chapters 5 & 6).*\n\n"
            "When treatment assignment is non-random, Inverse Probability Weighting (IPW) and Doubly Robust / Augmented IPW (AIPW)\n"
            "allow us to balance covariates and estimate causal effects under conditional exchangeability.\n"
            "This notebook explores:\n"
            "1. **Confounded Observational Studies**: Management training on employee engagement (`management_training.csv`).\n"
            "2. **Propensity Score Diagnostics**: Overlap and common support via `.plot_propensity()`.\n"
            "3. **Covariate Balance & Love Plot**: Evaluating pre- and post-weighting standardized mean differences via `.love_plot()`.\n"
            "4. **Inverse Probability Weighting (IPW)** for ATE and ATT.\n"
            "5. **Augmented IPW (Doubly Robust AIPW)**: Combining propensity weights with outcome regression models for double protection.\n"
            "6. **Causal Treatment Effect Distribution**: First-class `StudentsT` inference."
        ),
        new_code_cell(
            "import set_paths\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "from make_data import load_management_training\n"
            "from probability.distributions import Ratio, Boolean, StudentsT\n"
            "from probability.models import IPW, AIPW, DoublyRobust"
        ),
        new_markdown_cell(
            "## 5.1 The Management Training Dataset\n"
            "We want to estimate the causal effect of an intensive management training program (`intervention`) on team `engagement_score`.\n"
            "Managers self-selected or were selected based on `tenure`, `n_of_reports`, `gender`, `last_engagement_score`, `department_score`, and `department_size`."
        ),
        new_code_cell(
            "mgt_df = load_management_training().dropna()\n"
            "print(f\"Sample size: {len(mgt_df)} managers\")\n"
            "print(f\"Training participation rate: {mgt_df['intervention'].mean():.1%}\")\n"
            "mgt_df[['intervention', 'engagement_score', 'tenure', 'n_of_reports', 'last_engagement_score']].head()"
        ),
        new_markdown_cell(
            "### Raw Unadjusted Difference\n"
            "A simple comparison of means shows a naive difference that is confounded by baseline department score and past engagement."
        ),
        new_code_cell(
            "y_treated = mgt_df[mgt_df['intervention'] == 1]['engagement_score'].mean()\n"
            "y_control = mgt_df[mgt_df['intervention'] == 0]['engagement_score'].mean()\n"
            "print(f\"Treated Mean Engagement: {y_treated:.3f}\")\n"
            "print(f\"Control Mean Engagement: {y_control:.3f}\")\n"
            "print(f\"Raw Unadjusted Difference: {y_treated - y_control:.3f}\")"
        ),
        new_markdown_cell(
            "## 5.2 Inverse Probability Weighting (IPW)\n"
            "We fit a logistic propensity score model $P(T=1 \\mid X)$ and calculate normalized (Hajek) IPW weights."
        ),
        new_code_cell(
            "covariate_cols = [\n"
            "    'tenure', 'n_of_reports', 'gender', 'last_engagement_score',\n"
            "    'department_score', 'department_size'\n"
            "]\n"
            "ipw_ate = IPW.fit(\n"
            "    treatment=mgt_df['intervention'],\n"
            "    outcome=mgt_df['engagement_score'],\n"
            "    covariates=mgt_df[covariate_cols],\n"
            "    estimand='ATE'\n"
            ")\n"
            "print('--- IPW ATE Estimation ---')\n"
            "ipw_ate.summary().round(3)"
        ),
        new_markdown_cell(
            "## 5.3 Covariate Balance Diagnostics & Love Plot\n"
            "A Love plot compares the absolute standardized mean difference (Cohen's d) for each covariate before and after weighting.\n"
            "Balance is considered achieved when weighted differences fall below the standard threshold of $0.10$."
        ),
        new_code_cell(
            "bal_summary = ipw_ate.balance_summary()\n"
            "print('--- Covariate Balance Summary Table ---')\n"
            "bal_summary.round(4)"
        ),
        new_code_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n\n"
            "ipw_ate.love_plot(\n"
            "    threshold=0.10,\n"
            "    ax=axes[0],\n"
            "    title=\"Love Plot: Covariate Balance (ATE IPW)\"\n"
            ")\n\n"
            "ipw_ate.plot_propensity(\n"
            "    ax=axes[1],\n"
            "    title=\"Propensity Score Common Support (Treated vs Control)\"\n"
            ")\n\n"
            "plt.tight_layout()"
        ),
        new_markdown_cell(
            "## 5.4 Augmented IPW (Doubly Robust Estimation)\n"
            "AIPW combines outcome models $\\mu_1(X) = E[Y \\mid T=1, X]$ and $\\mu_0(X) = E[Y \\mid T=0, X]$ with the propensity score $e(X)$.\n"
            "It has the **Doubly Robust** property: consistent if EITHER the outcome model OR the propensity model is correctly specified."
        ),
        new_code_cell(
            "aipw = AIPW.fit(\n"
            "    treatment=mgt_df['intervention'],\n"
            "    outcome=mgt_df['engagement_score'],\n"
            "    covariates=mgt_df[covariate_cols],\n"
            "    estimand='ATE'\n"
            ")\n"
            "print('--- Doubly Robust (AIPW) Treatment Effect ---')\n"
            "aipw.summary().round(3)"
        ),
        new_markdown_cell(
            "## 5.5 Treatment Effect Sampling Distribution (`StudentsT`)\n"
            "We inspect the doubly robust causal effect as a probability distribution."
        ),
        new_code_cell(
            "eff_dist = aipw.effect_distribution\n"
            "print(f\"Causal Effect Distribution: StudentsT(nu={eff_dist.nu:.0f}, mu={eff_dist.mu:.3f}, sigma={eff_dist.sigma:.3f})\")\n"
            "print(f\"P(Treatment Effect > 0):    {float(eff_dist > 0):.4%}\")\n"
            "print(f\"95% HDI / Confidence Interval: {eff_dist.interval(0.95)}\")"
        )
    ]
    nb.cells = cells
    nbformat.write(nb, str(p))
    execute_notebook(p)


def build_tb_chapter_16():
    p = ROOT_DIR / 'examples' / 'notebooks' / 'tb' / 'chapter_16__quadratic_laplace_approximation.ipynb'
    nb = new_notebook()
    cells = [
        new_markdown_cell(
            "# Think Bayes & Statistical Rethinking — Chapter 16: Laplace / Quadratic Approximation\n"
            "> *Reference: Paraphrased from Richard McElreath's Statistical Rethinking (Chapter 4) and Allen Downey's Think Bayes (Chapter 15).*\n\n"
            "While grid approximation is exact for low-dimensional models, parameter dimensions above $d=2$ quickly suffer from the curse of dimensionality.\n"
            "The **Laplace (or Quadratic) Approximation** (`LaplaceApproximation`, aliased as `QuadraticApproximation` and `Quap`)\n"
            "finds the Maximum A Posteriori (MAP) estimate $\\hat{\\theta}$ and inverts the negative Hessian matrix $\\mathbf{H}^{-1}$ to construct\n"
            "a Gaussian approximation to the joint posterior distribution:\n"
            "$$\\theta \\mid D \\approx \\mathcal{N}(\\hat{\\theta}, \\mathbf{H}^{-1})$$\n"
            "This notebook demonstrates:\n"
            "1. **Continuous Normal Likelihood** for adult human heights.\n"
            "2. **MAP Optimization & Numerical Hessian Inversion**.\n"
            "3. **Marginal Posterior Extraction** as exact `Normal` distributions.\n"
            "4. **Posterior Sampling & Forest Plot Visualizations**."
        ),
        new_code_cell(
            "import set_paths\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "from make_data import load_heights\n"
            "from probability.distributions import Normal, MVNormal\n"
            "from probability.models import LaplaceApproximation, Quap"
        ),
        new_markdown_cell(
            "## 16.1 Adult Heights Dataset\n"
            "We load human adult height measurements and filter to adults (age $\\ge 18$)."
        ),
        new_code_cell(
            "heights_df = load_heights()\n"
            "adults = heights_df[heights_df['age'] >= 18]['height'].values\n"
            "print(f\"Number of adult observations: {len(adults)}\")\n"
            "print(f\"Sample Mean: {np.mean(adults):.2f} cm, Sample SD: {np.std(adults, ddof=1):.2f} cm\")"
        ),
        new_markdown_cell(
            "## 16.2 Specifying the Log-Posterior\n"
            "We model heights as $h_i \\sim \\mathcal{N}(\\mu, \\sigma)$ with weakly informative priors:\n"
            "- $\\mu \\sim \\mathcal{N}(178, 20)$\n"
            "- $\\sigma \\sim \\text{Uniform}(0.1, 50)$"
        ),
        new_code_cell(
            "def log_prior(mu, sigma):\n"
            "    if sigma <= 0.1 or sigma >= 50.0:\n"
            "        return -np.inf\n"
            "    log_p_mu = -0.5 * ((mu - 178.0) / 20.0) ** 2\n"
            "    log_p_sigma = -np.log(50.0 - 0.1)\n"
            "    return log_p_mu + log_p_sigma\n\n"
            "def log_likelihood(mu, sigma, data):\n"
            "    n = len(data)\n"
            "    return -0.5 * n * np.log(2.0 * np.pi * sigma ** 2) - np.sum((data - mu) ** 2) / (2.0 * sigma ** 2)\n\n"
            "def log_post(theta):\n"
            "    mu, sigma = theta[0], theta[1]\n"
            "    lp = log_prior(mu, sigma)\n"
            "    if not np.isfinite(lp):\n"
            "        return -np.inf\n"
            "    return lp + log_likelihood(mu, sigma, adults)"
        ),
        new_markdown_cell(
            "## 16.3 Fitting the Quadratic Approximation (`Quap`)\n"
            "We fit the quadratic approximation via `LaplaceApproximation.fit()`."
        ),
        new_code_cell(
            "quap_model = LaplaceApproximation.fit(\n"
            "    log_posterior_fn=log_post,\n"
            "    start_params={'mu': 175.0, 'sigma': 10.0},\n"
            "    bounds={'mu': (120.0, 220.0), 'sigma': (0.5, 30.0)}\n"
            ")\n"
            "print('--- Laplace Approximation Summary Table ---')\n"
            "quap_model.summary(hdi=0.95).round(4)"
        ),
        new_markdown_cell(
            "## 16.4 Marginal Posterior Distributions\n"
            "The model provides analytical `Normal` distributions for each parameter's marginal posterior."
        ),
        new_code_cell(
            "mu_norm = quap_model.marginal('mu')\n"
            "sigma_norm = quap_model.marginal('sigma')\n\n"
            "print(f\"mu posterior:    Normal(mu={mu_norm.mu:.3f}, sigma={mu_norm.sigma:.3f})\")\n"
            "print(f\"mu 95% HDI:      {mu_norm.interval(0.95)}\")\n"
            "print(f\"sigma posterior: Normal(mu={sigma_norm.mu:.3f}, sigma={sigma_norm.sigma:.3f})\")\n"
            "print(f\"sigma 95% HDI:   {sigma_norm.interval(0.95)}\")"
        ),
        new_markdown_cell(
            "## 16.5 Posterior Sampling & Visualizations\n"
            "We can draw samples from the joint Gaussian posterior and render forest and marginal density plots."
        ),
        new_code_cell(
            "samples = quap_model.sample(n=2000, seed=42)\n"
            "print(f\"Drawn {len(samples)} posterior samples:\")\n"
            "samples.head()"
        ),
        new_code_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n\n"
            "quap_model.plot_forest(\n"
            "    hdi=0.95,\n"
            "    ax=axes[0],\n"
            "    title=\"Posterior Parameter Estimates (95% HDI)\"\n"
            ")\n\n"
            "quap_model.plot_marginal(\n"
            "    param='mu',\n"
            "    hdi=0.95,\n"
            "    ax=axes[1]\n"
            ")\n\n"
            "plt.tight_layout()"
        )
    ]
    nb.cells = cells
    nbformat.write(nb, str(p))
    execute_notebook(p)


def build_empirical_bayes_notebook():
    p = ROOT_DIR / 'examples' / 'notebooks' / 'empirical_bayes_shrinkage.ipynb'
    nb = new_notebook()
    cells = [
        new_markdown_cell(
            "# Empirical Bayes & Hierarchical Shrinkage Across Series\n"
            "> *Reference: Paraphrased from David Robinson's Introduction to Empirical Bayes.*\n\n"
            "In many real-world settings, we have multiple groups with very unequal sample sizes.\n"
            "Small-sample groups often exhibit extreme, volatile estimates purely due to measurement noise.\n"
            "**Empirical Bayes partial pooling (shrinkage)** borrows statistical strength across the entire ensemble of groups\n"
            "by estimating a shared prior hyperparameter, then shrinking each noisy group toward the grand prior.\n\n"
            "This notebook demonstrates complete shrinkage workflows across all three series types in `probability`:\n"
            "1. **Beta-Binomial Empirical Bayes on `BooleanSeries`** (Binary success rates / baseball batting averages).\n"
            "2. **Gamma-Poisson Empirical Bayes on `CountSeries`** (Event rates / microservice bug reports).\n"
            "3. **Normal-Normal James-Stein Empirical Bayes on `RatioSeries`** (Continuous group means / retail store sales)."
        ),
        new_code_cell(
            "import set_paths\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "from probability.distributions import Boolean, BooleanSeries, Count, CountSeries, Ratio, RatioSeries, Beta, Gamma, Normal"
        ),
        new_markdown_cell(
            "## 1. Beta-Binomial Shrinkage on `BooleanSeries`\n"
            "Scenario: Estimating player batting averages across $12$ baseball players.\n"
            "Veteran players have hundreds of at-bats, while rookies have only $2$ or $4$ at-bats (yielding extreme raw averages like $1.000$ or $0.000$)."
        ),
        new_code_cell(
            "np.random.seed(42)\n"
            "players = {\n"
            "    'Veteran_Star_A': [1] * 160 + [0] * 340,   # 500 AB, .320\n"
            "    'Veteran_Solid_B': [1] * 125 + [0] * 375,  # 500 AB, .250\n"
            "    'Veteran_Solid_C': [1] * 80 + [0] * 270,   # 350 AB, .229\n"
            "    'Regular_D':       [1] * 55 + [0] * 145,   # 200 AB, .275\n"
            "    'Regular_E':       [1] * 22 + [0] * 78,    # 100 AB, .220\n"
            "    'Bench_Player_F':  [1] * 8 + [0] * 32,     # 40 AB,  .200\n"
            "    'Backup_G':        [1] * 5 + [0] * 15,     # 20 AB,  .250\n"
            "    'Pinch_Hitter_H':  [1] * 3 + [0] * 7,      # 10 AB,  .300\n"
            "    'Rookie_Phenom_I': [1] * 3 + [0] * 0,      # 3 AB,  1.000 (Extreme!)\n"
            "    'Rookie_Cold_J':   [0] * 4,                # 4 AB,   .000 (Extreme!)\n"
            "}\n"
            "bool_series = BooleanSeries({name: Boolean(pd.Series(arr)) for name, arr in players.items()})\n"
            "prior_beta = bool_series.empirical_bayes_prior(method='mle')\n"
            "print(f\"Empirical Bayes Prior: Beta(alpha={prior_beta.alpha:.2f}, beta={prior_beta.beta:.2f})\")\n"
            "print(f\"Prior Grand Mean:      {prior_beta.mean():.3f}\")"
        ),
        new_code_cell(
            "summary_bool = bool_series.shrinkage_summary(hdi=0.95, prior=prior_beta)\n"
            "print('--- Beta-Binomial Shrinkage Table ---')\n"
            "display_cols = ['n', 'k', 'raw_mean', 'shrunk_mean', '95% HDI lower', '95% HDI upper', 'weight_data']\n"
            "summary_bool[display_cols].round(3)"
        ),
        new_code_cell(
            "fig, ax = plt.subplots(figsize=(10, 5))\n"
            "bool_series.plot_shrinkage(\n"
            "    hdi=0.95,\n"
            "    prior=prior_beta,\n"
            "    ax=ax,\n"
            "    order_by='raw_mean',\n"
            "    title=\"Beta-Binomial Empirical Bayes Shrinkage across Baseball Batting Averages\"\n"
            ")\n"
            "plt.tight_layout()"
        ),
        new_markdown_cell(
            "## 2. Gamma-Poisson Shrinkage on `CountSeries`\n"
            "Scenario: Bug / incident reports observed across $10$ microservices.\n"
            "Some legacy microservices have $300+$ hours of log exposure, while new microservices have only $1$ to $3$ hours of traffic."
        ),
        new_code_cell(
            "services = {\n"
            "    'Auth_API':        [1, 2, 1, 1, 2, 0, 1] * 50,  # 350 hrs, ~1.14/hr\n"
            "    'Billing_Service': [2, 1, 3, 2, 1, 2, 1] * 30,  # 210 hrs, ~1.71/hr\n"
            "    'Search_Indexer':  [3, 2, 1, 4, 2, 3] * 15,     # 90 hrs,  ~2.50/hr\n"
            "    'Email_Worker':    [0, 1, 1, 0, 2, 1] * 8,      # 48 hrs,  ~0.88/hr\n"
            "    'Recommendations': [2, 1, 0, 1, 3] * 4,         # 20 hrs,  ~1.40/hr\n"
            "    'Mobile_Sync':     [4, 5],                      # 2 hrs,   ~4.50/hr (Extreme!)\n"
            "    'Beta_Feature':    [0],                         # 1 hr,    ~0.00/hr (Extreme!)\n"
            "}\n"
            "count_series = CountSeries({name: Count(pd.Series(arr)) for name, arr in services.items()})\n"
            "prior_gamma = count_series.empirical_bayes_prior(method='mle')\n"
            "print(f\"Empirical Bayes Prior: Gamma(alpha={prior_gamma.alpha:.2f}, beta={prior_gamma.beta:.2f})\")\n"
            "print(f\"Prior Grand Mean Rate: {prior_gamma.mean():.3f} errors/hr\")"
        ),
        new_code_cell(
            "summary_count = count_series.shrinkage_summary(hdi=0.95, prior=prior_gamma)\n"
            "print('--- Gamma-Poisson Shrinkage Table ---')\n"
            "display_cols = ['exposure', 'total_count', 'raw_rate', 'shrunk_mean', '95% HDI lower', '95% HDI upper', 'weight_data']\n"
            "summary_count[display_cols].round(3)"
        ),
        new_code_cell(
            "fig, ax = plt.subplots(figsize=(10, 5))\n"
            "count_series.plot_shrinkage(\n"
            "    hdi=0.95,\n"
            "    prior=prior_gamma,\n"
            "    ax=ax,\n"
            "    order_by='raw_rate',\n"
            "    title=\"Gamma-Poisson Empirical Bayes Shrinkage across Microservices\"\n"
            ")\n"
            "plt.tight_layout()"
        ),
        new_markdown_cell(
            "## 3. Normal-Normal James-Stein Shrinkage on `RatioSeries`\n"
            "Scenario: Average order value (AOV) across $9$ retail store locations.\n"
            "Flagship stores have hundreds of transactions, while popup kiosks have $2$ or $3$ sales."
        ),
        new_code_cell(
            "branches = {\n"
            "    'Downtown_Flagship': np.random.normal(loc=82.0, scale=18.0, size=400),\n"
            "    'Metro_Mall':        np.random.normal(loc=79.0, scale=17.0, size=250),\n"
            "    'Westside_Center':   np.random.normal(loc=84.0, scale=19.0, size=150),\n"
            "    'Suburban_Plaza':    np.random.normal(loc=76.0, scale=16.0, size=90),\n"
            "    'University_Campus': np.random.normal(loc=72.0, scale=15.0, size=35),\n"
            "    'Airport_PopUp':     np.array([125.0, 130.0, 120.0]),  # n=3, raw mean $125.0\n"
            "    'Station_Kiosk':     np.array([40.0, 44.0]),            # n=2, raw mean $42.0\n"
            "}\n"
            "ratio_series = RatioSeries({name: Ratio(pd.Series(arr)) for name, arr in branches.items()})\n"
            "prior_normal = ratio_series.empirical_bayes_prior(method='mle')\n"
            "print(f\"Empirical Bayes Prior: Normal(mu={prior_normal.mu:.2f}, sigma={prior_normal.sigma:.2f})\")\n"
            "print(f\"Prior Grand Mean AOV:  ${prior_normal.mu:.2f}\")"
        ),
        new_code_cell(
            "summary_ratio = ratio_series.shrinkage_summary(hdi=0.95, prior=prior_normal)\n"
            "print('--- Normal-Normal (James-Stein) Shrinkage Table ---')\n"
            "display_cols = ['n', 'raw_mean', 'raw_se', 'shrunk_mean', 'shrunk_std', '95% HDI lower', '95% HDI upper', 'weight_data']\n"
            "summary_ratio[display_cols].round(3)"
        ),
        new_code_cell(
            "fig, ax = plt.subplots(figsize=(10, 5))\n"
            "ratio_series.plot_shrinkage(\n"
            "    hdi=0.95,\n"
            "    prior=prior_normal,\n"
            "    ax=ax,\n"
            "    order_by='raw_mean',\n"
            "    title=\"Normal-Normal (James-Stein) Shrinkage across Retail Branches\"\n"
            ")\n"
            "plt.tight_layout()"
        )
    ]
    nb.cells = cells
    nbformat.write(nb, str(p))
    execute_notebook(p)


def main():
    print("=" * 70)
    print("BUILDING AND EXECUTING ALL NEW SHOWCASE NOTEBOOKS")
    print("=" * 70)
    print("\n1. Fixing and re-executing Think Bayes Chapter 2 & 3...")
    fix_chapter_02_03()

    print("\n2. Building and executing CIP Chapter 4 (Regression & FWL)...")
    build_cip_chapter_04()

    print("\n3. Building and executing CIP Chapter 5 & 6 (Propensity & Doubly Robust)...")
    build_cip_chapter_05_06()

    print("\n4. Building and executing Think Bayes Chapter 16 (Laplace / Quap)...")
    build_tb_chapter_16()

    print("\n5. Building and executing Empirical Bayes Shrinkage across Series...")
    build_empirical_bayes_notebook()

    print("\nAll notebooks built and executed successfully!")


if __name__ == '__main__':
    main()
