from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbclient import NotebookClient

NOTEBOOKS_DIR = Path(__file__).resolve().parent


def execute_and_save(nb, path: Path):
    print(f"Executing: {path.name}...")
    client = NotebookClient(
        nb,
        timeout=180,
        resources={'metadata': {'path': str(path.parent)}}
    )
    client.execute()
    nbformat.write(nb, str(path))
    print(f"  Successfully executed and saved: {path.name}")


def build_decision_analysis_notebook():
    path = NOTEBOOKS_DIR / 'decision_analysis_and_project_networks.ipynb'
    nb = new_notebook()
    cells = [
        new_markdown_cell(
            "# Decision Analysis, Risk Profiles & Project Networks\n"
            "> *References: Clemen & Reilly, Making Hard Decisions: An Introduction to Decision Analysis; Project Management Institute (PMI) Practice Standard for Project Risk Management.*\n\n"
            "Quantitative decision-making requires systematically modeling risks and stochastic rewards.\n"
            "This notebook demonstrates how the `probability` library unifies:\n"
            "1. **Expected Value of Perfect Information (EVPI)** in decision problems (e.g. Clemen's classic Oil Drilling problem).\n"
            "2. **Decision Trees with Continuous Uncertainty**: Incorporating stochastic costs and payoffs (`Normal`, `PERT`) directly into multi-stage decision trees.\n"
            "3. **Cumulative Risk Profiles**: Constructing exact probability distributions (lotteries) over terminal outcomes to evaluate risk and stochastic dominance.\n"
            "4. **Stochastic Project Networks (`ProjectNetwork`)**: Modeling complex precedence DAGs with uncertain task durations (`PERT`).\n"
            "5. **Monte Carlo Project Simulation**: Computing dynamic critical path probabilities, task criticality indices, and the project completion time distribution wrapped as a first-class `Ratio` distribution."
        ),
        new_code_cell(
            "import set_paths\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n\n"
            "from probability.distributions.continuous.normal import Normal\n"
            "from probability.distributions.continuous.pert import PERT\n"
            "from probability.models.decision_tree import DecisionTree\n"
            "from probability.models.project_network import ProjectNetwork, ProjectTask\n"
            "from probability.plotting import new_axes"
        ),
        new_markdown_cell(
            "## 1. Oil Drilling & Expected Value of Perfect Information (EVPI)\n"
            "Consider the classic oil drilling decision problem from Clemen's *Making Hard Decisions*:\n"
            "A wildcatter must decide whether to **Drill** or **Don't Drill**.\n"
            "The geological state of the site is uncertain:\n"
            "- **Dry** ($50\\%$ probability): Drilling loses $-\\$100\\text{k}$.\n"
            "- **Wet** ($30\\%$ probability): Drilling profits $+\\$150\\text{k}$.\n"
            "- **Soaking** ($20\\%$ probability): Drilling profits $+\\$500\\text{k}$.\n\n"
            "If the wildcatter does not drill, the payoff is $\\$0$ in all states.\n"
            "**Expected Value of Perfect Information (EVPI)** answers: *What is the maximum amount the wildcatter should pay for a perfectly accurate seismic survey before deciding?*"
        ),
        new_code_cell(
            "payoffs = {\n"
            "    'Drill': {'Dry': -100.0, 'Wet': 150.0, 'Soaking': 500.0},\n"
            "    'DontDrill': {'Dry': 0.0, 'Wet': 0.0, 'Soaking': 0.0}\n"
            "}\n"
            "state_probs = {'Dry': 0.50, 'Wet': 0.30, 'Soaking': 0.20}\n\n"
            "df_payoffs = pd.DataFrame(payoffs).T\n"
            "print('Payoff Matrix ($k):')\n"
            "display(df_payoffs)\n\n"
            "evpi = DecisionTree.calculate_evpi(payoffs, state_probs, minimize=False)\n"
            "print(f'\\nCalculated EVPI: ${evpi:,.2f}k (${evpi * 1000:,.0f})')\n"
            "print('Interpretation: The decision-maker should pay at most $50,000 for perfect foresight.')"
        ),
        new_markdown_cell(
            "## 2. Multi-Stage Decision Trees with Continuous Uncertainty\n"
            "In many engineering and business decisions, costs are not fixed numbers—they are stochastic random variables.\n"
            "Here, an enterprise evaluates two product development strategies:\n"
            "- **Strategy A: In-House R&D**: Initial research cost is stochastic, modeled as a continuous `Normal(mu=50, sigma=5)`.\n"
            "  - Succeeds with probability $70\\%$.\n"
            "  - If it fails, the firm faces a secondary **Pivot Decision**:\n"
            "    - *Acquire a Startup*: Cost $\\$60\\text{k}$, $85\\%$ success.\n"
            "    - *Abandon Project*: Cost $\\$0$.\n"
            "- **Strategy B: License Partner**: Fixed cost of $\\$35\\text{k}$ with $80\\%$ success.\n\n"
            "We solve by backward induction for cost minimization (`minimize=True`)."
        ),
        new_code_cell(
            "dt = DecisionTree()\n"
            "dt.add_decision(name='R&D_Strategy')\n\n"
            "# Choice A: In-House R&D with continuous Normal cost distribution\n"
            "dt.add_option(name='InHouse_RD', p_success=0.70, amount=Normal(50, 5), parent_name='R&D_Strategy')\n\n"
            "# Pivot decision upon in-house failure\n"
            "dt.add_decision(name='Pivot_Decision', parent_name='InHouse_RD')\n"
            "dt.add_option(name='Acquire_Startup', p_success=0.85, amount=60.0, parent_name='Pivot_Decision', final=True)\n"
            "dt.add_option(name='Abandon_Project', p_success=1.00, amount=0.0, parent_name='Pivot_Decision', final=True)\n\n"
            "# Choice B: Licensing\n"
            "dt.add_option(name='License_Partner', p_success=0.80, amount=35.0, parent_name='R&D_Strategy', final=True)\n\n"
            "# Solve by backward induction for cost minimization\n"
            "dt.solve(minimize=True)\n\n"
            "print('--- Solved Expected Costs ---')\n"
            "print(f\"In-House R&D Expected Cost:     ${dt.chance_node('InHouse_RD').expected_amount:.2f}k\")\n"
            "print(f\"License Partner Expected Cost:   ${dt.chance_node('License_Partner').expected_amount:.2f}k\")"
        ),
        new_markdown_cell(
            "## 3. Cumulative Risk Profiles\n"
            "Expected value alone does not tell the full story—a decision maker cares about downside risk.\n"
            "A **Risk Profile** summarizes the complete probability distribution (lottery) over all terminal outcomes under the optimal policy.\n"
            "We inspect the table and plot the cumulative risk profile step CDF."
        ),
        new_code_cell(
            "# Exact risk profile table\n"
            "rp = dt.risk_profile('InHouse_RD', policy='optimal')\n"
            "display(rp[['outcome', 'probability', 'amount', 'cumulative_probability']])\n\n"
            "# Plot cumulative risk profile\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "dt.plot_risk_profile(ax=ax)\n"
            "ax.set_title('Cumulative Risk Profile (Optimal Policy)')\n"
            "ax.set_xlabel('Cost ($k)')\n"
            "ax.set_ylabel('Cumulative Probability')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        new_markdown_cell(
            "## 4. Stochastic Project Scheduling (`ProjectNetwork`)\n"
            "Complex projects consist of interdependent tasks governed by precedence constraints.\n"
            "Standard Critical Path Method (CPM) assumes deterministic task durations.\n"
            "However, realistic tasks follow beta or `PERT(min, mode, max)` distributions.\n\n"
            "Let's construct an 11-task project network:\n"
            "- Task `A` initiates the project.\n"
            "- Tasks `B1, B2, B3` run concurrently.\n"
            "- Tasks `C, D1, D2, E` represent engineering pipelines.\n"
            "- Tasks `F, G` integrate intermediate results.\n"
            "- Task `H` provides final sign-off."
        ),
        new_code_cell(
            "net = ProjectNetwork()\n"
            "net.add_task(ProjectTask('A', PERT(1, 4, 7)))\n"
            "net.add_task(ProjectTask('B1', PERT(2, 5, 8)), parents='A')\n"
            "net.add_task(ProjectTask('B2', PERT(3, 6, 9)), parents='A')\n"
            "net.add_task(ProjectTask('B3', PERT(2, 5, 14)), parents='A')\n"
            "net.add_task(ProjectTask('C', PERT(6, 21, 30)), parents='B1')\n"
            "net.add_task(ProjectTask('D1', PERT(5, 14, 17)), parents='B2')\n"
            "net.add_task(ProjectTask('D2', PERT(2, 11, 14)), parents='B2')\n"
            "net.add_task(ProjectTask('E', PERT(6, 21, 30)), parents='B3')\n"
            "net.add_task(ProjectTask('F', PERT(5, 8, 17)), parents=['C', 'D1'])\n"
            "net.add_task(ProjectTask('G', PERT(3, 9, 15)), parents=['D2', 'E'])\n"
            "net.add_task(ProjectTask('H', PERT(3, 12, 21)), parents=['F', 'G'], end=True)\n\n"
            "# Visualize DAG structure\n"
            "fig, ax = plt.subplots(figsize=(10, 5))\n"
            "net.add_percentiles()\n"
            "net.draw(ax=ax)\n"
            "plt.title('Project Network Precedence DAG')\n"
            "plt.show()\n\n"
            "print('Deterministic 95th-percentile path lengths:')\n"
            "display(net.path_lengths())"
        ),
        new_markdown_cell(
            "## 5. Monte Carlo Simulation, Shifting Critical Paths & Task Criticality Index\n"
            "In deterministic CPM, one single path is designated 'the critical path'.\n"
            "In reality, because task durations are random variables, **different paths become critical on different runs**!\n\n"
            "Using `.critical_path_probabilities()`, we calculate the exact frequency with which each path bottlenecked the project.\n"
            "Using `.task_criticality_index()`, we rank tasks by $P(\\text{Task on critical path})$."
        ),
        new_code_cell(
            "# 1. Critical path probabilities across 2,000 Monte Carlo runs\n"
            "crit_probs = net.critical_path_probabilities(n_samples=2000, random_state=42)\n"
            "print('Critical Path Probabilities:')\n"
            "display(pd.DataFrame({'Probability': crit_probs}))\n\n"
            "# 2. Task Criticality Index P(Task is on critical path)\n"
            "crit_index = net.task_criticality_index(n_samples=2000, random_state=42)\n"
            "print('\\nTask Criticality Index:')\n"
            "display(pd.DataFrame({'Criticality Index': crit_index}))"
        ),
        new_markdown_cell(
            "## 6. Project Completion Time Distribution as a `Ratio` Distribution\n"
            "The overall completion time is not a single number, but an empirical distribution.\n"
            "`net.completion_time_distribution()` returns a first-class `Ratio` distribution instance, unlocking all sampling statistics, quantiles, and KDE plotting."
        ),
        new_code_cell(
            "completion_dist = net.completion_time_distribution(n_samples=2000, random_state=42)\n\n"
            "print(f'Expected Duration:        {completion_dist.mean():.2f} days')\n"
            "print(f'Median Duration:          {completion_dist.median():.2f} days')\n"
            "print(f'Standard Error:           {completion_dist.se():.3f} days')\n"
            "print(f'95% Probability Interval: [{completion_dist.quantile(0.025):.2f}, {completion_dist.quantile(0.975):.2f}] days')\n"
            "print(f'P(Completion <= 50 days): {(completion_dist.data <= 50).mean():.2%}')\n"
            "print(f'P(Completion <= 60 days): {(completion_dist.data <= 60).mean():.2%}')\n\n"
            "# Plot KDE distribution with mean and 95th percentile\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "net.plot_completion_time(n_samples=2000, random_state=42, ax=ax)\n"
            "plt.title('Monte Carlo Project Completion Time Distribution')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        )
    ]
    nb.cells = cells
    execute_and_save(nb, path)


def build_bayesian_workflow_notebook():
    path = NOTEBOOKS_DIR / 'bayesian_workflow_elicitation_and_rope.ipynb'
    nb = new_notebook()
    cells = [
        new_markdown_cell(
            "# The Modern Bayesian Workflow: Prior Elicitation, PyMC Export & ROPE Decision Making\n"
            "> *References: John K. Kruschke, Doing Bayesian Data Analysis: A Tutorial with R, JAGS, and Stan (2nd Edition, Chapter 12: Bayesian Approaches to Testing a Point (\"Null\") Hypothesis); Andrew Gelman et al., Bayesian Data Analysis (3rd Edition).*\n\n"
            "A modern, robust Bayesian modeling workflow consists of:\n"
            "1. **Prior Elicitation**: Translating human expert knowledge or historical engineering benchmarks into parametric priors (`Beta`, `Gamma`, `Normal`, `LogNormal`).\n"
            "2. **Model Bridging (`.to_pymc()`)**: Exporting `probability` distributions into PyMC random variables inside a probabilistic model.\n"
            "3. **MCMC Trace Ingestion (`Ratio.from_samples()`)**: Ingesting raw posterior draws or ArviZ `InferenceData` into first-class `Ratio` distributions without coupling to heavy external packages.\n"
            "4. **Highest Density Interval (HDI)**: Identifying the shortest credible parameter region containing $95\\%$ posterior mass.\n"
            "5. **Region of Practical Equivalence (ROPE) Decision Rules**: Conducting rigorous Bayesian null hypothesis testing using Kruschke's decision framework (`.prob_in_rope()`, `.rope_decision()`, `.plot_rope()`)."
        ),
        new_code_cell(
            "import set_paths\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n\n"
            "from probability.distributions import (\n"
            "    Beta, Gamma, LogNormal, Normal,\n"
            "    BetaBinomialConjugate, Ratio\n"
            ")"
        ),
        new_markdown_cell(
            "## 1. Prior Elicitation From Intuitive Intervals & Quantiles\n"
            "Domain experts rarely speak in terms of abstract mathematical parameters like $\\alpha = 3.2, \\beta = 14.8$.\n"
            "Instead, practitioners express beliefs through intuitive percentiles:\n"
            "- *\"I am 95% confident the website conversion rate is between 12% and 28%.\"*\n"
            "- *\"I am 90% confident the average server request rate is between 5 and 15 requests/sec.\"*\n"
            "- *\"Median annual household expenditure is $45k, with the 95th percentile at $110k.\"*\n\n"
            "`probability` solves this with exact analytical and numerical elicitation methods."
        ),
        new_code_cell(
            "# 1. Elicit a Beta prior from 95% interval [0.12, 0.28]\n"
            "prior_conversion = Beta.from_interval(lower=0.12, upper=0.28, prob=0.95)\n"
            "print(f'Elicited Conversion Rate Prior: {prior_conversion}')\n"
            "print(f'  Theoretical 95% interval:     {prior_conversion.interval(0.95)}')\n\n"
            "# 2. Elicit a Gamma prior from 90% interval [5.0, 15.0]\n"
            "prior_arrival = Gamma.from_interval(lower=5.0, upper=15.0, prob=0.90)\n"
            "print(f'\\nElicited Arrival Rate Prior:    {prior_arrival}')\n"
            "print(f'  Theoretical 90% interval:     {prior_arrival.interval(0.90)}')\n\n"
            "# 3. Elicit a LogNormal prior from median ($45k) and 95th percentile ($110k)\n"
            "prior_expenditure = LogNormal.from_quantiles(q1=0.50, v1=45.0, q2=0.95, v2=110.0)\n"
            "print(f'\\nElicited Expenditure Prior:     {prior_expenditure}')\n"
            "print(f'  Theoretical Median:           {prior_expenditure.median():.2f}')\n"
            "print(f'  Theoretical 95th percentile:  {prior_expenditure.ppf().at(0.95):.2f}')\n\n"
            "# Plot elicited priors\n"
            "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
            "prior_conversion.plot(ax=axes[0])\n"
            "axes[0].set_title(f'Beta Prior for Conversion\\n{prior_conversion}')\n\n"
            "prior_arrival.plot(ax=axes[1])\n"
            "axes[1].set_title(f'Gamma Prior for Arrival Rate\\n{prior_arrival}')\n\n"
            "prior_expenditure.plot(ax=axes[2])\n"
            "axes[2].set_title(f'LogNormal Prior for Expenditure\\n{prior_expenditure}')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        new_markdown_cell(
            "## 2. Bridging Priors to PyMC (`.to_pymc()`)\n"
            "When building hierarchical or custom probabilistic models, `probability` distributions seamlessly convert into PyMC random variables via `.to_pymc(name, **kwargs)`.\n\n"
            "Because PyMC is imported lazily inside `.to_pymc()`, your core library remains completely lightweight and dependency-free.\n"
            "Inside an active PyMC model, you can write:\n"
            "```python\n"
            "with pm.Model() as model:\n"
            "    theta = prior_conversion.to_pymc('theta')\n"
            "    obs = pm.Binomial('obs', n=n_trials, p=theta, observed=k_successes)\n"
            "    trace = pm.sample(2000)\n"
            "```"
        ),
        new_code_cell(
            "import inspect\n"
            "print('to_pymc signature on Beta:')\n"
            "print(inspect.signature(prior_conversion.to_pymc))\n"
            "print('\\nDocstring:')\n"
            "print(prior_conversion.to_pymc.__doc__)"
        ),
        new_markdown_cell(
            "## 3. Conjugate Posterior Updating & MCMC Sample Wrapping (`Ratio.from_samples`)\n"
            "Suppose we conduct an experiment and observe $k = 48$ conversions out of $n = 160$ users ($30\\%$ observed rate).\n"
            "Using `BetaBinomialConjugate`, we update the prior analytically.\n"
            "Then, we simulate MCMC draws and wrap them into a first-class `Ratio` distribution using `.from_samples()`."
        ),
        new_code_cell(
            "conjugate = BetaBinomialConjugate(\n"
            "    alpha=prior_conversion.alpha,\n"
            "    beta=prior_conversion.beta,\n"
            "    n=160,\n"
            "    k=48\n"
            ")\n"
            "posterior_beta = conjugate.posterior()\n"
            "print(f'Exact Conjugate Posterior: {posterior_beta}')\n"
            "print(f'Posterior Mean:            {posterior_beta.mean():.4f}')\n"
            "print(f'Posterior 95% Interval:    {posterior_beta.interval(0.95)}')\n\n"
            "# Simulate 4,000 MCMC-like posterior draws\n"
            "np.random.seed(42)\n"
            "mcmc_samples = posterior_beta.rvs(4000)\n\n"
            "# Wrap into first-class Ratio distribution\n"
            "post_ratio = Ratio.from_samples(mcmc_samples, name='Conversion Rate')\n"
            "print(f'\\nWrapped Ratio from Samples: Mean = {post_ratio.mean():.4f}, Std = {post_ratio.std():.4f}')"
        ),
        new_markdown_cell(
            "## 4. Highest Density Interval (HDI) vs. Equal-Tailed Credible Interval\n"
            "The Highest Density Interval (HDI) identifies the points of highest posterior probability density containing $95\\%$ of the probability mass.\n"
            "For asymmetric or bounded distributions, the HDI is narrower and more representative of the most credible parameter values than equal-tailed quantiles."
        ),
        new_code_cell(
            "hdi_95 = post_ratio.hdi(prob=0.95)\n"
            "ci_95 = post_ratio.ci(percent=0.95)\n\n"
            "print(f'95% Highest Density Interval (HDI): [{hdi_95[0]:.4f}, {hdi_95[1]:.4f}] (Width: {hdi_95[1] - hdi_95[0]:.4f})')\n"
            "print(f'95% Equal-Tailed Interval (CI):     [{ci_95[0]:.4f}, {ci_95[1]:.4f}] (Width: {ci_95[1] - ci_95[0]:.4f})')"
        ),
        new_markdown_cell(
            "## 5. Region of Practical Equivalence (ROPE) & Kruschke Decision Rules\n"
            "Classical null hypothesis testing evaluates a razor-thin point null ($H_0: \\theta = 0.20$), which will virtually always be rejected given a large enough sample.\n"
            "In contrast, **ROPE analysis** (Kruschke, 2018) defines an interval of negligible practical difference $[\\theta_{\\text{low}}, \\theta_{\\text{high}}]$ around the null value.\n\n"
            "For example, if the baseline conversion rate is $20\\%$, management might declare that any conversion rate between $18\\%$ and $22\\%$ ($[0.18, 0.22]$) is practically indistinguishable from status quo.\n\n"
            "**Decision Rules**:\n"
            "- **`reject_null`**: The $95\\%$ HDI falls *completely outside* the ROPE $\\to$ Decisive evidence of a substantive, real effect.\n"
            "- **`accept_null`**: The $95\\%$ HDI falls *completely inside* the ROPE $\\to$ Decisive evidence of practical equivalence.\n"
            "- **`undecided`**: The $95\\%$ HDI overlaps the ROPE boundary $\\to$ Data is inconclusive; more evidence is needed."
        ),
        new_code_cell(
            "rope = (0.18, 0.22)\n\n"
            "prob_in = post_ratio.prob_in_rope(*rope)\n"
            "prob_out = post_ratio.prob_outside_rope(*rope)\n"
            "decision = post_ratio.rope_decision(*rope, hdi_prob=0.95)\n\n"
            "print(f'ROPE:                        {rope}')\n"
            "print(f'Posterior Mass in ROPE:      {prob_in:.2%}')\n"
            "print(f'Posterior Mass Outside:      {prob_out:.2%}')\n"
            "print(f'Formal Decision (prob=0.95):  {decision.upper()}')\n"
            "print('Conclusion: We REJECT the null hypothesis. The conversion rate is practically superior to baseline.')\n\n"
            "# Visualize with .plot_rope()\n"
            "fig, ax = plt.subplots(figsize=(8, 4))\n"
            "post_ratio.plot_rope(low=rope[0], high=rope[1], hdi_prob=0.95, ax=ax)\n"
            "plt.title('ROPE Analysis: Meaningful Improvement (Reject Null)')\n"
            "plt.show()"
        ),
        new_markdown_cell(
            "## 6. The Three Decisions Compared: Reject, Accept, and Undecided\n"
            "To see Kruschke's framework in action across all three canonical outcomes, let's compare:\n"
            "1. **Meaningful Effect (`reject_null`)**: New treatment clearly outperforms the baseline.\n"
            "2. **Practical Equivalence (`accept_null`)**: A large sample confirms the treatment effect is negligible and trapped inside the ROPE.\n"
            "3. **Uncertain / Underpowered (`undecided`)**: Small sample with wide HDI overlapping the ROPE."
        ),
        new_code_cell(
            "# Create samples for the three scenarios\n"
            "np.random.seed(42)\n"
            "s_reject = Ratio.from_samples(np.random.normal(loc=0.28, scale=0.02, size=2000), name='Effect')\n"
            "s_accept = Ratio.from_samples(np.random.normal(loc=0.20, scale=0.007, size=2000), name='Equivalence')\n"
            "s_undecided = Ratio.from_samples(np.random.normal(loc=0.21, scale=0.035, size=2000), name='Uncertain')\n\n"
            "fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))\n\n"
            "for r, title, ax in zip(\n"
            "    [s_reject, s_accept, s_undecided],\n"
            "    ['1. Reject Null (Substantive Effect)', '2. Accept Null (Practical Equivalence)', '3. Undecided (Insufficient Precision)'],\n"
            "    axes\n"
            "):\n"
            "    dec = r.rope_decision(0.18, 0.22, hdi_prob=0.95)\n"
            "    r.plot_rope(0.18, 0.22, hdi_prob=0.95, ax=ax)\n"
            "    ax.set_title(f\"{title}\\nDecision: '{dec}'\")\n\n"
            "plt.tight_layout()\n"
            "plt.show()"
        )
    ]
    nb.cells = cells
    execute_and_save(nb, path)


if __name__ == '__main__':
    build_decision_analysis_notebook()
    build_bayesian_workflow_notebook()
