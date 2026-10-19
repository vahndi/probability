from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbclient import NotebookClient

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent


def build_brml_chapter_23():
    nb_path = ROOT_DIR / 'examples' / 'notebooks' / 'brml' / 'chapter_23__discrete_state_markov_models.ipynb'
    nb = new_notebook()

    cells = [
        new_markdown_cell(
            "# Chapter 23 — Discrete State Markov Models (Markov Chains & HMMs)\n"
            "> *Reference: Paraphrased from David Barber's Bayesian Reasoning and Machine Learning (Chapter 23: Discrete-State Markov Models) & Bishop PRML Chapter 13.*\n\n"
            "Many real-world processes generate sequences of observations ordered in time or space (e.g., text, speech, genomic sequences, financial records, weather). "
            "When dealing with sequential data, the assumption that observations are independent and identically distributed (i.i.d.) fails.\n\n"
            "This notebook demonstrates:\n"
            "1. **Discrete-Time Markov Chains (`MarkovChain`)**:\n"
            "   - First-order Markov property: $P(S_t \\mid S_{t-1}, \\dots, S_0) = P(S_t \\mid S_{t-1})$\n"
            "   - Multi-step transition matrices $T^n$\n"
            "   - Stationary distributions $\\pi T = \\pi$\n"
            "   - Trajectory simulation\n"
            "   - Absorbing chains (fundamental matrix $N = (I - Q)^{-1}$, absorption probabilities $B = NR$, and expected time to absorption)\n"
            "   - Transition diagram plotting\n"
            "2. **Hidden Markov Models (`HMM`)**:\n"
            "   - Hidden state sequence $Z_{1:T}$ and observable emissions $X_{1:T}$\n"
            "   - The classic **Dishonest Casino** benchmark (Fair vs. Loaded die)\n"
            "   - Sequence marginal likelihood $P(X_{1:T})$ via scaled Forward algorithm\n"
            "   - Real-time online **Filtering** $P(Z_t \\mid X_{1:t})$\n"
            "   - Offline retrospective **Smoothing** $P(Z_t \\mid X_{1:T})$ via Forward-Backward\n"
            "   - Most probable hidden trajectory via **Viterbi Decoding** (dynamic programming in log-space)\n"
            "   - Posterior probability curves and decoded sequence visualization\n"
            "3. **Bridging to Bayesian Networks**:\n"
            "   - Unrolling temporal chains into exact `BayesianNetwork` DAGs (`.to_bayesian_network(n_steps)`)."
        ),
        new_code_cell(
            "import set_paths\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from probability.discrete import Discrete, Conditional, BayesianNetwork, MarkovChain, HMM\n"
            "%matplotlib inline"
        ),
        new_markdown_cell(
            "## Part 1: Discrete-Time Markov Chains (`MarkovChain`)\n\n"
            "### 1.1 Weather Forecasting (Two-State Markov Chain)\n\n"
            "Consider a simple weather model with states $\\mathcal{S} = \\{\\text{Sunny}, \\text{Rainy}\\}$:\n"
            "- If today is Sunny, tomorrow is Sunny with probability 0.8 and Rainy with probability 0.2.\n"
            "- If today is Rainy, tomorrow is Sunny with probability 0.4 and Rainy with probability 0.6."
        ),
        new_code_cell(
            "weather_transitions = pd.DataFrame([\n"
            "    [0.80, 0.20],\n"
            "    [0.40, 0.60]\n"
            "], index=['Sunny', 'Rainy'], columns=['Sunny', 'Rainy'])\n\n"
            "weather_mc = MarkovChain(\n"
            "    transitions=weather_transitions,\n"
            "    initial_distribution={'Sunny': 1.0, 'Rainy': 0.0}\n"
            ")\n"
            "weather_mc"
        ),
        new_markdown_cell(
            "### 1.2 Transition Diagram\n\n"
            "We can inspect the state transition graph using `.draw()`:"
        ),
        new_code_cell(
            "fig, ax = plt.subplots(figsize=(6, 4))\n"
            "weather_mc.draw(ax=ax, node_color='#72B7B2', font_size=12)\n"
            "plt.show()"
        ),
        new_markdown_cell(
            "### 1.3 Multi-Step Predictions & Stationary Distribution\n\n"
            "The probability of transitioning between states after $n$ steps is given by powers of the transition matrix, $T^n$.\n"
            "As $n \\to \\infty$, an ergodic chain converges to its unique **stationary distribution** $\\pi$ satisfying:\n"
            "$$\\pi T = \\pi, \\quad \\sum_{i} \\pi_i = 1$$\n\n"
            "Analytically for our 2-state weather matrix:\n"
            "$$\\pi_S = 0.8 \\pi_S + 0.4 \\pi_R \\implies 0.2 \\pi_S = 0.4 \\pi_R \\implies \\pi_S = 2\\pi_R$$\n"
            "Since $\\pi_S + \\pi_R = 1$, we have $\\pi_S = \\frac{2}{3} \\approx 0.667$ and $\\pi_R = \\frac{1}{3} \\approx 0.333$."
        ),
        new_code_cell(
            "# Multi-step transition matrices\n"
            "print('1-Step Transition Matrix T^1:')\n"
            "display(weather_mc.n_step_transition_matrix(1))\n\n"
            "print('2-Step Transition Matrix T^2:')\n"
            "display(weather_mc.n_step_transition_matrix(2))\n\n"
            "print('10-Step Transition Matrix T^10 (converged):')\n"
            "display(weather_mc.n_step_transition_matrix(10))\n\n"
            "# Exact stationary distribution\n"
            "stat_dist = weather_mc.stationary_distribution()\n"
            "print('\\nStationary Distribution pi:')\n"
            "print(f'P(Sunny) = {stat_dist.p(state=\"Sunny\"):.4f}')\n"
            "print(f'P(Rainy) = {stat_dist.p(state=\"Rainy\"):.4f}')"
        ),
        new_markdown_cell(
            "### 1.4 Simulating Trajectories\n\n"
            "We can generate a synthetic realized trajectory across $N$ time steps using `.simulate()`:"
        ),
        new_code_cell(
            "np.random.seed(42)\n"
            "trajectory = weather_mc.simulate(n_steps=60, random_state=42)\n"
            "print('First 20 simulated steps:', trajectory[:20])\n\n"
            "# Plot simulated timeline\n"
            "fig, ax = plt.subplots(figsize=(12, 3))\n"
            "numeric_traj = [1 if s == 'Sunny' else 0 for s in trajectory]\n"
            "ax.step(range(len(numeric_traj)), numeric_traj, where='mid', color='#4E79A7', lw=2)\n"
            "ax.set_yticks([0, 1])\n"
            "ax.set_yticklabels(['Rainy', 'Sunny'], fontsize=11)\n"
            "ax.set_xlabel('Time Step (Days)')\n"
            "ax.set_title('Simulated Weather Trajectory (60 Days)')\n"
            "ax.grid(axis='x', alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        new_markdown_cell(
            "### 1.5 Absorbing Markov Chains (Gambler's Ruin)\n\n"
            "An **absorbing Markov chain** has at least one absorbing state $s$ ($T_{ss} = 1$), and every transient state can eventually reach an absorbing state.\n\n"
            "Consider a Gambler starting with money $k \\in \\{1, 2, 3, 4\\}$ playing a fair coin toss game against a house with total capital $N = 5$:\n"
            "- State 0: Ruin (Broke, Absorbing)\n"
            "- States 1, 2, 3, 4: Active gambling (Transient, win +1 with $p=0.5$, lose -1 with $p=0.5$)\n"
            "- State 5: Goal / Bankrupt the House (Absorbing)\n\n"
            "Using the fundamental matrix $N = (I - Q)^{-1}$:\n"
            "- $N_{ij}$: Expected number of visits to transient state $j$ before absorption when starting at $i$\n"
            "- $t = N \\mathbf{1}$: Expected time (number of steps) until absorption\n"
            "- $B = NR$: Absorption probabilities (probability of ending in ruin vs. victory)"
        ),
        new_code_cell(
            "# 6 states: 0, 1, 2, 3, 4, 5\n"
            "states = ['0', '1', '2', '3', '4', '5']\n"
            "T = np.zeros((6, 6))\n"
            "T[0, 0] = 1.0  # State 0 absorbing\n"
            "T[5, 5] = 1.0  # State 5 absorbing\n"
            "for i in range(1, 5):\n"
            "    T[i, i - 1] = 0.5  # Lose $1\n"
            "    T[i, i + 1] = 0.5  # Win $1\n\n"
            "gambler_mc = MarkovChain(pd.DataFrame(T, index=states, columns=states))\n"
            "print('Is absorbing chain:', gambler_mc.is_absorbing_chain())\n"
            "print('Absorbing states:', gambler_mc.absorbing_states)\n"
            "print('Transient states:', gambler_mc.transient_states)"
        ),
        new_code_cell(
            "# Fundamental matrix N = (I - Q)^-1\n"
            "N_mat = gambler_mc.fundamental_matrix()\n"
            "print('Fundamental Matrix N:')\n"
            "display(N_mat)\n\n"
            "# Expected duration until absorption\n"
            "exp_time = gambler_mc.expected_time_to_absorption()\n"
            "print('\\nExpected steps to absorption (ruin or victory) from each starting capital:')\n"
            "display(exp_time)\n\n"
            "# Absorption probabilities B = N * R\n"
            "abs_prob = gambler_mc.absorption_probabilities()\n"
            "print('\\nAbsorption probabilities (ending broke vs. winning $5):')\n"
            "display(abs_prob)"
        ),
        new_markdown_cell(
            "Notice how from starting capital $k$:\n"
            "- Expected time to absorption is exactly $k(N - k)$: e.g. for $k=2, N=5$, time $= 2 \\times 3 = 6$ steps!\n"
            "- Probability of reaching the goal ($5) is exactly $k / N$: e.g. for $k=2$, prob $= 2/5 = 0.40$!"
        ),
        new_markdown_cell(
            "## Part 2: Hidden Markov Models (`HMM`)\n\n"
            "In a **Hidden Markov Model**, the underlying state sequence $Z_1, Z_2, \\dots, Z_T$ is latent (unobserved). "
            "At each time step $t$, the system emits an observation $X_t$ according to state-conditional emission probabilities $P(X_t \\mid Z_t)$.\n\n"
            "### 2.1 The Dishonest Casino Benchmark (BRML Example 23.1)\n\n"
            "A casino uses two dice:\n"
            "1. **Fair Die ($F$)**: All faces $1, \\dots, 6$ have equal probability $\\frac{1}{6} \\approx 0.1667$.\n"
            "2. **Loaded Die ($L$)**: Face $6$ occurs with probability $0.50$, and faces $1, \\dots, 5$ each have probability $0.10$.\n\n"
            "The croupier occasionally switches between the dice:\n"
            "- $P(Z_t = F \\mid Z_{t-1} = F) = 0.95, \\quad P(Z_t = L \\mid Z_{t-1} = F) = 0.05$\n"
            "- $P(Z_t = L \\mid Z_{t-1} = L) = 0.90, \\quad P(Z_t = F \\mid Z_{t-1} = L) = 0.10$\n"
            "- Prior: $P(Z_1 = F) = 0.80, \\quad P(Z_1 = L) = 0.20$"
        ),
        new_code_cell(
            "casino_states = ['Fair', 'Loaded']\n"
            "casino_emissions = [1, 2, 3, 4, 5, 6]\n\n"
            "transitions = pd.DataFrame([\n"
            "    [0.95, 0.05],\n"
            "    [0.10, 0.90]\n"
            "], index=casino_states, columns=casino_states)\n\n"
            "emissions = pd.DataFrame([\n"
            "    [1.0 / 6] * 6,\n"
            "    [0.10, 0.10, 0.10, 0.10, 0.10, 0.50]\n"
            "], index=casino_states, columns=casino_emissions)\n\n"
            "casino_hmm = HMM(\n"
            "    states=casino_states,\n"
            "    emissions=casino_emissions,\n"
            "    transitions=transitions,\n"
            "    emission_matrix=emissions,\n"
            "    initial_distribution={'Fair': 0.8, 'Loaded': 0.2}\n"
            ")\n"
            "casino_hmm"
        ),
        new_markdown_cell(
            "### 2.2 Simulating the Casino Process\n\n"
            "Let's sample 100 consecutive rolls from the croupier and observe the hidden state switches:"
        ),
        new_code_cell(
            "np.random.seed(123)\n"
            "true_states, observed_rolls = casino_hmm.sample(n_steps=100, random_state=123)\n"
            "print('First 30 true states:   ', ''.join(['F' if s == 'Fair' else 'L' for s in true_states[:30]]))\n"
            "print('First 30 observed rolls:', observed_rolls[:30])"
        ),
        new_markdown_cell(
            "### 2.3 Likelihood of the Observation Sequence\n\n"
            "Using Rabiner's scaled Forward algorithm, we can evaluate the exact log-likelihood $\\ln P(X_{1:T})$ without numerical underflow:"
        ),
        new_code_cell(
            "log_lik = casino_hmm.log_likelihood(observed_rolls)\n"
            "print(f'Total log-likelihood ln P(X_{{1:100}}) = {log_lik:.2f}')\n"
            "print(f'Average log-likelihood per roll       = {log_lik / len(observed_rolls):.4f}')"
        ),
        new_markdown_cell(
            "### 2.4 Online Filtering vs. Offline Smoothing\n\n"
            "- **Filtering** $P(Z_t \\mid x_{1:t})$: Uses only observations up to the current time step (causal / online).\n"
            "- **Smoothing** $P(Z_t \\mid x_{1:T})$: Uses all observations before and after step $t$ via the Backward pass (retrospective / offline).\n\n"
            "Notice how smoothing provides cleaner, more confident estimates of when the loaded die was active!"
        ),
        new_code_cell(
            "filtered_df = casino_hmm.filter(observed_rolls)\n"
            "smoothed_df = casino_hmm.smooth(observed_rolls)\n\n"
            "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)\n"
            "# Filtered\n"
            "casino_hmm.plot_states(observed_rolls, smoothed=False, true_states=true_states, ax=ax1)\n"
            "ax1.set_title('Filtering: P(Z_t = Loaded | x_{1:t}) — Online / Causal', fontsize=12)\n"
            "# Smoothed\n"
            "casino_hmm.plot_states(observed_rolls, smoothed=True, true_states=true_states, ax=ax2)\n"
            "ax2.set_title('Smoothing: P(Z_t = Loaded | x_{1:T}) — Forward-Backward / Full Sequence', fontsize=12)\n\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        new_markdown_cell(
            "### 2.5 Viterbi Decoding (Global Maximum a Posteriori Sequence)\n\n"
            "Finding the single most probable sequence of hidden states $\\arg\\max_{z_{1:T}} P(z_{1:T} \\mid x_{1:T})$ is solved globally by the dynamic programming **Viterbi algorithm** in log-space.\n\n"
            "Let's compare the Viterbi decoded path against the ground truth hidden states:"
        ),
        new_code_cell(
            "decoded_path, max_log_prob = casino_hmm.decode(observed_rolls)\n\n"
            "# Compute accuracy\n"
            "matches = sum(1 for true, dec in zip(true_states, decoded_path) if true == dec)\n"
            "accuracy = matches / len(true_states)\n"
            "print(f'Viterbi sequence log-probability: {max_log_prob:.2f}')\n"
            "print(f'Decoded vs. True State Agreement: {accuracy * 100:.1f}% ({matches}/{len(true_states)})')\n\n"
            "# Visual comparison\n"
            "fig, ax = plt.subplots(figsize=(13, 3))\n"
            "y_true = [1 if s == 'Loaded' else 0 for s in true_states]\n"
            "y_viterbi = [0.98 if s == 'Loaded' else -0.02 for s in decoded_path]\n\n"
            "ax.plot(range(1, 101), y_true, label='True Hidden State (1=Loaded, 0=Fair)', color='black', lw=2)\n"
            "ax.plot(range(1, 101), y_viterbi, label='Viterbi Decoded Path', color='#E41A1C', linestyle='--', lw=2)\n"
            "ax.set_yticks([0, 1])\n"
            "ax.set_yticklabels(['Fair', 'Loaded'], fontsize=11)\n"
            "ax.set_xlabel('Roll Index t')\n"
            "ax.set_title('Viterbi State Decoding vs. Ground Truth')\n"
            "ax.legend(loc='upper right')\n"
            "ax.grid(alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        new_markdown_cell(
            "## Part 3: Temporal Unrolling into Bayesian Networks\n\n"
            "Both `MarkovChain` and `HMM` seamlessly unroll into a `BayesianNetwork` DAG via `.to_bayesian_network(n_steps)`.\n"
            "This enables exact graph-based reasoning, d-separation queries, and conditional independence tests on sequential models."
        ),
        new_code_cell(
            "unrolled_bn = casino_hmm.to_bayesian_network(n_steps=3)\n"
            "print('Unrolled HMM Bayesian Network nodes:', unrolled_bn.nodes)\n\n"
            "# Verify the Markov property: X_0 is conditionally independent of X_2 given Z_1\n"
            "is_indep = unrolled_bn.is_independent('X_0', 'X_2', given=['Z_1'])\n"
            "print(f'Is X_0 independent of X_2 given Z_1? {is_indep}')\n\n"
            "# Exact variable elimination query on the unrolled network:\n"
            "# If rolls at t=0 and t=1 are both 6, what is the posterior probability that Z_1 was Loaded?\n"
            "q_res = unrolled_bn.query('Z_1', evidence={'X_0': 6, 'X_1': 6})\n"
            "print('\\nP(Z_1 | X_0=6, X_1=6):')\n"
            "print(f'P(Loaded) = {q_res.p(Z_1=\"Loaded\"):.4f}')\n"
            "print(f'P(Fair)   = {q_res.p(Z_1=\"Fair\"):.4f}')"
        ),
        new_markdown_cell(
            "## Summary\n\n"
            "In this notebook, we demonstrated:\n"
            "- **`MarkovChain`**: Computing $n$-step transitions, solving for stationary distributions, calculating fundamental matrices for absorbing chains, and simulating trajectories.\n"
            "- **`HMM`**: The Dishonest Casino benchmark, exact scaled sequence likelihood, online causal filtering vs. retrospective full-sequence smoothing, and dynamic programming Viterbi decoding.\n"
            "- **Temporal Unrolling**: Converting sequential models directly into Directed Acyclic Graphs with `BayesianNetwork`."
        )
    ]

    nb.cells = cells
    nbformat.write(nb, str(nb_path))
    print(f"Built notebook: {nb_path.name}")

    print("Executing notebook...")
    client = NotebookClient(
        nb,
        timeout=180,
        resources={'metadata': {'path': str(nb_path.parent)}}
    )
    client.execute()
    nbformat.write(nb, str(nb_path))
    print(f"Successfully executed and saved: {nb_path.name}")


if __name__ == '__main__':
    build_brml_chapter_23()
