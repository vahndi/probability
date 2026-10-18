import sys
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbclient import NotebookClient

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent


def build_brml_chapter_03():
    nb_path = ROOT_DIR / 'examples' / 'notebooks' / 'brml' / 'chapter_03__belief_networks.ipynb'
    nb = new_notebook()

    cells = [
        new_markdown_cell(
            "# Chapter 03 — Belief Networks (Bayesian Networks)\n"
            "> *Reference: Paraphrased from David Barber's Bayesian Reasoning and Machine Learning (Chapter 3: Belief Networks).*\n\n"
            "A **Belief Network** (or **Bayesian Network**) is a Directed Acyclic Graph (DAG) whose nodes represent discrete random variables "
            "and directed edges represent direct causal or probabilistic dependencies.\n\n"
            "In this notebook, we cover:\n"
            "1. **The Classic Alarm Network** (Barber Example 3.1 / Pearl): Burglary, Earthquake, Alarm, and phone calls.\n"
            "2. **DAG Structure & Visualization**: Defining local Conditional Probability Tables (CPTs) and drawing the network.\n"
            "3. **D-Separation & Conditional Independence**: Verifying Markov properties, forks, chains, and colliders (explaining away).\n"
            "4. **Exact Inference via Variable Elimination**: Computing exact posterior distributions $P(T \\mid E=e)$ without full joint expansion.\n"
            "5. **Explaining Away in Action**: How observing an earthquake dramatically reduces the probability of a burglary.\n"
            "6. **The Asia / Chest Clinic Network** (Lauritzen & Spiegelhalter / Barber Example 3.2): 8-variable medical diagnostic network."
        ),
        new_code_cell(
            "import set_paths\n"
            "import matplotlib.pyplot as plt\n"
            "from probability.discrete import Discrete, Conditional, BayesianNetwork\n"
            "%matplotlib inline"
        ),
        new_markdown_cell(
            "## 1. The Classic Alarm Network (Pearl / Barber Example 3.1)\n\n"
            "Consider a home burglary alarm system:\n"
            "- $B$: Burglary occurred ($P(B=1) = 0.001$)\n"
            "- $E$: Earthquake occurred ($P(E=1) = 0.002$)\n"
            "- $A$: Alarm rings (triggered by Burglary, Earthquake, or both)\n"
            "- $J$: John calls when he hears the alarm\n"
            "- $M$: Mary calls when she hears the alarm\n\n"
            "Let us define each variable's CPT using `Discrete` (for root nodes) and `Conditional` (for children)."
        ),
        new_code_cell(
            "# Root priors\n"
            "burglary = Discrete.from_probs({'yes': 0.001, 'no': 0.999}, variables='burglary')\n"
            "earthquake = Discrete.from_probs({'yes': 0.002, 'no': 0.998}, variables='earthquake')\n\n"
            "# Conditional tables\n"
            "alarm = Conditional.from_probs({\n"
            "    ('yes', 'yes', 'yes'): 0.95, ('no', 'yes', 'yes'): 0.05,\n"
            "    ('yes', 'yes', 'no'):  0.94, ('no', 'yes', 'no'):  0.06,\n"
            "    ('yes', 'no',  'yes'): 0.29, ('no', 'no',  'yes'): 0.71,\n"
            "    ('yes', 'no',  'no'):  0.001, ('no', 'no',  'no'): 0.999,\n"
            "}, joint_variables='alarm', conditional_variables=['burglary', 'earthquake'])\n\n"
            "john_calls = Conditional.from_probs({\n"
            "    ('yes', 'yes'): 0.90, ('no', 'yes'): 0.10,\n"
            "    ('yes', 'no'):  0.05, ('no', 'no'):  0.95,\n"
            "}, joint_variables='john_calls', conditional_variables='alarm')\n\n"
            "mary_calls = Conditional.from_probs({\n"
            "    ('yes', 'yes'): 0.70, ('no', 'yes'): 0.30,\n"
            "    ('yes', 'no'):  0.01, ('no', 'no'):  0.99,\n"
            "}, joint_variables='mary_calls', conditional_variables='alarm')\n\n"
            "# Construct the Bayesian Network\n"
            "alarm_net = BayesianNetwork([burglary, earthquake, alarm, john_calls, mary_calls])\n"
            "print(alarm_net)\n"
            "print('Nodes:', alarm_net.nodes)\n"
            "print('Edges:', alarm_net.edges)"
        ),
        new_markdown_cell(
            "### Visualizing the DAG Structure\n"
            "We can render the directed graphical model directly via `.draw()`."
        ),
        new_code_cell(
            "fig, ax = plt.subplots(figsize=(8, 6))\n"
            "alarm_net.draw(ax=ax, layout='shell', node_color='#B2DF8A', node_size=3000, font_size=11)\n"
            "plt.show()"
        ),
        new_markdown_cell(
            "## 2. D-Separation & Conditional Independence (BRML Section 3.3)\n\n"
            "The DAG structure encodes conditional independencies through the **d-separation** criterion:\n"
            "1. **Collider (v-structure)**: Burglary and Earthquake are marginally independent: $B \\perp E \\mid \\emptyset$.\n"
            "   However, conditioning on their common child (Alarm) makes them **dependent**: $B \\not\\perp E \\mid A$.\n"
            "2. **Markov Chain**: John and Mary only communicate through the Alarm: $J \\perp M \\mid A$.\n"
            "   Without knowing if the alarm sounded, their calls are dependent: $J \\not\\perp M \\mid \\emptyset$."
        ),
        new_code_cell(
            "print('=== D-Separation Queries ===')\n"
            "print('1. Burglary independent of Earthquake?            ', alarm_net.is_independent('burglary', 'earthquake'))\n"
            "print('2. Burglary independent of Earthquake given Alarm?', alarm_net.is_independent('burglary', 'earthquake', given='alarm'))\n"
            "print('3. JohnCalls independent of MaryCalls given Alarm? ', alarm_net.is_independent('john_calls', 'mary_calls', given='alarm'))\n"
            "print('4. JohnCalls independent of MaryCalls without Alarm?', alarm_net.is_independent('john_calls', 'mary_calls'))"
        ),
        new_markdown_cell(
            "## 3. Exact Inference via Variable Elimination\n\n"
            "Suppose both John and Mary call to report hearing the alarm: $\\text{Evidence} = \\{J=\\text{'yes'}, M=\\text{'yes'}\\}$.\n"
            "What is the updated posterior probability of a Burglary, $P(B \\mid J=\\text{'yes'}, M=\\text{'yes'})$?\n\n"
            "Using `.query()`, the network uses **Variable Elimination** to sum out the unobserved hidden variables ($E$ and $A$)."
        ),
        new_code_cell(
            "post_b = alarm_net.query(target='burglary', evidence={'john_calls': 'yes', 'mary_calls': 'yes'})\n"
            "print('Posterior Distribution for Burglary:')\n"
            "print(post_b.data)\n"
            "print(f\"\\nPrior Probability of Burglary:     {burglary.p(burglary='yes'):.4f} (0.1%)\")\n"
            "print(f\"Posterior Probability (given J, M): {post_b.p(burglary='yes'):.4f} (28.4%)\")"
        ),
        new_markdown_cell(
            "## 4. Explaining Away: The Interplay of Causes\n\n"
            "Now suppose we turn on the radio and confirm that a major **Earthquake** just struck ($E=\\text{'yes'}$).\n"
            "What happens to the probability of a burglary now?\n"
            "Because an earthquake can explain the alarm ringing and John/Mary calling, it **explains away** the evidence for a burglary."
        ),
        new_code_cell(
            "post_b_earthquake = alarm_net.query(\n"
            "    target='burglary',\n"
            "    evidence={'john_calls': 'yes', 'mary_calls': 'yes', 'earthquake': 'yes'}\n"
            ")\n"
            "p_explained_away = post_b_earthquake.p(burglary='yes')\n"
            "print(f\"P(Burglary | JohnCalls, MaryCalls, Earthquake): {p_explained_away:.4f} (~0.7%)\")\n\n"
            "# Visualizing the belief update\n"
            "fig, ax = plt.subplots(figsize=(7, 4))\n"
            "bars = ax.bar(\n"
            "    ['Prior P(B)', 'P(B | John, Mary)', 'P(B | John, Mary, Earthquake)'],\n"
            "    [burglary.p(burglary='yes'), post_b.p(burglary='yes'), p_explained_away],\n"
            "    color=['#4A90E2', '#E74C3C', '#2ECC71'],\n"
            "    width=0.5\n"
            ")\n"
            "for bar in bars:\n"
            "    yval = bar.get_height()\n"
            "    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f\"{yval:.1%}\", ha='center', va='bottom', fontweight='bold')\n"
            "ax.set_ylabel('Probability of Burglary')\n"
            "ax.set_title('Explaining Away: The Impact of Earthquake on Burglary Belief')\n"
            "ax.set_ylim(0, 0.35)\n"
            "plt.show()"
        ),
        new_markdown_cell(
            "## 5. The Asia / Chest Clinic Network (Lauritzen & Spiegelhalter / BRML Example 3.2)\n\n"
            "The **Asia network** is one of the most famous benchmark belief networks in machine learning and medical informatics.\n"
            "It models shortness of breath (**Dyspnoea**, $D$) resulting from either **Tuberculosis** ($T$), **Lung Cancer** ($L$), or **Bronchitis** ($B$):\n"
            "- $A$: Visited Asia (increases risk of Tuberculosis)\n"
            "- $S$: Smoker (increases risk of Lung Cancer and Bronchitis)\n"
            "- $T$: Tuberculosis (depends on $A$)\n"
            "- $L$: Lung Cancer (depends on $S$)\n"
            "- $B$: Bronchitis (depends on $S$)\n"
            "- $E$: Either Tuberculosis or Lung Cancer ($T \\lor L$)\n"
            "- $X$: Positive Chest X-Ray (depends on $E$)\n"
            "- $D$: Dyspnoea (depends on $E$ and $B$)\n\n"
            "Let us construct this 8-variable diagnostic network."
        ),
        new_code_cell(
            "# 1. Priors\n"
            "asia = Discrete.from_probs({'yes': 0.01, 'no': 0.99}, variables='asia')\n"
            "smoke = Discrete.from_probs({'yes': 0.50, 'no': 0.50}, variables='smoke')\n\n"
            "# 2. Tuberculosis | Asia\n"
            "tub = Conditional.from_probs({\n"
            "    ('yes', 'yes'): 0.05, ('no', 'yes'): 0.95,\n"
            "    ('yes', 'no'):  0.01, ('no', 'no'):  0.99,\n"
            "}, joint_variables='tub', conditional_variables='asia')\n\n"
            "# 3. Lung Cancer | Smoke\n"
            "cancer = Conditional.from_probs({\n"
            "    ('yes', 'yes'): 0.10, ('no', 'yes'): 0.90,\n"
            "    ('yes', 'no'):  0.01, ('no', 'no'):  0.99,\n"
            "}, joint_variables='cancer', conditional_variables='smoke')\n\n"
            "# 4. Bronchitis | Smoke\n"
            "bronch = Conditional.from_probs({\n"
            "    ('yes', 'yes'): 0.60, ('no', 'yes'): 0.40,\n"
            "    ('yes', 'no'):  0.30, ('no', 'no'):  0.70,\n"
            "}, joint_variables='bronch', conditional_variables='smoke')\n\n"
            "# 5. Either (Tub or Cancer)\n"
            "either = Conditional.from_probs({\n"
            "    ('yes', 'yes', 'yes'): 1.0, ('no', 'yes', 'yes'): 0.0,\n"
            "    ('yes', 'yes', 'no'):  1.0, ('no', 'yes', 'no'):  0.0,\n"
            "    ('yes', 'no',  'yes'): 1.0, ('no', 'no',  'yes'): 0.0,\n"
            "    ('yes', 'no',  'no'):  0.0, ('no', 'no',  'no'): 1.0,\n"
            "}, joint_variables='either', conditional_variables=['tub', 'cancer'])\n\n"
            "# 6. X-Ray | Either\n"
            "xray = Conditional.from_probs({\n"
            "    ('yes', 'yes'): 0.98, ('no', 'yes'): 0.02,\n"
            "    ('yes', 'no'):  0.05, ('no', 'no'):  0.95,\n"
            "}, joint_variables='xray', conditional_variables='either')\n\n"
            "# 7. Dyspnoea | Either, Bronchitis\n"
            "dysp = Conditional.from_probs({\n"
            "    ('yes', 'yes', 'yes'): 0.90, ('no', 'yes', 'yes'): 0.10,\n"
            "    ('yes', 'yes', 'no'):  0.70, ('no', 'yes', 'no'):  0.30,\n"
            "    ('yes', 'no',  'yes'): 0.80, ('no', 'no',  'yes'): 0.20,\n"
            "    ('yes', 'no',  'no'):  0.10, ('no', 'no',  'no'): 0.90,\n"
            "}, joint_variables='dysp', conditional_variables=['either', 'bronch'])\n\n"
            "asia_net = BayesianNetwork([asia, smoke, tub, cancer, bronch, either, xray, dysp])\n"
            "print(asia_net)"
        ),
        new_markdown_cell(
            "### Visualizing the Asia Network DAG"
        ),
        new_code_cell(
            "fig, ax = plt.subplots(figsize=(10, 7))\n"
            "asia_net.draw(ax=ax, layout='spring', node_color='#FDBF6F', node_size=2800, font_size=11)\n"
            "plt.show()"
        ),
        new_markdown_cell(
            "### Clinical Diagnosis Scenarios\n\n"
            "Suppose a patient presents with shortness of breath (**Dyspnoea** = `yes`) and a **positive X-ray** (`yes`).\n"
            "Let us evaluate the probability of Lung Cancer vs. Tuberculosis for this patient under different patient histories:\n"
            "1. **Baseline**: Dyspnoea & Positive X-ray only.\n"
            "2. **Smoker**: Patient is also known to be a smoker.\n"
            "3. **Asia Traveler**: Patient recently returned from Asia."
        ),
        new_code_cell(
            "# Scenario 1: Symptoms only (Dyspnoea + Positive X-ray)\n"
            "p_cancer_s1 = asia_net.p('cancer', 'yes', given={'dysp': 'yes', 'xray': 'yes'})\n"
            "p_tub_s1    = asia_net.p('tub', 'yes', given={'dysp': 'yes', 'xray': 'yes'})\n\n"
            "# Scenario 2: Smoker\n"
            "p_cancer_s2 = asia_net.p('cancer', 'yes', given={'dysp': 'yes', 'xray': 'yes', 'smoke': 'yes'})\n"
            "p_tub_s2    = asia_net.p('tub', 'yes', given={'dysp': 'yes', 'xray': 'yes', 'smoke': 'yes'})\n\n"
            "# Scenario 3: Non-smoker, visited Asia\n"
            "p_cancer_s3 = asia_net.p('cancer', 'yes', given={'dysp': 'yes', 'xray': 'yes', 'smoke': 'no', 'asia': 'yes'})\n"
            "p_tub_s3    = asia_net.p('tub', 'yes', given={'dysp': 'yes', 'xray': 'yes', 'smoke': 'no', 'asia': 'yes'})\n\n"
            "print(f\"Scenario 1 (Dysp + X-Ray):           Cancer = {p_cancer_s1:.1%}, Tuberculosis = {p_tub_s1:.1%}\")\n"
            "print(f\"Scenario 2 (+ Smoker):               Cancer = {p_cancer_s2:.1%}, Tuberculosis = {p_tub_s2:.1%}\")\n"
            "print(f\"Scenario 3 (+ Non-smoker, Visited Asia): Cancer = {p_cancer_s3:.1%}, Tuberculosis = {p_tub_s3:.1%}\")"
        ),
        new_code_cell(
            "import pandas as pd\n"
            "df_results = pd.DataFrame({\n"
            "    'Scenario': ['Symptoms (Dysp, Xray)', '+ Smoker', '+ Visited Asia (Non-smoker)'],\n"
            "    'P(Cancer)': [p_cancer_s1, p_cancer_s2, p_cancer_s3],\n"
            "    'P(Tuberculosis)': [p_tub_s1, p_tub_s2, p_tub_s3]\n"
            "})\n"
            "ax = df_results.plot.bar(\n"
            "    x='Scenario', y=['P(Cancer)', 'P(Tuberculosis)'],\n"
            "    figsize=(9, 5), color=['#E41A1C', '#377EB8'], rot=0\n"
            ")\n"
            "ax.set_ylabel('Posterior Probability')\n"
            "ax.set_title('Differential Diagnosis in the Asia Network (BRML Example 3.2)')\n"
            "ax.set_ylim(0, 1.0)\n"
            "ax.grid(axis='y', alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        new_markdown_cell(
            "## Summary\n"
            "In this notebook, we demonstrated:\n"
            "- How `BayesianNetwork` automatically builds a DAG from local `Discrete` and `Conditional` CPTs.\n"
            "- How **d-separation** correctly identifies conditional independencies (such as v-structures and explaining away).\n"
            "- How **Variable Elimination** computes exact posterior queries $P(T \\mid E=e)$ for high-dimensional models (like the 8-node Asia network) efficiently."
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
    build_brml_chapter_03()
