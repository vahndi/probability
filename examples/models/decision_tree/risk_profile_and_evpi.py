"""
Decision Tree Uncertainty, Risk Profiles, and Expected Value of Perfect Information (EVPI).
Demonstrates:
1. Adding distribution-valued and scalar costs/payoffs to DecisionTree.
2. Computing and plotting the cumulative Risk Profile (empirical lottery over terminal payoffs).
3. Calculating Expected Value of Perfect Information (EVPI) analytically and from tree models.
"""
import matplotlib.pyplot as plt
import pandas as pd

from probability.distributions.continuous.normal import Normal
from probability.models.decision_tree import DecisionTree


def run_oil_drilling_evpi():
    print("=" * 60)
    print("1. Clemen Oil Drilling: Expected Value of Perfect Information (EVPI)")
    print("=" * 60)
    # Payoffs:
    # Drill: Dry (-$100k), Wet (+$150k), Soaking (+$500k)
    # Don't Drill: Dry ($0), Wet ($0), Soaking ($0)
    # State probabilities: Dry 50%, Wet 30%, Soaking 20%
    payoffs = {
        'Drill': {'Dry': -100.0, 'Wet': 150.0, 'Soaking': 500.0},
        'DontDrill': {'Dry': 0.0, 'Wet': 0.0, 'Soaking': 0.0}
    }
    state_probs = {'Dry': 0.50, 'Wet': 0.30, 'Soaking': 0.20}

    print("Payoff Matrix:")
    print(pd.DataFrame(payoffs).T)

    evpi = DecisionTree.calculate_evpi(payoffs, state_probs, minimize=False)
    print(f"\nCalculated EVPI: ${evpi:,.2f}k")
    print(f"Interpretation: The maximum a rational decision-maker should pay for a perfect")
    print(f"geological survey before deciding whether to drill is ${evpi * 1000:,.0f}.\n")


def run_decision_tree_risk_profiles():
    print("=" * 60)
    print("2. Multi-Stage Decision Tree with Distribution-Valued Payoffs")
    print("=" * 60)

    dt = DecisionTree()
    dt.add_decision(name='R&D_Strategy')

    # Choice A: In-house aggressive R&D (High cost, high payoff variance)
    # Initial R&D cost: Normal distribution ~ N(50, 5)
    dt.add_option(name='InHouse_RD', p_success=0.70, amount=Normal(50, 5), parent_name='R&D_Strategy')

    # If in-house fails, can pivot to Acquisition or Abandon
    dt.add_decision(name='Pivot_Decision', parent_name='InHouse_RD')
    dt.add_option(name='Acquire_Startup', p_success=0.85, amount=60.0, parent_name='Pivot_Decision', final=True)
    dt.add_option(name='Abandon_Project', p_success=1.00, amount=0.0, parent_name='Pivot_Decision', final=True)

    # Choice B: Licensing partnership (Low fixed cost, moderate certainty)
    dt.add_option(name='License_Partner', p_success=0.80, amount=35.0, parent_name='R&D_Strategy', final=True)

    # Solve for cost minimization
    dt.solve(minimize=True)

    print("Solved Expected Costs:")
    print(f"InHouse_RD expected cost:     ${dt.chance_node('InHouse_RD').expected_amount:.2f}k")
    print(f"License_Partner expected cost: ${dt.chance_node('License_Partner').expected_amount:.2f}k")
    print(f"Optimal Initial Strategy:      {min(['InHouse_RD', 'License_Partner'], key=lambda c: dt.chance_node(c).expected_amount)}")

    # Calculate exact risk profiles
    rp = dt.risk_profile('InHouse_RD', policy='optimal')
    print("\nInHouse_RD Risk Profile (Optimal Policy):")
    print(rp[['outcome', 'probability', 'amount', 'cumulative_probability']])

    fig, ax = plt.subplots(figsize=(8, 4))
    dt.plot_risk_profile(ax=ax)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_oil_drilling_evpi()
    run_decision_tree_risk_profiles()
