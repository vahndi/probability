import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from probability.distributions import Boolean, BooleanSeries


def main():
    print("=" * 75)
    print("EMPIRICAL BAYES BETA-BINOMIAL SHRINKAGE ON BOOLEANSERIES")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # 1. Simulating 12 ad campaigns with heterogeneous sample sizes
    # Typical scenario: Ad campaigns or web pages with wildly varying traffic.
    # Small campaigns suffer from extreme noisy sample rates (100% or 0%),
    # whereas large campaigns have reliable estimates around the true ~15% rate.
    # -------------------------------------------------------------------------
    np.random.seed(42)
    campaigns = {
        'Search_Brand':     {'k': 95, 'n': 500},  # large sample, ~19%
        'Search_Generic':   {'k': 60, 'n': 400},  # large sample, ~15%
        'Display_Retarget': {'k': 32, 'n': 200},  # medium sample, 16%
        'Display_Affiliate':{'k': 12, 'n': 100},  # medium sample, 12%
        'Social_Video':     {'k': 18, 'n': 90},   # medium sample, 20%
        'Newsletter_Promo': {'k': 8,  'n': 40},   # small sample, 20%
        'Partner_Referral': {'k': 3,  'n': 25},   # small sample, 12%
        'Influencer_A':     {'k': 1,  'n': 2},    # tiny sample: 50%
        'Influencer_B':     {'k': 3,  'n': 3},    # tiny sample: 100% (extreme!)
        'Influencer_C':     {'k': 0,  'n': 4},    # tiny sample: 0%   (extreme!)
        'Event_Sponsorship':{'k': 0,  'n': 1},    # tiny sample: 0%
        'Test_Ad_Variant':  {'k': 2,  'n': 2},    # tiny sample: 100%
    }

    # Construct BooleanSeries
    series_dict = {}
    for name, counts in campaigns.items():
        k, n = counts['k'], counts['n']
        outcomes = [True] * k + [False] * (n - k)
        series_dict[name] = Boolean(pd.Series(outcomes))

    bool_series = BooleanSeries(series_dict)
    bool_series.data.name = 'conversion'
    bool_series.data.index.name = 'campaign'

    # -------------------------------------------------------------------------
    # 2. Fit Empirical Bayes Prior
    # -------------------------------------------------------------------------
    prior_mle = bool_series.empirical_bayes_prior(method='mle')
    prior_mom = bool_series.empirical_bayes_prior(method='moments')

    print(f"\nEmpirical Bayes Prior (MLE):     Beta(alpha={prior_mle.alpha:.3f}, beta={prior_mle.beta:.3f})")
    print(f"Prior Grand Mean:                {prior_mle.mean():.4f} ({prior_mle.mean():.2%})")
    print(f"Prior 95% HDI:                   {prior_mle.hdi(0.95)}")
    print(f"Empirical Bayes Prior (Moments): Beta(alpha={prior_mom.alpha:.3f}, beta={prior_mom.beta:.3f})")

    # -------------------------------------------------------------------------
    # 3. Shrinkage Summary Table
    # -------------------------------------------------------------------------
    summary_df = bool_series.shrinkage_summary(hdi=0.95, prior=prior_mle)
    print("\n--- Empirical Bayes Shrinkage Summary ---")
    display_cols = ['n', 'k', 'raw_mean', 'shrunk_mean', '95% HDI lower', '95% HDI upper', 'weight_data']
    print(summary_df[display_cols].round(4).to_string())

    # Notice how Influencer_B (3/3 = 100%) was shrunk down to a sensible estimate:
    inf_b_raw = summary_df.loc['Influencer_B', 'raw_mean']
    inf_b_shrunk = summary_df.loc['Influencer_B', 'shrunk_mean']
    inf_b_weight = summary_df.loc['Influencer_B', 'weight_data']
    print(f"\nHighlight - Influencer_B:")
    print(f"  Raw Rate:     {inf_b_raw:.1%}")
    print(f"  Shrunk Rate:  {inf_b_shrunk:.1%}")
    print(f"  Data Weight:  {inf_b_weight:.1%} (Prior weight = {1.0 - inf_b_weight:.1%})")

    # And Search_Brand (95/500 = 19%):
    sb_raw = summary_df.loc['Search_Brand', 'raw_mean']
    sb_shrunk = summary_df.loc['Search_Brand', 'shrunk_mean']
    sb_weight = summary_df.loc['Search_Brand', 'weight_data']
    print(f"\nHighlight - Search_Brand:")
    print(f"  Raw Rate:     {sb_raw:.1%}")
    print(f"  Shrunk Rate:  {sb_shrunk:.1%}")
    print(f"  Data Weight:  {sb_weight:.1%} (Prior weight = {1.0 - sb_weight:.1%})")

    # -------------------------------------------------------------------------
    # 4. Visualization: plot_shrinkage()
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    bool_series.plot_shrinkage(
        ax=ax,
        hdi=0.95,
        prior=prior_mle,
        order_by='raw_mean',
        title="Empirical Bayes Beta-Binomial Shrinkage Across Marketing Campaigns"
    )
    plt.tight_layout()
    print("\nPlot generated successfully.")


if __name__ == "__main__":
    main()
