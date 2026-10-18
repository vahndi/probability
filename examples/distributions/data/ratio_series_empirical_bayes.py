import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from probability.distributions import Ratio, RatioSeries


def main():
    print("=" * 75)
    print("EMPIRICAL BAYES NORMAL-NORMAL (JAMES-STEIN) SHRINKAGE ON RATIOSERIES")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # 1. Simulating 10 store branches with varying customer transaction counts
    # Scenario: Average order value (AOV) across retail branches.
    # Established flagship stores have hundreds of transactions with means near ~$80.
    # Brand new popup locations have only 2-5 transactions, causing extreme
    # sample means (e.g. $125 or $42) due to small sample noise.
    # -------------------------------------------------------------------------
    np.random.seed(42)
    branches = {
        'Downtown_Flagship': np.random.normal(loc=82.0, scale=18.0, size=500),
        'Metro_Mall':        np.random.normal(loc=79.0, scale=17.0, size=350),
        'Westside_Center':   np.random.normal(loc=84.0, scale=19.0, size=200),
        'Suburban_Plaza':    np.random.normal(loc=76.0, scale=16.0, size=120),
        'Airport_Terminal':  np.random.normal(loc=88.0, scale=22.0, size=75),
        'University_Campus': np.random.normal(loc=72.0, scale=15.0, size=40),
        'Subway_Kiosk':      np.random.normal(loc=74.0, scale=18.0, size=15),
        'Station_PopUp':     np.array([125.0, 130.0, 120.0]),                   # n=3, raw mean = $125.0
        'Express_Cart':      np.array([40.0, 44.0]),                            # n=2, raw mean = $42.0
        'Harbor_Event':      np.array([95.0, 105.0]),                           # n=2, raw mean = $100.0
    }

    series_dict = {
        name: Ratio(pd.Series(arr))
        for name, arr in branches.items()
    }

    ratio_series = RatioSeries(series_dict)
    ratio_series.data.name = 'order_value'
    ratio_series.data.index.name = 'branch'

    # -------------------------------------------------------------------------
    # 2. Fit Empirical Bayes Normal Prior
    # -------------------------------------------------------------------------
    prior_mle = ratio_series.empirical_bayes_prior(method='mle')
    prior_mom = ratio_series.empirical_bayes_prior(method='moments')

    print(f"\nEmpirical Bayes Prior (MLE):     Normal(mu={prior_mle.mu:.3f}, sigma={prior_mle.sigma:.3f})")
    print(f"Prior Grand Mean AOV:            ${prior_mle.mu:.2f}")
    print(f"Prior 95% HDI:                   ${prior_mle.interval(0.95)[0]:.2f} to ${prior_mle.interval(0.95)[1]:.2f}")
    print(f"Empirical Bayes Prior (Moments): Normal(mu={prior_mom.mu:.3f}, sigma={prior_mom.sigma:.3f})")

    # -------------------------------------------------------------------------
    # 3. Shrinkage Summary Table
    # -------------------------------------------------------------------------
    summary_df = ratio_series.shrinkage_summary(hdi=0.95, prior=prior_mle)
    print("\n--- Empirical Bayes Normal-Normal Shrinkage Summary ---")
    display_cols = ['n', 'raw_mean', 'raw_se', 'shrunk_mean', 'shrunk_std', '95% HDI lower', '95% HDI upper', 'weight_data']
    print(summary_df[display_cols].round(3).to_string())

    # Highlight shrinkage for small popup stores
    popup_raw = summary_df.loc['Station_PopUp', 'raw_mean']
    popup_shrunk = summary_df.loc['Station_PopUp', 'shrunk_mean']
    popup_w = summary_df.loc['Station_PopUp', 'weight_data']
    print(f"\nHighlight - Station_PopUp (n=3):")
    print(f"  Raw Sample Mean:  ${popup_raw:.2f}")
    print(f"  Shrunk Posterior: ${popup_shrunk:.2f} (pulled down toward prior grand mean ${prior_mle.mu:.2f})")
    print(f"  Data Weight:      {popup_w:.1%} (Prior weight = {1.0 - popup_w:.1%})")

    cart_raw = summary_df.loc['Express_Cart', 'raw_mean']
    cart_shrunk = summary_df.loc['Express_Cart', 'shrunk_mean']
    cart_w = summary_df.loc['Express_Cart', 'weight_data']
    print(f"\nHighlight - Express_Cart (n=2):")
    print(f"  Raw Sample Mean:  ${cart_raw:.2f}")
    print(f"  Shrunk Posterior: ${cart_shrunk:.2f} (pulled up toward prior grand mean ${prior_mle.mu:.2f})")
    print(f"  Data Weight:      {cart_w:.1%} (Prior weight = {1.0 - cart_w:.1%})")

    flagship_raw = summary_df.loc['Downtown_Flagship', 'raw_mean']
    flagship_shrunk = summary_df.loc['Downtown_Flagship', 'shrunk_mean']
    flagship_w = summary_df.loc['Downtown_Flagship', 'weight_data']
    print(f"\nHighlight - Downtown_Flagship (n=500):")
    print(f"  Raw Sample Mean:  ${flagship_raw:.2f}")
    print(f"  Shrunk Posterior: ${flagship_shrunk:.2f} (retains sample data)")
    print(f"  Data Weight:      {flagship_w:.1%} (dominated by data)")

    # -------------------------------------------------------------------------
    # 4. Visualization: plot_shrinkage()
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ratio_series.plot_shrinkage(
        ax=ax,
        hdi=0.95,
        prior=prior_mle,
        order_by='raw_mean',
        title="Empirical Bayes Normal-Normal Shrinkage Across Retail Branches"
    )
    plt.tight_layout()
    print("\nPlot generated successfully.")


if __name__ == "__main__":
    main()
