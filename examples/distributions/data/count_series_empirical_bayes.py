import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from probability.distributions import Count, CountSeries


def main():
    print("=" * 75)
    print("EMPIRICAL BAYES GAMMA-POISSON SHRINKAGE ON COUNTSERIES")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # 1. Simulating 10 software modules with varying observation windows
    # Scenario: Bug / error report counts observed across microservices.
    # Some new or niche services have very few hours of log exposure (1-5 hours),
    # leading to volatile raw error rates (e.g. 5.0 or 0.0 errors/hr).
    # Established services have hundreds of hours with stable rates around ~1.5/hr.
    # -------------------------------------------------------------------------
    np.random.seed(42)
    services = {
        'Auth_Service':     {'counts': [1, 2, 1, 1, 2, 0, 1, 2] * 50},  # 400 hrs, rate ~ 1.25
        'Billing_API':      {'counts': [2, 1, 3, 2, 1, 2, 1, 1] * 35},  # 280 hrs, rate ~ 1.63
        'Checkout_Cart':    {'counts': [1, 1, 2, 0, 1, 2] * 25},        # 150 hrs, rate ~ 1.17
        'Search_Indexer':   {'counts': [3, 2, 1, 4, 2, 3] * 15},        # 90 hrs, rate ~ 2.50
        'Email_Notifier':   {'counts': [0, 1, 1, 0, 2, 1] * 10},        # 60 hrs, rate ~ 0.83
        'Recommendations':  {'counts': [2, 1, 0, 1, 3] * 4},            # 20 hrs, rate ~ 1.40
        'Legacy_Importer':  {'counts': [0, 1, 0, 0, 1]},                # 5 hrs, rate ~ 0.40
        'Export_Worker':    {'counts': [4, 5]},                          # 2 hrs, rate ~ 4.50 (extreme!)
        'Beta_Feature_X':   {'counts': [0]},                             # 1 hr, rate ~ 0.00 (extreme!)
        'Mobile_Sync':      {'counts': [3]},                             # 1 hr, rate ~ 3.00
    }

    series_dict = {
        name: Count(pd.Series(info['counts']))
        for name, info in services.items()
    }

    count_series = CountSeries(series_dict)
    count_series.data.name = 'errors'
    count_series.data.index.name = 'service'

    # -------------------------------------------------------------------------
    # 2. Fit Empirical Bayes Gamma Prior
    # -------------------------------------------------------------------------
    prior_mle = count_series.empirical_bayes_prior(method='mle')
    prior_mom = count_series.empirical_bayes_prior(method='moments')

    print(f"\nEmpirical Bayes Prior (MLE):     Gamma(alpha={prior_mle.alpha:.3f}, beta={prior_mle.beta:.3f})")
    print(f"Prior Grand Mean Rate:           {prior_mle.mean():.4f} errors/hr")
    print(f"Prior 95% HDI:                   {prior_mle.hdi(0.95)}")
    print(f"Empirical Bayes Prior (Moments): Gamma(alpha={prior_mom.alpha:.3f}, beta={prior_mom.beta:.3f})")

    # -------------------------------------------------------------------------
    # 3. Shrinkage Summary Table
    # -------------------------------------------------------------------------
    summary_df = count_series.shrinkage_summary(hdi=0.95, prior=prior_mle)
    print("\n--- Empirical Bayes Gamma-Poisson Shrinkage Summary ---")
    display_cols = ['exposure', 'total_count', 'raw_rate', 'shrunk_mean', '95% HDI lower', '95% HDI upper', 'weight_data']
    print(summary_df[display_cols].round(4).to_string())

    # Highlight shrinkage for tiny-sample Export_Worker (n=2, raw=4.50):
    exp_raw = summary_df.loc['Export_Worker', 'raw_rate']
    exp_shrunk = summary_df.loc['Export_Worker', 'shrunk_mean']
    exp_w = summary_df.loc['Export_Worker', 'weight_data']
    print(f"\nHighlight - Export_Worker:")
    print(f"  Raw Rate:     {exp_raw:.2f} errors/hr")
    print(f"  Shrunk Rate:  {exp_shrunk:.2f} errors/hr (pulled down toward {prior_mle.mean():.2f})")
    print(f"  Data Weight:  {exp_w:.1%} (Prior weight = {1.0 - exp_w:.1%})")

    # -------------------------------------------------------------------------
    # 4. Negative Binomial Predictive Distributions
    # -------------------------------------------------------------------------
    preds = count_series.predictive_distributions(prior=prior_mle, time_exposure=10.0)
    print("\n--- Future 10-Hour Error Predictions (Negative Binomial) ---")
    for name in ['Auth_Service', 'Export_Worker', 'Beta_Feature_X']:
        dist = preds[name]
        mean_pred = dist.mean()
        print(f"  {name:16s}: E[Y_10hr] = {mean_pred:.1f} errors")

    # -------------------------------------------------------------------------
    # 5. Visualization: plot_shrinkage()
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    count_series.plot_shrinkage(
        ax=ax,
        hdi=0.95,
        prior=prior_mle,
        order_by='raw_rate',
        title="Empirical Bayes Gamma-Poisson Shrinkage Across Microservices"
    )
    plt.tight_layout()
    print("\nPlot generated successfully.")


if __name__ == "__main__":
    main()
