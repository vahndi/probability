from pandas import DataFrame


def make_cookies_observations() -> DataFrame:

    return DataFrame({
        'bowl': ['bowl 1'] * 40 + ['bowl 2'] * 40,
        'flavor': (
            ['vanilla'] * 30 + ['chocolate'] * 10 +
            ['vanilla'] * 20 + ['chocolate'] * 20
        )
    })
