import numpy as np
from pandas import DataFrame


def make_cookies_observations() -> DataFrame:
    return DataFrame({
        'bowl': ['bowl 1'] * 40 + ['bowl 2'] * 40,
        'flavor': (
            ['vanilla'] * 30 + ['chocolate'] * 10 +
            ['vanilla'] * 20 + ['chocolate'] * 20
        )
    })


def make_euro_coin_data() -> dict:
    """The Euro Problem (Think Bayes Ch 3): 140 heads in 250 spins."""
    return {'k': 140, 'n': 250}


def make_world_cup_data() -> dict:
    """The World Cup Problem (Think Bayes Ch 5): Goals scored in tournament matches."""
    return {
        'france_goals': [2, 1, 0, 4, 2, 1],   # 6 matches before final
        'croatia_goals': [2, 3, 2, 1, 2, 2],  # 6 matches before final
    }


def make_height_sample_data(n: int = 50, seed: int = 42) -> np.ndarray:
    """Normal parameter estimation (Think Bayes Ch 15): sample of adult heights in cm."""
    np.random.seed(seed)
    return np.random.normal(loc=175.0, scale=7.5, size=n)


def make_ab_test_data() -> dict:
    """Comparing proportions (Think Bayes Ch 4): A/B testing two variants."""
    return {
        'variant_a': {'k': 140, 'n': 250},
        'variant_b': {'k': 120, 'n': 250},
    }
