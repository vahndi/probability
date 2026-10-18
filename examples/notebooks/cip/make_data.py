from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from pandas import DataFrame

DATA_DIR = Path(__file__).resolve().parent / 'data'
GITHUB_DATA_URL = (
    'https://raw.githubusercontent.com/matheusfacure/'
    'causal-inference-in-python-code/main/causal-inference-in-python/data'
)


def _load_csv(filename: str) -> DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    local_path = DATA_DIR / filename
    if not local_path.exists():
        urlretrieve(f'{GITHUB_DATA_URL}/{filename}', local_path)
    return pd.read_csv(local_path)


def load_xmas_sales() -> DataFrame:
    """
    Load the Chapter 1 xmas_sales.csv dataset.
    """
    return _load_csv('xmas_sales.csv')


def make_potential_outcomes_table(observed_only: bool = False) -> DataFrame:
    """
    Create the Chapter 1 toy potential-outcomes table.
    """
    return DataFrame({
        'i': [1, 2, 3, 4, 5, 6],
        'y0': [200, 120, 300, np.nan, np.nan, np.nan] if observed_only
        else [200, 120, 300, 450, 600, 600],
        'y1': [np.nan, np.nan, np.nan, 500, 600, 800] if observed_only
        else [220, 140, 400, 500, 600, 800],
        't': [0, 0, 0, 1, 1, 1],
        'x': [0, 0, 1, 0, 0, 1],
    })


def load_cross_sell_email() -> DataFrame:
    """
    Load the Chapter 2 cross_sell_email.csv dataset.
    """
    return _load_csv('cross_sell_email.csv')


def load_enem_scores() -> DataFrame:
    """
    Load the Chapter 2 enem_scores.csv dataset.
    """
    return _load_csv('enem_scores.csv')


def load_wage() -> DataFrame:
    """
    Load the Chapter 3 wage.csv dataset.
    """
    return _load_csv('wage.csv')


def load_management_training() -> DataFrame:
    """
    Load the Chapter 5 management_training.csv dataset.
    """
    return _load_csv('management_training.csv')


