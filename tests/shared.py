from numpy import random
from pandas import ExcelFile, Series, read_excel, DataFrame

from tests.paths import FN_PANDAS_TESTS

xlsx = ExcelFile(str(FN_PANDAS_TESTS))


def read_distribution(sheet_name: str) -> Series:
    data = read_excel(xlsx, sheet_name)
    variables = [c for c in data.columns if c != 'p' and not c.startswith('_')]
    return data.set_index(variables)['p']


def make_joint_data():

    random.seed(123)
    return DataFrame(random.randint(1, 4, (100, 4)),
                     columns=['A', 'B', 'C', 'D'])


def get_joint_distribution(data_set: DataFrame) -> Series:
    return (
        data_set.groupby(data_set.columns.tolist()).size() / len(data_set)
    ).rename('p')


def series_are_equivalent(series_1: Series, series_2: Series) -> bool:
    """
    Determine if the 2 series share the same items.
    N.B. where series 1 has p(0), the associated item may be missing from series 2.
    """
    d1 = series_1.copy().reset_index()
    cols_1 = sorted([c for c in d1.columns if c != 'p'])
    cols_p = cols_1 + ['p']
    s1 = d1[cols_p].set_index(cols_1)['p']
    d2 = series_2.copy().reset_index()
    cols_2 = sorted([c for c in d2.columns if c != 'p'])
    if cols_1 != cols_2:
        return False
    s2 = d2[cols_p].set_index(cols_2)['p']
    for k, v in s1.iteritems():
        if v == 0:
            continue
        if k not in s2.keys() or abs(s2[k] - v) > 1e-10:
            return False
    return True
