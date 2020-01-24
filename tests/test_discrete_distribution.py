from numpy import random
from pandas import DataFrame, Series

from probability.pandas.prob_utils import margin, condition


def make_joint_data():

    random.seed(123)
    return DataFrame(random.randint(1, 4, (100, 4)),
                     columns=['A', 'B', 'C', 'D'])


def get_joint_distribution(data_set: DataFrame) -> Series:
    return (
        data_set.groupby(data_set.columns.tolist()).size() / len(data_set)
    ).rename('p')


if __name__ == '__main__':

    joint_data = make_joint_data()
    print(joint_data)
    joint_dist = get_joint_distribution(joint_data)
    print(joint_dist)
    mA = margin(joint_dist, 'C', 'D')
    print(mA)
    condA = condition(joint_dist, 'C', 'D')
    print(condA)
