from pandas import Series

from examples.pandas.create_data import get_fruit_box_data
from probability.pandas.dataset import DataSet
from probability.pandas.discrete_distribution import DiscreteDistribution

data = get_fruit_box_data()
dataset = DataSet(data)

joint = dataset.joint()
marginal_box = joint.marginalize('box')
marginal_fruit = joint.marginalize('fruit')
conditional_box = joint.condition('box')
prior_data = Series(data={'blue': 0.6, 'red': 0.4})
prior_data.index.name = 'box'
prior = DiscreteDistribution(data=prior_data)
test1 = conditional_box * prior
