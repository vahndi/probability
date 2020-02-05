from probability.pandas import DiscreteDistribution
from tests.shared import read_distribution_data, series_are_equivalent
p_abcd = DiscreteDistribution(read_distribution_data('P(A,B,C,D)'))

p1 = p_abcd.given(A=1, B=2)
