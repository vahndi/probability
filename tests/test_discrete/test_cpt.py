from unittest.case import TestCase
from itertools import product
from pandas import DataFrame, MultiIndex, Index

from probability.discrete.cpt import CPT


class TestCPT(TestCase):

    def setUp(self) -> None:

        self.data = DataFrame(
            data=[[0.1, 0.1, 0.1, 0.2, 0.2, 0.2],
                  [0.1, 0.2, 0.3, 0.2, 0.3, 0.4],
                  [0.8, 0.7, 0.6, 0.6, 0.5, 0.4]],
            columns=MultiIndex.from_tuples(
                tuples=product(['easy', 'hard'], ['low', 'medium', 'high']),
                names=['difficulty', 'intelligence']
            ),
            index=Index(data=['C', 'B', 'A'], name='grade')
        )
        self.cpt = CPT.from_data_frame(data=self.data)

    def test_variable(self):

        self.assertEqual(self.cpt.variable, 'grade')