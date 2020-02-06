from itertools import product
from unittest import TestCase

from pandas import DataFrame, Series

from probability.pandas import DiscreteDistribution


class TestExamples(TestCase):

    def test_fruit_box(self):
        fruit_box_data = DataFrame({
            'box': ['red'] * 8 + ['blue'] * 4,
            'fruit': ['apple'] * 2 + ['orange'] * 7 + ['apple'] * 3
        })
        # P(box,fruit)
        p_bf = DiscreteDistribution.from_observations(fruit_box_data)
        self.assertEqual(p_bf.name, 'P(box,fruit)')
        for box, fruit in product(['red', 'blue'], ['apple', 'orange']):
            data_frac = fruit_box_data.loc[
                (fruit_box_data['box'] == box) &
                (fruit_box_data['fruit'] == fruit)
            ].shape[0] / len(fruit_box_data)
            self.assertEqual(data_frac, p_bf.p(box=box, fruit=fruit))
        # P(box)
        p_b = DiscreteDistribution.from_dict(data={'blue': 0.6, 'red': 0.4}, var_names='box')
        self.assertEqual(p_b.name, 'P(box)')
        self.assertEqual(p_b.p(box='blue'), 0.6)
        self.assertEqual(p_b.p(box='red'), 0.4)
        # P(fruit|box)
        p_f__b = p_bf.condition('box')
        self.assertEqual(p_f__b.name, 'P(fruit|box)')
        p_fb = p_f__b * p_b
        self.assertEqual(p_fb.name, 'P(fruit,box)')
        p_f = p_fb.margin('fruit')
        self.assertEqual(p_f.name, 'P(fruit)')
        p_b__f = p_fb.condition('fruit')
        self.assertEqual(p_b__f.name, 'P(box|fruit)')
        p_b__f_orange = p_fb.given(fruit='orange')
        self.assertEqual(p_b__f_orange.name, 'P(box|fruit=orange)')

    def test_darts(self):

        p_region = DiscreteDistribution.from_dict({r: 1 / 20 for r in range(1, 21)}, var_names='region')
        self.assertEqual(p_region.name, 'P(region)')
        p_not_20 = p_region.given(region__ne=20)
        self.assertEqual(p_not_20.name, 'P(region|region≠20)')
        self.assertAlmostEqual((p_not_20.data - Series(1/19, index=range(1, 20))).abs().sum(), 0)
