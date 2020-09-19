from unittest.case import TestCase

from probability.distributions import Beta


class TestDistributionCalculation(TestCase):

    def setUp(self) -> None:

        b1 = Beta(700, 300)
        b2 = Beta(600, 400)
        b3 = Beta(500, 500)
        self.b1__mul__b2 = b1 * b2
        self.b3__mul__b1__mul__b2 = b3 * self.b1__mul__b2
        self.b1__mul__comp__b1 = b1 * (1 - b1)

    def test_b1__mul__b2_name(self):

        self.assertEqual('Beta(α=700, β=300) * Beta(α=600, β=400)',
                         self.b1__mul__b2.name)

    def test_b1__mul__b2_result(self):

        result = self.b1__mul__b2.execute()
        self.assertAlmostEqual(0.42, result.mean(), 2)
        self.assertAlmostEqual(0.014, result.std(), 3)

    def test_b3__mul__b1__mul__b2_name(self):

        self.assertEqual(
            'Beta(α=500, β=500) * (Beta(α=700, β=300) * Beta(α=600, β=400))',
            self.b3__mul__b1__mul__b2.name
        )

    def test_b3__mul__b1__mul__b2_result(self):

        result = self.b3__mul__b1__mul__b2.execute()
        self.assertAlmostEqual(0.21, result.mean(), 2)
        self.assertAlmostEqual(0.0096, result.std(), 4)

    def test_b1__mul__comp__b1_name(self):

        self.assertEqual(
            'Beta(α=700, β=300) * (1 - Beta(α=700, β=300))',
            self.b1__mul__comp__b1.name
        )

    def test_b1__mul__comp__b1_result(self):

        result = self.b1__mul__comp__b1.execute()
        self.assertAlmostEqual(0.21, result.mean(), 2)
        self.assertAlmostEqual(0.006, result.std(), 3)
