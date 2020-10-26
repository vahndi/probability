from unittest.case import TestCase

from probability.discrete import Discrete, Conditional


class TestChapter03(TestCase):

    def setUp(self) -> None:

        self.r = Discrete.binary(0.2, 'rain')
        self.s = Discrete.binary(0.1, 'sprinkler')
        self.j__r = Conditional.from_probs({
                (1, 1): 1,
                (1, 0): 0.2,
                (0, 1): 0,
                (0, 0): 0.8
            },
            joint_variables='jack',
            conditional_variables='rain'
        )
        self.t__r_s = Conditional.from_probs({
                (1, 1, 0): 1,
                (1, 1, 1): 1,
                (1, 0, 1): 0.9,
                (1, 0, 0): 0,
                (0, 1, 0): 0,
                (0, 1, 1): 0,
                (0, 0, 1): 0.1,
                (0, 0, 0): 1
            },
            joint_variables='tracey',
            conditional_variables=['rain', 'sprinkler']
        )

    def test__3_1_11(self):

        r_s = self.r * self.s
        r_s_t = self.t__r_s * r_s
        s__t = r_s_t.given(tracey=1).p(sprinkler=1)
        self.assertAlmostEqual(0.3382, s__t, 4)

    def test__3_1_15(self):

        r_s = self.r * self.s
        j_t__r_s = self.j__r * self.t__r_s
        j_r_s_t = j_t__r_s * r_s
        j_s_t = j_r_s_t.marginal('jack', 'sprinkler', 'tracey')
        s__t1_j1 = j_s_t.given(tracey=1, jack=1).p(sprinkler=1)
        self.assertAlmostEqual(0.1604, s__t1_j1, 4)
