from unittest.case import TestCase

from probability.discrete import Discrete, Conditional


class TestChapter03(TestCase):

    def setUp(self) -> None:

        pass

    def test__3_1_1(self):

        r = Discrete.binary(0.2, 'rain')
        s = Discrete.binary(0.1, 'sprinkler')
        j__r = Conditional.from_probs({
                (1, 1): 1,
                (1, 0): 0.2
            },
            joint_variables='jack',
            conditional_variables='rain'
        )
        t__r_s = Conditional.from_probs({
                (1, 1, 0): 1,
                (1, 1, 1): 1,
                (1, 0, 1): 0.9,
                (1, 0, 0): 0
            },
            joint_variables='tracey',
            conditional_variables=['rain', 'sprinkler']
        )
        r_s = r * s
        r_s_t = t__r_s * r_s
        s__t = r_s_t.given(tracey=1).p(sprinkler=1)
        self.assertAlmostEqual(0.3382, s__t, 4)
