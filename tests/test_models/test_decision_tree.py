from unittest.case import TestCase

from pandas import DataFrame

from probability.models.decision_tree.actions import Action, ActionsGroup

MAX_YEAR = 2
ANNUAL_INFLATION = 1.2


def can_try(action: Action, year: int) -> bool:

    return year + action.duration <= 3


class TestDecisionTree(TestCase):

    def setUp(self) -> None:

        XNat = Action(
            name='X-NAT',
            p_success=2 / 3,
            init_cost=50,
            duration=0,
            cost_inflation=ANNUAL_INFLATION
        )
        XIvf = Action(
            name='X-IVF',
            p_success=1,
            init_cost=100,
            duration=0,
            cost_inflation=ANNUAL_INFLATION
        )
        MCap = Action(
            name='M-CAP',
            p_success=1 / 2,
            init_cost=10,
            duration=2,
            cost_inflation=ANNUAL_INFLATION
        )
        self.actions_group = ActionsGroup(
            XNat, XIvf, MCap,
            can_try_action=can_try
        )

    def _get_amounts(self, max_depth: int) -> DataFrame:

        dt = self.actions_group.make_tree(max_depth=max_depth)
        dt.solve()
        amounts = dt.amounts(require_success=True)
        return amounts

    def test_build_tree_from_actions_group(self):

        costs = self._get_amounts(max_depth=3)
        unique_expected_costs = sorted(costs['expected_amount_1'].unique())
        expected = [64.0, 78.0, 86.0, 100.0, 108.0, 110.0, 150.0, 160.0]
        for exp, act in zip(expected, unique_expected_costs):
            self.assertAlmostEqual(exp, act)
        self.assertEqual(len(costs), 28)
