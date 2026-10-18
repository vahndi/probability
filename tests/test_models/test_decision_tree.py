from unittest.case import TestCase
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame

from probability.distributions.continuous.normal import Normal
from probability.models.decision_tree import DecisionTree
from probability.models.decision_tree.actions import Action, ActionsGroup
from probability.models.decision_tree.nodes import ChanceNode, AmountNode

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

    def test_distribution_valued_amounts(self):
        dt = DecisionTree()
        dt.add_decision(name='D1')
        # Amount is a Normal distribution with mean 50, std 5
        dt.add_option(name='D1.a', p_success=0.8, amount=Normal(50, 5), parent_name='D1', final=True)
        # Amount is a scalar 40
        dt.add_option(name='D1.b', p_success=0.8, amount=40, parent_name='D1', final=True)

        dt.solve(minimize=True)
        # For D1.a: expected cost is 50.0
        self.assertAlmostEqual(dt.chance_node('D1.a').expected_amount, 50.0)
        # For D1.b: expected cost is 40.0
        self.assertAlmostEqual(dt.chance_node('D1.b').expected_amount, 40.0)
        # Best decision is min(50.0, 40.0) = 40.0
        self.assertAlmostEqual(dt.decision_node('D1').expected_amount, 40.0)

    def test_risk_profile(self):
        dt = DecisionTree()
        dt.add_decision(name='D1')
        dt.add_option(name='D1.a', p_success=0.7, amount=50, parent_name='D1')
        dt.add_decision(name='D1.a.D2', parent_name='D1.a')
        dt.add_option(name='D1.a.D2.a', p_success=0.8, amount=60, parent_name='D1.a.D2', final=True)
        dt.add_option(name='D1.a.D2.b', p_success=0.6, amount=25, parent_name='D1.a.D2', final=True)
        dt.solve(minimize=True)

        rp = dt.risk_profile('D1.a', policy='optimal')
        self.assertEqual(len(rp), 3)
        self.assertAlmostEqual(rp['probability'].sum(), 1.0)
        # Total expected cost calculated from risk profile must equal solved expected amount:
        # 0.7 * 50 + 0.3 * (0.6 * 75 + 0.4 * 75) = 35 + 22.5 = 57.5
        expected_cost = (rp['probability'] * rp['amount']).sum()
        self.assertAlmostEqual(expected_cost, dt.chance_node('D1.a').expected_amount)
        # Cumulative probabilities must be non-decreasing
        self.assertTrue((rp['cumulative_probability'].diff().dropna() >= 0).all())

    def test_plot_risk_profile(self):
        dt = DecisionTree()
        dt.add_decision(name='D1')
        dt.add_option(name='D1.a', p_success=0.7, amount=50, parent_name='D1', final=True)
        dt.add_option(name='D1.b', p_success=0.5, amount=20, parent_name='D1', final=True)
        dt.solve(minimize=True)

        fig, ax = plt.subplots()
        ret_ax = dt.plot_risk_profile(ax=ax)
        self.assertIs(ret_ax, ax)
        plt.close(fig)

    def test_calculate_evpi_clemen_oil_drilling(self):
        # Clemen textbook example: Oil drilling
        # Actions: Drill, Don't Drill
        # States: Dry (0.5), Wet (0.3), Soaking (0.2)
        # Drill payoffs: Dry: -100, Wet: 150, Soaking: 500
        # Don't Drill: 0, 0, 0
        payoffs = {
            'Drill': {'Dry': -100.0, 'Wet': 150.0, 'Soaking': 500.0},
            'DontDrill': {'Dry': 0.0, 'Wet': 0.0, 'Soaking': 0.0}
        }
        probs = {'Dry': 0.5, 'Wet': 0.3, 'Soaking': 0.2}

        # EMV(Drill) = 0.5(-100) + 0.3(150) + 0.2(500) = -50 + 45 + 100 = 95.0
        # EMV(DontDrill) = 0.0
        # EMV* = 95.0
        # EV|PI = 0.5(0) + 0.3(150) + 0.2(500) = 45 + 100 = 145.0
        # EVPI = 145.0 - 95.0 = 50.0
        evpi = DecisionTree.calculate_evpi(payoffs, probs, minimize=False)
        self.assertAlmostEqual(evpi, 50.0)

    def test_evpi_from_tree(self):
        # Represent the oil drilling problem directly in DecisionTree
        dt = DecisionTree()
        dt.add_decision('DrillDecision')
        d = dt.decision_node('DrillDecision')

        c_drill = dt.add_chance_node(ChanceNode('Drill', p_success=1.0, amount=0, depth=1), parent=d)
        dt.add_amount_node(AmountNode('Drill.Dry', probability=0.5, amount=-100, depth=1), parent=c_drill)
        dt.add_amount_node(AmountNode('Drill.Wet', probability=0.3, amount=150, depth=1), parent=c_drill)
        dt.add_amount_node(AmountNode('Drill.Soaking', probability=0.2, amount=500, depth=1), parent=c_drill)

        c_dont = dt.add_chance_node(ChanceNode('DontDrill', p_success=1.0, amount=0, depth=1), parent=d)
        dt.add_amount_node(AmountNode('DontDrill.Dry', probability=0.5, amount=0, depth=1), parent=c_dont)
        dt.add_amount_node(AmountNode('DontDrill.Wet', probability=0.3, amount=0, depth=1), parent=c_dont)
        dt.add_amount_node(AmountNode('DontDrill.Soaking', probability=0.2, amount=0, depth=1), parent=c_dont)

        dt.solve(minimize=False)
        self.assertAlmostEqual(d.expected_amount, 95.0)

        tree_evpi = dt.evpi(minimize=False)
        self.assertAlmostEqual(tree_evpi, 50.0)
