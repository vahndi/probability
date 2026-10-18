from unittest import TestCase

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from probability.distributions import (
    BetaBinomialConjugate, BetaGeometricConjugate, Binomial, Boolean,
    Count, Dirichlet, GammaExponentialConjugate, Nominal, Normal,
    Ordinal, PERT, Ratio
)
from probability.distributions.conjugate.dirichlet_multinomial_conjugate import (
    DirichletMultinomialConjugate
)
from probability.distributions.conjugate.gamma_poisson_conjugate import (
    GammaPoissonConjugate
)
from probability.distributions.conjugate.normal_normal_conjugate import (
    NormalNormalConjugate
)
from probability.models.decision_tree.actions import Action, ActionsGroup
from probability.models.project_network.project_network import ProjectNetwork
from probability.models.project_network.project_task import ProjectTask
from probability.plotting import (
    add_h_density, add_rectangle, add_text, add_v_density,
    cross_fade, format_as_integer, format_as_percent,
    format_axis_integer, format_axis_percent,
    new_axes, new_subplots, set_alpha
)


class TestPlottingUtils(TestCase):

    def tearDown(self):
        plt.close('all')

    def test_color_utils(self):
        c = cross_fade('red', 'blue', 0.5)
        self.assertEqual(len(c), 4)
        self.assertAlmostEqual(c[0], 0.5)
        self.assertAlmostEqual(c[2], 0.5)

        rgba = set_alpha('green', 0.25)
        self.assertEqual(len(rgba), 4)
        self.assertAlmostEqual(rgba[3], 0.25)

    def test_number_utils(self):
        self.assertEqual(format_as_percent(0.1234, 1), '12.3%')
        self.assertEqual(format_as_integer(1234567), '1,234,567')

    def test_axes_utils(self):
        ax = new_axes(width=8, height=4)
        self.assertIsInstance(ax, Axes)

        fig, axes = new_subplots(n_rows=2, n_cols=2)
        self.assertIsInstance(fig, Figure)
        self.assertEqual(axes.shape, (2, 2))

        ret_ax = add_rectangle(ax, width=1.0, height=2.0, x_left=0.0, y_bottom=0.0)
        self.assertIs(ret_ax, ax)
        self.assertEqual(len(ax.patches), 1)

        ret_ax2 = add_text(ax, x=0.5, y=1.0, text='hello', bbox_face_color='white')
        self.assertIs(ret_ax2, ax)
        self.assertEqual(len(ax.texts), 1)

        format_axis_integer(ax.xaxis)
        format_axis_percent(ax.yaxis, num_decimals=1)

        dist = Normal(0, 1)
        y_vals = np.linspace(-2, 2, 21)
        s = dist.pdf().at(y_vals)
        add_v_density(ax, x=0, y_to_z=s, width=0.5)
        add_h_density(ax, y=0, x_to_z=s, height=0.5)


class TestDistributionAndModelPlotting(TestCase):

    def tearDown(self):
        plt.close('all')

    def test_continuous_1d_and_nd_plotting(self):
        norm = Normal(0, 1)
        ax = norm.plot(mean=True, median=True, std=True)
        self.assertIsInstance(ax, Axes)
        x = np.linspace(-3, 3, 50)
        ax2 = norm.pdf().plot(x=x, ax=ax)
        self.assertIs(ax2, ax)
        ax3 = norm.cdf().plot(x=x)
        self.assertIsInstance(ax3, Axes)

        dirichlet = Dirichlet([2, 3, 4])
        ax_d = dirichlet.plot(x=np.linspace(0, 1, 50))
        self.assertIsInstance(ax_d, Axes)
        ax_s = dirichlet.pdf().plot_simplex()
        self.assertIsInstance(ax_s, Axes)

    def test_discrete_1d_plotting(self):
        binom = Binomial(10, 0.4)
        ax = binom.plot(mean=True, median=True, std=True)
        self.assertIsInstance(ax, Axes)
        ax2 = binom.pmf().plot(k=list(range(11)), kind='line')
        self.assertIsInstance(ax2, Axes)

    def test_data_distributions_plotting(self):
        b = Boolean(pd.Series([True, False, True, True, False], name='flag'))
        cat = Nominal(pd.Series(['A', 'A', 'B', 'B', 'B'], name='group'))
        self.assertIsInstance(b.plot_bars(), Axes)
        self.assertIsInstance(b.plot_conditional_prob_densities(categorical=cat), Axes)

        bs = b | cat
        self.assertIsInstance(bs.plot_prob_densities(), Axes)

        c = Count(pd.Series([1, 2, 2, 3, 1, 2], name='cnt'))
        self.assertIsInstance(c.plot_bars(), Axes)
        cs = c | cat
        self.assertIsInstance(cs.plot_bars(), Axes)

        o = Ordinal(
            pd.Series(pd.Categorical([1, 2, 3, 2, 1], categories=[1, 2, 3], ordered=True), name='ord')
        )
        self.assertIsInstance(o.plot_bars(), Axes)
        os = o | cat
        self.assertIsInstance(os.plot_density_grid(), Axes)

        r = Ratio(pd.Series([1.0, 2.5, 3.2, 4.1, 2.8, 3.6], name='val'))
        self.assertIsInstance(r.plot_hist(), Axes)
        self.assertIsInstance(r.plot_kde(mean_line=True, median_line=True), Axes)
        self.assertIsInstance(r.plot_conditional_dist_densities(categorical=cat, fit_dist=Normal), Axes)

        rs = r | cat
        self.assertIsInstance(rs.plot_density_bars(bin_spacing=1.0), Axes)
        self.assertIsInstance(rs.plot_density_distributions(), Axes)

    def test_conjugate_plotting(self):
        fig = BetaBinomialConjugate(alpha=2, beta=2, n=10, k=6).plot(n_=10)
        self.assertIsInstance(fig, Figure)

        fig = BetaGeometricConjugate(alpha=2, beta=2, n=5, k=12).plot()
        self.assertIsInstance(fig, Figure)

        fig = DirichletMultinomialConjugate(
            alpha=[2, 3], x=[5, 7]
        ).plot()
        self.assertIsInstance(fig, Figure)

        fig = GammaExponentialConjugate(alpha=2, beta=2, n=5, x_mean=2.0).plot()
        self.assertIsInstance(fig, Figure)

        fig = GammaPoissonConjugate(alpha=2, beta=2, n=5, k=10).plot()
        self.assertIsInstance(fig, Figure)

        fig = NormalNormalConjugate(
            n=5, x_sum=2.5, mu_0=0, sigma_sq=1, sigma_0_sq=1
        ).plot()
        self.assertIsInstance(fig, Figure)

    def test_models_plotting(self):
        a1 = Action(name='A1', p_success=0.6, init_cost=50, duration=0)
        a2 = Action(name='A2', p_success=1.0, init_cost=100, duration=0)
        ag = ActionsGroup(a1, a2, can_try_action=lambda a, y: y + a.duration <= 3)
        dt = ag.make_tree(max_depth=2)
        dt.solve()
        ax = dt.draw()
        self.assertIsInstance(ax, Axes)

        net = ProjectNetwork()
        t1 = ProjectTask('A', PERT(1, 2, 3))
        t2 = ProjectTask('B', PERT(2, 4, 6))
        net.add_task(t1)
        net.add_task(t2, parents='A', end=True)
        net.add_percentiles(0.95)
        ax_pn = net.draw()
        self.assertIsInstance(ax_pn, Axes)
