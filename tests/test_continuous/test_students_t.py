from unittest.case import TestCase

from probability.distributions import StudentsT


class TestStudentsT(TestCase):

    def setUp(self) -> None:

        pass

    def test_fit(self):

        for nu in (1, 2, 5):
            students_t_orig = StudentsT(nu=nu)
            students_t_fit = StudentsT.fit(students_t_orig.rvs(100_000))
            self.assertAlmostEqual(students_t_orig.nu,
                                   students_t_fit.nu, delta=0.1)
            self.assertAlmostEqual(students_t_orig.mu,
                                   students_t_fit.mu, delta=0.1)
            self.assertAlmostEqual(students_t_orig.sigma,
                                   students_t_fit.sigma, delta=0.1)
