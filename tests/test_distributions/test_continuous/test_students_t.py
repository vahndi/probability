from unittest.case import TestCase

from probability.distributions import StudentsT


class TestStudentsT(TestCase):

    def setUp(self) -> None:

        pass

    def test_fit(self):

        for nu in (1, 2, 5):
            students_t_orig = StudentsT(nu=nu)
            students_t_fit = StudentsT.fit(
                students_t_orig.rvs(100_000, random_state=42)
            )
            self.assertAlmostEqual(students_t_orig.nu,
                                   students_t_fit.nu, delta=0.1)
            self.assertAlmostEqual(students_t_orig.mu,
                                   students_t_fit.mu, delta=0.1)
            self.assertAlmostEqual(students_t_orig.sigma,
                                   students_t_fit.sigma, delta=0.1)

    def test_sigma_sq_and_mode(self):

        st = StudentsT(nu=5, mu=3.5, sigma_sq=4.0)
        self.assertEqual(2.0, st.sigma)
        self.assertEqual(4.0, st.sigma_sq)
        self.assertEqual(3.5, st.mode())

