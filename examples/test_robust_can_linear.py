import logging
import toppra
import numpy as np

def accel_constraint():
  dof = 5
  np.random.seed(0)
  alim_ = np.random.rand(5)
  alim = np.vstack((-alim_, alim_)).T
  constraint = toppra.constraint.JointAccelerationConstraint(alim)

  np.random.seed(0)
  path = toppra.SplineInterpolator(np.linspace(0, 1, 5), np.random.randn(5, dof))
  return constraint, path

def test_basic(dist_scheme):
    "Basic initialization."
    cnst, path = accel_constraint()

    ro_cnst = toppra.constraint.RobustLinearConstraint(cnst, [0.1, 2, .3], dist_scheme)

    assert ro_cnst.get_constraint_type() == toppra.constraint.ConstraintType.CanonicalConic
    assert ro_cnst.get_dof() == 5

    a, b, c, P, _, _ = ro_cnst.compute_constraint_params(
        path, np.linspace(0, path.duration, 10))

    # assert a.shape == (10, 2 * path.get_dof())
    # assert b.shape == (10, 2 * path.get_dof())
    # assert c.shape == (10, 2 * path.get_dof())
    # assert P.shape == (10, 2 * path.get_dof(), 3, 3)

    # Linear params
    cnst.set_discretization_type(dist_scheme)
    a0, b0, c0, F0, g0, _, _ = cnst.compute_constraint_params(
        path, np.linspace(0, path.duration, 10))

    # Assert values
    for i in range(10):
        np.testing.assert_allclose(a[i], F0.dot(a0[i]))
        np.testing.assert_allclose(b[i], F0.dot(b0[i]))
        np.testing.assert_allclose(c[i], F0.dot(c0[i]) - g0)
    for i in range(10):
        for j in range(a0.shape[1]):
            np.testing.assert_allclose(P[i, j], np.diag([0.1, 2, .3]))


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format="%(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
  dist_scheme = toppra.constraint.DiscretizationType.Collocation
  test_basic(dist_scheme)


