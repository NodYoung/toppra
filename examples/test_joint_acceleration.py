import os, sys
sys.path.insert(0, os.path.join('..'))
import logging
import numpy as np
import numpy.testing as npt
import toppra as ta
import toppra.constraint as constraint
from toppra.constants import JACC_MAXU

ta.setup_logging("DEBUG")
logger = logging.getLogger('toppra')

def accel_constraint_setup(dof):
  if dof == 1:  # Scalar
    pi = ta.PolynomialPath([1, 2, 3])  # 1 + 2s + 3s^2
    ss = np.linspace(0, 1, 3)
    alim = (np.r_[-1., 1]).reshape(1, 2)  # Scalar case
    accel_const = constraint.JointAccelerationConstraint(alim, constraint.DiscretizationType.Collocation)
    data = (pi, ss, alim)
    return data, accel_const

  if dof == 2:
    coeff = [[1., 2, 3], [-2., -3., 4., 5.]]
    pi = ta.PolynomialPath(coeff)
    ss = np.linspace(0, 0.75, 4)
    alim = np.array([[-1., 2], [-2., 2]])
    accel_const = constraint.JointAccelerationConstraint(alim, constraint.DiscretizationType.Collocation)
    data = (pi, ss, alim)
    return data, accel_const

  if dof == 6:
    np.random.seed(10)
    N = 20
    way_pts = np.random.randn(10, 6)
    pi = ta.SplineInterpolator(np.linspace(0, 1, 10), way_pts)
    ss = np.linspace(0, 1, N + 1)
    vlim_ = np.random.rand(6)
    alim = np.vstack((-vlim_, vlim_)).T
    accel_const = constraint.JointAccelerationConstraint(alim, constraint.DiscretizationType.Collocation)
    data = (pi, ss, alim)
    return data, accel_const

  if dof == '6d':
    np.random.seed(10)
    N = 20
    way_pts = np.random.randn(10, 6)
    pi = ta.SplineInterpolator(np.linspace(0, 1, 10), way_pts)
    ss = np.linspace(0, 1, N + 1)
    alim_s = np.random.rand(6)
    alim = np.vstack((-alim_s, alim_s)).T
    accel_const = constraint.JointAccelerationConstraint(alim_s, constraint.DiscretizationType.Collocation)
    data = (pi, ss, alim)
    return data, accel_const

def test_constraint_params(df):
  """ Test constraint satisfaction with cvxpy.
  """
  (path, ss, alim), accel_const = accel_constraint_setup(df)
  logger.info(f'ss={ss}, vlim={alim}')

  # An user of the class
  a, b, c, F, g, ubound, xbound = accel_const.compute_constraint_params(path, ss)
  assert xbound is None
  logger.info(f'a={a}, b={b}, c={c}, F={F}, g={g}, ubound={ubound}, xbound={xbound}')

  N = ss.shape[0] - 1
  dof = path.dof
  logger.info(f'N={ss.shape[0]}, dof={path.dof}')

  ps = path(ss, 1)
  pss = path(ss, 2)

  F_actual = np.vstack((np.eye(dof), - np.eye(dof)))
  g_actual = np.hstack((alim[:, 1], - alim[:, 0]))

  logging.info('F={}, F_actual={}'.format(F, F_actual))
  logging.info('g={}, g_actual={}'.format(g, g_actual))
  npt.assert_allclose(F, F_actual)
  npt.assert_allclose(g, g_actual)
  for i in range(0, N + 1):
    logging.info('a[i]={}, ps[i]={}'.format(a[i], ps[i]))
    npt.assert_allclose(a[i], ps[i])
    logging.info('b[i]={}, pss[i]={}'.format(b[i], pss[i]))
    npt.assert_allclose(b[i], pss[i])
    logging.info('c[i]={}'.format(c[i]))
    npt.assert_allclose(c[i], np.zeros_like(ps[i]))
    assert ubound is None
    assert xbound is None


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format="%(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
  test_constraint_params(2)
