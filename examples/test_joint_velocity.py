import os, sys
sys.path.insert(0, os.path.join('..'))
import logging
import cvxpy as cvx
import numpy as np
import numpy.testing as npt
import toppra as ta
import toppra.constraint as constraint
from toppra.constants import TINY, JVEL_MAXSD, SMALL
from scipy.interpolate import CubicSpline

ta.setup_logging("DEBUG")
logger = logging.getLogger('toppra')

def velocity_pc_data(dof):
  if dof == 2:
    coeff = [[1., 2, 3], [-2., -3., 4., 5.]]
    pi = ta.PolynomialPath(coeff)
    ss = np.linspace(0, 0.75, 4)
    vlim = np.array([[-1., 2], [-2., 2]])
    velocity_constraint = constraint.JointVelocityConstraint(vlim)
    data = (pi, ss, vlim)
    return data, velocity_constraint
  if dof == 6:
    np.random.seed(10)
    N = 100
    way_pts = np.random.randn(10, 6)
    pi = ta.SplineInterpolator(np.linspace(0, 1, 10), way_pts)
    ss = np.linspace(0, 1, N + 1)
    vlim_ = np.random.rand(6) * 10 + 2.
    vlim = np.vstack((-vlim_, vlim_)).T
    vel_constraint = constraint.JointVelocityConstraint(vlim)
    data = (pi, ss, vlim)
    return data, vel_constraint

def test_constraint_satisfaction(dof):
  """ Test constraint satisfaction with cvxpy."""
  data, pc = velocity_pc_data(dof)
  path, ss, vlim = data
  logger.info(f'ss={ss}, vlim={vlim}')

  constraint_param = pc.compute_constraint_params(path, ss)
  _, _, _, _, _, _, xlimit = constraint_param
  logger.info(f'xlimit={xlimit}') # 路径上每个位置sdmin和sdmax的平方

  qs = path(ss, 1)
  logger.info(f'qs={qs}')
  N = ss.shape[0] - 1

  sd = cvx.Variable()

  for i in range(0, N + 1):
      # 2. Compute max sd from the data
      constraints = [qs[i] * sd <= vlim[:, 1],
                     qs[i] * sd >= vlim[:, 0],
                     sd >= 0, sd <= JVEL_MAXSD]
      prob = cvx.Problem(cvx.Maximize(sd), constraints)
      try:
          prob.solve(solver=cvx.ECOS, abstol=1e-9)
          xmax = sd.value ** 2

          prob = cvx.Problem(cvx.Minimize(sd), constraints)
          prob.solve(solver=cvx.ECOS, abstol=1e-9)
          xmin = sd.value ** 2
      except cvx.SolverError:
          continue
      logger.info(f'i={i}, xmax={xmax}, xmin={xmin}')
      # 3. They should agree
      npt.assert_allclose([xmin, xmax], xlimit[i], atol=SMALL)  # 使用直接解析公式计算和使用优化库计算的对比

      # Assert non-negativity
      assert xlimit[i, 0] >= 0

def test_jnt_vel_varying_basic():
  # constraint
  ss_wpts = np.r_[0, 1, 2]
  vlim_wpts = [[[-1, 2], [-1, 2]],
               [[-2, 3], [-2, 3]],
               [[-1, 0], [-1, 0]]]
  logging.info(np.array(vlim_wpts).shape) # (N, dof, 2)
  vlim_spl = CubicSpline(ss_wpts, vlim_wpts)
  logging.info(vlim_spl(1.0)) # (dof, 2)
  constraint = ta.constraint.JointVelocityConstraintVarying(vlim_spl)
  # path
  coeff = [[1., 2, 3], [-2., -3., 4., 5.]]  # [1+2*s+3*s^2, -2-3*s+4*s^2+5*s^3]
  path = ta.PolynomialPath(coeff)
  gridpoints = np.linspace(0, 2, 10)  # [0, 0.22222222, 0.44444444, 0.66666667, 0.88888889, 1.11111111, 1.33333333, 1.55555556, 1.77777778 2.]
  qs = path((gridpoints), 1)
  _, _, _, _, _, _, xlimit = constraint.compute_constraint_params(path, gridpoints)
  
  # constraint splines
  qs = path(gridpoints, 1)
  # logging.info('qs = {}'.format(qs))
  # test
  sd = cvx.Variable()
  for i in range(gridpoints.shape[0]):
    logging.info('i = {}'.format(i))
    vlim = vlim_spl(gridpoints[i])  # vlim.shape=(dof, 2)
    logging.info('vlim = {}'.format(vlim))

    # 2. compute max sd from the data
    constraints = [qs[i] * sd <= vlim[:, 1],
                   qs[i] * sd >= vlim[:, 0],
                   sd >= 0, sd <= JVEL_MAXSD]

    try:
        prob = cvx.Problem(cvx.Maximize(sd), constraints)
        prob.solve(solver=cvx.ECOS, abstol=1e-9)
        xmax = sd.value ** 2

        prob = cvx.Problem(cvx.Minimize(sd), constraints)
        prob.solve(solver=cvx.ECOS, abstol=1e-9)
        xmin = sd.value ** 2
    except cvx.SolverError:
        continue  # ECOS can't solve this problem.

    # 3. they should agree
    logging.info('[xmin, xmax]=[{}, {}]'.format(xmin, xmax))
    logging.info('xlimit[i] = {}'.format(xlimit[i]))
    npt.assert_allclose([xmin, xmax], xlimit[i], atol=SMALL)

    # assert non-negativity
    assert xlimit[i, 0] >= 0

def test_only_max_vel_given():
  c = ta.constraint.JointVelocityConstraint([1, 1.2, 2])
  logging.info(c.vlim)

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format="%(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
  # test_jnt_vel_varying_basic()
  # test_only_max_vel_given()
  test_constraint_satisfaction(2)


