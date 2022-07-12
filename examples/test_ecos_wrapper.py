import logging
import numpy as np
import toppra
import toppra.constraint as constraint
from toppra.solverwrapper.ecos_solverwrapper import ecosWrapper

def generate_path():
  np.random.seed(1)
  path = toppra.SplineInterpolator(np.linspace(0, 1, 5), np.random.randn(5, 3)) # np.linspace(0, 1, 5)=[0.0 0.25 0.5 0.75 1.0]
  # logging.info(path.duration) # 1.0
  return path

def generate_vel_accel_robustaccel():
  "Velocity + Acceleration + Robust Acceleration constraint"
  dtype_a, dtype_ra = (0, 0)
  vlims = np.array([[-1, 1], [-1, 2], [-1, 4]], dtype=float)
  alims = np.array([[-1, 1], [-1, 2], [-1, 4]], dtype=float)
  vel_cnst = constraint.JointVelocityConstraint(vlims)
  accl_cnst = constraint.JointAccelerationConstraint(alims, dtype_a)
  robust_accl_cnst = constraint.RobustLinearConstraint(
      accl_cnst, [0.5, 0.1, 2.0], dtype_ra)
  return vel_cnst, accl_cnst, robust_accl_cnst

def test_linear_constraints_only(i, g, x_ineq):
  vel_c, acc_c, robust_acc_c = generate_vel_accel_robustaccel()
  path = generate_path()
  path_dist = np.linspace(0, path.duration, 10 + 1)
  solver = ecosWrapper([vel_c, acc_c], path, path_dist)
  xmin, xmax = x_ineq
  xnext_min = 0
  xnext_max = 1
  result = solver.solve_stagewise_optim(i, None, g, xmin, xmax, xnext_min, xnext_max)
  logging.info(result)



if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format="%(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
  # logging.info(path.duration)
  test_linear_constraints_only(0, np.array([0.2, -1]), (-1, 1))
  # for i in [0, 5, 9]:
  #   for g in [np.array([0.2, -1]), np.array([0.5, 1]), np.array([2.0, 1])]:
  #     for x_ineq in [(-1, 1), (0.2, 0.2), (0.4, 0.3), (np.nan, np.nan)]:
  #       test_linear_constraints_only(i, g, x_ineq)
