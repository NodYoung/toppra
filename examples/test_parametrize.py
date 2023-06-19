import os, sys
sys.path.insert(0, os.path.join('..'))
import logging
import toppra
import toppra.interpolator
import numpy as np
import matplotlib.pyplot as plt


def setup_path():
  return toppra.SplineInterpolator([0, 1, 2], [(0, 0), (1, 2), (2, 0)])

def test_spline():
  path = setup_path()
  ss_waypoints, q_waypoints = path.waypoints
  ss_dense = np.linspace(ss_waypoints[0], ss_waypoints[-1], 500)
  fig, ax = plt.subplots(nrows=2, ncols=1)
  for i in range(path.waypoints[1].shape[1]):
    ax[0].plot(ss_dense, path(ss_dense)[:, i], '-', c='C%d' % i, linewidth=0.5, label='%d' % i)
    ax[0].plot(ss_waypoints, q_waypoints[:, i], '.', c='C%d' % i, markersize=2)
    ax[1].plot(ss_dense, path(ss_dense, 1)[:, i], '-', c='C%d' % i, linewidth=0.5, label='%d' % i)
  ax[0].legend()
  ax[1].legend()
  fig.suptitle("original path")
  plt.savefig('test_parametrize-spline1.png', dpi = 800)
  plt.clf()
  
  gridpoints = [0, 0.5, 1, 1.5, 2]
  velocities = [1, 2, 2, 1, 0]
  # xd = [1, 4, 4, 1, 0]
  # ud = [6.0, 0, -6.0, -2.0]
  path_new = toppra.ParametrizeSpline(path, gridpoints, velocities)
  ts = np.linspace(path_new.path_interval[0], path_new.path_interval[1], 500)
  # ss, vs, us = path_new._eval_params(ts)
  # fig, ax = plt.subplots(nrows=1, ncols=2)
  # ax[0].plot(ss, vs, label="v(s)")
  # ax[0].plot(gridpoints, velocities, "o", label="input", markersize=2)
  # # ax[0].plot(ss, us, label="u(s)")
  # ax[0].legend()
  # ax[0].set_title("velocity(path)")
  
  # ax[1].plot(ts, ss, label="s(t)")
  # ax[1].plot(ts, vs, label="v(t)")
  # # ax[1].plot(ts, us, label="a(t)")
  # ax[1].legend()
  # ax[1].set_title("path(time)")
  # plt.savefig('test_parametrize-const_accel2.png', dpi = 800)
  # plt.clf()
  fig, ax = plt.subplots(nrows=3, ncols=1)
  for i in range(path.waypoints[1].shape[1]):
    ax[0].plot(ts, path_new(ts, 0)[:, i], '-', c='C%d' % i, linewidth=0.5, label='%d' % i)
    ax[1].plot(ts, path_new(ts, 1)[:, i], '-', c='C%d' % i, linewidth=0.5)
    ax[2].plot(ts, path_new(ts, 2)[:, i], '-', c='C%d' % i, linewidth=0.5)
  ax[0].legend()
  fig.suptitle("retimed path")
  plt.savefig('test_parametrize-spline3.png', dpi = 800)
  plt.clf()
  

def test_const_accel():
  path = setup_path()
  
  ss_waypoints, q_waypoints = path.waypoints
  ss_dense = np.linspace(ss_waypoints[0], ss_waypoints[-1], 500)
  fig, ax = plt.subplots(nrows=2, ncols=1)
  for i in range(path.waypoints[1].shape[1]):
    ax[0].plot(ss_dense, path(ss_dense)[:, i], '-', c='C%d' % i, linewidth=0.5, label='%d' % i)
    ax[0].plot(ss_waypoints, q_waypoints[:, i], '.', c='C%d' % i, markersize=2)
    ax[1].plot(ss_dense, path(ss_dense, 1)[:, i], '-', c='C%d' % i, linewidth=0.5, label='%d' % i)
  ax[0].legend()
  ax[1].legend()
  fig.suptitle("original path")
  plt.savefig('test_parametrize-const_accel1.png', dpi = 800)
  plt.clf()
  gridpoints = [0, 0.5, 1, 1.5, 2]
  velocities = [1, 2, 2, 1, 0]
  # xd = [1, 4, 4, 1, 0]
  # ud = [6.0, 0, -6.0, -2.0]
  path_new = toppra.ParametrizeConstAccel(path, gridpoints, velocities)
  ts = np.linspace(path_new.path_interval[0], path_new.path_interval[1], 500)
  ss, vs, us = path_new._eval_params(ts)
  fig, ax = plt.subplots(nrows=1, ncols=2)
  ax[0].plot(ss, vs, label="v(s)")
  ax[0].plot(gridpoints, velocities, "o", label="input", markersize=2)
  # ax[0].plot(ss, us, label="u(s)")
  ax[0].legend()
  ax[0].set_title("velocity(path)")
  
  ax[1].plot(ts, ss, label="s(t)")
  ax[1].plot(path_new._ts, gridpoints, "o", label="input", markersize=2)
  ax[1].plot(ts, vs, label="v(t)")
  ax[1].plot(path_new._ts, velocities, "o", label="input", markersize=2)
  # ax[1].plot(ts, us, label="a(t)")
  ax[1].legend()
  ax[1].set_title("path(time)")
  plt.savefig('test_parametrize-const_accel2.png', dpi = 800)
  plt.clf()
  
  fig, ax = plt.subplots(nrows=3, ncols=1)
  for i in range(path.waypoints[1].shape[1]):
    ax[0].plot(ts, path_new(ts, 0)[:, i], '-', c='C%d' % i, linewidth=0.5, label='%d' % i)
    ax[1].plot(ts, path_new(ts, 1)[:, i], '-', c='C%d' % i, linewidth=0.5)
    ax[2].plot(ts, path_new(ts, 2)[:, i], '-', c='C%d' % i, linewidth=0.5)
  ax[0].legend()
  fig.suptitle("retimed path")
  plt.savefig('test_parametrize-const_accel3.png', dpi = 800)
  plt.clf()
  # try:
  #   path_new.plot_parametrization(show=False)
  #   plt.savefig('test_parametrize-const_accel.png', dpi = 800)
  # except Exception:
  #   # when run on CI, this fails because of some tkinter error
  #   pass
  assert path_new(0).shape == (2,)
  assert path_new([0]).shape == (1, 2)
  assert path_new.path_interval[1] > 0
  assert path_new.duration > 0
  path_new(np.r_[0, 0.1])
  path_new(np.r_[0, 0.1], 1)
  path_new(np.r_[0, 0.1], 2)


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format="%(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
  # test_const_accel()
  test_spline()
