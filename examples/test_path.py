import os, sys
sys.path.insert(0, os.path.join('..'))
import logging
import toppra
import toppra.interpolator
import numpy as np
import matplotlib.pyplot as plt

def setup_path_cubic_spline(param):
  start, end = param
  waypoints = [[0, 0.3, 0.5], [1, 2, 3], [0.0, 0.1, 0.2], [0, 0.5, 0]]
  ss = np.linspace(start, end, len(waypoints))  # ss=[0., 0.33333333, 0.66666667, 1.]
  path = toppra.interpolator.SplineInterpolator(ss, waypoints)
  return path, waypoints

def test_cubic_spline():
  path, waypoints = setup_path_cubic_spline([0, 1])
  # visualize ###############################################################
  ss_full = np.linspace(path.path_interval[0], path.path_interval[1], 100)
  fig, ax = plt.subplots(nrows=3, ncols=1)
  for i in range(len(waypoints[0])):
    ax[0].plot(ss_full, path(ss_full)[:, i], '-', c='C%d' % i, linewidth=0.5, label='%d' % i)
    ax[0].plot(path.waypoints[0], path.waypoints[1][:, i], '.', c='C%d' % i, markersize=2)
    ax[1].plot(ss_full, path(ss_full, 1)[:, i], '-', c='C%d' % i, linewidth=0.5, label='%d' % i)
    ax[2].plot(ss_full, path(ss_full, 2)[:, i], '-', c='C%d' % i, linewidth=0.5, label='%d' % i)
  ax[0].legend()
  ax[0].grid(True)
  # plt.show()
  plt.savefig('test_path-cubic_spline.png', dpi = 800)

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format="%(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
  test_cubic_spline()