import logging
import toppra
import toppra.interpolator
import numpy as np
import matplotlib.pyplot as plt

# params=[[0, 1], [1.5, 2.7]]
def setup_path(param):
  start, end = param
  waypoints = [[0, 0.3, 0.5], [1, 2, 3], [0.0, 0.1, 0.2], [0, 0.5, 0]]
  ss = np.linspace(start, end, len(waypoints))  # ss=[0., 0.33333333, 0.66666667, 1.]
  path = toppra.interpolator.SplineInterpolator(ss, waypoints)
  return path, waypoints

def test_basic_usage():
  path, waypoints = setup_path([0, 1])
  gridpoints_ept = toppra.interpolator.propose_gridpoints(path, 1e-2)
  assert gridpoints_ept[0] == path.path_interval[0]
  assert gridpoints_ept[-1] == path.path_interval[1]

  # The longest segment should be smaller than 0.1. This is to
  # ensure a reasonable response.
  assert np.max(np.diff(gridpoints_ept)) < 0.05

  # # visualize ###############################################################
  ss_full = np.linspace(path.path_interval[0], path.path_interval[1], 100)
  for i in range(len(waypoints[0])):
    plt.plot(ss_full, path(ss_full)[:, i], '-', c='C%d' % i, linewidth=0.5, label='%d' % i)
    plt.plot(gridpoints_ept, path(gridpoints_ept)[:, i], 'o', c='C%d' % i, markersize=1)
    plt.plot(path.waypoints[0], path.waypoints[1][:, i], '.', c='C%d' % i, markersize=5)
  plt.legend()
  plt.grid(True)
  # plt.show()
  plt.savefig('test_find_gridpoints-basic_usage.png', dpi = 800)


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format="%(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
  test_basic_usage()
