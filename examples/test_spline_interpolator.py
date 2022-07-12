import logging
import numpy as np
import numpy.testing as npt
from toppra import SplineInterpolator

def scalar_pi():
  sswp, wp, ss, path_interval = [[0, 0.3, 0.5], [1, 2, 3], [0.0, 0.1, 0.2, 0.3, 0.5], [0, 0.5]]
  pi = SplineInterpolator(sswp, wp)  # 1 + 2s + 3s^2
  logging.info('pi(sswp, 0)={}'.format(pi(sswp, 0)))
  logging.info(pi.path_interval)
  return pi, path_interval



if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format="%(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
  scalar_pi()