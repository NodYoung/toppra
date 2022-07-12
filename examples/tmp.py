import logging
import numpy as np


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format="%(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
  # path_dist = np.linspace(0, 1, 5)
  # logging.info(path_dist)
  logging.info(np.linspace(0, 2, 10).shape[0])
  # vlim_wpts = [[[-1, 2], [-1, 2]],
  #              [[-2, 3], [-2, 3]],
  #              [[-1, 0], [-1, 0]]]
  # logging.info(np.array(vlim_wpts).shape)
  # coeff = [[1., 2, 3], [-2., -3., 4., 5.]]
  # coeff = np.array(coeff, dtype=object)
  # logging.info(coeff.shape)
