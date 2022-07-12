import logging
import numpy as np
import numpy.testing as npt
from toppra import PolynomialPath

def test_scalar():
  """Scalar case."""
  pi = PolynomialPath([1, 2, 3], s_start=0, s_end=2)  # 1 + 2s + 3s^2
  assert pi.dof == 1
  a = pi.eval([0, 0.5, 1])
  npt.assert_allclose(pi.eval([0, 0.5, 1]).astype(np.float64), [1, 2.75, 6])
  npt.assert_allclose(pi.evald([0, 0.5, 1]).astype(np.float64), [2, 5, 8])
  npt.assert_allclose(pi.evaldd([0, 0.5, 1]).astype(np.float64), [6, 6, 6])
  npt.assert_allclose(pi.path_interval, np.r_[0, 2])

def test_2_dof():
  """Polynomial path with 2dof."""
  pi = PolynomialPath([[1, 2, 3], [-2, 3, 4, 5]])
  # [1 + 2s + 3s^2]
  # [-2 + 3s + 4s^2 + 5s^3]
  assert pi.dof == 2
  npt.assert_allclose(
      pi.eval([0, 0.5, 1]), [[1, -2], [2.75, 1.125], [6, 10]])
  npt.assert_allclose(
      pi.evald([0, 0.5, 1]), [[2, 3], [5, 10.75], [8, 26]])
  npt.assert_allclose(pi.evaldd([0, 0.5, 1]), [[6, 8], [6, 23], [6, 38]])

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format="%(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
  test_scalar()
  test_2_dof()

