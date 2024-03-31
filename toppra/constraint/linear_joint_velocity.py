"""This module implements the joint velocity constraint."""
import logging
import numpy as np
# from toppra._CythonUtils import (_create_velocity_constraint,
#                                  _create_velocity_constraint_varying)
from .linear_constraint import LinearConstraint
from toppra.constants import JVEL_MAXSD
import toppra as ta

logger = logging.getLogger('toppra')

def create_velocity_constraint(qs, vlim):
    """ Evaluate coefficient matrices for velocity constraints.

    A maximum allowable value for the path velocity is defined with
    `MAXSD`. This constant results in a lower bound on trajectory
    duration obtain by toppra.

    Args:
    ----
    qs: ndarray
        Path derivatives at each grid point.
    vlim: ndarray
        Velocity bounds.
    
    Returns:
    --------
    a: ndarray
    b: ndarray
    c: ndarray
        Coefficient matrices.
    """
    a = np.zeros((qs.shape[0], 2), dtype=np.float64)
    b = np.ones((qs.shape[0], 2), dtype=np.float64)
    c = np.zeros((qs.shape[0], 2), dtype=np.float64)
    b[:, 1] = -1
    for i in range(qs.shape[0]):
        sdmin = -JVEL_MAXSD
        sdmax = JVEL_MAXSD
        for k in range(qs.shape[1]):
            if qs[i, k] > 0:
                sdmax = min(vlim[k, 1] / qs[i, k], sdmax)   # \frac{\mathrm{d} s}{\mathrm{d} t} = \frac{\mathrm{d} \theta}{\mathrm{d} t} / \frac{\mathrm{d} \theta}{\mathrm{d} s}
                sdmin = max(vlim[k, 0] / qs[i, k], sdmin)   # qdt = qds * sdt
            elif qs[i, k] < 0:
                sdmax = min(vlim[k, 0] / qs[i, k], sdmax)
                sdmin = max(vlim[k, 1] / qs[i, k], sdmin)
        c[i, 0] = - sdmax**2
        c[i, 1] = max(sdmin, 0.)**2
    # logger.info(f'c={c}')
    return a, b, c


def create_velocity_constraint_varying(qs, vlim_grid):
    """ Evaluate coefficient matrices for velocity constraints.

    Args:
    ----
    qs: (N,) ndarray
        Path derivatives at each grid point.
    vlim_grid: (N, dof, 2) ndarray
        Velocity bounds at each grid point.
    
    Returns:
    --------
    a: ndarray
    b: ndarray
    c: ndarray
        Coefficient matrices.
    """
    # Evaluate sdmin, sdmax at each steps and fill the matrices.
    a = np.zeros((qs.shape[0], 2), dtype=float)
    b = np.ones((qs.shape[0], 2), dtype=float)
    c = np.zeros((qs.shape[0], 2), dtype=float)
    b[:, 1] = -1
    for i in range(qs.shape[0]):
        sdmin = -JVEL_MAXSD
        sdmax = JVEL_MAXSD
        for k in range(qs.shape[1]):
            if qs[i, k] > 0:
                sdmax = min(vlim_grid[i, k, 1] / qs[i, k], sdmax)
                sdmin = max(vlim_grid[i, k, 0] / qs[i, k], sdmin)
            elif qs[i, k] < 0:
                sdmax = min(vlim_grid[i, k, 0] / qs[i, k], sdmax)
                sdmin = max(vlim_grid[i, k, 1] / qs[i, k], sdmin)
        c[i, 0] = - sdmax**2
        c[i, 1] = max(sdmin, 0.)**2
    return a, b, c


class JointVelocityConstraint(LinearConstraint):
    """A Joint Velocity Constraint class.

    Parameters
    ----------
    vlim: np.ndarray
        Shape (dof, 2). The lower and upper velocity bounds of the j-th joint
        are given by vlim[j, 0] and vlim[j, 1] respectively.

    """

    def __init__(self, vlim):
        super(JointVelocityConstraint, self).__init__()
        vlim = np.array(vlim, dtype=float)
        if np.isnan(vlim).any():
            raise ValueError("Bad velocity given: %s" % vlim)
        if len(vlim.shape) == 1:
            self.vlim = np.vstack((-np.array(vlim), np.array(vlim))).T
        else:
            self.vlim = np.array(vlim, dtype=float)
        self.dof = self.vlim.shape[0]
        self._assert_valid_limits()

    def _assert_valid_limits(self):
        """Check that the velocity limits is valid."""
        assert self.vlim.shape[1] == 2, "Wrong input shape."
        for i in range(self.dof):
            if self.vlim[i, 0] >= self.vlim[i, 1]:
                raise ValueError("Bad velocity limits: {:} (lower limit) > {:} (higher limit)".format(
                    self.vlim[i, 0], self.vlim[i, 1]))
        self._format_string = "    Velocity limit: \n"
        for i in range(self.vlim.shape[0]):
            self._format_string += "      J{:d}: {:}".format(
                i + 1, self.vlim[i]) + "\n"

    def compute_constraint_params(self, path, gridpoints):
        if path.dof != self.get_dof():
            raise ValueError(
                "Wrong dimension: constraint dof ({:d}) not equal to path dof ({:d})"
                .format(self.get_dof(), path.dof))
        qs = path(gridpoints, 1)   # [len(gridpoints), dof], \frac{\mathrm{d} \theta}{\mathrm{d} s}, qds
        # _, _, xbound_ = _create_velocity_constraint(qs, self.vlim)  # 已知qds和qdt，计算sdt，见'cpdef _create_velocity_constraint'
        _, _, xbound_ = create_velocity_constraint(qs, self.vlim)
        xbound = np.array(xbound_)
        xbound[:, 0] = xbound_[:, 1]
        xbound[:, 1] = -xbound_[:, 0]
        return None, None, None, None, None, None, xbound
        # 返回值a, b, c, F, h, ubound, xbound, 其中a.shape=[N, dof], b.shape=[N, dof], c.shape=[N, dof], F.shape=[2*dof, dof], h.shape=[2*dof], ubound.shape=[N, 2], xbound.shape=[N, 2]
        # F * (a[i] * u + b[i] * x + c[i]) <= h, ubound[i, 0] <= u <= ubound[i, 1], xbound[i, 0] <= x <= xbound[i, 1]
        # 具体用法可参考cvxpy_solverwrapper.py，另外见基类LinearConstraint
        # 这里限制速度qdt，将其转换为sdt的平方，状态变量x=sdt^2, 即限制xbound


class JointVelocityConstraintVarying(LinearConstraint):
    """A Joint Velocity Constraint class.

    This class handle velocity constraints that vary along the path.

    Parameters
    ----------
    vlim_func: (float) -> np.ndarray
        A function that receives a scalar (float) and produce an array
        with shape (dof, 2). The lower and upper velocity bounds of
        the j-th joint are given by out[j, 0] and out[j, 1]
        respectively.
    """

    def __init__(self, vlim_func):
        super(JointVelocityConstraintVarying, self).__init__()
        self.dof = vlim_func(0).shape[0]
        self._format_string = "    Varying Velocity limit: \n"
        self.vlim_func = vlim_func

    def compute_constraint_params(self, path, gridpoints):
        if path.dof != self.get_dof():
            raise ValueError(
                "Wrong dimension: constraint dof ({:d}) not equal to path dof ({:d})"
                .format(self.get_dof(), path.dof))
        qs = path((gridpoints), 1)
        vlim_grid = np.array([self.vlim_func(s) for s in gridpoints])
        # logging.info('vlim_grid: {}'.format(vlim_grid))
        _, _, xbound_ = create_velocity_constraint_varying(qs, vlim_grid)
        xbound = np.array(xbound_)
        xbound[:, 0] = xbound_[:, 1]
        xbound[:, 1] = -xbound_[:, 0]
        return None, None, None, None, None, None, xbound
