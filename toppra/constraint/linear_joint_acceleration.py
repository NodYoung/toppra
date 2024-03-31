"""Joint acceleration constraint."""
import numpy as np
from .linear_constraint import LinearConstraint, canlinear_colloc_to_interpolate
from ..constraint import DiscretizationType
from ..interpolator import AbstractGeometricPath


class JointAccelerationConstraint(LinearConstraint):
    """The Joint Acceleration Constraint class.

    A joint acceleration constraint is given by

    .. math ::

                \ddot{\mathbf{q}}_{min} & \leq \ddot{\mathbf q}
                                                    &\leq \ddot{\mathbf{q}}_{max} \\\\
                \ddot{\mathbf{q}}_{min} & \leq \mathbf{q}'(s_i) u_i + \mathbf{q}''(s_i) x_i
                                                    &\leq \ddot{\mathbf{q}}_{max}
        即qddt_min <= qddt(=qds*u+qdds*x_i) <= qddt_max
    where :math:`u_i, x_i` are respectively the path acceleration and
    path velocity square at :math:`s_i`. For more detail see :ref:`derivationKinematics`.

    Rearranging the above pair of vector inequalities into the form
    required by :class:`LinearConstraint`, we have:
    转换为a_i=qdot_i, b_i=qddot_i, F=[1, -1], h=[qdd_max, qdd_min]
    - :code:`a[i]` := :math:`\mathbf q'(s_i)`
    - :code:`b[i]` := :math:`\mathbf q''(s_i)`
    - :code:`F` := :math:`[\mathbf{I}, -\mathbf I]^T`
    - :code:`h` := :math:`[\ddot{\mathbf{q}}_{max}^T, -\ddot{\mathbf{q}}_{min}^T]^T`
    """

    def __init__(self, alim, discretization_scheme=DiscretizationType.Interpolation):
        """Initialize the joint acceleration class.

        Parameters
        ----------
        alim: array
            Shape (dof, 2). The lower and upper acceleration bounds of the
            j-th joint are alim[j, 0] and alim[j, 1] respectively.

        discretization_scheme: :class:`.DiscretizationType`
            Can be either Collocation (0) or Interpolation
            (1). Interpolation gives more accurate results with slightly
            higher computational cost.
        """
        super(JointAccelerationConstraint, self).__init__()
        alim = np.array(alim, dtype=float)
        if np.isnan(alim).any():
            raise ValueError("Bad velocity given: %s" % alim)
        if len(alim.shape) == 1:
            self.alim = np.vstack((-np.array(alim), np.array(alim))).T
        else:
            self.alim = np.array(alim, dtype=float)
        self.dof = self.alim.shape[0]
        self.set_discretization_type(discretization_scheme)

        assert self.alim.shape[1] == 2, "Wrong input shape."
        self._format_string = "    Acceleration limit: \n"
        for i in range(self.alim.shape[0]):
            self._format_string += "      J{:d}: {:}".format(i + 1, self.alim[i]) + "\n"
        self.identical = True

    def compute_constraint_params(
        self, path: AbstractGeometricPath, gridpoints: np.ndarray, *args, **kwargs
    ):
        if path.dof != self.dof:
            raise ValueError(
                "Wrong dimension: constraint dof ({:d}) not equal to path dof ({:d})".format(
                    self.dof, path.dof
                )
            )
        ps_vec = (path(gridpoints, order=1)).reshape((-1, path.dof))  # [len(gridpoints), dof], \frac{\mathrm{d} \theta}{\mathrm{d} s}
        pss_vec = (path(gridpoints, order=2)).reshape((-1, path.dof))  # \frac{\mathrm{d^2} \theta}{\mathrm{d} s^2}
        dof = path.dof
        F_single = np.zeros((dof * 2, dof))
        g_single = np.zeros(dof * 2)
        g_single[0:dof] = self.alim[:, 1]
        g_single[dof:] = -self.alim[:, 0]
        F_single[0:dof, :] = np.eye(dof)
        F_single[dof:, :] = -np.eye(dof)
        if self.discretization_type == DiscretizationType.Collocation:
            return (
                ps_vec,
                pss_vec,
                np.zeros_like(ps_vec),
                F_single,
                g_single,
                None,
                None,
            )
            # 返回值a, b, c, F, h, ubound, xbound, 其中a.shape=[N, dof], b.shape=[N, dof], c.shape=[N, dof], F.shape=[2*dof, dof], h.shape=[2*dof], ubound.shape=[N, 2], xbound.shape=[N, 2]
            # F * (a[i] * u + b[i] * x + c[i]) <= h, ubound[i, 0] <= u <= ubound[i, 1], xbound[i, 0] <= x <= xbound[i, 1]
            # 具体用法可参考cvxpy_solverwrapper.py
            # 这里限制加速度，qds*sdd+qdds*sd^2 = qddt_max, u=sdd, x=sd^2
        elif self.discretization_type == DiscretizationType.Interpolation:
            return canlinear_colloc_to_interpolate(
                ps_vec,
                pss_vec,
                np.zeros_like(ps_vec),
                F_single,
                g_single,
                None,
                None,
                gridpoints,
                identical=True,
            )
        else:
            raise NotImplementedError("Other form of discretization not supported!")
