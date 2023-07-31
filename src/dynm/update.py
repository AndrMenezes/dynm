"""Update methods for dynamic linear models using kalman filter."""
import numpy as np
import pandas as pd
import copy
from scipy import stats
from src.utils import tidy_parameters
from scipy.linalg import block_diag


def update_dlm(mod, y: float, x: float = None):
    """Update space state moments.

        Parameters
        ----------
        y : float
            Observation.
        x : float
            Regressor value for transfer function.
        z : float
            Regressor value for dynamic regression.

        Returns
        -------
        type
            Description of returned object.
    """
    f, q = mod.forecast(x=x)
    a, R = _calc_aR(mod)

    F = self._build_F(z=z)
    A = (Rt @ F) / q
    et = y - f

    # Estimate observational variance
    d = self.d + (et**2) / q
    n = self.n + 1
    s = np.ravel(d / n)[0]
    r = np.ravel((d / self.d) * (self.n / n))[0]

    r_matrix = np.ones([Rt.shape[1], Rt.shape[1]])
    if self.order[0] == 0:
        r_matrix = r * r_matrix
    else:
        r_matrix[self.index__ar_response_block[0],
                 self.index__ar_response_block[0]] = r

    # Kalman filter update
    self.m = at + A * et
    self.C = r_matrix * (Rt - q * A @ A.T)
    self.d = d
    self.n = n
    self.s = s
