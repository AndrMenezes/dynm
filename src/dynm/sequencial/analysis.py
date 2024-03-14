"""Utils functions."""
import numpy as np
from dynm.sequencial.filter import _foward_filter
from dynm.sequencial.smooth import _backward_smoother
from copy import copy
from dynm.utils.summary import get_predictive_log_likelihood


class Analysis():
    """Class for fitting, forecast and update dynamic linear models."""

    def __init__(self, mod):
        """Define model.

        Define model with observation/system equations components \
        and initial information for prior moments.

        Parameters
        ----------
        m0 : np.ndarray
            prior mean for state space components.
        C0 : np.ndarray
            prior covariance for state space components.
        delta : float
            discount factor.

        """
        self.mod = copy.deepcopy(mod)

    def fit(self,
            y: np.ndarray,
            X: dict = {},
            level: float = 0.05,
            smooth: bool = False):
        """Short summary.

        Parameters
        ----------
        y : np.ndarray
            Description of parameter `y`.
        x : np.ndarray
            Description of parameter `x`.

        Returns
        -------
        type
            Description of returned object.

        """
        # Fit
        foward_dict = _foward_filter(mod=self.mod, y=y, X=X, level=level)
        self.dict_filter = copy(foward_dict.get('filter'))
        self.dict_state_params = copy(foward_dict.get('state_params'))
        self.dict_state_evolution = copy(foward_dict.get('state_evolution'))

        if smooth:
            backward_dict = _backward_smoother(mod=self.mod, X=X, level=level)
            self.dict_smooth = copy(backward_dict.get('smooth'))
            self.dict_smooth_params = copy(backward_dict.get('smooth_params'))

        self.llk = get_predictive_log_likelihood(mod=self.mod)

        return self
