"""Utils functions."""
import numpy as np
import copy
from dynm.utils.algebra import _calc_predictive_mean_and_var, _calc_aR
from dynm.utils.algebra import _build_Gnonlinear, _build_W


class AutoRegressive():
    """Class for fitting, forecast and update dynamic linear models."""

    def __init__(self, m0: np.ndarray, C0: np.ndarray, order: int,
                 discount_factors: np.ndarray = None,
                 W: np.ndarray = None, V: float = None):
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
        self.order = order
        self.m = m0.reshape(-1, 1)
        self.C = C0

        if V is None:
            self.n = 1
            self.d = 1
            self.s = 1
            self.estimate_V = True
        else:
            self.s = V
            self.estimate_V = False

        self.discount_factors = discount_factors
        if W is None:
            self.estimate_W = True
        else:
            self.W = W
            self.estimate_W = False

        self.F = self._build_F()
        self.G = self._build_G()

        # Get index for blocks
        block_idx = np.cumsum([order, order])
        self.index_dict = {
            'response': np.arange(0, block_idx[0]),
            'decay': np.arange(block_idx[0], block_idx[1])
        }

    def forecast(self, k: int = 1):
        """Forecast y at time (t+k).

        Parameters

        ----------
        F : np.ndarray
            Design matrix.
        G : np.ndarray
            State matrix.

        Returns
        -------
        type
            Description of returned object.

        """
        f, q = _calc_predictive_mean_and_var(mod=self)
        return np.ravel(f), np.ravel(q)

    def _build_F(self):
        F = np.zeros(2 * self.order)
        F[0] = 1
        return F.reshape(-1, 1)

    def _build_G(self):
        m = self.m
        order = self.order
        G = _build_Gnonlinear(m=m, order=order)
        return G

    def _build_h(self, G: np.array):
        G_ = copy.deepcopy(G)
        idx = np.ix_(self.index_dict.get('response'),
                     self.index_dict.get('decay'))

        G_[idx] = G_[idx] * 0.0

        m = self.m.T
        h = (G_ - G) @ m.T

        return h

    def _calc_aR(self):
        m = self.m
        order = self.order

        a, R = _calc_aR(mod=self)
        G = self._build_Gnonlinear(m=m, order=order)
        a += self._build_h(G=G)

        return a, R

    def _build_P(self, G: np.array):
        return G @ self.C @ G.T

    def _build_W(self, P: np.array):
        if self.estimate_W:
            W = _build_W(mod=self, P=P)
            W[1:self.order, 1:self.order] = W[1:self.order, 1:self.order] * 0.0
            W[0, 0] = self.s
        else:
            W = self.W
        return W
