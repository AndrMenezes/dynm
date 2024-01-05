"""Utils functions."""
import numpy as np
import copy
from dynm.algebra import _calc_predictive_mean_and_var
from dynm.algebra import _build_Gnonlinear, _build_W
from scipy.linalg import block_diag


class TransferFunction():
    """Class for fitting, forecast and update dynamic linear models."""

    def __init__(self, m0: np.ndarray, C0: np.ndarray,
                 order: int, ntfm: int,
                 discount_factors: np.ndarray = None,
                 W: np.ndarray = None,
                 V: float = None):
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
        self.ntfm = ntfm
        self.m = m0.reshape(-1, 1)
        self.C = C0

        if V is None:
            self.n = 1
            self.d = 1
            self.s = 1
            self.discount_factors = discount_factors
        else:
            self.s = V
            self.estimate_V = False

        self.discount_factors = discount_factors
        if W is None:
            self.estimate_W = True
        else:
            self.W = W
            self.estimate_W = False

        # Get index for blocks
        self.index_dict = {}

        for n in range(ntfm):
            block_idx = n * (2 * order + 1) + np.cumsum([order, order, 1])
            self.index_dict[n] = {
                'all': np.arange(n * (2 * order + 1), block_idx[2]),
                'response': np.arange(n * (2 * order + 1), block_idx[0]),
                'decay': np.arange(block_idx[0], block_idx[1]),
                'pulse': np.arange(block_idx[1], block_idx[2])}

        # Build F and G
        self.F = self._build_F()
        self.G = self._build_G(x=np.repeat(0, ntfm))

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
        F = np.array([])

        for i in range(self.ntfm):
            Fi = np.zeros(2 * self.order + 1)
            Fi[0] = 1

            F = np.hstack((F, Fi))

        return F.reshape(-1, 1)

    def _build_G(self, x: np.array):
        m = self.m
        order = self.order
        ntfm = self.ntfm

        G = np.empty([0, 0])
        for n in range(ntfm):
            idx_ = np.concatenate((
                self.index_dict.get(n).get('response'),
                self.index_dict.get(n).get('decay'),
                self.index_dict.get(n).get('pulse')))

            m_ = m[idx_]
            Gi = _build_Gnonlinear(m=m_.reshape(-1, 1), order=order)

            xn = np.ravel(x[n])
            H = np.zeros(Gi.shape[0]).reshape(-1, 1)
            H[0, 0] = xn
            Gi = np.block([[Gi, H], [H.T * 0, 1]])
            G = block_diag(G, Gi)

        return G

    def _build_h(self, G: np.array):
        ntfm = self.ntfm
        G_ = copy.deepcopy(G)

        for n in range(ntfm):
            idx = np.ix_(self.index_dict.get(n).get('response'),
                         self.index_dict.get(n).get('decay'))
            G_[idx] = G_[idx] * 0.0

        m = self.m.T
        h = (G_ - G) @ m.T

        return h

    def _build_P(self, G: np.array):
        return G @ self.C @ G.T

    def _build_W(self, P: np.array):
        if self.estimate_W:
            W = _build_W(mod=self, P=P)

            for n in range(self.ntfm):
                idx = np.ix_(self.index_dict.get(n).get('response')[1:],
                             self.index_dict.get(n).get('response')[1:])

                W[idx] = W[idx] * 0.0
        else:
            W = self.W
        return W
