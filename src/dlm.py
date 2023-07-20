"""Dynamic Linear Model with transfer function."""
import numpy as np
from packages.dlmtf.src.algebra import _build_W
from scipy.linalg import block_diag


class DLM():
    """Class for fitting, forecast and update dynamic linear models."""

    def __init__(self, m0: np.ndarray, C0: np.ndarray,
                 ntrend: int, nregn: int,
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
        self.ntrend = ntrend
        self.nregn = nregn
        self.m = m0.reshape(-1, 1)  # Validar entrada de dimensões
        self.C = C0

        if V is None:
            self.n = 1
            self.d = 1
            self.s = 1
            self.estimate_V = True
        else:
            self.s = np.sqrt(V)
            self.estimate_V = False

        self.discount_factors = discount_factors
        if W is None:
            self.estimate_W = True
        else:
            self.W = W
            self.estimate_W = False

        self.F = self._build_F(x=0)
        self.G = self._build_G()

        # Get index for blocks
        block_idx = np.cumsum([ntrend, nregn])
        self.index_dict = {
            'trend': np.arange(0, block_idx[0]),
            'reg': np.arange(block_idx[0], block_idx[1])
        }

        # Validate entries section

    def _build_Ftrend(self):
        ntrend = self.ntrend
        Ftrend = np.ones(ntrend)

        if ntrend == 2:
            Ftrend[1] = 0

        return Ftrend

    def _build_Fregn(self, x: np.array):
        nregn = self.nregn
        # Fregn = np.ones(nregn) * (x - self.m[0])
        Fregn = np.ones(nregn) * x
        return Fregn

    def _build_F(self, x: np.array = None):
        Ftrend = self._build_Ftrend()
        Fregn = self._build_Fregn(x=x)
        F = np.block([Ftrend, Fregn]).reshape(-1, 1)
        return F

    def _build_Gtrend(self):
        ntrend = self.ntrend
        Gtrend = np.identity(ntrend)

        if ntrend == 2:
            Gtrend[0, 1] = 1

        return Gtrend

    def _build_Gregn(self):
        nregn = self.nregn
        Gregn = np.identity(nregn)
        return Gregn

    def _build_G(self):
        Gtrend = self._build_Gtrend()
        Gregn = self._build_Gregn()

        G = block_diag(Gtrend, Gregn)
        return G

    def _update_F(self, x: np.array = None):
        F = self.F
        # F[self.index_dict.get('reg'), 0] = np.ravel(x) - self.m[0]
        F[self.index_dict.get('reg'), 0] = np.ravel(x)
        return F

    def _build_P(self, G: np.array):
        return G @ self.C @ G.T

    def _build_W(self, P: np.array):
        if self.estimate_W:
            W = _build_W(mod=self, P=P)
        else:
            W = self.W
        return W
