"""Dynamic Linear Model with transfer function."""
import numpy as np
from dynm.utils.algebra import _build_W
from scipy.linalg import block_diag


class DLM():
    """Class for fitting, forecast and update dynamic linear models."""

    def __init__(self,
                 m0: np.ndarray,
                 C0: np.ndarray,
                 ntrend: int, nregn: int,
                 seas_period: int = None, seas_harm_components: list = None,
                 discount_factors: float = None,
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
        self.nseas = 2 * len(seas_harm_components)
        self.seas_period = seas_period
        self.seas_harm_components = seas_harm_components
        self.m = m0.reshape(-1, 1)  # Validar entrada de dimensões
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

        Fregn = np.ones(nregn) * x
        return Fregn

    def _build_Fseas(self):
        seas_harm_components = self.seas_harm_components

        p = len(seas_harm_components)
        n = 2 * p

        Fseas = np.zeros([n, 1])
        Fseas[0:n:2] = 1

        return Fseas.T

    def _build_F(self, x: np.array = None):
        Ftrend = self._build_Ftrend()
        Fregn = self._build_Fregn(x=x)
        Fseas = self._build_Fseas()
        F = np.block([Ftrend, Fregn, Fseas]).reshape(-1, 1)
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

    def _build_Gseas(self):
        seas_period = self.seas_period
        seas_harm_components = self.seas_harm_components

        p = len(seas_harm_components)
        n = 2 * p
        Gseas = np.zeros([n, n])

        for j in range(p):
            c = np.cos(2*np.pi*seas_harm_components[j] / seas_period)
            s = np.sin(2*np.pi*seas_harm_components[j] / seas_period)
            idx = 2*j
            Gseas[idx:(idx+2), idx:(idx+2)] = np.array([[c, s], [-s, c]])

        return Gseas

    def _build_G(self):
        Gtrend = self._build_Gtrend()
        Gregn = self._build_Gregn()
        Gseas = self._build_Gseas()

        G = block_diag(Gtrend, Gregn, Gseas)
        return G

    def _update_F(self, x: np.array = None):
        F = self.F
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
