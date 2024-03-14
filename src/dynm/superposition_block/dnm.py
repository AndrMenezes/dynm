"""Utils functions."""
import numpy as np
import copy
from dynm.sub_model.nullmodel import NullModel
from dynm.sub_model.autoregressive import AutoRegressive
from dynm.sub_model.transfer_function import TransferFunction
from scipy.linalg import block_diag


class DynamicNonLinearModel():
    """Class for fitting, forecast and update dynamic linear models."""

    def __init__(self, model_dict: dict, V: float = None):
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
        self.model_dict = copy.deepcopy(model_dict)
        self.V = V

        self._set_submodels()
        self._set_gamma_distribution_parameters()
        self._concatenate_regression_vector()
        self._concatenate_evolution_matrix()
        self._concatenate_prior_mean()
        self._concatenate_prior_covariance_matrix()
        self._set_submodels_block_index()
        self._set_parameters_name()

    def _set_submodels(self):
        if self.model_dict.get('arm') is not None:
            arm = AutoRegressive(
                m0=self.model_dict.get('arm').get('m0'),
                C0=self.model_dict.get('arm').get('C0'),
                discount_factors=self.model_dict.get('arm').get('del'),
                order=self.model_dict.get('arm').get('order'),
                W=self.model_dict.get('arm').get('W'))
        else:
            arm = NullModel()

        if self.model_dict.get('tfm') is not None:
            tfm = TransferFunction(
                m0=self.model_dict.get('tfm').get('m0'),
                C0=self.model_dict.get('tfm').get('C0'),
                discount_factors=self.model_dict.get('tfm').get('del'),
                lambda_order=self.model_dict.get('tfm').get('lambda_order'),
                gamma_order=self.model_dict.get('tfm').get('gamma_order'),
                ntfm=self.model_dict.get('tfm').get('ntfm'),
                W=self.model_dict.get('tfm').get('W'))
        else:
            tfm = NullModel()

        self.arm = arm
        self.tfm = tfm

    def _set_gamma_distribution_parameters(self):
        self.n = 1
        self.t = 0

        if self.V is None:
            self.d = 1
            self.s = 1
            self.estimate_V = True
        else:
            self.d = 0
            self.s = self.V
            self.estimate_V = False

        if self.arm.order > 0:
            self.v = 0
        else:
            self.v = self.s

    def _concatenate_regression_vector(self):
        self.F = np.vstack((self.arm.F, self.tfm.F))

    def _concatenate_evolution_matrix(self):
        self.G = block_diag(self.arm.G, self.tfm.G)

    def _concatenate_prior_mean(self):
        self.a = np.vstack((self.arm.m, self.tfm.m))
        self.m = np.vstack((self.arm.m, self.tfm.m))

    def _concatenate_prior_covariance_matrix(self):
        self.R = block_diag(self.arm.C, self.tfm.C)
        self.C = block_diag(self.arm.C, self.tfm.C)

    def _set_submodels_block_index(self):
        nparams_arm = len(self.arm.m)
        nparams_tfm = len(self.tfm.m)

        block_idx = np.cumsum([nparams_arm, nparams_tfm])

        idx_arm = np.arange(block_idx[0], block_idx[1])
        idx_tfm = np.arange(block_idx[1], block_idx[2])

        grid_arm_x, grid_arm_y = np.meshgrid(idx_arm, idx_arm)
        grid_tfm_x, grid_tfm_y = np.meshgrid(idx_tfm, idx_tfm)

        self.model_index_dict = {'arm': idx_arm, 'tfm': idx_tfm}

        self.grid_index_dict = {
            'arm': (grid_arm_x, grid_arm_y),
            'tfm': (grid_tfm_x, grid_tfm_y)
        }

    def _set_parameters_name(self):
        ar__response_labels = \
            ['xi_' + str(i+1) for i in range(self.arm.order)]

        ar__decay_labels = \
            ['phi_' + str(i+1) for i in range(self.arm.order)]

        tf__response_labels = \
            ['E_' + str(i+1) for i in range(self.tfm.lambda_order)]

        tf__decay_labels = \
            ['lambda_' + str(i+1) for i in range(self.tfm.lambda_order)]

        pulse_labels = \
            ['gamma_' + str(i+1) for i in range(self.tfm.gamma_order)]

        names_parameters = [
            ar__response_labels +
            ar__decay_labels +
            self.tfm.ntfm *
            (tf__response_labels + tf__decay_labels + pulse_labels)]

        self.names_parameters = names_parameters

    def _build_F(self):
        F = np.vstack((self.arm.F, self.tfm.F))

        return F

    def _build_G(self, x: np.array):
        G_arm = self.arm._build_G()
        G_tfm = self.tfm._build_G(x=x)

        G = block_diag(G_arm, G_tfm)

        return G

    def _build_W(self, G: np.array):
        grid_arm_x, grid_arm_y = self.grid_index_dict.get('arm')
        grid_tfm_x, grid_tfm_y = self.grid_index_dict.get('tfm')

        G_arm = G[grid_arm_x, grid_arm_y].T
        G_tfm = G[grid_tfm_x, grid_tfm_y].T

        P_arm = self.arm._build_P(G=G_arm)
        P_tfm = self.tfm._build_P(G=G_tfm)

        W_arm = self.arm._build_W(P=P_arm)
        W_tfm = self.tfm._build_W(P=P_tfm)

        W = block_diag(W_arm, W_tfm)

        return W

    def _build_h(self, G: np.array):
        grid_arm_x, grid_arm_y = self.grid_index_dict.get('arm')
        grid_tfm_x, grid_tfm_y = self.grid_index_dict.get('tfm')

        G_arm = G[grid_arm_x, grid_arm_y].T
        G_tfm = G[grid_tfm_x, grid_tfm_y].T

        h_arm = self.arm._build_h(G=G_arm)
        h_tfm = self.tfm._build_h(G=G_tfm)

        h = np.vstack([h_arm, h_tfm])

        return h

    def _update_submodels_moments(self):
        idx_arm = self.model_index_dict.get('arm')
        idx_tfm = self.model_index_dict.get('tfm')

        grid_arm_x, grid_arm_y = self.grid_index_dict.get('arm')
        grid_tfm_x, grid_tfm_y = self.grid_index_dict.get('tfm')

        self.arm.m = self.m[idx_arm]
        self.tfm.m = self.m[idx_tfm]

        self.arm.C = self.C[grid_arm_x, grid_arm_y]
        self.tfm.C = self.C[grid_tfm_x, grid_tfm_y]

        self.arm.s = self.s
        self.tfm.s = self.s

        if self.arm.order > 0:
            self.v = 0
        else:
            self.v = self.s
