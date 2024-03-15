"""Dynamic Linear Model with transfer function."""
import numpy as np
import copy
from scipy.linalg import block_diag
from dynm.sub_model.polynomial import Polynomial
from dynm.sub_model.regression import Regression
from dynm.sub_model.seasonal_fourier import SeasonalFourier
from dynm.sub_model.nullmodel import NullModel


class DynamicLinearModel():
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
        if self.model_dict.get('polynomial') is not None:
            submod_dict = self.model_dict.get('polynomial')
            self.ntrend = submod_dict.get('ntrend')

            polynomial_mod = Polynomial(
                m0=submod_dict.get('m0'),
                C0=submod_dict.get('C0'),
                discount=submod_dict.get('discount'),
                ntrend=submod_dict.get('ntrend'),
                W=submod_dict.get('W'))
        else:
            polynomial_mod = NullModel()

        if self.model_dict.get('regression') is not None:
            submod_dict = self.model_dict.get('regression')
            self.nregn = submod_dict.get('nregn')

            regression_mod = Regression(
                m0=submod_dict.get('m0'),
                C0=submod_dict.get('C0'),
                discount=submod_dict.get('discount'),
                nregn=submod_dict.get('nregn'),
                W=submod_dict.get('W'))

        else:
            regression_mod = NullModel()

        if self.model_dict.get('seasonal') is not None:
            submod_dict = self.model_dict.get('seasonal')

            self.nseas = 2 * len(submod_dict.get('seas_harm_components'))
            self.seas_period = submod_dict.get('seas_period')
            self.seas_harm_components = submod_dict.get('seas_harm_components')

            seasonal_mod = SeasonalFourier(
                m0=submod_dict.get('m0'),
                C0=submod_dict.get('C0'),
                discount=submod_dict.get('discount'),
                seas_period=submod_dict.get('seas_period'),
                seas_harm_components=submod_dict.get('seas_harm_components'),
                W=submod_dict.get('W'))

        else:
            seasonal_mod = NullModel()

        self.polynomial_model = polynomial_mod
        self.regression_model = regression_mod
        self.seasonal_model = seasonal_mod

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

    def _concatenate_regression_vector(self):
        self.F = np.vstack((self.polynomial_model.F,
                            self.regression_model.F,
                            self.seasonal_model.F))

    def _concatenate_evolution_matrix(self):
        self.G = block_diag(self.polynomial_model.G,
                            self.regression_model.G,
                            self.seasonal_model.G)

    def _concatenate_prior_mean(self):
        self.a = np.vstack((self.polynomial_model.m,
                            self.regression_model.m,
                            self.seasonal_model.m))
        self.m = np.vstack((self.polynomial_model.m,
                            self.regression_model.m,
                            self.seasonal_model.m))

    def _concatenate_prior_covariance_matrix(self):
        self.R = block_diag(self.polynomial_model.C,
                            self.regression_model.C,
                            self.seasonal_model.C)
        self.C = block_diag(self.polynomial_model.C,
                            self.regression_model.C,
                            self.seasonal_model.C)

    def _set_submodels_block_index(self):
        nparams_polynomial = len(self.polynomial_model.m)
        nparams_regression = len(self.regression_model.m)
        nparams_seasonal = len(self.seasonal_model.m)

        block_idx = np.cumsum([nparams_polynomial,
                               nparams_regression,
                               nparams_seasonal])

        idx_poly = np.arange(0, block_idx[0])
        idx_regn = np.arange(block_idx[0], block_idx[1])
        idx_seas = np.arange(block_idx[1], block_idx[2])

        grid_poly_x, grid_poly_y = np.meshgrid(idx_poly, idx_poly)
        grid_regn_x, grid_regn_y = np.meshgrid(idx_regn, idx_regn)
        grid_seas_x, grid_seas_y = np.meshgrid(idx_seas, idx_seas)

        self.model_index_dict = {
            'polynomial': idx_poly,
            'regression': idx_regn,
            'seasonal': idx_seas
        }

        self.grid_index_dict = {
            'polynomial': (grid_poly_x, grid_poly_y),
            'regression': (grid_regn_x, grid_regn_y),
            'seasonal': (grid_seas_x, grid_seas_y)
        }

    def _set_parameters_name(self):
        level_labels = \
            ['intercept_' + str(i+1)
             for i in range(self.polynomial_model.ntrend)]

        regn_labels = \
            ['beta_' + str(i+1)
             for i in range(self.regression_model.nregn)]

        seas_labels = \
            ['seas_harm_' + str(i+1)
             for i in range(self.seasonal_model.nseas)]

        names_parameters = (level_labels + regn_labels + seas_labels)
        self.names_parameters = names_parameters

    def _build_F(self, x: np.array = None):
        F_poly = self.polynomial_model.F
        F_regn = self.regression_model._update_F(x=x)
        F_seas = self.seasonal_model.F

        F = np.block([F_poly, F_regn, F_seas]).reshape(-1, 1)

        return F

    def _build_G(self):
        G_poly = self.polynomial_model.G
        G_regn = self.regression_model.G
        G_seas = self.seasonal_model.G

        G = block_diag(G_poly, G_regn, G_seas)

        return G

    def _build_W(self):
        P_poly = self.polynomial_model._build_P()
        P_regn = self.regression_model._build_P()
        P_seas = self.seasonal_model._build_P()

        W_poly = self.polynomial_model._build_W(P=P_poly)
        W_regn = self.regression_model._build_W(P=P_regn)
        W_seas = self.seasonal_model._build_W(P=P_seas)

        W = block_diag(W_poly, W_regn, W_seas)

        return W

    def _update_submodels_F(self):
        idx_poly = self.model_index_dict.get('polynomial')
        idx_regn = self.model_index_dict.get('regression')
        idx_seas = self.model_index_dict.get('seasonal')

        self.polynomial_model.F = self.F[idx_poly]
        self.regression_model.F = self.F[idx_regn]
        self.seasonal_model.F = self.F[idx_seas]

    def _update_submodels_G(self):
        grid_poly_x, grid_poly_y = self.grid_index_dict.get('polynomial')
        grid_regn_x, grid_regn_y = self.grid_index_dict.get('regression')
        grid_seas_x, grid_seas_y = self.grid_index_dict.get('seasonal')

        self.polynomial_model.G = self.G[grid_poly_x, grid_poly_y]
        self.regression_model.G = self.G[grid_regn_x, grid_regn_y]
        self.seasonal_model.G = self.G[grid_seas_x, grid_seas_y]

    def _update_submodels_moments(self):
        idx_poly = self.model_index_dict.get('polynomial')
        idx_regn = self.model_index_dict.get('regression')
        idx_seas = self.model_index_dict.get('seasonal')

        grid_poly_x, grid_poly_y = self.grid_index_dict.get('polynomial')
        grid_regn_x, grid_regn_y = self.grid_index_dict.get('regression')
        grid_seas_x, grid_seas_y = self.grid_index_dict.get('seasonal')

        self.polynomial_model.m = self.m[idx_poly]
        self.regression_model.m = self.m[idx_regn]
        self.seasonal_model.m = self.m[idx_seas]

        self.polynomial_model.C = self.C[grid_poly_x, grid_poly_y]
        self.regression_model.C = self.C[grid_regn_x, grid_regn_y]
        self.seasonal_model.C = self.C[grid_seas_x, grid_seas_y]
