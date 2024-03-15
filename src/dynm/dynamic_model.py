"""Utils functions."""
import numpy as np
import pandas as pd
from copy import deepcopy as copy
from dynm.utils.algebra import _calc_predictive_mean_and_var
from dynm.superposition_block.dlm import DynamicLinearModel
from dynm.superposition_block.dnm import DynamicNonLinearModel
from scipy.linalg import block_diag
from dynm.utils.summary import summary
from dynm.utils.format_result import _build_predictive_df, _build_posterior_df
from dynm.utils.format_input import set_X_dict
from dynm.sequencial.filter import _foward_filter
from dynm.sequencial.smooth import _backward_smoother
from dynm.utils.summary import get_predictive_log_likelihood


class BayesianDynamicModel():
    """Class for fitting, forecast and update dynamic linear models."""

    def __init__(self, model_dict: dict, V: float = None, W: dict = None):
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
        self.model_dict = copy(model_dict)
        self.V = V

        self._set_superposition_blocks()

        self._set_gamma_distribution_parameters()

        self._concatenate_regression_vector()

        self._concatenate_evolution_matrix()

        self._concatenate_prior_mean()

        self._concatenate_prior_covariance_matrix()

        self._set_superposition_block_index()

        self._set_parameters_name()

    def _set_superposition_blocks(self):
        dlm = DynamicLinearModel(model_dict=self.model_dict, V=self.V)
        dnm = DynamicNonLinearModel(model_dict=self.model_dict, V=self.V)

        self.dlm = dlm
        self.dnm = dnm

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

        if self.dnm.autoregressive_model.order > 0:
            self.v = 0
        else:
            self.v = self.s

    def _concatenate_regression_vector(self):
        self.F = np.vstack((self.dlm.F, self.dnm.F))

    def _concatenate_evolution_matrix(self):
        self.G = block_diag(self.dlm.G, self.dnm.G)

    def _concatenate_prior_mean(self):
        self.a = np.vstack((self.dlm.m, self.dnm.m))
        self.m = np.vstack((self.dlm.m, self.dnm.m))

    def _concatenate_prior_covariance_matrix(self):
        self.R = block_diag(self.dlm.C, self.dnm.C)
        self.C = block_diag(self.dlm.C, self.dnm.C)

    def _set_superposition_block_index(self):
        nparams_dlm = len(self.dlm.m)
        nparams_dnm = len(self.dnm.m)

        block_idx = np.cumsum([nparams_dlm, nparams_dnm])

        idx_dlm = np.arange(0, block_idx[0])
        idx_dnm = np.arange(block_idx[0], block_idx[1])

        grid_dlm_x, grid_dlm_y = np.meshgrid(idx_dlm, idx_dlm)
        grid_dnm_x, grid_dnm_y = np.meshgrid(idx_dnm, idx_dnm)

        self.model_index_dict = {
            'dlm': idx_dlm,
            'dnm': idx_dnm
        }

        self.grid_index_dict = {
            'dlm': (grid_dlm_x, grid_dlm_y),
            'dnm': (grid_dnm_x, grid_dnm_y)
        }

    def _set_parameters_name(self):
        dlm_names_parameters = self.dlm.names_parameters
        dnm_names_parameters = self.dnm.names_parameters

        names_parameters = dlm_names_parameters.extend(dnm_names_parameters)
        self.names_parameters = names_parameters

    def _build_F(self, x: np.array = None):
        F_dlm = self.dlm._build_F(x=x)
        F_dnm = self.dnm.F

        F = np.vstack((F_dlm, F_dnm))

        return F

    def _build_G(self, x: np.array = None):
        G_dlm = self.dlm.G
        G_dnm = self.dnm._build_G(x=x)

        G = block_diag(G_dlm, G_dnm)

        return G

    def _build_W(self):
        W_dlm = self.dlm._build_W()
        W_dnm = self.dnm._build_W()

        W = block_diag(W_dlm, W_dnm)

        return W

    def _build_h(self):
        h_dlm = np.zeros([self.dlm.G.shape[0], 1])
        h_dnm = self.dnm._build_h()

        h = np.vstack([h_dlm, h_dnm])

        return h

    def _calc_prior_mean_and_var(self):
        a = self.G @ self.m + self.h
        P = self.G @ self.C @ self.G.T
        R = (P + self.W)

        return a, R

    def _calc_predictive_mean_and_var(self):
        f = np.ravel(self.F.T @ self.a)[0]
        q = np.ravel(self.F.T @ self.R @ self.F + self.s)[0]
        return f, q

    def _update(self, y: float, X: dict):
        self.t += 1

        self.F = self._build_F(x=X.get('regression'))
        self.G = self._build_G(x=X.get('transfer_function'))

        self._update_superposition_block_F()
        self._update_superposition_block_G()

        self.W = self._build_W()
        self.h = self._build_h()

        if y is None or np.isnan(y):
            self.m = self.a
            self.C = self.R

            self.a = self.G @ self.m + self.h
            self.R = self.G @ self.C @ self.G.T

            self._update_superposition_block_moments()
        else:
            self.a, self.R = self._calc_prior_mean_and_var()
            self.f, self.q = self._calc_predictive_mean_and_var()

            self.A = (self.R @ self.F) / self.q
            self.e = y - self.f

            self._estimate_observational_variance()

            self._kalman_filter_update()

            self._update_superposition_block_moments()

    def _estimate_observational_variance(self):
        if self.estimate_V:
            self.r = (self.n + self.e**2 / self.q) / (self.n + 1)
            self.n = self.n + 1
            self.s = self.s * self.r
            self.d = self.s * self.n
        else:
            self.r = 1

    def _kalman_filter_update(self):
        self.a = self.a
        self.R = self.R
        self.m = self.a + self.A * self.e
        self.C = self.r * (self.R - self.q * self.A @ self.A.T)
        self._update_superposition_block_moments()

    def _update_superposition_block_F(self):
        idx_dlm = self.model_index_dict.get('dlm')
        idx_dnm = self.model_index_dict.get('dnm')

        self.dlm.F = self.F[idx_dlm]
        self.dnm.F = self.F[idx_dnm]

        self.dlm._update_submodels_F()
        self.dnm._update_submodels_F()

    def _update_superposition_block_G(self):
        grid_dlm_x, grid_dlm_y = self.grid_index_dict.get('dlm')
        grid_dnm_x, grid_dnm_y = self.grid_index_dict.get('dnm')

        self.dlm.G = self.G[grid_dlm_x, grid_dlm_y]
        self.dnm.G = self.G[grid_dnm_x, grid_dnm_y]

        self.dlm._update_submodels_G()
        self.dnm._update_submodels_G()

    def _update_superposition_block_moments(self):
        idx_dlm = self.model_index_dict.get('dlm')
        idx_dnm = self.model_index_dict.get('dnm')

        grid_dlm_x, grid_dlm_y = self.grid_index_dict.get('dlm')
        grid_dnm_x, grid_dnm_y = self.grid_index_dict.get('dnm')

        self.dlm.m = self.m[idx_dlm]
        self.dnm.m = self.m[idx_dnm]

        self.dlm.C = self.C[grid_dlm_x, grid_dlm_y]
        self.dnm.C = self.C[grid_dnm_x, grid_dnm_y]

        self.dlm.s = self.s
        self.dnm.s = self.s

        if self.dnm.autoregressive_model.order > 0:
            self.v = 0
        else:
            self.v = self.s
            self.dnm.v = self.s

        self.dlm._update_submodels_moments()
        self.dnm._update_submodels_moments()

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
        foward_dict = _foward_filter(mod=self, y=y, X=X, level=level)
        self.dict_filter = copy(foward_dict.get('filter'))
        self.dict_state_params = copy(foward_dict.get('state_params'))
        self.dict_state_evolution = copy(foward_dict.get('state_evolution'))

        if smooth:
            backward_dict = _backward_smoother(mod=self, X=X, level=level)
            self.dict_smooth = copy(backward_dict.get('smooth'))
            self.dict_smooth_params = copy(backward_dict.get('smooth_params'))

        self.llk = get_predictive_log_likelihood(mod=self)

        return self

    def _predict(
            self,
            k: int,
            X: dict = {},
            level: float = 0.05):
        ak = copy(self.m)
        Rk = copy(self.C)

        dict_state_params = {'a': [], 'R': []}
        dict_kstep_forecast = {'t': [], 'f': [], 'q': []}

        Xt = {'dlm': [], 'tfm': []}
        copy_X = set_X_dict(mod=self, nobs=k, X=X)

        # K steps-a-head forecast
        for t in range(k):
            Xt['dlm'] = copy_X['dlm'][t, :]
            Xt['tfm'] = copy_X['tfm'][t, :, :]

            # Predictive distribution moments
            F = self._build_F(X=Xt)
            G = self._build_G(X=Xt)
            W = self._build_W(G=G)
            h = self._build_h(G=G)

            ak = G @ ak + h
            Rk = G @ Rk @ G.T + W

            # Predictive
            f, q = _calc_predictive_mean_and_var(
                F=F, a=ak, R=Rk, s=self.v)

            # Append results
            dict_kstep_forecast['t'].append(t+1)
            dict_kstep_forecast['f'].append(np.ravel(f)[0])
            dict_kstep_forecast['q'].append(np.ravel(q)[0])

            # Dict state params
            dict_state_params['a'].append(ak)
            dict_state_params['R'].append(Rk)

        df_predictive = pd.DataFrame(dict_kstep_forecast)

        # Get posterior and predictive dataframes
        df_predictive = _build_predictive_df(
            mod=self, dict_predict=dict_kstep_forecast, level=level)

        df_predict_aR = _build_posterior_df(
            mod=self,
            dict_posterior=dict_state_params,
            entry_m="a",
            entry_v="R",
            t=k,
            level=level)

        # Creat dict of results
        dict_results = {'predictive': df_predictive,
                        'parameters': df_predict_aR}
        return dict_results

    def _summary(self):
        str_summary = summary(mod=self)
        return str_summary
