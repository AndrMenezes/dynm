"""Dynamic Linear Model with transfer function."""
import numpy as np
import pandas as pd
import copy
from dynm.utils.algebra import _build_W
from scipy.linalg import block_diag
from dynm.model.polynomial import Polynomial
from dynm.model.regression import Regression
from dynm.model.seasonal_fourier import SeasonalFourier
from dynm.sequencial.filter import _foward_filter
from dynm.sequencial.smooth import _backward_smoother
from dynm.utils.summary import summary
from dynm.utils.format_result import _build_predictive_df, _build_posterior_df
from dynm.utils.format_input import set_X_dict
from dynm.utils.summary import get_predictive_log_likelihood
from dynm.utils.algebra import _calc_predictive_mean_and_var


class BayesianDynamicModel():
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
        copy_model_dict = copy.deepcopy(model_dict)

        if model_dict.get('polynomial') is not None:
            submod_dict = model_dict.get('polynomial')
            self.ntrend = submod_dict.get('ntrend')

            polynomial_mod = Polynomial(
                m0=submod_dict.get('m0'),
                C0=submod_dict.get('C0'),
                discount=submod_dict.get('discount'),
                ntrend=submod_dict.get('ntrend'),
                W=submod_dict.get('W'))

            # Update dict
            copy_model_dict['polynomial']['mod'] = polynomial_mod

        if model_dict.get('regression') is not None:
            submod_dict = model_dict.get('regression')
            self.nregn = submod_dict.get('nregn')

            regression_mod = Regression(
                m0=submod_dict.get('m0'),
                C0=submod_dict.get('C0'),
                discount=submod_dict.get('discount'),
                nregn=submod_dict.get('nregn'),
                W=submod_dict.get('W'))

            # Update dict
            copy_model_dict['regression']['mod'] = regression_mod

        if model_dict.get('seasonal_fourier') is not None:
            submod_dict = model_dict.get('seasonal_fourier')

            self.nseas = 2 * len(submod_dict.get('seas_harm_components'))
            self.seas_period = submod_dict.get('seas_period')
            self.seas_harm_components = submod_dict.get('seas_harm_components')

            seasonal_fourier_mod = SeasonalFourier(
                m0=submod_dict.get('m0'),
                C0=submod_dict.get('C0'),
                discount=submod_dict.get('discount'),
                seas_period=submod_dict.get('seas_period'),
                seas_harm_components=submod_dict.get('seas_harm_components'),
                W=submod_dict.get('W'))

            # Update dict
            copy_model_dict['seasonal_fourier']['mod'] = seasonal_fourier_mod

        # Gamma distribution parameters
        self.n = 1
        self.t = 0

        if V is None:
            self.d = 1
            self.s = 1
            self.estimate_V = True
        else:
            self.d = 0
            self.s = V
            self.estimate_V = False

        if self.arm.order > 0:
            self.v = 0
        else:
            self.v = self.s

        # Concatenate models -------- #
        # Regression vector and evolution matrix
        self.F = np.vstack([copy_model_dict[m]['mod'].F.reshape(-1, 1)
                            for m in copy_model_dict.keys()])

        self.G = block_diag(*(copy_model_dict[m]['mod'].G
                            for m in copy_model_dict.keys()))

        # # Priori and Posterior moments
        self.m = np.vstack([copy_model_dict[m]['mod'].m.reshape(-1, 1)
                            for m in copy_model_dict.keys()])

        self.C = block_diag(*(copy_model_dict[m]['mod'].C
                            for m in copy_model_dict.keys()))

        # Get index for blocks
        self.model_dict = copy_model_dict
        nparams = [copy_model_dict[m]['mod'].m.shape[0]
                   for m in copy_model_dict.keys()]
        block_idx = np.cumsum(nparams)
        # index_dict = {m: block_idx for m in copy_model_dict.keys()}

        # self.index_dict = {
        #     'trend': np.arange(0, block_idx[0]),
        #     'reg': np.arange(block_idx[0], block_idx[1])
        # }

        # Validate entries section

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

    def _calc_fq(self, X: dict = {}):
        # Predictive distribution moments
        F = self._build_F(X=X)
        G = self._build_G(X=X)
        W = self._build_W(G=G)
        h = self._build_h(G=G)

        a = G @ self.m + h
        R = G @ self.C @ G.T + W

        f, q = _calc_predictive_mean_and_var(F=F, a=a, R=R, s=self.v)
        return f, q

    def summary(self):
        str_summary = summary(mod=self)
        return str_summary

    def _build_F(self, X: dict = {}):
        self.F = np.vstack([
            self.model_dict[m]['mod']._update_F(x=X.get(m))
            for m in self.model_dict.keys()])

        F_dlm = self.dlm._update_F(x=X.get('dlm'))
        F = np.vstack((F_dlm, self.arm.F, self.tfm.F))

        return F

    def _build_G(self, X: dict):
        G_dlm = self.dlm.G
        G_arm = self.arm._build_G()
        G_tfm = self.tfm._build_G(x=X.get('tfm'))

        G = block_diag(G_dlm, G_arm, G_tfm)

        return G

    def _build_W(self, G: np.array):
        grid_dlm_x, grid_dlm_y = self.grid_index_dict.get('dlm')
        grid_arm_x, grid_arm_y = self.grid_index_dict.get('arm')
        grid_tfm_x, grid_tfm_y = self.grid_index_dict.get('tfm')

        G_dlm = G[grid_dlm_x, grid_dlm_y].T
        G_arm = G[grid_arm_x, grid_arm_y].T
        G_tfm = G[grid_tfm_x, grid_tfm_y].T

        P_dlm = self.dlm._build_P(G=G_dlm)
        P_arm = self.arm._build_P(G=G_arm)
        P_tfm = self.tfm._build_P(G=G_tfm)

        W_dlm = self.dlm._build_W(P=P_dlm)
        W_arm = self.arm._build_W(P=P_arm)
        W_tfm = self.tfm._build_W(P=P_tfm)

        W = block_diag(W_dlm, W_arm, W_tfm)

        return W
