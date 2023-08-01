"""Utils functions."""
import numpy as np
import pandas as pd
from src.algebra import _calc_predictive_mean_and_var
from src.dlm_autoregressive import AutoRegressive
from src.dlm_transfer_function import TransferFunction
from src.dlm import DLM
from src.dlm_nullmodel import NullModel
from src.utils import tidy_parameters
from scipy import stats
from scipy.linalg import block_diag


class Analysis():
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
        if model_dict.get('dlm') is not None:
            dlm = DLM(
                m0=model_dict.get('dlm').get('m0'),
                C0=model_dict.get('dlm').get('C0'),
                discount_factors=model_dict.get('dlm').get('del'),
                ntrend=model_dict.get('dlm').get('ntrend'),
                nregn=model_dict.get('dlm').get('nregn'),
                W=model_dict.get('dlm').get('W'))
        else:
            dlm = NullModel()

        if model_dict.get('arm') is not None:
            arm = AutoRegressive(
                m0=model_dict.get('arm').get('m0'),
                C0=model_dict.get('arm').get('C0'),
                discount_factors=model_dict.get('arm').get('del'),
                order=model_dict.get('arm').get('order'),
                W=model_dict.get('arm').get('W'))
        else:
            arm = NullModel()

        if model_dict.get('tfm') is not None:
            tfm = TransferFunction(
                m0=model_dict.get('tfm').get('m0'),
                C0=model_dict.get('tfm').get('C0'),
                discount_factors=model_dict.get('tfm').get('del'),
                order=model_dict.get('tfm').get('order'),
                ntfm=model_dict.get('tfm').get('ntfm'),
                W=model_dict.get('tfm').get('W'))
        else:
            tfm = NullModel()

        # Concatenate models ------------------------------------------------ #
        self.dlm = dlm
        self.arm = arm
        self.tfm = tfm
        self.model_dict = model_dict

        # Regression vector and evolution matrix
        self.F = np.vstack((dlm.F, arm.F, tfm.F))
        self.G = block_diag(dlm.G, arm.G, tfm.G)

        # Priori and Posterior moments
        self.a = np.vstack((dlm.m, arm.m, tfm.m))
        self.m = np.vstack((dlm.m, arm.m, tfm.m))
        self.R = block_diag(dlm.C, arm.C, tfm.C)
        self.C = block_diag(dlm.C, arm.C, tfm.C)

        # Gamma distribution parameters
        self.n = 1
        self.d = 1
        self.t = 0

        if V is None:
            self.s = 1
            self.estimate_V = True
        else:
            self.s = V
            self.estimate_V = False

        if self.arm.order > 0:
            self.v = 0
        else:
            self.v = self.s

        # Get index for blocks ---------------------------------------------- #
        p_dlm = len(dlm.m)
        p_ar = len(arm.m)
        p_tf = len(tfm.m)

        block_idx = np.cumsum([p_dlm, p_ar, p_tf])

        idx_dlm = np.arange(0, block_idx[0])
        idx_arm = np.arange(block_idx[0], block_idx[1])
        idx_tfm = np.arange(block_idx[1], block_idx[2])

        grid_dlm_x, grid_dlm_y = np.meshgrid(idx_dlm, idx_dlm)
        grid_arm_x, grid_arm_y = np.meshgrid(idx_arm, idx_arm)
        grid_tfm_x, grid_tfm_y = np.meshgrid(idx_tfm, idx_tfm)

        self.model_index_dict = {
            'dlm': idx_dlm, 'arm': idx_arm, 'tfm': idx_tfm}

        self.grid_index_dict = {
            'dlm': (grid_dlm_x, grid_dlm_y),
            'arm': (grid_arm_x, grid_arm_y),
            'tfm': (grid_tfm_x, grid_tfm_y)
        }

        # Get parameters names ---------------------------------------------- #
        level_labels = \
            ['intercept_' + str(i+1) for i in range(self.dlm.ntrend)]

        regn_labels = \
            ['beta_' + str(i+1) for i in range(self.dlm.nregn)]

        ar__response_labels = \
            ['xi_' + str(i+1) for i in range(self.arm.order)]

        ar__decay_labels = \
            ['phi_' + str(i+1) for i in range(self.arm.order)]

        tf__response_labels = \
            ['E_' + str(i+1) for i in range(self.tfm.order)]

        tf__decay_labels = \
            ['lambda_' + str(i+1) for i in range(self.tfm.order)]

        pulse_labels = ['gamma_1']

        names_parameters = (
            level_labels + regn_labels +
            ar__response_labels + ar__decay_labels +
            self.tfm.ntfm *
            (tf__response_labels + tf__decay_labels + pulse_labels))

        self.names_parameters = names_parameters

    def fit(self, y: np.ndarray, X: dict = {}, level: float = 0.05):
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
        nobs = len(y)
        dict_1step_forecast = {'t': [], 'y': [], 'f': [], 'q': [], 's': []}
        dict_state_params = {'m': [], 'C': []}
        Xt = {'dlm': [], 'tfm': []}
        copy_X = X.copy()

        if X.get('dlm') is None:
            x = np.array([None]*(nobs+1)).reshape(-1, 1)
            copy_X['dlm'] = x

        if X.get('tfm') is None:
            z = np.array([None]*(nobs+1)).reshape(-1, 1)
            copy_X['tfm'] = z

        for t in range(nobs):
            # Predictive distribution moments
            Xt['dlm'] = copy_X['dlm'][t, :]
            Xt['tfm'] = copy_X['tfm'][t, :]
            f, q = self._forecast(X=Xt)

            # Append results
            dict_1step_forecast['t'].append(t)
            dict_1step_forecast['y'].append(y[t])
            dict_1step_forecast['f'].append(np.ravel(f)[0])
            dict_1step_forecast['q'].append(np.ravel(q)[0])
            dict_1step_forecast['s'].append(np.ravel(self.s)[0])

            # Update model
            self.update(y=y[t], X=Xt)

            # Dict state params
            dict_state_params['m'].append(self.m)
            dict_state_params['C'].append(self.C)

        self.dict_state_params = dict_state_params
        df_predictive = pd.DataFrame(dict_1step_forecast)

        # Organize the posterior parameters
        df_posterior = tidy_parameters(
            dict_parameters=dict_state_params,
            entry_m="m", entry_v="C",
            names_parameters=self.names_parameters)

        dlm_lb = list(np.repeat('dlm', len(self.dlm.m)))
        arm_lb = list(np.repeat('arm', len(self.arm.m)))
        tfm_lb = list(np.repeat(
            ['tfm_' + str(i + 1) for i in range(self.tfm.ntfm)],
            2 * self.tfm.order + 1))

        mod_lb = nobs * (dlm_lb + arm_lb + tfm_lb)
        df_posterior["mod"] = mod_lb

        n_parms = len(df_posterior[["parameter", "mod"]].drop_duplicates())
        t_index = np.arange(0, len(df_posterior) / n_parms) + 1
        df_posterior["t"] = np.repeat(t_index, n_parms)
        df_posterior["t"] = df_posterior["t"].astype(int)

        # Compute credible intervals
        df_posterior["ci_lower"] = stats.t.ppf(
            q=level/2, df=df_posterior["t"].values + 1,
            loc=df_posterior["mean"].values,
            scale=np.sqrt(df_posterior["variance"].values) + 10e-300)
        df_posterior["ci_upper"] = stats.t.ppf(
            q=1-level/2,
            df=df_posterior["t"].values + 1,
            loc=df_posterior["mean"].values,
            scale=np.sqrt(df_posterior["variance"].values) + 10e-300)

        df_predictive["ci_lower"] = stats.t.ppf(
            q=level/2, df=df_predictive["t"].values + 1,
            loc=df_predictive["f"].values,
            scale=np.sqrt(df_predictive["q"].values) + 10e-300)
        df_predictive["ci_upper"] = stats.t.ppf(
            q=1-level/2, df=df_predictive["t"].values + 1,
            loc=df_predictive["f"].values,
            scale=np.sqrt(df_predictive["q"].values) + 10e-300)

        dict_results = {'filter': df_predictive, 'posterior': df_posterior}
        return dict_results

    def _forecast(self, X: dict):
        F_dlm = self.dlm._update_F(x=X.get('dlm'))
        F = np.vstack((F_dlm, self.arm.F, self.tfm.F))

        a, R = self._calc_aR(X=X)
        f, q = _calc_predictive_mean_and_var(F=F, a=a, R=R, s=self.v)
        return f, q

    def _k_steps_a_head_forecast(self, k: int, X: dict = {},
                                 level: float = 0.05):
        ak = self.m
        Rk = self.C

        dict_state_params = {'a': [], 'R': []}
        dict_kstep_forecast = {'t': [], 'f': [], 'q': []}

        # Organize transfer function values
        Xt = {'dlm': [], 'tfm': []}
        if X.get('dlm') is None:
            x = np.array([None]*(k+1)).reshape(-1, 1)
            X['dlm'] = x

        if X.get('tfm') is None:
            z = np.array([None]*(k+1)).reshape(-1, 1)
            X['tfm'] = z

        # K steps-a-head forecast
        for t in range(k):
            Xt['dlm'] = X['dlm'][t, :]
            Xt['tfm'] = X['tfm'][t, :]

            F_dlm = self.dlm._update_F(x=Xt.get('dlm'))
            F = np.vstack((F_dlm, self.arm.F, self.tfm.F))

            # Predictive distribution moments
            G = self._build_G(X=Xt)
            W = self._build_W(G=G)

            ak = G @ ak
            Rk = G @ Rk @ G.T + W

            # Predictive
            f, q = _calc_predictive_mean_and_var(
                F=F, a=ak, R=Rk, s=self.v)

            # Append results
            dict_kstep_forecast['t'].append(t)
            dict_kstep_forecast['f'].append(np.ravel(f)[0])
            dict_kstep_forecast['q'].append(np.ravel(q)[0])

            # Dict state params
            dict_state_params['a'].append(ak)
            dict_state_params['R'].append(Rk)

        df_predictive = pd.DataFrame(dict_kstep_forecast)

        # Organize the posterior parameters
        df_predict_aR = tidy_parameters(
            dict_parameters=dict_state_params,
            entry_m="a", entry_v="R",
            names_parameters=self.names_parameters)

        dlm_lb = list(np.repeat('dlm', len(self.dlm.m)))
        arm_lb = list(np.repeat('arm', len(self.arm.m)))
        tfm_lb = list(np.repeat(
            ['tfm_' + str(i + 1) for i in range(self.tfm.ntfm)],
            2 * self.tfm.order + 1))

        mod_lb = k * (dlm_lb + arm_lb + tfm_lb)
        df_predict_aR["mod"] = mod_lb

        n_parms = len(df_predict_aR[["parameter", "mod"]].drop_duplicates())
        t_index = np.arange(0, len(df_predict_aR) / n_parms) + 1
        df_predict_aR["t"] = np.repeat(t_index, n_parms)
        df_predict_aR["t"] = df_predict_aR["t"].astype(int)

        # Compute credible intervals
        df_predict_aR["ci_lower"] = stats.t.ppf(
            q=level/2, df=self.t + 1,
            loc=df_predict_aR["mean"].values,
            scale=np.sqrt(df_predict_aR["variance"].values) + 10e-300)
        df_predict_aR["ci_upper"] = stats.t.ppf(
            q=1-level/2, df=self.t + 1,
            loc=df_predict_aR["mean"].values,
            scale=np.sqrt(df_predict_aR["variance"].values) + 10e-300)

        df_predictive["ci_lower"] = stats.t.ppf(
            q=level/2, df=self.t + 1,
            loc=df_predictive["f"].values,
            scale=np.sqrt(df_predictive["q"].values) + 10e-300)
        df_predictive["ci_upper"] = stats.t.ppf(
            q=1-level/2, df=self.t + 1,
            loc=df_predictive["f"].values,
            scale=np.sqrt(df_predictive["q"].values) + 10e-300)

        dict_results = {'filter': df_predictive, 'parameters': df_predict_aR}
        return dict_results

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

    def _build_h(self, G: np.array):
        grid_dlm_x, grid_dlm_y = self.grid_index_dict.get('dlm')
        grid_arm_x, grid_arm_y = self.grid_index_dict.get('arm')
        grid_tfm_x, grid_tfm_y = self.grid_index_dict.get('tfm')

        G_dlm = G[grid_dlm_x, grid_dlm_y].T
        G_arm = G[grid_arm_x, grid_arm_y].T
        G_tfm = G[grid_tfm_x, grid_tfm_y].T

        h_dlm = np.zeros([G_dlm.shape[0], 1])
        h_arm = self.arm._build_h(G=G_arm)
        h_tfm = self.tfm._build_h(G=G_tfm)

        h = np.vstack([h_dlm, h_arm, h_tfm])

        return h

    def _calc_aR(self, X: dict):
        G = self._build_G(X=X)
        W = self._build_W(G=G)
        h = self._build_h(G=G)

        a = G @ self.m + h
        P = G @ self.C @ G.T
        R = P + W

        return a, R

    def update(self, y: float, X: dict):
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
        self.t += 1

        if y is None or np.isnan(y):
            self.m = self.a
            self.C = self.R

            # Get priors a, R for time t + 1 from the posteriors m, C
            G = self._build_G(X=X)
            h = self._build_h(G=G)

            self.a = G @ self.m + h
            self.R = G @ self.C @ self.G.T
        else:
            self.dlm.F = self.dlm._update_F(x=X.get('dlm'))
            self.F = np.vstack((self.dlm.F, self.arm.F, self.tfm.F))

            # Need a better solution for this!
            if self.arm.order > 0:
                self.v = 0
            else:
                self.v = self.s

            a, R = self._calc_aR(X=X)
            f, q = _calc_predictive_mean_and_var(F=self.F, a=a, R=R, s=self.v)

            A = (R @ self.F) / q
            et = y - f

            # Estimate observational variance
            if self.estimate_V:
                r = (self.n + et**2 / q) / (self.n + 1)
                self.n = self.n + 1
                self.s = self.s * r
            else:
                r = 1

            # Kalman filter update
            self.a = a
            self.R = R
            self.m = a + A * et
            self.C = r * (R - q * A @ A.T)

            # Update submodels mean and covariance posterior
            idx_dlm = self.model_index_dict.get('dlm')
            idx_arm = self.model_index_dict.get('arm')
            idx_tfm = self.model_index_dict.get('tfm')

            grid_dlm_x, grid_dlm_y = self.grid_index_dict.get('dlm')
            grid_arm_x, grid_arm_y = self.grid_index_dict.get('arm')
            grid_tfm_x, grid_tfm_y = self.grid_index_dict.get('tfm')

            self.dlm.m = self.m[idx_dlm]
            self.arm.m = self.m[idx_arm]
            self.tfm.m = self.m[idx_tfm]

            self.dlm.C = self.C[grid_dlm_x, grid_dlm_y]
            self.arm.C = self.C[grid_arm_x, grid_arm_y]
            self.tfm.C = self.C[grid_tfm_x, grid_tfm_y]

            self.dlm.s = self.s
            self.arm.s = self.s
            self.tfm.s = self.s
