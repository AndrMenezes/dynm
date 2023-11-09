"""Test autoregressive model parameters estimation."""
import numpy as np
import unittest
from dynm.analysis import Analysis
from copy import copy

# Simulating the data
nobs = 500
sd_y = 0.02
true_lambda_1 = .8
true_lambda_2 = -.5
true_lambda_3 = .8
true_lambda_4 = -.5
true_gamma_1 = 5
true_gamma_2 = 4

m0 = np.array([0, 0, .8, -.5, 5, 0, 0, .7, -.4, 4])
w0 = np.array([0, 0, 0, 0, 1e-4, 0, 0, 0, 0, 1e-4])

y = np.zeros(nobs)
x = np.zeros([nobs, 2])
E1 = np.zeros(nobs)
E2 = np.zeros(nobs)
E3 = np.zeros(nobs)
E4 = np.zeros(nobs)

lambda_1 = np.zeros(nobs)
lambda_2 = np.zeros(nobs)
lambda_3 = np.zeros(nobs)
lambda_4 = np.zeros(nobs)

gamma_1 = np.zeros(nobs)
gamma_2 = np.zeros(nobs)

# Initial values
E1[0] = m0[0]
E2[0] = m0[1]

lambda_1[0] = m0[2]
lambda_2[0] = m0[3]

gamma_1[0] = m0[4]

# ---- #
E3[0] = m0[5]
E4[0] = m0[6]

lambda_3[0] = m0[7]
lambda_4[0] = m0[8]

gamma_2[0] = m0[9]

W = np.identity(10)
np.fill_diagonal(W, val=w0)

mu_x = np.array([0, 0])
sigma_x = np.identity(2) * .10

# Simulate time series
np.random.seed(1111)
for t in range(0, nobs-1):
    vt = np.random.normal(loc=0, scale=sd_y, size=1)
    omega = np.random.multivariate_normal(mean=m0 * 0.0, cov=W)
    x[t+1, :] = np.random.multivariate_normal(mean=mu_x, cov=sigma_x, size=1)

    # Evolution
    gamma_1[t+1] = gamma_1[t] + omega[4]
    gamma_2[t+1] = gamma_2[t] + omega[9]

    lambda_1[t+1] = lambda_1[t] + omega[2]
    lambda_2[t+1] = lambda_2[t] + omega[3]
    lambda_3[t+1] = lambda_3[t] + omega[7]
    lambda_4[t+1] = lambda_4[t] + omega[8]

    E1[t+1] = lambda_1[t] * E1[t] + lambda_2[t] * \
        E2[t] + gamma_1[t+1] * x[t+1, 0] + omega[0]
    E2[t+1] = E1[t] + omega[1]

    E3[t+1] = lambda_3[t] * E3[t] + lambda_4[t] * \
        E4[t] + gamma_2[t+1] * x[t+1, 1] + omega[5]
    E4[t+1] = E3[t] + omega[6]

    # Observation
    y[t+1] = E1[t+1] + E3[t+1] + vt

# Estimation
m0 = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0])
C0 = np.identity(10) * 9

tfm_del = np.repeat(.995, 10)

X = {'tfm': x}


class TestAnalysisTF(unittest.TestCase):
    """Tests Analysis results for Transfer Function Model."""

    def test__estimates_known_W(self):
        """Test parameters estimation with know W."""
        model_dict = {
            'tfm': {'m0': m0, 'C0': C0, 'order': 2, "W": W, "ntfm": 2}
        }

        # Fit
        mod = Analysis(model_dict=model_dict, V=sd_y**2)
        fit_results = mod.fit(y=y, X=X)
        m = mod.m

        self.assertTrue(np.abs(m[2] - true_lambda_1) < .1)
        self.assertTrue(np.abs(m[3] - true_lambda_2) < .1)
        self.assertTrue(np.abs(m[4] - true_gamma_1) < 1)
        self.assertTrue(np.abs(m[7] - true_lambda_3) < .1)
        self.assertTrue(np.abs(m[8] - true_lambda_4) < .1)
        self.assertTrue(np.abs(m[9] - true_gamma_2) < 1)

    def test__estimates_discount(self):
        """Test parameters estimation with discount."""
        model_dict = {
            'tfm': {'m0': m0, 'C0': C0, 'order': 2, "ntfm": 2,
                    "del": np.repeat(1, 10)}
        }

        # Fit
        mod = Analysis(model_dict=model_dict, V=sd_y**2)
        fit_results = mod.fit(y=y, X=X)
        m = mod.m

        self.assertTrue(np.abs(m[2] - true_lambda_1) < .1)
        self.assertTrue(np.abs(m[3] - true_lambda_2) < .1)
        self.assertTrue(np.abs(m[4] - true_gamma_1) < 1)
        self.assertTrue(np.abs(m[7] - true_lambda_3) < .1)
        self.assertTrue(np.abs(m[8] - true_lambda_4) < .1)
        self.assertTrue(np.abs(m[9] - true_gamma_2) < 1)

    def test__analysis_with_nan(self):
        """Test parameters estimation with nan in y."""
        model_dict = {
            'tfm': {'m0': m0, 'C0': C0, 'order': 2, "ntfm": 2,
                    "del": np.repeat(1, 10)}
        }

        copy_y = copy(y)
        copy_y[50] = np.nan

        # Fit
        mod = Analysis(model_dict=model_dict, V=sd_y**2)
        fit_results = mod.fit(y=copy_y, X=X)

        forecast_df = fit_results.get('filter')
        m = mod.m

        self.assertTrue(np.abs(m[2] - true_lambda_1) < .2)
        self.assertTrue(np.abs(m[3] - true_lambda_2) < .2)
        self.assertTrue(np.abs(m[4] - true_gamma_1) < 1)
        self.assertTrue(np.abs(m[7] - true_lambda_3) < .2)
        self.assertTrue(np.abs(m[8] - true_lambda_4) < .2)
        self.assertTrue(np.abs(m[9] - true_gamma_2) < 1)
        self.assertTrue(forecast_df.f.notnull().all())

    def test__k_steps_ahead_forecast_performance(self):
        """Test k steps a head performance."""
        model_dict = {
            'tfm': {'m0': m0, 'C0': C0, 'order': 2, "ntfm": 2,
                    "del": np.repeat(1, 10)}
        }

        # Insample and outsample sets
        tr__y = y[:450]
        te__y = y[450:]

        tr__X = {'tfm': x[:450, :]}
        te__X = {'tfm': x[450:, :]}

        # Fit
        mod = Analysis(model_dict=model_dict, V=sd_y**2)
        fit_results = mod.fit(y=tr__y, X=tr__X)

        # Forecasting
        forecast_results = mod._k_steps_a_head_forecast(k=50, X=te__X)
        forecast_df = forecast_results.get('filter')
        parameters_df = forecast_results.get('parameters')

        mape = np.mean(np.abs(forecast_df.f - te__y) / te__y)

        self.assertTrue(mape < 1)
        self.assertTrue(len(parameters_df) == 500)
        self.assertTrue(forecast_df.notnull().all().all())
        self.assertTrue(parameters_df.notnull().all().all())

    def test__k_steps_ahead_forecast_values(self):
        """Test k steps a head performance."""
        model_dict = {
            'tfm': {'m0': m0, 'C0': C0, 'order': 2, "ntfm": 2,
                    "del": np.repeat(1, 10)}
        }

        # Fit
        mod = Analysis(model_dict=model_dict, V=sd_y**2)
        fit_results = mod.fit(y=y, X=X)

        # Forecasting
        Xte = {'tfm': X.get('tfm')[40, :]}
        f, q = mod._forecast(X=Xte)

        reshape_Xte = {'tfm': X.get('tfm')[40, :].reshape(-1, 2)}
        forecast_df = mod\
            ._k_steps_a_head_forecast(k=1, X=reshape_Xte)\
            .get('filter')

        fk = forecast_df.f.values
        qk = forecast_df.q.values

        self.assertTrue(np.isclose(f, fk))
        self.assertTrue(np.isclose(q, qk))
