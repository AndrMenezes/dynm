"""Test autoregressive model parameters estimation."""
import numpy as np
import unittest
from src.analysis import Analysis

# Simulating the data
nobs = 500
sd_y = 0.02
true_phi_1 = 0.8
true_phi_2 = -0.5

y = np.zeros(nobs)
x = np.zeros(nobs)
xi_1 = np.zeros(nobs)
xi_2 = np.zeros(nobs)

# First observation
np.random.seed(1111)
for t in range(1, nobs):
    # Random errors
    nu = np.random.normal(loc=0, scale=sd_y, size=1)

    # Evolution
    xi_1[t] = true_phi_1 * xi_1[t - 1] + true_phi_2 * xi_2[t - 1] + nu
    xi_2[t] = xi_1[t-1]

    # Observation
    y[t] = xi_1[t]

# Estimation
m0 = np.array([0, 0, 1, 0])
C0 = np.identity(4)
W = np.identity(4)

np.fill_diagonal(C0, val=[9, 9, 9, 9])
np.fill_diagonal(W, val=[sd_y**2, 0, 0, 0])


class TestAnalysisAR(unittest.TestCase):
    """Tests Analysis results for AutoRegressive Model."""

    def test__estimates_known_W(self):
        """Test parameters estimation with know W."""
        model_dict = {
            'arm': {'m0': m0, 'C0': C0, 'order': 2, "W": W}
        }

        # Fit
        mod = Analysis(model_dict=model_dict, V=sd_y**2)
        fit_results = mod.fit(y=y)
        m = mod.m

        self.assertTrue(np.abs(m[2] - true_phi_1) < .1)
        self.assertTrue(np.abs(m[3] - true_phi_2) < .1)

    def test__estimates_discount(self):
        """Test parameters estimation with discount."""
        model_dict = {
            'arm': {'m0': m0, 'C0': C0, 'order': 2,
                    "del": np.array([1, 1, 1, 1])}
        }

        # Fit
        mod = Analysis(model_dict=model_dict, V=sd_y**2)
        fit_results = mod.fit(y=y)
        m = mod.m

        self.assertTrue(np.abs(m[2] - true_phi_1) < .1)
        self.assertTrue(np.abs(m[3] - true_phi_2) < .1)

    def test__k_steps_a_head_forecast(self):
        """Test k steps a head performance."""
        model_dict = {
            'arm': {'m0': m0, 'C0': C0, 'order': 2,
                    "del": np.array([1, 1, 1, 1])}
        }

        # Insample and outsample sets
        tr__y = y[:450]
        te__y = y[450:]

        # Fit
        mod = Analysis(model_dict=model_dict, V=sd_y**2)
        fit_results = mod.fit(y=tr__y)

        # Forecasting
        forecast_results = mod._k_steps_a_head_forecast(k=50)
        forecast_df = forecast_results.get('filter')
        parameters_df = forecast_results.get('parameters')

        mape = np.mean(np.abs(forecast_df.f - te__y) / te__y)

        self.assertTrue(mape < 1)
        self.assertTrue(len(parameters_df) == 200)
        self.assertTrue(forecast_df.notnull().all().all())
        self.assertTrue(parameters_df.notnull().all().all())
