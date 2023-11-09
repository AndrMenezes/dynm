"""Test dlm model parameters estimation."""
import numpy as np
import unittest
from dynm.analysis import Analysis
from scipy.linalg import block_diag

# Simulating the data
nobs = 80
sd_y = 0.02

y = np.zeros(nobs)
theta = np.zeros([nobs, 5])

# Initial information
y[0] = 10
theta[0, :] = np.array([10, 1, .25, 1, .25])

# First observation
np.random.seed(1111)
for t in range(1, nobs):
    # Random errors
    nu = np.random.normal(loc=0, scale=sd_y, size=1)

    # Regression vector
    F = np.array([1, 1, 0, 1, 0]).T

    # Evolution
    Gtrend = np.array([[1]])
    Gseas = np.array([[0.8660254,  0.5,  0.,  0.],
                      [-0.5,  0.8660254,  0.,  0.],
                      [0.,  0.,  0.5,  0.8660254],
                      [0.,  0., -0.8660254,  0.5]])
    G = block_diag(Gtrend, Gseas)

    theta[t] = G @ theta[t-1]

    # Observation
    y[t] = F.T @ theta[t] + nu

# Estimation
m0 = np.array([10, 0, 0, 0, 0])
C0 = np.identity(5)
W = np.identity(5) * 0

np.fill_diagonal(C0, val=[9, 9, 9, 9, 9])


class TestAnalysisAR(unittest.TestCase):
    """Tests Analysis results for AutoRegressive Model."""

    def test__estimates_known_W(self):
        """Test parameters estimation with know W."""
        model_dict = {
            'dlm': {
                'm0': m0,
                'C0': C0,
                'ntrend': 1,
                'nregn': 0,
                "seas_period": 12,
                "seas_harm_components": [1, 2],
                "W": W}
        }

        # Fit
        mod = Analysis(model_dict=model_dict, V=sd_y**2)
        fit_results = mod.fit(y=y)
        forecast_df = fit_results.get('filter')

        mape = np.mean(np.abs(forecast_df.f - forecast_df.y) / forecast_df.y)

        self.assertTrue(mape < .05)

    def test__estimates_discount(self):
        """Test parameters estimation with discount."""
        model_dict = {
            'dlm': {
                'm0': m0,
                'C0': C0,
                'ntrend': 1,
                'nregn': 0,
                "seas_period": 12,
                "seas_harm_components": [1, 2],
                "del": np.repeat(1, 5)}
        }

        # Fit
        mod = Analysis(model_dict=model_dict, V=sd_y**2)
        fit_results = mod.fit(y=y)
        forecast_df = fit_results.get('filter')

        mape = np.mean(np.abs(forecast_df.f - forecast_df.y) / forecast_df.y)

        self.assertTrue(mape < .05)

    def test__k_steps_ahead_forecast_performance(self):
        """Test k steps a head performance."""
        model_dict = {
            'dlm': {
                'm0': m0,
                'C0': C0,
                'ntrend': 1,
                'nregn': 0,
                "seas_period": 12,
                "seas_harm_components": [1, 2],
                "del": np.repeat(1, 5)}
        }

        # Insample and outsample sets
        tr__y = y[:60]
        te__y = y[60:]

        # Fit
        mod = Analysis(model_dict=model_dict, V=sd_y**2)
        fit_results = mod.fit(y=tr__y)

        # Forecasting
        forecast_results = mod._k_steps_ahead_forecast(k=20)
        forecast_df = forecast_results.get('filter')
        parameters_df = forecast_results.get('parameters')

        mape = np.mean(np.abs(forecast_df.f - te__y) / te__y)

        self.assertTrue(mape < .05)

    def test__k_steps_a_head_forecast_values(self):
        """Test k steps a head values."""
        model_dict = {
            'dlm': {
                'm0': m0,
                'C0': C0,
                'ntrend': 1,
                'nregn': 0,
                "seas_period": 12,
                "seas_harm_components": [1, 2],
                "del": np.repeat(1, 5)}
        }

        # Fit
        mod = Analysis(model_dict=model_dict, V=sd_y**2)
        fit_results = mod.fit(y=y)

        # Forecasting
        f, q = mod._forecast()

        forecast_df = mod\
            ._k_steps_a_head_forecast(k=1)\
            .get('filter')

        fk = forecast_df.f.values
        qk = forecast_df.q.values

        self.assertTrue(np.isclose(f, fk))
        self.assertTrue(np.isclose(q, qk))
