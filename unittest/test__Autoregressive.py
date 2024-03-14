"""Test autoregressive model parameters estimation."""
import numpy as np
import unittest
from dynm.analysis import Analysis
from copy import copy

# Simulating the data
nobs = 500
sd_y = 0.02
phi_1 = 0.8
phi_2 = -0.5

y = np.zeros(nobs)
x = np.zeros(nobs)
xi_1 = np.zeros(nobs)
xi_2 = np.zeros(nobs)

# First observation
np.random.seed(1234)
for t in range(1, nobs):
    # Random errors
    nu = np.random.normal(loc=0, scale=sd_y, size=1)

    # Evolution
    xi_1[t] = phi_1 * xi_1[t - 1] + phi_2 * xi_2[t - 1] + nu
    xi_2[t] = xi_1[t-1]

    # Observation
    y[t] = xi_1[t]

# Estimation
m0 = np.array([0, 0, 1, 0])
C0 = np.identity(4)
W = np.identity(4)

np.fill_diagonal(C0, val=[9, .001, 9, 9])
np.fill_diagonal(W, val=[sd_y**2, 0, 0, 0])


class TestAutoregressive(unittest.TestCase):
    """Tests Analysis results for AutoRegressive Model."""

    def test__estimates_known_W_and_V(self):
        """Test parameters estimation with know W and V."""
        model_dict = {
            'arm': {'m0': m0, 'C0': C0, 'order': 2, "W": W}
        }

        # Fit
        mod = Analysis(model_dict=model_dict, V=sd_y**2).fit(y=y)
        m = mod.m

        self.assertTrue(np.abs(m[2] - phi_1) < .1)
        self.assertTrue(np.abs(m[3] - phi_2) < .1)

    def test__estimates_discount(self):
        """Test parameters estimation with discount."""
        model_dict = {
            'arm': {'m0': m0, 'C0': C0, 'order': 2,
                    "del": np.array([1, 1, 1, 1])}
        }

        # Fit
        mod = Analysis(model_dict=model_dict).fit(y=y)
        m = mod.m

        self.assertTrue(np.abs(m[2] - phi_1) < .25)
        self.assertTrue(np.abs(m[3] - phi_2) < .25)

    def test__analysis_with_nan(self):
        """Test parameters estimation with nan in y."""
        model_dict = {
            'arm': {'m0': m0, 'C0': C0, 'order': 2,
                    "del": np.array([1, 1, 1, 1])}
        }

        copy_y = copy(y)
        copy_y[50] = np.nan

        # Fit
        mod = Analysis(model_dict=model_dict).fit(y=copy_y)

        forecast_df = mod.dict_filter.get('predictive')
        m = mod.m

        self.assertTrue(np.abs(m[2] - phi_1) < .2)
        self.assertTrue(np.abs(m[3] - phi_2) < .2)
        self.assertTrue(forecast_df.f.notnull().all())

    def test__k_steps_ahead_forecast_performance(self):
        """Test k steps a head performance."""
        model_dict = {
            'arm': {'m0': m0, 'C0': C0, 'order': 2,
                    "del": np.array([1, 1, 1, 1])}
        }

        # Insample and outsample sets
        tr__y = y[:450]
        te__y = y[450:]

        # Fit
        mod = Analysis(model_dict=model_dict).fit(y=tr__y)

        # Forecasting
        forecast_results = mod._k_steps_a_head_forecast(k=50)
        forecast_df = forecast_results.get('predictive')
        parameters_df = forecast_results.get('parameters')

        mape = np.mean(np.abs(forecast_df.f - te__y) / te__y)

        self.assertTrue(mape < 1)
        self.assertTrue(len(parameters_df) == 200)
        self.assertTrue(forecast_df.notnull().all().all())
        self.assertTrue(parameters_df.notnull().all().all())

    def test__k_steps_ahead_forecast_values(self):
        """Test k steps a head values."""
        model_dict = {
            'arm': {'m0': m0, 'C0': C0, 'order': 2,
                    "del": np.array([1, 1, 1, 1])}
        }

        # Fit
        mod = Analysis(model_dict=model_dict).fit(y=y)

        # Forecasting
        f, q = mod._forecast()

        forecast_df = mod\
            ._k_steps_a_head_forecast(k=1)\
            .get('predictive')

        fk = forecast_df.f.values
        qk = forecast_df.q.values

        self.assertTrue(np.isclose(f, fk))
        self.assertTrue(np.isclose(q, qk))

    def test__smoothed_posterior_variance(self):
        """Test smooth posterior variance."""
        model_dict = {
            'arm': {'m0': m0, 'C0': C0, 'order': 2,
                    "del": np.array([1, 1, 1, 1])}
        }

        # Fit
        mod = Analysis(model_dict=model_dict).fit(y=y, smooth=True)
        smooth_posterior = mod.dict_smooth.get('posterior')

        min_var = smooth_posterior.variance.min()
        self.assertTrue(min_var >= 0.0)

    def test__smoothed_predictive_variance(self):
        """Test smooth predictive variance."""
        model_dict = {
            'arm': {'m0': m0, 'C0': C0, 'order': 2,
                    "del": np.array([1, 1, 1, 1])}
        }

        # Fit
        mod = Analysis(model_dict=model_dict).fit(y=y, smooth=True)
        smooth_predictive = mod.dict_smooth.get('predictive')

        min_var = smooth_predictive.q.min()
        self.assertTrue(min_var >= 0.0)

    def test__smoothed_predictive_errors(self):
        """Test smooth predictive mape."""
        model_dict = {
            'arm': {'m0': m0, 'C0': C0, 'order': 2,
                    "del": np.array([1, 1, 1, 1])}
        }

        # Fit
        mod = Analysis(model_dict=model_dict).fit(y=y, smooth=True)

        filter_predictive = mod\
            .dict_filter.get('predictive')\
            .sort_values('t')

        smooth_predictive = mod\
            .dict_smooth.get('predictive')\
            .sort_values('t')

        f = filter_predictive.f.values
        fk = smooth_predictive.f.values

        mse1 = np.mean((f-y)**2)
        mse2 = np.mean((fk-y)**2)

        self.assertTrue(mse2/mse1 <= 1.0)