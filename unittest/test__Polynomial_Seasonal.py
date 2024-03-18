"""Test dlm model parameters estimation."""
import numpy as np
import unittest
from dynm.dynamic_model import BayesianDynamicModel
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


class TestDLM(unittest.TestCase):
    """Tests BayesianDynamicModel results for Dynamic Linear Model."""

    def test__estimates_known_W_and_V(self):
        """Test parameters estimation with know W and V."""
        model_dict = {
            "polynomial": {
                "m0": np.array([10]),
                "C0": np.array([[9]]),
                "ntrend": 1,
                "W": np.array([[0]]),
            },
            "seasonal": {
                "m0": np.array([0, 0, 0, 0]),
                "C0": np.identity(4) * 9,
                "seas_period": 12,
                "seas_harm_components": [1, 2],
                "W": np.identity(4) * 0
            }
        }

        # Fit
        mod = BayesianDynamicModel(model_dict=model_dict, V=sd_y**2).fit(y=y)
        forecast_df = mod.dict_filter.get("predictive")

        mape = np.mean(np.abs(forecast_df.f - forecast_df.y) / forecast_df.y)

        self.assertTrue(mape < .05)

    def test__estimates_discount(self):
        """Test parameters estimation with discount."""
        model_dict = {
            "polynomial": {
                "m0": np.array([10]),
                "C0": np.array([[9]]),
                "ntrend": 1,
                "discount": 1
            },
            "seasonal": {
                "m0": np.array([0, 0, 0, 0]),
                "C0": np.identity(4),
                "seas_period": 12,
                "seas_harm_components": [1, 2],
                "discount": 1
            }
        }

        # Fit
        mod = BayesianDynamicModel(model_dict=model_dict, V=sd_y**2).fit(y=y)
        forecast_df = mod.dict_filter.get("predictive")

        mape = np.mean(np.abs(forecast_df.f - forecast_df.y) / forecast_df.y)

        self.assertTrue(mape < .05)

    def test__predict_calc_predictive_mean_and_var_performance(self):
        """Test k steps a head performance."""
        model_dict = {
            "polynomial": {
                "m0": np.array([10]),
                "C0": np.array([[9]]),
                "ntrend": 1,
                "discount": 1
            },
            "seasonal": {
                "m0": np.array([0, 0, 0, 0]),
                "C0": np.identity(4),
                "seas_period": 12,
                "seas_harm_components": [1, 2],
                "discount": 1
            }
        }

        # Insample and outsample sets
        tr__y = y[:60]
        te__y = y[60:]

        # Fit
        mod = BayesianDynamicModel(model_dict=model_dict).fit(y=tr__y)

        # Forecasting
        forecast_results = mod._predict(k=20)
        forecast_df = forecast_results.get("predictive")
        mape = np.mean(np.abs(forecast_df.f - te__y) / te__y)

        self.assertTrue(mape < .05)

    def test__predict_values(self):
        """Test k steps a head values."""
        model_dict = {
            "polynomial": {
                "m0": np.array([10]),
                "C0": np.array([[9]]),
                "ntrend": 1,
                "discount": 1
            },
            "seasonal": {
                "m0": np.array([0, 0, 0, 0]),
                "C0": np.identity(4),
                "seas_period": 12,
                "seas_harm_components": [1, 2],
                "discount": 1
            }
        }

        # Fit
        mod = BayesianDynamicModel(model_dict=model_dict).fit(y=y)

        # Forecasting
        f, q = mod._calc_predictive_mean_and_var()

        forecast_df = mod\
            ._predict(k=1)\
            .get("predictive")

        fk = forecast_df.f.values
        qk = forecast_df.q.values

        self.assertTrue(np.isclose(f, fk))
        self.assertTrue(np.isclose(q, qk))

    def test__smoothed_posterior_variance(self):
        """Test smooth posterior variance."""
        model_dict = {
            "polynomial": {
                "m0": np.array([10]),
                "C0": np.array([[9]]),
                "ntrend": 1,
                "discount": 1
            },
            "seasonal": {
                "m0": np.array([0, 0, 0, 0]),
                "C0": np.identity(4),
                "seas_period": 12,
                "seas_harm_components": [1, 2],
                "discount": 1
            }
        }

        # Fit
        mod = BayesianDynamicModel(model_dict=model_dict).fit(y=y, smooth=True)
        smooth_posterior = mod.dict_smooth.get("posterior")

        min_var = smooth_posterior.variance.min()
        self.assertTrue(min_var >= 0.0)

    def test__smoothed_predictive_variance(self):
        """Test smooth predictive variance."""
        model_dict = {
            "polynomial": {
                "m0": np.array([10]),
                "C0": np.array([[9]]),
                "ntrend": 1,
                "discount": 1
            },
            "seasonal": {
                "m0": np.array([0, 0, 0, 0]),
                "C0": np.identity(4),
                "seas_period": 12,
                "seas_harm_components": [1, 2],
                "discount": 1
            }
        }

        # Fit
        mod = BayesianDynamicModel(model_dict=model_dict).fit(y=y, smooth=True)
        smooth_predictive = mod.dict_smooth.get("predictive")

        min_var = smooth_predictive.q.min()
        self.assertTrue(min_var >= 0.0)

    def test__smoothed_predictive_errors(self):
        """Test smooth predictive mape."""
        model_dict = {
            "polynomial": {
                "m0": np.array([10]),
                "C0": np.array([[9]]),
                "ntrend": 1,
                "discount": 1
            },
            "seasonal": {
                "m0": np.array([0, 0, 0, 0]),
                "C0": np.identity(4),
                "seas_period": 12,
                "seas_harm_components": [1, 2],
                "discount": 1
            }
        }

        # Fit
        mod = BayesianDynamicModel(model_dict=model_dict).fit(y=y, smooth=True)

        filter_predictive = mod\
            .dict_filter.get("predictive")\
            .sort_values("t")

        smooth_predictive = mod\
            .dict_smooth.get("predictive")\
            .sort_values("t")

        f = filter_predictive.f.values
        fk = smooth_predictive.f.values

        mse1 = np.mean((f-y)**2)
        mse2 = np.mean((fk-y)**2)

        self.assertTrue(mse2/mse1 <= 1.0)
