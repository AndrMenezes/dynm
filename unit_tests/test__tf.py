"""Test autoregressive model parameters estimation."""
import numpy as np
import unittest
from src.analysis import Analysis

# Simulating the data
nobs = 500
sd_y = 0.02
true_gamma = 2.5
true_lambda_1 = 0.8
true_lambda_2 = -0.5

y = np.zeros(nobs)
x = np.zeros(nobs)
E_1 = np.zeros(nobs)
E_2 = np.zeros(nobs)

# First observation
np.random.seed(1111)
for t in range(1, nobs):
    # Random errors
    nu = np.random.normal(loc=0, scale=sd_y, size=1)
    x[t] = np.random.normal(loc=0, scale=1, size=1)

    # Evolution
    E_1[t] = true_lambda_1 * E_1[t - 1] + \
        true_lambda_2 * E_2[t - 1] + \
        x[t] * true_gamma
    E_2[t] = E_1[t-1]

    # Observation
    y[t] = E_1[t] + nu

# Estimation
m0 = np.array([0, 0, 1, 0, 0])
C0 = np.identity(5)
W = np.identity(5)

np.fill_diagonal(C0, val=[9, 9, 9, 9, 9])
np.fill_diagonal(W, val=[0, 0, 0, 0])

X = {'tfm': x.reshape(-1, 1)}


class TestAnalysisTF(unittest.TestCase):
    """Tests Analysis results for Transfer Function Model."""

    def test__estimates_known_W(self):
        """Test parameters estimation with know W."""
        model_dict = {
            'tfm': {'m0': m0, 'C0': C0, 'order': 2, "W": W, "ntfm": 1}
        }

        # Fit
        mod = Analysis(model_dict=model_dict, V=sd_y**2)
        fit_results = mod.fit(y=y, X=X)
        m = mod.m

        self.assertTrue(np.abs(m[2] - true_lambda_1) < .1)
        self.assertTrue(np.abs(m[3] - true_lambda_2) < .1)

    def test__estimates_discount(self):
        """Test parameters estimation with discount."""
        model_dict = {
            'tfm': {'m0': m0, 'C0': C0, 'order': 2, "ntfm": 1,
                    "del": np.array([1, 1, 1, 1, 1])}
        }

        # Fit
        mod = Analysis(model_dict=model_dict, V=sd_y**2)
        fit_results = mod.fit(y=y, X=X)
        m = mod.m

        self.assertTrue(np.abs(m[2] - true_lambda_1) < .1)
        self.assertTrue(np.abs(m[3] - true_lambda_2) < .1)
