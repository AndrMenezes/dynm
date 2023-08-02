"""Test dlm model parameters estimation."""
import numpy as np
import matplotlib.pyplot as plt
from src.analysis import Analysis
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

model_dict = {
    'dlm': {'m0': m0, 'C0': C0, 'ntrend': 1, 'nregn': 0,
            "seas_period": 12, "seas_harm_components": [1, 2],
            "del": np.repeat(1, 5)}
}

# Fit
mod = Analysis(model_dict=model_dict, V=sd_y**2)
fit_results = mod.fit(y=y)
forecast_df = fit_results.get('filter').query("t > 10")

# Plot results
f = forecast_df["f"]
y = forecast_df["y"]
ci_lower = forecast_df["ci_lower"]
ci_upper = forecast_df["ci_upper"]
plt.plot(y.values, color="black", marker="o")
plt.plot(f.values, color="red", marker="o")
plt.plot(ci_lower.values, color="red")
plt.plot(ci_upper.values, color="red")
plt.show()
