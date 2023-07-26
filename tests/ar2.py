import numpy as np
from src.analysis import Analysis


# Simulating the data
###############################################################################
# Simulating the data
nobs = 500
sd_y = 0.02
true_phi_1 = 0.8
true_phi_2 = -0.5

y = np.zeros(nobs)
x = np.zeros(nobs)
xi_1 = np.zeros(nobs)
xi_2 = np.zeros(nobs)

# Initial values
xi_1[0] = 0

# Random errors
nu = np.random.normal(loc=0, scale=sd_y, size=nobs)

# First observation
y[0] = xi_1[0] + nu[0]

np.random.seed(1111)
for t in range(1, nobs):
    # Evolution
    xi_1[t] = true_phi_1 * xi_1[t - 1] + true_phi_2 * xi_2[t - 1] + nu[t]
    xi_2[t] = xi_1[t-1]

    # Observation
    y[t] = xi_1[t]

# Estimation
m0 = np.array([0, 0, 1, 0])
C0 = np.identity(4)
np.fill_diagonal(C0, val=[9, 9, 9, 9])
W = np.identity(4)
np.fill_diagonal(W, val=[sd_y**2, 0, 0, 0])

model_dict = {
    'arm': {'m0': m0, 'C0': C0, 'order': 2, "W": W}
}

# Fit
mod = Analysis(model_dict=model_dict, V=sd_y**2)
fit_results = mod.fit(y=y)

filter_df = fit_results.get('filter')
posterior_df = fit_results.get('posterior')
