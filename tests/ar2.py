import pandas as pd
import numpy as np
from src.analysis import Analysis
from src.algebra import _build_Gnonlinear, _build_W

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

np.random.seed(1212)
for t in range(1, nobs):
    # Evolution
    xi_1[t] = true_phi_1 * xi_1[t - 1] + true_phi_2 * xi_2[t - 1] + nu[t]
    xi_2[t] = xi_1[t-1]

    # Observation
    y[t] = xi_1[t]


data = pd.read_csv("data_ar2.csv")
y = data["y"].values

# Estimation
m0 = np.array([0, 0, 1, 0])
C0 = np.identity(4)
np.fill_diagonal(C0, val=[9, 9, 9, 9])
W = np.identity(4)
np.fill_diagonal(W, val=[0.2**2, 0, 0, 0])

model_dict = {
    'arm': {'m0': m0, 'C0': C0, 'order': 2,
            'W': W}
}


# Fit
mod = Analysis(model_dict=model_dict, V=0.2**2)
fit_results = mod.fit(y=data["y"])

mod.dict_state_params["R"]

filter_df = fit_results.get('filter')
posterior_df = fit_results.get('posterior')

###############################################################################
# Understanding the implementation
v = 0.04
F = np.array([1, 0, 0, 0]).reshape(-1, 1)
W = np.identity(4)
np.fill_diagonal(W, val=[v, 0, 0, 0])


def g(theta):
    return np.array(
        [theta[2] * theta[0] + theta[3] * theta[1],
         theta[0],
         theta[2],
         theta[3]]).reshape(-1, 1)


# Prior
m = np.array([0, 0, 1, 0]).reshape(-1, 1)
C = np.identity(4)
np.fill_diagonal(C, val=9)

dict_1step_forecast = {'t': [], 'y': [], 'f': [], 'q': [], 's': []}
dict_state_params = {'m': [], 'C': [], 'a': [], 'R': []}
for t in range(len(y)):

    # Prior for time t
    G_m = _build_Gnonlinear(m=m, order=2)
    a = g(theta=m)
    # print(a, a1)
    R = G_m @ C @ G_m.T + W

    # Predictive
    f = F.T @ a
    q = F.T @ R @ F + v

    # Error
    e = y[t] - f
    A = (R @ F) / q

    # Kalman update
    m = a + A * e
    C = R - q * A @ A.T

    dict_1step_forecast['t'].append(t)
    dict_1step_forecast['y'].append(y[t])
    dict_1step_forecast['f'].append(np.ravel(f)[0])
    dict_1step_forecast['q'].append(np.ravel(q)[0])
    dict_state_params['m'].append(m)
    dict_state_params['C'].append(C)
    dict_state_params['a'].append(a)
    dict_state_params['R'].append(R)
