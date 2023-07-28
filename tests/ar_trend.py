import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.analysis import Analysis

###############################################################################
# Simulating the data
nobs = 100
sd_y = 0.02
sd_mu = 2e-6
sd_beta = 2e-4
true_phi_1 = 0.8
true_phi_2 = -0.5

y = np.zeros(nobs)
x = np.zeros(nobs)
xi_1 = np.zeros(nobs)
xi_2 = np.zeros(nobs)
mu = np.zeros(nobs)
beta = np.zeros(nobs)

# Initial values
xi_1[0] = 0
mu[0] = 5
beta[0] = 0.0001

# Random errors
nu = np.random.normal(loc=0, scale=sd_y, size=nobs)
omega_mu = np.random.normal(loc=0, scale=sd_mu, size=nobs)
omega_beta = np.random.normal(loc=0, scale=sd_beta, size=nobs)

# First observation
y[0] = mu[0] + xi_1[0] + nu[0]

np.random.seed(1212)
for t in range(1, nobs):
    # Evolution
    xi_1[t] = true_phi_1 * xi_1[t - 1] + true_phi_2 * xi_2[t - 1] + nu[t]
    xi_2[t] = xi_1[t-1]
    beta[t] = beta[t - 1] + omega_beta[t]
    mu[t] = mu[t - 1] + beta[t] + omega_mu[t]

    # Observation
    y[t] = mu[t] + xi_1[t]

data_ar_trend = pd.DataFrame(data={"t": t, "y": y, "mu": mu, "beta": beta,
                                   "xi_1": xi_1, "xi_2": xi_2})

data_ar_trend.to_csv("ar_trend.csv", index=False)


# Estimation
m0_trend = np.array([0, 0])
C0_trend = np.identity(2)
np.fill_diagonal(C0_trend, val=10)
W_trend = np.identity(2)
np.fill_diagonal(W_trend, val=[sd_mu**2, sd_beta**2])

m0_ar = np.array([0, 0, 1, 0])
C0_ar = np.identity(4)
np.fill_diagonal(C0_ar, val=10)
C0_ar[1][1] = 0
W_ar = np.identity(4)
np.fill_diagonal(W_ar, val=[sd_y**2, 0, 0, 0])


model_dict = {
    'dlm': {'m0': m0_trend, 'C0': C0_trend, 'ntrend': 2, 'nregn': 0,
            'W': W_trend},
    'arm': {'m0': m0_ar, 'C0': C0_ar, 'order': 2, 'W': W_ar}
}


# Fit
mod = Analysis(model_dict=model_dict, V=sd_y**2)
fit_results = mod.fit(y=y)

filter_df = fit_results.get('filter')
posterior_df = fit_results.get('posterior')
posterior_df.tail(6)


mu_est = posterior_df[posterior_df["parameter"] == "intercept_1"]["mean"]
plt.plot(mu, color="black", marker="o")
plt.plot(mu_est.values, color="red", marker="o")
plt.show()


beta_est = posterior_df[posterior_df["parameter"] == "intercept_2"]["mean"]
plt.plot(beta, color="black", marker="o")
plt.plot(beta_est.values, color="red", marker="o")
plt.show()

phi_1_est = posterior_df[posterior_df["parameter"] == "phi_1"]
plt.plot(true_phi_1, color="black", marker="o")
plt.plot(phi_1_est["mean"].values, color="red", marker="o")
plt.show()
