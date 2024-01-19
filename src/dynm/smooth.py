"""Smoothing for Dynamic Linear Models."""
import numpy as np
import pandas as pd
from dynm.utils import tidy_parameters, set_X_dict
from dynm.utils import create_mod_label_column, add_credible_interval_studentt


def _backward_smoother(mod, X: dict = {}, level: float = 0.05):
    """Perform backward smoother.

    That is, obtain the smoothing moments of the one-step ahead predictive
    distribution and state space posterior distribution.

    Parameters
    ----------
    dict_state_parms : dict
        dictionary with the posterior (m and C) and prior (a and R) moments
        for the state space parameters along time.

    Returns
    -------
    List: It contains the following components:
        - `df_predictive_smooth`: pd.DataFrame with the smoothing moments
        of predictive distribution.

        - `df_posterior_smooth`: pd.DataFrame with the smoothing moments
        of posterior state space distribution.
    """
    nobs = len(mod.dict_state_params.get('a'))
    copy_X = set_X_dict(nobs=nobs, X=X)

    # Initialize the model components and posterior/prior parameters
    a = mod.dict_state_params.get('a')
    R = mod.dict_state_params.get('R')
    m = mod.dict_state_params.get('m')
    C = mod.dict_state_params.get('C')

    # Get state evolution matrix
    G = mod.dict_state_evolution.get('G')

    # Dictionaty to save predictive and posterior parameters
    Xk = {'dlm': [], 'tfm': []}
    Xk['dlm'] = copy_X['dlm'][nobs-1, :]
    Xk['tfm'] = copy_X['tfm'][nobs-1, :]

    FT = mod._build_F(X=Xk)

    ak = m[nobs-1]
    Rk = C[nobs-1]
    fk = FT.T @ ak
    qk = (FT.T @ Rk @ FT).round(10)
    dict_smooth_parms = {
        "t": [nobs],
        "ak": [ak],
        "Rk": [Rk],
        "fk": [fk.item()],
        "qk": [qk.item()]}

    # Perform smoothing
    for k in range(1, nobs):
        Xk['dlm'] = copy_X['dlm'][nobs-k, :]
        Xk['tfm'] = copy_X['tfm'][nobs-k, :]

        Fk = mod._build_F(X=Xk)
        Gk = G[nobs-k]

        # B_{t-k}
        B_t_k = C[nobs-k-1] @ Gk.T @ np.linalg.pinv(
            R[nobs-k], rcond=1e-10, hermitian=True)

        # a_t(-k) and R_t(-k)
        ak = m[nobs-k-1] + B_t_k @ (ak - a[nobs-k])
        Rk = C[nobs-k-1] + B_t_k @ (Rk - R[nobs-k]) @ B_t_k.T

        # f_t(-k) and q_t(-k)
        fk = Fk.T @ ak
        qk = (Fk.T @ Rk @ Fk).round(10)

        # Saving parameters
        dict_smooth_parms["ak"].append(ak)
        dict_smooth_parms["Rk"].append(Rk)
        dict_smooth_parms["fk"].append(fk.item())
        dict_smooth_parms["qk"].append(qk.item())
        dict_smooth_parms["t"].append(nobs-k)

    mod.dict_smooth_parms = dict_smooth_parms

    # Organize the predictive smooth parameters
    dict_filter = {key: dict_smooth_parms[key] for key in (
        dict_smooth_parms.keys() & {"t", "fk", "qk", "df"})}
    df_predictive = pd.DataFrame(dict_filter)

    # Organize the posterior parameters
    df_posterior = tidy_parameters(
        dict_parameters=dict_smooth_parms,
        entry_m="ak", entry_v="Rk",
        names_parameters=mod.names_parameters)

    # Create model labels
    df_posterior["mod"] = create_mod_label_column(mod=mod, t=mod.t)

    # Add time column on posterior_df
    t_index = mod.t - np.arange(0, mod.t)
    df_posterior["t"] = np.repeat(t_index, mod.p)
    df_posterior["t"] = df_posterior["t"].astype(int)

    # Round variance
    df_posterior["variance"] = df_posterior["variance"].round(10)
    df_predictive["qk"] = df_predictive["qk"].round(10)

    # Compute credible intervals
    df_posterior = add_credible_interval_studentt(
        pd_df=df_posterior, entry_m="mean",
        entry_v="variance", level=.05)

    df_predictive = add_credible_interval_studentt(
        pd_df=df_predictive, entry_m="fk",
        entry_v="qk", level=.05)

    # Creat dict of results
    dict_results = {'predictive': df_predictive, 'posterior': df_posterior}

    return dict_results
