"""Utils functions."""
import numpy as np
import pandas as pd
from scipy import stats
from typing import List
from scipy import stats


def tidy_parameters(dict_parameters: dict, entry_m: str, entry_v: str,
                    names_parameters: List,
                    index_seas_parameters: List = None,
                    F: np.ndarray = None):
    """Transform the state space moments from dictionary of to a \
    pd.DataFrame in long format.

    Parameters
    ----------
    dict_parameters : dict
        A dictionary with the posterior (m and C) and prior (a and R) moments
        for the state space parameters along time.
    entry_m : str
        An entry name in the `dict_parameters` of mean vector of
        state space parameters.
    entry_v : str
        An entry name in the `dict_parameters` of covariance matrix of
        state space parameters.
    names_parameters : List
        A List with the names of each state parameter components.
    index_seas_parameters : List
        A List indicating the corresponding index for the seasonalities model
        components.
    F : np.ndarray
        An array with the of known constants representing the model components.

    Returns
    -------
    pd.DataFrame
        A DataFrame in tidy format with columns parameter, mean, and
        variance for each time.
    """

    def _get_mean(x: np.ndarray):
        """Get the mean vector from np.nadarray and transform in pd.DataFrame.

        Parameters
        ----------
        x : np.ndarray
            The mean vector of state space parameters. It could be the prior or
            posterior moments.

        Returns
        -------
        pd.DataFrame
            The extracted mean values.
        """
        df_out = pd.DataFrame(
            data={"mean": x[:, 0]}, index=names_parameters)
        if index_seas_parameters:
            j = 1
            lt = []
            for iseas in index_seas_parameters:
                m_seas = x[iseas][:, 0]
                F_seas = F[iseas]
                sum_seas = F_seas.T @ m_seas
                lt.append(pd.DataFrame(
                    data={"mean": sum_seas}, index=["Sum Seas " + str(j)]))
                j = j + 1
            df_out = pd.concat([df_out, pd.concat(lt)])
        return df_out

    def _get_var(x: np.ndarray):
        """Get the diagonal elements of covariance matrix from np.nadarray\
        and transform in pd.DataFrame.

        Parameters
        ----------
        x : np.ndarray
            The covariance matrix of state space parameters.
            It could be the prior or posterior moments.

        Returns
        -------
        pd.DataFrame
            The extracted variance values.
        """
        df_out = pd.DataFrame(
            data={"variance": np.diag(x)}, index=names_parameters)
        if index_seas_parameters:
            j = 1
            lt = []
            for iseas in index_seas_parameters:
                cov_seas = x[np.ix_(iseas, iseas)]
                F_seas = F[iseas]
                sum_seas = F_seas.T @ cov_seas @ F_seas
                lt.append(pd.DataFrame(
                    data={"variance": sum_seas[:, 0]},
                    index=["Sum Seas " + str(j)]))
                j = j + 1
            df_out = pd.concat([df_out, pd.concat(lt)])

        return df_out

    df_mean_parms = pd.concat(
        list(map(_get_mean, dict_parameters[entry_m])))
    df_var_parms = pd.concat(
        list(map(_get_var, dict_parameters[entry_v])))
    df_state_parameters = pd.concat(
        [df_mean_parms.reset_index(),
         df_var_parms.reset_index(drop=True)], axis=1)
    df_state_parameters.rename(columns={"index": "parameter"}, inplace=True)

    return df_state_parameters[["parameter", "mean", "variance"]]


def set_X_dict(nobs: int, X: dict = {}):
    copy_X = X.copy()

    # Organize transfer function values
    if X.get('dlm') is None:
        x = np.array([None]*(nobs+1)).reshape(-1, 1)
        copy_X['dlm'] = x

    if X.get('tfm') is None:
        z = np.array([None]*(nobs+1)).reshape(-1, 1)
        copy_X['tfm'] = z

    return copy_X


def add_credible_interval_studentt(
        pd_df: pd.DataFrame,
        entry_m: str,
        entry_v: str,
        level=float):
    df = pd_df["t"].values + 1
    mu = pd_df[entry_m].values
    sigma = np.sqrt(pd_df[entry_v].values + 10e-300)

    # Calculate intervals
    pd_df["ci_lower"] = stats.t.ppf(q=level/2, df=df, loc=mu, scale=sigma)
    pd_df["ci_upper"] = stats.t.ppf(q=1-level/2, df=df, loc=mu, scale=sigma)

    return pd_df


def add_credible_interval_gamma(
        pd_df: pd.DataFrame,
        entry_a: str,
        entry_b: str,
        level=float):
    a = 1 / pd_df[entry_a].values
    b = pd_df[entry_b].values + 10e-300

    # Calculate intervals
    pd_df["ci_lower"] = stats.gamma.ppf(q=level/2, a=a, scale=b)
    pd_df["ci_upper"] = stats.gamma.ppf(q=1-level/2, a=a, scale=b)

    return pd_df


def create_mod_label_column(mod, t: int):
    dlm_lb = list(np.repeat('dlm', len(mod.dlm.m)))
    arm_lb = list(np.repeat('arm', len(mod.arm.m)))
    tfm_lb = list(np.repeat(
        ['tfm_' + str(i + 1) for i in range(mod.tfm.ntfm)],
        2 * mod.tfm.order + 1))

    mod_lb = t * (dlm_lb + arm_lb + tfm_lb)

    return mod_lb
