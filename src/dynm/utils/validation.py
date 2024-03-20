"""Methods for validation."""


def validate_polynomial_model_dict_keys(model_dict: dict):
    """
    Validate keys in a dictionary representing a polynomial model.

    Args:
        model_dict (dict): Dictionary containing model keys.

    Raises:
        ValueError: If required keys are missing.
    """
    user_keys = set(list(model_dict.keys()))
    expected_keys = set(['m0', 'C0', 'ntrend', 'discount', 'W'])
    missing_keys = expected_keys - user_keys

    if (missing_keys == {'discount'}) | (missing_keys == {'W'}):
        pass
    else:
        error_message = ("Missing elements in polynomial model:" +
                         str(missing_keys))
        raise ValueError(error_message)


def validate_regression_model_dict_keys(model_dict: dict):
    """
    Validate keys in a dictionary representing a regression model.

    Args:
        model_dict (dict): Dictionary containing model keys.

    Raises:
        ValueError: If required keys are missing.
    """
    user_keys = set(list(model_dict.keys()))
    expected_keys = set(['m0', 'C0', 'nregn', 'discount', 'W'])
    missing_keys = expected_keys - user_keys

    if (missing_keys == {'discount'}) | (missing_keys == {'W'}):
        pass
    else:
        error_message = ("Missing elements in regression model:" +
                         str(missing_keys))
        raise ValueError(error_message)


def validate_seasonal_model_dict_keys(model_dict: dict):
    """
    Validate keys in a dictionary representing a seasonal model.

    Args:
        model_dict (dict): Dictionary containing model keys.

    Raises:
        ValueError: If required keys are missing.
    """
    user_keys = set(list(model_dict.keys()))
    expected_keys = set(['m0', 'C0', 'seas_period',
                         'seas_harm_components', 'discount', 'W'])
    missing_keys = expected_keys - user_keys

    if (missing_keys == {'discount'}) | (missing_keys == {'W'}):
        pass
    else:
        error_message = ("Missing elements in seasonal model:" +
                         str(missing_keys))
        raise ValueError(error_message)


def validate_transfer_function_model_dict_keys(model_dict: dict):
    """
    Validate keys in a dictionary representing a transfer function model.

    Args:
        model_dict (dict): Dictionary containing model keys.

    Raises:
        ValueError: If required keys are missing.
    """
    user_keys = set(list(model_dict.keys()))
    expected_keys = set(['m0', 'C0', 'ntfm',
                         'lambda_order', 'gamma_order',
                         'discount', 'W'])
    missing_keys = expected_keys - user_keys

    if (missing_keys == {'discount'}) | (missing_keys == {'W'}):
        pass
    else:
        error_message = ("Missing elements in transfer function model:" +
                         str(missing_keys))
        raise ValueError(error_message)


def validate_autoregressive_model_dict_keys(model_dict: dict):
    """
    Validate keys in a dictionary representing an autoregressive model.

    Args:
        model_dict (dict): Dictionary containing model keys.

    Raises:
        ValueError: If required keys are missing.
    """
    user_keys = set(list(model_dict.keys()))
    expected_keys = set(['m0', 'C0', 'order', 'discount', 'W'])
    missing_keys = expected_keys - user_keys

    if (missing_keys == {'discount'}) | (missing_keys == {'W'}):
        pass
    else:
        error_message = ("Missing elements in autoregressive model:" +
                         str(missing_keys))
        raise ValueError(error_message)


def validate_model_dict_polynomial_mean_array(model_dict: dict):
    """
    Validate prior mean array shape for polynomial model.

    Args:
        model_dict (dict): Dictionary containing model keys.

            Obrigatory keys: {'m0', 'C0', 'ntrend''}.

            Optional keys (choose one): {'W', 'discount'}.

    Raises:
        ValueError: If prior mean array and ntrend are incompatible.
    """
    ntrend = model_dict.get('ntrend')
    m0 = model_dict.get('m0')

    if m0.shape[0] == ntrend:
        pass
    else:
        error_message = ("Prior mean array" +
                         "and declared ntrend are incomplatible")
        raise ValueError(error_message)


def validate_model_dict_regression_mean_array(model_dict: dict):
    """
    Validate prior mean array shape for regression model.

    Args:
        model_dict (dict): Dictionary containing model keys.

            Obrigatory keys: {'m0', 'C0', 'nregn'}.

            Optional keys (choose one): {'W', 'discount'}.

    Raises:
        ValueError: If prior mean array and nregn are incompatible.
    """
    nregn = model_dict.get('nregn')
    m0 = model_dict.get('m0')

    if m0.shape[0] == nregn:
        pass
    else:
        error_message = ("Prior mean array" +
                         "and declared nregn are incomplatible")
        raise ValueError(error_message)


def validate_model_dict_seasonal_mean_array(model_dict: dict):
    """
    Validate prior mean array shape for seasonal model.

    Args:
        model_dict (dict): Dictionary containing model keys.

            Obrigatory keys: {'m0', 'C0', 'seas_period',
                              'seas_harm_components'}.

            Optional keys (choose one): {'W', 'discount'}.

    Raises:
        ValueError: If prior mean array and seas_harm_components are
            incompatible.
    """
    seas_harm_components = model_dict.get('seas_harm_components')
    nseas = 2 * len(seas_harm_components)
    m0 = model_dict.get('m0')

    if m0.shape[0] == nseas:
        pass
    else:
        error_message = ("Prior mean array" +
                         "and declared seas_harm_components" +
                         "are incomplatible")
        raise ValueError(error_message)


def validate_model_dict_transfer_function_mean_array(model_dict: dict):
    """
    Validate prior mean array shape for transfer function model.

    Args:
        model_dict (dict): Dictionary containing model keys.

            Obrigatory keys: {'m0', 'C0', 'lambda_order',
                              'gamma_order', 'ntfm'}.

            Optional keys (choose one): {'W', 'discount'}.

    Raises:
        ValueError: If prior mean array and model parameters are incompatible.
    """
    gamma_order = model_dict.get('gamma_order')
    lambda_order = model_dict.get('lambda_order')
    ntfm = model_dict.get('ntfm')
    nparams = (gamma_order + 2 * lambda_order) * ntfm
    m0 = model_dict.get('m0')

    if m0.shape[0] == nparams:
        pass
    else:
        error_message = ("Prior mean array" +
                         "and declared ntfm, gamma_order, lambda_order" +
                         "are incomplatible")
        raise ValueError(error_message)


def validate_model_dict_autoregressive_mean_array(model_dict: dict):
    """
    Validate prior mean array shape for autoregressive model.

    Args:
        model_dict (dict): Dictionary containing model keys.

            Obrigatory keys: {'m0', 'C0', 'order'}.

            Optional keys (choose one): {'W', 'discount'}.

    Raises:
        ValueError: If prior mean array and model parameters are incompatible.
    """
    order = model_dict.get('order')
    nparams = 2 * order
    m0 = model_dict.get('m0')

    if m0.shape[0] == nparams:
        pass
    else:
        error_message = ("Prior mean array and declared order" +
                         "are incomplatible")
        raise ValueError(error_message)


def validate_model_dict_polynomial_covariance_matrix(model_dict: dict):
    """
    Validate prior covariance matrix shape for polynomial model.

    Args:
        model_dict (dict): Dictionary containing model keys.

            Obrigatory keys: {'m0', 'C0', 'ntrend''}.

            Optional keys (choose one): {'W', 'discount'}.

    Raises:
        ValueError: If prior covariance matrix and ntrend are incompatible.
    """
    ntrend = model_dict.get('ntrend')
    C0 = model_dict.get('C0')

    if (C0.shape[0] == ntrend) & (C0.shape[1] == ntrend):
        pass
    else:
        error_message = ("Prior covariance matrix" +
                         "and declared ntrend are incomplatible")
        raise ValueError(error_message)


def validate_model_dict_regression_covariance_matrix(model_dict: dict):
    """
    Validate prior covariance matrix shape for regression model.

    Args:
        model_dict (dict): Dictionary containing model keys.

            Obrigatory keys: {'m0', 'C0', 'nregn'}.

            Optional keys (choose one): {'W', 'discount'}.

    Raises:
        ValueError: If prior covariance matrix and nregn are incompatible.
    """
    nregn = model_dict.get('nregn')
    C0 = model_dict.get('C0')

    if (C0.shape[0] == nregn) & (C0.shape[1] == nregn):
        pass
    else:
        error_message = ("Prior covariance matrix" +
                         "and declared nregn are incomplatible")
        raise ValueError(error_message)


def validate_model_dict_seasonal_covariance_matrix(model_dict: dict):
    """
    Validate prior covariance matrix shape for seasonal model.

    Args:
        model_dict (dict): Dictionary containing model keys.

            Obrigatory keys: {'m0', 'C0', 'seas_period',
                              'seas_harm_components'}.

            Optional keys (choose one): {'W', 'discount'}.

    Raises:
        ValueError: If prior covariance matrix and seas_harm_components
        are incompatible.
    """
    seas_harm_components = model_dict.get('seas_harm_components')
    nseas = 2 * len(seas_harm_components)
    C0 = model_dict.get('C0')

    if (C0.shape[0] == nseas) & (C0.shape[1] == nseas):
        pass
    else:
        error_message = ("Prior covariance matrix" +
                         "and declared seas_harm_components" +
                         "are incomplatible")
        raise ValueError(error_message)


def validate_model_dict_transfer_function_covariance_matrix(model_dict: dict):
    """
    Validate prior covariance matrix shape for transfer function model.

    Args:
        model_dict (dict): Dictionary containing model keys.

            Obrigatory keys: {'m0', 'C0', 'seas_period',
                              'seas_harm_components'}.

            Optional keys (choose one): {'W', 'discount'}.

    Raises:
        ValueError: If prior covariance matrix and seas_harm_components
        are incompatible.
    """
    gamma_order = model_dict.get('gamma_order')
    lambda_order = model_dict.get('lambda_order')
    ntfm = model_dict.get('ntfm')
    nparams = (gamma_order + 2 * lambda_order) * ntfm
    C0 = model_dict.get('C0')

    if (C0.shape[0] == nparams) & (C0.shape[1] == nparams):
        pass
    else:
        error_message = ("Prior covariance matrix" +
                         "and declared ntfm, gamma_order, lambda_order" +
                         "are incomplatible")
        raise ValueError(error_message)


def validate_model_dict_autoregressive_covariance_matrix(model_dict: dict):
    """
    Validate prior covariance matrix shape for autoregressive model.

    Args:
        model_dict (dict): Dictionary containing model keys.

            Obrigatory keys: {'m0', 'C0', 'order'}.

            Optional keys (choose one): {'W', 'discount'}.

    Raises:
        ValueError: If prior mean array and model parameters are incompatible.
    """
    order = model_dict.get('order')
    nparams = 2 * order
    C0 = model_dict.get('C0')

    if (C0.shape[0] == nparams) & (C0.shape[1] == nparams):
        pass
    else:
        error_message = ("Prior mean array and declared order" +
                         "are incomplatible")
        raise ValueError(error_message)
