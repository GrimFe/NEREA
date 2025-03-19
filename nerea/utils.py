from collections.abc import Iterable
import numpy as np
import pandas as pd
import warnings

__all__ = ['integral_v_u', 'ratio_uncertainty', 'ratio_v_u', 'product_v_u', '_make_df', 'get_fit_R2']

def integral_v_u(s: pd.Series) -> tuple[float]:
    """
    Compute the integral (sum) of a series and its associated uncertainty.

    Parameters
    ----------
    s : pd.Series
        The series of values to sum.

    Returns
    -------
    v : float
        The sum of the series.
    u : float
        The absolute uncertainty of the sum, calculated as the inverse of the square root of the sum.

    Examples
    --------
    >>> import numpy as np
    >>> from nerea.utils import integral_v_u
    >>> s = np.array([1, 2, 3, 4])
    >>> v, u = integral_v_u(s)
    >>> print(f"Sum: {v}, Uncertainty: {u}")
    Sum: 10, Uncertainty: 3.1622776601683794
    """
    v = s.sum()
    u = np.sqrt(v)
    return v, u

def ratio_uncertainty(n, un, d, ud) -> tuple[float]:
    """
    Compute the uncertainty of a ratio given the values and uncertainties of the numerator and denominator.

    Parameters
    ----------
    n : float
        The value of the numerator.
    un : float
        The absolute uncertainty of the numerator.
    d : float
        The value of the denominator.
    ud : float
        The absolute uncertainty of the denominator.

    Returns
    -------
    float
        The absolute uncertainty of the ratio.

    Examples
    --------
    >>> from nerea.utils import ratio_uncertainty
    >>> n, un = 10, 0.5
    >>> d, ud = 5, 0.2
    >>> u_ratio = ratio_uncertainty(n, un, d, ud)
    >>> print(f"Uncertainty of the ratio: {u_ratio}")
    Uncertainty of the ratio: 0.1118033988749895
    """
    return np.sqrt((1 / d * un)**2 + (n / d**2 * ud)**2)

def ratio_v_u(n: pd.DataFrame, d: pd.DataFrame) -> tuple[float]:
    """
    Compute the value and uncertainty of a ratio given objects with value and uncertainty attributes.

    Parameters
    ----------
    n : pd.DataFrame
        An object with `value` and `uncertainty` attributes representing the numerator.
    d : pd.DataFrame
        An object with `value` and `uncertainty` attributes representing the denominator.

    Returns
    -------
    v : float
        The value of the ratio.
    u : float
        The uncertainty of the ratio.

    Examples
    --------
    >>> class Measurement:
    ...     def __init__(self, value, uncertainty):
    ...         self.value = value
    ...         self.uncertainty = uncertainty
    ...
    >>> from nerea.utils import ratio_v_u
    >>> n = Measurement(10, 0.5)
    >>> d = Measurement(5, 0.2)
    >>> v, u = ratio_v_u(n, d)
    >>> print(f"Ratio: {v}, Uncertainty: {u}")
    Ratio: 2.0, Uncertainty: 0.1118033988749895
    """
    v = n.value / d.value
    u = ratio_uncertainty(n.value, n.uncertainty, d.value, d.uncertainty)
    return v, u

def product_v_u(factors: Iterable[pd.DataFrame]) -> tuple[float]:
    """
    Computes the product of a number of values and propagates their uncertainty
    to the result.

    Takes
    -----
    factors : Iterable[pd.DataFrame]
        the factors to multiply. Each dataframe should come with `value` and
        `uncertainty [%]` columns.
    """
    v = np.prod([x.value for x in factors])
    # it is easier to work in relative terms for products and turn to absolute later on
    u = np.sqrt(np.sum([(x['uncertainty [%]']/100)**2 for x in factors])) * v
    return v, u

def sum_v_u(addends: Iterable[pd.DataFrame]) -> tuple[float]:
    a = pd.concat(addends)
    return a["value"].sum(), np.sum(a["uncertainty"] **2)

def _make_df(v, u, relative=True) -> pd.DataFrame:
    """
    Create a pandas DataFrame with the given value and uncertainty.

    Parameters
    ----------
    v : float
        The value to store in the DataFrame.
    u : float
        The uncertainty to store in the DataFrame.
    relative : bool, optional
        flag to enable calulation of the relative uncertainty too.

    Returns
    -------
    pd.DataFrame
        A DataFrame with `value` and `uncertainty` columns.

    Examples
    --------
    >>> from nerea.utils import _make_df
    >>> v, u = 10, 0.5
    >>> df = _make_df(v, u)
    >>> print(df)
            value  uncertainty
    value    10.0          0.5
    """
    if not isinstance(v, Iterable):
        rel = u / v * 100 if relative else np.nan
        out = pd.DataFrame({'value': v, 'uncertainty': u, 'uncertainty [%]': rel},
                           index=['value'])
    else:
        rel = u / v * 100 if relative else [np.nan] * len(v)
        out = pd.DataFrame({'value': v, 'uncertainty': u, 'uncertainty [%]': rel},
                           index=['value'] * len(v))
    return out

def get_fit_R2(y, fvec, weight: Iterable[float]=None):
    # calculation of fit R2
    # we use weighted average to account for uncertainties on y
    # https://stats.stackexchange.com/questions/439590/how-does-r-compute-r-squared-for-weighted-least-squares
    w = np.array([1] * len(y)) if weight is None else weight
    weighted_mean_y = np.average(y, weights=w)
    r2 = 1 - (fvec ** 2).sum() / np.sum((y - weighted_mean_y) ** 2 * w)
    return r2

def smoothing(data: pd.Series, method: str="moving_average", *args, **kwargs) -> pd.DataFrame:
    """
    Calculates the sum of 'counts' and the minimum value of 'channel' for each
    group based on the integer division of 'channel' by 10.
    Contains the data used to find `max` and hence `R`.

    Parameters
    ----------
    data : ps.Series
        the data to smooth
    method : str, optional
        the method to use. Allowed options are:
            - moving_average
                (requires window)
            - savgol_filter
                (requires window_length
                        polyorder)
        Defailt is "moving_average"
    * args :
        arguments for the chosen method
    **kwargs :
        arguments for the chosen method

    Returns
    -------
    pd.DataFrame
    """
    s = data.copy()
    match method:
        case "moving_average":
            if kwargs.get("window") is None: kwargs["window"] = 10
            s = s.rolling(*args, **kwargs).mean().fillna(0)
        case "savgol_filter":
            from scipy.signal import savgol_filter
            s = savgol_filter(s, *args, **kwargs)
            if any(s < 0):
                warnings.warn("Using Savgol Filter smoothing, negative values appear.")
        case _:
            raise Exception(f"The chosen method {method} is not allowed")
    return s
