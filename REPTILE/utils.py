import numpy as np
import pandas as pd

def integral_v_u(s: pd.Series):
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
    >>> from REPTILE.utils import integral_v_u
    >>> s = np.array([1, 2, 3, 4])
    >>> v, u = integral_v_u(s)
    >>> print(f"Sum: {v}, Uncertainty: {u}")
    Sum: 10, Uncertainty: 3.1622776601683794
    """
    v = s.sum()
    u = np.sqrt(v)
    return v, u

def ratio_uncertainty(n, un, d, ud):
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
    >>> from REPTILE.utils import ratio_uncertainty
    >>> n, un = 10, 0.5
    >>> d, ud = 5, 0.2
    >>> u_ratio = ratio_uncertainty(n, un, d, ud)
    >>> print(f"Uncertainty of the ratio: {u_ratio}")
    Uncertainty of the ratio: 0.1118033988749895
    """
    return np.sqrt((1 / d * un)**2 + (n / d**2 * ud)**2)

def ratio_v_u(n: pd.DataFrame, d: pd.DataFrame):
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
    >>> from REPTILE.utils import ratio_v_u
    >>> n = Measurement(10, 0.5)
    >>> d = Measurement(5, 0.2)
    >>> v, u = ratio_v_u(n, d)
    >>> print(f"Ratio: {v}, Uncertainty: {u}")
    Ratio: 2.0, Uncertainty: 0.1118033988749895
    """
    v = n.value / d.value
    u = ratio_uncertainty(n.value, n.uncertainty, d.value, d.uncertainty)
    return v, u

def _make_df(v, u, relative=True):
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
    >>> from REPTILE.utils import _make_df
    >>> v, u = 10, 0.5
    >>> df = _make_df(v, u)
    >>> print(df)
            value  uncertainty
    value    10.0          0.5
    """
    out = pd.DataFrame({'value': v, 'uncertainty': u, 'uncertainty [%]': u / v * 100},
                       index=['value']) if relative else pd.DataFrame({'value': v, 'uncertainty': u, 'uncertainty [%]': np.nan},
                                                                      index=['value'])
    return out
