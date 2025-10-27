from collections.abc import Iterable
import numpy as np
import pandas as pd

__all__ = ['integral_v_u', 'time_integral_v_u', 'ratio_uncertainty', 'ratio_v_u', 'product_v_u', 'dot_product_v_u',
           '_make_df']

def integral_v_u(s: pd.Series) -> tuple[float, float]:
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
        The absolute uncertainty of the sum, calculated as the inverse of
        the square root of the sum.

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

def time_integral_v_u(s: pd.DataFrame) -> tuple[float, float]:
    """
    Compute the time integral (c.dot(dt)) of a series and its associated uncertainty.

    Parameters
    ----------
    s : pd.DataFrame
        The series of values to inegrate.
        Has `Time` and `value` columns.

    Returns
    -------
    v : float
        The sum of the series.
    u : float
        The absolute uncertainty of the sum, calculated as the inverse of the square root of the sum.

    Notes
    -----
    s is assumed to be steps-post and the data are treated accordingly, hence s
    should end 1 time step after the desired end of integration.
    To allow calculation of time differences, s should start 1 time spep before
    the desired start time.

    """
    v = (s.value * s.Time.diff().shift(-1).apply(lambda x: x.total_seconds())).sum()
    u = np.sqrt(v)
    return v, u

def ratio_uncertainty(n, un, d, ud) -> tuple[float, float]:
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

def ratio_v_u(n: pd.DataFrame, d: pd.DataFrame) -> tuple[float, float]:
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

def product_v_u(factors: Iterable[pd.DataFrame]) -> tuple[float, float]:
    """
    Computes the product of a number of values and propagates their uncertainty
    to the result.

    Takes
    -----
    factors : Iterable[pd.DataFrame]
        the factors to multiply. Each dataframe should come with `value` and
        `uncertainty [%]` columns.
    """
    v = pd.concat([df["value"] for df in factors], axis=1).prod(axis=1)
    # it is easier to work in relative terms for products and turn to absolute later on
    u = (pd.concat([df["uncertainty [%]"] for df in factors], axis=1) / 100
         ).pow(2).sum(axis=1)
    return v, np.sqrt(u) * v

def dot_product_v_u(a: pd.DataFrame, b: pd.DataFrame) -> tuple[float, float]:
    """
    Calculates value and uncertainty of the dot product
    of two vectors.

    Params
    ------
    a: pd.DataFrame
        First vector: a data frame with `'value'` and
        `'uncertainty'` columns.
    b: pd.DataFrame
        Second vector: a data frame with `'value'` and
        `'uncertainty'` columns.
    
    Retruns
    -------
    v : float
        The dot product value.
    u : float
        The dot product absolute uncertainty.
    """
    idx = a.index.intersection(b.index)
    a_ = a.loc[idx]
    b_ = b.loc[idx]
    v = a_.value @ b_.value
    u = (a_.value **2 @ b_.uncertainty **2) + (a_.uncertainty **2 @ b_.value **2)
    return v, np.sqrt(u)

def sum_v_u(addends: Iterable[pd.DataFrame]) -> tuple[float, float]:
    """
    Calculates value and uncertainty of a sum.

    Params
    ------
    a: Iterable[pd.DataFrame]
        Lists all items to sum. Each is a data frame
        with `'value'` and `'uncertainty'` columns.
    
    Retruns
    -------
    v : float
        The sum value.
    u : float
        The sum absolute uncertainty.
    """
    a = pd.concat(addends)
    return a["value"].sum(), np.sum(a["uncertainty"] **2)

def _make_df(v, u, relative: bool=True, idx: pd.Index=None) -> pd.DataFrame:
    """
    Create a pandas DataFrame with the given value and uncertainty.

    Parameters
    ----------
    v : Iterable | float
        The value to store in the DataFrame.
    u : Iterable | float
        The uncertainty to store in the DataFrame.
    relative : bool, optional
        Flag to enable calulation of the relative uncertainty too.
    idx : pd.Index, optional
        Index of the output dataframe.
        Default is None, index is set to `'value'`.

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
    if not isinstance(v, Iterable) and not isinstance(v, (str, bytes)):
        idx_ = idx if idx is not None else ['value']
        rel = u / v * 100 if relative else np.nan
        out = pd.DataFrame({'value': v, 'uncertainty': u, 'uncertainty [%]': rel},
                           index=idx_)
    elif isinstance(v, (str, bytes)):
        raise TypeError(f"{type(v)} is not an allowed type for _make_df")
    else:
        v_, u_ = np.array(v), np.array(u)
        rel = u_ / v_ * 100 if relative else [np.nan] * len(v_)
        idx_ = idx if idx is not None else ['value'] * len(v_)
        out = pd.DataFrame({'value': v_, 'uncertainty': u_, 'uncertainty [%]': rel},
                           index=idx_)
    return out
