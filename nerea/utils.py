from collections.abc import Iterable, Callable
from functools import reduce
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import warnings
from scipy.optimize import curve_fit

from inspect import signature

__all__ = ['integral_v_u', 'time_integral_v_u', 'ratio_uncertainty', 'ratio_v_u', 'product_v_u', 'dot_product_v_u',
           '_make_df', 'fitting_polynomial', 'polynomial', 'get_fit_R2', 'polyfit', 'smoothing',
           '_normalize_array', 'get_relative_array', 'impurity_correction']

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

def time_integral_v_u(s: pd.DataFrame) -> tuple[float]:
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
    v = pd.concat([df["value"] for df in factors], axis=1).prod(axis=1)
    # it is easier to work in relative terms for products and turn to absolute later on
    u = (pd.concat([df["uncertainty [%]"] for df in factors], axis=1) / 100
         ).pow(2).sum(axis=1)
    return v, np.sqrt(u) * v

def dot_product_v_u(a: pd.DataFrame, b: pd.DataFrame) -> tuple[float]:
    idx = a.index.intersection(b.index)
    a_ = a.loc[idx]
    b_ = b.loc[idx]
    v = a_.value @ b_.value
    u = (a_.value **2 @ b_.uncertainty **2) + (a_.uncertainty **2 @ b_.value **2)
    return v, np.sqrt(u)

def sum_v_u(addends: Iterable[pd.DataFrame]) -> tuple[float]:
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

def polynomial(order: int, c: Iterable[float], x: float):
    if len(c) != order + 1:
            raise ValueError(f"Expected {order + 1} coefficients, got {len(c)}")
        # returns the polynomial value @x
    return sum([c[i] * x **(order - i) for i in range(order + 1)])

def fitting_polynomial(order: int) -> float:
    def poly(x, *c):
        if len(c) != order + 1:
            raise ValueError(f"Expected {order + 1} coefficients, got {len(c)}")
        # returns the polynomial value @x
        return sum([c[i] * x **(order - i) for i in range(order + 1)])
    return poly

def get_fit_R2(y: Iterable[float], fvec: Iterable[float], weight: Iterable[float]=None) -> float:
    """
    Calculates the R2 of a fit from the fitted points and the fitting line residuals.

    Paramters
    ---------
    y : Iterable[float]
        the fitted y points
    fvec : Iterable[float]
        residuals of the fitting line corresponding to those of y
    weight : Iterable[float], optional
        the weights for R2 weighting. Degaults to None, meaning w = 1

    Returns:
    --------
    r2 : float
        the fit R^2.

    Notes
    -----
    assumes y and fvec share x.
    """
    # calculation of fit R2
    # we use weighted average to account for uncertainties on y
    # https://stats.stackexchange.com/questions/439590/how-does-r-compute-r-squared-for-weighted-least-squares
    w = np.array([1] * len(y)) if weight is None else weight
    weighted_mean_y = np.average(y, weights=w)
    r2 = 1 - (np.array(fvec) ** 2).sum() / np.sum((y - weighted_mean_y) ** 2 * w)
    return r2

def polyfit(order: int, data: pd.DataFrame) -> tuple[np.array, np.array, Callable]:
    """
    Fits the data with a polynomial of chosen order.

    Parameters:
    -----------
    order : int
        fitting polynomial order.
    data : pd.DataFrame
        Dataframe with data to fit. Columns should be:
            - x, abscissa
            - y, ordinate
            - u, y-uncertainty

    Returns:
    --------
    coef : np.array
        fit coefficients
    coef_cov : np.array
        fit coefficients covariance matrix
    
    Notes:
    ------
    Replaces NaN and negative or zero uncertianties with
    non-zero small values to allow fitting.
    """
    data_ = data.copy()
    zero_u = data_.query("u > 0").u.min() * 1e-3  # Replacing meaningless uncertainties with this
    data_.u = data_.u.fillna(zero_u)
    data_.loc[data_.u <= 0, 'u'] = zero_u
    coef, coef_cov, out, _, _ = curve_fit(fitting_polynomial(order),
                                          data_.x,
                                          data_.y,
                                          sigma=data_.u,
                                          absolute_sigma=True,
                                          full_output=True,
                                          p0=[1] * (order + 1)  ## needed to set number of fit params
                                          )
    r2 = get_fit_R2(data_.y, out["fvec"], weight=1 / data_.u **2)
    warnings.warn(f"CR reactivity curve fit R^2 = {r2}")
    return coef, coef_cov

def smoothing(data: pd.Series,
              smoothing_method: str="moving_average",
              renormalize: bool=False,
              **kwargs) -> pd.DataFrame:
    """
    Calculates the sum of 'counts' and the minimum value of 'channel' for each
    group based on the integer division of 'channel' by 10.
    Contains the data used to find `max` and hence `R`.

    Parameters
    ----------
    data : ps.Series
        the data to smooth
    smoothing_method : str, optional
        the method to use. Allowed options are:
            - moving_average
                (requires window)
            - ewm
            - savgol_filter
                (requires window_length
                        polyorder)
            - fit
                (requires ch_before_max,
                        order)
        Defailt is "moving_average"
    renormalize : bool, optional
        defines whether the smoothed data shall be
        renormalized to the data integral.
    **kwargs :
        arguments for the chosen method

    Returns
    -------
    pd.DataFrame
    """
    kwargs = {'window': 10, 'window_length': 10,
              'ch_before_max': 10, 'order': 3, 'ch_lld': 0} | kwargs
    if not np.any([k in kwargs.keys() for k in ["com", "span", "halflife", "alpha"]]):
        kwargs['span'] = 10
    s = data.copy()
    s.astype('float64')
    match smoothing_method:
        case "moving_average":
            kwargs = {k: v for k, v in kwargs.items() if k in set(signature(pd.Series.rolling).parameters)}
            s = s.rolling(**kwargs).mean().fillna(0)
        case "ewm": # exponentially weighted mean
            kwargs = {k: v for k, v in kwargs.items() if k in set(signature(pd.Series.ewm).parameters)}
            s = s.ewm(**kwargs).mean()
        case "savgol_filter":
            from scipy.signal import savgol_filter
            kwargs = {k: v for k, v in kwargs.items() if k in set(signature(savgol_filter).parameters)}
            s = savgol_filter(s, **kwargs)
            if any(s < 0):
                warnings.warn("Using Savgol Filter smoothing, negative values appear.")
            s = pd.Series(s, index=data.index)
        case "fit":
            start = s.loc[kwargs['ch_lld']:].idxmax() - kwargs['ch_before_max']
            start_ = start if start > 0 else 0
            order = kwargs['order']
            kwargs = {k: v for k, v in kwargs.items() if k in set(signature(curve_fit).parameters)}
            # truncate fit at first zero after max if any
            if s.iloc[-1] == 0:
                lst_ch = s.loc[s.idxmax():][s.loc[s.idxmax():] == 0].index[0]
                zeros = pd.Series(np.zeros(s.loc[lst_ch:].shape[0]),
                                index=range(lst_ch + 1,
                                            s.loc[lst_ch:].shape[0] + lst_ch + 1),
                                name='counts')
            else:
                lst_ch = s.index[-1]
                zeros = None
            s = s.loc[start_:lst_ch]
            coef, _ = curve_fit(fitting_polynomial(order),
                                s.index,
                                s.values,
                                p0=[1] * (order + 1),
                                **kwargs
                                )
            s = pd.Series(fitting_polynomial(order)(s.index, *coef), index=s.index, name='counts')
            s = pd.concat([data.loc[:start_ - 1], s.loc[start_:lst_ch], zeros])
        case _:
            raise Exception(f"The chosen method {smoothing_method} is not allowed.")
    if renormalize:
        s *= (data.sum() / s.sum())
    return s

def _normalize_array(a: pd.DataFrame,
                     d: str) -> pd.DataFrame:
    """
    Calculates array a normalized to its entry
    with index d. Supports `get_relative_array()`.

    Paramters
    ---------
    a : pd.DataFrame
        the array to make relative.
        Has columns for its value and uncertainty.
    den : str
        flag to make the array relative to one entry `den`.

    Return
    ------
    pd.DataFrame
    """
    a_ = a.copy()
    idx = a_.index
    den = pd.concat([a_.loc[d]] * a_.shape[0],
                    axis=1).T.reset_index()
    a_ = _make_df(*ratio_v_u(a_.reset_index(),
                             den))[['value', 'uncertainty']]
    a_.index = idx
    # the denominator is fully correlated with itself:
    a_.loc[d, 'uncertainty'] = 0
    return a_

def get_relative_array(a: dict[str, float] | pd.DataFrame,
                       den: str = '') -> pd.DataFrame:
    """
    Transforms composition array making it relative to its main component.

    Paramters
    ---------
    a : dict[str, float] | pd.DataFrame
        the array to make relative.
        `key` is the nuclide string identifier (e.g., `'U235'`),
        and `value` is its array value.
        Has columns for its value and uncertainty.
    den : str, optional
        flag to make the array relative to one entry `den`.
        Default is `''` to normalize to the maximum.

    Return
    ------
    pd.DataFrame
    """
    a_ = pd.DataFrame(a, index=['value', 'uncertainty']
            ).T if not isinstance(a, pd.DataFrame) else a.copy()
    if den == '':
        match a_.value.max():
            case 1:
                pass
            case 100:
                a_.value /= 100
                a_.uncertainty /= 100
            case _:
                a_ = _normalize_array(a_, a_.value.idxmax())
    else:
        a_ = _normalize_array(a_, den)
    return a_

def impurity_correction(one_group_xs: dict[str, float],
                        composition: pd.DataFrame,
                        xs_den: str='',
                        drop_main: bool=False,
                        **kwargs) -> pd.DataFrame:
        """
        Calculates the fission chamber calibration coefficient.

        Paramters
        ---------
        one_group_xs : dict[str, float]
            the one group cross sections of the fission
            chamber components. `key` is the nuclide
            string identifier (e.g., `'U235'`), and `value`
            is its one group cross section.
            Has columns for its value and uncertainty.
        composition : pd.DataFrame
            the fission chamber composition relative to
            the deposit main nuclide. `key` is the nuclide
            string identifier (e.g., `'U235'`), and `value`
            is its atomic abundance relative to the main one.
            Has columns for its value and uncertainty.
        xs_den : str, optional
            the cross section entry of `one_group_xs` to
            normalize the impurity correction to.
            Default is `''` for no cross section normalization.
        drop_main : bool, optional
            flag to drop the main nuclide.
            Default is False.
        **kwargs 
            keyword arguments passed to `_make_df()`

        Returns 
        -------
        pd.DataFrame

        Note
        ----
        The implementation features a separate calculation of
        value and uncertainty to proper account for uncertianties
        in dot products where the elements of the two vectors
        are variable.
        This also covers for the setting of u(N_main) to 0 in
        `get_relative_composition` when renormalization is required.
        """
        if composition.shape[0] <= 1 and drop_main:
            # one cannot remove main if there is one nuclide ony
            warnings.warn(f"Removing main from processing of mono-isotopic deposit.")
            out = _make_df(0, 0, relative=False)
        else:
            one_over_xsd = 1 / one_group_xs.loc[xs_den].value if xs_den else 1
            c = get_relative_array(composition)
            main = c.query("value == 1").index.values[0]
            if drop_main:
                c = c.drop(main)
                comp = composition.drop(main)  # comp is used to compute uncertainties
            else:
                comp = composition.copy()
            # Only the xs in the composition array are taken
            # xs normalization is performed at the end
            if not isinstance(one_group_xs, pd.DataFrame):
                xs = pd.DataFrame(one_group_xs,
                                index=['value', 'uncertainty']
                                ).T.loc[c.index]
            else:
                xs = one_group_xs.copy().loc[c.index]
            ## BEST ESTIMATE
            v, _ = dot_product_v_u(c, xs)
            v = v * one_over_xsd

            ## UNCERTAINTY
            na = composition.loc[main]
            # use unchanged composition uncertainty data as I treat the ratio
            # Ni / Na separately.
            # the variance from Ni is just the sum of variances from each Ni
            var_ni = 1 / na.value **2 * (xs.value **2 @ comp.uncertainty **2) * one_over_xsd **2
            # If the main nuclide was not removed, its var_ni quota should be
            # removed as it is anyways correlated 1 with itself
            if not drop_main:
                var_ni -= (na.uncertainty / na.value * xs.loc[main].value * one_over_xsd) **2
            # the variance from Na instead takes the sensitivity of the output to
            # Na, which is the dot product of Ni and xs
            # here again we use the unmodified composition array
            # Same as before I need to subtract one contribution ot var_na
            # if not drop_main as main is still in xs and comp
            if drop_main:
                var_na = 1 / na.value **4 * (xs.value @ comp.value
                                        ) **2 * na.uncertainty **2 * one_over_xsd **2
            else:
                var_na = 1 / na.value **4 * (xs.drop(main).value @ comp.drop(main).value
                                        ) **2 * na.uncertainty **2 * one_over_xsd **2
            # the sensitivity of the output to xi is just the normalized
            # composition array in composition_
            var_xi = c.value **2 @ xs.uncertainty **2 * one_over_xsd **2
            # If normalization to a cross section is required, then
            # sensitivity to it is computed taking the normalized
            # concentration array and the cross section value
            if xs_den:
                x = one_group_xs.loc[xs_den]
                var_xd = 1 / x.value **4 * (c.value @ xs.value
                                            ) **2 * x.uncertainty **2
            else:
                var_xd = 0
            u = np.sqrt(var_ni + var_na + var_xi + var_xd)
            out = _make_df(v, u, **kwargs)
        return out
