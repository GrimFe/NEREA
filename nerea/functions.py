import warnings
import pandas as pd
import numpy as np

from collections.abc import Iterable, Callable
from scipy.optimize import curve_fit
from inspect import signature

from .constants import ATOMIC_MASS
from .utils import ratio_v_u, _make_df

__all__ = ['fitting_polynomial', 'polynomial', 'get_fit_R2', 'polyfit', 'smoothing', 'impurity_correction']


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

def smoothing(data: pd.Series, smoothing_method: str="moving_average", **kwargs) -> pd.DataFrame:
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
            - savgol_filter
                (requires window_length
                        polyorder)
        Defailt is "moving_average"
    **kwargs :
        arguments for the chosen method

    Returns
    -------
    pd.DataFrame
    """
    kwargs = {'window': 10, 'window_length': 10} | kwargs
    if not np.any([k in kwargs.keys() for k in ["com", "span", "halflife", "alpha"]]):
        kwargs['span'] = 10
    s = data.copy()
    match smoothing_method:
        case "moving_average":
            kwargs = {k: v for k, v in kwargs.items() if k in set(signature(pd.Series.rolling).parameters)}
            s = s.rolling(**kwargs).mean().fillna(0)
        case "ewm": # exponentially weighted mean
            kwargs = {k: v for k, v in kwargs.items() if k in set(signature(pd.Series.ewm).parameters)}
            s = s.ewm(**kwargs).mean()
            s *= (data.sum() / s.sum())
        case "savgol_filter":
            from scipy.signal import savgol_filter
            kwargs = {k: v for k, v in kwargs.items() if k in set(signature(savgol_filter).parameters)}
            s = savgol_filter(s, **kwargs)
            if any(s < 0):
                warnings.warn("Using Savgol Filter smoothing, negative values appear.")
            s = pd.Series(s, index=data.index)
        case _:
            raise Exception(f"The chosen method {smoothing_method} is not allowed.")
    return s

def impurity_correction(imp: pd.DataFrame, one_g_xs: pd.DataFrame, num_dep: str, den_dep: str):
    # normalize Serpent output per unit mass
    v = one_g_xs['value'] / ATOMIC_MASS.T['value']
    u = one_g_xs['uncertainty'] / ATOMIC_MASS.T['value']
    xs = pd.concat([v.dropna(), u.dropna()], axis=1)
    xs.columns = ["value", "uncertainty"]

    # normalize impurities and one group xs to the numerator deposit
    imp_v, imp_u = ratio_v_u(imp, imp.loc[num_dep])
    xs_v, xs_u = ratio_v_u(xs, xs.loc[den_dep])
    
    # remove information on the main isotope
    # will sum over all impurities != self.numerator.deposit_id
    imp_v = imp_v.drop(num_dep)
    imp_u = imp_u.drop(num_dep)
    xs_v = xs_v.drop(num_dep)
    xs_u = xs_u.drop(num_dep)
    
    # compute correction and its uncertainty
    if not all([i in xs_v.index] for i in imp_v.index):
        warnings.warn("Not all impurities were provided with a cross section.")
    correction = sum((imp_v * xs_v).dropna())
    correction_variance = sum(((xs_v * imp_u) **2 + (imp_v * xs_u) **2).dropna())
    relative = True if imp_v.shape[0] != 0 else False
    return _make_df(correction, np.sqrt(correction_variance), relative=relative)
