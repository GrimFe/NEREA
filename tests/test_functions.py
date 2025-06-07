import pytest
import numpy as np
import pandas as pd
from nerea.functions import *

@pytest.fixture
def synthetic_one_g_xs_data():
    data = pd.DataFrame({'U236': [0.07, 0.001], 'U234': [0.08, 0.002],
                         'U238': [0.9, 0.003], 'U235': [0.6, 0.004]}).T.reset_index()
    data.columns = ['nuclide', 'value', 'uncertainty']
    return data.set_index('nuclide')

def test_polynomial():
    assert polynomial(2, [1, 2, 3], 1) == 6.

def test_fitting_polynomial():
    assert fitting_polynomial(2)(1, *[1,2,3]) == polynomial(2, [1, 2, 3], 1) == 6.

def test_polyfit():
    # It looks like the fit differs here and on the GitHub PC -> assert with tolerance
    data = pd.DataFrame({'x': [1, 2, 3],
                         'y': [2, 4, 6],
                         'u': [0.01, 0.01, 0.01]})
    coef, coef_cov = polyfit(1, data)
    np.testing.assert_almost_equal(coef,
                                   np.array([ 2.00000000e+00, -2.18001797e-12]))
    np.testing.assert_almost_equal(coef_cov, np.array(
                                            [[ 1.00000000e-05, -7.31498003e-10],
                                             [-7.31498003e-10,  1.87281265e-13]]),
                                    decimal=5)
    
    # test zero uncertainties
    data = pd.DataFrame({'x': [1, 2, 3],
                         'y': [2, 4, 6],
                         'u': [0, 0.01, 0.01]})
    data_ = pd.DataFrame({'x': [1, 2, 3],
                         'y': [2, 4, 6],
                         'u': [0.01 * 1e-3, 0.01, 0.01]})
    coef, coef_cov = polyfit(1, data)
    coef_, coef_cov_ = polyfit(1, data_)
    np.testing.assert_almost_equal(coef, coef_)
    np.testing.assert_almost_equal(coef_cov, coef_cov_)

    # test NaN uncertainties
    data = pd.DataFrame({'x': [1, 2, 3],
                         'y': [2, 4, 6],
                         'u': [np.nan, 0.01, 0.01]})
    data_ = pd.DataFrame({'x': [1, 2, 3],
                         'y': [2, 4, 6],
                         'u': [0.01 * 1e-3, 0.01, 0.01]})
    coef, coef_cov = polyfit(1, data)
    coef_, coef_cov_ = polyfit(1, data_)
    np.testing.assert_almost_equal(coef, coef_)
    np.testing.assert_almost_equal(coef_cov, coef_cov_)

def test_get_fit_R2():
    y = [1, 2, 3, 4, 5, 6, 7]
    f = [0, 0, 0, 0, 0, 0, 0]
    assert get_fit_R2(y, f) == 1

    f = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    r2 = 1 - sum(np.array(f) **2) / np.sum((4 - np.array(y)) **2)
    assert get_fit_R2(y, f) == r2

    w = [1, 1, 1, 1, 1, 1, 1]
    assert get_fit_R2(y, f, w) == r2

    w = [2, 2, 2, 2, 2, 2, 2]
    assert get_fit_R2(y, f, w) == 0.975

def test_smoothing():
    data = pd.Series([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.])
    ma = smoothing(data)
    pd.testing.assert_series_equal(ma, pd.Series([0.] * 9 + [5.5, 6.5, 7.5, 8.5]))
    sf = smoothing(data, smoothing_method='savgol_filter', window_length=2, polyorder=1)
    pd.testing.assert_series_equal(sf, pd.Series([1.] + list(np.arange(2, 13) + 0.5) + [13.]), atol=1e-5)
    ewm = smoothing(data, 'ewm', span=2)
    pd.testing.assert_series_equal(ewm, pd.Series([1.0649573834741848,
                                                   1.8636754210798234,
                                                   2.785273156778637,
                                                   3.7805987113333557,
                                                   4.814311477358505,
                                                   5.866042730400385,
                                                   6.9256331946335825,
                                                   7.988479104572819,
                                                   9.052624733235037,
                                                   10.117275497527087,
                                                   11.182118655723848,
                                                   12.247033956860777,
                                                   13.311975977021957]), atol=1e-5)

def impurity_correction(synthetic_one_g_xs_data):
    w1, uw1, w2, uw2, wd, uwd = .1, .01, .2, .02, .7, .07
    x1, ux1, x2, ux2, xd, uxd = .07 / 236., .001 / 236., .08 / 234., .002 / 234., .6 / 235.043923, .004 / 235.043923
    v = (w1 / wd * x1 / xd) + (w2 / wd * x2 / xd)

    W1, X1 = w1 / wd, x1 / xd
    vW1, vX1 = (uw1 / wd) **2 + (w1 / wd **2 * uwd) **2, (ux1 / xd) **2 + (x1 / xd **2 * uxd) **2 
    W2, X2 = w2 / wd, x2 / xd
    vW2, vX2 = (uw2 / wd) **2 + (w2 / wd **2 * uwd) **2, (ux2 / xd) **2 + (x2 / xd **2 * uxd) **2
    u = np.sqrt(vW1 * X1**2 + W1**2 * vX1 + vW2 * X2**2 + W2**2 * vX2)

    data = pd.DataFrame({'value': [v], 'uncertainty': [u], 'uncertainty [%]': u / v * 100}, index=['value'])

    imp = pd.DataFrame({'U236': [0.1, 0.01], 'U234': [0.2, 0.02], 'U238': [0.7, 0.07]}).T.reset_index()
    imp.columns = ['nuclide', 'value', 'uncertainty']
    nerea_ = impurity_correction(
        imp=imp,
        one_g_xs=synthetic_one_g_xs_data,
        dep_num="U238",
        dep_den="U235"
    )

    np.testing.assert_equal(data.index.values, nerea_.index.values)
    np.testing.assert_equal(data.columns.values, nerea_.columns.values)
    np.testing.assert_almost_equal(data['value'].values, nerea_['value'].values, decimal=5)
    np.testing.assert_almost_equal(data['uncertainty'].values, nerea_['uncertainty'].values, decimal=6)
