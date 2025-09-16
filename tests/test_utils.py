import pytest
import types

import numpy as np
import pandas as pd
from nerea.utils import *
import datetime

EXPECTED_RATIO = 2.0
EXPECTED_UNCERTAINTY = 0.282842712474619

def test_integral_v_u():
    s = np.array([1, 2, 3, 4])
    v, u = integral_v_u(s)
    assert v == pytest.approx(10, 1e-9)
    assert u == pytest.approx(0.31622776601683794 * v, rel=1e-9)

def test_time_integral_v_u():
    d = pd.DataFrame({"Time": [datetime.datetime(2025, 3, 3, 14, 22, 0) + datetime.timedelta(seconds=i) for i in [10, 15, 25, 30]],
                      "value": [2, 5, 1, 5]})
    v, u = time_integral_v_u(d)
    assert v == 65
    assert u == np.sqrt(65)

def test_ratio_v_u():
    v1, u1, v2, u2 = 10, 1, 5, 0.5
    sample_data1 = pd.DataFrame({'value': v1, 'uncertainty': u1}, index=['value'])
    sample_data2 = pd.DataFrame({'value': v2, 'uncertainty': u2}, index=['value'])
    ratio, uncertainty = ratio_v_u(sample_data1, sample_data2)
    assert ratio.values[0] == pytest.approx(EXPECTED_RATIO, rel=1e-9)
    assert uncertainty.values[0] == pytest.approx(EXPECTED_UNCERTAINTY, rel=1e-9)

def test_ratio_uncertainty():
    v1, u1, v2, u2 = 10, 1, 5, 0.5
    expected_uncertainty = 0.282842712474619
    uncertainty = ratio_uncertainty(v1, u1, v2, u2)
    assert uncertainty == pytest.approx(expected_uncertainty, rel=1e-9)

def test_product_v_u():
    product = product_v_u([pd.DataFrame({'value': x[0],
                                         'uncertainty [%]': x[1]}, index=['value']) for x in
                                                            [(10, 10), (5, 2), (10, 20)]])
    EXPECTED_VALUE = 500
    EXPECTED_UNCERTAINTY = np.sqrt(2 **2 + 10 **2 + 20 **2) / 100 * EXPECTED_VALUE
    assert product[0].value == EXPECTED_VALUE
    assert product[1].value == EXPECTED_UNCERTAINTY

def test_dot_product_v_u():
    v1, u1, v2, u2 = 1, 0.1, 2, 0.2
    a = pd.DataFrame({'value': [v1, v2],
                      'uncertainty': [u1, u2]}
                      , index=[1, 2])
    b = pd.DataFrame({'value': [v1, v2],
                      'uncertainty': [u1, u2]}
                      , index=[2, 1])
    v, u = dot_product_v_u(a, b)
    u_ = np.sqrt((v1 * u2) **2 + (v2 * u1) **2 + (v1 * u2) **2 + (v2 * u1) **2)
    assert v == v1 * v2 + v1 * v2
    assert u == u_

def test_make_df():
    df = _make_df(EXPECTED_RATIO, EXPECTED_UNCERTAINTY)
    expected_df = pd.DataFrame({'value': [EXPECTED_RATIO],
                                'uncertainty': [EXPECTED_UNCERTAINTY],
                                'uncertainty [%]': [EXPECTED_UNCERTAINTY / EXPECTED_RATIO * 100]},
                                index=['value'])
    pd.testing.assert_frame_equal(df, expected_df)

def test_make_df_not_relative():
    df = _make_df(EXPECTED_RATIO, EXPECTED_UNCERTAINTY, relative=False)
    expected_df = pd.DataFrame({'value': [EXPECTED_RATIO],
                                'uncertainty': [EXPECTED_UNCERTAINTY],
                                'uncertainty [%]': np.nan},
                                index=['value'])
    pd.testing.assert_frame_equal(df, expected_df)

def test_make_df_iterable():
    it = _make_df(np.array([1, 2]), np.array([0.01, 0.01]), relative=True)
    target = pd.DataFrame({'value': np.array([1, 2]),
                           'uncertainty': [0.01, 0.01],
                           'uncertainty [%]': [1., 0.5]},
                           index=['value', 'value'])
    pd.testing.assert_frame_equal(it, target)
    # now with list input
    it = _make_df([1, 2], [0.01, 0.01], relative=True)
    pd.testing.assert_frame_equal(it, target)

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
    pd.testing.assert_series_equal(ewm, pd.Series([1.0,
                                                   1.75,
                                                   2.615384615384615,
                                                   3.55,
                                                   4.520661157024794,
                                                   5.508241758241757,
                                                   6.5032021957914,
                                                   7.501219512195123,
                                                   8.500457270602583,
                                                   9.500169353746104,
                                                   10.500062095672497,
                                                   11.500022580159568,
                                                   12.500008153936282]), atol=1e-5)
    data = pd.Series([1.,2.,3.,4.,5.,6.,7.,8.,7.,6.,5.,4.,1.])
    fit = smoothing(data, smoothing_method='fit')
    pd.testing.assert_series_equal(fit, pd.Series([0.9505494505493358,
                                                   1.934065934061414,
                                                   3.0309690309429116,
                                                   4.153846153767962,
                                                   5.215284715110696,
                                                   6.127872127545245,
                                                   6.804195803645741,
                                                   7.1568431559863175,
                                                   7.098401597141105,
                                                   6.541458539684238,
                                                   5.398601396189846,
                                                   3.5824175792320596,
                                                   1.0054945013850145]), atol=1e-5)
    fit = smoothing(data, smoothing_method='fit', order=5, ch_before_max=2)
    pd.testing.assert_series_equal(fit, pd.Series([1.0,
                                                   2.0,
                                                   3.0,
                                                   4.0,
                                                   5.0,
                                                   5.981351972771222,
                                                   7.100233079894906,
                                                   7.790209747459471,
                                                   7.198135116458502,
                                                   5.953379808482282,
                                                   4.937062694705332,
                                                   4.051281664868952,
                                                   0.9883443962731917]), atol=1e-5)    
    # test renormalization
    data = pd.Series([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.])
    ewm = smoothing(data, 'ewm', span=2, renormalize=True)
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

def test_normalize_array():
    a = pd.DataFrame({"value": [10, 5],
                      "uncertainty": [1, .5]},
                      index=["A", "B"])
    expected = pd.DataFrame({"value": [1, .5],
                             "uncertainty": [0, np.sqrt(
                                 .1**2 + .1**2
                                 ) * .5]},
                             index=["A", "B"])
    pd.testing.assert_frame_equal(_normalize_array(a, "A"), expected)

def test_get_relative_array():
    expected = pd.DataFrame({"value": [1, .5],
                             "uncertainty": [.1, .05]},
                             index=["A", "B"])
    c = {"A": [1, 0.1], "B": [0.5, 0.05]}
    pd.testing.assert_frame_equal(get_relative_array(c), expected)
    c = expected.copy()
    pd.testing.assert_frame_equal(get_relative_array(c), expected)
    c = expected * 100
    pd.testing.assert_frame_equal(get_relative_array(c), expected)
    c = pd.DataFrame({"value": [2, 1],
                      "uncertainty": [.2, .1]},
                      index=["A", "B"])
    expected = pd.DataFrame({"value": [1, .5],
                             "uncertainty": [0, np.sqrt(
                                 (1 / 2 * .1)**2 +
                                 (1 / 2**2 * .2) **2
                             )]},
                             index=["A", "B"])
    pd.testing.assert_frame_equal(get_relative_array(c), expected)

def test_impurity_correction():
    w1, uw1, w2, uw2, wd, uwd = .1, .01, .2, .02, .7, .07
    x1, ux1, x2, ux2, xd, uxd = .07 / 236., .001 / 236., .08 / 234.040916, .002 / 234.040916, .6 / 235.043923, .004 / 235.043923
    v = (w1/wd * x1/xd) + (w2/wd * x2/xd)

    s_w1, s_w2 = 1 / wd * x1 / xd, 1 / wd * x2 / xd
    s_x1, s_x2 = w1 / wd * 1 / xd, w2 / wd * 1 / xd
    s_wd = 1 / wd **2 * (w1 * x1 / xd + w2 * x2 / xd)
    s_xd = 1 / xd **2 * (w1 / wd * x1 + w2 / wd * x2)
    u = np.sqrt((s_w1 * uw1) **2 + (s_w2 * uw2) **2 +
                (s_x1 * ux1) **2 + (s_x2 * ux2) **2 +
                (s_wd * uwd) **2 + (s_xd * uxd) **2 )
    data = pd.DataFrame({'value': [v], 'uncertainty': [u], 'uncertainty [%]': u / v * 100}, index=['value'])

    synthetic_one_g_xs_data = pd.DataFrame({'U236': [x1, ux1],
                                            'U234': [x2, ux2],
                                            'U238': [0.9, 0.003],
                                            'U235': [xd, uxd]}
                                            ).T.reset_index()
    synthetic_one_g_xs_data.columns = ['nuclide', 'value', 'uncertainty']
    synthetic_one_g_xs_data = synthetic_one_g_xs_data.set_index('nuclide')

    composition = pd.DataFrame({'value': [w1, w2, wd],
                                'uncertainty': [uw1, uw2, uwd]},
                                index=['U236', 'U234', 'U238'])

    nerea_ = impurity_correction(synthetic_one_g_xs_data, composition, drop_main=True, xs_den='U235')
    np.testing.assert_equal(data.index.values, nerea_.index.values)
    np.testing.assert_equal(data.columns.values, nerea_.columns.values)
    np.testing.assert_almost_equal(data['value'].values, nerea_['value'].values, decimal=5)
    np.testing.assert_almost_equal(data['uncertainty'].values, nerea_['uncertainty'].values, decimal=6)

    ## NO XS DEN
    w1, uw1, w2, uw2, wd, uwd = .1, .01, .2, .02, .7, .07
    x1, ux1, x2, ux2, xd, uxd = .07 / 236., .001 / 236., .08 / 234.040916, .002 / 234.040916, 1, 0
    v = (w1/wd * x1/xd) + (w2/wd * x2/xd)

    s_w1, s_w2 = 1 / wd * x1 / xd, 1 / wd * x2 / xd
    s_x1, s_x2 = w1 / wd * 1 / xd, w2 / wd * 1 / xd
    s_wd = 1 / wd **2 * (w1 * x1 / xd + w2 * x2 / xd)
    s_xd = 1 / xd **2 * (w1 / wd * x1 + w2 / wd * x2)
    u = np.sqrt((s_w1 * uw1) **2 + (s_w2 * uw2) **2 +
                (s_x1 * ux1) **2 + (s_x2 * ux2) **2 +
                (s_wd * uwd) **2 + (s_xd * uxd) **2 )
    data = pd.DataFrame({'value': [v], 'uncertainty': [u], 'uncertainty [%]': u / v * 100}, index=['value'])

    synthetic_one_g_xs_data = pd.DataFrame({'U236': [x1, ux1],
                                            'U234': [x2, ux2],
                                            'U238': [0.9, 0.003],
                                            'U235': [xd, uxd]}
                                            ).T.reset_index()
    synthetic_one_g_xs_data.columns = ['nuclide', 'value', 'uncertainty']
    synthetic_one_g_xs_data = synthetic_one_g_xs_data.set_index('nuclide')

    composition = pd.DataFrame({'value': [w1, w2, wd],
                                'uncertainty': [uw1, uw2, uwd]},
                                index=['U236', 'U234', 'U238'])

    nerea_ = impurity_correction(synthetic_one_g_xs_data, composition, drop_main=True)
    np.testing.assert_equal(data.index.values, nerea_.index.values)
    np.testing.assert_equal(data.columns.values, nerea_.columns.values)
    np.testing.assert_almost_equal(data['value'].values, nerea_['value'].values, decimal=5)
    np.testing.assert_almost_equal(data['uncertainty'].values, nerea_['uncertainty'].values, decimal=6)

    ## NO DROP_MAIN
    w1, uw1, w2, uw2, wd, uwd = .1, .01, .2, .02, .7, .07
    x1, ux1, x2, ux2, xd_, uxd_ = .07 / 236., .001 / 236., .08 / 234.040916, .002 / 234.040916, .6 / 235.043923, .004 / 235.043923
    xd, uxd = 0.9 / 238.050783, 0.003 / 238.050783
    v = (w1/wd * x1/xd_) + (w2/wd * x2/xd_) + (wd/wd * xd/xd_)

    s_w1, s_w2 = 1 / wd * x1 / xd_, 1 / wd * x2 / xd_
    s_x1, s_x2 = w1 / wd * 1 / xd_, w2 / wd * 1 / xd_
    s_xd = 1 / xd_
    s_wd = 1 / wd **2 * (w1 * x1 / xd_ + w2 * x2 / xd_)
    s_xd_ = 1 / xd_ **2 * (w1 / wd * x1 + w2 / wd * x2 + xd)
    u = np.sqrt((s_w1 * uw1) **2 + (s_w2 * uw2) **2 +
                (s_x1 * ux1) **2 + (s_x2 * ux2) **2 + (s_xd * uxd) **2 +
                (s_wd * uwd) **2 + (s_xd_ * uxd_) **2 )
    data = pd.DataFrame({'value': [v], 'uncertainty': [u], 'uncertainty [%]': u / v * 100}, index=['value'])

    synthetic_one_g_xs_data = pd.DataFrame({'U236': [x1, ux1],
                                            'U234': [x2, ux2],
                                            'U238': [xd, uxd],
                                            'U235': [xd_, uxd_]}
                                            ).T.reset_index()
    synthetic_one_g_xs_data.columns = ['nuclide', 'value', 'uncertainty']
    synthetic_one_g_xs_data = synthetic_one_g_xs_data.set_index('nuclide')

    composition = pd.DataFrame({'value': [w1, w2, wd],
                                'uncertainty': [uw1, uw2, uwd]},
                                index=['U236', 'U234', 'U238'])

    nerea_ = impurity_correction(synthetic_one_g_xs_data, composition, drop_main=False, xs_den='U235')
    np.testing.assert_equal(data.index.values, nerea_.index.values)
    np.testing.assert_equal(data.columns.values, nerea_.columns.values)
    np.testing.assert_almost_equal(data['value'].values, nerea_['value'].values, decimal=5)
    np.testing.assert_almost_equal(data['uncertainty'].values, nerea_['uncertainty'].values, decimal=6)

    ## MONO-ISOTOPIC deposit
    # only uncertainty = 0 makes sense
    w, uw = 1, 0
    x, ux = .6 / 235.043923, .004 / 235.043923
    v, u = x, ux

    data = pd.DataFrame({'value': [v], 'uncertainty': [u], 'uncertainty [%]': u / v * 100}, index=['value'])

    synthetic_one_g_xs_data = pd.DataFrame({'U236': [x1, ux1],
                                            'U234': [x2, ux2],
                                            'U238': [xd, uxd],
                                            'U235': [x, ux]}
                                            ).T.reset_index()
    synthetic_one_g_xs_data.columns = ['nuclide', 'value', 'uncertainty']
    synthetic_one_g_xs_data = synthetic_one_g_xs_data.set_index('nuclide')

    composition = pd.DataFrame({'value': [w],
                                'uncertainty': [uw]},
                                index=['U235'])

    nerea_ = impurity_correction(synthetic_one_g_xs_data, composition, drop_main=False, xs_den='')
    np.testing.assert_equal(data.index.values, nerea_.index.values)
    np.testing.assert_equal(data.columns.values, nerea_.columns.values)
    np.testing.assert_almost_equal(data['value'].values, nerea_['value'].values, decimal=5)
    np.testing.assert_almost_equal(data['uncertainty'].values, nerea_['uncertainty'].values, decimal=6)

    # and the absolute value of w does not matter
    w, uw = 10, 0
    x, ux = .6 / 235.043923, .004 / 235.043923
    v, u = x, ux

    data = pd.DataFrame({'value': [v], 'uncertainty': [u], 'uncertainty [%]': u / v * 100}, index=['value'])

    synthetic_one_g_xs_data = pd.DataFrame({'U236': [x1, ux1],
                                            'U234': [x2, ux2],
                                            'U238': [xd, uxd],
                                            'U235': [x, ux]}
                                            ).T.reset_index()
    synthetic_one_g_xs_data.columns = ['nuclide', 'value', 'uncertainty']
    synthetic_one_g_xs_data = synthetic_one_g_xs_data.set_index('nuclide')

    composition = pd.DataFrame({'value': [w],
                                'uncertainty': [uw]},
                                index=['U235'])

    nerea_ = impurity_correction(synthetic_one_g_xs_data, composition, drop_main=False, xs_den='')
    np.testing.assert_equal(data.index.values, nerea_.index.values)
    np.testing.assert_equal(data.columns.values, nerea_.columns.values)
    np.testing.assert_almost_equal(data['value'].values, nerea_['value'].values, decimal=5)
    np.testing.assert_almost_equal(data['uncertainty'].values, nerea_['uncertainty'].values, decimal=6)

    # Now test dropping main doees not break NEREA
    x1, ux1, x2, ux2, xd, uxd = .07 / 236., .001 / 236., .08 / 234.040916, .002 / 234.040916, .6 / 235.043923, .004 / 235.043923

    w, uw = 1, 0
    x, ux = .6 / 235.043923, .004 / 235.043923
    v, u = x, ux

    data = pd.DataFrame({'value': [0], 'uncertainty': [0], 'uncertainty [%]': np.nan}, index=['value'])

    synthetic_one_g_xs_data = pd.DataFrame({'U236': [x1, ux1],
                                            'U234': [x2, ux2],
                                            'U238': [xd, uxd],
                                            'U235': [x, ux]}
                                            ).T.reset_index()
    synthetic_one_g_xs_data.columns = ['nuclide', 'value', 'uncertainty']
    synthetic_one_g_xs_data = synthetic_one_g_xs_data.set_index('nuclide')

    composition = pd.DataFrame({'value': [w],
                                'uncertainty': [uw]},
                                index=['U235'])

    nerea_ = impurity_correction(synthetic_one_g_xs_data, composition, drop_main=True, xs_den='')
    np.testing.assert_equal(data.index.values, nerea_.index.values)
    np.testing.assert_equal(data.columns.values, nerea_.columns.values)
    np.testing.assert_almost_equal(data['value'].values, nerea_['value'].values, decimal=5)
    np.testing.assert_almost_equal(data['uncertainty'].values, nerea_['uncertainty'].values, decimal=6)
