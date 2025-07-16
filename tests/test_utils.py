import pytest
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
                                         'uncertainty [%]': x[1]}, index=[0]) for x in
                                                            [(10, 10), (5, 2), (10, 20)]])
    EXPECTED_VALUE = 500
    EXPECTED_UNCERTAINTY = np.sqrt(2 **2 + 10 **2 + 20 **2) / 100 * EXPECTED_VALUE
    assert product[0] == EXPECTED_VALUE
    assert product[1] == EXPECTED_UNCERTAINTY

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
