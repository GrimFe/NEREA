import pytest
import numpy as np
import pandas as pd
from nerea.utils import *

EXPECTED_RATIO = 2.0
EXPECTED_UNCERTAINTY = 0.282842712474619

def test_integral_v_u():
    s = np.array([1, 2, 3, 4])
    v, u = integral_v_u(s)
    assert v == pytest.approx(10, 1e-9)
    assert u == pytest.approx(0.31622776601683794 * v, rel=1e-9)

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
    pass
