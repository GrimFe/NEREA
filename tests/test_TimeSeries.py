import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from nerea.time_series import TimeSeries
from nerea.utils import _make_df

@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'Time': [datetime(2024, 5, 19, 20, 5, 0) + timedelta(seconds=i) for i in range(7)],
        'value': [0, 1, 2, 3, 4, 5, 6]
    })
    return data

@pytest.fixture
def ts(sample_data):
    return TimeSeries(sample_data, timebase=1.)

def test_smooth_data(ts):
    # the logic of this is tested in test_functions
    # here we test local behaviour
    smoothed = ts.smooth_data(**{'smoothing_method': 'moving_average', 'window': 2, 'renormalize': False})
    assert isinstance(smoothed, TimeSeries)
    pd.testing.assert_series_equal(smoothed.data.Time, ts.data.Time)
    pd.testing.assert_series_equal(smoothed.data.value, pd.Series([0, .5, 1.5, 2.5, 3.5, 4.5, 5.5], name='value'))

def test_cut_data(ts):
    st = datetime(2024, 5, 19, 20, 5, 0) + timedelta(seconds=1)
    et = datetime(2024, 5, 19, 20, 5, 0) + timedelta(seconds=3)
    cut = ts.cut_data(st, et)
    assert isinstance(cut, TimeSeries)
    pd.testing.assert_frame_equal(cut.data, ts.data.query("Time >= @st and Time < @et"))

def test_average(ts):
    st = datetime(2024, 5, 19, 20, 5, 0)
    result = ts.average(st, 3)
    v, u = 1., np.sqrt(3) / 3
    test = pd.DataFrame({'value': v,
                         'uncertainty': u,
                         'uncertainty [%]': u * 100}, index=['value'])
    pd.testing.assert_frame_equal(result, test)

    result = ts.average(st, 3, uncertainty='std')
    v, u = 1., np.std([0, 1, 2], ddof=1)
    test = pd.DataFrame({'value': v,
                         'uncertainty': u,
                         'uncertainty [%]': u * 100}, index=['value'])
    pd.testing.assert_frame_equal(result, test)

    result = ts.average(st, 3, uncertainty='sem')
    v, u = 1., np.std([0, 1, 2], ddof=1) / np.sqrt(3)
    test = pd.DataFrame({'value': v,
                         'uncertainty': u,
                         'uncertainty [%]': u * 100}, index=['value'])
    pd.testing.assert_frame_equal(result, test)
