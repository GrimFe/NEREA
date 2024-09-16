import pytest
import pandas as pd
from nerea.comparisons import FrameCompare

@pytest.fixture
def frame1():
    return pd.DataFrame({'value': 1.01, 'uncertainty': 0.03,
                         'uncertainty [%]': 2.9702970297029703,
                         'VAR_FRAC_C_n': None, 'VAR_FRAC_C_d': None}, index=['value'])

@pytest.fixture
def frame2():
    return pd.DataFrame({'value': 1, 'uncertainty': 0.02,
                         'uncertainty [%]': 2,
                         'VAR_FRAC_C_n': None, 'VAR_FRAC_C_d': None}, index=['value'])

@pytest.fixture
def sample_fc(frame1, frame2):
    return FrameCompare(frame1, frame2)

def test_frame_compare(sample_fc):
    expected_df = pd.DataFrame({'value': 1.01,
                                'uncertainty': 0.036167,
                                'uncertainty [%]': 3.580875,
                                'VAR_FRAC_C_n_n': None,
                                'VAR_FRAC_C_d_n': None,
                                'VAR_FRAC_C_n': 0.0009,
                                'VAR_FRAC_C_n_d': None,
                                'VAR_FRAC_C_d_d': None,
                                'VAR_FRAC_C_d': 0.00040804}, index=['value'])
    pd.testing.assert_frame_equal(expected_df, sample_fc.compute(), check_exact=False, atol=0.00001)
    