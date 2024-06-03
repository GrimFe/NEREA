import pytest
import pandas as pd
from REPTILE.Calculated import CalculatedTraverse

@pytest.fixture
def sample_data():
    return pd.DataFrame({'value': [1, 2], 'uncertainty': [.5, .5],
                         'uncertainty [%]': [50, 25], 'traverse': ['A', 'B']}, index=['value','value'])

@pytest.fixture
def sample_c(sample_data):
    return CalculatedTraverse(sample_data, 'M', 'nuclide')

def test_initialization(sample_data, sample_c):
    pd.testing.assert_frame_equal(sample_data, sample_c.data)
    assert sample_c.model_id == 'M'
    assert sample_c.deposit_id == 'nuclide'

def test_calculate(sample_c):
    target_df = pd.DataFrame({'value': [0.5,  1],
                              'uncertainty': [0.279508, 0.353553],
                              'uncertainty [%]': [55.901699, 35.355339],
                              'traverse': ['A', 'B']})
    pd.testing.assert_frame_equal(target_df, sample_c.calculate())
