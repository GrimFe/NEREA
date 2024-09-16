import pytest
import pandas as pd
from nerea.calculated import CalculatedSpectralIndex

@pytest.fixture
def sample_data():
    return pd.DataFrame({'value': 1.01, 'uncertainty': .5}, index=['value'])

@pytest.fixture
def sample_c(sample_data):
    return CalculatedSpectralIndex(sample_data, 'M', ['D1', 'D2'])

def test_initialization(sample_data, sample_c):
    pd.testing.assert_frame_equal(sample_data, sample_c.data)
    assert sample_c.model_id == 'M'
    assert sample_c.deposit_ids == ['D1', 'D2']

def test_calculate(sample_data, sample_c):
    pd.testing.assert_frame_equal(sample_data, sample_c.calculate())
