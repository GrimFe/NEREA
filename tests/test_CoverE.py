import pytest
from PSICHE.ReactionRate import ReactionRate, AverageReactionRate
from PSICHE.FissionFragmentSpectrum import FissionFragmentSpectrum, FissionFragmentSpectra
from PSICHE.EffectiveMass import EffectiveMass
from PSICHE.PowerMonitor import PowerMonitor
from PSICHE.SpectralIndex import SpectralIndex
from PSICHE.CoverE import C, CoverE
from datetime import datetime
import pandas as pd

@pytest.fixture
def sample_spectrum_data():
    # Sample data for testing
    data = {
        "channel":[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
        "counts": [0, 0, 0, 0, 1, 3, 1, 4, 1, 5,  1,  3,  4,  2,  4,  1,  3,  5,  80, 65, 35, 5,  20, 25, 35, 55, 58, 60, 62, 70, 65, 50, 45, 40, 37, 34, 25, 20, 13, 5,  1,  0]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_integral_data():
    data = {
        "channel": [ 6.,  8., 10., 12., 14., 16., 18., 20., 22., 24.],
        "value": [60, 80, 88, 87, 87, 88, 86, 85, 82, 78],
        "uncertainty": [.1, .2, .3, .4, .5, .6, .7, .8, .9, .1]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_power_monitor_data():
    data = {
        "Time": [datetime(2024, 5, 18, 20, 30, 15 + i) for i in range(20)],
        "value": [100, 101, 99, 98, 101, 100, 99, 98, 102, 102] * 2
    }
    return pd.DataFrame(data)

@pytest.fixture
def fission_fragment_spectrum_1(sample_spectrum_data):
    return FissionFragmentSpectrum(start_time=datetime(2024, 5, 18, 20, 30, 15),
                                   life_time=10, real_time=10,
                                   data=sample_spectrum_data, campaign_id="A", experiment_id="B",
                                   detector_id="C1", deposit_id="D1", location_id="E", measurement_id="F1")

@pytest.fixture
def fission_fragment_spectrum_2(sample_spectrum_data):
    return FissionFragmentSpectrum(start_time=datetime(2024, 5, 18, 20, 30, 15),
                                   life_time=10, real_time=10,
                                   data=sample_spectrum_data, campaign_id="A", experiment_id="B",
                                   detector_id="C2", deposit_id="D2", location_id="E", measurement_id="F2")

@pytest.fixture
def effective_mass_1(sample_integral_data):
    return EffectiveMass(deposit_id="D1", detector_id="C1", integral=sample_integral_data, bins=42)

@pytest.fixture
def effective_mass_2(sample_integral_data):
    return EffectiveMass(deposit_id="D2", detector_id="C2", integral=sample_integral_data, bins=42)

@pytest.fixture
def power_monitor(sample_power_monitor_data):
        return PowerMonitor(experiment_id="B", data=sample_power_monitor_data, start_time=datetime(2024, 5, 29, 12, 25, 10), campaign_id='C', monitor_id='M')

@pytest.fixture
def rr_1(fission_fragment_spectrum_1, effective_mass_1, power_monitor):
    return ReactionRate(fission_fragment_spectrum_1, effective_mass_1, power_monitor)

@pytest.fixture
def rr_2(fission_fragment_spectrum_2, effective_mass_2, power_monitor):
    return ReactionRate(fission_fragment_spectrum_2, effective_mass_2, power_monitor)

@pytest.fixture
def sample_spectral_index(rr_1, rr_2):
    return SpectralIndex(rr_1, rr_2)

@pytest.fixture
def sample_c_data():
    return pd.DataFrame({'value': 1.01, 'uncertainty': .5}, index=['value'])

@pytest.fixture
def sample_c(sample_c_data):
    return C(sample_c_data, 'M', ['D1', 'D2'])

@pytest.fixture
def sample_ce(sample_c, sample_spectral_index):
    return CoverE(sample_c, sample_spectral_index)

def test_compute(sample_ce):
    expected_df = pd.DataFrame({'value': 1.01, 'uncertainty': 0.50669436}, index=['value'])
    pd.testing.assert_frame_equal(expected_df, sample_ce.compute(), check_exact=False, atol=0.00000001)

def test_minus_one_per_cent(sample_ce):
    expected_df = pd.DataFrame({'value': 1., 'uncertainty': 50.66943608}, index=['value'])
    pd.testing.assert_frame_equal(expected_df, sample_ce.minus_one_percent(), check_exact=False, atol=0.00000001)
