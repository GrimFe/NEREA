import pytest
from PSICHE.ReactionRate import ReactionRate, AverageReactionRate
from PSICHE.FissionFragmentSpectrum import FissionFragmentSpectrum, FissionFragmentSpectra
from PSICHE.EffectiveMass import EffectiveMass
from PSICHE.PowerMonitor import PowerMonitor
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
def fission_fragment_spectrum(sample_spectrum_data):
    return FissionFragmentSpectrum(start_time=datetime(2024, 5, 18, 20, 30, 15),
                                   life_time=10, real_time=10,
                                   data=sample_spectrum_data, campaign_id="A", experiment_id="B",
                                   detector_id="C", deposit_id="D", location_id="E", measurement_id="F")

@pytest.fixture
def effective_mass(sample_integral_data):
    return EffectiveMass(deposit_id="D", detector_id="C", integral=sample_integral_data, bins=42)

@pytest.fixture
def power_monitor(sample_power_monitor_data):
        return PowerMonitor(experiment_id="B", data=sample_power_monitor_data, start_time=datetime(2024, 5, 29, 12, 25, 10), campaign_id='C', monitor_id='M')

@pytest.fixture
def rr(fission_fragment_spectrum, effective_mass, power_monitor):
    return ReactionRate(fission_fragment_spectrum, effective_mass, power_monitor)

def test_reaction_rate_measurement_id(rr):
    assert rr.measurement_id == "F"

def test_reaction_rate_campaign_id(rr):
    assert rr.campaign_id == "A"

def test_reaction_rate_experiment_id(rr):
    assert rr.experiment_id == "B"

def test_reaction_rate_location_id(rr):
    assert rr.location_id == "E"

def test_reaction_rate_deposit_id(rr):
    assert rr.deposit_id == "D"

def test_per_unit_mass(rr):
    expected_series = pd.Series([14., 14., 1.01045977,  0.05807369], index=['channel fission fragment spectrum', 'channel effective mass', 'value', 'uncertainty'], name=4)
    pd.testing.assert_series_equal(expected_series, rr.per_unit_mass(ch_tolerance=.5))

def test_compute(rr):
    expected_df = pd.DataFrame({'value': 0.010104597701149425,'uncertainty': 0.0005807457360837471}, index=['value'])
    pd.testing.assert_frame_equal(expected_df, rr.compute())
