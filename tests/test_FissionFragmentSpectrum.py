import pytest
import datetime
import pandas as pd
from PSICHE.FissionFragmentSpectrum import FissionFragmentSpectrum

@pytest.fixture
def sample_spectrum_data():
    # Sample data for testing
    data = {
        "channel": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "counts": [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 50, 25, 10, 5, 3, 1, 0, 0, 0]
    }
    return data

@pytest.fixture
def sample_spectrum(sample_spectrum_data):
    return FissionFragmentSpectrum(start_time=datetime.datetime(2024, 5, 18, 20, 30, 15),
                                   life_time=10,
                                   real_time=10,
                                   data=pd.DataFrame(sample_spectrum_data),
                                   campaign_id="campaign",
                                   experiment_id="experiment",
                                   detector_id="detector",
                                   deposit_id="deposit", 
                                   location_id="location",
                                   measurement_id="measurement")

def test_smoothing(sample_spectrum):
    smoothed_data = sample_spectrum.smooth
    EXPECTED_MEAN_0 = 0
    EXPECTED_MEAN_1 = 325.0
    EXPECTED_MEAN_2 = 375.0
    assert all(smoothed_data.query("channel < 10").counts == [EXPECTED_MEAN_0] * 9)
    assert smoothed_data.query("channel == 10").counts.values[0] == pytest.approx(EXPECTED_MEAN_1, rel=1e-9)
    assert smoothed_data.query("channel == 11").counts.values[0] == pytest.approx(EXPECTED_MEAN_2, rel=1e-9)

def test_max(sample_spectrum):
    max_data = sample_spectrum.max
    assert max_data["channel"][0] == 11
    assert max_data["counts"][0] == 375.0

def test_R(sample_spectrum):
    expected_df = pd.Series([18.0, 174.4], index=['channel', 'counts'], name=17)
    pd.testing.assert_series_equal(expected_df, sample_spectrum.R)

def test_rebin(sample_spectrum):
    expected_df = pd.DataFrame({ "counts": [325.0, 2551.9], "channel": [1, 2]},
                                index=pd.RangeIndex(0, 2))
    pd.testing.assert_frame_equal(expected_df, sample_spectrum.rebin(2))

def test_integrate(sample_spectrum):
    expected_df = pd.DataFrame({'value': [2876.9, 2876.9, 2876.9000000000005, 2876.9000000000005,
                                          2551.9, 1811.8999999999999, 1140.8999999999999, 587.6,
                                          587.6, 193.8],
                                'uncertainty': [0.018643936577179342, 0.018643936577179342,
                                                0.018643936577179342, 0.018643936577179342,
                                                0.019795577409806874, 0.02349269754893474,
                                                0.029605759709491655, 0.04125333907726532,
                                                0.04125333907726532, 0.07183285265343592],
                                'channel':  [2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  9., 10.]})
    pd.testing.assert_frame_equal(expected_df, sample_spectrum.integrate(bins=10))
