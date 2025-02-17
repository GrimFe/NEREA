import pytest
import datetime
import pandas as pd
from nerea.fission_fragment_spectrum import FissionFragmentSpectrum
from nerea.reaction_rate import ReactionRate
from nerea.constants import KNBS, XS_FAST

@pytest.fixture
def sample_spectrum_data():
    # Sample data for testing
    data = {
        "channel": [  1,   2,   3,   4,   5,   6,   7,   8,   9,  10, 11,  12, 13, 14, 15, 16, 17, 18, 19, 20],
        "counts":  [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 50, 25, 10,  5,  3,  1,  0,  0,  0]
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

def test_get_max(sample_spectrum):
    max_data = sample_spectrum.get_max()
    assert max_data["channel"][0] == 11
    assert max_data["counts"][0] == 600

def test_get_R(sample_spectrum):
    expected_df = pd.Series([50, 12], index=['counts', 'channel'], name=11)
    pd.testing.assert_series_equal(expected_df, sample_spectrum.get_R())

def test_rebin(sample_spectrum):
    expected_df = pd.DataFrame({ "counts": [325.0, 2551.9], "channel": [1, 2]},
                                index=pd.RangeIndex(0, 2))
    pd.testing.assert_frame_equal(expected_df, sample_spectrum.rebin(2))

def test_integrate(sample_spectrum):
    expected_df = pd.DataFrame({'channel':  [1., 2., 2., 3., 3., 4., 4., 5., 5., 6.],
                                'value': [2876.9, 2876.9, 2876.9, 2876.9, 2876.9,
                                            2876.9, 2876.9, 2876.9, 2876.9, 2551.9],
                                'uncertainty': [53.63674114, 53.63674114,
                                                53.63674114, 53.63674114,
                                                53.63674114, 53.63674114,
                                                53.63674114, 53.63674114,
                                                53.63674114, 50.51633399],
                                'uncertainty [%]': [1.86439366, 1.86439366,
                                                    1.86439366, 1.86439366,
                                                    1.86439366, 1.86439366,
                                                    1.86439366, 1.86439366,
                                                    1.86439366, 1.97955774]})
    pd.testing.assert_frame_equal(expected_df, sample_spectrum.integrate(bins=10), check_exact=False, atol=0.00001)

def test_calibrate(sample_spectrum):
    avg, duration = 27000, 100
    sample_monitor = ReactionRate(data = pd.DataFrame({"Time": [sample_spectrum.start_time + datetime.timedelta(seconds=i) for i in range(duration)],
                                                  "value": [avg] * duration}),
                             start_time=sample_spectrum.start_time,
                             campaign_id='CALIBRATION',
                             experiment_id="CALIBRATION 1",
                             detector_id='test',
                             deposit_id="U235",
                             timebase= 1)
    expected_df = pd.DataFrame({'channel': [1., 2., 3., 3., 4., 4., 5., 6., 6., 7.],
                                'value': [320.28691336, 320.28691336, 320.28691336, 320.28691336, 320.28691336, 320.28691336, 320.28691336, 320.28691336,
                                          320.28691336, 284.1044785],
                                'uncertainty': [33.21006196, 33.21006196, 33.21006196, 33.21006196, 33.21006196, 33.21006196, 33.21006196, 33.21006196,
                                                33.21006196, 29.51894558],
                                'uncertainty [%]': [10.36884761, 10.36884761, 10.36884761, 10.36884761, 10.36884761, 10.36884761, 10.36884761, 10.36884761,
                                                    10.36884761, 10.39017256]},
                                index=['value'] * 10)

    # fractional composition
    composition = {'U235': [1, 0.1]}
    c = sample_spectrum.calibrate(k=KNBS["BR1-MARK3"],
                                  composition=composition,
                                  monitor=sample_monitor,
                                  one_group_xs=XS_FAST,
                                  bins=None)
    pd.testing.assert_frame_equal(expected_df, c.data)

    # per cent composition
    composition = {'U235': [1, 0.1]}
    c = sample_spectrum.calibrate(k=KNBS["BR1-MARK3"],
                                  composition=composition,
                                  monitor=sample_monitor,
                                  one_group_xs=XS_FAST,
                                  bins=None)
    pd.testing.assert_frame_equal(expected_df, c.data)

    # pd.DataFrame composition
    composition = pd.DataFrame({'U235': [1, 0.1]}, index=['value', 'uncertainty']).T
    c = sample_spectrum.calibrate(k=KNBS["BR1-MARK3"],
                                  composition=composition,
                                  monitor=sample_monitor,
                                  one_group_xs=XS_FAST,
                                  bins=None)
    pd.testing.assert_frame_equal(expected_df, c.data)

    # test other effective mass attributes
    assert sample_spectrum.deposit_id == c.deposit_id
    assert sample_spectrum.detector_id == c.detector_id
    assert sample_spectrum.data.channel.max() == c.bins
    pd.testing.assert_frame_equal(composition, c.composition)

    # test effective mass bins
    composition = pd.DataFrame({'U235': [1, 0.1]}, index=['value', 'uncertainty']).T
    c = sample_spectrum.calibrate(k=KNBS["BR1-MARK3"],
                                  composition=composition,
                                  monitor=sample_monitor,
                                  one_group_xs=XS_FAST,
                                  bins=10)
    assert 10 == c.bins
