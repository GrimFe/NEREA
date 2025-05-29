import pytest
import datetime
import pandas as pd
import numpy as np
from nerea.fission_fragment_spectrum import FissionFragmentSpectrum
from nerea.reaction_rate import ReactionRate
from nerea.constants import KNBS

XS_FAST = pd.DataFrame({"value": np.array([72.88, 1133.12, 1489.03, 572.23, 284.95, 1264.48,
                                   1971.88, 2132.61, 1308.2, np.nan, 1115.87, 1321.81,
                                   1024.24]) * 1e-27,   ## fast xs JEFF-3.1.1 [mb] then converted to cm^2
                         "uncertainty": [0., 0., 0., 0., 0., 0.,
                                   0., 0., 0., 0., 0., 0.,
                                   0.]   ## to be computed [b]
                         }, index=["Th232", "U234", "U235", "U236", "U238", "Np237",
                                   "Pu238", "Pu239", "Pu240", "Pu241", "Pu242", "Am241",
                                   "Am243"])

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
    EXPECTED_MEAN_0 = 0

    # test default (Moving Average window = 10)
    smoothed_data = sample_spectrum.smooth().data
    EXPECTED_MEAN_1 = 325.0
    EXPECTED_MEAN_2 = 375.0
    assert all(smoothed_data.query("channel < 10").counts == [EXPECTED_MEAN_0] * 9)
    assert smoothed_data.query("channel == 10").counts.values[0] == pytest.approx(EXPECTED_MEAN_1, rel=1e-9)
    assert smoothed_data.query("channel == 11").counts.values[0] == pytest.approx(EXPECTED_MEAN_2, rel=1e-9)
    assert all(smoothed_data.channel == sample_spectrum.data.channel)

    # test Moving Average window = 5
    smoothed_data = sample_spectrum.smooth(window=5).data
    EXPECTED_MEAN_1 = 200
    EXPECTED_MEAN_2 = 250
    assert all(smoothed_data.query("channel < 5").counts == [EXPECTED_MEAN_0] * 4)
    assert smoothed_data.query("channel == 5").counts.values[0] == pytest.approx(EXPECTED_MEAN_1, rel=1e-9)
    assert smoothed_data.query("channel == 6").counts.values[0] == pytest.approx(EXPECTED_MEAN_2, rel=1e-9)
    assert all(smoothed_data.channel == sample_spectrum.data.channel)

    # # test Savgol Filter
    # smoothed_data = sample_spectrum.smooth(method="savgol_filter", window_length=5, polyorder=3).data
    # target = [100, 150, 200, 250, 300, 350, 400, 450, 500, 601.428571, 452.142857, 190.571429,
    #           -19.1428571, 10.6000000, 4.6571428, 2.65714286, 1.08571429, 0.0857142857,
    #           -0.0571428571, 0.0142857143]
    # for i, row in smoothed_data.iterrows():
    #     np.testing.assert_almost_equal(row.counts, target[i], decimal=5)
    # assert all(smoothed_data.channel == sample_spectrum.data.channel)

def test_get_max(sample_spectrum):
    max_data = sample_spectrum.get_max(smooth=False)
    assert max_data["channel"][0] == 11
    assert max_data["counts"][0] == 600

def test_get_R(sample_spectrum):
    expected_df = pd.DataFrame({"channel": 12, "counts": 50}, index=[11])
    pd.testing.assert_frame_equal(expected_df, sample_spectrum.get_R(smooth=False))

def test_rebin(sample_spectrum):
    expected_df = pd.DataFrame({"channel": [1, 2], "counts": [325.0, 2551.9]},
                                index=pd.RangeIndex(0, 2))
    pd.testing.assert_frame_equal(expected_df, sample_spectrum.rebin(2))

    # test it gives the same data if rebinned to the same amount of bins
    pd.testing.assert_frame_equal(sample_spectrum.data,
                                  sample_spectrum.rebin(20, smooth=False))
    
    # test it gives smoothened data if rebinned to the same amount of bins, buth smooth is True
    pd.testing.assert_frame_equal(sample_spectrum.smooth().data,
                                  sample_spectrum.rebin(20, smooth=True))
    
    # test it also gives the same when smoothing and bins=None
    pd.testing.assert_frame_equal(sample_spectrum.smooth().data,
                                  sample_spectrum.rebin(bins=None, smooth=True))

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
                                                    1.86439366, 1.97955774],
                                'R': [.15, .2, .25, .3, .35, .4, .45, .5, .55, .6]})
    pd.testing.assert_frame_equal(expected_df, sample_spectrum.integrate(bins=10), check_exact=False, atol=0.00001)
    # testing integer llds
    expected_df = pd.DataFrame({'channel':  [1., 2.],
                                'value': [2876.9, 2876.9],
                                'uncertainty': [53.63674114, 53.63674114],
                                'uncertainty [%]': [1.86439366, 1.86439366],
                                'R': [np.nan, np.nan]})
    pd.testing.assert_frame_equal(expected_df, sample_spectrum.integrate(bins=10, llds=[1., 2.], r=False), check_exact=False, atol=0.00001)

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
                                'value': [439.08776331513275, 427.9547064359458, 411.25512111716546, 411.25512111716546, 388.9890073587916,
                                          388.9890073587916, 361.15636516082424, 327.7571945232634, 327.7571945232634, 288.79149544610914],
                                'uncertainty': [45.328774766486085, 44.19313472096307, 42.48966619235774, 42.48966619235774, 40.21835702157304,
                                                40.21835702157304, 37.37918673739926, 33.972120356027794, 33.972120356027794, 29.99709489018778],
                                'uncertainty [%]': [10.323397405623822, 10.326591589331588, 10.331705068361337, 10.331705068361337, 10.339201432619625,
                                                    10.339201432619625, 10.349862370764024, 10.365026587880601, 10.365026587880601, 10.3871115885355],
                                "R": [.15, .2, .25, .3, .35, .4, .45, .5, .55, .6]},
                                index=['value'] * 10)

    # fractional composition
    composition = {'U235': [1, 0.1]}
    c = sample_spectrum.calibrate(k=KNBS["BR1-MARK3"],
                                  composition=composition,
                                  monitor=sample_monitor,
                                  one_group_xs=XS_FAST,
                                  bins=None,
                                  smooth=False)
    pd.testing.assert_frame_equal(expected_df, c.data)

    # per cent composition
    composition = {'U235': [1, 0.1]}
    c = sample_spectrum.calibrate(k=KNBS["BR1-MARK3"],
                                  composition=composition,
                                  monitor=sample_monitor,
                                  one_group_xs=XS_FAST,
                                  bins=None,
                                  smooth=False)
    pd.testing.assert_frame_equal(expected_df, c.data)

    # pd.DataFrame composition
    composition = pd.DataFrame({'U235': [1, 0.1]}, index=['value', 'uncertainty']).T
    c = sample_spectrum.calibrate(k=KNBS["BR1-MARK3"],
                                  composition=composition,
                                  monitor=sample_monitor,
                                  one_group_xs=XS_FAST,
                                  bins=None,
                                  smooth=False)
    pd.testing.assert_frame_equal(expected_df, c.data)

    # test other effective mass attributes
    assert sample_spectrum.deposit_id == c.deposit_id
    assert sample_spectrum.detector_id == c.detector_id
    assert sample_spectrum.data.channel.max() == c.bins
    pd.testing.assert_frame_equal(composition.reset_index(names='nuclide'), c.composition)

    # test effective mass bins
    composition = pd.DataFrame({'U235': [1, 0.1]}, index=['value', 'uncertainty']).T
    c = sample_spectrum.calibrate(k=KNBS["BR1-MARK3"],
                                  composition=composition,
                                  monitor=sample_monitor,
                                  one_group_xs=XS_FAST,
                                  bins=10,
                                  smooth=False)
    assert 10 == c.bins
