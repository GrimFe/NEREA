import pytest
import datetime
import pandas as pd
import numpy as np
from nerea.pulse_height_spectrum import PulseHeightSpectrum
from nerea.count_rate import CountRate
from nerea.constants import KNBS
from nerea.classes import Xs

XS_FAST = Xs(pd.DataFrame({"value": np.array([72.88, 1133.12, 1489.03, 572.23, 284.95, 1264.48,
                                   1971.88, 2132.61, 1308.2, np.nan, 1115.87, 1321.81,
                                   1024.24]) * 1e-27,   ## fast xs JEFF-3.1.1 [mb] then converted to cm^2
                         "uncertainty": np.array([0., 0., 1., 0., 2., 0.,
                                   0., 0., 0., 0., 0., 0.,
                                   0.]) * 1e-27  # just for testing
                         }, index=["Th232", "U234", "U235", "U236", "U238", "Np237",
                                   "Pu238", "Pu239", "Pu240", "Pu241", "Pu242", "Am241",
                                   "Am243"]))

@pytest.fixture
def sample_spectrum_data():
    # Sample data for testing
    data = {
        "channel": [  1,   2,   3,   4,   5,   6,   7,   8,   9,  10, 11,  12, 13, 14, 15, 16, 17, 18, 19, 20],
        "value":  [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 50, 25, 10,  5,  3,  1,  0,  0,  0]
    }
    return data

@pytest.fixture
def sample_spectrum(sample_spectrum_data):
    return PulseHeightSpectrum(start_time=datetime.datetime(2024, 5, 18, 20, 30, 15),
                                   live_time=10,
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
    assert all(smoothed_data.query("channel < 10").value == [EXPECTED_MEAN_0] * 9)
    assert smoothed_data.query("channel == 10").value.values[0] == pytest.approx(EXPECTED_MEAN_1, rel=1e-9)
    assert smoothed_data.query("channel == 11").value.values[0] == pytest.approx(EXPECTED_MEAN_2, rel=1e-9)
    assert all(smoothed_data.channel == sample_spectrum.data.channel)

    # test Moving Average window = 5
    smoothed_data = sample_spectrum.smooth(window=5).data
    EXPECTED_MEAN_1 = 200
    EXPECTED_MEAN_2 = 250
    assert all(smoothed_data.query("channel < 5").value == [EXPECTED_MEAN_0] * 4)
    assert smoothed_data.query("channel == 5").value.values[0] == pytest.approx(EXPECTED_MEAN_1, rel=1e-9)
    assert smoothed_data.query("channel == 6").value.values[0] == pytest.approx(EXPECTED_MEAN_2, rel=1e-9)
    assert all(smoothed_data.channel == sample_spectrum.data.channel)

def test_get_max(sample_spectrum):
    max_data = sample_spectrum.get_max(smooth=False)
    assert max_data["channel"][0] == 11
    assert max_data["value"][0] == 600

def test_get_R(sample_spectrum):
    expected_df = pd.DataFrame({"channel": 12, "value": 50}, index=[11])
    pd.testing.assert_frame_equal(expected_df, sample_spectrum.get_R(smooth=False))

def test_rebin(sample_spectrum):
    expected_df = pd.DataFrame({"channel": [1, 2], "value": [325.0, 2551.9]},
                                index=pd.RangeIndex(0, 2))
    pd.testing.assert_frame_equal(expected_df, sample_spectrum.rebin(2).data)

    # test it gives the same data if rebinned to the same amount of bins
    pd.testing.assert_frame_equal(sample_spectrum.data,
                                  sample_spectrum.rebin(20, smooth=False).data)
    
    # test it gives smoothened data if rebinned to the same amount of bins, buth smooth is True
    pd.testing.assert_frame_equal(sample_spectrum.smooth().data,
                                  sample_spectrum.rebin(20, smooth=True).data)
    
    # test it also gives the same when smoothing and bins=None
    pd.testing.assert_frame_equal(sample_spectrum.smooth().data,
                                  sample_spectrum.rebin(bins=None, smooth=True).data)

def test_integrate(sample_spectrum):
    expected_df = pd.DataFrame({'channel':  np.int32([1, 2, 2, 3, 3, 4, 4, 5, 5, 6]),
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
    pd.testing.assert_frame_equal(expected_df,
                                  sample_spectrum.integrate(bins=10,
                                                            raw_integral=False,
                                                            renormalize=False),
                                  check_exact=False, atol=0.00001)
    # testing integer llds
    expected_df = pd.DataFrame({'channel':  np.int32([1, 2]),
                                'value': [2876.9, 2876.9],
                                'uncertainty': [53.63674114, 53.63674114],
                                'uncertainty [%]': [1.86439366, 1.86439366],
                                'R': [np.nan, np.nan]})
    pd.testing.assert_frame_equal(expected_df,
                                  sample_spectrum.integrate(bins=10,
                                                            llds=[1., 2.],
                                                            r=False,
                                                            raw_integral=False,
                                                            renormalize=False),
                                  check_exact=False, atol=0.00001)
    # testing integration of raw data
    expected_df = pd.DataFrame({'channel':  np.int32([1, 2]),
                                'value': [3944., 3844.],
                                'uncertainty': [62.80127387243033, 62.],
                                'uncertainty [%]': [1.592324388246205, 1.6129032258064515],
                                'R': [np.nan, np.nan]})
    pd.testing.assert_frame_equal(expected_df,
                                  sample_spectrum.integrate(bins=None,
                                                            llds=[1., 2.],
                                                            r=False,
                                                            raw_integral=True,
                                                            renormalize=False),
                                  check_exact=False, atol=0.00001)
    # testing single-lld integration
    expected_df = pd.DataFrame({'channel':  np.int32([2]),
                                'value': [3844.],
                                'uncertainty': [62.],
                                'uncertainty [%]': [1.6129032258064515],
                                'R': [np.nan]})
    pd.testing.assert_frame_equal(expected_df,
                                  sample_spectrum.integrate(bins=20,
                                                            llds=[2.],
                                                            r=False,
                                                            raw_integral=True,
                                                            renormalize=False),
                                  check_exact=False, atol=0.00001)
    expected_df = pd.DataFrame({'channel':  np.int32([2]),
                                'value': [3694.],
                                'uncertainty': [60.778285596090974],
                                'uncertainty [%]': [1.6453244611827553],
                                'R': [np.nan]})
    pd.testing.assert_frame_equal(expected_df,
                                  sample_spectrum.integrate(bins=10,
                                                            llds=[2.],
                                                            r=False,
                                                            raw_integral=True,
                                                            renormalize=False),
                                  check_exact=False, atol=0.00001)
    expected_df = pd.DataFrame({'channel':  np.int32([2]),
                                'value': [3844.],
                                'uncertainty': [62.],
                                'uncertainty [%]': [1.6129032258064515],
                                'R': [np.nan]})
    pd.testing.assert_frame_equal(expected_df,
                                  sample_spectrum.integrate(bins=20,
                                                            llds=2.,
                                                            r=False,
                                                            raw_integral=True,
                                                            renormalize=False),
                                  check_exact=False, atol=0.00001)
    expected_df = pd.DataFrame({'channel':  np.int32([2]),
                                'value': [3844.],
                                'uncertainty': [62.],
                                'uncertainty [%]': [1.6129032258064515],
                                'R': [np.nan]})
    pd.testing.assert_frame_equal(expected_df,
                                  sample_spectrum.integrate(bins=20,
                                                            llds=2,
                                                            r=False,
                                                            raw_integral=True,
                                                            renormalize=False),
                                  check_exact=False, atol=0.00001)

def test_get_calibration_coefficient(sample_spectrum):
    # The only uncertainty that makes sense for a mono-isotopic
    # deposit is 0
    composition_ = pd.DataFrame({'U235': [1, 0]}, index=['value', 'uncertainty']).T
    expected_df = pd.DataFrame({'value': 6.02214076e23 / 235.043923 / 1e6 * 1489.03 * 1e-27,
                            "uncertainty": 6.02214076e23 / 235.043923 / 1e6 * 1 * 1e-27,
                            "uncertainty [%]": 1 / 1489.03 * 100}, index=['value'])
    pd.testing.assert_frame_equal(expected_df,
                sample_spectrum._get_calibration_coefficient(XS_FAST, composition_))
    # now with two nuclides
    composition_ = pd.DataFrame({'U235': [.9, .09], 'U238': [.1, .01]}, index=['value', 'uncertainty']).T
    x8, ux8, x5, ux5 = 284.95 * 1e-27, 2 * 1e-27, 1489.03 * 1e-27, 1 * 1e-27
    v = 6.02214076e23 / 235.043923 / 1e6 * (.1 / .9 * x8 + x5)
    u = 6.02214076e23 / 235.043923 / 1e6 * np.sqrt((1 / .9 * x8 * .01) **2 \
                                                + (.1 / .9**2 * x8 * .09) **2 \
                                                + (.1 / .9 * ux8) **2 \
                                                + (ux5) **2)
    expected_df = pd.DataFrame({'value': v,
                                "uncertainty": u,
                                "uncertainty [%]": u / v * 100}, index=['value'])
    pd.testing.assert_frame_equal(expected_df,
                sample_spectrum._get_calibration_coefficient(XS_FAST, composition_))

def test_calibrate(sample_spectrum):
    avg, duration = 27000, 100
    sample_monitor = CountRate(data = pd.DataFrame({"Time": [sample_spectrum.start_time + datetime.timedelta(seconds=i) for i in range(duration)],
                                                  "value": [avg] * duration}),
                             start_time=sample_spectrum.start_time,
                             campaign_id='CALIBRATION',
                             experiment_id="CALIBRATION 1",
                             detector_id='test',
                             deposit_id="U235",
                             timebase= 1)
    k, uk = 8720., 0.02 * 8720.
    m, um = 27000, 51.96152422706632
    c, uc = 3.815094702900616e-09, 2.562134210123782e-12
    kmc, ukmc = k * m * c, np.sqrt((uk * m * c)**2 + (k * um * c)**2 + (k * m * uc)**2)

    phs_t = np.array([394.4, 384.4, 369.4, 369.4, 349.4, 349.4, 324.4, 294.4, 294.4,
        259.4])
    uphs_t = np.array([6.280127387, 6.2        , 6.07782856 , 6.07782856 , 5.911006682,
        5.911006682, 5.695612346, 5.425863987, 5.425863987, 5.093132631])
    v = phs_t / kmc
    u = np.sqrt((1 / kmc * uphs_t)**2 + (phs_t / kmc **2 * ukmc)**2)
    expected_df = pd.DataFrame({'channel': [1, 2, 3, 3, 4, 4, 5, 6, 6, 7],
                                'value': v,
                                'uncertainty': u,
                                'uncertainty [%]': u / v * 100,
                                "R": [.15, .2, .25, .3, .35, .4, .45, .5, .55, .6]})
    expected_df['channel'] = expected_df['channel'].astype('int32')

    # fractional composition
    composition = {'U235': [1, 0]}
    c = sample_spectrum.calibrate(k=KNBS["BR1-MARK3"],
                                  composition=composition,
                                  monitor=sample_monitor,
                                  one_group_xs=XS_FAST,
                                  bins=None,
                                  smooth=False)
    pd.testing.assert_frame_equal(expected_df, c.data, check_exact=False, atol=1e-8)

    # pd.DataFrame composition
    composition = pd.DataFrame({'U235': [1, 0]}, index=['value', 'uncertainty']).T
    c = sample_spectrum.calibrate(k=KNBS["BR1-MARK3"],
                                  composition=composition,
                                  monitor=sample_monitor,
                                  one_group_xs=XS_FAST,
                                  bins=None,
                                  smooth=False)
    pd.testing.assert_frame_equal(expected_df, c.data, check_exact=False, atol=1e-8)

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
