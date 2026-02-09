import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from nerea.count_rate import CountRate
from nerea.classes import EffectiveDelayedParams
from nerea.classes import EffectiveDelayedParams
from nerea.utils import _make_df

@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'Time': [datetime(2024, 5, 19, 20, 5, 0) + timedelta(seconds=i) for i in range(7)],
        'value': [0, 10, 15, 10, 20, 15, 10]
    })
    return data

@pytest.fixture
def power_monitor(sample_data):
    return CountRate(data=sample_data, campaign_id="C1", experiment_id="E1",
                        start_time=datetime(2024, 5, 19, 20, 5, 0), detector_id='M', deposit_id='dep')

@pytest.fixture
def sample_data_02s():
    data = pd.DataFrame({
        'Time': [datetime(2024, 5, 19, 20, 5, 0) + timedelta(seconds=i / 5) for i in range(35)],
        ## data report the count rate so it's just increased multiplicity
        'value': [0] * 5 + [10] * 5 + [15] * 5 + [10] * 5 + [20] * 5 + [15] * 5 + [10] * 5
    })
    return data

@pytest.fixture
def power_monitor_02s(sample_data_02s):
    return CountRate(data=sample_data_02s, campaign_id="C1", experiment_id="E1",
                        start_time=datetime(2024, 5, 19, 20, 5, 0), detector_id='M', deposit_id='dep', timebase=0.2)

@pytest.fixture
def sample_data_uncertain_time_binning():
    data = pd.DataFrame({
        'Time': [datetime(2024, 5, 19, 20, 5, 0) + timedelta(seconds=i) for i in [0, 0.999, 2.001, 3, 4, 5, 5.999]],
        'value': [0, 10, 15, 10, 20, 15, 10]
    })
    return data

@pytest.fixture
def power_monitor_uncertain_time_binning(sample_data_uncertain_time_binning):
    return CountRate(data=sample_data_uncertain_time_binning, campaign_id="C1", experiment_id="E1",
                        start_time=datetime(2024, 5, 19, 20, 5, 0), detector_id='M', deposit_id='dep')

@pytest.fixture
def plateau_data():
    counts = [0,0,0,0,0,.3,.3,.4,.1,.2,.5,0,.0,1,1,1.5,2,2.5,2,3,3.5,4,4.2,3.8,4.2,3.9,3.9,4.2,4.1,4,
          0,0,0,0,0,.3,.3,.4,.1,.2,.5,0,.0,1,1,1.5,2,2.5,2,3,3.5,4,4.2,3.8,4.2,3.9,3.9,4.2,4.1,4,
          0,0,0,0,0,.3,.3,.4,.1,.2,.5,0,.0,1,1,1.5,2,2.5,2,3,3.5,4,4.2,3.8,4.2,3.9,3.9,4.2,4.1,4,
          39700,40000,40100,40200,39800,39700,39800,39900,40200,
          39900,39900,40200,40100,40000,39700,40000,40100,40200,39800,39700,
          39800,39900,40200,39900,39900,40200,40100,
          40000,39700,40000,40100,40200,39800,
          39900,39900,40200,40100,40000,39700,40000,40100,40200,39800,39700,
          39800,39900,40200,39900,39900,40200,40100,
          40000,39700,40000,40100,40200,39800,39800,39700,39800,39900,40200,
          39900,39900,40200,40100,40000,39700, 39800,39700,39800,39900,40200,
          39900,39900,40200,40100,40000,39700,
          39900,39900,40200,40100,40000,39700,40000,40100,40200,39800,39700,
          39800,39900,40200,39900,39900,40200,40100,
          39900,39900,40200,40100,40000,39700,40000,40100,40200,39800,39700,
          39800,39900,40200,39900,39900,40200,40100,
          40000,39700,40000,40100,40200,39800,39800,39700,39800,39900,40200,
          39900,39900,40200,40100,40000,39700,
          40000,39700,40000,40100,40200,39800,39700,
          39800,39900,40200,39900,39900,
          39900,40200,40100,40000,39700,40000,40100,40200,39800,39700,
          39800,39900,40200,39900,39900,40200,40100,
          40000,39700,40000,40100,40200,
          40000,39700,40000,40100,40200,39800,39700,
          39800,39900,40200,39900,39900,40200,40100,
          39900,39900,40200,40100,40000,39700,40000,40100,40200,39800,39700,
          39800,39900,40200,39900,39900,40200,40100,
          40000,39700,40000,40100,40200,39800,39800,39700,39800,39900,40200,
          39900,39900,40200,40100,40000,39700,
          3.7,3.8,3.9,4,4,4.2,4.1,3.5,3.2,3,2.5,2.2,2,2,1.5,1.5,1.6,1,1,1,
          .5,.6,.4,.3,.5,.3,.5,.6,.1,.3,.2,.1,0,0,0,0,0,0,0,
          0,0,0,0,0,.3,.3,.4,.1,.2,.5,0,.0,1,1,1.5,2,2.5,2,3,3.5,4,4.2,3.8,4.2,3.9,3.9,4.2,4.1,4,
          0,0,0,0,0,.3,.3,.4,.1,.2,.5,0,.0,1,1,1.5,2,2.5,2,3,3.5,4,4.2,3.8,4.2,3.9,3.9,4.2,4.1,4,
          3.7,3.8,3.9,4,4,4.2,4.1,3.5,3.2,3,2.5,2.2,2,2,1.5,1.5,1.6,1,1,1,
          .5,.6,.4,.3,.5,.3,.5,.6,.1,.3,.2,.1,0,0,0,0,0,0,0,
          3.7,3.8,3.9,4,4,4.2,4.1,3.5,3.2,3,2.5,2.2,2,2,1.5,1.5,1.6,1,1,1,
          .5,.6,.4,.3,.5,.3,.5,.6,.1,.3,.2,.1,0,0,0,0,0,0,0,]
    time = [datetime(2024,5,27,13,19,20) + timedelta(seconds=i) for i in range(len(counts))]
    return pd.DataFrame({'Time': time, 'value': counts})

@pytest.fixture
def rr_plateau(plateau_data):
    return CountRate(plateau_data, plateau_data.Time.min(),
                        campaign_id='A', experiment_id='B',
                        detector_id=1, deposit_id='dep')

@pytest.fixture
def plateau_monitor(plateau_data):
    data_ = plateau_data.copy()
    data_.value = [15000] * len(data_.value)
    return CountRate(data_, data_.Time.min(),
                        campaign_id='A', experiment_id='B',
                        detector_id=2, deposit_id='dep')

@pytest.fixture
def dtc_monitor():
    t_start = datetime(2025, 4, 3, 15, 33, 20)
    return CountRate(
        data=pd.DataFrame({"Time": [t_start, t_start + timedelta(seconds=1)], "value": [1e6, 1e6]}),
        start_time=t_start,
        campaign_id="TEST",
        experiment_id="TEST_DTC_MONITOR",
        detector_id="A",
        deposit_id="U235",
        timebase=1
    )

@pytest.fixture
def linear_monitor():
    t_start = datetime(2025, 4, 3, 15, 33, 20)
    return CountRate(
        data=pd.DataFrame({"Time": [t_start] + [t_start + timedelta(seconds=i) for i in range(1, 5)], "value": [1, 2, 3, 4, 5]}),
        start_time=t_start,
        campaign_id="TEST",
        experiment_id="TEST_LINEAR_MONITOR",
        detector_id="A",
        deposit_id="U235",
        timebase=1
    )

@pytest.fixture
def exponential_monitor():
    counts = [10050.        ,  9700.        , 10300.        ,  9908.        ,
       10050.        ,  9700.        , 10300.        ,  9908.        ,
       10050.        ,  9700.        , 10300.        ,  9908.        ,
       10050.        ,  9700.        , 10300.        ,  9908.        ,
       10050.        ,  9700.        , 10300.        ,  9908.        ,
       10050.        ,  9700.        , 10300.        ,  9908.        ,
       10050.        ,  9700.        , 10300.        ,  9908.        ,
       10050.        ,  9700.        , 10300.        ,  9908.        ,
       10050.        ,  9700.        , 10300.        ,  9908.        ,
       10050.        ,  9700.        , 10300.        ,  9908.        ,
       10205.26539385, 10190.06030508, 10534.60638164, 10605.70617995,
       10749.44020793, 10968.52419814, 11370.74649383, 11545.7363885 ,
       11663.6458216 , 12017.20722169, 12249.82576268, 12412.21968879,
       12778.52915873, 12937.47377821, 13299.53833755, 13536.52111918,
       13874.96052003, 14262.61344635, 14454.27778699, 14743.5494138 ,
       15079.18223329, 15034.02672851, 15436.24714356, 15823.1206704 ,
       16141.47220188, 16713.00383297, 16772.88364406, 17209.66305423,
       17583.26350302, 17936.62315281, 18268.0000186 , 18998.50345006,
       19250.20044118, 19651.98517352, 19696.98781926, 20283.29362904,
       20808.51131891, 21163.77208409, 21392.5437547 , 21862.62045796,
       22444.97352695, 22781.21682144, 23403.77726871, 24027.84237773,
       23849.65310135, 24793.52836332, 25482.98225099, 25875.30626789,
       26457.86940532, 26836.09209241, 27262.94393846, 27725.79359112,
       28583.51038749, 29091.75735898, 29670.36202313, 30283.67543766,
       31031.28478653, 31719.91231029, 31970.9090208 , 32900.54447419,
       33437.39156945, 34246.14436673, 35235.63850054, 35595.09969342,
       36451.57508   , 37187.51010199, 37831.10078657, 38801.97217786,
       39627.13826139, 40303.0429204 , 41510.99759457, 41930.87449026,
       43145.95066861, 43489.62613014, 44603.28665924, 45173.50764359,
       46948.36420313, 47144.43376751, 48169.99529848, 49446.52244024,
       49790.41053071, 51507.49109536, 52366.8190593 , 53246.05149746,
       54230.21762011, 56163.98465553, 56907.51125403, 58071.36205947,
       59560.62009647, 60358.49641309, 61656.70563861, 62539.88815849,
       64547.34469704, 65631.03518449, 66641.72355218, 68251.04979896,
       69226.06006311, 71208.31568526, 72576.15641079, 74500.61794504,
       74503.61794504, 74495.61794504, 74515.61794504, 74500.61794504,
       74503.61794504, 74495.61794504, 74515.61794504, 74500.61794504,
       74503.61794504, 74495.61794504, 74515.61794504, 74500.61794504,
       74503.61794504, 74495.61794504, 74515.61794504, 74500.61794504,
       74503.61794504, 74495.61794504, 74515.61794504, 74500.61794504,
       74503.61794504, 74495.61794504, 74515.61794504, 74500.61794504,
       74503.61794504, 74495.61794504, 74515.61794504, 74500.61794504,
       74503.61794504, 74495.61794504, 74515.61794504, 74500.61794504,
       74503.61794504, 74495.61794504, 74515.61794504, 74500.61794504,
       74503.61794504, 74495.61794504, 74515.61794504, 74500.61794504]
    # data generated with period T = 50s
    t_start = datetime(2025, 4, 3, 15, 33, 20)
    times = [t_start + timedelta(seconds=i) for i in range(180)]
    return CountRate(
        data=pd.DataFrame({"Time": times, "value": counts}),
        start_time=t_start,
        campaign_id="TEST",
        experiment_id="TEST_EXPONENTIAL_MONITOR",
        detector_id="A",
        deposit_id="U235",
        timebase=10,
        _dead_time_corrected=True  # here I assume it to me already corrected to ease the testing
    )

@pytest.fixture
def cut_exponential_monitor(exponential_monitor):
    # data generated with period T = 50s
    data = exponential_monitor.data.iloc[40:-40]
    return CountRate(
        data=data,
        start_time=data.Time.min(),
        campaign_id="TEST",
        experiment_id="TEST_EXPONENTIAL_MONITOR",
        detector_id="A",
        deposit_id="U235",
        timebase=10,
        _dead_time_corrected=True  # here I assume it to me already corrected to ease the testing
    )

def test_average(power_monitor):
    expected_df = pd.DataFrame({'value': 11.66666667,
                                'uncertainty': 1.9720265943665387,
                                'uncertainty [%]': 16.903085094570333},
                                index=['value'])
    pd.testing.assert_frame_equal(expected_df, power_monitor.average(power_monitor.start_time,
                                                                     3), check_exact=False, atol=0.00001)

@pytest.fixture
def linear_monitor():
    t_start = datetime(2025, 4, 3, 15, 33, 20)
    return CountRate(
        data=pd.DataFrame({"Time": [t_start] + [t_start + timedelta(seconds=i) for i in range(1, 5)], "value": [1, 2, 3, 4, 5]}),
        start_time=t_start,
        campaign_id="TEST",
        experiment_id="TEST_LINEAR_MONITOR",
        detector_id="A",
        deposit_id="U235",
        timebase=1
    )

@pytest.fixture
def exponential_monitor():
    counts = [10050.        ,  9700.        , 10300.        ,  9908.        ,
       10050.        ,  9700.        , 10300.        ,  9908.        ,
       10050.        ,  9700.        , 10300.        ,  9908.        ,
       10050.        ,  9700.        , 10300.        ,  9908.        ,
       10050.        ,  9700.        , 10300.        ,  9908.        ,
       10050.        ,  9700.        , 10300.        ,  9908.        ,
       10050.        ,  9700.        , 10300.        ,  9908.        ,
       10050.        ,  9700.        , 10300.        ,  9908.        ,
       10050.        ,  9700.        , 10300.        ,  9908.        ,
       10050.        ,  9700.        , 10300.        ,  9908.        ,
       10205.26539385, 10190.06030508, 10534.60638164, 10605.70617995,
       10749.44020793, 10968.52419814, 11370.74649383, 11545.7363885 ,
       11663.6458216 , 12017.20722169, 12249.82576268, 12412.21968879,
       12778.52915873, 12937.47377821, 13299.53833755, 13536.52111918,
       13874.96052003, 14262.61344635, 14454.27778699, 14743.5494138 ,
       15079.18223329, 15034.02672851, 15436.24714356, 15823.1206704 ,
       16141.47220188, 16713.00383297, 16772.88364406, 17209.66305423,
       17583.26350302, 17936.62315281, 18268.0000186 , 18998.50345006,
       19250.20044118, 19651.98517352, 19696.98781926, 20283.29362904,
       20808.51131891, 21163.77208409, 21392.5437547 , 21862.62045796,
       22444.97352695, 22781.21682144, 23403.77726871, 24027.84237773,
       23849.65310135, 24793.52836332, 25482.98225099, 25875.30626789,
       26457.86940532, 26836.09209241, 27262.94393846, 27725.79359112,
       28583.51038749, 29091.75735898, 29670.36202313, 30283.67543766,
       31031.28478653, 31719.91231029, 31970.9090208 , 32900.54447419,
       33437.39156945, 34246.14436673, 35235.63850054, 35595.09969342,
       36451.57508   , 37187.51010199, 37831.10078657, 38801.97217786,
       39627.13826139, 40303.0429204 , 41510.99759457, 41930.87449026,
       43145.95066861, 43489.62613014, 44603.28665924, 45173.50764359,
       46948.36420313, 47144.43376751, 48169.99529848, 49446.52244024,
       49790.41053071, 51507.49109536, 52366.8190593 , 53246.05149746,
       54230.21762011, 56163.98465553, 56907.51125403, 58071.36205947,
       59560.62009647, 60358.49641309, 61656.70563861, 62539.88815849,
       64547.34469704, 65631.03518449, 66641.72355218, 68251.04979896,
       69226.06006311, 71208.31568526, 72576.15641079, 74500.61794504,
       74503.61794504, 74495.61794504, 74515.61794504, 74500.61794504,
       74503.61794504, 74495.61794504, 74515.61794504, 74500.61794504,
       74503.61794504, 74495.61794504, 74515.61794504, 74500.61794504,
       74503.61794504, 74495.61794504, 74515.61794504, 74500.61794504,
       74503.61794504, 74495.61794504, 74515.61794504, 74500.61794504,
       74503.61794504, 74495.61794504, 74515.61794504, 74500.61794504,
       74503.61794504, 74495.61794504, 74515.61794504, 74500.61794504,
       74503.61794504, 74495.61794504, 74515.61794504, 74500.61794504,
       74503.61794504, 74495.61794504, 74515.61794504, 74500.61794504,
       74503.61794504, 74495.61794504, 74515.61794504, 74500.61794504]
    # data generated with period T = 50s
    t_start = datetime(2025, 4, 3, 15, 33, 20)
    times = [t_start + timedelta(seconds=i) for i in range(180)]
    return CountRate(
        data=pd.DataFrame({"Time": times, "value": counts}),
        start_time=t_start,
        campaign_id="TEST",
        experiment_id="TEST_EXPONENTIAL_MONITOR",
        detector_id="A",
        deposit_id="U235",
        timebase=10,
        _dead_time_corrected=True  # here I assume it to me already corrected to ease the testing
    )

def test_average(power_monitor, power_monitor_uncertain_time_binning, power_monitor_02s):
    expected_df = pd.DataFrame({'value': 8.75,
                                'uncertainty': 1.479019945774904,
                                'uncertainty [%]': 16.903085094570333},
                                index=['value'])
    pd.testing.assert_frame_equal(expected_df, power_monitor.average(power_monitor.start_time,
                                                                     4), check_exact=False, atol=0.00001)
    # now with timebase < 1s
    pd.testing.assert_frame_equal(expected_df, power_monitor_02s.average(
                                                                    power_monitor_02s.start_time,
                                                                    4), check_exact=False, atol=0.00001)
    # now with uncertain time binning
    expected_df = pd.DataFrame({'value': 8.751249999999999,
                                'uncertainty': 1.4791255862840045,
                                'uncertainty [%]': 16.90187786069424},
                                index=['value'])
    pd.testing.assert_frame_equal(expected_df, power_monitor_uncertain_time_binning.average(
                                                            power_monitor_uncertain_time_binning.start_time, 4),
                                                                            check_exact=False, atol=0.00001)

def test_integrate(power_monitor):
    expected_df = pd.DataFrame({'value': [5, 12.5, 17.5],
                                'uncertainty': [1.5811388300841898, 2.5, 2.958039891549808],
                                'uncertainty [%]': [31.622776601683793, 20.0, 16.903085094570333]})
    pd.testing.assert_frame_equal(expected_df, power_monitor.integrate(2), check_exact=False, atol=0.00001)

def test_plateau(rr_plateau):
    MIN_T = datetime(2024,5,27,13,21,1)
    MAX_T = datetime(2024,5,27,13,24,20)
    COUNTS = 7992600.0

    assert rr_plateau.plateau(1).Time.min() == MIN_T
    assert rr_plateau.plateau(1).Time.max() == MAX_T
    assert rr_plateau.plateau(1).value.sum() == COUNTS

def test_plateau_timebase(rr_plateau):
    MIN_T = datetime(2024,5,27,13,20,59)
    MAX_T = datetime(2024,5,27,13,24,21)
    COUNTS = 8112600.0

    assert rr_plateau.plateau(2, timebase=7).Time.min() == MIN_T
    assert rr_plateau.plateau(2, timebase=7).Time.max() == MAX_T
    assert rr_plateau.plateau(2, timebase=7).value.sum() == COUNTS

def test_per_unit_power(rr_plateau, plateau_monitor):
    expected_df = pd.DataFrame({'value': 532.84,
                                'uncertainty': 0.36143842,
                                'uncertainty [%]': 0.06783245}, index=['value'])
    pd.testing.assert_frame_equal(expected_df, rr_plateau.per_unit_power(plateau_monitor))

def test_per_unit_time_power(rr_plateau, plateau_monitor):
    MIN_T = datetime(2024,5,27,13,21,1)
    MAX_T = datetime(2024,5,27,13,24,20)
    D = (MAX_T - MIN_T).seconds
    V, U = 532.84 / D, 0.36143842 / D
    target = rr_plateau.per_unit_time_power(plateau_monitor)
    assert np.isclose(target.value.values[0], V, atol=0.00001)
    assert np.isclose(target.uncertainty.values[0], U, atol=0.00001)

def test_smooth(plateau_monitor):
    # individual smoothings tested in test_utils.py
    data = plateau_monitor.smooth()
    assert data.campaign_id == "A"
    assert data.experiment_id == "B"
    assert data.start_time == datetime(2024,5,27,13,19,20)
    assert data.detector_id == 2
    assert data.deposit_id == 'dep'

def test_dead_time_corrected(dtc_monitor):
    target = pd.DataFrame({"Time": [datetime(2025, 4, 3, 15, 33, 20),
                                    datetime(2025, 4, 3, 15, 33, 21)],
                           "value": [1122066.89701783, 1122066.89701783]})
    pd.testing.assert_frame_equal(dtc_monitor.dead_time_corrected().data, target)
    assert dtc_monitor.dead_time_corrected().start_time == dtc_monitor.start_time
    assert dtc_monitor.dead_time_corrected().campaign_id == dtc_monitor.campaign_id
    assert dtc_monitor.dead_time_corrected().experiment_id == dtc_monitor.experiment_id
    assert dtc_monitor.dead_time_corrected().detector_id == dtc_monitor.detector_id
    assert dtc_monitor.dead_time_corrected().deposit_id == dtc_monitor.deposit_id
    assert dtc_monitor.dead_time_corrected().timebase == dtc_monitor.timebase
    assert dtc_monitor.dead_time_corrected()._dead_time_corrected == True

def test_cut(exponential_monitor, cut_exponential_monitor):
    ts = datetime(2025, 4, 3, 15, 34, 0)
    te = datetime(2025, 4, 3, 15, 35, 40)
    cut = exponential_monitor.cut(ts, te)
    pd.testing.assert_frame_equal(
        cut.data,
        cut_exponential_monitor.data
    )
    assert(cut.start_time == ts)
    assert(cut.campaign_id == exponential_monitor.campaign_id)
    assert(cut.experiment_id == exponential_monitor.experiment_id)
    assert(cut.detector_id == exponential_monitor.detector_id)
    assert(cut.deposit_id == exponential_monitor.deposit_id)
    assert(cut.timebase == exponential_monitor.timebase)
    assert(cut._dead_time_corrected == exponential_monitor._dead_time_corrected)

def test_linear_fit(linear_monitor):
    fitted_data, popt, pcov, out = linear_monitor._linear_fit(preprocessing=None)
    np.testing.assert_array_equal(fitted_data, linear_monitor.data.value.values)
    np.testing.assert_almost_equal(popt, np.array([1., 1.]))
    np.testing.assert_almost_equal(pcov, np.array([[0.] * 2,
                                                   [0.] * 2]))
    np.testing.assert_almost_equal(out['fvec'], np.array([0.] * 5))

def test_get_asymptotic_period(cut_exponential_monitor, caplog):
    # test logging
    with caplog.at_level("INFO"):
        cut_exponential_monitor.get_asymptotic_period()
    msg = 'Reactor period fit R^2 = 0.9999'
    records = caplog.records
    assert any(
        record.levelname == "INFO" and record.message == msg
        for record in records
    )
    # no scan
    target = pd.DataFrame({'value': 49.49970769,
                           'uncertainty': 0.05527809,
                           'uncertainty [%]': 0.11167358},
                           index=['value'])
    pd.testing.assert_frame_equal(
        cut_exponential_monitor.get_asymptotic_period(log=False),
        target)
    # scan
    target = pd.DataFrame({'value': 49.51226637073453,
                           'uncertainty': 0.055947612206386645,
                           'uncertainty [%]': 0.11299747781179309},
                           index=['value'])
    pd.testing.assert_frame_equal(
        cut_exponential_monitor.get_asymptotic_period(1., 20., 1e-2, log=True),
        target)
    # test logging with scan
    with caplog.at_level("INFO"):
        cut_exponential_monitor.get_asymptotic_period(1., 20., 1e-2, log=True)
    msg = 'Reactor period fit R^2 = 0.9999'
    records = caplog.records
    assert any(
        record.levelname == "INFO" and record.message == msg
        for record in records
    )


def test_get_reactivity(cut_exponential_monitor):
    dd = EffectiveDelayedParams(_make_df(np.array([1, 2]), np.array([0.01, 0.01]), relative=True),
                                _make_df(np.array([10, 20]), np.array([0.1, 0.1]), relative=True))
    T = 49.49970769
    uT = 0.05527809
    data = cut_exponential_monitor.get_reactivity(dd)

    a = (1 / (1 + T) * 0.1)**2
    b = (1 / (1 + 2*T) * 0.1)**2
    c = (10 / (1 + T)**2 * T * 0.01)**2
    d = (20 / (1 + 2*T)**2 * T * 0.01)**2
    e = (10 / (1 + T)**2 * uT)**2 + (20 * 2 / (1 + 2*T)**2 * uT)**2

    target = pd.DataFrame({"value": [10 / (1 + T) + 20 / (1 + 2 * T)],
                           "uncertainty": [np.sqrt(a + b + c + d + e)],
                           "uncertainty [%]": [np.sqrt(a + b + c + d + e) / (10 / (1 + T) + 20 / (1 + 2 * T)) * 100],
                           "VAR_PORT_T": [e],
                           "VAR_PORT_B": [a + b],
                           "VAR_PORT_L": [c + d]}, index=["value"])
    pd.testing.assert_frame_equal(data, target)
