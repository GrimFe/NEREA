import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from nerea.reaction_rate import ReactionRate
from nerea.calculated import EffectiveDelayedParams
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
    return ReactionRate(data=sample_data, campaign_id="C1", experiment_id="E1",
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
    return ReactionRate(plateau_data, plateau_data.Time.min(),
                        campaign_id='A', experiment_id='B',
                        detector_id=1, deposit_id='dep')

@pytest.fixture
def plateau_monitor(plateau_data):
    data_ = plateau_data.copy()
    data_.value = [15000] * len(data_.value)
    return ReactionRate(data_, data_.Time.min(),
                        campaign_id='A', experiment_id='B',
                        detector_id=2, deposit_id='dep')


@pytest.fixture
def dtc_monitor():
    t_start = datetime(2025, 4, 3, 15, 33, 20)
    return ReactionRate(
        data=pd.DataFrame({"Time": [t_start, t_start + timedelta(seconds=1)], "value": [1e6, 1e6]}),
        start_time=t_start,
        campaign_id="TEST",
        experiment_id="TEST_DTC_MONITOR",
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
    return ReactionRate(
        data=pd.DataFrame({"Time": times, "value": counts}),
        start_time=t_start,
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

def test_average_timebase(power_monitor):
    expected_df = pd.DataFrame({'value': 11.66666667 * 2,
                                'uncertainty': 1.9720265943665387 * 2,
                                'uncertainty [%]': 16.903085094570333},
                                index=['value'])
    power_monitor.timebase = 2
    pd.testing.assert_frame_equal(expected_df, power_monitor.average(power_monitor.start_time,
                                                                     3), check_exact=False, atol=0.00001)

def test_integrate(power_monitor):
    expected_df = pd.DataFrame({'value': [12.5, 15, 12.5],
                                'uncertainty': [5. / 2, 5.47722558 / 2, 5. / 2],
                                'uncertainty [%]': [40. / 2, 36.51483717 / 2, 40. / 2]})
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

def test_smooth():
    pass

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

def test_get_asymptotic_counts(exponential_monitor):
    t_start, t_end = datetime(2025, 4, 3, 15, 33, 58), datetime(2025, 4, 3, 15, 35, 39)
    asymptotic = exponential_monitor.get_asymptotic_counts(t_left=0.012).data
    assert asymptotic.Time.iloc[0] == t_start
    assert asymptotic.Time.iloc[-1] == t_end

    # test continuity in time
    assert ((asymptotic.index.values - np.roll(asymptotic.index.values, 1))[1:] == 1).all()

def test_period(exponential_monitor):
    target = pd.DataFrame({"value": [49.61310925],
                           "uncertainty": [8.27759428],
                           "uncertainty [%]": [16.68428849]},
                           index=["value"])
    pd.testing.assert_frame_equal(exponential_monitor.get_asymptotic_counts(t_left=0.012).period, target)

def test_get_reactivity(exponential_monitor):
    dd = EffectiveDelayedParams(_make_df(np.array([1, 2]), np.array([0.01, 0.01]), relative=True),
                                _make_df(np.array([10, 20]), np.array([0.1, 0.1]), relative=True))
    T = 49.61310925
    uT = 8.27759428
    data = exponential_monitor.get_asymptotic_counts(t_left=0.012).get_reactivity(dd)

    a = (1 / (1 + T) * 0.1)**2
    b = (1 / (1 + 2*T) * 0.1)**2
    c = (10 / (1 + T)**2 * T * 0.01)**2
    d = (20 / (1 + 2*T)**2 * T * 0.01)**2
    e = (10 / (1 + T)**2 * uT)**2 + (20 * 2 / (1 + 2*T)**2 * uT)**2

    target = pd.DataFrame({"value": [10 / (1 + T) + 20 / (1 + 2 * T)],
                           "uncertainty": [np.sqrt(a + b + c + d + e)],
                           "uncertainty [%]": [np.sqrt(a + b + c + d + e) / (10 / (1 + T) + 20 / (1 + 2 * T)) * 100],
                           "VAR_FRAC_T": [e],
                           "VAR_FRAC_B": [a + b],
                           "VAR_FRAC_L": [c + d]}, index=["value"])
    pd.testing.assert_frame_equal(data, target)
