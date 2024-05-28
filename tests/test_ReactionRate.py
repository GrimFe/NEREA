import pytest
import pandas as pd
from datetime import datetime, timedelta
from REPTILE.ReactionRate import ReactionRate

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
          3700,4000,4100,4200,3800,3700,3800,3900,4200,
          3900,3900,4200,4100,4000,3700,4000,4100,4200,3800,3700,3800,3900,4200,3900,3900,4200,4100,
          4000,3700,4000,4100,4200,3800,
          3900,3900,4200,4100,4000,3700,4000,4100,4200,3800,3700,3800,3900,4200,3900,3900,4200,4100,
          4000,3700,4000,4100,4200,3800, 3800,3700,3800,3900,4200,
          3900,3900,4200,4100,4000,3700, 3800,3700,3800,3900,4200,
          3900,3900,4200,4100,4000,3700,
          3900,3900,4200,4100,4000,3700,4000,4100,4200,3800,3700,3800,3900,4200,3900,3900,4200,4100,
          4000,3700,4000,4100,4200,3800,3800,3700,3800,3900,4200,
          3900,3900,4200,4100,4000,3700,
          3900,3900,4200,4100,4000,3700,4000,4100,4200,3800,3700,3800,3900,4200,3900,3900,4200,4100,
          4000,3700,4000,4100,4200,3800,3800,3700,3800,3900,4200,
          3900,3900,4200,4100,4000,3700,
          4200,3800, 3900,3900,4200,4100,4000,3700, 4200,3800, 3900,3900,4200,4100,4000,3700, 4200,3800, 3900,3900,4200,4100,4000,3700,
          4200,3800, 3900,3900,4200,4100,3800,3700,3800,3900,4200,
          3900,3900,4200,4100,4000,3700,
          4000,3700, 4200,3800, 3900,3900,4200,4100,4000,3700, 4200,3800, 3900,3900,4200,4100,4000,3700,
          4200,3800, 3900,3900,4200,4100,4000,3700, 4200,3800, 3900,3900,4200,4100,4000,3700, 4200,3800, 3900,3900,4200,4100,4000,3700,
          4100,4000,3700,4000,4100,4200,3800,3700,3800,3900,4200,3900,3900,4200,4100,
          4000,3700,4000,4100,4200,3800, 4200,3800, 3900,3900,4200,4100,4000,3700, 4200,3800, 3900,3900,4200,4100,4000,3700, 4200,3800,
          3900,3900,4200,4100,4000,3700, 4100,4000,3700,4000,4100,4200,3800,3700,3800,3900,4200,3900,3900,4200,4100,
          4000,3700,4000,4100,4200,3800, 4100,4000,3700,4000,4100,4200,3800,3700,3800,3900,4200,3900,3900,4200,4100,
          4000,3700,4000,4100,4200,3800,
          4200,3800, 3900,3900,4200,4100,4000,3700, 4200,3800, 3900,3900,4200,4100,4000,3700, 4200,3800, 3900,3900,4200,4100,4000,3700,
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

def test_average(power_monitor):
    expected_df = pd.DataFrame({'value': 11.66666667,
                                'uncertainty': 1.9720265943665387,
                                'uncertainty [%]': 16.903085094570333},
                                index=['value'])
    pd.testing.assert_frame_equal(expected_df, power_monitor.average(power_monitor.start_time,
                                                                     3), check_exact=False, atol=0.00001)

def test_integrate(power_monitor):
    expected_df = pd.DataFrame({'value': [12.5, 15, 12.5],
                                'uncertainty': [5. / 2, 5.47722558 / 2, 5. / 2],
                                'uncertainty [%]': [40. / 2, 36.51483717 / 2, 40. / 2]})
    pd.testing.assert_frame_equal(expected_df, power_monitor.integrate(2), check_exact=False, atol=0.00001)

def test_plateau(rr_plateau):
    MIN_T = datetime(2024,5,27,13,20,51)
    MAX_T = datetime(2024,5,27,13,26,39)
    COUNTS = 1356527.7

    assert rr_plateau.plateau(2).Time.min() == MIN_T
    assert rr_plateau.plateau(2).Time.max() == MAX_T
    assert rr_plateau.plateau(2).value.sum() == COUNTS

def test_plateau_timebase(rr_plateau):
    MIN_T = datetime(2024,5,27,13,23,12)
    MAX_T = datetime(2024,5,27,13,26,33)
    COUNTS = 798403.7

    assert rr_plateau.plateau(2, timebase=7).Time.min() == MIN_T
    assert rr_plateau.plateau(2, timebase=7).Time.max() == MAX_T
    assert rr_plateau.plateau(2, timebase=7).value.sum() == COUNTS

def test_per_unit_power(rr_plateau, plateau_monitor):
    expected_df = pd.DataFrame({'value': 90.43518,
                                'uncertainty': 0.087154,
                                'uncertainty [%]': 0.096372}, index=['value'])
    pd.testing.assert_frame_equal(expected_df, rr_plateau.per_unit_power(plateau_monitor))
