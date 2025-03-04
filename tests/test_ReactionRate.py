import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from nerea.reaction_rate import ReactionRate
#  +\
                # [datetime(2024, 5, 19, 20, 5, 0) + timedelta(seconds=i) for i in [0, 1.001, 1.999, 3, 4, 5, 6]
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
def sample_data_uncertain_time_binning():
    data = pd.DataFrame({
        'Time': [datetime(2024, 5, 19, 20, 5, 0) + timedelta(seconds=i) for i in [0, 0.999, 2.001, 3, 4, 5, 5.999]],
        'value': [0, 10, 15, 10, 20, 15, 10]
    })
    return data

@pytest.fixture
def power_monitor_uncertain_time_binning(sample_data_uncertain_time_binning):
    return ReactionRate(data=sample_data_uncertain_time_binning, campaign_id="C1", experiment_id="E1",
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

def test_average(power_monitor, power_monitor_uncertain_time_binning):
    expected_df = pd.DataFrame({'value': 8.75,
                                'uncertainty': 1.479019945774904,
                                'uncertainty [%]': 16.903085094570333},
                                index=['value'])
    pd.testing.assert_frame_equal(expected_df, power_monitor.average(power_monitor.start_time,
                                                                     4), check_exact=False, atol=0.00001)
    # now with uncertain time binning
    expected_df = pd.DataFrame({'value': 8.751249999999999,
                                'uncertainty': 1.4791255862840045,
                                'uncertainty [%]': 16.90187786069424},
                                index=['value'])
    pd.testing.assert_frame_equal(expected_df, power_monitor_uncertain_time_binning.average(
                                                                                power_monitor.start_time, 4),
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
