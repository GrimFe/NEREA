import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from REPTILE.ReactionRate import ReactionRate, ReactionRates

@pytest.fixture
def sample_data1():
    data = pd.DataFrame({
        'Time': [datetime(2024, 5, 19, 11, 19, 20) + timedelta(seconds=i) for i in range(7)],
        'value': [0, 10, 15, 10, 20, 15, 10]
    })
    return data

@pytest.fixture
def sample_data2():
    data = pd.DataFrame({
        'Time': [datetime(2024, 5, 19, 11, 19, 20) + timedelta(seconds=i) for i in range(7)],
        'value': [0, 1, 2, 1, 2, 2, 1]
    })
    return data

@pytest.fixture
def power_monitor_1(sample_data1):
    return ReactionRate(data=sample_data1, campaign_id="C1", experiment_id="E1",
                        start_time=datetime(2024, 5, 19, 20, 5, 0), detector_id='M', deposit_id='dep')

@pytest.fixture
def power_monitor_2(sample_data2):
    return ReactionRate(data=sample_data2, campaign_id="C1", experiment_id="E1",
                        start_time=datetime(2024, 5, 19, 20, 5, 0), detector_id='M', deposit_id='dep')

@pytest.fixture
def pms(power_monitor_1, power_monitor_2):
    return ReactionRates({1: power_monitor_1, 2: power_monitor_2})

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
    data_.value = data_.value.apply(lambda x: 600 if x > 3000 else 1)
    return ReactionRate(data_, data_.Time.min(),
                        campaign_id='A', experiment_id='B',
                        detector_id=2, deposit_id='dep')

def test_deposit_id(pms, power_monitor_1):
    assert pms.deposit_id == power_monitor_1.deposit_id
    assert pms.deposit_id == 'dep'

def test_best(pms, power_monitor_1):
    assert pms.best == power_monitor_1

def test_per_unit_power(rr_plateau, plateau_monitor):
    expected_df = pd.DataFrame({'value': 13321.,
                                'uncertainty': 38.83779782,
                                'uncertainty [%]': 0.29155317},
                                index=['value'])
    pd.testing.assert_frame_equal(expected_df,
                                  ReactionRates({1: rr_plateau,
                                                 2: plateau_monitor}
                                                 ).per_unit_power(2)[1])
