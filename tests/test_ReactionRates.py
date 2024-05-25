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
                        start_time=datetime(2024, 5, 19, 20, 5, 0), detector_id='M')

@pytest.fixture
def power_monitor_2(sample_data2):
    return ReactionRate(data=sample_data2, campaign_id="C1", experiment_id="E1",
                        start_time=datetime(2024, 5, 19, 20, 5, 0), detector_id='M')

@pytest.fixture
def pms(power_monitor_1, power_monitor_2):
    return ReactionRates([power_monitor_1, power_monitor_2])

def test_best(pms, power_monitor_1):
    assert pms.best == power_monitor_1
