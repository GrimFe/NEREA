import pytest
import pandas as pd
from datetime import datetime, timedelta
from REPTILE.PowerMonitor import PowerMonitor

@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'Time': [datetime(2024, 5, 19, 20, 5, 0) + timedelta(seconds=i) for i in range(7)],
        'value': [0, 10, 15, 10, 20, 15, 10]
    })
    return data

@pytest.fixture
def power_monitor(sample_data):
    return PowerMonitor(data=sample_data, campaign_id="C1", experiment_id="E1",
                        start_time=datetime(2024, 5, 19, 20, 5, 0), monitor_id='M')

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
