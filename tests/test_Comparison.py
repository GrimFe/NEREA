import pytest
from nerea.experimental import NormalizedFissionFragmentSpectrum, SpectralIndex, Traverse
from nerea.fission_fragment_spectrum import FissionFragmentSpectrum
from nerea.effective_mass import EffectiveMass
from nerea.reaction_rate import ReactionRate
from nerea.comparisons import _Comparison
from nerea.calculated import CalculatedSpectralIndex, CalculatedTraverse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


@pytest.fixture
def sample_spectrum_data():
    # Sample data for testing
    data = {
        "channel":[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
        "counts": [0, 0, 0, 0, 1, 3, 1, 4, 1, 5,  1,  3,  4,  2,  4,  1,  3,  5,  80, 65, 35, 5,  20, 25, 35, 55, 58, 60, 62, 70, 65, 50, 45, 40, 37, 34, 25, 20, 13, 5,  1,  0]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_integral_data():
    data = {
        "channel": [ 6.,  8., 10., 12., 14., 16., 18., 20., 22., 24.],
        "value": [60, 80, 88, 87, 87, 88, 86, 85, 82, 78],
        "uncertainty": [.1, .2, .3, .4, .5, .6, .7, .8, .9, .1]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_power_monitor_data():
    data = {
        "Time": [datetime(2024, 5, 18, 20, 30, 15 + i) for i in range(20)],
        "value": [100, 101, 99, 98, 101, 100, 99, 98, 102, 102] * 2
    }
    return pd.DataFrame(data)

@pytest.fixture
def fission_fragment_spectrum_1(sample_spectrum_data):
    return FissionFragmentSpectrum(start_time=datetime(2024, 5, 18, 20, 30, 15),
                                   life_time=10, real_time=10,
                                   data=sample_spectrum_data, campaign_id="A", experiment_id="B",
                                   detector_id="C1", deposit_id="D1", location_id="E", measurement_id="F1")

@pytest.fixture
def fission_fragment_spectrum_2(sample_spectrum_data):
    return FissionFragmentSpectrum(start_time=datetime(2024, 5, 18, 20, 30, 15),
                                   life_time=10, real_time=10,
                                   data=sample_spectrum_data, campaign_id="A", experiment_id="B",
                                   detector_id="C2", deposit_id="D2", location_id="E", measurement_id="F2")

@pytest.fixture
def effective_mass_1(sample_integral_data):
    return EffectiveMass(deposit_id="D1", detector_id="C1", data=sample_integral_data, bins=42)

@pytest.fixture
def effective_mass_2(sample_integral_data):
    return EffectiveMass(deposit_id="D2", detector_id="C2", data=sample_integral_data, bins=42)

@pytest.fixture
def power_monitor(sample_power_monitor_data):
        return ReactionRate(experiment_id="B", data=sample_power_monitor_data, start_time=datetime(2024, 5, 29, 12, 25, 10), campaign_id='C', detector_id='M', deposit_id='dep')

@pytest.fixture
def rr_1(fission_fragment_spectrum_1, effective_mass_1, power_monitor):
    return NormalizedFissionFragmentSpectrum(fission_fragment_spectrum_1, effective_mass_1, power_monitor)

@pytest.fixture
def rr_2(fission_fragment_spectrum_2, effective_mass_2, power_monitor):
    return NormalizedFissionFragmentSpectrum(fission_fragment_spectrum_2, effective_mass_2, power_monitor)

@pytest.fixture
def sample_spectral_index(rr_1, rr_2):
    return SpectralIndex(rr_1, rr_2)

@pytest.fixture
def sample_c_si_data():
    return pd.DataFrame({'value': 1.01, 'uncertainty': .05, 'VAR_FRAC_C_n': None, 'VAR_FRAC_C_d': None, 'VAR_FRAC_C': 0.05 **2}, index=['value'])

@pytest.fixture
def sample_c(sample_c_si_data):
    return CalculatedSpectralIndex(sample_c_si_data, 'M', ['D1', 'D2'])

@pytest.fixture
def sample_si_ce(sample_c, sample_spectral_index):
    return _Comparison(sample_c, sample_spectral_index)

@pytest.fixture
def sample_si_cc(sample_c):
    return _Comparison(sample_c, sample_c)

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

@pytest.fixture
def rr1():
    time = [datetime(2024,5,27,13,19,20) + timedelta(seconds=i) for i in range(len(counts))]
    data =  pd.DataFrame({'Time': time, 'value': counts})
    return ReactionRate(data, data.Time.min(),
                        campaign_id='A', experiment_id='B',
                        detector_id=1, deposit_id='dep')

@pytest.fixture
def rr2():
    time = [datetime(2024,5,27,15,12,42) + timedelta(seconds=i) for i in range(len(counts))]
    data = pd.DataFrame({'Time': time, 'value': np.array(counts) / 2})
    return ReactionRate(data, data.Time.min(),
                        campaign_id='A', experiment_id='B',
                        detector_id=1, deposit_id='dep')

@pytest.fixture
def monitor1(rr1):
    data_ = rr1.data.copy()
    data_.value = data_.value.apply(lambda x: 600 if x > 1000 else 1)
    return ReactionRate(data_, data_.Time.min(),
                        campaign_id='A', experiment_id='B',
                        detector_id=2, deposit_id='dep')

@pytest.fixture
def monitor2(rr2):
    data_ = rr2.data.copy()
    data_.value = data_.value.apply(lambda x: 600 if x > 1000 else 1)
    return ReactionRate(data_, data_.Time.min(),
                        campaign_id='A', experiment_id='B',
                        detector_id=2, deposit_id='dep')

@pytest.fixture
def sample_traverse_rr(rr1, rr2):
    return Traverse({'loc A': rr1, 'loc B': rr2})

@pytest.fixture
def sample_c_traverse_data():
    return pd.DataFrame({'value': [1.01, 0.5],
                         'uncertainty': [0.01, 0.01],
                         'traverse': ['loc A', 'loc B']})

@pytest.fixture
def sample_c_traverse(sample_c_traverse_data):
    return CalculatedTraverse(sample_c_traverse_data, 'M', 'dep')

@pytest.fixture
def sample_ce_traverse(sample_c_traverse, sample_traverse_rr):
    return _Comparison(sample_c_traverse, sample_traverse_rr)

def test_deposit_ids(sample_si_ce):
    assert sample_si_ce.deposit_ids == ['D1', 'D2']

def test_compute_si(sample_si_ce):
    expected_df = pd.DataFrame({'value': 1.01,
                                'uncertainty': 0.08323682675073317,
                                'uncertainty [%]': 8.241269975320115,
                                'VAR_FRAC_C_n': None, 
                                'VAR_FRAC_C_d': None,
                                'VAR_FRAC_C': 0.0025000000000000005,
                                'VAR_FRAC_FFS_n': 0.0011603913092935957,
                                'VAR_FRAC_EM_n': 3.369335447218919e-05,
                                'VAR_FRAC_PM_n': 0.0010201000000000001,
                                'VAR_FRAC_t_n': 0.,
                                'VAR_FRAC_FFS_d': 0.0011603913092935955,
                                'VAR_FRAC_EM_d': 3.369335447218918e-05,
                                'VAR_FRAC_PM_d': 0.0010201000000000001,
                                'VAR_FRAC_t_d': 0.,
                                'VAR_FRAC_1GXS': 0.},
                                index=['value'])
    pd.testing.assert_frame_equal(expected_df, sample_si_ce.compute(), check_exact=False, atol=0.00001)
    # check that sum(VAR_FRAC) == uncertainty **2
    np.testing.assert_almost_equal(expected_df[[c for c in expected_df.columns if c.startswith("VAR_FRAC")]].sum(axis=1).iloc[0],
                                   expected_df['uncertainty'].iloc[0] **2, decimal=5)

def test_compute_traverse(sample_ce_traverse, monitor1, monitor2):
    expected_df = pd.DataFrame({'value': [1., 0.99250555],
                                'uncertainty': [0.01483835, 0.02256028],
                                'uncertainty [%]': [1.48383507, 2.27306339],
                                'traverse': ['loc A', 'loc B']})
    pd.testing.assert_frame_equal(expected_df, sample_ce_traverse.compute(monitors=[monitor1, monitor2]))

def test_minus_one_per_cent(sample_si_ce):
    expected_df = pd.DataFrame({'value': 1.,
                                'uncertainty': 8.323682675073316,
                                'uncertainty [%]': np.nan,
                                'VAR_FRAC_C_n': None, 
                                'VAR_FRAC_C_d': None,
                                'VAR_FRAC_C': 25.000000000000004,
                                'VAR_FRAC_FFS_n': 11.603913092935958,
                                'VAR_FRAC_EM_n': 0.33693354472189185,
                                'VAR_FRAC_PM_n': 10.201,
                                'VAR_FRAC_t_n': 0.,
                                'VAR_FRAC_FFS_d': 11.603913092935956,
                                'VAR_FRAC_EM_d': 0.3369335447218918,
                                'VAR_FRAC_PM_d': 10.201,
                                'VAR_FRAC_t_d': 0.,
                                'VAR_FRAC_1GXS': 0.},
                                index=['value'])
    pd.testing.assert_frame_equal(expected_df, sample_si_ce.minus_one_percent(), check_exact=False, atol=0.00001)

def test_si_cc(sample_si_cc):
    target_df = pd.DataFrame({'value': 1.,
                              'uncertainty': 0.07001057239470768,
                              'uncertainty [%]': 7.001057239470768,
                              'VAR_FRAC_C_n_n': None,
                              'VAR_FRAC_C_d_n': None,
                              'VAR_FRAC_C_n': 0.0024507401235173026,
                              'VAR_FRAC_C_n_d': None,
                              'VAR_FRAC_C_d_d': None,
                              'VAR_FRAC_C_d': 0.0024507401235173026}, index=['value'])
    pd.testing.assert_frame_equal(target_df, sample_si_cc.compute())
