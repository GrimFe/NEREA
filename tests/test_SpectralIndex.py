import pytest
from nerea.experimental import NormalizedFissionFragmentSpectrum, SpectralIndex
from nerea.fission_fragment_spectrum import FissionFragmentSpectrum
from nerea.effective_mass import EffectiveMass
from nerea.reaction_rate import ReactionRate
from datetime import datetime
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
        "uncertainty": [.1, .2, .3, .4, .5, .6, .7, .8, .9, .1],
        "R": [.15, .2, .25, .3, .35, .4, .45, .5, .55, .6]
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
def sample_spectrum_1(sample_spectrum_data):
    return FissionFragmentSpectrum(start_time=datetime(2024, 5, 18, 20, 30, 15),
                                   life_time=10, real_time=10,
                                   data=sample_spectrum_data, campaign_id="A", experiment_id="B",
                                   detector_id="C1", deposit_id="U238", location_id="E", measurement_id="F1")

@pytest.fixture
def sample_spectrum_2(sample_spectrum_data):
    return FissionFragmentSpectrum(start_time=datetime(2024, 5, 18, 20, 30, 15),
                                   life_time=10, real_time=10,
                                   data=sample_spectrum_data, campaign_id="A", experiment_id="B",
                                   detector_id="C2", deposit_id="U235", location_id="E", measurement_id="F2")

@pytest.fixture
def effective_mass_1(sample_integral_data):
    data = pd.DataFrame({'U236': [0.1, 0.01], 'U234': [0.2, 0.02], 'U238': [0.7, 0.07]}).T.reset_index()
    data.columns = ['nuclide', 'value', 'uncertainty']
    return EffectiveMass(deposit_id="U238", detector_id="C1", data=sample_integral_data, bins=42, composition=data)

@pytest.fixture
def effective_mass_2(sample_integral_data):
    data = pd.DataFrame({'U234': [0.15, 0.015], 'U238': [0.25, 0.025], 'U235': [0.6, 0.06]}).T.reset_index()
    data.columns = ['nuclide', 'value', 'uncertainty']
    return EffectiveMass(deposit_id="U235", detector_id="C2", data=sample_integral_data, bins=42, composition=data)

@pytest.fixture
def power_monitor(sample_power_monitor_data):
        return ReactionRate(experiment_id="B", data=sample_power_monitor_data, start_time=datetime(2024, 5, 29, 12, 25, 10), campaign_id='C', detector_id='M', deposit_id='dep')

@pytest.fixture
def rr_1(sample_spectrum_1, effective_mass_1, power_monitor):
    return NormalizedFissionFragmentSpectrum(sample_spectrum_1, effective_mass_1, power_monitor)

@pytest.fixture
def rr_2(sample_spectrum_2, effective_mass_2, power_monitor):
    return NormalizedFissionFragmentSpectrum(sample_spectrum_2, effective_mass_2, power_monitor)

@pytest.fixture
def si(rr_1, rr_2):
    return SpectralIndex(rr_1, rr_2)

@pytest.fixture
def synthetic_one_g_xs_data():
    data = pd.DataFrame({'U236': [0.07, 0.001], 'U234': [0.08, 0.002],
                         'U238': [0.9, 0.003], 'U235': [0.6, 0.004]}).T.reset_index()
    data.columns = ['nuclide', 'value', 'uncertainty']
    return data.set_index('nuclide')

def test_deposit_ids(si):
    assert si.deposit_ids == ['U238', 'U235']

def test_get_long_output(si):
    n_proc = si.numerator.process(raw_integral=False, renormalize=False, long_output=True)
    d_proc = si.denominator.process(raw_integral=False, renormalize=False, long_output=True)
    expected_df = pd.DataFrame({'FFS_n': 8.79100000e+02,
                                'VAR_FFS_n': 8.79100000e+02,
                                'EM_n': 87,
                                'VAR_EM_n': 2.50000000e-01,
                                'PM_n': 1.00000000e+02,
                                'VAR_PM_n': 1.00000000e+01,
                                't_n': 10.,
                                'VAR_t_n': 0.,
                                'FFS_d': 8.79100000e+02,
                                'VAR_FFS_d': 8.79100000e+02,
                                'EM_d': 87,
                                'VAR_EM_d': 2.50000000e-01,
                                'PM_d': 1.00000000e+02,
                                'VAR_PM_d': 1.00000000e+01,
                                't_d': 10.,
                                'VAR_t_d': 0.,
                                '1GXS': 0,
                                'VAR_1GXS': None}, index=['value'])
    pd.testing.assert_frame_equal(expected_df, si._get_long_output(n_proc, d_proc, None))
    # check that the variances are unchanged when relate to numerator and denominator
    assert n_proc["VAR_FFS"].value == expected_df["VAR_FFS_n"].value
    assert d_proc["VAR_FFS"].value == expected_df["VAR_FFS_d"].value
    assert n_proc["VAR_EM"].value == expected_df["VAR_EM_n"].value
    assert d_proc["VAR_EM"].value == expected_df["VAR_EM_d"].value
    assert n_proc["VAR_PM"].value == expected_df["VAR_PM_n"].value
    assert d_proc["VAR_PM"].value == expected_df["VAR_PM_d"].value
    assert n_proc["VAR_t"].value == expected_df["VAR_t_n"].value
    assert d_proc["VAR_t"].value == expected_df["VAR_t_d"].value

    ## test empty is returned if no long output was passed to the num and den processing
    n_proc = si.numerator.process(raw_integral=False, renormalize=False)
    d_proc = si.denominator.process(raw_integral=False, renormalize=False)
    pd.testing.assert_frame_equal(pd.DataFrame(), si._get_long_output(n_proc, d_proc, None))

def test_process(si):
    expected_df = pd.DataFrame({'value': 1.,
                                'uncertainty': 0.06588712284729072,
                                'uncertainty [%]': 6.5887122847290716,
                                'VAR_PORT_FFS_n': 0.001138,
                                'VAR_PORT_EM_n': 0.000033,
                                'VAR_PORT_PM_n': 0.001000,
                                'VAR_PORT_t_n': 0.,
                                'VAR_PORT_FFS_d': 0.001138,
                                'VAR_PORT_EM_d': 0.000033,
                                'VAR_PORT_PM_d': 0.001000,
                                'VAR_PORT_t_d': 0.,
                                'VAR_PORT_1GXS': 0.}, index= ['value'])
    pd.testing.assert_frame_equal(expected_df,
                                  si.process(numerator_kwargs={'raw_integral': False, 'renormalize': False},
                                             denominator_kwargs={'raw_integral': False, 'renormalize': False}),
                                  check_exact=False, atol=0.00001)
    # check that sum(VAR_PORT) == uncertainty **2
    np.testing.assert_almost_equal(expected_df[[c for c in expected_df.columns if c.startswith("VAR_PORT")]].sum(axis=1).iloc[0],
                                   expected_df['uncertainty'].iloc[0] **2, decimal=5)

def test_compute_correction(si, synthetic_one_g_xs_data):
    w1, uw1, w2, uw2, wd, uwd = .1, .01, .2, .02, .7, .07
    x1, ux1, x2, ux2, xd, uxd = .07 / 236., .001 / 236., .08 / 234., .002 / 234., .6 / 235.043923, .004 / 235.043923
    v = (w1/wd * x1/xd) + (w2/wd * x2/xd)

    W1, X1 = w1 / wd, x1 / xd
    vW1, vX1 = (uw1 / wd) **2 + (w1 / wd **2 * uwd) **2, (ux1 / xd) **2 + (x1 / xd **2 * uxd) **2 
    W2, X2 = w2 / wd, x2 / xd
    vW2, vX2 = (uw2 / wd) **2 + (w2 / wd **2 * uwd) **2, (ux2 / xd) **2 + (x2 / xd **2 * uxd) **2
    u = np.sqrt(vW1 * X1**2 + W1**2 * vX1 + vW2 * X2**2 + W2**2 * vX2)

    data = pd.DataFrame({'value': [v], 'uncertainty': [u], 'uncertainty [%]': u / v * 100}, index=['value'])

    nerea_ = si._compute_correction(synthetic_one_g_xs_data)

    np.testing.assert_equal(data.index.values, nerea_.index.values)
    np.testing.assert_equal(data.columns.values, nerea_.columns.values)
    np.testing.assert_almost_equal(data['value'].values, nerea_['value'].values, decimal=5)
    np.testing.assert_almost_equal(data['uncertainty'].values, nerea_['uncertainty'].values, decimal=6)

def test_compute_with_correction(si, synthetic_one_g_xs_data):
    w1, uw1, w2, uw2, wd, uwd = .1, .01, .2, .02, .7, .07
    x1, ux1, x2, ux2, xd, uxd = .07, .001, .08, .002, .6, .004
    v = (w1/wd * x1/xd) + (w2/wd * x2/xd)

    W1, X1 = w1 / wd, x1 / xd
    vW1, vX1 = (uw1 / wd) **2 + (w1 / wd **2 * uwd) **2, (ux1 / xd) **2 + (x1 / xd **2 * uxd) **2 
    W2, X2 = w2 / wd, x2 / xd
    vW2, vX2 = (uw2 / wd) **2 + (w2 / wd **2 * uwd) **2, (ux2 / xd) **2 + (x2 / xd **2 * uxd) **2
    u = np.sqrt(vW1 * X1**2 + W1**2 * vX1 + vW2 * X2**2 + W2**2 * vX2)

    v_ = 1 - v
    u_ = np.sqrt(0.06588712284729072 **2 + u **2)

    data = pd.DataFrame({'value': [v_], 'uncertainty': [u_], 'uncertainty [%]': u_ / v_ * 100}, index=['value'])

    nerea_ = si.process(synthetic_one_g_xs_data,
                        numerator_kwargs={'raw_integral': False, 'renormalize': False},
                        denominator_kwargs={'raw_integral': False, 'renormalize': False})
    np.testing.assert_equal(data.index.values, nerea_.index.values)
    np.testing.assert_equal(data.columns.values, nerea_[['value', 'uncertainty', 'uncertainty [%]']].columns.values)
    np.testing.assert_almost_equal(data['value'].values, nerea_['value'].values, decimal=4)
    np.testing.assert_almost_equal(data['uncertainty'].values, nerea_['uncertainty'].values, decimal=5)
