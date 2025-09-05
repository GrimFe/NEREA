import pytest
from nerea.experimental import NormalizedFissionFragmentSpectrum
from nerea.fission_fragment_spectrum import FissionFragmentSpectrum
from nerea.effective_mass import EffectiveMass
from nerea.reaction_rate import ReactionRate
from nerea.utils import _make_df
from datetime import datetime
import pandas as pd
import numpy as np

@pytest.fixture
def sample_spectrum_data():
    # Sample data for testing
    data = {
        "channel":[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
        "value": [0, 0, 0, 0, 1, 3, 1, 4, 1, 5,  1,  3,  4,  2,  4,  1,  3,  5,  80, 65, 35, 5,  20, 25, 35, 55, 58, 60, 62, 70, 65, 50, 45, 40, 37, 34, 25, 20, 13, 5,  1,  0]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_integral_data():
    data = {
        "channel": [ 6.,  8., 10., 12., 14., 16., 18., 20., 22., 24.],
        "value":   [60,  80,  88,  87,  87,  88,  86,  85,  82,  78],
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
def fission_fragment_spectrum(sample_spectrum_data):
    return FissionFragmentSpectrum(start_time=datetime(2024, 5, 18, 20, 30, 15),
                                   life_time=10, real_time=10,
                                   data=sample_spectrum_data, campaign_id="A", experiment_id="B",
                                   detector_id="C", deposit_id="D", location_id="E", measurement_id="F")

@pytest.fixture
def effective_mass(sample_integral_data):
    return EffectiveMass(deposit_id="D", detector_id="C", data=sample_integral_data, bins=42)

@pytest.fixture
def power_monitor(sample_power_monitor_data):
        return ReactionRate(experiment_id="B", data=sample_power_monitor_data,
                            start_time=datetime(2024, 5, 29, 12, 25, 10), campaign_id='C', detector_id='M', deposit_id='dep')

@pytest.fixture
def nffs(fission_fragment_spectrum, effective_mass, power_monitor):
    return NormalizedFissionFragmentSpectrum(fission_fragment_spectrum, effective_mass, power_monitor)

def test_reaction_rate_measurement_id(nffs):
    assert nffs.measurement_id == "F"

def test_reaction_rate_campaign_id(nffs):
    assert nffs.campaign_id == "A"

def test_reaction_rate_experiment_id(nffs):
    assert nffs.experiment_id == "B"

def test_reaction_rate_location_id(nffs):
    assert nffs.location_id == "E"

def test_reaction_rate_deposit_id(nffs):
    assert nffs.deposit_id == "D"

def test_time_normalization(nffs):
    pd.testing.assert_frame_equal(nffs._time_normalization, _make_df(.1, 0.))
    # test variance
    tmp = nffs
    tmp.fission_fragment_spectrum.life_time_uncertainty = .1
    tmp.fission_fragment_spectrum.real_time_uncertainty = .1
    pd.testing.assert_frame_equal(tmp._time_normalization, _make_df(.1, 1/10 **2 * .1))

def test_power_normalization(nffs):
    pd.testing.assert_frame_equal(nffs._power_normalization,
                                  _make_df(.01, 0.00031622776601683794))

def test_get_long_output(nffs):
    expected_df = pd.DataFrame({'FFS': 879.0999999999999, 'VAR_FFS': 879.0999999999999,
                                'EM': 87, 'VAR_EM': .5 **2,
                                'PM': 1 / .01, 'VAR_PM': 0.00031622776601683794 **2 / .01 **4,
                                't': 1 / .1, 'VAR_t': 0.}, index=['value'])
    pd.testing.assert_frame_equal(expected_df,
                                  nffs._get_long_output(nffs.plateau(raw_integral=False, renormalize=False),
                                                        nffs._time_normalization,
                                                        nffs._power_normalization,
                                                        raw_integral=False,
                                                        renormalize=False))
    pd.testing.assert_frame_equal(expected_df,
                                  nffs._get_long_output(nffs.plateau(raw_integral=False, renormalize=False),
                                                        nffs._time_normalization,
                                                        nffs._power_normalization,
                                                        bins=2,
                                                        raw_integral=False,
                                                        renormalize=False))

def test_per_unit_mass_R(nffs):
    expected_df = pd.DataFrame({
        'value': [14.77333333, 11.08 , 10.07272727, 10.15287356, 10.1045977,
                  9.92954545, 10.09767442, 10.05529412,  9.97195122,  9.93974359],
        'uncertainty': [0.49681835, 0.37318533, 0.34006171, 0.34478792, 0.3457126,
                        0.34266489, 0.35237775, 0.3567267 , 0.36549703, 0.35720442],
        'uncertainty [%]': [3.36294012, 3.36809864, 3.3760639 , 3.39596385, 3.42133962,
                            3.45096254, 3.4896921 , 3.54765061, 3.66525089, 3.59369858],
        'VAR_PORT_FFS': [0.24622222, 0.1385    , 0.11446281, 0.1166997 , 0.1161448 ,
                    0.11283574, 0.11741482, 0.11829758, 0.12160916, 0.12743261],
        'VAR_PORT_EM': [0.00060625, 0.00076729, 0.00117916, 0.00217901, 0.0033724 ,
                   0.00458349, 0.00675526, 0.00895636, 0.01197892, 0.00016239],
        'CH_FFS': [6.,  8., 10., 12., 14., 16., 18., 20., 22., 24.],
        'CH_EM': [6.,  8., 10., 12., 14., 16., 18., 20., 22., 24.],
        "R": [.15, .2, .25, .3, .35, .4, .45, .5, .55, .6]})
    pd.testing.assert_frame_equal(expected_df,
                                  nffs._per_unit_mass_R(nffs.fission_fragment_spectrum.integrate(
                                                                raw_integral=False,
                                                                renormalize=False),
                                                        nffs.effective_mass.integral),
                                  check_exact=False, atol=0.00001)

def test_per_unit_mass_ch(nffs):
    expected_df = pd.DataFrame({
        'value': [14.77333333, 11.08 , 10.07272727, 10.15287356, 10.1045977,
                  9.92954545, 10.09767442, 10.05529412,  9.97195122,  9.93974359],
        'uncertainty': [0.49681835, 0.37318533, 0.34006171, 0.34478792, 0.3457126,
                        0.34266489, 0.35237775, 0.3567267 , 0.36549703, 0.35720442],
        'uncertainty [%]': [3.36294012, 3.36809864, 3.3760639 , 3.39596385, 3.42133962,
                            3.45096254, 3.4896921 , 3.54765061, 3.66525089, 3.59369858],
        'VAR_PORT_FFS': [0.24622222, 0.1385    , 0.11446281, 0.1166997 , 0.1161448 ,
                    0.11283574, 0.11741482, 0.11829758, 0.12160916, 0.12743261],
        'VAR_PORT_EM': [0.00060625, 0.00076729, 0.00117916, 0.00217901, 0.0033724 ,
                   0.00458349, 0.00675526, 0.00895636, 0.01197892, 0.00016239],
        'CH_FFS': [6,  8, 10, 12, 14, 16, 18, 20, 22, 24],
        'CH_EM': [6,  8, 10, 12, 14, 16, 18, 20, 22, 24],
        "R": [np.nan] * 10})
    nffs.effective_mass.integral.R = [np.nan] * len(nffs.effective_mass.integral.R)
    # No need to set llds as the R channels already allign absolute channels as well in this case
    pd.testing.assert_frame_equal(expected_df,
                                  nffs._per_unit_mass_ch(nffs.fission_fragment_spectrum.integrate(
                                                                    raw_integral=False,
                                                                    renormalize=False),
                                                         nffs.effective_mass.integral),
                                  check_exact=False, atol=0.00001)

def test_per_unit_mass(nffs):
    # R calibration
    pd.testing.assert_frame_equal(nffs._per_unit_mass_R(nffs.fission_fragment_spectrum.integrate(
                                                                    raw_integral=False,
                                                                    renormalize=False),
                                                        nffs.effective_mass.integral),
                                  nffs.per_unit_mass(raw_integral=False, renormalize=False),
                                  check_exact=False, atol=0.00001)
    # channel calibration
    nffs.effective_mass.integral.R = [np.nan] * len(nffs.effective_mass.integral.R)
    pd.testing.assert_frame_equal(nffs._per_unit_mass_ch(nffs.fission_fragment_spectrum.integrate(
                                                                        llds=nffs.effective_mass.integral.channel,
                                                                        r=False,
                                                                        raw_integral=False,
                                                                        renormalize=False),
                                                         nffs.effective_mass.integral),
                                  nffs.per_unit_mass(llds=nffs.effective_mass.integral.channel,
                                                     r=False,
                                                     raw_integral=False,
                                                     renormalize=False),
                                  check_exact=False, atol=0.00001)

def test_per_unit_mass_and_time(nffs):
    expected_df = pd.DataFrame({
        'CH_FFS': [6.,  8., 10., 12., 14., 16., 18., 20., 22., 24.],
        'CH_EM': [6.,  8., 10., 12., 14., 16., 18., 20., 22., 24.],
        'value': [1.47733333, 1.108     , 1.00727273, 1.01528736, 1.01045977,
                  0.99295455, 1.00976744, 1.00552941, 0.99719512, 0.99397436],
        'uncertainty': [0.04968184, 0.03731853, 0.03400617, 0.03447879, 0.03457126,
                        0.03426649, 0.03523777, 0.03567267, 0.0365497 , 0.03572044],
        'uncertainty [%]': [3.36294012, 3.36809864, 3.3760639 , 3.39596385, 3.42133962,
                            3.45096254, 3.4896921 , 3.54765061, 3.66525089, 3.59369858]})
    pd.testing.assert_frame_equal(expected_df,
                                   nffs.per_unit_mass_and_time(raw_integral=False, renormalize=False),
                                   check_exact=False, atol=0.00001)

def test_per_unit_mass_and_power(nffs):
    expected_df = pd.DataFrame({
        'CH_FFS': [6.,  8., 10., 12., 14., 16., 18., 20., 22., 24.],
        'CH_EM': [6.,  8., 10., 12., 14., 16., 18., 20., 22., 24.],
        'value': [0.14773333, 0.1108    , 0.10072727, 0.10152874, 0.10104598,
                  0.09929545, 0.10097674, 0.10055294, 0.09971951, 0.09939744],
        'uncertainty': [0.00681968, 0.00511892, 0.00465942, 0.00471126, 0.00470765,
                        0.00464774, 0.00475535, 0.00477873, 0.0048273 , 0.00475808],
        'uncertainty [%]': [4.61620691, 4.61996628, 4.62577642, 4.64032008, 4.65892314,
                            4.68072029, 4.70934719, 4.75245461, 4.84087431, 4.78692694]})
    pd.testing.assert_frame_equal(expected_df,
                                   nffs.per_unit_mass_and_power(raw_integral=False, renormalize=False),
                                   check_exact=False, atol=0.00001)

def test_plateau(nffs):
    expected_df = pd.Series({'value': 10.104598,
                             'uncertainty': 0.345713,
                             'uncertainty [%]': 3.421340,
                             'VAR_PORT_FFS': 0.1161448,
                             'VAR_PORT_EM': 0.0033724,
                             'CH_FFS': 14.000000,
                             'CH_EM': 14.000000,
                             'R': 0.35}, name=4).to_frame().T
    expected_df.index = ['value']
    pd.testing.assert_frame_equal(expected_df, nffs.plateau(raw_integral=False, renormalize=False))
    # check that sum(VAR_PORT) == uncertainty **2
    np.testing.assert_almost_equal(expected_df[[c for c in expected_df.columns if c.startswith("VAR_PORT")]].sum(axis=1).iloc[0],
                                   expected_df['uncertainty'].iloc[0] **2, decimal=5)

def test_process(nffs):
    expected_df = pd.DataFrame({'value': 0.010104597701149425,
                                'uncertainty': 0.0004707654400802892,
                                'uncertainty [%]': 4.658923135819038,
                                'VAR_PORT_FFS':  1.161448e-7,
                                'VAR_PORT_EM': 0.0033724e-7,
                                'VAR_PORT_PM': 1.0210289470207425e-07,
                                'VAR_PORT_t': 0.}, index=['value'])
    pd.testing.assert_frame_equal(expected_df,
                                  nffs.process(raw_integral=False,
                                               renormalize=False),
                                  check_exact=False, atol=0.00001)
    # check that sum(VAR_PORT) == uncertainty **2
    np.testing.assert_almost_equal(expected_df[[c for c in expected_df.columns if c.startswith("VAR_PORT")]].sum(axis=1).iloc[0],
                                   expected_df['uncertainty'].iloc[0] **2, decimal=5)
