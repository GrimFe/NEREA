import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from nerea.control_rod import *
from nerea.reaction_rate import ReactionRate
from nerea.classes import EffectiveDelayedParams
from nerea.utils import _make_df

@pytest.fixture
def edd():
    dd = EffectiveDelayedParams(lambda_i = pd.DataFrame({'value': 1, 'uncertainty': 1}, index=['value']),
                                beta_i = pd.DataFrame({'value': 1, 'uncertainty': 1}, index=['value']))
    return dd

@pytest.fixture
def edd_low_uncertainty():
    dd = EffectiveDelayedParams(lambda_i = pd.DataFrame({'value': 1, 'uncertainty': .01}, index=['value']),
                                beta_i = pd.DataFrame({'value': 1, 'uncertainty': .01}, index=['value']))
    return dd

@pytest.fixture
def periods():
    return {60: 400,
            200: 300,
            250: 250}

@pytest.fixture
def rrs(periods):
    counts = {}
    tend = 3000
    x = np.linspace(0, tend, 3001)
    for h, p in periods.items():
        counts[h] = np.exp(x[10:-10] / p)
        counts[h] = np.concatenate(([counts[h][0] - 20] * 10, counts[h], [counts[h][-1]] * 10))
        counts[h] = ReactionRate(pd.DataFrame({"Time": [datetime(2025, 5, 19, 19, 31, 20) + timedelta(seconds=i) for i in range(tend+1)],
                                                    "value": counts[h]}),
                                    start_time=datetime(2025, 5, 19, 19, 31, 20),
                                    campaign_id="TEST",
                                    experiment_id="TEST01",
                                    detector_id="DET",
                                    deposit_id="DEP")
    return counts

@pytest.fixture
def cr(rrs):
    cr = ControlRodCalibration(reaction_rates=rrs, critical_height=0, name="CR0")
    return cr

@pytest.fixture
def dnc(rrs):
    dnc = DifferentialNoCompensation(reaction_rates=rrs, critical_height=0, name="CR0")
    return dnc

@pytest.fixture
def inc(rrs):
    inc = IntegralNoCompensation(reaction_rates=rrs, critical_height=0, name="CR0")
    return inc

@pytest.fixture
def dc(rrs):
    dc = DifferentialCompensation(reaction_rates=rrs, critical_height=0, name="CR0")
    return dc

@pytest.fixture
def ic(rrs):
    ic = IntegralCompensation(reaction_rates=rrs, critical_height=0, name="CR0")
    return ic

def test_evaluate_integral_differential_cr():
    pd.testing.assert_frame_equal(
        evaluate_integral_differential_cr(1, 2, [1,2,3], np.ones((3,3))),
        _make_df(4 + 1/3, 1.8333333333333333))

def test_evaluate_integral_integral_cr():
    pd.testing.assert_frame_equal(
         evaluate_integral_integral_cr(1, 2, [1,2,3], np.ones((3,3))),
        _make_df(6, 3.))

def test_get_rhos(cr, edd, rrs):
    import warnings
    warnings.filterwarnings("ignore")

    rhos = cr._get_rhos(edd,
                        ac_kwargs={'smooth_kwargs': {'method': 'savgol_filter', 'polyorder': 3, 'window_length': 10}})
    pd.testing.assert_frame_equal(rhos.iloc[0].to_frame().T, pd.DataFrame({"value": 0., "uncertainty": 0., "uncertainty [%]": 0.,
                                                                           "h": 0., "VAR_PORT_T": 0., "VAR_PORT_B": 0.,
                                                                           "VAR_PORT_L": 0.}, index=['value']))
    ## Testing formatting: reactivity is tested in test_ReactionRate
    ref = rrs[60].dead_time_corrected().get_asymptotic_counts(smooth_kwargs={'method': 'savgol_filter',
                                                                             'polyorder': 3, 'window_length': 10}
                                                                             ).get_reactivity(edd).assign(h=60.)
    pd.testing.assert_frame_equal(rhos.iloc[1].to_frame().T, ref[["value", "uncertainty", "uncertainty [%]", "h",
                                                                  "VAR_PORT_T", "VAR_PORT_B", "VAR_PORT_L"]])
    ref = rrs[200].dead_time_corrected().get_asymptotic_counts(smooth_kwargs={'method': 'savgol_filter',
                                                                             'polyorder': 3, 'window_length': 10}
                                                                             ).get_reactivity(edd).assign(h=200.)
    pd.testing.assert_frame_equal(rhos.iloc[2].to_frame().T, ref[["value", "uncertainty", "uncertainty [%]", "h",
                                                                   "VAR_PORT_T", "VAR_PORT_B", "VAR_PORT_L"]])
    ref = rrs[250].dead_time_corrected().get_asymptotic_counts(smooth_kwargs={'method': 'savgol_filter',
                                                                             'polyorder': 3, 'window_length': 10}
                                                                             ).get_reactivity(edd).assign(h=250.)
    pd.testing.assert_frame_equal(rhos.iloc[3].to_frame().T, ref[["value", "uncertainty", "uncertainty [%]", "h",
                                                                   "VAR_PORT_T", "VAR_PORT_B", "VAR_PORT_L"]])
    warnings.filterwarnings("always")
    assert rhos.shape == (4 ,7)

def test_differential_curve_no_compensation(dnc, edd_low_uncertainty):
    import warnings
    warnings.filterwarnings("ignore")
    cols = ['value', 'uncertainty', 'VAR_PORT_T', 'VAR_PORT_B', 'VAR_PORT_L', 'h']
    # not testing uncertainty [%] as there differences with GitHub PC are more visible
    curve = dnc.get_reactivity_curve(edd_low_uncertainty,
                                     ac_kwargs={'smooth_kwargs': {'method': 'savgol_filter', 'polyorder': 3, 'window_length': 10}})
    pd.testing.assert_frame_equal(curve.iloc[0].to_frame().T[cols],
                                  pd.DataFrame({'value': 4.15633386e-05,
                                                'uncertainty': 6.85597030e-07,
                                                'uncertainty [%]': 1.64952348e+00,
                                                'VAR_PORT_T': 1.25401603e-13,
                                                'VAR_PORT_B': 1.72751112e-13,
                                                'VAR_PORT_L': 1.71890573e-13,
                                                'h': 3.000000e+01}, index=['value'])[cols])
    warnings.filterwarnings("always")
    assert curve.shape == (3, 7)

def test_integral_curve_no_compensation(inc, edd_low_uncertainty):
    import warnings
    warnings.filterwarnings("ignore")
    curve = inc.get_reactivity_curve(edd_low_uncertainty,
                                     ac_kwargs={'smooth_kwargs': {'method': 'savgol_filter', 'polyorder': 3, 'window_length': 10}})
    pd.testing.assert_frame_equal(curve,
                                  inc._get_rhos(edd_low_uncertainty,
                                               ac_kwargs={'smooth_kwargs': {'method': 'savgol_filter', 'polyorder': 3, 'window_length': 10}}))
    warnings.filterwarnings("always")

def test_differential_curve_compensation(dc, edd_low_uncertainty):
    import warnings
    warnings.filterwarnings("ignore")
    cols = ['value', 'uncertainty', 'VAR_PORT_T', 'VAR_PORT_B', 'VAR_PORT_L', 'h']
    # not testing uncertainty [%] as there differences with GitHub PC are more visible
    curve = dc.get_reactivity_curve(edd_low_uncertainty,
                                    ac_kwargs={'smooth_kwargs': {'method': 'savgol_filter', 'polyorder': 3, 'window_length': 10}})
    pd.testing.assert_frame_equal(curve.iloc[0].to_frame().T[cols],
                                  pd.DataFrame({'value': 4.15633386e-05,
                                                'uncertainty': 6.85597030e-07,
                                                'uncertainty [%]': 1.64952348e+00,
                                                'VAR_PORT_T': 1.25401603e-13,
                                                'VAR_PORT_B': 1.72751112e-13,
                                                'VAR_PORT_L': 1.71890573e-13,
                                                'h': 3.000000e+01}, index=['value'])[cols])
    pd.testing.assert_frame_equal(curve.iloc[1].to_frame().T[cols],
                                  pd.DataFrame({'value': 2.37329788e-05,
                                                'uncertainty': 3.67662065e-07,
                                                'uncertainty [%]': 1.54916105e+00,
                                                'VAR_PORT_T': 2.28982109e-14,
                                                'VAR_PORT_B': 5.63254285e-14,
                                                'VAR_PORT_L': 5.59517547e-14,
                                                'h': 1.300000e+02}, index=['value'])[cols])
    warnings.filterwarnings("always")
    assert curve.shape == (3, 7)

def test_integral_curve_compensation(ic, edd_low_uncertainty):
    import warnings
    warnings.filterwarnings("ignore")
    cols = ['value', 'uncertainty', 'VAR_PORT_T', 'VAR_PORT_B', 'VAR_PORT_L', 'h']
    # not testing uncertainty [%] as there differences with GitHub PC are more visible
    curve = ic.get_reactivity_curve(edd_low_uncertainty,
                                    ac_kwargs={'smooth_kwargs': {'method': 'savgol_filter', 'polyorder': 3, 'window_length': 10}})
    pd.testing.assert_frame_equal(curve.iloc[0].to_frame().T,
                                  pd.DataFrame({'value': 0., 'uncertainty': 0., 'uncertainty [%]': 0., 'VAR_PORT_T': 0., 'VAR_PORT_B': 0.,
                                                'VAR_PORT_L': 0., 'h': 0.}, index=['value']))
    pd.testing.assert_frame_equal(curve.iloc[1].to_frame().T[cols],
                                  pd.DataFrame({'value': 2.49380032e-03,
                                                'uncertainty': 4.11358218e-05,
                                                'uncertainty [%]': 1.64952348e+00,
                                                'VAR_PORT_T': 4.51445772e-10,
                                                'VAR_PORT_B': 6.21904003e-10,
                                                'VAR_PORT_L': 6.18806062e-10,
                                                'h': 6.000000e+01}, index=['value'])[cols])
    warnings.filterwarnings("always")
    assert curve.shape == (4, 7)

def test_get_reactivity_worth(inc, edd_low_uncertainty):
    cols = ['value', 'uncertainty', 'VAR_PORT_X1', 'VAR_PORT_X0']
    # not testing uncertainty [%] as there differences with GitHub PC are more visible
    target = pd.DataFrame({'value': 2.49450213e-03,
                           'uncertainty': 3.55392057e-05,
                           'uncertainty [%]': 1.42470136e+00,
                           'VAR_PORT_X1': 1.15874710e-09,
                           'VAR_PORT_X0': 1.04288039e-10},
                           index=['value'])
    rw = inc.get_reactivity_worth(60, 200, edd_low_uncertainty, order=1)
    pd.testing.assert_frame_equal(rw[cols], target[cols], atol=1e-6)
    ## With fitting I get the theoretical value within 1 pcm
    target_rw = inc.get_reactivity_curve(edd_low_uncertainty,
                                         ac_kwargs={'smooth_kwargs': {'method': 'savgol_filter',
                                                                      'polyorder': 3,
                                                                      'window_length': 10}}
                                        ).iloc[1,0]
    assert abs(2.49450213e-03 - target_rw)  <= 1e-5
