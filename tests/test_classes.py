import pytest

import pandas as pd

from nerea.classes import *
from nerea.constants import ATOMIC_MASS


def test_xs_copy():
    xs = Xs(pd.DataFrame({"value": [1., 2.], "uncertainty": [1., 2.]},
                         index=["A", "B"]),
            volume=2.,
            mass_normalized=True,
            volume_normalized=True)
    xsc = xs.copy()
    pd.testing.assert_frame_equal(xs.data, xsc.data)
    assert xsc.volume == xs.volume
    assert xsc.volume_normalized == xs.volume_normalized
    assert xsc.mass_normalized == xs.mass_normalized

def test_xs_normalized():
    start_data = pd.DataFrame({"value": [1., 2.], "uncertainty": [1., 2.]},
                         index=["U235", "U238"])
    xs = Xs(start_data,
            volume=1,
            mass_normalized=False,
            volume_normalized=False)
    mass_norm = pd.DataFrame({"value": [1. / ATOMIC_MASS.loc['U235', 'value'],
                                        2. / ATOMIC_MASS.loc['U238', 'value']],
                              "uncertainty": [1. / ATOMIC_MASS.loc['U235', 'value'],
                                             2. / ATOMIC_MASS.loc['U238', 'value']]},
                         index=["U235", "U238"])
    # full normalization volume = 1
    xsc = xs.copy().normalized
    pd.testing.assert_frame_equal(xsc.data[['value', 'uncertainty']], mass_norm)
    assert xsc.mass_normalized == True
    assert xsc.volume_normalized == True
    # full normalization volume = 2
    xs.volume = 2.
    xsc = xs.copy().normalized
    pd.testing.assert_frame_equal(xsc.data[['value', 'uncertainty']], mass_norm / 2.)
    assert xsc.mass_normalized == True
    assert xsc.volume_normalized == True
    # mass normalization only (even with volume = 2)
    xs.volume_normalized = True
    xsc = xs.copy().normalized
    pd.testing.assert_frame_equal(xsc.data[['value', 'uncertainty']], mass_norm)
    assert xsc.volume_normalized == True
    # no normalization when both normalizations are set to True
    xs.mass_normalized = True
    xsc = xs.copy().normalized
    pd.testing.assert_frame_equal(xsc.data[['value', 'uncertainty']], start_data)
