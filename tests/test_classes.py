import pytest

import pandas as pd

from nerea.classes import *
from nerea.constants import ATOMIC_MASS


def test_xs_copy():
    xs = Xs(pd.DataFrame({"value": [1, 2], "uncertainty": [1, 2]},
                         index=["A", "B"]),
            volume=2,
            mass_normalized=True,
            volume_normalized=True)
    xsc = xs.copy()
    pd.testing.assert_frame_equal(xs.data, xsc.data)
    assert xsc.volume == xs.volume
    assert xsc.volume_normalized == xs.volume_normalized
    assert xsc.mass_normalized == xs.mass_normalized

def test_xs_normalized():
    start_data = pd.DataFrame({"value": [1, 2], "uncertainty": [1, 2]},
                         index=["U235", "U238"])
    xs = Xs(start_data,
            volume=1,
            mass_normalized=False,
            volume_normalized=False)
    mass_norm = pd.DataFrame({"value": [1 / ATOMIC_MASS.loc['U235', 'value'],
                                        2 / ATOMIC_MASS.loc['U238', 'value']],
                              "uncertainty": [1 / ATOMIC_MASS.loc['U235', 'value'],
                                             2 / ATOMIC_MASS.loc['U238', 'value']]},
                         index=["U235", "U238"])
    pd.testing.assert_frame_equal(xs.data, mass_norm)
    assert xs.mass_normalized == True
    assert xs.volume_normalized == True
    xs.volume =2
    pd.testing.assert_frame_equal(xs.data, mass_norm / 2)
    assert xs.mass_normalized == True
    assert xs.volume_normalized == True
    xs.volume_normalized = True
    pd.testing.assert_frame_equal(xs.data, mass_norm)
    assert xs.volume_normalized == True
    xs.mass_normalized = True
    pd.testing.assert_frame_equal(xs.data, start_data)
