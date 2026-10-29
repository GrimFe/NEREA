import pytest

import pandas as pd

from nerea.classes import *
from nerea.constants import ATOMIC_MASS


@pytest.fixture
def base_test_data():
    """Fixture che fornisce i dati iniziali di partenza e il DataFrame atteso per la massa."""
    start_data = pd.DataFrame(
        {"value": [1.0, 2.0], "uncertainty": [1.0, 2.0]}, 
        index=["U235", "U238"]
    )
    
    mass_norm = pd.DataFrame(
        {
            "value": [1.0 / ATOMIC_MASS.loc['U235', 'value'], 2.0 / ATOMIC_MASS.loc['U238', 'value']],
            "uncertainty": [1.0 / ATOMIC_MASS.loc['U235', 'value'], 2.0 / ATOMIC_MASS.loc['U238', 'value']]
        },
        index=["U235", "U238"]
    )
    
    return start_data, mass_norm

def test_xs_copy():
    xs = Xs(pd.DataFrame({"value": [1., 2.], "uncertainty": [1., 2.]},
                         index=["A", "B"]),
            volume=2.,
            atomic_mass_normalized=True,
            volume_normalized=True)
    xsc = xs.copy()
    pd.testing.assert_frame_equal(xs.data, xsc.data)
    assert xsc.volume == xs.volume
    assert xsc.volume_normalized == xs.volume_normalized
    assert xsc.atomic_mass_normalized == xs.atomic_mass_normalized

def test_xs_per_unit_volume(base_test_data):
    """Verifica la normalizzazione per unità di volume isolata."""
    start_data, _ = base_test_data
    
    # Test con volume = 2
    xs = Xs(start_data.copy(), volume=2.0, volume_normalized=False)
    xsc = xs.per_unit_volume
    
    pd.testing.assert_frame_equal(xsc.data[['value', 'uncertainty']], start_data / 2.0)
    assert xsc.volume_normalized is True

    # Verifica che non ri-normalizzi se già impostato su True
    xsc_already_norm = xsc.per_unit_volume
    pd.testing.assert_frame_equal(xsc_already_norm.data[['value', 'uncertainty']], start_data / 2.0)

def test_xs_per_unit_mass(base_test_data):
    """Verifica la normalizzazione per unità di massa atomica isolata."""
    start_data, mass_norm = base_test_data
    
    xs = Xs(start_data.copy(), atomic_mass_normalized=False)
    xsc = xs.per_unit_mass
    
    pd.testing.assert_frame_equal(xsc.data[['value', 'uncertainty']], mass_norm)
    assert xsc.atomic_mass_normalized is True

    # Verifica che non ri-normalizzi se già impostato su True
    xs_true = Xs(start_data.copy(), atomic_mass_normalized=True)
    xsc_true = xs_true.per_unit_mass
    pd.testing.assert_frame_equal(xsc_true.data[['value', 'uncertainty']], start_data)

def test_xs_normalized(base_test_data):
    """Verifica la proprietà combinata .normalized (volume + massa)."""
    start_data, mass_norm = base_test_data
    
    # Caso 1: Full normalization con volume = 1
    xs1 = Xs(start_data.copy(), volume=1.0, atomic_mass_normalized=False, volume_normalized=False)
    xsc1 = xs1.normalized
    pd.testing.assert_frame_equal(xsc1.data[['value', 'uncertainty']], mass_norm)
    assert xsc1.atomic_mass_normalized is True
    assert xsc1.volume_normalized is True
    
    # Caso 2: Full normalization con volume = 2
    xs2 = Xs(start_data.copy(), volume=2.0, atomic_mass_normalized=False, volume_normalized=False)
    xsc2 = xs2.normalized
    pd.testing.assert_frame_equal(xsc2.data[['value', 'uncertainty']], mass_norm / 2.0)
    assert xsc2.atomic_mass_normalized is True
    assert xsc2.volume_normalized is True

    # Caso 3: Solo normalizzazione di massa (volume già normalizzato)
    xs3 = Xs(start_data.copy(), volume=2.0, atomic_mass_normalized=False, volume_normalized=True)
    xsc3 = xs3.normalized
    pd.testing.assert_frame_equal(xsc3.data[['value', 'uncertainty']], mass_norm)
    assert xsc3.atomic_mass_normalized is True
    assert xsc3.volume_normalized is True
