
import pytest
import pandas as pd
from PSICHE.EffectiveMass import EffectiveMass

@pytest.fixture
def sample_effective_mass():
    deposit_id = "deposit_1"
    detector_id = "detector_1"
    integral = pd.DataFrame({'channel': [15, 20, 30], 'value': [10, 20, 30], 'uncertainty': [1, 2, 3]})
    bins = 4096
    return EffectiveMass(deposit_id, detector_id, integral, bins)

def test_EffectiveMass_properties(sample_effective_mass):
    assert isinstance(sample_effective_mass.deposit_id, str)
    assert isinstance(sample_effective_mass.detector_id, str)
    assert isinstance(sample_effective_mass.integral, pd.DataFrame)
    assert isinstance(sample_effective_mass.bins, int)

def test_EffectiveMass_R_channel_property(sample_effective_mass):
    assert isinstance(sample_effective_mass.R_channel, int)
    assert sample_effective_mass.R_channel == 100

# def test_EffectiveMass_from_xls_classmethod():
#     file_path = r"tests/Deposit_Detector.xlsx"
#     sample_data = {
#         "channel": [1, 2, 3, 4, 5],
#         "value": [10, 20, 30, 40, 50]
#     }
#     sample_df = pd.DataFrame(sample_data)
#     sample_df.to_excel(file_path, index=False)
    
#     mass = EffectiveMass.from_xls(str(file_path))
#     pd.testing.assert_frame_equal(mass.integral, sample_df)
