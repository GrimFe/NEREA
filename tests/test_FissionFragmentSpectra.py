import pytest
import datetime
import pandas as pd
from PSICHE.FissionFragmentSpectrum import FissionFragmentSpectrum, FissionFragmentSpectra

@pytest.fixture
def sample_spectrum_data():
    # Sample data for testing
    data = {
        "channel": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "counts": [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 50, 25, 10, 5, 3, 1, 0, 0, 0]
    }
    return data

@pytest.fixture
def sample_spectrum_data2():
    # Sample data for testing
    data = {
        "channel": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "counts": [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 5, 3, 1, 1, 0, 0, 0, 0, 0]
    }
    return data

@pytest.fixture
def sample_spectrum_1(sample_spectrum_data):
    return FissionFragmentSpectrum(start_time=datetime.datetime(2024, 5, 18, 20, 30, 15),
                                   life_time=10,
                                   real_time=10,
                                   data=pd.DataFrame(sample_spectrum_data),
                                   campaign_id="campaign",
                                   experiment_id="experiment",
                                   detector_id="detector",
                                   deposit_id="deposit", 
                                   location_id="location",
                                   measurement_id="measurement")

@pytest.fixture
def sample_spectrum_2(sample_spectrum_data2):
    return FissionFragmentSpectrum(start_time=datetime.datetime(2024, 5, 18, 20, 30, 15),
                                   life_time=10,
                                   real_time=10,
                                   data=pd.DataFrame(sample_spectrum_data2),
                                   campaign_id="campaign",
                                   experiment_id="experiment",
                                   detector_id="detector",
                                   deposit_id="deposit", 
                                   location_id="location",
                                   measurement_id="measurement")


@pytest.fixture
def sample_ffsa(sample_spectrum_1, sample_spectrum_2):
    data = []
    for f in [sample_spectrum_1, sample_spectrum_2]:
        data.append(f.data.set_index('channel'))
    dct = {
            'start_time': sample_spectrum_1.start_time,
            'life_time': sample_spectrum_1.life_time,
            'real_time': sample_spectrum_1.real_time,
            'campaign_id': sample_spectrum_1.campaign_id,
            'experiment_id': sample_spectrum_1.experiment_id,
            'detector_id': sample_spectrum_1.detector_id,
            'deposit_id': sample_spectrum_1.deposit_id,
            'location_id': sample_spectrum_1.location_id,
            'measurement_id': sample_spectrum_1.measurement_id,
            'data': sum(data).reset_index(),
            'spectra': [sample_spectrum_1, sample_spectrum_2]
        }
    return FissionFragmentSpectra(**dct)

def test_best(sample_ffsa, sample_spectrum_1):
    assert sample_ffsa.best is sample_spectrum_1