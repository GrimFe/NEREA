# import pytest
# from PSICHE.ReactionRate import ReactionRate, AverageReactionRate
# from PSICHE.FissionFragmentSpectrum import FissionFragmentSpectrum, FissionFragmentSpectra
# from PSICHE.EffectiveMass import EffectiveMass
# from PSICHE.PowerMonitor import PowerMonitor
# import pandas as pd

# def test_average_reaction_rate_initialization(fission_fragment_spectrum, effective_mass, power_monitor):
#     spectra = FissionFragmentSpectra([fission_fragment_spectrum, fission_fragment_spectrum])
#     avg_reaction_rate = AverageReactionRate(spectra, effective_mass, power_monitor)
#     assert avg_reaction_rate.fission_fragment_spectra == spectra
#     assert avg_reaction_rate.effective_mass == effective_mass
#     assert avg_reaction_rate.power_monitor == power_monitor

# def test_average_reaction_rate_campaign_id(fission_fragment_spectrum, effective_mass, power_monitor):
#     spectra = FissionFragmentSpectra([fission_fragment_spectrum, fission_fragment_spectrum])
#     avg_reaction_rate = AverageReactionRate(spectra, effective_mass, power_monitor)
#     assert avg_reaction_rate.campaign_id == "A"

# def test_average_reaction_rate_experiment_id(fission_fragment_spectrum, effective_mass, power_monitor):
#     spectra = FissionFragmentSpectra([fission_fragment_spectrum, fission_fragment_spectrum])
#     avg_reaction_rate = AverageReactionRate(spectra, effective_mass, power_monitor)
#     assert avg_reaction_rate.experiment_id == "B"

# def test_average_reaction_rate_location_id(fission_fragment_spectrum, effective_mass, power_monitor):
#     spectra = FissionFragmentSpectra([fission_fragment_spectrum, fission_fragment_spectrum])
#     avg_reaction_rate = AverageReactionRate(spectra, effective_mass, power_monitor)
#     assert avg_reaction_rate.location_id == "E"

# def test_average_reaction_rate_deposit_id(fission_fragment_spectrum, effective_mass, power_monitor):
#     spectra = FissionFragmentSpectra([fission_fragment_spectrum, fission_fragment_spectrum])
#     avg_reaction_rate = AverageReactionRate(spectra, effective_mass, power_monitor)
#     assert avg_reaction_rate.deposit_id == "D"

# def test_average_reaction_rate_compute(fission_fragment_spectrum, effective_mass, power_monitor):
#     spectra = FissionFragmentSpectra([fission_fragment_spectrum, fission_fragment_spectrum])
#     avg_reaction_rate = AverageReactionRate(spectra, effective_mass, power_monitor)
#     result = avg_reaction_rate.compute()

#     assert 'value' in result
#     assert 'uncertainty' in result
