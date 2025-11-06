<h1 align="center"
[![codecov](https://codecov.io/gh/GrimFe/NEREA/branch/v0.1/graph/badge.svg)](https://codecov.io/gh/GrimFe/NEREA)
[![Tests](https://github.com/GrimFe/NEREA/actions/workflows/unit_test.yml/badge.svg)](https://github.com/GrimFe/NEREA/actions/workflows/unit_test.yml)
[![PyPI version](https://img.shields.io/pypi/v/nerea.svg)](https://pypi.org/project/nerea/)

![here](https://github.com/GrimFe/NEREA/blob/v0.1/img/logo.png)
</h1>

<h1 align="center" style="font-family:simplifica">NEREA</h1>

NEREA (Neutron Energy-integrated Reactor Experiment Analysis) is a Python package designed for the analysis and evaluation of spectral indices and reaction rates from fission fragment spectra. The package provides a comprehensive set of tools for handling, processing, and analyzing nuclear data, specifically focusing on fission fragment spectra, effective mass, and reaction rates.

## Main features

- **Pulse Height Spectrum Analysis**: Tools to handle and analyze pulse height spectrum data.
- **Effective Mass Calculation**: Methods to compute effective mass from integral data (i.e., fission chamber calibration).
- **Reaction Rate Computation**: Functions to calculate reaction rates per unit power and mass using pulse height spectra and effective masses.
- **Spectral Index Calculation**: Tools to compute spectral indices by comparing reaction rates.
- **Control Rod Calibration**: Tools to compute the control rod calibration curve and measure reactivity worth.
- **Fission Traverse Processing**: Tools to process fission traverse experiments.
- **C/E Calculation**: Compute C/E values from simulated and experimental data.

## 🔧 Installation

To install NEREA with pip:
```sh
pip install nerea
```
Or from Test PyPI:
```sh
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple nerea
```

Otherwise, clone the repository and use pip to install the dependencies:
```sh
git clone https://github.com/GrimFe/NEREA.git
cd nerea
pip install .
```

To import NEREA (without installation)
```
import sys
sys.path.insert(0, r"path/to/nerea")
import nerea
```
## 🗺️ Structure

A schematic of the main structure of NEREA: ![here](https://github.com/GrimFe/NEREA/blob/v0.1/img/Schematic_v0.1.0.png)
Some of the main features of NEREA are:
* DATA - dedicated to the interface with detector raw data and preprocessing:
   - `EffectiveMass` or [`EM`](https://github.com/GrimFe/NEREA/blob/main/nerea/effective_mass.py)
   - `PulseHeightSpectrum` or [`PHS`](https://github.com/GrimFe/NEREA/blob/main/nerea/pulse_height_spectrum.py)
   - `PulseHeightSpectra` or [`FFSa`](https://github.com/GrimFe/NEREA/blob/main/nerea/pulse_height_spectrum.py)
   - `CountRate` or [`RR`](https://github.com/GrimFe/NEREA/blob/main/nerea/count_rate.py)
   - `CountRates` or [`RRs`](https://github.com/GrimFe/NEREA/blob/main/nerea/count_rates.py)
* Experiemntal - objects created out of the DATA and related processing:
  - `NormalizedPulseHeightSpectrum` or [`NPHS`](https://github.com/GrimFe/NEREA/blob/main/nerea/experimental.py)
  - `SpectralIndex` or [`SI`](https://github.com/GrimFe/NEREA/blob/main/nerea/experimental.py)
  - `Traverse` or [`Traverse`](https://github.com/GrimFe/NEREA/blob/main/nerea/experimental.py)
  - `ControlRod` or [`CR`](https://github.com/GrimFe/NEREA/blob/main/nerea/control_rod.py).
* CALCULATED - objects crated from model outputs:
  - `CalculatedSpectralIndex` or [`CSI`](https://github.com/GrimFe/NEREA/blob/main/nerea/calculated.py)
  - `CalculatedTraverse` or [`CT`](https://github.com/GrimFe/NEREA/blob/main/nerea/calculated.py)
* C/E - comparison of calculations to experiments:
  - `CoverE` or [`CE`](https://github.com/GrimFe/NEREA/blob/main/nerea/comparisons.py)
  - `CoverC` or [`CC`](https://github.com/GrimFe/NEREA/blob/main/nerea/comparisons.py)
  - `EoverE` or [`EE`](https://github.com/GrimFe/NEREA/blob/main/nerea/comparisons.py)
* Useful functions are stored in
  - [`utils.py`](https://github.com/GrimFe/NEREA/blob/main/nerea/utils.py)
  - [`functions.py`](https://github.com/GrimFe/NEREA/blob/main/nerea/utils.py).

## 💡 Examples and Concept

NEREA comes with examples in the docstrings and [test](https://github.com/GrimFe/NEREA/blob/main/tests) that can serve a similar purpose.

NEREA outputs are normally formatted as `pd.DataFrame` by the [`_make_df()` function](https://github.com/GrimFe/NEREA/blob/main/nerea/utils.py).
A standard workflow for spectral index processing is outlined in the following (slightly simplified with respect to some options), for control rod calibration and fission traverse measurement we refer you to docstrings and tests. Any keyword argument to customize your processing beyond [default options](https://github.com/GrimFe/NEREA/blob/main/nerea/defaults.py) can be passed at any stage of the Normalized PHS processing. The SpectralIndex.processing() can take numerator and denominator kwargs.
<pre> ```python
import nerea
import pandas as pd

# FISSION CHAMBER CALIBRATION
# Composition and monitor for first detector
composition1 = pd.DataFrame({"value": 100, "uncertainty": 0}, index=["U235"])
pm_cal1 = nerea.CountRate.from_ascii("PM1.txt", detector=1, deposit_id="U235")
em1 = (
    nerea.PulseHeightSpectrum.from_formatted_TKA("cal1.TKA").calibrate(
        nerea.constants.KNBS,
        composition1,
        monitor=pm_cal1,
        one_group_xs=nerea.Xs(nerea.constants.XS_FAST)
    )
)

# Composition and monitor for second detector
composition2 = pd.DataFrame({"value": 100, "uncertainty": 0}, index=["U238"])
pm_cal2 = nerea.CountRate.from_ascii("PM2.txt", detector=2, deposit_id="U235")
em2 = (
    nerea.PulseHeightSpectrum.from_formatted_TKA("cal2.TKA").calibrate(
        nerea.constants.KNBS,
        composition2,
        monitor=pm_cal2,
        one_group_xs=nerea.Xs(nerea.constants.XS_FAST)
    )
)

# SPECTRAL INDEX PROCESSING
# Measured spectra
phs1 = nerea.PulseHeightSpectrum.from_formatted_TKA("meas1.TKA")
phs2 = nerea.PulseHeightSpectrum.from_formatted_TKA("meas2.TKA")

# Count rates for measurements
pm1 = nerea.CountRate.from_ascii("M1.txt", detector=1, deposit_id="U235")
pm2 = nerea.CountRate.from_ascii("M2.txt", detector=1, deposit_id="U235")

# Compute spectral index
si = nerea.SpectralIndex(
    nerea.NormalizedPulseHeightSpectrum(phs1, em1, pm1),
    nerea.NormalizedPulseHeightSpectrum(phs2, em2, pm2)
)

# C/E calculation
# Calculated spectral index from model detectors
c = nerea.CalculatedSpectralIndex.from_detectors("model_det0.m", ["d1", "d2"])

# C/E - 1 [%] calculation
ce = nerea.CoverE(c, si).process(minus_one_percent=True)

# ce is a pandas.DataFrame with columns 'value' and 'uncertainty'

``` </pre>


## 🤝 Acknowledgments
NEREA was conceived and developed as a part of the PhD thesis on *Neutron Data Benchmarking at the VENUS-F zero power reactor for MYRRHA* in the framework of a collaboration between [SCK CEN](https://www.sckcen.be) and [ULB](http://www.ulb.ac.be), supported financially by Association Vinçotte Nuclear (AVN).

The authors thank Agnese Carlotti for designing the logo of NEREA.

## 📋 Reference
### The NEREA package
- F. Grimaldi, F. Di Corce, A. Krása, J. Wagemans, G. Vittiglio, [*The NEREA Python Package for Integral Experiment Analysis: a Fission Chamber Calibration Case Study*]

### Spectral Indices (first use) 
- F. Grimaldi F. Di Croce, A. Krása, G. de Izarra, L. Barbot, P. Blaise, P.E. Labeau, L. Fiorito. G. Vittiglio, J. Wagemans, [*The CoRREx neutron spectrum filtering campaign at VENUS-F for calculation-to-experiment discrepancy interpretation*](https://doi.org/10.1016/j.anucene.2025.111425),     Annals of Nuclear Energy, Volume 219, 2025, 111425, ISSN 0306-4549.

## 🌍 Publications
This is a not complete list of publications featuring NEREA.
- F. Grimaldi F. Di Croce, A. Krása, G. de Izarra, L. Barbot, P. Blaise, P.E. Labeau, L. Fiorito. G. Vittiglio, J. Wagemans, [*The CoRREx neutron spectrum filtering campaign at VENUS-F for calculation-to-experiment discrepancy interpretation*](https://doi.org/10.1016/j.anucene.2025.111425),     Annals of Nuclear Energy, Volume 219, 2025, 111425, ISSN 0306-4549.
