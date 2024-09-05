# NEREA 

NEREA (Neutron Energy-integrated Reactor Experiment Analysis) is a Python package designed for the analysis and evaluation of spectral indices and reaction rates from fission fragment spectra. The package provides a comprehensive set of tools for handling, processing, and analyzing nuclear data, specifically focusing on fission fragment spectra, effective mass, and reaction rates.

## Features

- **Fission Fragment Spectrum Analysis**: Tools to handle and analyze fission fragment spectra data.
- **Effective Mass Calculation**: Methods to compute effective mass from integral data.
- **Reaction Rate Computation**: Functions to calculate reaction rates using fission fragment spectra and effective mass.
- **Spectral Index Calculation**: Tools to compute spectral indices by comparing reaction rates.
- **C/E Calculation**: Compute C/E values from simulated and experimental data.

## 🔧 Installation

To install the package, clone the repository and use pip to install the dependencies:
```sh
git clone https://github.com/GrimFe/NEREA.git
cd nerea
pip install .
```

To import NEREA (without installation)
```
import sys
sys.path.append(r"path/to/nerea")
import nerea
```
## 🗺️ Structure

A schematic of the main structure of NEREA can be found ![here](https://github.com/GrimFe/NEREA/blob/main/img/Structure.jpg).
Some of the main features of NEREA are:
* DATA - dedicated to the interface with detector raw data and preprocessing:
   - `EffectiveMass` or [`EM`](https://github.com/GrimFe/NEREA/blob/main/nerea/EffectiveMass.py)
   - `FissionFragmentSpectrum` or [`FFS`](https://github.com/GrimFe/NEREA/blob/main/nerea/FissionFragmentSpectrum.py)
   - `FissionFragmentSpectra` or [`FFSa`](https://github.com/GrimFe/NEREA/blob/main/nerea/FissionFragmentSpectrum.py)
   - `ReactionRate` or [`RR`](https://github.com/GrimFe/NEREA/blob/main/nerea/ReactionRate.py)
   - `ReactionRates` or [`RRs`](https://github.com/GrimFe/NEREA/blob/main/nerea/ReactionRate.py)
* COMPUTABLES - objects created out of the DATA and related processing:
  - `NormalizedFissionFragmentSpectrum` or [`NFFS`](https://github.com/GrimFe/NEREA/blob/main/nerea/Experimental.py)
  - `SpectralIndex` or [`SI`](https://github.com/GrimFe/NEREA/blob/main/nerea/Experimental.py)
  - `Traverse` or [`Traverse`](https://github.com/GrimFe/NEREA/blob/main/nerea/Experimental.py)
* CALCULATED - objects crated from model outputs:
  - `CalculatedSpectralIndex` or [`CSI`](https://github.com/GrimFe/NEREA/blob/main/nerea/Calculated.py)
  - `CalculatedTraverse` or [`CT`](https://github.com/GrimFe/NEREA/blob/main/nerea/Calculated.py)
* C/E - comparison of calculations to experiments:
  - `CoverE` or [`CE`](https://github.com/GrimFe/NEREA/blob/main/nerea/Comparisons.py)
* Useful functions are stored in [`utils.py`](https://github.com/GrimFe/NEREA/blob/main/nerea/utils.py).

## 🎮 Examples

NEREA comes with examples in the docstrings and test that can serve a similar purpose.
