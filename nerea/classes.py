from dataclasses import dataclass

from typing import Self

import pandas as pd
import serpentTools as sts

from .utils import _make_df, ratio_v_u
from .constants import ATOMIC_MASS

__all__ = ["EffectiveDelayedParams", "Xs"]


@dataclass(slots=True)
class EffectiveDelayedParams:
    """
    ``nerea.EffectiveDelayedParams``
    ================================
    Class storing and pre-processing effective delayed
    parameters.

    Attributes
    ----------
    **lambda_i**: ``pd.DataFrame``
        precursor-group-wise effective decay constant.
    **beta_i**: ``pd.DataFrame``
        precursor-group-wise effective decay fraction.
    """
    lambda_i: pd.DataFrame
    beta_i: pd.DataFrame

    @classmethod
    def from_sts(cls, file: str) -> Self:
        """
        `nerea.EffectiveDelayedParams.from_sts()`
        -----------------------------------------
        Creates an instance using data extracted from a Serpent res.m.

        Parameters
        ----------
        **file** : ``str``
            The file path from which data will be read.

        Returns
        -------
        `nerea.EffectiveDelayedParams`
            An instance of the `nerea.EffectiveDelayedParams` class created from
            the specified file."""
        bi = sts.read(file).resdata['adjIfpImpBetaEff'].reshape(-1, 2)[1:, :]
        li = sts.read(file).resdata['adjIfpImpLambda'].reshape(-1, 2)[1:, :]
        bi = _make_df(bi[:, 0], bi[:, 1] * bi[:, 0])
        li = _make_df(li[:, 0], li[:, 1] * li[:, 0])
        return cls(li, bi)


@dataclass(slots=True)
class Xs:
    """
    ``nerea.Xs``
    ============
    Class storing one group cross section data.

    Attributes
    ----------
    **data**: ``pd.DataFrame``
        data frame with cross section data (index is nuclide identifier).
    **mass_normalized**: ``bool``, optional
        whether the cross section is mass-normalized.
        Default is `False`.
    **volume_normalized**: ``bool``, optional
        whether the cross section is volume-normalized.
        Default is `False`.
    **volume**: ``float``, optional
        volume for volume normalization. Default is `1.0`."""
    data : pd.DataFrame
    mass_normalized : bool=False
    volume_normalized : bool=False
    volume: float = 1.

    def copy(self) -> Self:
        """
        `nerea.Xs.copy()`
        -----------------

        Copies the `nerea.Xs` isntance.

        Returns
        -------
        `nerea.Xs`"""
        return self.__class__(self.data.copy(),
                              self.mass_normalized,
                              self.volume_normalized,
                              self.volume)

    @classmethod
    def from_file(cls,
                  file: str,
                  read: dict[str, str],
                  *args, **kwargs) -> Self:
        """
        `nerea.Xs.from_file()`
        ----------------------
        Create Xs object from serpent detector output file.

        Parameters
        ----------
        **file**: ``str``
            Serpent output file path from which data will be read.
        **read**: ``dict[str, str]``
            The nuclide (`key`) associated with each
            detector (`value`).
        **args, **kwargs
            Additional arguments for instance creation
            
            - **mass_normalized** (``bool``, optional), whether the cross section is mass-normalized.
            - **volume_normalized** (``bool``, optional), whether the cross section is volume-normalized.
            - **volume** (``float``, optional), volume for volume normalization.

        Returns
        -------
        `nerea.Xs`"""
        data = pd.DataFrame({n: sts.read(file).detectors[d].bins[0][-2:]
                             for n, d in read.items()}).T
        data.columns = ['value', 'uncertainty']
        data.index.name = 'nuclide'
        # uncertainy is absolute
        data.uncertainty = data.uncertainty * data.value
        return cls(data, *args, **kwargs)
    
    @property
    def normalized(self) -> Self:
        """
        `nerea.Xs.normalized()`
        -----------------------
        Normalizes the cross section data per unit
        volume and mass.

        Returns
        -------
        `nerea.Xs`"""
        if not self.volume_normalized:
            self.data /= self.volume
        if not self.mass_normalized:
            idx = self.data.index.copy()
            self.data = _make_df(*ratio_v_u(self.data, ATOMIC_MASS),
                                 relative=False)[['value', 'uncertainty']
                                                 ].dropna()
            self.data.index = idx
        self.volume_normalized = True
        self.mass_normalized = True
        return self
