from dataclasses import dataclass

from typing import Self

import pandas as pd
import serpentTools as sts

from .utils import _make_df, ratio_v_u
from .constants import ATOMIC_MASS

__all__ = ["EffectiveDelayedParams", "Xs"]


@dataclass(slots=True)
class EffectiveDelayedParams:
    lambda_i: pd.DataFrame
    beta_i: pd.DataFrame

    @classmethod
    def from_sts(cls, file: str) -> Self:
        """
        Creates an instance using data extracted from a Serpent res.m.

        Parameters
        ----------
        file : str
            The file path from which data will be read.

        Returns
        -------
        EffectiveDelayedParams
            An instance of the `EffectiveDelayedParams` class created from
            the specified file.
        """
        bi = sts.read(file).resdata['adjIfpImpBetaEff'].reshape(-1, 2)[1:, :]
        li = sts.read(file).resdata['adjIfpImpLambda'].reshape(-1, 2)[1:, :]
        bi = _make_df(bi[:, 0], bi[:, 1] * bi[:, 0])
        li = _make_df(li[:, 0], li[:, 1] * li[:, 0])
        return cls(li, bi)


@dataclass(slots=True)
class Xs:
    data : pd.DataFrame
    mass_normalized : bool=False
    volume_normalized : bool=False
    volume: float = 1.

    def copy(self):
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
        Create Xs object from serpent file.

        Parameters
        ----------
        file : str
            The file path from which data will be read.
        read : dict[str, str]
            The nuclide `key` associated with each
            detector `value`.
        """
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
        Normalizes the cross section data per unit
        volume and mass.
        """
        if not self.volume_normalized:
            self.data /= self.volume
        if not self.mass_normalized:
            idx = self.data.index.copy()
            self.data = _make_df(*ratio_v_u(self.data, ATOMIC_MASS)).dropna()
            self.data.index = idx
        self.volume_normalized = True
        self.mass_normalized = True
        return self
