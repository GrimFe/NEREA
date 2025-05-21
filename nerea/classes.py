from dataclasses import dataclass

from typing import Self

import pandas as pd
import serpentTools as sts

from .utils import _make_df

__all__ = ["EffectiveDelayedParams"]


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
