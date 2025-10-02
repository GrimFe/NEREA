from collections.abc import Iterable
from typing import Self
from dataclasses import dataclass

import serpentTools as sts
import pandas as pd

from .utils import _make_df, ratio_v_u
from .constants import ATOMIC_MASS

__all__ = ['_Calculated',
           'CalculatedSpectralIndex',
           'CalculatedTraverse']

@dataclass(slots=True)
class _Calculated:
    def calculate(self) -> None:
        """
        Placeholder for inheriting classes.
        """
        return None

@dataclass(slots=True)
class CalculatedSpectralIndex(_Calculated):
    data: pd.DataFrame
    model_id: str
    deposit_ids: list[str]  # 0: num, 1: den

    @classmethod
    def from_sts(cls, file: str, detector_name: str, **kwargs) -> Self:
        """
        Creates an instance using data extracted from a Serpent det.m
        file for a specific detector.

        Parameters
        ----------
        file : str
            The file path from which data will be read.
        detector_name : str
            The name of the detector from which data will be extracted.

        Returns
        -------
        CalculatedSpectralIndex
            An instance of the `CalculatedSpectralIndex` class created
            from the specified file.

        Examples
        --------
        >>> c_instance = CalculatedSpectralIndex.from_sts('file_det0.m',
                                                    'SI_detector', model_id='Model1')
        """
        v, u = sts.read(file).detectors[detector_name].bins[0][-2:]
        n, d = kwargs['deposit_ids'][0], kwargs['deposit_ids'][1]
        mass_norm = ATOMIC_MASS.loc[d]['value'] / ATOMIC_MASS.loc[n]['value']
        # Serpent detector uncertainty is relative
        kwargs['data'] = _make_df(v * mass_norm, u * v * mass_norm
                                  ).assign(VAR_PORT_C_n=None,
                                           VAR_PORT_C_d=None)
        return cls(**kwargs)

    @classmethod
    def from_sts_detectors(cls, file: str, detector_names: dict[str, str], **kwargs) -> Self:
        """
        Creates an instance using data extracted from a Serpent det.m
        file for multiple detectors.

        Parameters
        ----------
        file : str
            The file path from which data will be read.
        detector_names : dict[str,str]
            Keys can be `numerator` or `denominator`, while values are the names of
            the detectors from which data will be extracted.

        Returns
        -------
        CalculatedSpectralIndex
            An instance of the `CalculatedSpectralIndex` class created from
            the specified file.

        Examples
        --------
        >>> c_instance = CalculatedSpectralIndex.from_sts_detectors('file.det', 
                                                    ['detector1', 'detector2'], model_id='Model1')
        """
        v1, u1_ = sts.read(file).detectors[detector_names['numerator']].bins[0][-2:]
        v2, u2_ = sts.read(file).detectors[detector_names['denominator']].bins[0][-2:]
        # Serpent detector uncertainty is relative
        u1, u2 = u1_ * v1, u2_ * v2
        n, d = kwargs['deposit_ids'][0], kwargs['deposit_ids'][1]
        v, u = ratio_v_u(_make_df(v=v1 / ATOMIC_MASS.loc[n]['value'],
                                  u=u1 / ATOMIC_MASS.loc[n]['value']),
                         _make_df(v=v2 / ATOMIC_MASS.loc[d]['value'],
                                  u=u2 / ATOMIC_MASS.loc[d]['value']))
        S1, S2= 1 / v2, v1 / v2 **2
        kwargs['data'] = _make_df(v, u).assign(VAR_PORT_C_n=(S1 * u1) **2,
                                               VAR_PORT_C_d=(S2 * u2) **2)
        return cls(**kwargs)

    def calculate(self):
        """
        Computes the C value. Alias for self.data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the C value.

        Examples
        --------
        >>> c_instance = CalculatedSpectralIndex(data=pd.DataFrame({'value': [0.5]}),
                            model_id='Model1', deposit_ids='Dep1')
        >>> c_instance.compute()
           value
        0    0.5
        """
        return self.data

@dataclass(slots=True)
class CalculatedTraverse(_Calculated):
    data: pd.DataFrame
    model_id: str
    deposit_id: str

    @classmethod
    def from_sts(cls, file: str, detector_names: list[str], **kwargs) -> Self:
        """
        Creates an instance using data extracted from a Serpent det.m
        file for multiple detectors.

        Parameters
        ----------
        file : str
            The file path from which data will be read.
        detector_names : Iterable[str]
            The names of the detectors from which data will be extracted.
        normalization : str, optional
            The detector name to normalize the traveres to.
            Defaults to None, normalizing to the one with the highest counts.

        Returns
        -------
        CalculatedTraverse
            An instance of the `CalculatedTraverse` class created from
            the specified file.
        """
        out = []
        for d in detector_names:
            v, u = sts.read(file).detectors[d].bins[0][-2:]
            out.append(_make_df(v, u * v).assign(traverse=d))
        out = pd.concat(out)
        return cls(data=out, **kwargs)

    def calculate(self, normalization: str=None):
        """
        Computes the C value. Normalized self.data.

        Parameters
        ----------
        normalization : str, optional
            The detector name to normalize the traveres to.
            Defaults to None, normalizing to the one with the highest counts.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the C value.

        """
        out = []
        max_d = self.data.query("value == @self.data.value.max()").traverse.iloc[0]
        norm_d = max_d if normalization is None else normalization
        den = self.data.query('traverse == @norm_d')
        den = _make_df(den.value.iloc[0], den.uncertainty.iloc[0])
        num = _make_df(self.data.value, self.data.uncertainty)
        out = _make_df(*ratio_v_u(num, den)).assign(traverse=self.data.traverse.values)
        return out.reset_index(drop=True)
