from collections.abc import Iterable
from dataclasses import dataclass

import serpentTools as sts
import numpy as np
import pandas as pd

from PSICHE import SpectralIndex
from PSICHE.utils import ratio_v_u, _make_df


__all__ = ['C', 'CoverE']

@dataclass
class C:
    data: pd.DataFrame
    model_id: str
    deposit_ids: list[str]

    @classmethod
    def from_sts(cls, file: str, detector_name: str, **kwargs):
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
        C
            An instance of the `C` class created from the specified file.

        Examples
        --------
        >>> c_instance = C.from_sts('file.det', 'SI_detector', model_id='Model1')
        """
        ## works with relative uncertainties for the moment.
        ## Shall be homogenized with the rest of the API
        v, u = sts.read(file).detectors[detector_name].bins[0][-2:]
        kwargs['data'] = _make_df(v, u)
        return cls(**kwargs)

    @classmethod
    def from_sts_detectors(cls, file: str, detector_names: Iterable[str], **kwargs):
        """
        Creates an instance using data extracted from a Serpent det.m
        file for multiple detectors.

        Parameters
        ----------
        file : str
            The file path from which data will be read.
        detector_names : Iterable[str]
            The names of the detectors from which data will be extracted.

        Returns
        -------
        C
            An instance of the `C` class created from the specified file.

        Examples
        --------
        >>> c_instance = C.from_sts_detectors('file.det', ['detector1', 'detector2'], model_id='Model1')
        """
        v1, u1 = sts.read(file).detectors[detector_names[0]].bins[0][-2:]
        v2, u2 = sts.read(file).detectors[detector_names[1]].bins[0][-2:]
        kwargs['data'] = _make_df(ratio_v_u(_make_df(v1, u1), _make_df(v2, u2)))
        return cls(**kwargs)

    def compute(self):
        """
        Computes the cover-e value.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the cover-e value.

        Examples
        --------
        >>> c_instance = C(data=pd.DataFrame({'value': [0.5]}), model_id='Model1', deposit_ids='Dep1')
        >>> c_instance.compute()
           value
        0    0.5
        """
        return self.data


@dataclass
class CoverE:
    c: C
    e: SpectralIndex

    def __post_init__(self):
        self._check_consistency()

    def _check_consistency(self) -> None:
        if not self.c.deposit_ids == self.e.deposit_ids:
            raise Exception("Inconsistent deposits between C and E.")

    def compute(self, *args, **kwargs) -> pd.DataFrame:
        """
        Computes the cover-e value.

        Parameters
        ----------
        *args : Any
            Positional arguments to be passed to the `SpectralIndex.compute()` method.
        **kwargs : Any
            Keyword arguments to be passed to the `SpectralIndex.compute()` method.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the cover-e value.

        Examples
        --------
        >>> c_instance = C(data=pd.DataFrame({'value': [0.5]}), model_id='Model1', deposit_ids='Dep1')
        >>> ffs_num = ReactionRate(...)  # Assuming proper initialization
        >>> ffs_den = ReactionRate(...)  # Assuming proper initialization
        >>> spectral_index = SpectralIndex(numerator=ffs_num, denominator=ffs_den)
        >>> cover_e_instance = CoverE(c=c_instance, e=spectral_index)
        >>> cover_e_instance.compute()
           value  uncertainty
        0    ...          ...
        """
        v, u = ratio_v_u(self.c.compute(), self.e.compute(*args, **kwargs))
        return _make_df(v, u)

    def minus_one_percent(self, *args, **kwargs):
        """
        Computes the cover-e value and subtracts 1%, adjusting the uncertainty accordingly.

        Parameters
        ----------
        *args : Any
            Positional arguments to be passed to the `SpectralIndex.compute()` method.
        **kwargs : Any
            Keyword arguments to be passed to the `SpectralIndex.compute()` method.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the adjusted cover-e value.

        Examples
        --------
        >>> c_instance = C(data=pd.DataFrame({'value': [1.05]}), model_id='Model1', deposit_ids='Dep1')
        >>> ffs_num = ReactionRate(...)  # Assuming proper initialization
        >>> ffs_den = ReactionRate(...)  # Assuming proper initialization
        >>> spectral_index = SpectralIndex(numerator=ffs_num, denominator=ffs_den)
        >>> cover_e_instance = CoverE(c=c_instance, e=spectral_index)
        >>> cover_e_instance.minus_one_percent()
           value  uncertainty
        0    5.0          ...
        """
        out = self.compute(*args, **kwargs)
        return _make_df((out['value'] - 1) * 100, out['uncertainty'] * 100)
