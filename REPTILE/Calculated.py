from collections.abc import Iterable
from dataclasses import dataclass

import serpentTools as sts
import pandas as pd

from REPTILE.utils import _make_df, ratio_v_u

@dataclass(slots=True)
class Calculated:
    def calculate(self) -> None:
        """
        Placeholder for inheriting classes.
        """
        return None

@dataclass(slots=True)
class CalculatedSpectralIndex(Calculated):
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
        CalculatedSpectralIndex
            An instance of the `CalculatedSpectralIndex` class created
            from the specified file.

        Examples
        --------
        >>> c_instance = CalculatedSpectralIndex.from_sts('file.det',
                                                    'SI_detector', model_id='Model1')
        """
        ## works with relative uncertainties for the moment.
        ## Shall be homogenized with the rest of the API
        v, u = sts.read(file).detectors[detector_name].bins[0][-2:]
        kwargs['data'] = _make_df(v, u * v)  # Serpent detector uncertainty is relative
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
        CalculatedSpectralIndex
            An instance of the `CalculatedSpectralIndex` class created from
            the specified file.

        Examples
        --------
        >>> c_instance = CalculatedSpectralIndex.from_sts_detectors('file.det', 
                                                    ['detector1', 'detector2'], model_id='Model1')
        """
        v1, u1 = sts.read(file).detectors[detector_names[0]].bins[0][-2:]
        v2, u2 = sts.read(file).detectors[detector_names[1]].bins[0][-2:]
        kwargs['data'] = _make_df(ratio_v_u(_make_df(v1, u1 * v1),  # Serpent detector uncertainty is relative
                                            _make_df(v2, u2 * v2)))  # Serpent detector uncertainty is relative
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
class CalculatedTraverse:
    data: pd.DataFrame
    model_id: str
    deposit_id: str

    @classmethod
    def from_sts(cls, file: str, detector_names: list[str], **kwargs):
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
        CalculatedTraverse
            An instance of the `CalculatedTraverse` class created from
            the specified file.
        """
        out = []
        for d in detector_names:
            v, u = sts.read(file).detectors[detector_names].bins[0][-2:]
            out.append(_make_df(v, u).assign(traverse=d))
        return cls(data=pd.concat(out, ignore_index=True), **kwargs)

    def calculate(self):
        """
        Computes the C value. Alias for self.data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the C value.

        """
        return self.data
    