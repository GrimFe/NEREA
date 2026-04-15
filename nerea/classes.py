from dataclasses import dataclass

from typing import Self
from datetime import datetime, timedelta
import warnings
import linecache
from collections.abc import Iterable
import matplotlib.pyplot as plt

import pandas as pd
import serpentTools as sts

from .utils import _make_df, ratio_v_u
from .constants import ATOMIC_MASS, BASE_DATE
from .defaults import DEFAULT_DETECTOR_DEPOSIT

__all__ = ["EffectiveDelayedParams", "Xs", "TimeSeriesImporter"]


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


@dataclass(slots=True)
class TimeSeriesImporter:
    """
    ``nerea.TimeSeriesImporter``
    ============================
    Class to import time dependent data.

    Attributes
    ----------
    **data**: ``pd.DataFrame``
        data frame with time dependent data.
    **metadata**: ``dict``
        storing metadata for class creation.
    """
    data: pd.DataFrame
    metadata: dict[str, any]
 
    @classmethod
    def from_ads(cls, file: str, formatted: bool, **kwargs) -> Self:
        """
        `nerea.TimeSeriesImporter.from_ads()`
        -------------------------------------
        Method to create a ``nerea.TimeSeriesImporter`` instance
        from an ASCII file generated by ADS DAQ.

        Parameters
        ----------
        **file** : ``str``
            Path to the ASCII file.
        **kwargs
            additional arguments for class creation
            **detector_id** (``int|str``): metadata for detector identification
            **deposit_id** (``str``): metadata for detector deposit
            **experiment_id** (``str``): metadata for experiment identification
            **campaign_id** (``str``): metadata for experimental campaign identification.

        Returns
        -------
        ``nerea.TimeSeriesImporter``
            A new ``nerea.TimeSeriesImporter`` instance.

        Note
        ----
        - ``deposit_id`` set to ``'U235'`` by default if not explicitly passed
        - ``experiment_id`` inferred from file name if `formatted == True`
        - ``campaign_id`` inferred from file name if `formatted == True`
        - ``detector_id`` kwarg is also used to select the detector to read.
        """
        start_time = datetime.strptime(linecache.getline(file, 1), "%d-%m-%Y %H:%M:%S\n")
        read = pd.read_csv(file, sep='\t', skiprows=[0,1], decimal=',')
        read["Time"] = read["Time"].apply(lambda x: start_time + timedelta(seconds=x))
        d = kwargs['detector_id']
        if isinstance(d, int) or (isinstance(d, str) and d.isnumeric()):
            d = f"Det {d}"
        data = read[["Time", d]].rename(columns={d: "value"})

        md = {
            'start_time': datetime.strptime(linecache.getline(file, 1), "%d-%m-%Y %H:%M:%S\n"),
            'timebase': (data['Time'][1] - data['Time'][0]).total_seconds(),
            'deposit_id': kwargs.get('deposit_id', DEFAULT_DETECTOR_DEPOSIT)
            }
        if formatted:
            metadata = file.split('\\')[-1].split('.')[0]
            md['campaign_id'], md['experiment_id'] = metadata.split('_')
        return cls(data, kwargs | md)

    @classmethod
    def from_phspa(cls, file: str, formatted: bool, **kwargs) -> Self:
        """
        `nerea.TimeSeriesImporter.from_phspa()`
        -------------------------------------
        Method to create a ``nerea.TimeSeriesImporter`` instance
        from an ASCII file generated by PHSPA DAQ.

        Parameters
        ----------
        **file** : ``str``
            Path to the ASCII file.
        **kwargs
            additional arguments for class creation
            **detector_id** (``int|str``): metadata for detector identification
            **deposit_id** (``str``): metadata for detector deposit
            **experiment_id** (``str``): metadata for experiment identification
            **campaign_id** (``str``): metadata for experimental campaign identification.

        Returns
        -------
        ``nerea.TimeSeriesImporter``
            A new ``nerea.TimeSeriesImporter`` instance.

        Note
        ----
        - ``deposit_id`` set to ``'U235'`` by default if not explicitly passed
        - ``experiment_id`` inferred from file name if `formatted == True`
        - ``campaign_id`` inferred from file name if `formatted == True`
        - ``detector_id`` inferred from file name if `formatted == True`
        """
        data = pd.read_csv(file, sep="\t", skiprows=18, decimal=',').iloc[:,:-1]
        data.columns = ["Time", "value"]
        data.Time = data.Time.apply(lambda x: BASE_DATE + timedelta(days=x))
        warnings.warn("Average timebase considered for PHSPA acquisitions.")
        md = {'timebase': data.Time.diff().dt.total_seconds().mean(),
              'start_time': data.Time.min(),
              'deposit_id': kwargs.get('deposit_id', DEFAULT_DETECTOR_DEPOSIT)}

        if formatted:
            metadata = file.split('\\')[-1].split('.')[0]
            md['campaign_id'], md['experiment_id'], md['detector_id'] = metadata.split('_')
        return cls(file, kwargs | md)

    @classmethod
    def from_br1(cls, file: str, **kwargs) -> Self:
        """
        `nerea.TimeSeriesImporter.from_br1()`
        -------------------------------------
        Method to create a ``nerea.TimeSeriesImporter`` instance
        from an ASCII file generated by NBS DAQ.

        Parameters
        ----------
        **file** : ``str``
            Path to the ASCII file.
        **kwargs
            additional arguments for class creation
            **detector_id** (``int|str``): metadata for detector identification
            **deposit_id** (``str``): metadata for detector deposit
            **experiment_id** (``str``): metadata for experiment identification
            **campaign_id** (``str``): metadata for experimental campaign identification.

        Returns
        -------
        ``nerea.TimeSeriesImporter``
            A new ``nerea.TimeSeriesImporter`` instance.

        Note
        ----
        - ``deposit_id`` set to ``'U235'`` by default if not explicitly passed
        - ``experiment_id`` inferred from file name if `formatted == True`
        - ``campaign_id`` set to ``'CAL'`` if not explicitly passed
        - ``detector_id`` set tot ``'NBS'`` if not explicitly passed
        """
        data = pd.read_csv(file, sep=';', header=None)[[0,4]]
        data.columns = ["Time", "value"]
        data["Time"] = pd.to_datetime(data["Time"])
        warnings.warn("Average timebase considered for BR1 acquisitions.")
        md = {'timebase': data.Time.diff().dt.total_seconds().mean(),
              'start_time': data.Time.iloc[0],
              'campaign_id': kwargs.get('campaign_id', 'CAL'),
              'detector_id': kwargs.get('detector_id', 'NBS'),
              'deposit_id': kwargs.get('deposit_id', DEFAULT_DETECTOR_DEPOSIT)
        }
        return cls(data, kwargs | md)

    @classmethod
    def from_vf(cls, file: str, **kwargs) -> Self:
        """
        `nerea.TimeSeriesImporter.from_vf()`
        ------------------------------------
        Method to create a ``nerea.TimeSeriesImporter`` instance
        from an ASCII file generated by the VENUS-F
        monitoring system.

        Parameters
        ----------
        **file** : ``str``
            Path to the ASCII file.
        **kwargs
            additional arguments for class creation
            **detector_id** (``int|str``): metadata for detector identification
            **deposit_id** (``str``): metadata for detector deposit
            **experiment_id** (``str``): metadata for experiment identification
            **campaign_id** (``str``): metadata for experimental campaign identification.


        Returns
        -------
        ``nerea.CountRate``
            A new ``nerea.CountRate`` instance.

        Note
        ----
        - ``deposit_id`` set to ``'U235'`` by default if not explicitly passed
        - ``experiment_id`` inferred from file name if `formatted == True`
        - ``campaign_id`` inferred from file name if `formatted == True`
        - reads experiment date from file name (format: %Y-%m-%d)."""
        data = pd.read_csv(file, encoding='unicode_escape', sep=r'\s+', index_col=False)
        metadata = file.split('\\')[-1].split('.')[0]
        st = metadata.split('_')[2]
        data["Time"] = pd.to_datetime(st + ' '+ data.time.astype(str),
                                      format="%Y-%m-%d %H:%M:%S")
        data["value"] = data[kwargs['detector_id']]
        
        md = {'campaign_id': metadata.split('_')[0],
              'experiment_id': metadata.split('_')[1],
              'start_time': st,
              'timebase': data.Time.iloc[1] - data.Time.iloc[1]
              }
        return cls(data[["Time", "value"]], kwargs | md)

    @classmethod
    def from_ascii(cls,
                   file: str,
                   filetype: str='infer',
                   formatted: bool=False,
                   **kwargs) -> Self:
        """
        `nerea.TimeSeriesImporter.from_ascii()`
        ---------------------------------------
        Method to create a ``nerea.TimeSeriesImporter`` instance
        from an ASCII file.

        Parameters
        ----------
        **file** : ``str``
            Path to the ASCII file.
        **filetype** : ``str``, optional
            Type of ASCII file to process.
            Default is ``'infer'`` to infer it from
            file extension.
        **kwargs
            additional arguments for class creation
            **detector_id** (``int|str``): metadata for detector identification
            **deposit_id** (``str``): metadata for detector deposit
            **experiment_id** (``str``): metadata for experiment identification
            **campaign_id** (``str``): metadata for experimental campaign identification.

        Returns
        -------
        ``nerea.TimeSeriesImporter``
            A new ``TimeSeriesImporter`` instance.
            
        Notes
        -----
        - `deposit_id` set to `'U235'` by default if not explicitly passed
        - `experiment_id` inferred from file name if `formatted == True`
        - `campaign_id` inferred from file name if `formatted == True`
        - formats:
            ``'CAMP_EXP_DET.log'`` (phspa)
            ``'CAMP_EXP.ads'`` (ads)
            ``'CAMP_EXP_DATE.vf'`` (vf)
        """
        ft = file.split('.')[-1] if filetype == 'infer' else filetype
        match ft:
            case 'ads':
                out = cls.from_ads(file, formatted, **kwargs)
            case 'phspa':
                out = cls.from_phspa(file, formatted, **kwargs)
            case 'log':
                out = cls.from_phspa(file, formatted=False, **kwargs)
            case 'br1':
                out = cls.from_br1(file, **kwargs)
            case 'vf':
                out = cls.from_vf(file, **kwargs)
            case _:
                raise ValueError("ASCII file type processing not implemented")
        return out

    @classmethod
    def from_files(cls, files: Iterable[str], filetype: str='infer', **kwargs) -> Self:
        """
        `nerea.CountRate.from_files()`
        ------------------------------
        Method to create a ``nerea.CountRate`` instance
        joing data from ASCII files of the same type.

        Parameters
        ----------
        **file** : ``str``
            Path to the ASCII file.
        **filetype** : ``str``, optional
            Type of ASCII file to process.
            Default is ``'infer'`` to infer it from
            file extension.
        **kwargs
            additional arguments for class creation
            **deposit_id** (``str``): metadata for detector deposit
            **detector_id** (``str``): metadata for detector identification
            **experiment_id** (``str``): metadata for experiment identification
            **campaign_id** (``str``): metadata for experimental campaign identification.

        Returns
        -------
        ``nerea.CountRate``
            A new ``CountRate`` instance."""
        data = []
        vlines = []
        for i, f in enumerate(files):
            rr = cls.from_ascii(f, filetype, **kwargs)
            data.append(rr.data)
            vlines.append(data[-1].Time.iloc[-1])
            if i == 0:
                _kwargs = {'campaign_id': rr.campaign_id,
                           'experiment_id': rr.experiment_id,
                           'detector_id': rr.detector_id,
                           'deposit_id': rr.deposit_id}
        data = pd.concat(data, ignore_index=True)
        _kwargs['timebase'] = data.Time.diff().dt.total_seconds().mean()
        _kwargs['start_time'] = data.Time.min()
        _kwargs['_vlines'] = vlines
        return cls(data, _kwargs)
