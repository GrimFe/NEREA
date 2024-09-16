from collections.abc import Iterable
from dataclasses import dataclass
import pandas as pd
import numpy as np
import warnings

from datetime import datetime, timedelta

from .utils import integral_v_u, _make_df
from .effective_mass import EffectiveMass

__all__ = [
    "FissionFragmentSpectrum",
    "FissionFragmentSpectra"]

@dataclass(slots=True)
class FissionFragmentSpectrum:
    start_time: datetime
    life_time: int
    real_time: int
    data: pd.DataFrame
    campaign_id: str
    experiment_id: str
    detector_id: str
    deposit_id: str
    location_id: str
    measurement_id: str

    @property
    def smooth(self) -> pd.DataFrame:
        """
        Calculates the sum of 'counts' and the minimum value of 'channel' for each
        group based on the integer division of 'channel' by 10.
        Contains the data used to find `max` and hence `R`.

        Returns
        -------
        pd.DataFrame
            The smoothened fission fragment spectrum data.

        Examples
        --------
        >>> ffs = FissionFragmentSpectrum(...)
        >>> smoothened_data = ffs.smooth
        """
        df = self.data.copy()
        df['counts'] = df.rolling(10).mean()['counts'].fillna(0)
        return df

    def get_max(self, bins: int=None) -> pd.DataFrame:
        """
        Finds the channel with the maximum count value in a DataFrame.

        Parameters
        ----------
        bins : int, optional
            Number of bins for rebinned spectrum.
            Recommended values are 4096, 2048, 1024, 512.
            Defaults to `None` for no rebinning.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'channel' and 'counts' columns.

        Examples
        --------
        >>> ffs = FissionFragmentSpectrum(...)
        >>> max_data = ffs.get_max(...)
        """
        ## Implementing what is in MATLAB
        # reb = self.rebin(bins)
        # lst_ch = reb[reb.counts > 0].channel.max()
        # fst_ch = np.floor(lst_ch / 10)
        # df = reb.set_index("channel").loc[fst_ch:lst_ch]

        ## New implementation (skipping the first channels to cut noise)
        reb = self.rebin(bins)
        lst_ch = reb[reb.counts > 0].channel.max()
        fst_ch = reb[reb.counts > 0].channel.min() + np.floor(lst_ch / 10)
        df = reb[reb.channel > fst_ch]
        return pd.DataFrame({"channel": [df.counts.idxmax() + 1], "counts": [df.counts.max()]})

    def get_R(self, bins: int=None) -> pd.DataFrame:
        """
        Filters data in channels above the channel of the spectrum maximum
        and returns the first row with counts <= than the maximum.

        Parameters
        ----------
        bins : int, optional
            Number of bins for rebinned spectrum.
            Recommended values are 4096, 2048, 1024, 512.
            Defaults to `None` for no rebinning.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'channel' and 'counts' columns.

        Examples
        --------
        >>> ffs = FissionFragmentSpectrum(...)
        >>> r_data = ffs.get_R(...)
        """
        reb = self.rebin(bins)
        data = reb.query("channel > @self.get_max(@bins).channel[0]")
        return data[data.counts <= self.get_max(bins).counts[0] / 2].iloc[0]

    def rebin(self, bins: int=None):
        """
        Rebins the spectrum.

        Parameters
        ----------
        bins : int, optional
            Number of bins for rebinned spectrum.
            Recommended values are 4096, 2048, 1024, 512.
            Defaults to `None` for no rebinning.

        Returns
        -------
        pd.DataFrame
            Rebinned spectrum data.

        Examples
        --------
        >>> ffs = FissionFragmentSpectrum(...)
        >>> rebinned_data = ffs.rebin()
        """
        if bins is not None:
            max_bins = self.data.channel.max()
            if bins > max_bins:
                warnings.warn(f"The maximum amount of allowed bins is {max_bins}. Bins set to {max_bins}.")
            df = self.smooth.copy()
            bins_ = int(min(bins, max_bins))
            df['bins'] = pd.cut(df['channel'], bins=list(range(0, max_bins + 1, int(max_bins / bins_))))
            df = df.groupby('bins', as_index=False
                            ).agg({'counts': 'sum'}).drop('bins', axis=1
                                                            ).assign(channel=range(1, bins_+1))
        else:
            df = self.data[['counts', 'channel']]
        return df

    def integrate(self, bins: int=None) -> pd.DataFrame:
        """
        Calculates the integral of data based on specified channels (as a function of R)
        and returns a DataFrame with channel, value, and uncertainty columns.

        Parameters
        ----------
        bins : int, optional
            Number of bins for integration.
            Defaults to `None`, which uses all the bins.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'channel', 'value', and 'uncertainty' columns.

        Examples
        --------
        >>> ffs = FissionFragmentSpectrum(...)
        >>> integral_data = ffs.integrate()
        """
        out = []
        bins_ = self.data.channel.max() if bins is None else bins
        reb = self.rebin(bins_)
        for ch in np.array(range(15, 65, 5)) / 100:
            ch_ = np.floor(ch * self.get_R(bins_).channel)
            v, u = integral_v_u(reb.query("channel >= @ch_").counts)
            out.append(_make_df(v, u).assign(channel=ch_))
        return pd.concat(out, ignore_index=True)[['channel', 'value', 'uncertainty', 'uncertainty [%]']]

    def calibrate(spectrum, bins: int=None) -> EffectiveMass:
        """
        Computes the fission chamber effective mass from the fission
        fragment spectrum.

        Takes
        -----
        spectrum : pass
            pass
        bins : int, optional
            Number of bins for integration.
            Defaults to `None`, which uses all the bins.
    
        Examples
        --------
        >>> spectrum = pass
        >>> em = FFS.from_TKA(file).calibrate(spectrum)
        """
        pass

    @classmethod
    def from_TKA(cls, file: str, **kwargs):
        """
        Reads data from a TKA file to create a `FissionFragmentSpectrum`
        instance.

        Parameters
        ----------
        file : str
            TKA file path.
        *args : Any
            Positional arguments to be passed to the `FissionFragmentSpectrum()` initialization.
        **kwargs : Any
            Keyword arguments to be passed to the `FissionFragmentSpectrum()` initialization.

        Returns
        -------
        FissionFragmentSpectrum
            Fission fragment spectrum instance.

        Examples
        --------
        >>> ffs = FissionFragmentSpectrum.from_TKA('filename.TKA', ...)
        """
        data = pd.read_csv(file, header=None)
        data = data.iloc[2:].reset_index(drop=True).reset_index()        
        data.columns = ['channel', 'counts']
        data.channel += 1
        # GENIE overwrites the first two counts with time indications
        # Here two zeros are added at the beginning of the data
        data = pd.concat([pd.DataFrame({'channel': [-2, -1], 'counts': [0, 0]}),
                          data], ignore_index=True)
        data['channel'] += 2
        kwargs['data'] = data
        return cls(**kwargs)

    @classmethod
    def from_formatted_TKA(cls, file: str):
        """
        Reads data from a formatted TKA file and extracts metadata from the
        file name to create a `FissionFragmentSpectrum` instance.
        The filename is expected to be formatted as:
        {Campaign}_{Experiment}_{Detector}_{Deposit}_{Location}_{Measurement}.TKA
        Requires a text file with the same name with time information.

        Parameters
        ----------
        file : str
            TKA file path.

        Returns
        -------
        FissionFragmentSpectrum
            Fission fragment spectrum instance.

        Examples
        --------
        >>> ffs = FissionFragmentSpectrum.from_TKA(f
                    '{Campaign}_{Experiment}_{Detector}_{Deposit}_{Location}_{Measurement}.TKA')
        """
        with open(file.replace('.TKA', '.txt'), 'r') as f:
            start, life, real = f.readlines()
        campaign_id, experiment_id, detector_id, deposit_id, location_id, measurement_id = file.split('\\')[-1].split('_')
        dct = {
            'start_time': datetime.strptime(start, "%Y-%m-%d %H:%M:%S\n"),
            'life_time': float(life),
            'real_time': float(real),
            'campaign_id':campaign_id,
            'experiment_id': experiment_id,
            'detector_id': detector_id,
            'deposit_id': deposit_id,
            'location_id': location_id,
            'measurement_id': measurement_id.replace(".TKA", ""),
        }
        return cls.from_TKA(file, **dct)


@dataclass(slots=True)
class FissionFragmentSpectra():
    """
    This class works under the assumption that no measurement time was
    lost in the process of measuring the fission fragment spectra.

    That is that the `self.data` will be the channel-wise sum of the counts
    of the listed fission fragment spectra, while the start time of the
    measurement will be the minimum start time and life and real times will
    be the sum of the respective times in the listed fission fragment spectra.

    For a more solid behaviour, resort to the `AverageReactionRate` class.

    """
    spectra: Iterable[FissionFragmentSpectrum]

    def __post_init__(self) -> None:
        self._check_consistency()

    def __iter__(self):
        return self.spectra.__iter__()

    def __getitem__(self, item) -> FissionFragmentSpectrum:
         return self.spectra[item]

    def _check_consistency(self) -> None:
        """
        Checks the consistency of
            - campaign_id
            - experiment_id
            - detector_id
            - deposit_id
            - location_id
        among self.spectra

        """
        # should also check the consistency of the ffs?
        should = ['campaign_id', 'experiment_id']
        must = ['detector_id', 'deposit_id', 'location_id']
        for attr in should:
            if not all([getattr(ffs, attr) == getattr(self.spectra[0], attr) for ffs in self.spectra]):
                warnings.warn(f"Inconsistent {attr} among different FissionFragmentSpectrum instances.")
        for attr in must:
            if not all([getattr(ffs, attr) == getattr(self.spectra[0], attr) for ffs in self.spectra]):
                raise Exception(f"Inconsistent {attr} among different FissionFragmentSpectrum instances.")

    @property
    def best(self) -> FissionFragmentSpectrum:
        """
        Returns the fission fragment spectrum with the highest sum value.

        Returns
        -------
        FissionFragmentSpectrum
            Fission fragment spectrum with the highest integral count.

        Examples
        --------
        >>> ffss = FissionFragmentSpectra(...)
        >>> best_spectrum = ffss.best
        """
        max = self.spectra[0].data.counts.sum()
        out = self.spectra[0]
        for s in self.spectra[1:]:
            if s.data.counts.sum() > max:
                out = s
        return out

    def merge(self) -> FissionFragmentSpectrum:
        """
        Returns a Fission Fragment Spectrum instance containing merged
        informatio of the fission fragment spectra in `self.spectra`.

        Returns
        -------
        FissionFragmentSpectrum
            merged fission fragment specrtum instance

        Example
        -------
        >>> ffs = FissionFragmentSpectra([ffs1, ffs2]).merge()
        """
        pass

    @classmethod
    def from_formatted_TKA(cls, files: Iterable[str]):
        """
        Reads a list of files to create a FissionFragmentSpectra object.
        Each filename is expected to be formatted as:
        {Campaign}_{Experiment}_{Detector}_{Deposit}_{Location}_{Measurement}.TKA
        Each file requires a text file with the same name with time information.

        Parameters
        ----------
        files : Iterable[str]
            List of file paths.

        Returns
        -------
        FissionFragmentSpectra
            Fission fragment spectra instance.

        Examples
        --------
        >>> ffss = FissionFragmentSpectra.from_TKA(['file1.TKA', 'file2.TKA'])
        """
        data = []
        for file in files:
            data.append(cls.from_formatted_TKA(file))
        return cls(data)
