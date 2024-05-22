from collections.abc import Iterable
from dataclasses import dataclass
import pandas as pd
import numpy as np
import warnings

from datetime import datetime, timedelta

from PSICHE.utils import integral_v_u, _make_df

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

    @property
    def max(self) -> pd.DataFrame:
        """
        Finds the channel with the maximum count value in a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'channel' and 'counts' columns.

        Examples
        --------
        >>> ffs = FissionFragmentSpectrum(...)
        >>> max_data = ffs.max
        """
        lst_ch = self.smooth[self.smooth.counts > 0].channel.max()
        fst_ch = np.floor(lst_ch / 10)
        data = self.smooth.set_index("channel").loc[fst_ch:lst_ch]
        return pd.DataFrame({"channel": [data.counts.idxmax()], "counts": [data.counts.max()]})

    @property
    def R(self) -> pd.DataFrame:
        """
        Filters data in channels above the channel of the spectrum maximum
        and returns the first row with counts <= than the maximum.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'channel' and 'counts' columns.

        Examples
        --------
        >>> ffs = FissionFragmentSpectrum(...)
        >>> r_data = ffs.R
        """
        data = self.smooth.query("channel > @self.max.channel[0]")
        return data[data.counts <= self.max.counts[0] / 2].iloc[0]
    
    def rebin(self, bins: int):
        """
        Rebins the spectrum.

        Parameters
        ----------
        bins : int, optional
            Number of bins for rebinned spectrum.
            Recommended values are 4096, 2048, 1024, 512.

        Returns
        -------
        pd.DataFrame
            Rebinned spectrum data.

        Examples
        --------
        >>> ffs = FissionFragmentSpectrum(...)
        >>> rebinned_data = ffs.rebin()
        """
        max_bins = self.data.channel.max()
        if bins > max_bins:
            warnings.warn(f"The maximum amount of allowed bins is {max_bins}. Bins set to {max_bins}.")
        df = self.smooth.copy()
        bins_ = int(min(bins, max_bins))
        df['bins'] = pd.cut(df['channel'], bins=list(range(0, max_bins + 1, int(max_bins / bins_))))
        return df.groupby('bins', as_index=False
                            ).agg({'counts': 'sum'}).drop('bins', axis=1
                                                            ).assign(channel=range(1, bins_+1))

    def integrate(self, bins: int = None) -> pd.DataFrame:
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
            ch_ = np.floor(ch * self.R.channel)
            v, u = integral_v_u(reb.query("channel >= @ch_").counts)
            out.append(_make_df(v, u).assign(channel=ch_))
        return pd.concat(out, ignore_index=True)

    @classmethod
    def from_TKA(cls, file: str):
        """
        Reads data from a TKA file and extracts metadata from the file name
        to create a `FissionFragmentSpectrum` instance.
        The filename is expected to be formatted as:
        {Campaign}_{Experiment}_{Detector}_{Deposit}_{Location}_{Measurement}.TKA

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
        >>> ffs = FissionFragmentSpectrum.from_TKA('filename.TKA')
        """
        data = pd.read_csv(file, header=None)
        with open(file.replace('.TKA', '.txt'), 'r') as f:
            start, life, real = f.readlines()
        data = data.iloc[2:].reset_index(drop=True).reset_index()        
        data.columns = ['channel', 'counts']
        data.channel += 1
        # GENIE overwrites the first two counts with time indications
        # Here two zeros are added at the beginning of the data
        data = pd.concat([pd.DataFrame({'channel': [-2, -1], 'counts': [0, 0]}),
                          data], ignore_index=True)
        data['channel'] += 2
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
            'data': data
        }
        return cls(**dct)


@dataclass(slots=True)
class FissionFragmentSpectra(FissionFragmentSpectrum):
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

    @classmethod
    def from_TKA(cls, files: Iterable[str]):
        """
        Reads a list of files to create a FissionFragmentSpectra object.
        Each filename is expected to be formatted as:
        {Campaign}_{Experiment}_{Detector}_{Deposit}_{Location}_{Measurement}.TKA

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
        out = []
        data = []
        for file in files:
            out.append(FissionFragmentSpectrum.from_TKA(file))
            data.append(out[-1].data.set_index('channel'))
        dct = {
            'start_time': min([ffs.start_time for ffs in out]),
            'life_time': sum([ffs.life_time for ffs in out]),
            'real_time': sum([ffs.real_time for ffs in out]),
            'campaign_id': out[-1].campaign_id,
            'experiment_id': out[-1].experiment_id,
            'detector_id': out[-1].detector_id,
            'deposit_id': out[-1].deposit_id,
            'location_id': out[-1].location_id,
            'measurement_id': out[-1].measurement_id,
            'data': sum(data).reset_index(),
            'spectra': out
        }
        return cls(**dct)
