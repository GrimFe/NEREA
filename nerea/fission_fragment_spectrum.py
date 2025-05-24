from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self
import pandas as pd
import numpy as np
import warnings

from datetime import datetime, timedelta
from scipy.signal import savgol_filter

from .utils import integral_v_u, _make_df, ratio_v_u, product_v_u
from .constants import AVOGADRO, ATOMIC_MASS
from .effective_mass import EffectiveMass
from .reaction_rate import ReactionRate
from .defaults import *

__all__ = [
    "FissionFragmentSpectrum",
    "FissionFragmentSpectra"]

@dataclass(slots=True)
class FissionFragmentSpectrum:
    start_time: datetime
    data: pd.DataFrame
    campaign_id: str
    experiment_id: str
    detector_id: str
    deposit_id: str
    location_id: str
    measurement_id: str
    life_time: int
    real_time: int
    life_time_uncertainty: float = 0.
    real_time_uncertainty: float = 0.

    def smooth(self, method: str="moving_average", *args, **kwargs) -> Self:
        """
        Calculates the sum of 'counts' and the minimum value of 'channel' for each
        group based on the integer division of 'channel' by 10.
        Contains the data used to find `max` and hence `R`.

        Parameters
        ----------
        method : str, optional
            the method to use. Allowed options are:
                - moving_average
                    (requires window)
                - savgol_filter
                    (requires window_length
                            polyorder)
            Defailt is "moving_average"
        * args :
            arguments for the chosen method
        **kwargs :
            arguments for the chosen method

        Returns
        -------
        self.__class__
            With the smoothed fission fragment spectrum as data.

        Examples
        --------
        >>> ffs = FissionFragmentSpectrum(...)
        >>> smoothened_data = ffs.smooth()
        """
        df = self.data.copy()
        match method:
            case "moving_average":
                if kwargs.get("window") is None: kwargs["window"] = 10
                df['counts'] = df.rolling(*args, **kwargs).mean()['counts'].fillna(0)
            case "savgol_filter":
                df['counts'] = savgol_filter(df['counts'], *args, **kwargs)
                if any(df['counts'] < 0):
                    warnings.warn("Using Savgol Filter smoothing negative counts appear.")
            case _:
                raise Exception(f"The chosen method {method} is not allowed")
        out = self.__class__(
            start_time = self.start_time,
            data = df,
            campaign_id = self.campaign_id,
            experiment_id = self.experiment_id,
            detector_id = self.detector_id,
            deposit_id = self.deposit_id,
            location_id = self.location_id,
            measurement_id = self.measurement_id,
            life_time = self.life_time,
            real_time = self.real_time,
            life_time_uncertainty = self.life_time_uncertainty,
            real_time_uncertainty = self.real_time_uncertainty
        )        
        return out

    def rebin(self, bins: int=None, smooth: bool=True, *args, **kwargs) -> pd.DataFrame:
        """
        Rebins the spectrum.

        Parameters
        ----------
        bins : int, optional
            Number of bins for rebinned spectrum.
            Recommended values are 4096, 2048, 1024, 512.
            Defaults to `None` for no rebinning.
        smooth : bool, optional
            Flag to rebin smoothened spectrum.
            Defaults to True.
        *args :
            arguments for self.smooth()
        **kwargs :
            arguments for self.smooth()

        Returns
        -------
        pd.DataFrame
            Rebinned spectrum data.

        Examples
        --------
        >>> ffs = FissionFragmentSpectrum(...)
        >>> rebinned_data = ffs.rebin()
        """
        df = self.smooth(*args, **kwargs).data if smooth else self.data.copy()
        if bins is not None:
            max_bins = self.data.channel.max()
            if bins > max_bins:
                warnings.warn(f"The maximum amount of allowed bins is {max_bins}. Bins set to {max_bins}.")
            bins_ = int(min(bins, max_bins))
            df['bins'] = pd.cut(df['channel'], bins=list(range(0, max_bins + 1, int(max_bins / bins_))))
            df = df.groupby('bins', as_index=False, observed=False
                            ).agg({'counts': 'sum'}).drop('bins', axis=1
                                                            ).assign(channel=range(1, bins_+1))
        return df[['channel', 'counts']]

    def get_max(self, fst_ch: int=None, bin_kwargs: dict=None) -> pd.DataFrame:
        """
        Finds the channel with the maximum count value in a DataFrame.

        Parameters
        ----------
        fst_ch : int, optional
            Left channel to search the maximum from.
            Defaults to `None` for automatic 1/10 of total
            channels acquired.
        bin_kwargs : dict, optional
            - bins : int
            - smooth : bool

        Returns
        -------
        pd.DataFrame
            DataFrame with 'channel' and 'counts' columns.

        Examples
        --------
        >>> ffs = FissionFragmentSpectrum(...)
        >>> max_data = ffs.get_max(...)
        """
        bin_kw = DEFAULT_BIN_KWARGS if bin_kwargs is None else bin_kwargs

        reb = self.rebin(**bin_kw)
        if fst_ch is None:
            lst_ch = reb[reb.counts > 0].channel.max()
            fst_ch = reb[reb.counts > 0].channel.min() + np.floor(lst_ch / 10)
        df = reb[reb.channel > fst_ch]
        return pd.DataFrame({"channel": [df.counts.idxmax() + 1], "counts": [df.counts.max()]})

    def get_R(self, bin_kwargs: dict=None, max_kwargs: dict=None) -> pd.DataFrame:
        """
        Filters data in channels above the channel of the spectrum maximum
        and returns the first row with counts <= than the maximum.

        Parameters
        ----------
        bin_kwargs : dict, optional
            - bins : int
            - smooth : bool
        max_kwargs : dict, optional
            Other kwargs for self.get_max()
            - fst_ch : int

        Returns
        -------
        pd.DataFrame
            DataFrame with 'channel' and 'counts' columns.

        Examples
        --------
        >>> ffs = FissionFragmentSpectrum(...)
        >>> r_data = ffs.get_R(...)
        """
        bin_kw = DEFAULT_BIN_KWARGS if bin_kwargs is None else bin_kwargs
        max_kw = DEFAULT_MAX_KWARGS if max_kwargs is None else max_kwargs

        reb = self.rebin(**bin_kw)
        data = reb.query("channel > @self.get_max(bin_kwargs=@bin_kw, **max_kw).channel[0]")
        return data[data.counts <= self.get_max(bin_kwargs=bin_kw, **max_kw).counts[0] / 2].iloc[0].to_frame().T[["channel", "counts"]]

    def integrate(self, bin_kwargs: dict=None, max_kwargs: dict=None,
                  llds: Iterable[int | float]=[.15, .2, .25, .3, .35, .4, .45, .5, .55, .6]) -> pd.DataFrame:
        """
        Calculates the integral of data based on specified channels (as a function of R)
        and returns a DataFrame with channel, value, and uncertainty columns.
    
        Parameters
        ----------
        bin_kwargs : dict, optional
            - bins : int
            - smooth : bool
        max_kwargs : dict, optional
            kwargs for self.get_max().
            - fst_ch : int
        llds : Iterable[int|float], optional
            low level discriminations to integrate from.
            If integer -> interpreted as channel.
            If fractional -> interpreted as fractions of self.get_R().channel.
            Defaults to 10 llds between [0.15, 0.6].

        Returns
        -------
        pd.DataFrame
            DataFrame with 'channel', 'value', and 'uncertainty' columns.

        Notes
        -----
        max_kwargs is used only with float llds to search for R.

        Examples
        --------
        >>> ffs = FissionFragmentSpectrum(...)
        >>> integral_data = ffs.integrate()
        """
        bin_kw = DEFAULT_BIN_KWARGS if bin_kwargs is None else bin_kwargs
        max_kw = DEFAULT_MAX_KWARGS if max_kwargs is None else max_kwargs
        if bin_kw.get('bins') is None: bin_kw['bins'] = self.data.channel.max()

        out = []
        reb = self.rebin(**bin_kw)
        for ch in llds:
            channel_discri = isinstance(ch, int) or ch.is_integer()
            ch_ = ch if channel_discri else np.floor(ch * self.get_R(bin_kwargs=bin_kw, max_kwargs=max_kwargs).channel.iloc[0])
            v, u = integral_v_u(reb.query("channel >= @ch_").counts)
            out.append(_make_df(v, u).assign(channel=ch_, R=np.nan if channel_discri else ch))
        return pd.concat(out, ignore_index=True)[['channel', 'value', 'uncertainty', 'uncertainty [%]', 'R']]

    @staticmethod
    def _get_calibration_coefficient(one_group_xs: dict[str, float], composition: dict[str, float] | pd.DataFrame):
        """
        Calculates the fission chamber calibration coefficient.

        Paramters
        ---------
        one_group_xs : dict[str, float]
            the one group cross sections of the fission
            chamber components. `key` is the nuclide
            string identifier (e.g., `'U235'`), and `value`
            is its one group cross section.
            Has columns for its value and uncertainty.
        composition : dict[str, float] | pd.DataFrame
            the fission chamber composition relative to
            the deposit main nuclide. `key` is the nuclide
            string identifier (e.g., `'U235'`), and `value`
            is its atomic abundance relative to the main one.
            Has columns for its value and uncertainty.
        """
        xs = pd.DataFrame(one_group_xs, index=['value', 'uncertainty']).T if not isinstance(one_group_xs, pd.DataFrame) else one_group_xs.copy()
        match composition.value.max():
            case 1:
                pass
            case 100:
                composition.value /= 100
                composition.uncertainty /= 100
            case _:
                raise ValueError("`composition` should be relative to the main isotope," +
                                 "which should be reported in it.")

        ## calculation of the sum over nuclides in the deposit (n * xs)
        main = composition.query("value == 1").index.values[0]
        xs_ = xs.loc[composition.index]
        a = _make_df(ratio_v_u(AVOGADRO, ATOMIC_MASS[main])[0],
                     ratio_v_u(AVOGADRO, ATOMIC_MASS[main])[1])
        c_v = a['value'].value * composition.value @ xs_.value
        c_u = np.sqrt((a['value'].value * composition.value @ xs_.uncertainty) **2 +
                      (a['value'].value * composition.uncertainty @ xs_.value) **2 +
                      (a['uncertainty'].value * composition.value @ xs_.value) **2)
        return _make_df(c_v / 1e6, c_u / 1e6)  # EM in ug

    def calibrate(self,
                  k: pd.DataFrame,
                  composition: dict[str, float] | pd.DataFrame,
                  monitor: ReactionRate,
                  one_group_xs: dict[str, float],
                  **kwargs) -> EffectiveMass:
        """
        Computes the fission chamber effective mass from the fission
        fragment spectrum.

        Takes
        -----
        k : pd.DataFrame,
            the facility calibration factor as in
            "Miniature fission chambers calibration in
            pulse mode: interlaboratory comparison at
            the SCK•CEN BR1 and CEA CALIBAN reactors".
            Has columns for its value and uncertainty.
        composition : dict[str, float] | pd.DataFrame
            the fission chamber composition relative to
            the deposit main nuclide. `key` is the nuclide
            string identifier (e.g., `'U235'`), and `value`
            is its atomic abundance relative to the main one.
            Has columns for its value and uncertainty.
        one_group_xs : dict[str, float]
            the one group cross sections of the fission
            chamber components. `key` is the nuclide
            string identifier (e.g., `'U235'`), and `value`
            is its one group cross section.
            Has columns for its value and uncertainty.
        monitor : nerea.ReactionRate
            the counts of the monitor fission chamber used
            during calibration.
        **kwargs:
            arguments for self.integrate()
            bin_kwargs : dict, optional
                - bins : int
                - smooth : bool
            max_kwargs : dict, optional
                - fst_ch : int
            llds : Iterable[int|float]
    
        Examples
        --------
        >>> spectrum = pass
        >>> em = FFS.from_TKA(file).calibrate(spectrum)
        """
        kwargs['bin_kwargs'] = kwargs['bin_kwargs'] if kwargs.get('bin_kwargs') else DEFAULT_BIN_KWARGS
        kwargs['max_kwargs'] = kwargs['max_kwargs'] if kwargs.get('max_kwargs') else DEFAULT_MAX_KWARGS
        if kwargs['bin_kwargs'].get('bins') is None: kwargs['bin_kwargs']['bins'] = self.data.channel.max()

        composition_ = pd.DataFrame(composition, index=['value', 'uncertainty']).T if not isinstance(composition, pd.DataFrame) else composition.copy()
        c = self._get_calibration_coefficient(one_group_xs, composition_)

        pm = monitor.average(self.start_time, self.real_time)
        kmc = _make_df(product_v_u([k, pm, c])[0], product_v_u([k, pm, c])[1])
        integral = []
        for _, row in self.integrate(**kwargs).iterrows():
            v, u = ratio_v_u(row, _make_df(self.life_time, self.life_time_uncertainty))
            integral.append(_make_df(v, u).assign(channel=row.channel, R=row.R))
        integral = pd.concat(integral)

        data = pd.concat([_make_df(v, u) for v, u in zip(ratio_v_u(integral, kmc)[0],
                                                         ratio_v_u(integral, kmc)[1])]
                                                         ).assign(channel=integral.channel,
                                                                  R=integral.R)
        return EffectiveMass(data=data[["channel", "value", "uncertainty", "uncertainty [%]", "R"]],
                             composition=composition_, detector_id=self.detector_id, deposit_id=self.deposit_id,
                             bins=kwargs['bin_kwargs']['bins'])

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
        data = pd.concat([pd.DataFrame({'channel': [-1, -0], 'counts': [0, 0]}),
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
