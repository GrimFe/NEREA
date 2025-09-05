from collections.abc import Iterable
from inspect import signature
from dataclasses import dataclass
from typing import Self
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime, timedelta

from .utils import integral_v_u, _make_df, ratio_v_u, product_v_u, smoothing
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

    def smooth(self, **kwargs) -> Self:
        """
        Calculates the sum of 'value' and the minimum value of 'channel' for each
        group based on the integer division of 'channel' by 10.
        Contains the data used to find `max` and hence `R`.

        Parameters
        ----------
        **kwargs :
            arguments for the chosen utils.smoothing

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
        df['value'] = smoothing(df['value'], **kwargs)
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

    def rebin(self, bins: int=None, smooth: bool=True, **kwargs) -> pd.DataFrame:
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
        df = self.smooth(**kwargs).data if smooth else self.data.copy()
        if bins is not None:
            max_bins = self.data.channel.max()
            if bins > max_bins:
                warnings.warn(f"The maximum amount of allowed bins is {max_bins}. Bins set to {max_bins}.")
            bins_ = int(min(bins, max_bins))
            df['bins'] = pd.cut(df['channel'], bins=list(range(0, max_bins + 1, int(max_bins / bins_))))
            df = df.groupby('bins', as_index=False, observed=False
                            ).agg({'value': 'sum'}).drop('bins', axis=1
                                                            ).assign(channel=range(1, bins_+1))
        return df[['channel', 'value']]

    def get_max(self, fst_ch: int=None, **kwargs) -> pd.DataFrame:
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
            Defailt is empty, rading from nerea.defaults.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'channel' and 'value' columns.

        Examples
        --------
        >>> ffs = FissionFragmentSpectrum(...)
        >>> max_data = ffs.get_max(...)
        """
        kwargs = DEFAULT_BIN_KWARGS | kwargs

        reb = self.rebin(**kwargs)
        if fst_ch is None:
            lst_ch = reb[reb.value > 0].channel.max()
            fst_ch = reb[reb.value > 0].channel.min() + np.floor(lst_ch / 10)
        df = reb[reb.channel > fst_ch]
        return pd.DataFrame({"channel": [df.value.idxmax() + 1], "value": [df.value.max()]})

    def get_R(self, **kwargs) -> pd.DataFrame:
        """
        Filters data in channels above the channel of the spectrum maximum
        and returns the first row with value <= than the maximum.

        Parameters
        ----------
        bin_kwargs : dict, optional
            - bins : int
            - smooth : bool
            Defailt is empty, rading from nerea.defaults.
        max_kwargs : dict, optional
            Paramters for self.get_max()
            - fst_ch : int
            Defailt is empty, rading from nerea.defaults.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'channel' and 'value' columns.

        Examples
        --------
        >>> ffs = FissionFragmentSpectrum(...)
        >>> r_data = ffs.get_R(...)
        """
        kwargs = DEFAULT_MAX_KWARGS | DEFAULT_BIN_KWARGS | kwargs

        reb = self.rebin(**kwargs)
        max_ch = self.get_max(**kwargs).channel[0]
        data = reb.query("channel > @max_ch")
        return data[data.value <= self.get_max(**kwargs).value[0] / 2].iloc[0].to_frame().T[["channel", "value"]]

    def discriminators(self,
                       llds: Iterable[int | float]=[.15, .2, .25, .3, .35, .4, .45, .5, .55, .6],
                       **kwargs) -> np.array:
        """
        Calculates the discrimination levels to process.

        Parameters
        ----------
        llds : Iterable[int | float], optional
            Low level discriminators to consider.
            Iteger -> interpreted as absolute channel
            Float -> interpreted as fractiosn of R
            Default is 10 uniformly spaced from 0.15 to 0.65.
        bin_kwargs : dict, optional
            - bins : int
            - smooth : bool
            Defailt is empty, rading from nerea.defaults.
        max_kwargs : dict, optional
            Paramters for self.get_max()
            - fst_ch : int
            Defailt is empty, rading from nerea.defaults.

        Returns
        -------
        np.array
        """
        kwargs = DEFAULT_MAX_KWARGS | DEFAULT_BIN_KWARGS | kwargs
        r = kwargs.get('r', True)
        out = []
        for ch in llds:
            out.append(int(np.floor(ch * self.get_R(**kwargs).channel.iloc[0])) if r else int(np.round(ch)))
        return np.array(out)

    def integrate(self,
                  llds: Iterable[int | float]=[.15, .2, .25, .3, .35, .4, .45, .5, .55, .6],
                  r: bool=True,
                  raw_integral: bool=True,
                  **kwargs) -> pd.DataFrame:
        """
        Calculates the integral of data based on specified channels (as a function of R)
        and returns a DataFrame with channel, value, and uncertainty columns.

        Parameters
        ----------
        bin_kwargs : dict, optional
            - bins : int
            - smooth : bool
            Defailt is empty, rading from nerea.defaults.
        max_kwargs : dict, optional
            Paramters for self.get_max().
            - fst_ch : int
            Defailt is empty, rading from nerea.defaults.
        llds : Iterable[int|float] | int, optional
            low level discriminator(s) to integrate from.
            Defaults to 10 llds between [0.15, 0.6].
        r : bool, optional
            Defines whether the discriminators are absolute or
            fractions of the R channel.
            Default is True.
        raw_integral : bool, optional
            Defines whether to integrate the raw data or the
            smoothed ones.
            Default is False.

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
        llds_ = llds if isinstance(llds, Iterable) else [llds]
        kwargs = DEFAULT_MAX_KWARGS | DEFAULT_BIN_KWARGS | kwargs | {'llds': llds_, 'r': r}

        out = []
        data = self.data if raw_integral else self.rebin(**kwargs)
        data.value = data.value.astype('float64')
        discri = self.discriminators(**kwargs)
        for ch in discri:
            out.append(_make_df(*integral_v_u(data.query("channel >= @ch").value)))
        return pd.concat(out, ignore_index=True
                         ).assign(channel=discri, R=llds_ if r else [np.nan] * len(llds_)
                                  )[['channel', 'value', 'uncertainty', 'uncertainty [%]', 'R']]

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
                  visual: bool=False,
                  savefig: str='',
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
            r : are the llds fractions of the R channel?
    
        Examples
        --------
        >>> spectrum = pass
        >>> em = FFS.from_TKA(file).calibrate(spectrum)
        """
        kwargs = DEFAULT_MAX_KWARGS | DEFAULT_BIN_KWARGS | kwargs

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
        if visual or savefig:
            ax = self.plot(**kwargs)
            if savefig:
                ax.figure.savefig(savefig)
                plt.close()
        return EffectiveMass(data=data[["channel", "value", "uncertainty", "uncertainty [%]", "R"]],
                             composition=composition_.reset_index(names='nuclide'),
                             detector_id=self.detector_id,
                             deposit_id=self.deposit_id,
                             bins=self.rebin(**kwargs).channel.max())

    def plot(self, ax: plt.Axes=None, c: str='k', **kwargs) ->plt.Axes:
        """
        Plots the pulse height spectrum data.

        Parameters
        ----------
        phs_kwargs: dict
            Parameters to process the spectrum before plotting.
            - bin_kwargs
            - max_kwargs
            - llds
            - r
        ax : plt.Axes, optional
            Axes wehere to plot. Default is None.
        c : str, optional
            plot color. Default is `'k'`.

        Returns
        -------
        plt.Axes
        """
        kwargs = DEFAULT_MAX_KWARGS | DEFAULT_BIN_KWARGS | kwargs
        plt_kwargs = {k: v for k, v in kwargs.items() if k in set(signature(pd.DataFrame.plot).parameters)}

        ax = self.rebin(**kwargs).plot(x='channel', y='value', kind='scatter',
                                       s=10, c=c, ax=ax, **plt_kwargs)

        m = self.get_max(**kwargs)
        ax.scatter(x=m.channel.iloc[0], y=m.value.iloc[0], color='green', s=20, label="MAX")
        r = self.get_R(**kwargs)
        ax.scatter(x=r.channel.iloc[0], y=r.value.iloc[0], color='red', s=20, label="R")

        for i in self.discriminators(**kwargs):
            ax.axvline(i, color='red', alpha = 0.5, label=f"LLD: {i:.0f}", ls='--')
        ax.legend()
        ax.set_xlim([0, self.rebin(**kwargs).query("value >= 1").channel.iloc[-1]])
        ax.set_ylim([0, m.value.iloc[0] * 1.1])
        return ax

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
        data.columns = ['channel', 'value']
        data.channel += 1
        # GENIE overwrites the first two values with time indications
        # Here two zeros are added at the beginning of the data
        data = pd.concat([pd.DataFrame({'channel': [-1, -0], 'value': [0, 0]}),
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

    That is that the `self.data` will be the channel-wise sum of the values
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
        max = self.spectra[0].data.value.sum()
        out = self.spectra[0]
        for s in self.spectra[1:]:
            if s.data.value.sum() > max:
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
