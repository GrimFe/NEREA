from collections.abc import Iterable
from inspect import signature
from dataclasses import dataclass
from typing import Self
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime, timezone, timedelta

from .utils import integral_v_u, _make_df, ratio_v_u, product_v_u
from .functions import smoothing, get_relative_array, impurity_correction
from .constants import AVOGADRO, ATOMIC_MASS
from .effective_mass import EffectiveMass
from .count_rate import CountRate
from .defaults import *
from .classes import Xs

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "PulseHeightSpectrum",
    "PulseHeightSpectra"]

@dataclass(slots=True)
class PulseHeightSpectrum:
    """
    ``nerea.PulseHeightSpectrum``
    =============================
    Class storing and processing pulse height spectrum (PHS) data.
    Inherits from `nerea.Experimental`.

    Attributes
    ----------
    **start_time**: ``datetime.datetime``
        the PHS acquisition start date and time.
    **data**: ``pd.DataFrame``
        the PHS data.
    **campaign_id**: ``str``
        metadatata for expereimental campaign identification.
    **experiment_id**: ``str``
        metadatata for experiment identification.
    **detector_id**: ``str``
        metadatata for detector identification.
    **deposit_id**: ``str``
        metadatata for ionization chamber deposit identification.
    **location_id**: ``str``
        metadatata for expereimental location identification.
    **measurement_id**: ``str``
        metadatata for acquisition identification.
    **live_time**: ``int``
        the PHS acquisition live time.
    **real_time**: ``int``
        the PHS acquisition real time.
    **live_time_uncertainty**: ``float``, optional
        the PHS acquisition live time uncertainty.
        Default is ``0.0``.
    **real_time_uncertainty**: ``float``, optional
        the PHS acquisition real time uncertainty.
        Default is ``0.0``.
    __smoothing_verbose_printed: ``bool``, optional
        flag labelling whether the verbose message for smoothing
        was printed. Handled internally. Default is ``False``.
    __rebin_verbose_printed: ``bool``, optional
        flag labelling whether the verbose message for rebinning
        was printed. Handled internally. Default is ``False``.
    __max_verbose_printed: ``bool``, optional
        flag labelling whether the verbose message for maximum search
        was printed. Handled internally. Default is ``False``.
    __r_verbose_printed: ``bool``, optional
        flag labelling whether the verbose message for R channel
        was printed. Handled internally. Default is ``False``."""
    start_time: datetime
    data: pd.DataFrame
    campaign_id: str
    experiment_id: str
    detector_id: str
    deposit_id: str
    location_id: str
    measurement_id: str
    live_time: int
    real_time: int
    live_time_uncertainty: float = 0.0
    real_time_uncertainty: float = 0.0
    __smoothing_verbose_printed: bool=False
    __rebin_verbose_printed: bool=False
    __max_verbose_printed: bool=False
    __r_verbose_printed: bool=False

    def smooth(self, **kwargs) -> Self:
        """
        `nerea.PulseHeightSpectrum.smooth()`
        ------------------------------------
        Calculates the sum of 'value' and the minimum value of 'channel' for each
        group based on the integer division of 'channel' by 10.
        Contains the data used to find `max` and hence `R`.

        Parameters
        ----------
        **kwargs
            arguments for the chosen ``nerea.functions.smoothing``

        Returns
        -------
        ``nerea.PulseHeightSpectrum``
            With the smoothed pulse height spectrum as data.
            
        Notes
        -----
        Allowed methods are
            - ``'moving_average'`` (requires ``window``)
            - ``'ewm'``
            - ``'savgol_filter'`` (requires ``window_length``, ``polyorder``)
            - ``'fit'``(requires ``ch_before_max``, ``order``)"""
        if kwargs.get('verbose', False) and not self.__smoothing_verbose_printed:
            sm = kwargs.get('smoothing_method', 'no smoothing method')
            if kwargs.get('window', False):
                w = kwargs.get('window', False)
            else:
                w = kwargs.get('window_length', False)
            self.__smoothing_verbose_printed = True
            logger.info(f"Smoothing PHS with {sm} and window length {w}.")
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
            live_time = self.live_time,
            real_time = self.real_time,
            live_time_uncertainty = self.live_time_uncertainty,
            real_time_uncertainty = self.real_time_uncertainty
        )        
        return out

    def rebin(self, bins: int=None, smooth: bool=True, **kwargs) -> Self:
        """
        `nerea.PulseHeightSpectrum.rebin()`
        ------------------------------------
        Rebins the spectrum.

        Parameters
        ----------
        **bins** : ``int``, optional
            Number of bins for rebinned spectrum.
            Recommended values are 4096, 2048, 1024, 512.
            Defaults to `None` for no rebinning.
        **smooth** : ``bool``, optional
            Flag to rebin smoothened spectrum.
            Defaults to True.
        **kwargs
            Additional arguments for ``self.smooth()``

            - **renormalize** (``bool``): Whether to renormalize the smoothed spectrum.
            - **smoothing_method** (``str``): The mehtod to implement for smoothing.
            - arguments for the chosen ``utils.smoothing``

        Returns
        -------
        ``nerea.PulseHeightSpectrum``
            Rebinned spectrum."""
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
            if kwargs.get('verbose', False) and not self.__rebin_verbose_printed:
                self.__rebin_verbose_printed = True
                logger.info(f"Rebinning PHS to {bins} bins.")
        out = self.__class__(
                start_time = self.start_time,
                data = df[['channel', 'value']],
                campaign_id = self.campaign_id,
                experiment_id = self.experiment_id,
                detector_id = self.detector_id,
                deposit_id = self.deposit_id,
                location_id = self.location_id,
                measurement_id = self.measurement_id,
                live_time = self.live_time,
                real_time = self.real_time,
                live_time_uncertainty = self.live_time_uncertainty,
                real_time_uncertainty = self.real_time_uncertainty
            )        
        return out
    
    def _get_processing_chs(self, fst_ch: int|str=None, **kwargs) -> tuple[int]:
        """
        `nerea.PulseHeightSpectrum._get_processing_chs()`
        -------------------------------------------------
        Method to find the channels used to
        serach a PHS maximum from.

        Parameters
        ----------
        **fst_ch** : ``int | str``, optional
            Left channel to search the maximum from. Defaults to `None`
            for automatic 1/10 of total channels acquired.
            Or maximum search option
            - ``'valley'``: backwards valley search from 1/10 of acquired channels.
            - ``'iterative'``: to search for the valley
                            iteratively from the left.

        Returns
        -------
        ``tuple[int]``
            Channel to start max serach from, Last channel with counts."""
        kwargs = DEFAULT_BIN_KWARGS | kwargs

        reb = self.rebin(**kwargs).data
        lst_ch = reb[reb.value > 0].channel.max()
        if fst_ch is None:
            fst_ch = reb[reb.value > 0].channel.min() + np.floor(lst_ch / 10)
            if kwargs.get('verbose', False) and not self.__max_verbose_printed:
                logger.info(f"Searching PHS maximum from {fst_ch}.")
        elif fst_ch == 'valley':
            if kwargs.get('verbose', False) and not self.__max_verbose_printed:
                logger.info(f"Searching PHS maximum looking for PHS valley.")
            fst_ch = reb[reb.value > 0].channel.min() + np.floor(lst_ch / 10)
            # supposedly, this locates the rising edge of the PHS
            do = True
            while do:
                try:
                    vfst = reb.query('channel == @fst_ch').value.iloc[0]
                    vleft = reb.query('channel == @fst_ch - 50').value.iloc[0]
                    do = vleft < vfst
                    if do: fst_ch -= 50
                except IndexError:
                    # then the search has reached its end
                    do = False
        elif fst_ch == 'iterative':
            if kwargs.get('verbose', False) and not self.__max_verbose_printed:
                logger.info(f"Searching PHS maximum iteratively.")
            fst_ch = reb.query('value > 20').channel.iloc[0]
            do = True
            while do:
                new_ch = fst_ch + 70
                mfst = reb.query('channel >= @fst_ch').value.max()
                mnew = reb.query('channel >= @new_ch').value.max()
                do = (mfst != mnew)
                fst_ch = new_ch
        return fst_ch, lst_ch

    def get_max(self, **kwargs) -> pd.DataFrame:
        """
        `nerea.PulseHeightSpectrum.get_max()`
        -------------------------------------
        Finds the channel with the maximum count value in a DataFrame.

        Parameters
        ----------
        **kwargs
            Additional arguments for
            
            - ``self.rebin()``
                - **bins** (``int``): number of bins
                - **smooth** (``bool``): whether to smooth the PHS

            - ``self.smooth()`` only if ``smooth == True``
                - **renormalize** (``bool``): Whether to renormalize the smoothed spectrum.
                - **smoothing_method** (``str``): The mehtod to implement for smoothing.
                - arguments for the chosen ``utils.smoothing``.

        Returns
        -------
        ``pd.DataFrame``
            DataFrame with 'channel' and 'value' columns.

        Note
        --------
        First channel finding is handled by _get_fst_ch()."""
        kwargs = DEFAULT_BIN_KWARGS | kwargs

        reb = self.rebin(**kwargs).data
        fst_ch, _ = self._get_processing_chs(**kwargs)
        df = reb[reb.channel > fst_ch]
        if kwargs.get('verbose', False) and not self.__max_verbose_printed:
            self.__max_verbose_printed = True
            logger.info(f"PHS maximum found from first channel {fst_ch}: channel {df.value.idxmax() + 1}.")
        return pd.DataFrame({"channel": [df.value.idxmax() + 1], "value": [df.value.max()]})

    def get_R(self, **kwargs) -> pd.DataFrame:
        """
        `nerea.PulseHeightSpectrum.get_R()`
        -----------------------------------
        Filters data in channels above the channel of the spectrum maximum
        and returns the first row with value <= than the maximum.

        Parameters
        ----------
        **kwargs
            Additional arguments for
            
            - ``self.rebin()``
                - **bins** (``int``): number of bins
                - **smooth** (``bool``): whether to smooth the PHS

            - ``self.smooth()`` only if ``smooth == True``
                - **renormalize** (``bool``): Whether to renormalize the smoothed spectrum.
                - **smoothing_method** (``str``): The mehtod to implement for smoothing.
                - arguments for the chosen ``utils.smoothing``.

            - ``self.get_max()``
                - **fst_ch** (``int | str``): channel to start max search or max search method.

        Returns
        -------
        ``pd.DataFrame``
            DataFrame with 'channel' and 'value' columns."""
        kwargs = DEFAULT_MAX_KWARGS | DEFAULT_BIN_KWARGS | kwargs

        reb = self.rebin(**kwargs).data
        max_ch = self.get_max(**kwargs).channel[0]
        data = reb.query("channel > @max_ch")
        out = data[data.value <= self.get_max(**kwargs).value[0] / 2].iloc[0].to_frame().T[["channel", "value"]]
        if kwargs.get('verbose', False) and not self.__r_verbose_printed:
            self.__r_verbose_printed = True
            logger.info(f"PHS R channel found: {out.channel.iloc[0]}.")
        return out

    def discriminators(self,
                       llds: Iterable[int | float]=[.15, .2, .25, .3, .35, .4, .45, .5, .55, .6],
                       **kwargs) -> np.array:
        """
        `nerea.PulseHeightSpectrum.discriminators()`
        --------------------------------------------
        Calculates the discrimination levels to process.

        Parameters
        ----------
        **llds** : ``Iterable[int | float]``, optional
            Low level discriminators to consider.
            Iteger -> interpreted as absolute channel
            Float -> interpreted as fractiosn of R
            Default is 10 uniformly spaced from 0.15 to 0.65.
        **kwargs
            Additional arguments for
            
            - ``self.rebin()``
                - **bins** (``int``): number of bins
                - **smooth** (``bool``): whether to smooth the PHS

            - ``self.smooth()`` only if ``smooth == True``
                - **renormalize** (``bool``): Whether to renormalize the smoothed spectrum.
                - **smoothing_method** (``str``): The mehtod to implement for smoothing.
                - arguments for the chosen ``utils.smoothing``.

            - ``self.get_max()``
                - **fst_ch** (``int | str``): channel to start max search or max search method.

        Returns
        -------
        ``np.array``"""
        kwargs = DEFAULT_MAX_KWARGS | DEFAULT_BIN_KWARGS | kwargs
        r = kwargs.get('r', True)
        out = []
        for ch in llds:
            out.append(int(np.floor(ch * self.get_R(**kwargs).channel.iloc[0])) if r else int(np.round(ch)))
        return np.array(out, dtype=np.int32)

    def integrate(self,
                  llds: Iterable[int | float]=[.15, .2, .25, .3, .35, .4, .45, .5, .55, .6],
                  r: bool=True,
                  raw_integral: bool=True,
                  **kwargs) -> pd.DataFrame:
        """
        `nerea.PulseHeightSpectrum.integrate()`
        ---------------------------------------
        Calculates the integral of data based on specified channels (as a function of R)
        and returns a DataFrame with channel, value, and uncertainty columns.

        Parameters
        ----------
        **llds** : ``Iterable[int|float] | int``, optional
            low level discriminator(s) to integrate from.
            Defaults to 10 llds between [0.15, 0.6].
        **r** : ``bool``, optional
            Defines whether the discriminators are absolute or
            fractions of the R channel.
            Default is ``True``.
        **raw_integral** : ``bool``, optional
            Defines whether to integrate the raw data or the
            smoothed ones.
            Default is ``False``.

        **kwargs
        Additional arguments for

        - ``self.rebin()``
            - **bins** (``int``): number of bins
            - **smooth** (``bool``): whether to smooth the PHS

        - ``self.smooth()`` only if ``smooth == True``
            - **renormalize** (``bool``): Whether to renormalize the smoothed spectrum.
            - **smoothing_method** (``str``): The mehtod to implement for smoothing.
            - arguments for the chosen ``nerea.functions.smoothing``.

        - ``self.get_max()``
            - **fst_ch** (``int | str``): channel to start max search or max search method.

        Returns
        -------
        ``pd.DataFrame``
            DataFrame with ``'channel'``, ``'value'``, and ``'uncertainty'`` columns.

        Notes
        -----
        - ``llds`` are handled by ``self.discriminators``"""
        llds_ = llds if isinstance(llds, Iterable) else [llds]
        kwargs = DEFAULT_MAX_KWARGS | DEFAULT_BIN_KWARGS | kwargs | {'llds': llds_, 'r': r}

        out = []
        data2integrate_kw = kwargs | {'smooth': not raw_integral}
        data = self.rebin(**data2integrate_kw).data.copy()
        data.value = data.value.astype('float64')
        discri = self.discriminators(**kwargs)
        for ch in discri:
            out.append(_make_df(*integral_v_u(data.query("channel >= @ch").value)))
        if kwargs.get('verbose', False):
            r = kwargs.get('r', False)
            llds = kwargs.get('llds', False)
            logger.info(f"PHS integration with r method: {r}.")
            if not r: logger.info(f"LLDs are {llds}.")
        return pd.concat(out, ignore_index=True
                         ).assign(channel=discri, R=llds_ if r else [np.nan] * len(llds_)
                                  )[['channel', 'value', 'uncertainty', 'uncertainty [%]', 'R']]

    @staticmethod
    def _get_calibration_coefficient(one_group_xs: Xs,
                                     composition: dict[str, float] | pd.DataFrame) -> pd.DataFrame:
        """
        `nerea.PulseHeightSpectrum._get_calibration_coefficient()`
        ----------------------------------------------------------
        Calculates the fission chamber calibration coefficient.

        Paramters
        ---------
        **one_group_xs** : ``nerea.Xs``
            the one group cross sections of the fission
            chamber components. `key` is the nuclide
            string identifier (e.g., `'U235'`), and `value`
            is its one group cross section.
            Has columns for its value and uncertainty.
        **composition** : ``dict[str, float] | pd.DataFrame``
            the fission chamber composition relative to
            the total. `key` is the nuclide string identifier
            (e.g., `'U235'`), and `value` is its atomic
            abundance relative to the total. Has columns
            for its value and uncertainty.

        Returns
        -------
        ``pd.DataFrame``"""
        _c = get_relative_array(composition)
        main = _c.loc[_c.value.idxmax()].name
        # calculation of the sum over nuclides in the deposit (n * xs)
        # impurity_correction requires non-normalized composition
        ic = impurity_correction(one_group_xs,
                                 composition,
                                 xs_den='',
                                 drop_main=False)
        # this can be done after impurity correction calculation
        # because we assume no uncertainty on Nav and A
        a = _make_df(*ratio_v_u(AVOGADRO, ATOMIC_MASS.loc[main]))
        units_of_ug = _make_df(1e-6, 0)
        return _make_df(*product_v_u([a, units_of_ug, ic]))

    def calibrate(self,
                  k: pd.DataFrame,
                  composition: dict[str, float] | pd.DataFrame,
                  monitor: CountRate,
                  one_group_xs: Xs,
                  visual: bool=False,
                  savefig: str='',
                  **kwargs) -> EffectiveMass:
        """
        `nerea.PulseHeightSpectrum.calibrate()`
        ---------------------------------------
        Computes the fission chamber effective mass from the pulse
        height spectrum.

        Parameters
        ----------
        **k**: ``pd.DataFrame``,
            the facility calibration factor as in
            "Miniature fission chambers calibration in
            pulse mode: interlaboratory comparison at
            the SCK CEN BR1 and CEA CALIBAN reactors".
            Has columns for its value and uncertainty.
        **composition**: ``nerea.Xs``
            the fission chamber composition relative to
            the total. `key` is the nuclide string identifier
            (e.g., `'U235'`), and `value` is its atomic
            abundance relative to the total. Has columns
            for its value and uncertainty.
        **one_group_xs**: ``dict[str, float]``
            the one group cross sections of the fission
            chamber components. `key` is the nuclide
            string identifier (e.g., `'U235'`), and `value`
            is its one group cross section.
            Has columns for its value and uncertainty.
        **monitor**: ``nerea.ReactionRate``
            the counts of the monitor fission chamber used
            during calibration.
        **kwargs
        Additional arguments for

        - ``self.rebin()``
            - **bins** (``int``): number of bins
            - **smooth** (``bool``): whether to smooth the PHS

        - ``self.smooth()`` only if ``smooth == True``
            - **renormalize** (``bool``): Whether to renormalize the smoothed spectrum.
            - **smoothing_method** (``str``): The mehtod to implement for smoothing.
            - arguments for the chosen ``utils.smoothing``.

        - ``self.get_max()``
            - **fst_ch** (``int | str``): channel to start max search or max search method.

        - ``self.integrate()``
            - **llds** (``Iterable[int | float]``): Low level discriminators.
            - **r** (``bool``): Whether the ``llds`` are fractions of the R channel.

        Returns
        -------
        ``nerea.EffectiveMass``"""
        kwargs = DEFAULT_MAX_KWARGS | DEFAULT_BIN_KWARGS | kwargs

        composition_ = pd.DataFrame(composition, index=['value', 'uncertainty']
            ).T if not isinstance(composition, pd.DataFrame
                                  ) else composition.copy()
        c = self._get_calibration_coefficient(one_group_xs, composition_)

        pm = monitor.average(self.start_time, self.real_time)
        kmc = _make_df(*product_v_u([k, pm, c]))
        integral = self.integrate(**kwargs)
        integral.index = ['value'] * integral.shape[0]
        time = pd.concat([_make_df(self.live_time, self.live_time_uncertainty)
                          ] * integral.shape[0])
        integral = _make_df(*ratio_v_u(integral, time)).assign(channel=integral.channel.values,
                                     R=integral.R.values)
        data = _make_df(*ratio_v_u(integral, kmc)).assign(channel=integral.channel, R=integral.R)
        if visual or savefig:
            fig, axs = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'wspace': 0.4})
            # PHS plot
            self.plot(ax=axs[0], **kwargs)
            # EM table
            cell_text = [['{:.0f}'.format(r.channel),
                          '{:.2f}'.format(r.value)
                          ] for _, r in data.iterrows()]
            tab = axs[0].table(cellText=cell_text, colLabels=['ch', 'm [ug]'],
                               bbox=[1.01, 0, 0.275, 1])
            tab.auto_set_font_size(False)
            # Power monitor plot
            monitor.plot(ax=axs[1], start_time=self.start_time, duration=self.real_time)
            axs[1].set_xlim([self.start_time - timedelta(seconds=60),
                             self.start_time + timedelta(seconds=60 + self.real_time)])
            if savefig:
                fig.savefig(savefig)
                plt.close()
        return EffectiveMass(data=data[["channel", "value", "uncertainty", "uncertainty [%]", "R"]].reset_index(drop=True),
                             composition=composition_.reset_index(names='nuclide'),
                             detector_id=self.detector_id,
                             deposit_id=self.deposit_id,
                             bins=self.rebin(**kwargs).data.channel.max())

    def plot(self, ax: plt.Axes=None, c: str='k', **kwargs) ->plt.Axes:
        """
        `nerea.PulseHeightSpectrum.plot()`
        ----------------------------------
        Plots the pulse height spectrum data.

        Parameters
        ----------
        ax : ``plt.Axes``, optional
            Axes wehere to plot. Default is None.
        c : ``str``, optional
            plot color. Default is `'k'`.
        **kwargs
        Additional arguments for

        - ``self.rebin()``
            - **bins** (``int``): number of bins
            - **smooth** (``bool``): whether to smooth the PHS

        - ``self.smooth()`` only if ``smooth == True``
            - **renormalize** (``bool``): Whether to renormalize the smoothed spectrum.
            - **smoothing_method** (``str``): The mehtod to implement for smoothing.
            - arguments for the chosen ``utils.smoothing``.

        - ``self.get_max()``
            - **fst_ch** (``int | str``): channel to start max search or max search method.

        - ``self.integrate()``
            - **llds** (``Iterable[int | float]``): Low level discriminators.
            - **r** (``bool``): Whether the ``llds`` are fractions of the R channel.

        Returns
        -------
        ``plt.Axes``"""
        kwargs = DEFAULT_MAX_KWARGS | DEFAULT_BIN_KWARGS | kwargs
        plt_kwargs = {k: v for k, v in kwargs.items() if k in set(signature(pd.DataFrame.plot).parameters)}

        ax = self.rebin(**kwargs).data.plot(x='channel', y='value', kind='scatter',
                                       s=10, c=c, ax=ax, **plt_kwargs)

        m = self.get_max(**kwargs)
        ax.scatter(x=m.channel.iloc[0], y=m.value.iloc[0], color='green', s=20, label=f"MAX: {m.channel.iloc[0]:.0f}")
        r = self.get_R(**kwargs)
        ax.scatter(x=r.channel.iloc[0], y=r.value.iloc[0], color='red', s=20, label=f"R: {r.channel.iloc[0]:.0f}")

        for i in self.discriminators(**kwargs):
            ax.axvline(i, color='red', alpha = 0.5, label=f"LLD: {i:.0f}", ls='--')
        fst, lst = self._get_processing_chs(**kwargs)
        ax.axvline(fst, color='green', alpha = 0.5, label=f"FST: {fst:.0f}", ls='--')
        ax.legend()
        ax.set_xlim([0, lst])
        ax.set_ylim([0, m.value.iloc[0] * 1.1])
        return ax

    @classmethod
    def from_TKA(cls, file: str, **kwargs) -> Self:
        """
        `nerea.PulseHeightSpectrum.from_TKA()`
        --------------------------------------
        Reads data from a TKA file to create a `PulseHeightSpectrum` instance.

        Parameters
        ----------
        **file** : ``str``
            TKA file path.
        **kwargs
            Keyword arguments for class initialization.

            - **start_time** (``datetime.datetime``) the PHS acquisition start date and time.
            - **campaign_id** (``str``) expereimental campaign identifier.
            - **experiment_id** (``str``) experiment identifier.
            - **detector_id** (``str``) detector identifier.
            - **deposit_id** (``str``) ionization chamber deposit identifier.
            - **location_id** (``str``) expereimental location identifier.
            - **measurement_id** (``str``) acquisition identifier.
            - **live_time** (``int``) the PHS acquisition life time.
            - **real_time** (``int``) the PHS acquisition real time.
            - **live_time_uncertainty**: (``float``, optional) life time uncertainty. Default is `0.0`.
            - **real_time_uncertainty**: (``float``, optional) real time uncertainty. Default is `0.0`.

        Returns
        -------
        ``nerea.PulseHeightSpectrum``"""
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
    def from_formatted_TKA(cls, file: str, **kwargs) -> Self:
        """
        `nerea.PulseHeightSpectrum.from_formatted_TKA()`
        ------------------------------------------------
        Reads data from a formatted TKA file and extracts metadata from the
        file name to create a `PulseHeightSpectrum` instance.
        The filename is expected to be formatted as:
        {Campaign}_{Experiment}_{Detector}_{Deposit}_{Location}_{Measurement}.TKA
        Requires a text file with the same name with time information.

        Parameters
        ----------
        **file** : ``str``
            TKA file path.
        **kwargs
            Keyword arguments for class initialization
            - **live_time_uncertainty**: (``float``, optional) life time uncertainty. Default is `0.0`.
            - **real_time_uncertainty**: (``float``, optional) real time uncertainty. Default is `0.0`.
            Other nerea.PulseHeightSpectrum initialization kwargs can be overwritten.

        Returns
        -------
        ``nerea.PulseHeightSpectrum``

        Examples
        --------
        >>> ffs = PulseHeightSpectrum.from_formatted_TKA(
        f'{Campaign}_{Experiment}_{Detector}_{Deposit}_{Location}_{Measurement}.TKA')"""
        with open(file.replace('.TKA', '.txt'), 'r') as f:
            start, life, real = f.readlines()
        campaign_id, experiment_id, detector_id, deposit_id, location_id, measurement_id = file.split('\\')[-1].split('_')
        dct = {
            'start_time': datetime.strptime(start, "%Y-%m-%d %H:%M:%S\n"),
            'live_time': float(life),
            'real_time': float(real),
            'campaign_id':campaign_id,
            'experiment_id': experiment_id,
            'detector_id': detector_id,
            'deposit_id': deposit_id,
            'location_id': location_id,
            'measurement_id': measurement_id.replace(".TKA", ""),
        } | kwargs
        return cls.from_TKA(file, **dct)

    @classmethod
    def from_CNF(cls, file: str, **kwargs) -> Self:
        """
        `nerea.PulseHeightSpectrum.from_CNF()`
        --------------------------------------
        Reads data from a CNF file to create a `PulseHeightSpectrum` instance.

        Parameters
        ----------
        **file** : ``str``
            CNF file path.
        **kwargs
            Keyword arguments for class initialization.

            - **campaign_id** (``str``) expereimental campaign identifier.
            - **experiment_id** (``str``) experiment identifier.
            - **detector_id** (``str``) detector identifier.
            - **deposit_id** (``str``) ionization chamber deposit identifier.
            - **location_id** (``str``) expereimental location identifier.
            - **measurement_id** (``str``) acquisition identifier.
            - **live_time_uncertainty**: (``float``, optional) life time uncertainty. Default is `0.0`.
            - **real_time_uncertainty**: (``float``, optional) real time uncertainty. Default is `0.0`.

        Returns
        -------
        ``nerea.PulseHeightSpectrum``
        
        Notes
        -----
        **start_time**, **live_time**, **real_time** are read from the file."""
        def uint8_at(f, pos):
            f.seek(pos)
            return np.fromfile(f, dtype=np.dtype('<u1'), count=1)[0]

        def uint32_at(f, pos, count=1):
            f.seek(pos)
            if count == 1:
                return np.fromfile(f, dtype=np.dtype('<u4'), count=count)[0]
            else:
                return np.fromfile(f, dtype=np.dtype('<u4'), count=count)

        def uint64_at(f, pos):
            f.seek(pos)
            return np.fromfile(f, dtype=np.dtype('<u8'), count=1)[0]

        def uint16_at(f, pos):
            f.seek(pos)
            return np.fromfile(f, dtype=np.dtype('<u2'), count=1)[0]

        def time_at(f, pos):
            return ~uint64_at(f, pos)*1e-7

        def datetime_at(f, pos):
            return uint64_at(f, pos) / 10000000 - 3506716800
        
        with open(file, 'rb') as f:
            i = 0
            leave = False
            while not leave:
                # List of available section headers
                sec_header = 0x70 + i*0x30
                i += 1
                sec_id = uint32_at(f, sec_header)
                match sec_id:
                    case 73728:  # reads time data
                        offs_p = uint32_at(f, sec_header + 0x0a)
                        offs_t = offs_p + 0x30 + uint16_at(f, offs_p + 0x24)        
                        st = datetime.fromtimestamp(datetime_at(f, offs_t + 0x01),
                                                    tz=timezone.utc).replace(tzinfo=None)
                        rt = time_at(f, offs_t + 0x09)
                        lt = time_at(f, offs_t + 0x11)                
                    case 73733:  # reads counts
                        offs_c = uint32_at(f, sec_header + 0x0a)
                        n_channels = uint8_at(f, offs_p + 0x00ba) * 256
                        # Data in each channel
                        value = uint32_at(f, offs_c + 0x200, n_channels)
                        channel = np.arange(1, n_channels+1, 1)
                    case 0:  # leaves the loop
                        leave = True
                    case _:
                        pass
        return cls(start_time=st,
                   data=pd.DataFrame({'value': value, 'channel': channel}),
                   live_time=lt,
                   real_time=rt,
                   **kwargs)

    @classmethod
    def from_formatted_CNF(cls, file: str, **kwargs) -> Self:
        """
        `nerea.PulseHeightSpectrum.from_formatted_CNF()`
        ------------------------------------------------
        Reads data from a formatted CNF file and extracts metadata from the
        file name to create a `PulseHeightSpectrum` instance.
        The filename is expected to be formatted as:
        {Campaign}_{Experiment}_{Detector}_{Deposit}_{Location}_{Measurement}.CNF
        Requires a text file with the same name with time information.

        Parameters
        ----------
        **file** : ``str``
            CNF file path.
        **kwargs
            Keyword arguments for class initialization
            - **live_time_uncertainty**: (``float``, optional) life time uncertainty. Default is `0.0`.
            - **real_time_uncertainty**: (``float``, optional) real time uncertainty. Default is `0.0`.
            Other nerea.PulseHeightSpectrum initialization kwargs can be overwritten.

        Returns
        -------
        ``nerea.PulseHeightSpectrum``

        Examples
        --------
        >>> ffs = PulseHeightSpectrum.from_formatted_CNF(
        f'{Campaign}_{Experiment}_{Detector}_{Deposit}_{Location}_{Measurement}.CNF')"""
        campaign_id, experiment_id, detector_id, deposit_id, location_id, measurement_id = file.split('\\')[-1].split('_')
        dct = {
            'campaign_id':campaign_id,
            'experiment_id': experiment_id,
            'detector_id': detector_id,
            'deposit_id': deposit_id,
            'location_id': location_id,
            'measurement_id': measurement_id.replace(".CNF", ""),
        } | kwargs
        return cls.from_CNF(file, **dct)

@dataclass(slots=True)
class PulseHeightSpectra():
    """
    ``nerea.PulseHeightSpectra``
    ============================
    Class storing and processing pulse height spectrum (PHS) data.
    Inherits from `nerea.Experimental` over several acquisitions.

    This class works under the assumption that no measurement time was
    lost in the process of measuring the pulse height spectra.

    That is that the `self.data` will be the channel-wise sum of the values
    of the listed pulse height spectra, while the start time of the
    measurement will be the minimum start time and life and real times will
    be the sum of the respective times in the listed pulse height spectra.

    Attributes
    ----------
    **spectra**: ``Iterable[`nerea.PulseHeightSpectrum`]``
        the listed spectra included in the analysis.
    _enable_checks: ``bool``, optoinal
        flag enabling consistency checks. Default is `True`."""
    spectra: Iterable[PulseHeightSpectrum]
    _enable_checks: bool = True

    def __post_init__(self) -> None:
        if self._enable_checks:
            self._check_consistency()

    def __iter__(self):
        return self.spectra.__iter__()

    def __getitem__(self, item) -> PulseHeightSpectrum:
         return self.spectra[item]

    def _check_consistency(self) -> None:
        """
        `nerea.PulseHeightSpectra._check_consistency()`
        -----------------------------------------------
        Checks the consistency of
            - ``campaign_id``
            - ``experiment_id``
            - ``detector_id``
            - ``deposit_id``
            - ``location_id``
        among self.spectra"""
        # should also check the consistency of the ffs?
        should = ['campaign_id', 'experiment_id']
        must = ['detector_id', 'deposit_id', 'location_id']
        for attr in should:
            if not all([getattr(ffs, attr) == getattr(self.spectra[0], attr) for ffs in self.spectra]):
                warnings.warn(f"Inconsistent {attr} among different PulseHeightSpectrum instances.")
        for attr in must:
            if not all([getattr(ffs, attr) == getattr(self.spectra[0], attr) for ffs in self.spectra]):
                raise Exception(f"Inconsistent {attr} among different PulseHeightSpectrum instances.")

    @property
    def best(self) -> PulseHeightSpectrum:
        """
        `nerea.PulseHeightSpectra.best()`
        ---------------------------------
        Returns the pulse height spectrum with the highest sum value.

        Returns
        -------
        ``nerea.PulseHeightSpectrum``
            Pulse height spectrum with the highest integral count."""
        max = self.spectra[0].data.value.sum()
        out = self.spectra[0]
        for s in self.spectra[1:]:
            if s.data.value.sum() > max:
                out = s
        return out

    @classmethod
    def from_formatted_TKA(cls, files: Iterable[str], **kwargs) -> Self:
        """
        `nerea.PulseHeightSpectra.from_formatted_TKA()`
        -----------------------------------------------
        Reads a list of files to create a PulseHeightSpectra object.
        Each filename is expected to be formatted as:
        {Campaign}_{Experiment}_{Detector}_{Deposit}_{Location}_{Measurement}.TKA
        Each file requires a text file with the same name with time information.

        Parameters
        ----------
        **files** : ``Iterable[str]``
            List of file paths.
        **kwargs
            Keyword arguments for class initialization
            - **live_time_uncertainty**: (``float``, optional) life time uncertainty. Default is `0.0`.
            - **real_time_uncertainty**: (``float``, optional) real time uncertainty. Default is `0.0`.
            Other nerea.PulseHeightSpectrum initialization kwargs can be overwritten.
            The same is passed to all instances.

        Returns
        -------
        ``nerea.PulseHeightSpectra``"""
        data = []
        for file in files:
            data.append(cls.from_formatted_TKA(file, **kwargs))
        return cls(data)
