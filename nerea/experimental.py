import serpentTools as sts  ## impurity correction
from collections.abc import Iterable
from dataclasses import dataclass

from .pulse_height_spectrum import PulseHeightSpectrum
from .effective_mass import EffectiveMass
from .count_rate import CountRate, CountRates
from .utils import ratio_v_u, product_v_u, _make_df
from .functions import impurity_correction
from .constants import ATOMIC_MASS
from .defaults import *
from .classes import Xs

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

import logging

logger = logging.getLogger(__name__)

__all__ = ['_Experimental',
           'NormalizedPulseHeightSpectrum',
           'SpectralIndex',
           'Traverse']


def average_v_u(df: pd.DataFrame) -> tuple[float, float]:
    """
    `nerea.experimental.average_v_u()`
    ------------------------------------
    Computes average value and uncertainty.

    Parameters
    ----------
    **df** : ``pd.DataFrame``
        A data frame with `'value'` and `'uncertainty'`
        columns to take the average of.
    
    Returns
    -------
    ``tuple[float, float]``
        The average value and uncertainty."""
    v = df.value.mean()
    u = sum(df.uncertainty **2) / len(df.uncertainty)
    return v, u

@dataclass(slots=True)
class _Experimental:
    """
    ``nerea._Experimental``
    =======================
    Superclass for experimental results.
    """
    def process(self) -> None:
        """
        Placeholder for inheriting classes.
        """
        return None


@dataclass(slots=True)
class NormalizedPulseHeightSpectrum(_Experimental):
    """
    ``nerea.NormalizedPulseHeightSpectrum``
    =======================================
    Class storing and processing the pulse height spectrum
    (PHS) normalization per unit mass, power and time.
    Inherits from ``nerea.Experimental``.

    Attributes
    ----------
    **phs** : ``nerea.PulseHeightSpectrum``
        the pulse height spectrum to normalize.
    **effective_mass** : ``nerea.EffectiveMass``
        the effective mass of the fission chamber used to acquire
        the PHS.
    **power_monitor** : ``nerea.CountRate``
        the power monitor of the PHS acquisition.
    _enable_checks: ``bool``, optoinal
        flag enabling consistency checks. Default is ``True``."""
    phs: PulseHeightSpectrum
    effective_mass: EffectiveMass
    power_monitor: CountRate
    _enable_checks: bool = True

    def __post_init__(self) -> None:
        """"
        Runs consistency checks.
        """
        if self._enable_checks:
            self._check_consistency()

    def _check_consistency(self) -> None:
        """
        ``nerea.NormalizedPulseHeightSpectrum._check_consistency()``
        ------------------------------------------------------------
        Checks the consistency of:
            - ``experiment_id``
            - ``detector_id``
            - ``deposit_id``
        among ``self.pulse_height_spectrum``
        and also checks:
            - ``R_channel``
        between ``self.pulse_height_spectrum`` and ``self.effective_mass``
        via ``_check_ch_equality(tolerance=0.01)``.

        Raises
        ------
        Exception
            If there are inconsistencies among the IDs or R channel values."""
        if not self.phs.detector_id == self.effective_mass.detector_id:
            raise Exception('Inconsistent detectors among PulseHeightSpectrum and EffectiveMass')
        if not self.phs.deposit_id == self.effective_mass.deposit_id:
            raise Exception('Inconsistent deposits among PulseHeightSpectrum and EffectiveMass')
        if not self.phs.experiment_id == self.power_monitor.experiment_id:
            raise Exception('Inconsitent experiments among PulseHeightSpectrum and CountRate')
        if not self._check_ch_equality():
            ch = self.phs.get_R(bin_kwargs={'bins': self.effective_mass.bins}).channel
            msg = f"R channel difference: {((ch - self.effective_mass.R_channel) / self.effective_mass.R_channel * 100).iloc[0]} %"
            warnings.warn(msg)

    def _check_ch_equality(self, tolerance:float =0.01) -> bool:
        """
        ``nerea.NormalizedPulseHeightSpectrum._check_ch_equality()``
        ------------------------------------------------------------
        Checks consistency of the R channels of ``self.pulse_height_spectrum`` and
        ``self.effective_mass`` within a specified tolerance.
        The check happens only if the binning of the two objects is the same.
        
        Parameters
        ----------
        **tolerance** : ``float``, optional
            The acceptable relative difference between the R channel of
            ``self.pulse_height_spectrum`` and ``self.effective_mass``.
            Default is ``0.01``.

        Returns
        -------
        ``bool``
            Indicating whether the relative difference between the R channels
            is within tolerance."""
        if self.phs.data.channel.max() == self.effective_mass.bins:
            check = abs(self.phs.get_R(
                            bin_kwargs={'bins': self.effective_mass.bins}
                            ).channel.iloc[0] - self.effective_mass.R_channel
                        ) / self.effective_mass.R_channel < tolerance
        else:
            check = True
        return check 

    @property
    def measurement_id(self) -> str:
        """
        ``nerea.NormalizedPulseHeightSpectrum.measurement_id()``
        ------------------------------------------------------------
        The measurement ID associated with the pulse height spectrum.

        Returns
        -------
        ``str``
            The measurement ID attribute of the associated PHS."""
        return self.phs.measurement_id
    
    @property
    def campaign_id(self) -> str:
        """
        ``nerea.NormalizedPulseHeightSpectrum.campaign_id()``
        -----------------------------------------------------
        The campaign ID associated with the pulse height spectrum.

        Returns
        -------
        ``str``
            The campaign ID attribute of the associated PHS."""
        return self.phs.campaign_id
    
    @property
    def experiment_id(self) -> str:
        """
        ``nerea.NormalizedPulseHeightSpectrum.experiment_id()``
        -------------------------------------------------------
        The experiment ID associated with the pulse height spectrum.

        Returns
        -------
        ``str``
            The experiment ID attribute of the associated PHS."""
        return self.phs.experiment_id
    
    @property
    def location_id(self) -> str:
        """
        ``nerea.NormalizedPulseHeightSpectrum.location_id()``
        -----------------------------------------------------
        The location ID associated with the pulse height spectrum.

        Returns
        -------
        ``str``
            The location ID attribute of the associated PHS."""
        return self.phs.location_id

    @property
    def deposit_id(self) -> str:
        """
        ``nerea.NormalizedPulseHeightSpectrum.deposit_id()``
        ----------------------------------------------------
        The deposit ID associated with the pulse height spectrum.

        Returns
        -------
        ``str``
            The deposit ID attribute of the associated PHS."""
        return self.phs.deposit_id

    @property
    def _time_normalization(self) -> pd.DataFrame:
        """
        ``nerea.NormalizedPulseHeightSpectrum._time_normalization()``
        -------------------------------------------------------------
        The time normalization and correction to be multiplied by the
        pulse height spectrum per unit mass.

        Returns
        -------
        ``pd.DataFrame``
            with ``'value'`` and ``'uncertainty'`` columns.

        Note
        ----
        - it corresponds to 1 / time."""
        l = self.phs.live_time
        v = 1 / l
        u = np.sqrt((1 / self.phs.live_time **2 \
                     * self.phs.live_time_uncertainty)**2)
        return _make_df(v, u)

    @property
    def _power_normalization(self) -> pd.DataFrame:
        """
        ``nerea.NormalizedPulseHeightSpectrum.power_normalization()``
        -------------------------------------------------------------
        The power normalization to be multiplied by the pulse height
        spectrum per unit mass.

        Returns
        -------
        ``pd.DataFrame``
            with ``'value'`` and ``'uncertainty'`` columns.

        Note
        ----
        - it corresponds to 1 / average(count_rate)."""
        start_time = self.phs.start_time
        duration = self.phs.real_time
        pm = self.power_monitor.average(start_time, duration)
        return _make_df(*ratio_v_u(_make_df(1, 0), pm))

    def _get_long_output(self,
                         plateau: pd.DataFrame,
                         time: pd.DataFrame,
                         power: pd.DataFrame,
                         **kwargs) -> pd.DataFrame:
        """
        ``nerea.NormalizedPulseHeightSpectrum._get_long_output()``
        ----------------------------------------------------------
        The information to be included in the long output: component
        variances.

        Parameters
        ----------
        **plateau** : ``pd.DataFrame``
            output of ``self.plateau()``
        **time** : ``pd.DataFrame``
            output of ``self._time_normalization``
        **power** : ``pd.DataFrame``
            output of ``self._power_normalization``
        **kwargs
            arguments for ``self.pulse_height_spectrum.integrate()``
            - **llds** (``Iterable[int|float] | int``) low level discriminator(s).
            - **r** (``bool``): whether discriminators are absolute or fractions of R.
            - **raw_integral** (``bool``): whether to integrate the raw data or the smoothed ones.

            additional arguments for

            - ``self.pulse_height_spectrum.rebin()``
                - **bins** (``int``): number of bins
                - **smooth** (``bool``): whether to smooth the PHS

            - ``self.pulse_height_spectrum.smooth()`` only if ``smooth == True``
                - **renormalize** (``bool``): Whether to renormalize the smoothed spectrum.
                - **smoothing_method** (``str``): The mehtod to implement for smoothing.
                - arguments for the chosen ``nerea.functions.smoothing``.

            - ``self.pulse_height_spectrum.get_max()``
                - **fst_ch** (``int | str``): channel to start max search or max search method.

        Returns
        -------
        ``pd.DataFrame``
            (1 x N) DataFrame with information about variance values and variances of
            data used in the processing.

        Note
        ----
        - ``bins`` for PHS rebinning are set to ``self.effective_mass.bins``."""
        kwargs = DEFAULT_MAX_KWARGS | DEFAULT_BIN_KWARGS | kwargs
        # I always want to integrate over the same channels and binning as EM
        kwargs['bins'] = self.effective_mass.bins

        ch_ffs, ch_em = plateau['CH_FFS'].value, plateau['CH_EM'].value
        ffs = self.phs.integrate(**kwargs).query("channel==@ch_ffs")
        em = self.effective_mass.integral.query("channel==@ch_em")

        val_ffs, var_ffs = ffs.value.iloc[0], ffs.uncertainty.iloc[0] **2
        val_em, var_em = em.value.iloc[0], em.uncertainty.iloc[0] **2
        val_pm, var_pm = 1 / power.value, power.uncertainty **2 / power.value **4
        val_t, var_t = 1 / time.value,  time.uncertainty **2 / time.value **4
        df = pd.DataFrame({'FFS': val_ffs, 'VAR_FFS': var_ffs,
                           'EM': val_em, 'VAR_EM': var_em,
                           'PM': val_pm, 'VAR_PM': var_pm,
                           't': val_t, 'VAR_t': var_t}, index=['value'])
        return df
    
    def _per_unit_mass_R(self, phsi: pd.DataFrame, emi: pd.DataFrame) -> pd.DataFrame:
        """
        ``nerea.NormalizedPulseHeightSpectrum._per_unit_mass_R()``
        ----------------------------------------------------------
        The tabulated ratio of PHS.integrate() / EM.integral, integrated from
        discrimination levels computed as a function of the R channel.

        Parameters
        ----------
        **phsi** : ``pd.DataFrame``
            Output of ``self.pulse_height_spectrum.integrate().``
        **emi** : ``pd.DataFrame``
            Output of ``self.effective_mass.integral``.

        Returns
        -------
        ``pd.DataFrame``
            with information of the mass-normalized spectrum."""
        channels = sorted(set(emi.R).intersection(set(phsi.R)))
        if len(channels) < len(emi.R): warnings.warn("Neglecting some calibration channels.")
        if len(channels) < len(phsi.R): warnings.warn("Neglecting some integration channels.")
        return _make_df(*ratio_v_u(phsi, emi)).reset_index(drop=True).assign(
                            VAR_PORT_FFS = (phsi.uncertainty / emi.value) **2,
                            VAR_PORT_EM = (phsi.value / emi.value**2 * emi.uncertainty) **2,
                            CH_FFS = phsi.channel,
                            CH_EM = emi.channel,
                            R=emi.R)

    def _per_unit_mass_ch(self, phsi: pd.DataFrame, emi: pd.DataFrame) -> pd.DataFrame:
        """
        ``nerea.NormalizedPulseHeightSpectrum._per_unit_mass_ch()``
        ----------------------------------------------------------
        The tabulated ratio of PHS.integrate() / EM.integral, integrated from
        discrimination levels defined as absolute channels.

        Parameters
        ----------
        **phsi** : ``pd.DataFrame``
            Output of ``self.pulse_height_spectrum.integrate()``.
        **emi** : ``pd.DataFrame``
            Output of ``self.effectivemass.integral``.

        Returns
        -------
        ``pd.DataFrame``
            with information of the mass-normalized spectrum."""
        channels = sorted(set(emi.channel).intersection(set(phsi.channel)))
        if len(channels) < len(emi.channel): warnings.warn("Neglecting some calibration channels.")
        if len(channels) < len(phsi.channel): warnings.warn("Neglecting some integration channels.")
        return _make_df(*ratio_v_u(phsi, emi)).reset_index(drop=True).assign(
                                    VAR_PORT_FFS = (phsi.uncertainty / emi.value) **2,
                                    VAR_PORT_EM = (phsi.value / emi.value**2 * emi.uncertainty) **2,
                                    CH_FFS = phsi.channel,
                                    CH_EM = emi.channel,
                                    R=np.nan)

    def per_unit_mass(self, **kwargs) -> pd.DataFrame:
        """
        ``nerea.NormalizedPulseHeightSpectrum.per_unit_mass()``
        -------------------------------------------------------
        Normalizes ``self.pulse_height_spectrum()`` to the
        ``self.effective_mass`` based on the effective mass
        discrimination levels.

        Parameters
        ----------
        **kwargs
        arguments for ``self.pulse_height_spectrum.integrate()``

        - **llds** (``Iterable[int|float] | int``) low level discriminator(s).
        - **r** (``bool``): whether discriminators are absolute or fractions of R.
        - **raw_integral** (``bool``): whether to integrate the raw data or the smoothed ones.

        additional arguments for

        - ``self.pulse_height_spectrum.rebin()``
            - **bins** (``int``): number of bins
            - **smooth** (``bool``): whether to smooth the PHS

        - ``self.pulse_height_spectrum.smooth()`` only if ``smooth == True``
            - **renormalize** (``bool``): Whether to renormalize the smoothed spectrum.
            - **smoothing_method** (``str``): The mehtod to implement for smoothing.
            - arguments for the chosen ``nerea.functions.smoothing``.

        - ``self.pulse_height_spectrum.get_max()``
            - **fst_ch** (``int | str``): channel to start max search or max search method.

        Returns
        -------
        ``pd.DataFrame``
            with information of the mass-normalized spectrum.

        Note
        ----
        - `bins` for PHS rebinning are set to `self.effective_mass.bins`."""
        kwargs = DEFAULT_MAX_KWARGS | DEFAULT_BIN_KWARGS | kwargs
        # I always want to integrate over the same channels and binning as EM
        kwargs['bins'] = self.effective_mass.bins

        ffs = self.phs.integrate(**kwargs)
        em = self.effective_mass.integral
        if np.isnan(em.R).all() and np.isnan(ffs.R).all():
            data = self._per_unit_mass_ch(ffs, em)
        elif not(np.isnan(em.R).all() and np.isnan(ffs.R).all()):
            data = self._per_unit_mass_R(ffs, em)
        else:
            raise Exception("Inconsistent integration and integration methodologies: can not process discrimination levels.",
                            ValueError)
        return data

    def per_unit_mass_and_time(self, **kwargs) -> pd.DataFrame:
        """
        ``nerea.NormalizedPulseHeightSpectrum.per_unit_mass_and_time()``
        -----------------------------------------------------------------
        The integrated PHS normalized per unit mass and time.

        Parameters
        ----------
        **kwargs
        arguments for ``self.per_unit_mass()``

        - **llds** (``Iterable[int|float] | int``) low level discriminator(s).
        - **r** (``bool``): whether discriminators are absolute or fractions of R.
        - **raw_integral** (``bool``): whether to integrate the raw data or the smoothed ones.

        additional arguments for

        - ``self.pulse_height_spectrum.rebin()``
            - **bins** (``int``): number of bins
            - **smooth** (``bool``): whether to smooth the PHS

        - ``self.pulse_height_spectrum.smooth()`` only if ``smooth == True``
            - **renormalize** (``bool``): Whether to renormalize the smoothed spectrum.
            - **smoothing_method** (``str``): The mehtod to implement for smoothing.
            - arguments for the chosen ``nerea.functions.smoothing``.

        - ``self.pulse_height_spectrum.get_max()``
            - **fst_ch** (``int | str``): channel to start max search or max search method.

        Returns
        -------
        ``pd.DataFrame``
            DataFrame with the information of the mass- and time- normalized spectrum.

        Note
        ----
        - `bins` for PHS rebinning are set to `self.effective_mass.bins`."""
        kwargs = DEFAULT_MAX_KWARGS | DEFAULT_BIN_KWARGS | kwargs
        # I always want to integrate over the same channels and binning as EM
        kwargs['bins'] = self.effective_mass.bins
        pum = self.per_unit_mass(**kwargs)
        pum.index = ['value'] * pum.shape[0]
        time = pd.concat([self._time_normalization] * pum.shape[0])
        data = _make_df(*product_v_u([pum, time])).assign(
            CH_FFS=pum.CH_FFS, CH_EM=pum.CH_EM).reset_index(drop=True)
        return data[['CH_FFS', 'CH_EM', 'value', 'uncertainty', 'uncertainty [%]']]

    def per_unit_mass_and_power(self, **kwargs) -> pd.DataFrame:
        """
        ``nerea.NormalizedPulseHeightSpectrum.per_unit_mass_and_power()``
        -----------------------------------------------------------------
        The integrated PHS normalized per unit mass and power.

        Parameters
        ----------
        **kwargs
        arguments for ``self.per_unit_mass()``

        - **llds** (``Iterable[int|float] | int``) low level discriminator(s).
        - **r** (``bool``): whether discriminators are absolute or fractions of R.
        - **raw_integral** (``bool``): whether to integrate the raw data or the smoothed ones.

        additional arguments for

        - ``self.pulse_height_spectrum.rebin()``
            - **bins** (``int``): number of bins
            - **smooth** (``bool``): whether to smooth the PHS

        - ``self.pulse_height_spectrum.smooth()`` only if ``smooth == True``
            - **renormalize** (``bool``): Whether to renormalize the smoothed spectrum.
            - **smoothing_method** (``str``): The mehtod to implement for smoothing.
            - arguments for the chosen ``nerea.functions.smoothing``.

        - ``self.pulse_height_spectrum.get_max()``
            - **fst_ch** (``int | str``): channel to start max search or max search method.

        Returns
        -------
        ``pd.DataFrame``
            with information of the mass- and power- normalized spectrum.

        Note
        ----
        - `bins` for PHS rebinning are set to `self.effective_mass.bins`."""
        kwargs = DEFAULT_MAX_KWARGS | DEFAULT_BIN_KWARGS | kwargs
        # I always want to integrate over the same channels and binning as EM
        kwargs['bins'] = self.effective_mass.bins

        # ffs, em = self.pulse_height_spectrum, self.effective_mass
        pum = self.per_unit_mass(**kwargs)
        pum.index = ['value'] * pum.shape[0]
        power = pd.concat([self._power_normalization] * pum.shape[0])
        data = _make_df(*product_v_u([pum, power])).assign(
            CH_FFS=pum.CH_FFS, CH_EM=pum.CH_EM).reset_index(drop=True)
        return data[['CH_FFS', 'CH_EM', 'value', 'uncertainty', 'uncertainty [%]']]

    def per_unit_power_and_time(self, **kwargs) -> pd.DataFrame:
        """
        ``nerea.NormalizedPulseHeightSpectrum.per_unit_power_and_time()``
        -----------------------------------------------------------------
        The integrated PHS normalized per unit power and time.

        Parameters
        ----------
        **kwargs
        arguments for ``self.per_unit_mass()``

        - **llds** (``Iterable[int|float] | int``) low level discriminator(s).
        - **r** (``bool``): whether discriminators are absolute or fractions of R.
        - **raw_integral** (``bool``): whether to integrate the raw data or the smoothed ones.

        additional arguments for

        - ``self.pulse_height_spectrum.rebin()``
            - **bins** (``int``): number of bins
            - **smooth** (``bool``): whether to smooth the PHS

        - ``self.pulse_height_spectrum.smooth()`` only if ``smooth == True``
            - **renormalize** (``bool``): Whether to renormalize the smoothed spectrum.
            - **smoothing_method** (``str``): The mehtod to implement for smoothing.
            - arguments for the chosen ``nerea.functions.smoothing``.

        - ``self.pulse_height_spectrum.get_max()``
            - **fst_ch** (``int | str``): channel to start max search or max search method.

        Returns
        -------
        ``pd.DataFrame``
            with information of the power- and time- normalized spectrum."""
        phspa_int = self.phs.integrate(**kwargs).set_index(['channel', 'R'])
        idx = phspa_int.index
        return _make_df(*product_v_u([phspa_int.reset_index(drop=True),
                                      pd.concat([self._time_normalization] * phspa_int.shape[0], ignore_index=True),
                                      pd.concat([self._power_normalization] * phspa_int.shape[0], ignore_index=True)]),
                                      idx=idx).reset_index()

    def plateau(self, int_tolerance: float=.01, ch_tolerance: float=.01, **kwargs) -> pd.DataFrame:
        """
        ``nerea.NormalizedPulseHeightSpectrum.plateau()``
        -------------------------------------------------
        Computes the count rate per unit mass.

        Parameters
        ----------
        **int_tolerance** : ``float``, optional
            Tolerance for the integration check, by default ``0.01``.
        **ch_tolerance** : ``float``, optional
            Tolerance for the channel check, by default ``0.01``.
        **kwargs
        arguments for ``self.per_unit_mass()``

        - **llds** (``Iterable[int|float] | int``) low level discriminator(s).
        - **r** (``bool``): whether discriminators are absolute or fractions of R.
        - **raw_integral** (``bool``): whether to integrate the raw data or the smoothed ones.

        additional arguments for

        - ``self.pulse_height_spectrum.rebin()``
            - **bins** (``int``): number of bins
            - **smooth** (``bool``): whether to smooth the PHS

        - ``self.pulse_height_spectrum.smooth()`` only if ``smooth == True``
            - **renormalize** (``bool``): Whether to renormalize the smoothed spectrum.
            - **smoothing_method** (``str``): The mehtod to implement for smoothing.
            - arguments for the chosen ``nerea.functions.smoothing``.

        - ``self.pulse_height_spectrum.get_max()``
            - **fst_ch** (``int | str``): channel to start max search or max search method.

        Returns
        -------
        ``pd.DataFrame``
            containing the count rate per unit mass at convergence.
            It has ``'value'`` and ``'uncertainty'`` columns.

        Raises
        ------
        ValueError
            If the channel values differ beyond the specified tolerance."""
        kwargs = DEFAULT_MAX_KWARGS | DEFAULT_BIN_KWARGS | kwargs
        data = self.per_unit_mass(**kwargs)

        if data.shape[0] < 3:
            warnings.warn(f"No plateau can be found with {data.shape[0]} points. Considering last.")
            out = data.iloc[-1].to_frame().T
        else:
            if kwargs.get('verbose', False):
                msg = f"Normalized PHS plateau search with integral tolerance {int_tolerance} and channel tolerance {ch_tolerance}."
                logger.info(msg)
            
            # check where the values in the mass-normalized count rate converge withing tolerance
            vals = data.value.values
            mask_value = np.isclose(vals[1:], vals[:-1], rtol=int_tolerance)
            close_values = data.iloc[1:][mask_value]
            if close_values.shape[0] == 0:
                raise Exception("No convergence found with the given tolerance on the integral.", ValueError)

            # and where close values were found in successive rows 
            idx = close_values.index.values
            mask_successive = np.isclose(idx, np.roll(idx, shift=1), atol=1)
            plateau = close_values[mask_successive]
            if plateau.shape[0] == 0:
                raise Exception("No convergence found in neighbouring channels.", ValueError)

            # test if the channels are within tolerance
            mask_channel = np.abs(plateau['CH_FFS'] - plateau['CH_EM']) / plateau['CH_EM'] < ch_tolerance
            plateau = plateau[mask_channel]
            if plateau.shape[0] == 0:
                raise Exception("No convergence found with the given tolerance on the channel.", ValueError)

            # return first value of the plateau
            out = plateau.iloc[0].to_frame().T
        out.index = ['value']
        return out

    def process(self, long_output: bool=False, visual: bool=False,
                savefig: str='', **kwargs) -> pd.DataFrame:
        """
        ``nerea.NormalizedPulseHeightSpectrum.process()``
        ----------------------------------------------------------
        Computes the count rate.

        Parameters
        ----------
        **long_output** : ``bool``, optional
            Flag to turn on/off the full output information, whcih includes
            values and variances of all the processing elements, ``False`` by default.
        **visual** : ``bool``, optional
            Flag to display the processed data.
            Default is ``False``.
        **savefig** : ``str``, optional
            Filename to save the figure to.
            Default is ``''`` not saving.
        **kwargs
        arguments for ``self.plateau()``

        - **int_tolerance** (``float``): tolerance for integration check.
        - **ch_tolerance** (``float``): tolerance for channel check.

        additional arguments for

        - ``self.pulse_height_spectrum.integrate()``
            - **llds** (``Iterable[int|float] | int``) low level discriminator(s).
            - **r** (``bool``): whether discriminators are absolute or fractions of R.
            - **raw_integral** (``bool``): whether to integrate the raw data or the smoothed ones.

        - ``self.pulse_height_spectrum.rebin()``
            - **bins** (``int``): number of bins
            - **smooth** (``bool``): whether to smooth the PHS

        - ``self.pulse_height_spectrum.smooth()`` only if ``smooth == True``
            - **renormalize** (``bool``): Whether to renormalize the smoothed spectrum.
            - **smoothing_method** (``str``): The mehtod to implement for smoothing.
            - arguments for the chosen ``nerea.functions.smoothing``.

        - ``self.pulse_height_spectrum.get_max()``
            - **fst_ch** (``int | str``): channel to start max search or max search method.

        Returns
        -------
        ``pd.DataFrame``
            containing the count rate per unint mass and power.
            It has ``'value'`` and ``'uncertainty'`` columns.
        
        Note
        ----
        - `bins` for PHS rebinning are set to `self.effective_mass.bins`.

        Examples
        --------
        >>> ffs = PulseHeightSpectrum(data=pd.DataFrame({'value': [1.0, 2.0, 3.0], 'uncertainty': [0.1, 0.2, 0.3]}),
        ...                               detector_id='D1', deposit_id='Dep1', experiment_id='Exp1')
        >>> em = EffectiveMass(data=pd.DataFrame({'value': [0.5, 0.6, 0.7], 'uncertainty': [0.05, 0.06, 0.07]}),
        ...                    detector_id='D1', deposit_id='Dep1')
        >>> pm = CountRate(data=pd.DataFrame({'value': [10, 20, 30], 'uncertainty': [1, 2, 3]}), experiment_id='Exp1')
        >>> rr = NormalizedPulseHeightSpectrum(pulse_height_spectrum=ffs, effective_mass=em, power_monitor=pm)
        >>> rr.process()
            value  uncertainty
        0  35.6    2.449490"""
        kwargs = DEFAULT_MAX_KWARGS | DEFAULT_BIN_KWARGS | kwargs
        # I always want to integrate over the same channels and binning as EM
        kwargs['bins'] = self.effective_mass.bins

        plateau = self.plateau(**kwargs)        # FFS / EM @plateau and relative variance fractions
        power = self._power_normalization       # this is 1/PM
        time = self._time_normalization         # this is 1/t
        # compute variance fractions
        S_PLAT, S_PM, S_T = power.value * time.value, plateau.value * time.value, plateau.value * power.value
        df = _make_df(*product_v_u([plateau, power, time])
                      ).assign(VAR_PORT_FFS=plateau["VAR_PORT_FFS"] * S_PLAT **2,
                               VAR_PORT_EM=plateau["VAR_PORT_EM"] * S_PLAT **2,
                               VAR_PORT_PM=(S_PM * power.uncertainty) **2,
                               VAR_PORT_t=(S_T * time.uncertainty) **2)
        if visual or savefig:
            fig, _ = self.plot(plateau['CH_FFS'].value, **kwargs)
            if savefig:
                fig.savefig(savefig)
                plt.close()
        return df if not long_output else pd.concat([df,
                                                     self._get_long_output(plateau,
                                                                           time,
                                                                           power,
                                                                           **kwargs)
                                                    ], axis=1)

    def plot(self, discri: int=None, **kwargs) -> tuple[plt.Figure, Iterable[plt.Axes]]:
        """
        `nerea.NormalizedPulseHeightSpectrum.plot()`
        --------------------------------------------
        Default plotting method of PHS and monitor considered.
        It also reports tabulated effective mass values.

        Paramters
        ---------
        **discri**: ``int``, optional
            The discrimination level to highilight in the plots.
            It is in units of channel of self.pulse_height_spectrum.
            Default is None.
        **kwargs
        arguments for ``self.per_unit_mass()``, ``self.pulse_height_spectrum.integrate()``
        and ``self.pulse_height_spectrum.plot()``.

        -``self.per_unit_mass()``
            - **llds** (``Iterable[int|float] | int``) low level discriminator(s).
            - **r** (``bool``): whether discriminators are absolute or fractions of R.
            - **raw_integral** (``bool``): whether to integrate the raw data or the smoothed ones.

        - ``self.pulse_height_spectrum.rebin()``
            - **bins** (``int``): number of bins
            - **smooth** (``bool``): whether to smooth the PHS

        - ``self.pulse_height_spectrum.smooth()`` only if ``smooth == True``
            - **renormalize** (``bool``): Whether to renormalize the smoothed spectrum.
            - **smoothing_method** (``str``): The mehtod to implement for smoothing.
            - arguments for the chosen ``nerea.functions.smoothing``.

        - ``self.pulse_height_spectrum.get_max()``
            - **fst_ch** (``int | str``): channel to start max search or max search method.

        Returns
        -------
        ``tuple[plt.Figure, Iterable[plt.Axes]]``"""
        fig, axs = plt.subplots(2, 2, figsize=(15, 12),
                                height_ratios=[1, 1], width_ratios=[1, 1],
                                gridspec_kw={'wspace': 0.4})

        ## plot Effective Mass
        self.effective_mass.data.plot(x='channel',  y='value', ax=axs[0][0], kind='scatter', c='k')
        axs[0][0].set_xlabel("Calibration channel")
        cell_text = [['{:.0f}'.format(r.channel),
                      '{:.2f}'.format(r.value)
                      ] for _, r in self.effective_mass.data.iterrows()]
        tab = axs[0][0].table(cellText=cell_text, colLabels=['ch', 'm [ug]'],
                              bbox=[1.01, 0, 0.275, 1])
        tab.auto_set_font_size(False)
        axs[0][0].set_ylabel("Effective mass [ug]")

        ## plot Power Monitor
        self.power_monitor.plot(ax=axs[0][1],
                                start_time=self.phs.start_time,
                                duration=self.phs.real_time)

        ## plot PHS
        self.phs.plot(ax=axs[1][0], **kwargs)
        axs[1][0].set_xlabel("Measurement channel")
        axs[1][0].set_ylabel("Counts [-]")

    	## plot fission rate per unit mass
        pum = self.per_unit_mass(**kwargs)
        pum.plot(x='CH_FFS', y='value', ax=axs[1][1], kind='scatter', c='k')
        axs[1][1].set_xticks(pum['CH_FFS'])
        axs[1][1].set_xticklabels([f"{x:.0f}" for x in pum['CH_FFS']])
        axs[1][1].set_ylabel("Integral counts per unit mass [1/ug]")

        ax_top = axs[1][1].twiny()
        ax_top.set_xlim(axs[1][1].get_xlim())
        ax_top.set_xticks(axs[1][1].get_xticks())
        ax_top.set_xticklabels([f"{x:.0f}" for x in pum['CH_EM']])
        ax_top.set_xlabel("Calibration channel")
        axs[1][1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        axs[1][1].tick_params(axis='y', left=False, labelleft=False, right=True, labelright=True)
        axs[1][1].yaxis.set_label_position("right")
        axs[1][1].set_xlabel("Measurement channel")
        t = axs[1][1].yaxis.get_offset_text()
        t.set_x(1.01)
        axs[1][1].grid()

        # highlight discrimination level if passed
        if discri is not None:
            discri_r = self.phs.integrate(
                    **kwargs).query("channel == @discri").R.iloc[0]
            axs[0][0].scatter(x=self.effective_mass.data.query("R == @discri_r")['channel'].iloc[0],
                              y=self.effective_mass.data.query("R == @discri_r")['value'].iloc[0],
                              c='b', marker='s', label="Discriminator")
            axs[1][0].axvline(discri, c='b', label='Discriminator')
            axs[1][1].scatter(discri, pum.query("CH_FFS == @discri").value.iloc[0],
                          c='b', marker='s', label='Discriminator')
        return fig, axs


@dataclass
class SpectralIndex(_Experimental):
    """
    ``nerea.SpectralIndex``
    =======================
    Class storing and processing a spectral index.
    Inherits from ``nerea.Experimental``.

    Attributes
    ----------
    **numerator** : ``nerea.NormalizedPulseHeightSpectrum``
        the spectral index numerator.
    **denominator** : ``nerea.NormalizedPulseHeightSpectrum``
        the spectral index denominator.
    _enable_checks: ``bool``, optoinal
        flag enabling consistency checks. Default is ``True``.
    """
    numerator: NormalizedPulseHeightSpectrum
    denominator: NormalizedPulseHeightSpectrum
    _enable_checks: bool = True

    def __post_init__(self) -> None:
        """
        Runs consistency checks.
        """
        if self._enable_checks:
            self._check_consistency()

    def _check_consistency(self) -> None:
        """
        `nerea.SpectralIndex._check_consistency()`
        -------------------------------------------
        Checks the consistency of:
            - campaign_id
            - location_id
        among ``self.numerator`` and ``self.denominator``.

        Raises
        ------
        UserWarning
            If there are inconsistencies among the specified attributes."""
        should = ['campaign_id', 'location_id']
        for attr in should:
            if not getattr(self.numerator, attr
                           ) == getattr(self.denominator, attr):
                warnings.warn(f"Inconsistent {attr} among numerator and denominator.")

    @property
    def deposit_ids(self) -> list[str]:
        """
        `nerea.SpectralIndex.deposit_ids()`
        -----------------------------------
        The deposit IDs associated with the numerator and denominator.

        Returns
        -------
        ``list[str]``
            A list containing the deposit IDs of the numerator and denominator.

        Examples
        --------
        >>> from nerea.CountRate import CountRate
        >>> ffs_num = CountRate(..., deposit_id='Dep1')
        >>> ffs_den = CountRate(..., deposit_id='Dep2')
        >>> spectral_index = SpectralIndex(numerator=ffs_num, denominator=ffs_den)
        >>> spectral_index.deposit_ids
        ['Dep1', 'Dep2']"""
        return [self.numerator.deposit_id, self.denominator.deposit_id]

    def _compute_correction(self, one_g_xs: pd.DataFrame) -> pd.DataFrame:
        """
        `nerea.SpectralIndex._compute_correction()`
        -------------------------------------------
        Computes the impurity correction to the spectral index.

        Parameters
        ----------
        **one_g_xs** : ``nerea.Xs``
            with nuclides and one group cross sections as.

        Returns
        -------
        ``pd.DataFrame``
            with correction ``'value'`` and ``'uncertainty'`` columns.

        Raises
        ------
        UserWarning
            If xs is not given for all impurities."""
        comp = self.numerator.effective_mass.composition_.copy()
        # sum over impurities != self.numerator.deposit_id
        return impurity_correction(one_g_xs, comp, drop_main=True,
                                   xs_den=self.denominator.deposit_id,
                                   relative = True if comp.shape[0] != 0 else False
                                   ).dropna()

    def _get_long_output(self, num, den, k) -> pd.DataFrame:
        """
        `nerea.SpectralIndex._get_long_output()`
        ----------------------------------------
        The information to be included in the long output:
        values and variances of numerator and denominator if
        those were computed in the first place and variance
        of the impurity correction (if any of the others was
        computed).

        Parameters
        ----------
        **num** : ``pd.DataFrame``
            output of ``self.numerator.process()``
        **den** : ``pd.DataFrame``
            output of ``self.denominator.process()``
        **k** : ``pd.DataFrame``
            impurity correction

        Returns
        -------
        ``pd.DataFrame``
            (1 x N) DataFrame or empty pd.DataFrame if the varaince was not
            computed for none of `num` and `den`."""
        empty = True
        start_col = 7
        if 'VAR_FFS' in num.columns:
            num_ = num.rename(columns=dict(zip(num.columns[start_col:],
                                          [f'{c}_n' for c in num.columns[start_col:]]))
                                          ).iloc[:, start_col:]
            empty = False
        else:
            num_ = pd.DataFrame()

        if 'VAR_FFS' in den.columns:
            den_ = den.rename(columns=dict(zip(num.columns[start_col:],
                                          [f'{c}_d' for c in num.columns[start_col:]]))
                                          ).iloc[:, start_col:]
            empty = False
        else:
            den_ = pd.DataFrame()

        if not empty:
            k_ = pd.DataFrame({'1GXS': 0 if k is None else k['value'].iloc[0],
                               'VAR_1GXS': None if k is None else k['uncertainty'].iloc[0] **2},
                               index=['value'])
            out = pd.concat([num_, den_, k_], axis=1)
        else:
            out = pd.DataFrame()
        return out

    def process(self,
                one_g_xs: Xs = None,
                one_g_xs_file: str = None,
                nuc_dec_from_file : dict[str, str] = None,
                numerator_kwargs: dict={},
                denominator_kwargs: dict={},
                mass_normalized: bool=False) -> pd.DataFrame:
        """
        `nerea.SpectralIndex.process()`
        -------------------------------
        Computes the ratio of two count rates.

        Parameters
        ----------
        **one_g_xs** : ``nerea.Xs``, optional
            with nuclides and one group cross sections as.
            Defaults to ``None`` for no correction.
        **one_g_xs_file** : ``str``, optional
            the Serpent detector file to read the one group xs from.
            Default is ``None``.
        **nuc_dec_from_file** : ``dict[str, str]``, optional
            A dictionary mapping each nuclide with the detector associated
            with its cross section calculation in ``one_g_xs_file``.
        **numerator_kwargs** : dict[Any]
            Keyword arguments for `self.numerator.process()`

            - **long_output** (``bool``): whetehr to output full information
            - **visual** (``bool``): whether to display the processed data
            - **savefig** (``str``): filename to save the figure to.

            additional arguments for

            - ``self.plateau()``
                - **int_tolerance** (``float``): tolerance for integration check.
                - **ch_tolerance** (``float``): tolerance for channel check.

            - ``self.pulse_height_spectrum.integrate()``
                - **llds** (``Iterable[int|float] | int``) low level discriminator(s).
                - **r** (``bool``): whether discriminators are absolute or fractions of R.
                - **raw_integral** (``bool``): whether to integrate the raw data or the smoothed ones.

            - ``self.pulse_height_spectrum.rebin()``
                - **bins** (``int``): number of bins
                - **smooth** (``bool``): whether to smooth the PHS

            - ``self.pulse_height_spectrum.smooth()`` only if ``smooth == True``
                - **renormalize** (``bool``): Whether to renormalize the smoothed spectrum.
                - **smoothing_method** (``str``): The mehtod to implement for smoothing.
                - arguments for the chosen ``nerea.functions.smoothing``.

            - ``self.pulse_height_spectrum.get_max()``
                - **fst_ch** (``int | str``): channel to start max search or max search method.

            **denominator_kwargs** : dict[Any]
            Keyword arguments for `self.denominator.process()`
            - **long_output** (``bool``): whetehr to output full information
            - **visual** (``bool``): whether to display the processed data
            - **savefig** (``str``): filename to save the figure to.

            additional arguments for

            - ``self.plateau()``
                - **int_tolerance** (``float``): tolerance for integration check.
                - **ch_tolerance** (``float``): tolerance for channel check.

            - ``self.pulse_height_spectrum.integrate()``
                - **llds** (``Iterable[int|float] | int``) low level discriminator(s).
                - **r** (``bool``): whether discriminators are absolute or fractions of R.
                - **raw_integral** (``bool``): whether to integrate the raw data or the smoothed ones.

            - ``self.pulse_height_spectrum.rebin()``
                - **bins** (``int``): number of bins
                - **smooth** (``bool``): whether to smooth the PHS

            - ``self.pulse_height_spectrum.smooth()`` only if ``smooth == True``
                - **renormalize** (``bool``): Whether to renormalize the smoothed spectrum.
                - **smoothing_method** (``str``): The mehtod to implement for smoothing.
                - arguments for the chosen ``nerea.functions.smoothing``.

            - ``self.pulse_height_spectrum.get_max()``
                - **fst_ch** (``int | str``): channel to start max search or max search method.

        **mass_normalized** : ``bool``, optional
            defines whether the result is the ratio of fission rates or of
            fission rates per unit mass.
            Default is ``False``.

        Returns
        -------
        ``pd.DataFrame``
            with ``'value'`` and ``'uncertainty'`` columns.
            
        Note
        ----
        - Working in the effective mass framework, it is assumed that all cross sections
        for impurities are mass-normalized (nerea.Xs.normalized). Then the processed 
        spectral index result is multiplied by the ratio between numerator and denominator
        atomic mass to be consistent with the definition of one-group cross section ratio.
        Else the ``mass_normalized`` argument should be used passing consistent one group
        cross sections for impurity correction."""
        if numerator_kwargs.get('verbose', False):
            logger.info("PROCESSING SPECTRAL INDEX NUMERATOR.")
        num = self.numerator.process(**numerator_kwargs)
        if denominator_kwargs.get('verbose', False):
            logger.info("PROCESSING SPECTRAL INDEX DENOMINATOR.")
        den = self.denominator.process(**denominator_kwargs)
        v, u = ratio_v_u(num, den)
        if (one_g_xs is None and one_g_xs_file is None
            and self.numerator.effective_mass.composition_.shape[0] > 1):
            warnings.warn("Impurities in the fission chambers require one group xs" +\
                          " to be accounted for.")
        if one_g_xs_file is not None:
            read = Xs.from_file(one_g_xs_file, nuc_dec_from_file)
        else:
            read = None

        one_g_xs_ = read if one_g_xs is None else one_g_xs
        if one_g_xs_ is not None:
            k = self._compute_correction(one_g_xs_.normalized)
            v = v - k.value
            u = np.sqrt(u **2 + k.uncertainty **2)
        else: k = None

        # atomic mass ratio for EM renormalization
        # see docstring note
        # assumed to have no uncertainty
        if not mass_normalized:
            an = ATOMIC_MASS.loc[self.numerator.deposit_id].value 
            ad = ATOMIC_MASS.loc[self.denominator.deposit_id].value 
            mass_ratio = an / ad
        else:
            mass_ratio = 1
        df = _make_df(v * mass_ratio, u * mass_ratio)

        # compute fraction of variance
        var_cols = [c for c in num.columns if c.startswith("VAR_PORT")]
        
        var_num = num[var_cols] / den['value'].value **2 * mass_ratio **2
        var_num.columns = [f"{c}_n" for c in var_cols]

        var_den = den[var_cols] * (num['value'] / den['value'] **2).value **2 * mass_ratio **2
        var_den.columns = [f"{c}_d" for c in var_cols]

        # concatenate variances to `df`
        df =  pd.concat([df, var_num, var_den], axis=1).assign(
                                    VAR_PORT_1GXS=(k.uncertainty * mass_ratio) **2 if k is not None else 0.
                                    )
        return pd.concat([df, self._get_long_output(num, den, k)], axis=1)

@dataclass(slots=True)
class Traverse(_Experimental):
    """
    ``nerea.Traverse``
    ==================
    Class storing and processing a traverse data.
    Inherits from `nerea.Experimental`.

    Attributes
    ----------
    **count_rates** : ``dict[str, CountRate | CountRates]``
        Links traverse position to the measured count rate.
        ``key`` is the position identifier, ``value`` is the 
        corresponding ``nerea.CountRate`` or `nerea`.CountRates``.
        If ``nerea.CountRates``, the first is considered.
    _enable_checks: ``bool``, optoinal
        flag enabling consistency checks. Default is ``True``."""
    count_rates: dict[str, CountRate | CountRates]
    _enable_checks: bool = True
    
    def __post_init__(self):
        """
        Runs consistency checks.
        """
        if self._enable_checks:
            for item in self.count_rates.values():
                if not self._first.campaign_id == item.campaign_id:
                        warnings.warn("Not matching campaign ids.")
                if not self._first.deposit_id == item.deposit_id:
                        warnings.warn("Not matching deposit ids.")

    @property
    def _first(self) -> CountRate:
        """
        ``nerea.Traverse._first()``
        ---------------------------
        The first element of ``self.count_rates``.

        Returns
        -------
        ``nerea.CountRate``"""
        return list(self.count_rates.values())[0]

    @property
    def deposit_id(self) -> str:
        """
        ``nerea.Traverse.deposit_id()``
        -------------------------------
        The deposit id of the first count rate.

        Returns
        -------
        ``str``"""
        return self._first.deposit_id

    def process(self,
                monitors: Iterable[CountRate| int],
                normalization: int|str=None,
                visual: bool=False,
                savefig: str='',
                palette: str='tab10',
                **kwargs) -> pd.DataFrame:
        """
        ``nerea.Traverse.process()``
        ----------------------------
        Normalizes all the count rates to the power in ``monitors``
        and to the maximum value.

        Parameters
        ----------
        **monitors** : ``Iterable[CountRate | int]``
            ordered information on the power normalization.
            Should be ``nerea.CountRate`` when mapped to a
            ``nerea.CountRate`` and int when mapped to ``nerea.CountRates``.
            The normalization is passed to ``CountRate.per_unit_time_power()``
            or ``CountRates.per_unit_time_power()``.
        **normalization** : ``str``, optional
            The ``self.count_rates`` CountRate identifier to normalize the traveres to.
            Defaults to ``None``, normalizing to the one with the highest counts.
        **visual** : ``bool``, optional
            Plots the processed data.
            Default is ``False``.
        **savefig** : ``str``, optional
            File name to save the plotted data to.
            Default is `''` for not saving.
        **palette** : ``str``, optional
            Color palette to use for plotting.
            Default is ``'tab10'``.
        **kwargs
            for `nerea.CountRate.plateau()`.

            - **sigma** (``int``): standard deviations for plateau finding
            - **timebase** (``int``): time base for integration in plateau search.
        
        Returns
        -------
        ``pd.DataFrame``
            with ``'value'``, ``'uncertainty'``, ``'uncertainty [%]'`` and
            ``'traverse'`` columns.

        Note
        ----
        - Working with ``nerea.CountRates`` instances, the first count rate is used."""
        normalized, m = {}, 0
        # Normalize to power
        for i, (k, rr) in enumerate(self.count_rates.items()):
            n = rr.per_unit_time_power(monitors[i], **kwargs)
            normalized[k] = n if isinstance(rr, CountRate) else list(n.values())[0]
            if normalized[k]['value'].value > m:
                max_k, m = k, normalized[k].value[0]
        norm_k = max_k if normalization is None else normalization
        out = []
        for k, v in normalized.items():
            out.append(_make_df(*ratio_v_u(v, normalized[norm_k])).assign(traverse=k))
        # plot
        if visual or savefig:
            fig, _ = self.plot(monitors, palette, **kwargs)
            if savefig:
                fig.savefig(savefig)
                plt.close()
        return pd.concat(out, ignore_index=True)

    def plot(self,
             monitors: Iterable[CountRate| int],
             palette: str='tab10',
             **kwargs) -> tuple[plt.Figure, Iterable[plt.Axes]]:
        """
        ``nerea.Traverse.plot()``
        -------------------------
        Plot the data processed in Traverse.

        Parameters
        ----------
        **monitors** : ``Iterable[CountRate | int]``
            ordered information on the power normalization.
            Should be ``nerea.CountRate`` when mapped to a ``nerea.CountRate``
            and ``int`` when mapped to ``nerea.CountRates``. The normalization
            is passed to ``CountRate.per_unit_time_power()`` or
            ``CountRates.per_unit_time_power()``.
        **palette** : ``str``, optional
            plt palette to use for plotting.
            Default is ``'tab10'``.
        **kwargs
            for `nerea.CountRate.plateau()`.
            - **sigma** (``int``): standard deviations for plateau finding
            - **timebase** (``int``): time base for integration in plateau search.
        
        Returns
        -------
        ``tuple[plt.Figure, Iterable[plt.Axes]]``"""
        fig, axs = plt.subplots(len(self.count_rates), 2,
                              figsize=(15, 30 / len(self.count_rates)))
        j = 0
        for i, (k, rr) in enumerate(self.count_rates.items()):
            c = plt.get_cmap(palette)(j)
            plat = rr.plateau(**kwargs)
            dur = (plat.Time.max() - plat.Time.min()).total_seconds()
            # plot data
            rr.plot(start_time=plat.Time.min(), duration=dur, ax=axs[i][0], c=c)
            axs[i][0].plot([], [], c=c, label=f"Traverse count rate {k}")
            # plot monitor
            axs[i][1] = monitors[i].plot(plat.Time.min(), dur, ax=axs[i][1], c=c)
            axs[i][1].plot([], [], c=c, label=f"Monitor count rate {k}")

            h, l = axs[i][0].get_legend_handles_labels()
            axs[i][0].legend(h[1:], l[1:])
            h, l = axs[i][1].get_legend_handles_labels()
            axs[i][1].legend(h[1:], l[1:])

            j = j + 1 if i < plt.get_cmap(palette).N else 0
        return fig, axs
