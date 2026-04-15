from collections.abc import Iterable
from typing import Self
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
from inspect import signature

from .utils import ratio_v_u, _make_df, time_integral_v_u, integral_v_u
from .functions import get_fit_R2, smoothing, find_plateau
from .defaults import *
from .classes import EffectiveDelayedParams, TimeSeriesImporter

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "TimeSeries",
    "Position",
    "CountRate",
    "CountRates"]


@dataclass(slots=True)
class TimeSeries:
    """
    ``nerea.TimeSeries``
    ====================
    Class to handle time dependent data.
    Supports `nerea.CountRate` and `nerea.Position`.

    Attributes
    ----------
    **data**: ``pd.DataFrame``
        data frame with time dependent data.
    """
    data: pd.DataFrame
    timebase: float  # seconds

    def smooth_data(self, **kwargs) -> Self:
        """
        `nerea.TimeSeries.smooth()`
        ---------------------------
        Smooths the time series data to ease feature recognition.

        Parameters
        ----------
        **kwargs
        Argumnents for ``nerea.functions.smoothing()``
            - **renormalize** (``bool``): Whether to renormalize the data.
            - **smoothing_method** (``str``): The mehtod to implement for smoothing.
            - arguments for the chosen ``nerea.functions.smoothing``.
        
        Returns
        -------
        ``pd.TimeSeries``
            data frame with time and counts columns.
        
        Notes
        -----
        Allowed methods are
            - ``'moving_average'`` (requires ``window``)
            - ``'ewm'``
            - ``'savgol_filter'`` (requires ``window_length``, ``polyorder``)
            - ``'fit'``(requires ``ch_before_max``, ``order``)"""
        kw = DEFAULT_SMOOTH_KWARGS | kwargs
        out = pd.DataFrame({"Time": self.data["Time"],
                            "value": smoothing(self.data["value"], **kw)})
        return TimeSeries(data=out, timebase=self.timebase)

    def cut_data(self, start: datetime, end: datetime) -> Self:
        """
        `nerea.TimeSeries.cut()`
        -----------------------
        Cuts time series data from a set start to an end.
        
        Parameters
        ----------
        **start** : ``datetime``
            start time of the new `nerea.CountRate`.
        **end** : ``datetime``
            end time of the new `nerea.CountRate`.
        
        Returns
        -------
        ``nerea.TimeSeries``
            instance with truncated data.
            
        Notes
        -----
        Left boundary included, right boundary excluded."""
        data = self.data.query("Time >= @start and Time < @end")
        return TimeSeries(data, timebase=self.timebase)

    def plot_data(self,
        start_time: datetime=None,
        duration: int=None,
        experiment_time: bool=False,
        ax: plt.Axes=None,
        c: str='k',
        **kwargs) -> plt.Axes:
        """
        `nerea.TimeSeries.plot()`
        ------------------------
        Plot data in this TimeSeries instance.

        Parameters
        ----------
        **start_time** : ``datetime.datetime``, optional
            The time the count rate is considered from.
            Default is ``None`` for first acquisition time.
        **duration** : ``int``, optional
            The time-span the count rate is considered for.
            Default is ``None`` for until last acquisition time.
        **ax** : ``plt.Axes``, optional
            The ax where the plot is drawn.
            Defauls is ``None`` for a new axes.
        **c** : ``str``, optional
            The color of the plotted seriese.
            Default is ``'k'``.
        **kwargs
            Additional arguments for ``pd.DataFrame.plot()``
        
        Returns
        -------
        ``plt.Axes``
            with the plotted data."""
        start_time_ = start_time if start_time is not None else self.data.Time.min()
        duration_ = duration if duration is not None else (
            self.data.Time.max() - start_time_).total_seconds()
        data = self.cut_data(start_time_,
                             start_time_ + timedelta(seconds=duration_)
                             ).data
        if experiment_time:
            data.Time = range(data.Time.shape[0])
        ax = data.plot(x="Time",
                       y='value',
                       ax=ax,
                       color=c,
                       kind='scatter',
                       **kwargs)
        return ax

    def average(self,
                start_time: datetime,
                duration: float,
                uncertainty: str='poisson') -> pd.DataFrame:
        """
        `nerea.TimeSeries.average()`
        ---------------------------
        Calculates the average value and uncertainty of
        time series data within a specified duration.

        Parameters
        ----------
        **start_time** : ``datetime.datetime``
            The starting time for the data to be analyzed.
        **duration** : ``float``
            The length of time in seconds for which
            the average is calculated.
        **uncertainty** : ``str``, optional
            policy for uncertainty calculation.
            Available options are:
            - ``'poisson'``
            - ``'std'``
            - ``'sem'``
            Default is ``'poisson'``.

        Returns
        -------
        ``pd.DataFrame``
            data frame containing average `'value'` and `'uncertainty'` columns.

        Notes
        -----
        - uncertainty computed assuming Poisson distribution: 1/sqrt(`value`)
            
        Examples
        --------
        >>> from datetime import datetime
        >>> data = pd.DataFrame({'Time': pd.date_range('2021-01-01', periods=100, freq='S'),
                                 'value': np.random.rand(100)})
        >>> pm = CountRate(data=data, start_time=datetime(2021, 1, 1), 
                              campaign_id='C1', experiment_id='E1', detector_id='M1')
        >>> avg_df = pm.average(datetime(2021, 1, 1, 0, 0, 30), 10)
        >>> print(avg_df)"""
        # end_time should be 1 timebase after the real end time to use
        end_time = start_time + timedelta(seconds=duration + self.timebase)
        series = self.cut_data(start_time, end_time).data
        if series.empty:
            raise ValueError("No time series data in the requested interval.")
        v, u = time_integral_v_u(series)
        delta = (series.Time.iloc[-1] - series.Time.iloc[0]).total_seconds()
        v /= delta
        uncertainty_ = 'std' if v <= 0 else uncertainty
        if uncertainty_ == 'poisson':
            u /= delta
        elif uncertainty_ == 'std':
            # time_integral_v_u ignores last point
            u = np.std(series['value'][:-1], ddof=1)
        elif uncertainty_ == 'sem':
        # time_integral_v_u ignores last point
            u = np.std(series['value'][:-1], ddof=1
                       ) / np.sqrt(np.size(series['value']) - 1)
        else:
            raise ValueError(f"'{uncertainty}' is not an allowed uncertainty policy.")
        relative = True if v != 0 else False
        # Time Normalization of the Average
        # If the RR time binning is not 1 s, there is a chance the query truncated
        # the time series so that the difference between first and last time in
        # `series` are closer than duration. Hence I define delta.
        # iloc[-1] explained by the use of time_integral_v_u(): we stop at the 
        # beginning of the next step-post time stamp.
        return _make_df(v, u, relative)

    @classmethod
    def from_importer(cls, importer: TimeSeriesImporter):
        return cls(importer.data, importer.metadata['timebase'])


@dataclass(slots=True)
class Position(TimeSeries):
    """
    ``nerea.Position``
    ==================
    Class to handle time dependent position data.
    Inherits from ``nerea.TimeSeries``.

    Attributes
    ----------
    **data**: ``pd.DataFrame``
        data frame with time dependent data."""    
    def plateau(self,
                smooth: bool=False,
                **kwargs) -> pd.DataFrame:
        """
        `nerea.TimeSeries.plateau()`
        ----------------------------
        Identifies plateaus in the time series.

        Parameters
        ----------
        **smooth** : ``bool``
            flag to smooth the data before plateau search.
            Default is ``False``.

        **kwargs
        Additional arguments for
            
            - ``nerea.functions.find_plateau()``
                - **tol** (``float``) tolerance for plateau finding.
                - **absolute_tolerance** (``bool``) flag for absolute ``tol``.
                - **min_length** (``int``) minimal number of bins in plateau.

            - ``nerea.functions.smoothing()``
                - **renormalize** (``bool``): Whether to renormalize the data.
                - **smoothing_method** (``str``): The mehtod to implement for smoothing.
                - arguments for the chosen ``nerea.functions.smoothing``.
        
        Returns
        -------
        ``pd.DataFrame``
            data frame with time and counts columns.
        
        Notes
        -----
        Allowed methods are
            - ``'moving_average'`` (requires ``window``)
            - ``'ewm'``
            - ``'savgol_filter'`` (requires ``window_length``, ``polyorder``)
            - ``'fit'``(requires ``ch_before_max``, ``order``)"""
        plateau_kw = {k: v for k, v in kwargs.items() if k in set(signature(find_plateau).parameters)}
        data = self.smooth_data(**kwargs).data if smooth else self.data.copy()
        return find_plateau(data['Time'], data['value'], **plateau_kw)

    def from_plateau(self, **kwargs):
        out = []
        for i, p in self.plateau(**kwargs).T.iterrows():
            cut = self.cut_data(p['start'],
                                p['end'] + timedelta(seconds=self.timebase))
            out.append(cut.data.assign(PLATEAU=i))
        return self.__class__(pd.concat(out, ignore_index=True), timebase=self.timebase)


@dataclass(slots=True)
class CountRate(TimeSeries):
    """
    ``nerea.CountRate``
    ===================
    Class storing and processing count rate data acquired as 
    a function of time.
    Inherits from ``nerea.TimeSeries``.

    Attributes
    ----------
    **data**: ``pd.DataFrame``
        the count rate as a function of time data.
    **start_time**: ``datetime``
        acquisition start time.
    **campaign_id**: ``str``
        metadatada for experimental campaign identification.
    **experiment_id**: ``str``
        metadata for experiment identification
    **detector_id**: ``int|str``
        metadata for detector identification
    **deposit_id**: ``str``
        metadata for deposit identification
    **timebase**: ``float``, optional
        acquisition timebase in seconds. Default is ``1.0``.
    _dead_time_corrected: ``bool``, optional
        flag labelling whether the count rates have been
        corrected for dead time. Handled internally.
        Default is ``False``.
    _vlines: ``Iterable[datetime]``, optional
        lines to draw plotting. Handled internally.
        Default is ``[]``."""
    start_time: datetime
    campaign_id: str
    experiment_id: str
    detector_id: str
    deposit_id: str
    _dead_time_corrected: bool = False
    _vlines: Iterable[datetime] = field(default_factory=lambda: [])

    def _linear_fit(self,
                    preprocessing: str='log',
                    nonzero: bool=True
                    ) -> tuple[pd.Series, np.ndarray, np.ndarray, dict]:
        """
        `nerea.CountRate._linear_fit`
        -----------------------------
        Linearly fits monitor data after preprocessing.

        Parameters
        ----------
        **preprocessing** : ``str``, optional
            ``numpy`` function to apply to ``self.data`` prior to
            linear fitting. Default is ``'log'``.
        **nonzero** : ``bool``, optional
            queries non-zero values in ``self.data``.
            Default is ``True``.
            
        Returns
        -------
        tuple[pd.Series, np.ndarray, np.ndarray, dict]
        fit output:
            - The dependent data
            - Parameter values minimizing RMSE
            - Parameter covariance
            - information"""
        from scipy.optimize import curve_fit
        def linear_fit(x, a, b):
            return x / a + b  # Linear fit function (a = T)
        if nonzero:
            data = self.data[self.data.value != 0]
            if data.shape != self.data.shape:
                str1 = "Removing 0 counts from Count Rate to enable period log fit. "
                str2 = f"Removed {(self.data.shape[0] - data.shape[0])} rows."
                warnings.warn(str1 + str2)
        if preprocessing is not None:
            y = getattr(np, preprocessing)(data.value)  # apply preprocessing
        else:
            y = data.value
        x = (data.Time - self.start_time).dt.total_seconds()  # x must be in seconds from 0
        popt, pcov, out, _, _ = curve_fit(linear_fit, x, y,
                                          full_output=True,
                                          absolute_sigma=False)  # sigma scaled to math sample variance
        return y, popt, pcov, out

    def smooth(self, **kwargs) -> Self:
        """
        `nerea.CountRate.smooth()`
        --------------------------
        Smooths the count rate data to ease feature recognition.

        Parameters
        ----------
        **kwargs
        Argumnents for ``nerea.functions.smoothing()``
            - **renormalize** (``bool``): Whether to renormalize the data.
            - **smoothing_method** (``str``): The mehtod to implement for smoothing.
            - arguments for the chosen ``nerea.functions.smoothing``.
        
        Returns
        -------
        ``pd.CountRate``
            data frame with time and counts columns.
        
        Notes
        -----
        Allowed methods are
            - ``'moving_average'`` (requires ``window``)
            - ``'ewm'``
            - ``'savgol_filter'`` (requires ``window_length``, ``polyorder``)
            - ``'fit'``(requires ``ch_before_max``, ``order``)"""
        if kwargs.get('window') or kwargs.get('window_lenght'):
            w = kwargs['window'] if kwargs['smoothing_method'] == 'moving_average' else kwargs['window_length']
            if w < self.timebase:  ## if nor window nor window_length are passed w is False
                raise ValueError("Smoothing window length should be larger than the Count Rate timebase.")
        else:
            out = self.smooth_data(**kwargs).data
            return self.__class__(
                data=out,
                start_time=self.start_time,
                campaign_id=self.campaign_id,
                experiment_id=self.experiment_id,
                detector_id=self.detector_id,
                deposit_id=self.deposit_id,
                timebase=self.timebase,
                _dead_time_corrected=self._dead_time_corrected
            )

    def integrate(self, timebase: int, start_time: datetime | None = None) -> pd.DataFrame:
        """
        `nerea.CountRate.integrate()`
        -----------------------------
        Integrates data over a specified timebase starting
        from a given start time.

        Parameters
        ----------
        **timebase** : ``int``
            The interval of time in seconds over which to calculate the average.
            This interval is used to group the data for averaging.
        **start_time** : ``datetime``, optional
            The starting time for the integration process. Default is ``self.start_time``.

        Returns
        -------
        ``pd.DataFrame``
            data frame containing average `'value'` and `'uncertainty'` columns.

        Notes
        -----
        - uncertainty computed assuming Poisson distribution: 1/sqrt(`value`)

        Examples
        --------
        >>> from datetime import datetime
        >>> data = pd.DataFrame({'Time': pd.date_range('2021-01-01', periods=100, freq='S'),
                                 'value': np.random.rand(100)})
        >>> pm = CountRate(data=data, start_time=datetime(2021, 1, 1), 
                              campaign_id='C1', experiment_id='E1', detector_id='M1')
        >>> integrated_df = pm.integrate(10)
        >>> print(integrated_df)"""
        start_time_ = self.start_time if start_time is None else start_time
        out = []
        while start_time_ < self.data.Time.max():
            out.append(self.average(start_time_, timebase))
            start_time_ = start_time_ + timedelta(seconds=timebase)
        return pd.concat(out, ignore_index=True)

    def plateau(self, sigma: int=2, timebase: int=10) -> pd.DataFrame:
        """
        `nerea.CountRate.plateau()`
        ---------------------------
        The plateau with the largest integral counts in the detector.

        Parameters
        ----------
        **sigma** : ``int``, optional
            the amount of standard deviations to consider for the
            uncertainty on the plateau.
            Defaults to ``2``.
        **timebase** : ``int``, optional
            the time base for integration in plateau search in seconds.
            Defaults to ``10`` s.
        
        Returns
        -------
        ``pd.DataFrame``
            with ``'Time'`` and ``'value'`` columns."""
        time = self.data.Time.min()
        plateau_start_time, plateau_end_time = self.data.Time.min(), self.data.Time.min()
        sum, max = 1, -1
        while time < self.data.Time.max():
            # compute the integral over timebase
            time_plus_timedelta = time + timedelta(seconds=timebase)
            series = self.data.query("Time > @time and Time <= @time_plus_timedelta").value
            local_sum, _ = integral_v_u(series)
            # check if new plateau starts
            same_plateau = np.isclose(local_sum, sum,
                                      atol=(np.sqrt(local_sum) + np.sqrt(sum)) * sigma)
            # local_plateau = self.data.query("Time > @plateau_start_time and Time <= @time_plus_timedelta")
            if same_plateau:
                plateau_end_time = time_plus_timedelta
            else:
                if self.data.query("Time > @plateau_start_time and Time <= @plateau_end_time").value.sum() > max:
                    max = self.data.query("Time > @plateau_start_time and Time <= @plateau_end_time").value.sum()
                    max_plateau_start_time = plateau_start_time
                    max_plateau_end_time = plateau_end_time
                plateau_start_time = time_plus_timedelta
            # update iteration variables (next timebin next sum)
            time = time_plus_timedelta
            sum = local_sum
        if plateau_end_time == self.data.Time.min():
            raise Exception(f"No plateau found in for detector {self.detector_id} in experiment {self.experiment_id}.")
        return self.data.query("Time > @max_plateau_start_time and Time <= @max_plateau_end_time")

    def filter_on_position_plateau(self, position: Position, **kwargs) -> list[Self]:
        """
        `nerea.CountRate.filter_on_position_plateau()`
        ----------------------------------------------
        Filters the count rate data based on plateau found on
        a `nerea.Position` instance.

        Parameters
        ----------
        **position** : ``nerea.Position``
            the position based on which plateau the data should
            be filtered.

        **kwargs
        - **smooth** (``bool``) flag smoothing position data before plateau search.
        Additional arguments for
            
            - ``nerea.functions.find_plateau()``
                - **tol** (``float``) tolerance for plateau finding.
                - **absolute_tolerance** (``bool``) flag for absolute ``tol``.
                - **min_length** (``int``) minimal number of bins in plateau.

            - ``nerea.functions.smoothing()``
                - **renormalize** (``bool``): Whether to renormalize the data.
                - **smoothing_method** (``str``): The mehtod to implement for smoothing.
                - arguments for the chosen ``nerea.functions.smoothing``.
        
        Returns
        -------
        ``list[nere.CountRate]``
            corresponding to the different plateau."""
        out = []
        for _, p in position.plateau(**kwargs).T.iterrows():
            out.append(self.cut(p['start'], p['end'] + timedelta(seconds=self.timebase)))
        return out

    def per_unit_power(self, monitor: Self, check_count_rate_plateau: bool=True, **kwargs) -> pd.DataFrame:
        """
        `nerea.CountRate.per_unit_power()`
        ----------------------------------
        Normalizes the count rate to a power monitor.

        Parameters
        ----------
        **monitor** : ``nerea.CountRate``
            The power monitor for the count rate normalization.
        **check_count_rate_plateau** : ``bool``, optional
            flag to define whether to check for count rate plateau to
            compute the traverse or not. Default is ``True``.
        **kwargs
            arguments for ``self.plateau()``.
            - **sigma** (``int``): standard deviations for plateau finding.
            - **timebase** (``int``): integration timebase in seconds.
        
        Returns
        -------
        ``pd.DataFrame``
            with ``'value'`` and ``'uncertainty'`` columns.
            
        Notes
        -----
        ``kwargs`` ignored if ``check_count_rate_plateau == False``."""
        if check_count_rate_plateau:
            plateau = self.plateau(**kwargs)
        else:
            plateau = self.data
        duration = (plateau.Time.max() - plateau.Time.min()).seconds
        normalization = monitor.average(plateau.Time.min(), duration) 
        return _make_df(*ratio_v_u(_make_df(*integral_v_u(plateau.value)), normalization))

    def per_unit_time_power(self, monitor: Self, check_count_rate_plateau: bool=True, *args, **kwargs) -> pd.DataFrame:
        """
        `nerea.CountRate.per_unit_time_power()`
        ---------------------------------------
        Normalizes the count rate to a power monitor and gives
        the conunt rate per unit power.

        Parameters
        ----------
        **monitor** : CountRate
            The power monitor for the count rate normalization.
        **check_count_rate_plateau** : ``bool``, optional
            flag to define whether to check for count rate plateau to
            compute the traverse or not. Default is ``True``.
        
        Parameters
        ----------
        **monitor** : ``nerea.CountRate``
            The power monitor for the count rate normalization.
        **kwargs
            arguments for ``self.plateau()``.
            - **sigma** (``int``): standard deviations for plateau finding.
            - **timebase** (``int``): integration timebase in seconds.
        
        Returns
        -------
        ``pd.DataFrame``
            with ``'value'`` and ``'uncertainty'`` columns.
            
        Notes
        -----
        ``kwargs`` ignored if ``check_count_rate_plateau == False``."""
        if check_count_rate_plateau:
            plateau = self.plateau(**kwargs)
        else:
            plateau = self.data
        duration = (plateau.Time.max() - plateau.Time.min()).seconds
        unit_p = self.per_unit_power(monitor, check_count_rate_plateau, *args, **kwargs)
        return _make_df(unit_p.value / duration, unit_p.uncertainty / duration)

    def dead_time_corrected(self, tau_p: float = 88e-9, tau_np: float = 108e-9) -> Self:
        """
        `nerea.CountRate.dead_time_corrected()`
        ---------------------------------------
        Apply dead time correction to the count rate data.
        
        Parameters
        ----------
        **tau_p** : ``float``, optional
            paralizable dead time constant.
            Defaults to ``88e-9``.
        **tau_np** : ``float``, optional
            non-paralizable dead time constant.
            Defaults to ``108e-9``.
        
        Returns
        -------
        ``nerea.CountRate``
            instance with dead time corrected data."""
        if self._dead_time_corrected:
            pm = self.data.copy()
        else:
            from scipy import optimize
            def dead_time_correction(n, m, tp, tnp): 
                # Equation for dead time correction
                return n / ((1 - tp / tnp) * n * tp + np.exp(tp * n)) - m
            if self._dead_time_corrected:
                logger.info("Dead time correction already applied to this detector.")
            pm = self.data.copy()
            pm["value"] = pm.value.apply(lambda x:
                                        optimize.newton(lambda n:
                                                        dead_time_correction(n, x, tau_p, tau_np),
                                                        x))
        return self.__class__(pm,
                              start_time=self.start_time,
                              campaign_id=self.campaign_id,
                              experiment_id=self.experiment_id,
                              detector_id=self.detector_id,
                              deposit_id=self.deposit_id,
                              timebase=self.timebase,
                              _dead_time_corrected=True)

    def cut(self, start: datetime, end: datetime) -> Self:
        """
        `nerea.CountRate.cut()`
        -----------------------
        Cuts count rate data from a set start to an end.
        
        Parameters
        ----------
        **start** : ``datetime``
            start time of the new `nerea.CountRate`.
        **end** : ``datetime``
            end time of the new `nerea.CountRate`.
        
        Returns
        -------
        ``nerea.CountRate``
            instance with truncated data.
            
        Notes
        -----
        Left boundary included, right boundary excluded."""
        data = self.cut_data(start, end).data
        return self.__class__(data,
                              start_time=data.Time.min(),
                              campaign_id=self.campaign_id,
                              experiment_id=self.experiment_id,
                              detector_id=self.detector_id,
                              deposit_id=self.deposit_id,
                              timebase=self.timebase,
                              _dead_time_corrected=self._dead_time_corrected
                              )

    def get_asymptotic_period(self,
                              scan_dt: float=0.,
                              scan_dt0: float=20.,
                              scan_tol: float=1e-2,
                              log: bool=True) -> pd.DataFrame:
            """
            `nerea.CountRate.get_asymptotic_period()`
            -----------------------------------------
            Calculats the reactor period from a CountRate instance.

            Parameters
            ----------
            **scan_dt** : ``float``, optional
                time delta to evaluate asymptotic convergence.
                In seconds. Default is ``0.`` for no scan.
            **scan_dt0** : ``float``, optional
                seconds to skip from last: buffer time to have
                a reasonable period estimate to converge to.
                In seconds. Default is ``20.``.
            **scan_tol** : ``float``, optional
                tolerance to evaluate asymptotic convergence.
                Relative. Default is ``1e-2``.
            **log** : ``bool``, optional
                flag to log fit R^2. Default is ``True``.

            Returns
            -------
            ``pd.DataFrame``
                with reactor asymptotic period value and uncertainty.
                
            Note
            ----
            The scan is performed backwards, starting from the
            later time in ``self`` - ``scan_dt0``. time spacing is
            defined by ``scan_dt`` and tolerance by ``scan_tol``."""
            if scan_dt == 0:
                fitted_data, popt, pcov, out = self._linear_fit()
                period = _make_df(popt[0], np.sqrt(pcov[0, 0]))
                r2 = get_fit_R2(fitted_data, out['fvec'])
                if log:
                    logger.info(f"Reactor period fit R^2 = {r2:.4f}")
            else:
                # starting from end
                t0 = self.data.Time.max() - timedelta(seconds=scan_dt0)
                full_dt = (t0 - self.start_time).total_seconds()
                nbins = int(np.floor(full_dt / scan_dt))
                old_period = 0
                for i in np.linspace(scan_dt, full_dt, nbins):
                    ts = t0 - timedelta(seconds = i)
                    # period from ts to end of cut data
                    cr = self.cut(ts, self.data.Time.max())
                    period = cr.get_asymptotic_period(log=False
                                ).value.values[0]
                    if old_period == 0:
                        old_period = period
                    else:
                        if (period / old_period - 1) <= scan_tol:
                            old_period = period
                        else:
                            break
                period = self.cut(ts + timedelta(seconds=scan_dt),
                                  self.data.Time.max()
                                  ).get_asymptotic_period(log=log)
            return period

    def get_reactivity(self,
                       delayed_data: EffectiveDelayedParams,
                       ap_kwargs: dict={}) -> pd.DataFrame:
        """
        `nerea.CountRate.get_reactivity()`
        ----------------------------------
        Calculates the reactor reactivity based on the Count Rate-estimated
        reactor period and on effective nuclear data computed by Serpent.
        
        Parameters
        ----------
        **delayed_data** : ``nerea.EffectiveDelayedParams``
            effective delayed neutron paramters to use for
            reactivity calculation.

        Returns
        -------
        ``pd.DataFrame``
            data frame with ``'value'`` and ``'uncertainty'`` columns."""
        ap_kw = DEFAULT_AP_KWARGS | ap_kwargs
        bi = delayed_data.beta_i
        li = delayed_data.lambda_i

        # compute reactivity
        assert(all(self.data.value.values != np.nan))
        assert(all(self.data.value.values != 0))

        T = self.get_asymptotic_period(**ap_kw)
        rho = np.sum(bi.value / (1 + li.value * T.value))

        # variance portions
        VAR_PORT_T = np.sum((-bi.value * li.value / (1 + li.value * T.value)**2 * T.uncertainty) **2)
        VAR_PORT_B = np.sum((1 / (1 + li.value * T.value) * bi.uncertainty) **2)
        VAR_PORT_L = np.sum((-bi.value * T.value / (1 + li.value * T.value)**2 * li.uncertainty) **2)
        return _make_df(rho, np.sqrt(VAR_PORT_T + VAR_PORT_B + VAR_PORT_L)).assign(VAR_PORT_T=VAR_PORT_T,
                                                                                   VAR_PORT_B=VAR_PORT_B,
                                                                                   VAR_PORT_L=VAR_PORT_L)

    def plot(self,
        start_time: datetime=None,
        duration: int=None,
        experiment_time: bool=False,
        ax: plt.Axes=None,
        c: str='k',
        **kwargs) -> plt.Axes:
        """
        `nerea.TimeSeries.plot()`
        ------------------------
        Plot data in this TimeSeries instance.

        Parameters
        ----------
        **start_time** : ``datetime.datetime``, optional
            The time the count rate is considered from.
            Default is ``None`` for first acquisition time.
        **duration** : ``int``, optional
            The time-span the count rate is considered for.
            Default is ``None`` for until last acquisition time.
        **ax** : ``plt.Axes``, optional
            The ax where the plot is drawn.
            Defauls is ``None`` for a new axes.
        **c** : ``str``, optional
            The color of the plotted seriese.
            Default is ``'k'``.
        **kwargs
            Additional arguments for ``pd.DataFrame.plot()``
        
        Returns
        -------
        ``plt.Axes``
            with the plotted data."""
        ax = self.plot_data(start_time,
                            duration,
                            experiment_time,
                            ax,
                            c,
                            **kwargs)
        if not experiment_time and self._vlines:
            for v in self._vlines:
                ax.axvline(v, ls='--', c='gray', label="File joining")
            ax.legend()
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.set_ylabel("Power monitor count rate [1/s]")
        ax.tick_params(axis='y', left=False, labelleft=False, right=True, labelright=True)
        ax.yaxis.set_label_position("right")
        t = ax.yaxis.get_offset_text()
        t.set_x(1.01)
        return ax

    @classmethod
    def from_importer(cls, importer: TimeSeriesImporter):
        return cls(importer.data, **importer.metadata)


@dataclass(slots=True)
class CountRates:
    """
    ``nerea.CountRates``
    =============================
    Class storing and processing count rate data acquired as 
    a function of time. Stores data of many detectors/acquisitions.

    Attributes
    ----------
    **detectors** : ``dict[int, nerea.CountRate]``
        Links detector id and its conunt rate.
        ``key`` is the id and ``value`` the count rate.
    _enable_checks: ``bool``, optional
        flag to enable consistency checks. Default is ``True``."""
    detectors: dict[int | str, CountRate]
    _enable_checks: bool = True

    def __post_init__(self) -> None:
        """
        Runs consistency checks.
        """
        if self._enable_checks:
            self._check_consistency()

    def _check_consistency(self, time_tolerance: timedelta=timedelta(seconds=60),
                           timebase: int=100, sigma=1) -> None:
        """
        `nerea.CountRates._check_consistency()`
        ---------------------------------------
        Check the consistency of time and curve data with specified parameters.

        Parameters
        ----------
        **time_tolerance** : ``datetime.timedelta``, optional
            Parameter for ``self._check_time_consistency``. Defaults to ``60`` seconds.
        **timebase** : ``int``, optional
            Parameter for ``self._check_curve_consistency``. Defaults to ``100``.
        **sigma** : int, optional
            Parameter for ``self._check_curve_consistency``. Defaults to ``1``.

        Examples
        --------
        >>> data = pd.DataFrame({'Time': pd.date_range('2021-01-01', periods=100, freq='S'),
                                 'value': np.random.rand(100)})
        >>> pm1 = CountRate(data=data, start_time=datetime(2021, 1, 1), 
                               campaign_id='C1', experiment_id='E1', detector_id='M1')
        >>> pm2 = CountRate(data=data, start_time=datetime(2021, 1, 1), 
                               campaign_id='C1', experiment_id='E1', detector_id='M2')
        >>> pms = CountRates(detectors={1: pm1, 2: pm2})
        >>> pms._check_consistency()"""
        must = ['campaign_id', 'experiment_id']
        for attr in must:
            if not all([getattr(m, attr) == getattr(list(self.detectors.values())[0], attr)
                        for m in self.detectors.values()]):
                raise Exception(f"Inconsistent {attr} among different CountRate instances.")
        should = ['deposit_id']
        for attr in should:
            if not all([getattr(m, attr) == getattr(list(self.detectors.values())[0], attr)
                        for m in self.detectors.values()]):
                warnings.warn(f"Inconsistent {attr} among different CountRate instances.")
        self._check_time_consistency(time_tolerance)
        self._check_curve_consistency(timebase, sigma)

    def _check_time_consistency(self, time_tolerance: timedelta) -> None:
        """
        `nerea.CountRates._check_time_consistency()`
        --------------------------------------------
        Check if the start times of power detectors are
        consistent within a given time tolerance.

        Parameters
        ----------
        **time_tolerance** : ``datetime.timedelta``
            The maximum allowable difference in time between
            the start times of the power detectors in ``self.detectors``.

        Examples
        --------
        >>> data = pd.DataFrame({'Time': pd.date_range('2021-01-01', periods=100, freq='S'),
                                 'value': np.random.rand(100)})
        >>> pm1 = CountRate(data=data, start_time=datetime(2021, 1, 1), 
                               campaign_id='C1', experiment_id='E1', detector_id='M1')
        >>> pm2 = CountRate(data=data, start_time=datetime(2021, 1, 1), 
                               campaign_id='C1', experiment_id='E1', detector_id='M2')
        >>> pms = CountRates(detectors={1: pm1, 2: pm2})
        >>> pms._check_time_consistency(timedelta(seconds=60))"""
        ref = list(self.detectors.values())[0].start_time
        for monitor in self.detectors.values():
            if not abs(monitor.start_time - ref) < time_tolerance:
                warnings.warn(f"Power monitor start time difference > {time_tolerance}")

    def _check_curve_consistency(self, timebase: int, sigma: int=1) -> None:
        """
        `nerea.CountRates._check_curve_consistency()`
        ---------------------------------------------
        Compare data from multiple detectors to check for consistency
        within a sigma-uncertainty tolerance, based on a specified
        timebase.

        Parameters
        ----------
        **timebase** : ``int``
            The time interval in seconds for grouping the data.
            This parameter determines how the data are aggregated
            and compared between different detectors.
        **sigma** : ``int``, optional
            The uncertainty associated with the measurements. It
            is used to calculate the tolerance for checking the
            consistency between different power detectors. The
            tolerance is computed as the average uncertainty on the
            ratio of values between two detectors. Defaults to ``1``.

        Examples
        --------
        >>> data = pd.DataFrame({'Time': pd.date_range('2021-01-01', periods=100, freq='S'),
                                 'value': np.random.rand(100)})
        >>> pm1 = CountRate(data=data, start_time=datetime(2021, 1, 1), 
                               campaign_id='C1', experiment_id='E1', detector_id='M1')
        >>> pm2 = CountRate(data=data, start_time=datetime(2021, 1, 1), 
                               campaign_id='C1', experiment_id='E1', detector_id='M2')
        >>> pms = CountRates(detectors={1: pm1, 2: pm2})
        >>> pms._check_curve_consistency(10, 1)"""
        ref = list(self.detectors.values())[0].data.groupby(
                                                    pd.Grouper(key="Time", freq=f'{timebase}s', closed='right'),
                                                    observed=False
                                                    ).agg('sum').reset_index()
        ref['uncertainty'] = np.sqrt(ref['value'])  # absolute
        ref['value'] = ref['value'] / timebase  
        for monitor in list(self.detectors.values())[1:]:
            compute = monitor.data.groupby(pd.Grouper(key="Time", freq=f'{timebase}s', closed='right'),
                                           observed=False
                                            ).agg('sum').reset_index()
            compute['uncertainty'] = np.sqrt(compute['value'])  # absolute
            compute['value'] = compute['value'] / timebase
            # filtering noise below 100 counts in time
            start = max(ref.query('value >= 100').Time.min(), compute.query('value >= 100').Time.min())
            end = min(ref.query('value >= 100').Time.max(), compute.query('value >= 100').Time.max())
            qs = "Time >= @start and Time <= @end"
            v, u = ratio_v_u(compute.query(qs), ref.query(qs))
            # tolerance is scalar, therefore it is computed as average uncertainty on the ratio
            tol = np.mean(sigma * u)  # absolute
            if not (np.isclose(v, np.roll(v, shift=1), atol=tol)).all():
                warnings.warn(f"Power monitor {monitor.detector_id} inconsistent with {list(self.detectors.values())[0].detector_id}")

    @property
    def _first(self) -> CountRate:
        """
        `nerea.CountRates._first()`
        ---------------------------
        The first count rate in ``self.detectors``.

        Returns
        -------
        ``nerea.CountRate``
            the first count rate."""
        return list(self.detectors.values())[0]

    @property
    def campaign_id(self) -> str:
        """
        `nerea.CountRates.campaign_id()`
        --------------------------------
        Campaign id of the first count rate in ``self.detectors``.

        Returns
        -------
        ``str``
            the campaign id of the first detector."""
        return self._first.campaign_id

    @property
    def experiment_id(self) -> str | int:
        """
        `nerea.CountRates.experiment_id()`
        ----------------------------------
        Experiment id of the first count rate in ``self.detectors``.

        Returns
        -------
        ``str``
            the campaign id of the first detector."""
        return self._first.experiment_id

    @property
    def deposit_id(self) -> str:
        """
        `nerea.CountRates.deposit_id()`
        -------------------------------
        The deposit id of the first element of `self.detectors`.
        
        Returns
        -------
        ``str``
            the deposit id of the first detector."""
        return self._first.deposit_id

    @property
    def best(self) -> CountRate:
        """
        `nerea.CountRates.best()`
        -------------------------
        Returns the count rate with the highest sum value.

        Returns
        -------
        ``nerea.CountRate``
            Count rate with the highest integral count."""
        max = list(self.detectors.values())[0].data.value.sum()
        out = list(self.detectors.values())[0]
        for monitor in list(self.detectors.values())[1:]:
            if monitor.data.value.sum() > max:
                out = monitor
        return out

    def per_unit_power(self, monitor: int, **kwargs) -> dict[int, pd.DataFrame]:
        """
        `nerea.CountRates.per_unit_power()`
        -----------------------------------
        Normalizes the raction rate to a power monitor.

        Parameters
        ----------
        **monitor** : ``int``
            The ID of the count rate to be used as power
            monitor for the count rate normalization.
        **kwargs
            arguments for ``CountRate.plateau()``.
            - **sigma** (``int``): standard deviations for plateau finding.
            - **timebase** (``int``): integration timebase in seconds.

        Returns
        -------
        ``dict[int, pd.DataFrame]``
            with value and uncertainty of the normalized count rate
            integrated over time. Keys are the detector IDs as in
            self.detectors."""
        out = {}
        for key, detector in self.detectors.items():
            if key != monitor:
                out[key] = detector.per_unit_power(self.detectors[monitor], **kwargs)
        return out

    def per_unit_time_power(self, monitor: int, **kwargs) -> dict[int, pd.DataFrame]:
        """
        `nerea.CountRates.per_unit_time_power()`
        ----------------------------------------
        Normalizes the raction rate to a power monitor and takes the average over time.

        Parameters
        ----------
        **monitor** : ``int``
            The ID of the count rate to be used as power
            monitor for the count rate normalization.
        **kwargs
            arguments for ``CountRate.plateau()``.
            - **sigma** (``int``): standard deviations for plateau finding.
            - **timebase** (``int``): integration timebase in seconds.

        Returns
        -------
        ``dict[int, pd.DataFrame]``
            with value and uncertainty of the normalized count rate
            averaged over time. Keys are the detector IDs as in
            self.detectors."""
        out = {}
        for key, detector in self.detectors.items():
            if key != monitor:
                out[key] = detector.per_unit_time_power(self.detectors[monitor], **kwargs)
        return out

    def cut(self, starts: list, ends: list):
        dts = {}
        for i, (j, d) in enumerate(self.detectors.items()):
            dts[j] = d.cut(starts[i], ends[i])
        return self.__class__(dts, self._enable_checks)

    @classmethod
    def from_importers(cls, importers: list[TimeSeriesImporter], **kwargs) -> Self:
        """
        `nerea.CountRates.from_importers()`
        -----------------------------------
        Creates an instance of ``nerea.CountRates`` using data extracted
        from an ASCII file through ``nerea.TimeSeriesImporter``.

        Parameters
        ----------
        **importers** : list[TimeSeriesImporter]
            The importers to use for initialization.
        **kwargs
            additional arguments for class creation
            - **_enable_checks**: turns consistency checks on/off.

        Returns
        -------
        ``nerea.CountRates``
            initialized with the data from the ASCII file."""
        out = {}
        for i in importers:
            out[i.metadata['detector_id']] = CountRate.from_importer(i)
        return cls(out, **kwargs)
