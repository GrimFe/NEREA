from collections.abc import Iterable
from typing import Self
from dataclasses import dataclass
import numpy as np
import pandas as pd
import linecache
import serpentTools as sts
from datetime import datetime, timedelta
import warnings
import matplotlib.pyplot as plt

from .utils import ratio_v_u, _make_df, time_integral_v_u, integral_v_u
from .functions import get_fit_R2, smoothing
from .defaults import *
from .classes import EffectiveDelayedParams
from .constants import BASE_DATE

__all__ = [
    "ReactionRate",
    "ReactionRates"]

@dataclass(slots=True)
class ReactionRate:
    data: pd.DataFrame
    start_time: datetime
    campaign_id: str
    experiment_id: str
    detector_id: str
    deposit_id: str
    timebase: int = 1 ## in seconds
    _dead_time_corrected: bool = False

    @property
    def period(self) -> pd.DataFrame:
            """
            Calculate the reactor period from a ReactionRate instance.
            
            Returns
            -------
            pd.DataFrame
                with reactor period value and uncertainty
            """
            # Curve fitting to find the reactor period (T)
            fitted_data, popt, pcov, out = self._linear_fit()
            period = _make_df(popt[0], np.sqrt(pcov[0, 0]))
            r2 = get_fit_R2(fitted_data, out['fvec'])
            warnings.warn(f"Reactor period fit R^2 = {r2}")
            return period

    def _linear_fit(self, preprocessing: str='log', nonzero: bool=True):
        """
        Linearly fits monitor data after preprocessing.

        Parameters:
        -----------
        preprocessing : str, optional
            numpy function to apply to self.data prior to linear fitting.
            Default is 'log'.
        nonzero : bool, optional
            queries non-zero values in self.data. Default is True.
        """
        from scipy.optimize import curve_fit
        def linear_fit(x, a, b):
            return x / a + b  # Linear fit function (a = T)
        if nonzero:
            data = self.data[self.data.value != 0]
            if data.shape != self.data.shape:
                warnings.warn("Removing 0 counts from Reaction Rate to enable period log fit. Removed %s rows." % (self.data.shape[0] - data.shape[0]))
        if preprocessing is not None:
            y = getattr(np, preprocessing)(data.value)  # apply preprocessing
        else:
            y = data.value
        popt, pcov, out, _, _ = curve_fit(linear_fit,
                                          (data.Time - self.start_time).dt.seconds,  # x must be in seconds from 0
                                          y,
                                          full_output=True,
                                          absolute_sigma=True)
        return y, popt, pcov, out

    @property
    def period(self) -> pd.DataFrame:
            """
            Calculate the reactor period from a ReactionRate instance.
            
            Returns
            -------
            pd.DataFrame
                with reactor period value and uncertainty
            """
            # Curve fitting to find the reactor period (T)
            fitted_data, popt, pcov, out = self._linear_fit()
            period = _make_df(popt[0], np.sqrt(pcov[0, 0]))
            r2 = get_fit_R2(fitted_data, out['fvec'])
            warnings.warn(f"Reactor period fit R^2 = {r2}")
            return period

    def _linear_fit(self, preprocessing: str='log', nonzero: bool=True):
        """
        Linearly fits monitor data after preprocessing.

        Parameters:
        -----------
        preprocessing : str, optional
            numpy function to apply to self.data prior to linear fitting.
            Default is 'log'.
        nonzero : bool, optional
            queries non-zero values in self.data. Default is True.
        """
        from scipy.optimize import curve_fit
        def linear_fit(x, a, b):
            return x / a + b  # Linear fit function (a = T)
        if nonzero:
            data = self.data[self.data.value != 0]
            if data.shape != self.data.shape:
                warnings.warn("Removing 0 counts from Reaction Rate to enable period log fit. Removed %s rows." % (self.data.shape[0] - data.shape[0]))
        if preprocessing is not None:
            y = getattr(np, preprocessing)(data.value)  # apply preprocessing
        else:
            y = data.value
        popt, pcov, out, _, _ = curve_fit(linear_fit,
                                          (data.Time - self.start_time).dt.seconds,  # x must be in seconds from 0
                                          y,
                                          full_output=True,
                                          absolute_sigma=True)
        return y, popt, pcov, out

    def average(self, start_time: datetime, duration: float) -> pd.DataFrame:
        """
        Calculate the average value and uncertainty of a time series data within a specified duration.

        Parameters
        ----------
        start_time : datetime
            The starting time for the data to be analyzed.
        duration : float
            The length of time in seconds for which the average is calculated.

        Returns
        -------
        pd.DataFrame
            DataFrame containing:
            - 'value': The average value of the data within the specified time range.
            - 'uncertainty': The uncertainty value, calculated as 1/sqrt(N).

        Examples
        --------
        >>> from datetime import datetime
        >>> data = pd.DataFrame({'Time': pd.date_range('2021-01-01', periods=100, freq='S'),
                                 'value': np.random.rand(100)})
        >>> pm = ReactionRate(data=data, start_time=datetime(2021, 1, 1), 
                              campaign_id='C1', experiment_id='E1', detector_id='M1')
        >>> avg_df = pm.average(datetime(2021, 1, 1, 0, 0, 30), 10)
        >>> print(avg_df)
        """
        # end_time should be 1 timebase after the real end time to use
        end_time = start_time + timedelta(seconds=duration + self.timebase)
        series = self.data.query("Time >= @start_time and Time < @end_time")
        v, u = time_integral_v_u(series)
        relative = True if v != 0 else False
        # Time Normalization of the Average
        # If the RR time binning is not 1 s, there is a chance the query truncated
        # the time series so that the difference between first and last time in
        # `series` are closer than duration. Hence I define delta.
        # iloc[-1] explained by the use of time_integral_v_u(): we stop at the 
        # beginning of the next step-post time stamp.
        delta = (series.Time.iloc[-1] - series.Time.iloc[0]).total_seconds()
        return _make_df(v / delta, u / delta, relative)

    def smooth(self, **kwargs) -> Self:
        """
        Calculates the reaction rate moving average.

        Parameters
        ----------
        **kwargs :
            arguments to nerea.utils.smoothing()
            - method
            - arguments to the metod
        
        Returns
        -------
        pd.DataFrame
            with time and counts data.
        
        Notes
        -----
        Allowed methods are:
            - "moving average"
            - "savgol_filter"
        """
        if kwargs.get('window') or kwargs.get('window_lenght'):
            w = kwargs['window'] if kwargs['smoothing_method'] == 'moving_average' else kwargs['window_length']
            if w < self.timebase:  ## if nor window nor window_length are passed w is False
                raise ValueError("Smoothing window length should be larger than the Reaction Rate timebase.")
        else:
            out = pd.DataFrame({"Time": self.data["Time"],
                                "value": smoothing(self.data["value"], **kwargs)})
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
        Integrate data over a specified timebase starting from a given start time.

        Parameters
        ----------
        timebase : int
            The interval of time in seconds over which to calculate the average.
            This interval is used to group the data for averaging.
        start_time : datetime, optional
            The starting time for the integration process. Defaults to `self.start_time`.

        Returns
        -------
        pd.DataFrame
            DataFrame containing:
            - 'value': The average value of the data within the specified time range.
            - 'uncertainty': The uncertainty value, calculated as 1/sqrt(N).

        Examples
        --------
        >>> from datetime import datetime
        >>> data = pd.DataFrame({'Time': pd.date_range('2021-01-01', periods=100, freq='S'),
                                 'value': np.random.rand(100)})
        >>> pm = ReactionRate(data=data, start_time=datetime(2021, 1, 1), 
                              campaign_id='C1', experiment_id='E1', detector_id='M1')
        >>> integrated_df = pm.integrate(10)
        >>> print(integrated_df)
        """
        start_time_ = self.start_time if start_time is None else start_time
        out = []
        while start_time_ < self.data.Time.max():
            out.append(self.average(start_time_, timebase))
            start_time_ = start_time_ + timedelta(seconds=timebase)
        return pd.concat(out, ignore_index=True)

    def plateau(self, sigma: int=2, timebase: int=10) -> pd.DataFrame:
        """
        The plateau with the largest integral counts in the detector.

        Parameters
        ----------
        sigma : int, optional
            the amount of standard deviations to consider for the
            uncertainty on the plateau.
            Defaults to 2.
        timebase : int, optional
            the time base for integration in plateau search in seconds.
            Defaults to 10 s.
        
        Returns
        -------
        pd.DataFrame
            with a `Time` and a `value` column.
        """
        time = self.data.Time.min()
        plateau_start_time, plateau_end_time = self.data.Time.min(), self.data.Time.min()
        sum, max = 1, 0
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

    def per_unit_power(self, monitor, *args, **kwargs) -> pd.DataFrame:
        """
        Normalizes the raction rate to a power monitor.

        Parameters
        ----------
        monitor : ReactionRate
            The power monitor for the reaction rate normalization.
        *args : Any
            Positional arguments to be passed to the `ReactionRate.plateau()` method.
        **kwargs : Any
            Keyword arguments to be passed to the `ReactionRate.plateau()` method.
        
        Returns
        -------
        pd.DataFrame
            with value and uncertainty of the normalized reaction rate
            integrated over time.
        """
        plateau = self.plateau(*args, **kwargs)
        duration = (plateau.Time.max() - plateau.Time.min()).seconds
        normalization = monitor.average(plateau.Time.min(), duration) 
        return _make_df(*ratio_v_u(_make_df(*integral_v_u(plateau.value)), normalization))

    def per_unit_time_power(self, monitor, *args, **kwargs) -> pd.DataFrame:
        """
        Normalizes the raction rate to a power monitor and gives the conunt rate
        per unit power.

        Parameters
        ----------
        monitor : ReactionRate
            The power monitor for the reaction rate normalization.
        *args : Any
            Positional arguments to be passed to the `ReactionRate.plateau()` method.
        **kwargs : Any
            Keyword arguments to be passed to the `ReactionRate.plateau()` method.
        
        Returns
        -------
        pd.DataFrame
            with value and uncertainty of the normalized reaction rate
            averaged over time.
        """
        plateau = self.plateau(*args, **kwargs)
        duration = (plateau.Time.max() - plateau.Time.min()).seconds
        unit_p = self.per_unit_power(monitor, *args, **kwargs)
        return _make_df(unit_p.value / duration, unit_p.uncertainty / duration)

    def dead_time_corrected(self, tau_p: float = 88e-9, tau_np: float = 108e-9) -> Self:
        """
        Apply dead time correction to the reaction rate data.
        
        Parameters
        ----------
        tau_p : float, optional
            prompt dead time constant.
            Defaults to 88e-9.
        tau_np : float, optional
            non-prompt dead time constant.
            Defaults to 108e-9.
        
        Returns
        -------
        self.__class__
            instance with corrected data
        """
        if self._dead_time_corrected:
            pm = self.data.copy()
        else:
            from scipy import optimize
            def dead_time_correction(n, m, tp, tnp): 
                # Equation for dead time correction
                return n / ((1 - tp / tnp) * n * tp + np.exp(tp * n)) - m
            if self._dead_time_corrected:
                warnings.warn("Dead time correction already applied to this detector.")
            pm = self.data.copy()
            pm["value"] = pm.value.apply(lambda x:
                                        optimize.newton(lambda n:
                                                        dead_time_correction(n, x, tau_p, tau_np),
                                                        x))
        return self.__class__(pm,
                              self.start_time,
                              self.campaign_id,
                              self.experiment_id,
                              self.detector_id,
                              self.deposit_id,
                              self.timebase,
                              _dead_time_corrected=True)

    def get_reactivity(self, delayed_data: EffectiveDelayedParams) -> pd.DataFrame:
        """
        Calculates the reactor reactivity based on the Reaction Rate-estimated
        reactor period and on effective nuclear data computed by Serpent.
        
        Parameters
        ----------
        delayed_data_file : nerea.EffectiveDelayedParams
            effective delayed neutron paramters to use for reactivity calculation

        Returns
        -------
        pd.DataFrame
            with reactivity value and uncertainty
        """
        bi = delayed_data.beta_i
        li = delayed_data.lambda_i

        # compute reactivity
        T = self.period
        rho = np.sum(bi.value / (1 + li.value * T.value))

        # variance portions
        VAR_PORT_T = np.sum((-bi.value * li.value / (1 + li.value * T.value)**2 * T.uncertainty) **2)
        VAR_PORT_B = np.sum((1 / (1 + li.value * T.value) * bi.uncertainty) **2)
        VAR_PORT_L = np.sum((-bi.value * T.value / (1 + li.value * T.value)**2 * li.uncertainty) **2)
        return _make_df(rho, np.sqrt(VAR_PORT_T + VAR_PORT_B + VAR_PORT_L)).assign(VAR_PORT_T=VAR_PORT_T,
                                                                                   VAR_PORT_B=VAR_PORT_B,
                                                                                   VAR_PORT_L=VAR_PORT_L)

    def get_asymptotic_counts(self, t_left: float=3e-2, t_right: float=1e-2, smooth_kwargs: dict={}, dtc_kwargs: dict={}) -> Self:
        """
        Cut the power monitor data based on specific conditions to find the
        asymptotic exponential (after all harmonics have decayed).
        
        Parameters
        ----------
        t_left : float, optional
            tolerance to find the beginning of the asymptotic
            exponential counts. Default is 3e-2.
        t_right : float, optional
            tolerance to find the end of the asymptotic
            exponential counts. Default is 1e-2.
        smooth_kwargs: dict, optional
            arguments to pass to `self.smooth`.
            Default is `{}`.
        dtc_kwargs : dict, optional
            arguments to pass to `self.dead_time_corrected`.
            Default is `{}`.
            
        Returns
        -------
        self.__class__
            instance with truncated data

        Notes
        -----
        Inherently uses dead time corrected counts
        Inherently uses smoothed data to ease the search
        """
        smt_kw = DEFAULT_SMOOTH_KWARGS | smooth_kwargs
        if not self._dead_time_corrected:
            dtc_kw = DEFAULT_DTC_KWARGS | dtc_kwargs
            data = self.dead_time_corrected(**dtc_kw).smooth(**smt_kw).data
        else:
            data = self.smooth(**smt_kw).data
            if dtc_kwargs is not None:
                warnings.warn("Dead time corection already applied, ignoring kwargs.")
        log_double_derivative = np.log(data.value).diff().diff()
        max_ = data.value.idxmax()
        ## Right discrimination
        last = data.loc[:max_].loc[log_double_derivative.loc[:max_].abs() < t_right]
        if last.shape[0] == 0:
            warnings.warn(f"Right bound not found with t_right = {t_right}. Using end point.")
            last = data.loc[max_].to_frame().T.iloc[-1].name
        else:
            last = last.iloc[-1].name
        ## Left discrimination
        first = data.loc[:last].loc[log_double_derivative.loc[:last].abs() > t_left]
        if first.shape[0] == 0:
            warnings.warn(f"Left bound not found with t_left = {t_left}. Using start point.")
            first = data.loc[0].to_frame().T.iloc[-1].name
        else:
            first = first.iloc[-1].name
        return self.__class__(self.data[first:last],
                              start_time=self.data[first:last].Time.min(),
                              campaign_id=self.campaign_id,
                              experiment_id=self.experiment_id,
                              detector_id=self.detector_id,
                              deposit_id=self.deposit_id,
                              timebase=self.timebase,
                              _dead_time_corrected=self._dead_time_corrected
                              )

    def plot(self,
             start_time: datetime=None,
             duration: int=None,
             ax: plt.Axes=None,
             c: str='k') -> plt.Axes:
        """
        Plot data in ReactionRate.

        Parameters
        ----------
        start_time : datetime, optional
            The time the Reaction rate is considered from.
            Default is None for first acquisition time.
        duration : int, optional
            The time-span the Reaction rate is considered for.
            Default is None for until last acquisition time.
        ax : plt.Axes, optional
            The ax where the plot is drawn.
            Defauls is None for a new axes.
        c : str, optional
            The color of the plotted seriese.
            Default is `'k'`.
        
        Returns
        -------
        plt.Axes
        """
        start_time_ = start_time if start_time is not None else self.start_time
        duration_ = duration if duration is not None else (
            self.data.Time.max() - start_time_).total_seconds()
        ax = self.data.plot(x="Time", y='value', ax=ax, color=c)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.axvspan(self.start_time, start_time_, alpha=0.5, color='gray')
        ax.axvspan(start_time_ + timedelta(seconds=duration_),
                   self.data.Time.max(), alpha=0.5, color='gray',
                   label='Ingored')
        ax.set_ylabel("Power monitor count rate [1/s]")
        ax.tick_params(axis='y', left=False, labelleft=False, right=True, labelright=True)
        ax.yaxis.set_label_position("right")
        t = ax.yaxis.get_offset_text()
        t.set_x(1.01)
        return ax
    
    @classmethod
    def _from_formatted_ads(cls, file: str, **kwargs):
        """
        Method to create a `ReactionRate` instance
        from an ASCII file generated by ADS DAQ.

        Parameters
        ----------
        file : str
            Path to the ASCII file.
        kwargs to the class creation
            detector : int
                Detector number in the ASCII file.
            deposit_id : str
                Deposit of the detector.

        Returns
        -------
        ReactionRate
            A new `ReactionRate` instance.

        Note
        ----
        Infers `campaign_id` and `experiment_id`
        from file name.

        File format: `'CAMP_EXP.ads'`
        """
        detector = kwargs.pop('detector', None)
        if detector is None:
            raise ValueError("`'detector'` kwarg required to read ReactionRate.")
        start_time = datetime.strptime(linecache.getline(file, 1), "%d-%m-%Y %H:%M:%S\n")
        read = pd.read_csv(file, sep='\t', skiprows=[0,1], decimal=',')
        read["Time"] = read["Time"].apply(lambda x: start_time + timedelta(seconds=x))
        metadata = file.split('\\')[-1].split('.')[0]
        campaign_id, experiment_id = metadata.split('_')
        return cls(read[["Time", f"Det {detector}"]].rename(columns={f"Det {detector}": "value"}),
                   start_time=start_time,
                   campaign_id=campaign_id,
                   experiment_id=experiment_id,
                   detector_id=f"Det {detector}",
                   timebase=(read['Time'][1] - read['Time'][0]).total_seconds(),
                   **kwargs)
    
    @classmethod
    def _from_phspa(cls, file: str, **kwargs):
        """
        Method to create a `ReactionRate` instance
        from an ASCII file generated by PHSPA DAQ.

        Parameters
        ----------
        file : str
            Path to the ASCII file.
        kwargs to the class creation
            detector : int
                Detector number in the ASCII file.
            deposit_id : str
                Deposit of the detector.
            campaign_id : str
                ID of experimental campaign
            experiment_id : str
                ID of experiment in the campaign

        Returns
        -------
        ReactionRate
            A new `ReactionRate` instance.
        """
        detector = kwargs.pop('detector', None)
        if detector is None:
            raise ValueError("`'detector'` kwarg required to read ReactionRate.")
        data = pd.read_csv(file, sep="\t", skiprows=18, decimal=',').iloc[:,:-1]
        data.columns = ["Time", "value"]
        data.Time = data.Time.apply(lambda x: BASE_DATE + timedelta(days=x))
        warnings.warn("Average timebase considered for PHSPA acquisitions.")
        timebase = data.Time.diff().dt.total_seconds().mean()
        return cls(data,
                   start_time=data.Time.min(),
                   detector_id=detector,
                   timebase=timebase,
                   **kwargs)
    
    @classmethod
    def _from_formatted_phspa(cls, file: str, **kwargs):
        """
        Method to create a `ReactionRate` instance
        from an ASCII file generated by PHSPA DAQ.

        Parameters
        ----------
        file : str
            Path to the ASCII file.
        kwargs to the class creation
            deposit_id : str
                Deposit of the detector.
            campaign_id : str
                ID of experimental campaign
            experiment_id : str
                ID of experiment in the campaign

        Returns
        -------
        ReactionRate
            A new `ReactionRate` instance.

        Note
        ----
        Infers `detector_id`, `campaign_id` and
        `experiment_id` from file name.
        
        File format: `'CAMP_EXP_DET.log'`
        """
        metadata = file.split('\\')[-1].split('.')[0]
        campaign_id, experiment_id, det = metadata.split('_')
        return cls._from_phspa(file,
                               detector=det,
                               campaign_id=campaign_id,
                               experiment_id=experiment_id,
                               **kwargs)

    @classmethod
    def from_ascii(cls, file: str, filetype: str='infer', **kwargs):
        """
        Method to create a `ReactionRate` instance
        from an ASCII file.

        Parameters
        ----------
        file : str
            Path to the ASCII file.
        filetype : str, optional
            Type of ASCII file to process.
            Default is `'infer'` to infer it from
            file extension.
        kwargs to the class creation
            detector : int
                Detector number in the ASCII file.
            deposit_id : str
                Deposit of the detector.
            campaign_id : str
                ID of experimental campaign
            experiment_id : str
                ID of experiment in the campaign

        Returns
        -------
        ReactionRate
            A new `ReactionRate` instance.
        """
        ft = file.split('.')[-1] if filetype == 'infer' else filetype
        match ft:
            case 'ads':
                out = cls._from_formatted_ads(file, **kwargs)
            case 'phspa':
                out = cls._from_phspa(file, **kwargs)
            case 'log':
                out = cls._from_formatted_phspa(file, **kwargs)
            case _:
                raise ValueError("ASCII file type processing not implemented")
        return out


@dataclass(slots=True)
class ReactionRates:
    detectors: dict[int, ReactionRate]

    def __post_init__(self) -> None:
        self._check_consistency()

    def _check_consistency(self, time_tolerance: timedelta=timedelta(seconds=60),
                           timebase: int=100, sigma=1) -> None:
        """
        Check the consistency of time and curve data with specified parameters.

        Parameters
        ----------
        time_tolerance : timedelta, optional
            Parameter for `self._check_time_consistency`. Defaults to 60 seconds.
        timebase : int, optional
            Parameter for `self._check_curve_consistency`. Defaults to 100.
        sigma : int, optional
            Parameter for `self._check_curve_consistency`. Defaults to 1.

        Examples
        --------
        >>> data = pd.DataFrame({'Time': pd.date_range('2021-01-01', periods=100, freq='S'),
                                 'value': np.random.rand(100)})
        >>> pm1 = ReactionRate(data=data, start_time=datetime(2021, 1, 1), 
                               campaign_id='C1', experiment_id='E1', detector_id='M1')
        >>> pm2 = ReactionRate(data=data, start_time=datetime(2021, 1, 1), 
                               campaign_id='C1', experiment_id='E1', detector_id='M2')
        >>> pms = ReactionRates(detectors={1: pm1, 2: pm2})
        >>> pms._check_consistency()
        """
        must = ['campaign_id', 'experiment_id']
        for attr in must:
            if not all([getattr(m, attr) == getattr(list(self.detectors.values())[0], attr)
                        for m in self.detectors.values()]):
                raise Exception(f"Inconsistent {attr} among different ReactionRate instances.")
        should = ['deposit_id']
        for attr in should:
            if not all([getattr(m, attr) == getattr(list(self.detectors.values())[0], attr)
                        for m in self.detectors.values()]):
                warnings.warn(f"Inconsistent {attr} among different ReactionRate instances.")
        self._check_time_consistency(time_tolerance)
        self._check_curve_consistency(timebase, sigma)

    def _check_time_consistency(self, time_tolerance: timedelta) -> None:
        """
        Check if the start times of power detectors are consistent within a given time tolerance.

        Parameters
        ----------
        time_tolerance : timedelta
            The maximum allowable difference in time between the start times of the power detectors
            in `self.detectors`.

        Examples
        --------
        >>> data = pd.DataFrame({'Time': pd.date_range('2021-01-01', periods=100, freq='S'),
                                 'value': np.random.rand(100)})
        >>> pm1 = ReactionRate(data=data, start_time=datetime(2021, 1, 1), 
                               campaign_id='C1', experiment_id='E1', detector_id='M1')
        >>> pm2 = ReactionRate(data=data, start_time=datetime(2021, 1, 1), 
                               campaign_id='C1', experiment_id='E1', detector_id='M2')
        >>> pms = ReactionRates(detectors={1: pm1, 2: pm2})
        >>> pms._check_time_consistency(timedelta(seconds=60))
        """
        ref = list(self.detectors.values())[0].start_time
        for monitor in self.detectors.values():
            if not abs(monitor.start_time - ref) < time_tolerance:
                warnings.warn(f"Power monitor start time difference > {time_tolerance}")

    def _check_curve_consistency(self, timebase: int, sigma: int=1) -> None:
        """
        Compare data from multiple detectors to check for consistency within a sigma-uncertainty tolerance,
        based on a specified timebase.

        Parameters
        ----------
        timebase : int
            The time interval in seconds for grouping the data. This parameter determines how the data are
            aggregated and compared between different detectors.
        sigma : int, optional
            The uncertainty associated with the measurements. It is used to calculate the tolerance for
            checking the consistency between different power detectors. The tolerance is computed as the
            average uncertainty on the ratio of values between two detectors. Defaults to 1.

        Examples
        --------
        >>> data = pd.DataFrame({'Time': pd.date_range('2021-01-01', periods=100, freq='S'),
                                 'value': np.random.rand(100)})
        >>> pm1 = ReactionRate(data=data, start_time=datetime(2021, 1, 1), 
                               campaign_id='C1', experiment_id='E1', detector_id='M1')
        >>> pm2 = ReactionRate(data=data, start_time=datetime(2021, 1, 1), 
                               campaign_id='C1', experiment_id='E1', detector_id='M2')
        >>> pms = ReactionRates(detectors={1: pm1, 2: pm2})
        >>> pms._check_curve_consistency(10, 1)
        """
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
    def _first(self):
        return list(self.detectors.values())[0]

    @property
    def campaign_id(self):
        return self._first.campaign_id

    @property
    def experiment_id(self):
        return self._first.experiment_id

    @property
    def deposit_id(self):
        """
        The deposit id of the first element of `self.detectors`.
        """
        return self._first.deposit_id

    @property
    def best(self) -> ReactionRate:
        """
        Returns the power monitor with the highest sum value.

        Returns
        -------
        ReactionRate
            Power monitor with the highest integral count.

        Examples
        --------
        >>> pm = ReactionRate(...)
        >>> best_pm = pm.best
        """
        max = list(self.detectors.values())[0].data.value.sum()
        out = list(self.detectors.values())[0]
        for monitor in list(self.detectors.values())[1:]:
            if monitor.data.value.sum() > max:
                out = monitor
        return out

    def per_unit_power(self, monitor: int, *args, **kwargs) -> dict[int, pd.DataFrame]:
        """
        Normalizes the raction rate to a power monitor.

        Parameters
        ----------
        monitor : int
            The ID of the reaction rate to be used as power
            monitor for the reaction rate normalization.
        *args : Any
            Positional arguments to be passed to the `ReactionRate.plateau()` method.
        **kwargs : Any
            Keyword arguments to be passed to the `ReactionRate.plateau()` method.

        Returns
        -------
        dict[int, pd.DataFrame]
            with value and uncertainty of the normalized reaction rate
            integrated over time. Keys are the detector IDs as in
            self.detectors.
        """
        out = {}
        for key, detector in self.detectors.items():
            if key != monitor:
                out[key] = detector.per_unit_power(self.detectors[monitor],
                                                   *args, **kwargs)
        return out

    def per_unit_time_power(self, monitor: int, *args, **kwargs) -> dict[int, pd.DataFrame]:
        """
        Normalizes the raction rate to a power monitor and takes the average over time.

        Parameters
        ----------
        monitor : int
            The ID of the reaction rate to be used as power
            monitor for the reaction rate normalization.
        *args : Any
            Positional arguments to be passed to the `ReactionRate.plateau()` method.
        **kwargs : Any
            Keyword arguments to be passed to the `ReactionRate.plateau()` method.

        Returns
        -------
        dict[int, pd.DataFrame]
            with value and uncertainty of the normalized reaction rate
            averaged over time. Keys are the detector IDs as in
            self.detectors.
        """
        out = {}
        for key, detector in self.detectors.items():
            if key != monitor:
                out[key] = detector.per_unit_time_power(self.detectors[monitor],
                                                        *args, **kwargs)
        return out

    @classmethod
    def from_ascii(cls,
                   files: dict[str, tuple[Iterable[str]|Iterable[int]|None, Iterable[str]]],
                   filetypes: Iterable[str]='infer'):
        """
        Creates an instance of ReactionRate using data extracted from an ASCII file.

        The ASCII file should contain columns of data including timestamps and power readings.

        The filename is supposed to be formatted as:
        {Campaign}_{experiment} (ADS) or
        {Campaign}_{experiment}_{detector} (PHSPA)

        Parameters
        ----------
        files : dict[str, tuple[Iterable[str]|Iterable[int]|None, Iterable[str]]]
            Maps each file to the detectors to read there and
            the corresponding deposit id.
            - key: str
                Path to the ASCII files containing the power monitor data.
            -values: tuple
                first: Iterable[str]|Iterable[int]
                    detector ids for ADS files
                    or None
                    for PHSPA file (detector id inferred from filename)
                second: Iterable[str]
                    deposit ids
        filetype : Iterable[str], optional
            Type of ASCII file to process.
            Default is `'infer'` to infer it from
            file extension for each file.

        Returns
        -------
        ReactionRate
            An instance of the ReactionRate class initialized
            with the data from the ASCII file.

        Note
        ----
        Allowed only for formatted source files.

        ADS files requires detectors to be passed as an iterable
        in the same order as the ADS processed files.
        """
        ft = ['infer'] * len(files) if filetypes == 'infer' else filetypes
        out = {}
        for i, (f, (dets, deps)) in enumerate(files.items()):
            ft_ = f.split('.')[-1] if ft[i] == 'infer' else ft
            match ft_:
                case 'ads':
                    for d, d_ in zip(dets, deps):
                        out[d] = ReactionRate.from_ascii(f,
                                                        filetype=ft_,
                                                        detector=d,
                                                        deposit_id=d_)
                case 'phspa':
                    d = f.split('\\')[-1].split('.')[0].split('_')[-1]
                    d_ = deps[0]
                    out[d] = ReactionRate.from_ascii(f,
                                                     filetype=ft_,
                                                     detector=d,
                                                     deposit_id=d_)
                case 'log':
                    d = f.split('\\')[-1].split('.')[0].split('_')[-1]
                    d_ = deps[0]
                    out[d] = ReactionRate.from_ascii(f,
                                                     filetype=ft_,
                                                     deposit_id=d_)
        return cls(out)


