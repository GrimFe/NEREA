from collections.abc import Iterable
from typing import Self
from dataclasses import dataclass
import numpy as np
import pandas as pd
import linecache
import serpentTools as sts
from datetime import datetime, timedelta
import warnings
import logging

from .utils import ratio_v_u, _make_df, integral_v_u, product_v_u, sum_v_u, get_fit_R2

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
    dead_time_corrected: bool = False

    @property
    def period(self) -> pd.DataFrame:
        """
        Calculate the reactor period from a ReactionRate instance.
        
        Returns
        -------
        pd.DataFrame
            with reactor period value and uncertainty
        """
        from scipy.optimize import curve_fit
        def linear_fit(x, a, b):
            return x / a + b  # Linear fit function (a = T)

        # Curve fitting to find the reactor period (T)
        data = self.data[self.data.value != 0]
        if data.shape != self.data.shape:
            logging.warning("Removing 0 counts from Reaction Rate to enable period log fit. Removed %s rows.", self.data.shape[0] - data.shape[0])
        y = np.log(data.value)  # Log-transform the data to allow lineaer fit
        
        popt, pcov, out, _, _ = curve_fit(linear_fit,
                                          (data.Time - self.start_time).dt.seconds,  # x must be in seconds from 0
                                          y,
                                          full_output=True)

        period = _make_df(popt[0], np.sqrt(pcov[0, 0]) * popt[0])

        r2 = get_fit_R2(y, out['fvec'])
        logging.info("Reactor period fit R^2 = %s", r2)  # probably not functioning
        return period

    def average(self, start_time: datetime, duration: int) -> pd.DataFrame:
        """
        Calculate the average value and uncertainty of a time series data within a specified duration.

        Parameters
        ----------
        start_time : datetime
            The starting time for the data to be analyzed.
        duration : int
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
        end_time = start_time + timedelta(seconds=duration)
        series = self.data.query("Time > @start_time and Time <= @end_time")
        v, u = integral_v_u(series.value)
        relative = True if v != 0 else False
        return _make_df(v / duration * self.timebase, u / duration * self.timebase, relative)

    def moving_average(self, length: float) -> pd.DataFrame:
        """
        Calculates the power monitor moving average.

        Parameters
        ----------
        length : float
            the window span to average, in seconds.
        
        Returns
        -------
        pd.DataFrame
            with time and counts data.
        """
        if length < self.timebase:
            raise ValueError("Moving average window length should be larger than the Reaction Rate timebase.")
        elif length == self.timebase:
            out = self.data
        else:
            out = []
            time = self.start_time
            for i in range((self.data.Time.iloc[-1] - self.start_time).total_seconds() / length):
                v = self.average(time, length)
                out.append(pd.DataFrame({"Time": time, "value": v["value"].value}))
                time = time + timedelta(seconds=length)
            out = pd.concat(out, ignore_index=True)
        return out

    def integrate(self, timebase: int, start_time: datetime | None = None) -> pd.DataFrame:
        """
        Integrate data over a specified timebase starting from a given start time.

        Parameters
        ----------
        timebase : int
            The interval of time in seconds over which to calculate the average. This interval is used to group the data for averaging.
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
        v, u = integral_v_u(plateau.value)
        v, u = ratio_v_u(_make_df(v, u), normalization)
        return _make_df(v, u)

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
        v, u = unit_p.value / duration, unit_p.uncertainty / duration
        return _make_df(v, u)

    def dead_time_correction(self, tau_p: float = 88e-9, tau_np: float = 108e-9) -> Self:
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
        from scipy import optimize
        def dead_time_correction_m(n, m, tp, tnp): 
            # Equation for dead time correction
            return n / ((1 - tp / tnp) * n * tp + np.exp(tp * n)) - m
        if self.dead_time_corrected:
            warnings.warn("Dead time correction already applied to this detector.")
        pm = self.data.copy()
        pm["value"] = pm.value.apply(lambda x:
                                     optimize.newton(lambda n:
                                                     dead_time_correction_m(n, x, tau_p, tau_np),
                                                     x))
        return self.__class__(pm, self.start_time, self.campaign_id, self.experiment_id, self.detector_id,
                              self.deposit_id, self.timebase, dead_time_corrected=True)

    def get_reactivity(self, file) -> pd.DataFrame:
        """
        Calculates the reactor reactivity based on the Reaction Rate-estimated
        reactor period and on effective nuclear data computed by Serpent.
        
        Parameters
        ----------
        file : str
            path to the Serpent `res.m` output file to read effective delayed
            neutron data from.

        Returns
        -------
        pd.DataFrame
            with reactivity value and uncertainty
        """
        # get delayed data from Serpent simulation
        bi = sts.read(file).resdata['adjIfpImpBetaEff']
        li = sts.read(file).resdata['adjIfpImpLambda']
        bi = _make_df(bi[:, 0], bi[:, 1] * bi[:, 0])
        li = _make_df(li[:, 0], li[:, 1] * li[:, 0])

        # compute reactivity
        T = self.period
        rho = np.sum(bi.value / (1 + li.value * T.value))

        # variance portions
        VAR_FRAC_T = np.sum((-bi.value * li.value / (1 + li.value * T.value) * T.uncertainty) **2)
        VAR_FRAC_B = np.sum((1 / (1 + li.value * T.value) * bi.uncertainty) **2)
        VAR_FRAC_L = np.sum((-bi.value * T.value / (1 + li.value * T.value) * li.uncertainty) **2)
        
        return _make_df(rho, np.sqrt(VAR_FRAC_T + VAR_FRAC_B + VAR_FRAC_L)).assign(VAR_FRAC_T=VAR_FRAC_T,
                                                                                   VAR_FRAC_B=VAR_FRAC_B,
                                                                                   VAR_FRAC_L=VAR_FRAC_L)

    def get_asymptotic_counts(self, t_f: float = 3e-2, t_l: float = 1e-2, **kwargs) -> Self:
        """
        Cut the power monitor data based on specific conditions to find the
        asymptotic exponential (after all harmonics have decayed).
        
        Parameters
        ----------
        t_f : float
            threshold for the first significant change
        t_l : float
            threshold for the last significant change
        **kwargs
            arguments to pass to `self.moving_average`
        
        Returns
        -------
        self.__class__
            instance with truncated data
        """
        data = self.moving_average().value
        checker = np.log(data).diff().diff()
        if t_f > checker.max():
            raise ValueError(f'lower threshold should be lower than {checker.max():.2e}')
        max_ = data.idxmax()  # Time index of the maximum value
        last = data.loc[:max_].loc[checker.loc[:max_].abs() < t_l].iloc[-1].name
        first_1 = data.loc[:last].loc[checker.loc[:last].abs() > t_f].iloc[-1].name
        first = data.loc[first_1:].iloc[1].name
        out = data.loc[first:last]
        return self.__class__(out, out.Time.iloc[0], self.campaign_id,
                              self.experiment_id, self.detector_id, self.deposit_id,
                              kwargs['length'], self.dead_time_corrected)

    @classmethod
    def from_ascii(cls, file: str, detector: int, deposit_id: str):
        """
        Placeholder method to create a `ReactionRate` instance from an ASCII file.

        Parameters
        ----------
        file : str
            Path to the ASCII file.
        detector : int
            Detector number in the ASCII file.
        deposit_id : str
            Deposit of the detector.

        Returns
        -------
        ReactionRate
            A new `ReactionRate` instance.
        """
        start_time = datetime.strptime(linecache.getline(file, 1), "%d-%m-%Y %H:%M:%S\n")
        read = pd.read_csv(file, sep='\t', skiprows=[0,1], decimal=',')
        read["Time"] = read["Time"].apply(lambda x: start_time + timedelta(seconds=x))
        campaign_id, experiment_id = file.split('\\')[-1].split('_')
        out = cls(read[["Time", f"Det {detector}"]].rename(columns={f"Det {detector}": "value"}),
                  start_time=start_time,
                  campaign_id=campaign_id,
                  experiment_id=experiment_id,
                  detector_id=f"Det {detector}",
                  deposit_id=deposit_id,
                  timebase=(read['Time'][1] - read['Time'][0]).seconds)
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
                                                    pd.Grouper(key="Time", freq=f'{timebase}s', closed='right')
                                                    ).agg('sum').reset_index()
        ref['uncertainty'] = np.sqrt(ref['value'])  # absolute
        ref['value'] = ref['value'] / timebase
        for monitor in list(self.detectors.values())[1:]:
            compute = monitor.data.groupby(pd.Grouper(key="Time", freq=f'{timebase}s', closed='right')
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
                raise Exception(f"Power monitor {monitor.detector_id} inconsistent with {list(self.detectors.values())[0].detector_id}")

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
    def from_ascii(cls, file: str, detectors: Iterable[int], deposit_ids: Iterable[str]):
        """
        Creates an instance of ReactionRate using data extracted from an ASCII file.

        The ASCII file should contain columns of data including timestamps and power readings.

        The filename is supposed to be formatted as:
        {Campaign}_{experiment}.txt

        Parameters
        ----------
        file : str
            Path to the ASCII file containing the power monitor data.
        detectors : Iterable[int]
            Detector numbers to read from the ASCII file.
        deposit_ids : Iterable[str]
            Ordered deposits of the detectors. 

        Returns
        -------
        ReactionRate
            An instance of the ReactionRate class initialized with the data from the ASCII file.

        Example
        -------
        Consider an ASCII file `power_data.txt` with the following content:

        ```
        timestamp,power
        2023-01-01 00:00:00,100
        2023-01-01 01:00:00,150
        2023-01-01 02:00:00,200
        ```

        You can create a ReactionRate instance from this file as follows:

        ```python
        from nerea.ReactionRate import ReactionRate

        power_monitor = ReactionRate.from_ascii('path/to/power_data.txt', experiment_id='EXP123')
        print(power_monitor.data)
        ```

        This will output:

        ```
                    timestamp  power
        0 2023-01-01 00:00:00    100
        1 2023-01-01 01:00:00    150
        2 2023-01-01 02:00:00    200
        ```

        Note
        ----
        Ensure that the file path provided is correct and that the file format matches the expected structure.
        """
        out = {}
        for i, d in enumerate(detectors):
            out[d] = ReactionRate.from_ascii(file, d, deposit_id=deposit_ids[i])
        return cls(out)
