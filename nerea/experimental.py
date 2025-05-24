import serpentTools as sts  ## impurity correction
from collections.abc import Iterable
from dataclasses import dataclass
from .fission_fragment_spectrum import FissionFragmentSpectrum
from .effective_mass import EffectiveMass
from .reaction_rate import ReactionRate, ReactionRates
from .utils import ratio_v_u, product_v_u, _make_df
from .constants import ATOMIC_MASS
from . defaults import *

import pandas as pd
import numpy as np
import warnings

__all__ = ['_Experimental',
           'NormalizedFissionFragmentSpectrum',
           'SpectralIndex',
           'Traverse']


def average_v_u(df):
    v = df.value.mean()
    u = sum(df.uncertainty **2) / len(df.uncertainty)
    return v, u

@dataclass(slots=True)
class _Experimental:
    def process(self) -> None:
        """
        Placeholder for inheriting classes.
        """
        return None


@dataclass(slots=True)
class NormalizedFissionFragmentSpectrum(_Experimental):
    fission_fragment_spectrum: FissionFragmentSpectrum
    effective_mass: EffectiveMass
    power_monitor: ReactionRate

    def __post_init__(self) -> None:
        self._check_consistency()

    def _check_consistency(self) -> None:
        """
        Checks the consistency of:
            - experiment_id
            - detector_id
            - deposit_id
        among self.fission_fragment_spectrum
        and also checks:
            - R channel
        among self.fission_fragment_spectrum and effective_mass
        via _check_ch_equality(tolerance=0.01).

        Raises
        ------
        Exception
            If there are inconsistencies among the IDs or R channel values.
        """
        if not self.fission_fragment_spectrum.detector_id == self.effective_mass.detector_id:
            raise Exception('Inconsistent detectors among FissionFragmentSpectrum ans EffectiveMass')
        if not self.fission_fragment_spectrum.deposit_id == self.effective_mass.deposit_id:
            raise Exception('Inconsistent deposits among FissionFragmentSpectrum and EffectiveMass')
        if not self.fission_fragment_spectrum.experiment_id == self.power_monitor.experiment_id:
            raise Exception('Inconsitent experiments among FissionFragmentSpectrum and ReactionRate')
        if not self._check_ch_equality():
            ch = self.fission_fragment_spectrum.get_R(bin_kwargs={'bins': self.effective_mass.bins}).channel
            msg = f"R channel difference: {((ch - self.effective_mass.R_channel) / self.effective_mass.R_channel * 100).iloc[0]} %"
            warnings.warn(msg)

    def _check_ch_equality(self, tolerance:float =0.01) -> bool:
        """
        Checks consistency of the R channels of `self.fission_fragment_spectrum` and
        `self.effective_mass` within a specified tolerance.
        The check happens only if the binning of the two objects is the same.
        
        Parameters
        ----------
        tolerance : float, optional
            The acceptable relative difference between the R channel of
            `self.fission_fragment_spectrum` and `self.effective_mass`.
            Defaults to 0.01.

        Returns
        -------
        bool
            Indicating whether the relative difference between the R channels is within tolerance.
        """
        if self.fission_fragment_spectrum.data.channel.max() == self.effective_mass.bins:
            check = abs(self.fission_fragment_spectrum.get_R(
                            bin_kwargs={'bins': self.effective_mass.bins}
                            ).channel.iloc[0] - self.effective_mass.R_channel
                        ) / self.effective_mass.R_channel < tolerance
        else:
            check = True
        return check 

    @property
    def measurement_id(self) -> str:
        """
        The measurement ID associated with the fission fragment spectrum.

        Returns
        -------
        str
            The measurement ID attribute of the associated `FissionFragmentSpectrum`.
        """
        return self.fission_fragment_spectrum.measurement_id
    
    @property
    def campaign_id(self) -> str:
        """
        The campaign ID associated with the fission fragment spectrum.

        Returns
        -------
        str
            The campaign ID attribute of the associated `FissionFragmentSpectrum`.
        """
        return self.fission_fragment_spectrum.campaign_id
    
    @property
    def experiment_id(self) -> str:
        """
        The experiment ID associated with the fission fragment spectrum.

        Returns
        -------
        str
            The experiment ID attribute of the associated `FissionFragmentSpectrum`.
        """
        return self.fission_fragment_spectrum.experiment_id
    
    @property
    def location_id(self) -> str:
        """
        The location ID associated with the fission fragment spectrum.

        Returns
        -------
        str
            The location ID attribute of the associated `FissionFragmentSpectrum`.
        """
        return self.fission_fragment_spectrum.location_id

    @property
    def deposit_id(self):
        """
        The deposit ID associated with the fission fragment spectrum.

        Returns
        -------
        str
            The deposit ID attribute of the associated `FissionFragmentSpectrum`.
        """
        return self.fission_fragment_spectrum.deposit_id

    @property
    def _time_normalization(self) -> pd.DataFrame:
        """
        The time normalization and correction to be multiplied by the
        fission fragment spectrum per unit mass.

        Returns
        -------
        pd.DataFrame
            with normalization value and uncertainty
        """
        l = self.fission_fragment_spectrum.life_time
        v = 1 / l
        u = np.sqrt((1 / self.fission_fragment_spectrum.life_time **2 \
                     * self.fission_fragment_spectrum.life_time_uncertainty)**2)
        return _make_df(v, u)

    @property
    def _power_normalization(self) -> pd.DataFrame:
        """
        The power normalization to be multiplied by the fission fragment
        spectrum per unit mass.

        Returns
        -------
        pd.DataFrame
            with normalization value and uncertainty
        """
        start_time = self.fission_fragment_spectrum.start_time
        duration = self.fission_fragment_spectrum.real_time
        pm = self.power_monitor.average(start_time, duration)
        v, u = ratio_v_u(_make_df(1, 0), pm)
        return _make_df(v, u)

    def _get_long_output(self,
                         plateau: pd.DataFrame,
                         time: pd.DataFrame,
                         power: pd.DataFrame,
                         **kwargs) -> pd.DataFrame:
        """
        The information to be included in the long output: component
        variances.

        Parameters
        ----------
        plateaut : pd.DataFrame
            output of self.plateau()
        time : pd.DataFrame
            output of self._time_normalization
        power : pd.DataFrame
            output of self._power_normalization
        **kwargs
        Parameters for self.ffs.integrate()
            - bin_kwargs : dict, optional
                - bins : int (enforced to be same as EM.bins)
                - smooth : bool
            - max_kwargs : dict, optional
                kwargs for self.fission_fragment_spectrum.max().
                - fst_ch : int
            - llds : Iterable[int|float]

        Returns
        -------
        pd.DataFrame
            (1 x N) DataFrame with the information.
        """
        kwargs['bin_kwargs'] = kwargs['bin_kwargs'] if kwargs.get('bin_kwargs') else DEFAULT_BIN_KWARGS
        kwargs['max_kwargs'] = kwargs['max_kwargs'] if kwargs.get('max_kwargs') else DEFAULT_MAX_KWARGS
        # I always want to integrate over the same channels and binning as EM
        kwargs['bin_kwargs']['bins'] = self.effective_mass.bins

        ch_ffs, ch_em = plateau['CH_FFS'].value, plateau['CH_EM'].value
        ffs = self.fission_fragment_spectrum.integrate(**kwargs).query("channel==@ch_ffs")
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
    
    def _per_unit_mass_R(self, ffsi: pd.DataFrame, emi: pd.DataFrame) -> pd.DataFrame:
        """
        The tabulated ratio of FFS.integrate() / EM.integral, integrated from
        discrimination levels computed as a function of the R channel.

        Parameters
        ----------
        ffsi : pd.DataFrame
            Output of self.ffs.integrate().
        emi : pd.DataFrame
            Output of self.em.integral.

        Returns
        -------
        pd.DataFrame
            DataFrame with the information of the mass-normalized spectrum.
        """
        data = []
        channels = sorted(set(emi.R).intersection(set(ffsi.R)))
        if len(channels) < len(emi.R): warnings.warn("Neglecting calibration channels.")
        if len(channels) < len(ffsi.R): warnings.warn("Neglecting integration channels.")
        for r in channels:
            ffs_, em_ = ffsi.query("R==@r").iloc[0], emi.query("R==@r").iloc[0]
            v, u = ratio_v_u(ffs_, em_)
            data.append(_make_df(v, u).assign(
                                    VAR_FRAC_FFS = (ffs_.uncertainty / em_.value) **2,
                                    VAR_FRAC_EM = (ffs_.value / em_.value**2 * em_.uncertainty) **2,
                                    CH_FFS = ffs_.channel,
                                    CH_EM = em_.channel,
                                    R=r))
        return pd.concat(data, ignore_index=True)

    def _per_unit_mass_ch(self, ffsi: pd.DataFrame, emi: pd.DataFrame) -> pd.DataFrame:
        """
        The tabulated ratio of FFS.integrate() / EM.integral, integrated from
        discrimination levels defined as absolute channels.

        Parameters
        ----------
        ffsi : pd.DataFrame
            Output of self.ffs.integrate().
        emi : pd.DataFrame
            Output of self.em.integral.

        Returns
        -------
        pd.DataFrame
            DataFrame with the information of the mass-normalized spectrum.
        """
        data = []
        channels = sorted(set(emi.channel).intersection(set(ffsi.channel)))
        if len(channels) < len(emi.channel): warnings.warn("Neglecting calibration channels.")
        if len(channels) < len(ffsi.channel): warnings.warn("Neglecting integration channels.")
        for ch in channels:
            ffs_, em_ = ffsi.query("channel==@ch").iloc[0], emi.query("channel==@ch").iloc[0]
            v, u = ratio_v_u(ffs_, em_)
            data.append(_make_df(v, u).assign(
                                    VAR_FRAC_FFS = (ffs_.uncertainty / em_.value) **2,
                                    VAR_FRAC_EM = (ffs_.value / em_.value**2 * em_.uncertainty) **2,
                                    CH_FFS = ch,
                                    CH_EM = ch,
                                    R=np.nan))
        return pd.concat(data, ignore_index=True)

    def per_unit_mass(self, **kwargs) -> pd.DataFrame:
        """
        Normalizes the pulse height spectrum in self.ffs to the
        effective mass in self.em based on the effective mass
        discrimination levels.

        Parameters
        ----------
        **kwargs for self.ffs.integrate.
            - bin_kwargs : dict, optional
                - bins : int (enforced to be same as EM.bins)
                - smooth : bool
            - max_kwargs : dict, optional
                kwargs for self.fission_fragment_spectrum.max().
                - fst_ch : int
            - llds : Iterable[int|float]

        Returns
        -------
        pd.DataFrame
            DataFrame with the information of the mass-normalized spectrum.
        """
        kwargs['bin_kwargs'] = kwargs['bin_kwargs'] if kwargs.get('bin_kwargs') else DEFAULT_BIN_KWARGS
        kwargs['max_kwargs'] = kwargs['max_kwargs'] if kwargs.get('max_kwargs') else DEFAULT_MAX_KWARGS
        # I always want to integrate over the same channels and binning as EM
        kwargs['bin_kwargs']['bins'] = self.effective_mass.bins

        ffs = self.fission_fragment_spectrum.integrate(**kwargs)
        em = self.effective_mass.integral
        if np.isnan(em.R).all() and np.isnan(ffs.R).all():
            data = self._per_unit_mass_ch(ffs, em)
        elif not(np.isnan(em.R).all() and np.isnan(ffs.R).all()):
            data = self._per_unit_mass_R(ffs, em)
            warnings.warn("-------------- IN R --------------")
        else:
            raise Exception("Inconsistent integration and integration methodologies: can not process discrimination levels.",
                            ValueError)
        return data

    def per_unit_mass_and_time(self, **kwargs) -> pd.DataFrame:
        """
        The integrated FFS normalized per unit mass and time.

        Parameters
        ----------
        **kwargs
        Paramters for self.per_unit_mass
        - bin_kwargs : dict, optional
            - bins : int (enforced to be same as EM.bins)
            - smooth : bool
        - max_kwargs : dict, optional
            kwargs for self.fission_fragment_spectrum.max().
            - fst_ch : int
        - llds : Iterable[int|float]

        Returns
        -------
        pd.DataFrame
            DataFrame with the information of the mass- and time- normalized spectrum.
        """
        kwargs['bin_kwargs'] = kwargs['bin_kwargs'] if kwargs.get('bin_kwargs') else DEFAULT_BIN_KWARGS
        kwargs['max_kwargs'] = kwargs['max_kwargs'] if kwargs.get('max_kwargs') else DEFAULT_MAX_KWARGS
        # I always want to integrate over the same channels and binning as EM
        kwargs['bin_kwargs']['bins'] = self.effective_mass.bins

        data = pd.concat([_make_df(x[0], x[1]) for x in
                          [product_v_u([self._time_normalization.set_index(pd.Index([i])),
                                        self.per_unit_mass(**kwargs).loc[i].to_frame().T]) for i in
                                        self.per_unit_mass(**kwargs).index]],
                         ignore_index=True).assign(CH_FFS=self.per_unit_mass(**kwargs).CH_FFS,
                                                   CH_EM=self.per_unit_mass(**kwargs).CH_EM)
        return data[['CH_FFS', 'CH_EM', 'value', 'uncertainty', 'uncertainty [%]']]

    def per_unit_mass_and_power(self, **kwargs) -> pd.DataFrame:
        """
        The integrated FFS normalized per unit mass and power.

        Parameters
        ----------
        **kwargs for self.per_unit_mass
        - bin_kwargs : dict, optional
            - bins : int (enforced to be same as EM.bins)
            - smooth : bool
        - max_kwargs : dict, optional
            kwargs for self.fission_fragment_spectrum.max().
            - fst_ch : int
        - llds : Iterable[int|float]

        Returns
        -------
        pd.DataFrame
            DataFrame with the information of the mass- and power- normalized spectrum.
        """
        kwargs['bin_kwargs'] = kwargs['bin_kwargs'] if kwargs.get('bin_kwargs') else DEFAULT_BIN_KWARGS
        kwargs['max_kwargs'] = kwargs['max_kwargs'] if kwargs.get('max_kwargs') else DEFAULT_MAX_KWARGS
        # I always want to integrate over the same channels and binning as EM
        kwargs['bin_kwargs']['bins'] = self.effective_mass.bins

        ffs, em = self.fission_fragment_spectrum, self.effective_mass
        data = pd.concat([_make_df(x[0], x[1]) for x in
                          [product_v_u([self._power_normalization.set_index(pd.Index([i])),
                                        self.per_unit_mass(**kwargs).loc[i].to_frame().T]) for i in
                                        self.per_unit_mass(**kwargs).index]],
                         ignore_index=True).assign(CH_FFS=self.per_unit_mass(**kwargs).CH_FFS,
                                                   CH_EM=self.per_unit_mass(**kwargs).CH_EM)
        return data[['CH_FFS', 'CH_EM', 'value', 'uncertainty', 'uncertainty [%]']]

    def plateau(self, int_tolerance: float =.01, ch_tolerance: float =.01, **kwargs) -> pd.DataFrame:
        """
        Computes the reaction rate per unit mass.

        Parameters
        ----------
        int_tolerance : float, optional
            Tolerance for the integration check, by default 0.01.
        ch_tolerance : float, optional
            Tolerance for the channel check, by default 0.01.
        **kwargs:
            Paramters for self.per_unit_mass
            - bin_kwargs : dict, optional
                - bins : int
                - smooth : bool
            - max_kwargs : dict, optional
                kwargs for self.fission_fragment_spectrum.max().
                - fst_ch : int
            - llds : Iterable[int|float]

        Returns
        -------
        pd.DataFrame
            DataFrame containing the reaction rate per unit mass.

        Raises
        ------
        ValueError
            If the channel values differ beyond the specified tolerance.
        """
        kwargs['bin_kwargs'] = kwargs['bin_kwargs'] if kwargs.get('bin_kwargs') else DEFAULT_BIN_KWARGS
        kwargs['max_kwargs'] = kwargs['max_kwargs'] if kwargs.get('max_kwargs') else DEFAULT_MAX_KWARGS

        data = self.per_unit_mass(**kwargs)
        # check where the values in the mass-normalized count rate converge withing tolerance
        close_values = data[np.isclose(data.value, np.roll(data.value, shift=1), rtol=int_tolerance)]
        if close_values.shape[0] == 0:
            raise Exception("No convergence found with the given tolerance on the integral.", ValueError)
        # consider only values where the convergence is observed in two neighboring channels, i.e. atol=1
        plateau = close_values[np.isclose(close_values.index, np.roll(close_values.index, shift=1), atol=1)]
        if plateau.shape[0] == 0:
            raise Exception("No convergence found in neighbouring channels.", ValueError)
        # check the channels in which value does not differ more than ch_tolerance from the calibration ones
        plateau = plateau[abs(plateau['CH_FFS'] - plateau['CH_EM'])
                / plateau['CH_EM'] < ch_tolerance]
        if plateau.shape[0] == 0:
            raise Exception("No convergence found with the given tolerance on the channel.", ValueError)
        out = plateau.iloc[0].to_frame().T
        out.index = ['value']
        return out

    def process(self, long_output: bool=False, *args, **kwargs) -> pd.DataFrame:
        """
        Computes the reaction rate.

        Parameters
        ----------
        long_output : bool, optional
            Flag to turn on/off the full ouptup information, whcih includes
            values and variances of all the processing elements, False by default.
        *args : Any
            Positional arguments to be passed to the `self.plateau()` method.
        **kwargs : dict
            Keyword arguments to be passed to the `self.plateau()` method.
            - bin_kwargs : dict, optional
                - bins : int (enforced to be same as EM.bins)
                - smooth : bool
            - max_kwargs : dict, optional
                kwargs for self.fission_fragment_spectrum.max().
                - fst_ch : int
            - int_tolerance: float
            - ch_tolerance: float
            - llds : Iterable[int|float]

        Returns
        -------
        pd.DataFrame
            DataFrame containing the reaction rate.

        Examples
        --------
        >>> ffs = FissionFragmentSpectrum(data=pd.DataFrame({'value': [1.0, 2.0, 3.0], 'uncertainty': [0.1, 0.2, 0.3]}),
        ...                               detector_id='D1', deposit_id='Dep1', experiment_id='Exp1')
        >>> em = EffectiveMass(data=pd.DataFrame({'value': [0.5, 0.6, 0.7], 'uncertainty': [0.05, 0.06, 0.07]}),
        ...                    detector_id='D1', deposit_id='Dep1')
        >>> pm = ReactionRate(data=pd.DataFrame({'value': [10, 20, 30], 'uncertainty': [1, 2, 3]}), experiment_id='Exp1')
        >>> rr = NormalizedFissionFragmentSpectrum(fission_fragment_spectrum=ffs, effective_mass=em, power_monitor=pm)
        >>> rr.process()
            value  uncertainty
        0  35.6    2.449490
        """
        kwargs['bin_kwargs'] = kwargs['bin_kwargs'] if kwargs.get('bin_kwargs') else DEFAULT_BIN_KWARGS
        kwargs['max_kwargs'] = kwargs['max_kwargs'] if kwargs.get('max_kwargs') else DEFAULT_MAX_KWARGS
        # I always want to integrate over the same channels and binning as EM
        kwargs['bin_kwargs']['bins'] = self.effective_mass.bins

        plateau = self.plateau(*args, **kwargs) # FFS / EM @plateau and relative variance fractions
        power = self._power_normalization       # this is 1/PM
        time = self._time_normalization         # this is 1/t
        v, u = product_v_u([plateau, power, time])

        # compute variance fractions
        S_PLAT, S_PM, S_T = power.value * time.value, plateau.value * time.value, plateau.value * power.value
        df = _make_df(v, u).assign(VAR_FRAC_FFS=plateau["VAR_FRAC_FFS"] * S_PLAT **2,
                                   VAR_FRAC_EM=plateau["VAR_FRAC_EM"] * S_PLAT **2,
                                   VAR_FRAC_PM=(S_PM * power.uncertainty) **2,
                                   VAR_FRAC_t=(S_T * time.uncertainty) **2)
        return df if not long_output else pd.concat([df,
                                                     self._get_long_output(plateau,
                                                                           time,
                                                                           power,
                                                                           **kwargs)
                                                    ], axis=1)


@dataclass
class SpectralIndex(_Experimental):
    numerator: NormalizedFissionFragmentSpectrum
    denominator: NormalizedFissionFragmentSpectrum

    def __post_init__(self):
        self._check_consistency()

    def _check_consistency(self) -> None:
        """
        Checks the consistency of:
            - campaign_id
            - location_id
        among `self.numerator` and `self.denominator`.

        Raises
        ------
        UserWarning
            If there are inconsistencies among the specified attributes.
        """
        should = ['campaign_id', 'location_id']
        for attr in should:
            if not getattr(self.numerator, attr
                           ) == getattr(self.denominator, attr):
                warnings.warn(f"Inconsistent {attr} among numerator and denominator.")

    @property
    def deposit_ids(self) -> list[str]:
        """
        The deposit IDs associated with the numerator and denominator.

        Returns
        -------
        list[str]
            A list containing the deposit IDs of the numerator and denominator.

        Examples
        --------
        >>> from REPTILE.ReactionRate import ReactionRate
        >>> ffs_num = ReactionRate(..., deposit_id='Dep1')
        >>> ffs_den = ReactionRate(..., deposit_id='Dep2')
        >>> spectral_index = SpectralIndex(numerator=ffs_num, denominator=ffs_den)
        >>> spectral_index.deposit_ids
        ['Dep1', 'Dep2']
        """
        return [self.numerator.deposit_id, self.denominator.deposit_id]

    def _compute_correction(self, one_g_xs: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the impurity correction to the spectral index.

        Parameters
        ----------
        one_g_xs : pd.DataFrame, optional
            dataframe with nuclides as index, one group cross sections as `value`
            and absolute uncertainty as `uncertainty`.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the correction value and uncertainty.

        Raises
        ------
        UserWarning
            If xs is not given for all impurities.
        """
        imp = self.numerator.effective_mass.composition_.set_index('nuclide')
        imp.columns = ['value', 'uncertainty']

        # normalize Serpent output per unit mass
        v = one_g_xs['value'] / ATOMIC_MASS.T['value']
        u = one_g_xs['uncertainty'] / ATOMIC_MASS.T['value']
        xs = pd.concat([v.dropna(), u.dropna()], axis=1)
        xs.columns = ["value", "uncertainty"]

        # normalize impurities and one group xs to the numerator deposit
        imp_v, imp_u = ratio_v_u(imp,
                                 imp.loc[self.numerator.deposit_id])
        xs_v, xs_u = ratio_v_u(xs,
                               xs.loc[self.denominator.deposit_id])
        
        # remove information on the main isotope
        # will sum over all impurities != self.numerator.deposit_id
        imp_v = imp_v.drop(self.numerator.deposit_id)
        imp_u = imp_u.drop(self.numerator.deposit_id)
        xs_v = xs_v.drop(self.numerator.deposit_id)
        xs_u = xs_u.drop(self.numerator.deposit_id)
        
        # compute correction and its uncertainty
        if not all([i in xs_v.index] for i in imp_v.index):
            warnings.warn("Not all impurities were provided with a cross section.")
        correction = sum((imp_v * xs_v).dropna())
        correction_variance = sum(((xs_v * imp_u) **2 +
                                    (imp_v * xs_u) **2).dropna())
        relative = True if imp_v.shape[0] != 0 else False
        return _make_df(correction, np.sqrt(correction_variance), relative=relative)

    def _get_long_output(self, num, den, k) -> pd.DataFrame:
        """
        The information to be included in the long output:
        variances of numerator and denominator if those were
        computed in the first place and variance of the
        impurity correction (if any of the others was computed).

        Parameters
        ----------
        num : pd.DataFrame
            output of self.numerator.process()
        den : pd.DataFrame
            output of self.denominator.process
        k : pd.DataFrame
            impurity correction

        Returns
        -------
        pd.DataFrame
            (1 x N) DataFrame with the information or empty pd.DataFrame
            if the varaince was not computed for none of `num` and `den`.
        """
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
            den = pd.DataFrame()

        if not empty:
            k_ = pd.DataFrame({'1GXS': 0 if k is None else k['value'].iloc[0],
                               'VAR_1GXS': None if k is None else k['uncertainty'].iloc[0] **2},
                               index=['value'])
            out = pd.concat([num_, den_, k_], axis=1)
        else:
            out = pd.DataFrame()
        return out

    def process(self, one_g_xs: pd.DataFrame = None,
                one_g_xs_file: dict[str, tuple[str, str]] = None,
                *args, **kwargs) -> pd.DataFrame:
        """
        Computes the ratio of two reaction rates.

        Parameters
        ----------
        one_g_xs : pd.DataFrame, optional
            dataframe with nuclides as index, one group cross sections as `value`
            and absolute uncertainty as `uncertainty`.
            Defaults to None for no correction.
        one_g_xs_file : dict[str, tuple[str, str]], optional
            the Serpent detector file `value[1]` to read each one group xs
            `value[0]` from for each nuclide `key`. Alternative to `one_g_xs`.
            Defaults to None for no file.
        *args : Any
            Positional arguments to be passed to the
            `NormalizedFissionFragmentSpectrum.process()` method.
        **kwargs : Any
            Keyword arguments to be passed to the
            `NormalizedFissionFragmentSpectrum.process()` method.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the spectral index value and uncertainty.

        Examples
        --------
        >>> from REPTILE.ReactionRate import ReactionRate
        >>> ffs_num = ReactionRate(..., deposit_id='Dep1')
        >>> ffs_den = ReactionRate(..., deposit_id='Dep2')
        >>> spectral_index = SpectralIndex(numerator=ffs_num, denominator=ffs_den)
        >>> spectral_index.process()
            value  uncertainty
        0  0.95   0.034139
        """
        num, den = self.numerator.process(*args, **kwargs), self.denominator.process(*args, **kwargs)
        v, u = ratio_v_u(num, den)
        if (one_g_xs is None and one_g_xs_file is None
            and self.numerator.effective_mass.composition_.shape[0] > 1):
            warnings.warn("Impurities in the fission chambers require one group xs" +\
                          " to be accounted for.")
        if one_g_xs_file is not None:
            read = pd.DataFrame({nuc: sts.read(det[1]).detectors[det[0]].bins[0][-2:]
                                 for nuc, det in one_g_xs_file.items()}).T
            read.columns = ['value', 'uncertainty']
            read.index.name = 'nuclide'
            # uncertainy is absolute
            read.uncertainty = read.uncertainty * read.value
        else:
            read = None

        one_g_xs_ = read if one_g_xs is None else one_g_xs
        if one_g_xs_ is not None:
            k = self._compute_correction(one_g_xs_)
            v = v - k.value
            u = np.sqrt(u **2 + k.uncertainty **2)
        else: k = None
        df = _make_df(v, u)

        # compute fraction of variance
        var_cols = [c for c in num.columns if c.startswith("VAR_FRAC")]
        
        var_num = num[var_cols] / den['value'].value **2
        var_num.columns = [f"{c}_n" for c in var_cols]

        var_den = den[var_cols] * (num['value'] / den['value'] **2).value **2
        var_den.columns = [f"{c}_d" for c in var_cols]

        # concatenate variances to `df`
        df =  pd.concat([df, var_num, var_den], axis=1).assign(
                                    VAR_FRAC_1GXS=k.uncertainty **2 if k is not None else 0.
                                    )
        return pd.concat([df, self._get_long_output(num, den, k)], axis=1)

@dataclass(slots=True)
class Traverse(_Experimental):
    reaction_rates: dict[str, ReactionRate | ReactionRates]
    
    def __post_init__(self):
        for item in self.reaction_rates.values():
            if not self._first.campaign_id == item.campaign_id:
                    warnings.warn("Not matching campaign ids.")
            if not self._first.deposit_id == item.deposit_id:
                    warnings.warn("Not matching deposit ids.")

    @property
    def _first(self):
        return list(self.reaction_rates.values())[0]

    @property
    def deposit_id(self):
        return self._first.deposit_id

    def process(self, monitors: Iterable[ReactionRate| int], *args, normalization: int|str=None,
                **kwargs) -> pd.DataFrame:
        """
        Normalizes all the reaction rates to the power in `monitors`
        and to the maximum value.

        Parameters
        ----------
        monitors : Iterable[ReactionRate | int]
            ordered information on the power normalization.
            Should be `ReactionRate` when mapped to a `ReactionRate` and
            int when mapped to `ReactionRates`. The normalization is passed to
            `ReactionRate.per_unit_time_power()` or `ReactionRates.per_unit_time_power()`.
        *args : Any
            Positional arguments to be passed to the `ReactionRate.plateau()` method.
        normalization : str, optional
            The `self.reaction_rates` ReactionRate identifier to normalize the traveres to.
            Defaults to None, normalizing to the one with the highest counts.
        **kwargs : Any
            Keyword arguments to be passed to the `ReactionRate.plateau()` method.
        
        Returns
        -------
        pd.DataFrame
            with `value`, `uncertainty`, `uncertainty [%]`, `traverse` columns.

        Notes
        -----
        Working with `ReactionRates` instances, the first reaction rate is used.

        """
        normalized, m = {}, 0
        # Normalize to power
        for i, (k, rr) in enumerate(self.reaction_rates.items()):
            n = rr.per_unit_time_power(monitors[i], *args, **kwargs)
            normalized[k] = n if isinstance(rr, ReactionRate) else list(n.values())[0]
            if normalized[k]['value'].value > m:
                max_k, m = k, normalized[k].value[0]
        norm_k = max_k if normalization is None else normalization
        out = []
        for k, v in normalized.items():
            relative = False if k == norm_k else True
            v, u = ratio_v_u(v, normalized[norm_k])
            out.append(_make_df(v, u).assign(traverse=k))
        return pd.concat(out, ignore_index=True)
