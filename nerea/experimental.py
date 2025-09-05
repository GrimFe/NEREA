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
import matplotlib.pyplot as plt
import datetime

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
            - r : bool

        Returns
        -------
        pd.DataFrame
            (1 x N) DataFrame with the information.

        Note
        ----
        `bins` for PHS rebinning are set to `self.effective_mass.bins`
        """
        kwargs = DEFAULT_MAX_KWARGS | DEFAULT_BIN_KWARGS | kwargs
        # I always want to integrate over the same channels and binning as EM
        kwargs['bins'] = self.effective_mass.bins

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
                                    VAR_PORT_FFS = (ffs_.uncertainty / em_.value) **2,
                                    VAR_PORT_EM = (ffs_.value / em_.value**2 * em_.uncertainty) **2,
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
                                    VAR_PORT_FFS = (ffs_.uncertainty / em_.value) **2,
                                    VAR_PORT_EM = (ffs_.value / em_.value**2 * em_.uncertainty) **2,
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
            - r : bool

        Returns
        -------
        pd.DataFrame
            DataFrame with the information of the mass-normalized spectrum.

        Note
        ----
        `bins` for PHS rebinning are set to `self.effective_mass.bins`
        """
        kwargs = DEFAULT_MAX_KWARGS | DEFAULT_BIN_KWARGS | kwargs
        # I always want to integrate over the same channels and binning as EM
        kwargs['bins'] = self.effective_mass.bins

        ffs = self.fission_fragment_spectrum.integrate(**kwargs)
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
        - r : bool

        Returns
        -------
        pd.DataFrame
            DataFrame with the information of the mass- and time- normalized spectrum.

        Note
        ----
        `bins` for PHS rebinning are set to `self.effective_mass.bins`
        """
        kwargs = DEFAULT_MAX_KWARGS | DEFAULT_BIN_KWARGS | kwargs
        # I always want to integrate over the same channels and binning as EM
        kwargs['bins'] = self.effective_mass.bins

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
        - r : bool

        Returns
        -------
        pd.DataFrame
            DataFrame with the information of the mass- and power- normalized spectrum.

        Note
        ----
        `bins` for PHS rebinning are set to `self.effective_mass.bins`
        """
        kwargs = DEFAULT_MAX_KWARGS | DEFAULT_BIN_KWARGS | kwargs
        # I always want to integrate over the same channels and binning as EM
        kwargs['bins'] = self.effective_mass.bins

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
            - r : bool

        Returns
        -------
        pd.DataFrame
            DataFrame containing the reaction rate per unit mass.

        Raises
        ------
        ValueError
            If the channel values differ beyond the specified tolerance.
        """
        kwargs = DEFAULT_MAX_KWARGS | DEFAULT_BIN_KWARGS | kwargs
        
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

    def process(self, long_output: bool=False, visual: bool=False,
                savefig: str='', **kwargs) -> pd.DataFrame:
        """
        Computes the reaction rate.

        Parameters
        ----------
        long_output : bool, optional
            Flag to turn on/off the full ouptup information, whcih includes
            values and variances of all the processing elements, False by default.
        visual : bool, optional
            Flag to display the processed data.
            Default is False.
        savefig : str, optional
            Filename to save the figure to.
            Default is '' not saving.
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
            - r : bool
            - raw_integral : bool

        Returns
        -------
        pd.DataFrame
            DataFrame containing the reaction rate.
        
        Note
        ----
        `bins` for PHS rebinning are set to `self.effective_mass.bins`

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
        kwargs = DEFAULT_MAX_KWARGS | DEFAULT_BIN_KWARGS | kwargs
        # I always want to integrate over the same channels and binning as EM
        kwargs['bins'] = self.effective_mass.bins

        plateau = self.plateau(**kwargs)        # FFS / EM @plateau and relative variance fractions
        power = self._power_normalization       # this is 1/PM
        time = self._time_normalization         # this is 1/t
        v, u = product_v_u([plateau, power, time])

        # compute variance fractions
        S_PLAT, S_PM, S_T = power.value * time.value, plateau.value * time.value, plateau.value * power.value
        df = _make_df(v, u).assign(VAR_PORT_FFS=plateau["VAR_PORT_FFS"] * S_PLAT **2,
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
        Default plotting method of PHS and monitor considered.
        It also reports tabulated effective mass values.

        Paramters
        ---------
        discri: int, optional
            The discrimination level to highilight in the plots.
            It is in units of channel of self.fission_fragment_spectrum.
            Default is None.
        phs_kwargs: dict
            Parameters to process the spectrum before plotting.
            - bin_kwargs
            - max_kwargs
            - llds
            - r

        Returns
        -------
        tuple[plt.Figure, Iterable[plt.Axes]]
        """
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
                                start_time=self.fission_fragment_spectrum.start_time,
                                duration=self.fission_fragment_spectrum.real_time)

        ## plot PHS
        self.fission_fragment_spectrum.plot(ax=axs[1][0], **kwargs)
        axs[1][0].set_xlabel("Measurement channel")
        axs[1][0].set_ylabel("Counts [-]")

    	## plot fission rate per unit mass
        pum = self.per_unit_mass(**kwargs)
        pum.plot(x='CH_FFS', y='value', ax=axs[1][1], kind='scatter', c='k')
        axs[1][1].set_xticks(pum['CH_FFS'])
        axs[1][1].set_xticklabels([f"{x:.0f}" for x in pum['CH_FFS']])
        axs[1][1].set_ylabel("Fission rate per unit mass [1/ug]")

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
            discri_r = self.fission_fragment_spectrum.integrate(
                    **kwargs).query("channel == @discri").R.iloc[0]
            axs[0][0].scatter(x=self.effective_mass.data.query("R == @discri_r")['channel'].value,
                              y=self.effective_mass.data.query("R == @discri_r")['value'].value,
                              c='b', marker='s', label="Discriminator")
            axs[1][0].axvline(discri, c='b', label='Discriminator')
            axs[1][1].scatter(discri, pum.query("CH_FFS == @discri").value.iloc[0],
                          c='b', marker='s', label='Discriminator')
        return fig, axs


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
        imp = self.numerator.effective_mass.composition_
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
            den_ = pd.DataFrame()

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
                numerator_kwargs: dict={},
                denominator_kwargs: dict={}) -> pd.DataFrame:
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
        num = self.numerator.process(**numerator_kwargs)
        den = self.denominator.process(**denominator_kwargs)
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
        var_cols = [c for c in num.columns if c.startswith("VAR_PORT")]
        
        var_num = num[var_cols] / den['value'].value **2
        var_num.columns = [f"{c}_n" for c in var_cols]

        var_den = den[var_cols] * (num['value'] / den['value'] **2).value **2
        var_den.columns = [f"{c}_d" for c in var_cols]

        # concatenate variances to `df`
        df =  pd.concat([df, var_num, var_den], axis=1).assign(
                                    VAR_PORT_1GXS=k.uncertainty **2 if k is not None else 0.
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

    def process(self,
                monitors: Iterable[ReactionRate| int],
                normalization: int|str=None,
                visual: bool=False,
                savefig: str='',
                palette: str='tab10',
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
        normalization : str, optional
            The `self.reaction_rates` ReactionRate identifier to normalize the traveres to.
            Defaults to None, normalizing to the one with the highest counts.
        visual : bool, optional
            Plots the processed data.
            Default is False.
        savefig : str, optional
            File name to save the plotted data to.
            Default is `''` for not saving.
        palette : str, optional
            Color palette to use for plotting.
            Default is `'tab10'`.
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
            n = rr.per_unit_time_power(monitors[i], **kwargs)
            normalized[k] = n if isinstance(rr, ReactionRate) else list(n.values())[0]
            if normalized[k]['value'].value > m:
                max_k, m = k, normalized[k].value[0]
        norm_k = max_k if normalization is None else normalization
        out = []
        for k, v in normalized.items():
            v, u = ratio_v_u(v, normalized[norm_k])
            out.append(_make_df(v, u).assign(traverse=k))
        # plot
        if visual or savefig:
            fig, _ = self.plot(monitors, palette, **kwargs)
            if savefig:
                fig.savefig(savefig)
                plt.close()
        return pd.concat(out, ignore_index=True)

    def plot(self,
             monitors: Iterable[ReactionRate| int],
             palette: str='tab10',
             **kwargs) -> tuple[plt.Figure, Iterable[plt.Axes]]:
        """
        Plot the data processed in Traverse.

        Parameters
        ----------
        monitors : Iterable[ReactionRate | int]
            ordered information on the power normalization.
            Should be `ReactionRate` when mapped to a `ReactionRate` and
            int when mapped to `ReactionRates`. The normalization is passed to
            `ReactionRate.per_unit_time_power()` or `ReactionRates.per_unit_time_power()`.
        *args : Any
            Positional arguments to be passed to the `ReactionRate.plateau()` method.
        palette : str, optional
            plt palette to use for plotting.
            Defaults to `'tab10'`.
        **kwargs : Any
            Keyword arguments to be passed to the `ReactionRate.plateau()` method.
        
        Returns
        -------
        tuple[plt.Figure, Iterable[plt.Axes]]
        """
        fig, axs = plt.subplots(len(self.reaction_rates), 2,
                              figsize=(30 / len(self.reaction_rates), 15))
        j = 0
        for i, (k, rr) in enumerate(self.reaction_rates.items()):
            c = plt.get_cmap(palette)(j)
            plat = rr.plateau(**kwargs)
            dur = (plat.Time.max() - plat.Time.min()).total_seconds()
            # plot data
            rr.plot(start_time=plat.Time.min(), duration=dur, ax=axs[i][0], c=c)
            axs[i][0].plot([], [], c=c, label=k)
            # plot monitor
            axs[i][1] = monitors[i].plot(plat.Time.min(), dur, ax=axs[i][1], c=c)
            axs[i][1].plot([], [], c=c, label=k)

            h, l = axs[i][0].get_legend_handles_labels()
            axs[i][0].legend(h[1:], l[1:])
            h, l = axs[i][1].get_legend_handles_labels()
            axs[i][1].legend(h[1:], l[1:])

            j = j + 1 if i < plt.get_cmap(palette).N else 0
        return fig, axs