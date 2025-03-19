from dataclasses import dataclass
from collections.abc import Iterable
import logging

import numpy as np
import pandas as pd

from .reaction_rate import ReactionRate
from .utils import _make_df, get_fit_R2
from .defaults import *

__all__ = ["ControlRodCalibration"]

@dataclass(slots=True)
class ControlRodCalibration:
    reaction_rates: dict[float, ReactionRate]  # height and corresponding RR
    critical_height: float
    name: int

    def _get_rhos(self, delayed_data_file: str, dtc_kwargs: dict=None, ac_kwargs: dict=None) -> pd.DataFrame:
        """"
        Computes the reactivity associated with each reaction rate
        in self.reaction_rates.

        Parameters
        ---------
        delayed_data_file : str
            path to the Serpent `res.m` output file to read effective delayed
            neutron data from.
        dtc_kwargs : dict, optional
            kwargs for nerea.ReactionRate.dead_time_corrected.
            Default is None.
        ac_kwargs : dict, optional
            kwargs for nerea.ReactionRate.asymptotic_counts.
            Default is None.

        Returns
        -------
        pd.DataFrame
        """
        dtc_kw = DEFAULT_DTC_KWARGS if dtc_kwargs is None else dtc_kwargs
        ac_kw = DEFAULT_AC_KWARGS if ac_kwargs is None else ac_kwargs
        rhos = [_make_df(0, 0, False).assign(h=self.critical_height)]
        for h, r in self.reaction_rates.items():
            dtc = r.dead_time_corrected(**dtc_kw)
            ac = dtc.get_asymptotic_counts(**ac_kw)
            rho = ac.get_reactivity(delayed_data_file)
            rhos.append(rho.assign(h=h))
        rhos = pd.concat(rhos)
        return rhos.fillna(0)

    def differential_curve_no_compensation(self, delayed_data_file: str, dtc_kwargs: dict=None, ac_kwargs: dict=None) -> pd.DataFrame:
        """"
        Computes the differential reacitivty curve dr/dh for measurements without compensation.

        Parameters
        ---------
        delayed_data_file : str
            path to the Serpent `res.m` output file to read effective delayed
            neutron data from.
        dtc_kwargs : dict, optional
            kwargs for nerea.ReactionRate.dead_time_corrected.
            Default is None.
        ac_kwargs : dict, optional
            kwargs for nerea.ReactionRate.asymptotic_counts.
            Default is None.

        Returns
        -------
        pd.DataFrame
        """
        rho = self._get_rhos(delayed_data_file, dtc_kwargs, ac_kwargs)
        drho_v = (rho["value"].diff() / rho["h"].diff()).fillna(0).values

        VAR_FRAC_T = rho["VAR_FRAC_T"].rolling(2).sum() / rho["h"].diff() **2
        VAR_FRAC_B = rho["VAR_FRAC_B"].rolling(2).sum() / rho["h"].diff() **2
        VAR_FRAC_L = rho["VAR_FRAC_L"].rolling(2).sum() / rho["h"].diff() **2
        out = _make_df(drho_v, np.sqrt(VAR_FRAC_T + VAR_FRAC_B + VAR_FRAC_L)).assign(VAR_FRAC_T=VAR_FRAC_T,
                                                                                     VAR_FRAC_B=VAR_FRAC_B,
                                                                                     VAR_FRAC_L=VAR_FRAC_L,
                                                                                     h=rho['h'].rolling(2).mean()  # each differential corrsponds to the average of two heights
                                                                                     )
        return out.dropna().reset_index(drop=True)  # dropping the first NaN of diff()

    def integral_curve_no_compensation(self, delayed_data_file: str, dtc_kwargs: dict=None, ac_kwargs: dict=None) -> pd.DataFrame:
        """"
        Computes the integral reacitivty curve dr/dh for measurement without compensation.

        Parameters
        ---------
        delayed_data_file : str
            path to the Serpent `res.m` output file to read effective delayed
            neutron data from.
        dtc_kwargs : dict, optional
            kwargs for nerea.ReactionRate.dead_time_corrected.
            Default is None.
        ac_kwargs : dict, optional
            kwargs for nerea.ReactionRate.asymptotic_counts.
            Default is None.

        Returns
        -------
        pd.DataFrame

        Note
        ----
        alias for self._get_rhos
        """
        rho = self._get_rhos(delayed_data_file, dtc_kwargs, ac_kwargs)
        return rho

    def differential_curve_compensation(self, delayed_data_file: str, dtc_kwargs: dict=None, ac_kwargs: dict=None) -> pd.DataFrame:
        """"
        Computes the differential reacitivty curve dr/dh for measurements with compensation.

        Parameters
        ---------
        delayed_data_file : str
            path to the Serpent `res.m` output file to read effective delayed
            neutron data from.
        dtc_kwargs : dict, optional
            kwargs for nerea.ReactionRate.dead_time_corrected.
            Default is None.
        ac_kwargs : dict, optional
            kwargs for nerea.ReactionRate.asymptotic_counts.
            Default is None.

        Returns
        -------
        pd.DataFrame
        """
        rho = self._get_rhos(delayed_data_file, dtc_kwargs, ac_kwargs)
        # no need to differenciate rho as with compensation the measured reactivity
        # is already related with a differential movement of the control rod
        drho_v = rho.value / rho["h"].diff()
        
        VAR_FRAC_T = rho["VAR_FRAC_T"] / rho["h"].diff() **2
        VAR_FRAC_B = rho["VAR_FRAC_B"] / rho["h"].diff() **2
        VAR_FRAC_L = rho["VAR_FRAC_L"] / rho["h"].diff() **2
        out = _make_df(drho_v, np.sqrt(VAR_FRAC_T + VAR_FRAC_B + VAR_FRAC_L)).assign(VAR_FRAC_T=VAR_FRAC_T,
                                                                                     VAR_FRAC_B=VAR_FRAC_B,
                                                                                     VAR_FRAC_L=VAR_FRAC_L,
                                                                                     h=rho['h'].rolling(2).mean()  # each differential corrsponds to the average of two heights
                                                                                     )
        return out.dropna().reset_index(drop=True)  # dropping the first NaN of diff()

    def integral_curve_compensation(self, delayed_data_file: str, dtc_kwargs: dict=None, ac_kwargs: dict=None) -> pd.DataFrame:
        """"
        Computes the integral reacitivty curve dr/dh for measurement with compensation.

        Parameters
        ---------
        delayed_data_file : str
            path to the Serpent `res.m` output file to read effective delayed
            neutron data from.
        dtc_kwargs : dict, optional
            kwargs for nerea.ReactionRate.dead_time_corrected.
            Default is None.
        ac_kwargs : dict, optional
            kwargs for nerea.ReactionRate.asymptotic_counts.
            Default is None.

        Returns
        -------
        pd.DataFrame

        Note
        ----
        alias for self._get_rhos
        """
        rho = self._get_rhos(delayed_data_file, dtc_kwargs, ac_kwargs)
        return rho.loc[:, rho.columns != 'h'].cumsum().assign(h=rho['h'])

    def get_reactivity_curve(self, differential: bool, compensation: bool, delayed_data_file: str, dtc_kwargs: dict=None,
                ac_kwargs: dict=None) -> pd.DataFrame:
        if differential:
            if compensation: rho = self.differential_curve_compensation(delayed_data_file, dtc_kwargs, ac_kwargs)
            else: rho = self.differential_curve_no_compensation(delayed_data_file, dtc_kwargs, ac_kwargs)

        if not differential:
            if compensation: rho = self.integral_curve_compensation(delayed_data_file, dtc_kwargs, ac_kwargs)
            else: rho = self.integral_curve_no_compensation(delayed_data_file, dtc_kwargs, ac_kwargs)
        return rho

    def get_reactivity_worth(self, differential: bool, compensation: bool, x0: float, x1: float, delayed_data_file: str,
                             diff_fit_order: int=2, dtc_kwargs: dict=None, ac_kwargs: dict=None) -> pd.DataFrame:
        from scipy.optimize import curve_fit
        order = diff_fit_order if differential else diff_fit_order + 1
        def poly(c: Iterable[float], x: float):
            # returns the polynomial value @x
            return sum([c[i] * x **(order - i) for i in range(order + 1)])

        rho = self.get_rho(self, differential, compensation, delayed_data_file, dtc_kwargs, ac_kwargs)
        coef, coef_cov, out, _, _ = curve_fit(poly, rho.h, rho.value, sigma=rho.uncertainty, absolute_sigma=True, full_output=True)

        r2 = get_fit_R2(rho.value, out["fvec"], weight=1 / rho.uncertainty **2)
        logging.info("CR reactivity curve fit R^2 = %s", r2)  # probably not functioning

        # integral of fitted function
        def evaluate_integral(x: float):
            if differential:
                v = sum([c / (order + 1 - i) * x **(order + 1 - i) for i, c in enumerate(coef)])
                sens = np.array([x **(order + 1 - i) / (order + 1 - i) for i in range(order + 1)])
                u = np.sqrt(sens @ coef_cov @ sens)
            else:
                v = poly(x)
                sens = np.array([x **(order - i) for i in range(order + 1)])
                u = np.sqrt(sens @ coef_cov @ sens)
            return _make_df(v, u)
        i0 = evaluate_integral(x0)
        i1 = evaluate_integral(x1)

        return _make_df(i1.value - i0.value, np.sqrt(i1.uncertainty **2 + i0.uncertainty **2)).assign(VAR_FRAC_X1=i1.uncertainty **2,
                                                                                                      VAR_FRAC_X0=i0.uncertainty **2)
