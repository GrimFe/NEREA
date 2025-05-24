from dataclasses import dataclass
from collections.abc import Iterable
import warnings

import numpy as np
import pandas as pd

from .reaction_rate import ReactionRate
from .utils import _make_df, polyfit, polynomial
from .defaults import *
from .classes import EffectiveDelayedParams

__all__ = ["ControlRodCalibration",
           "DifferentialNoCompensation",
           "IntegralNoCompensation",
           "DifferentialCompensation",
           "IntegralCompensation",
           "evaluate_integral_differential_cr",
           "evaluate_integral_integral_cr"]

def evaluate_integral_differential_cr(x: float,
                                      order: int,
                                      coef: Iterable[float],
                                      coef_cov: Iterable[Iterable[float]]) -> pd.DataFrame:
    """
    Integrates a polynomial.

    Parameters:
    -----------
    x : float
        The point whereto evaluate the integral.
    order : int
        Polynomial order to integrate
    coef : Iterable[float]
        Polynomial coefficients.
    coef_cov : Iterable[Iterable[float]]
        Polynomial coefficients covariance matrix.
    
    Returns:
    --------
        pd.DataFrame
    
    Notes:
    ------
    Used for differential control rods.
    """
    v = sum([c / (order + 1 - i) * x **(order + 1 - i) for i, c in enumerate(coef)])
    sens = np.array([x **(order + 1 - i) / (order + 1 - i) for i in range(order + 1)])
    u = np.sqrt(sens @ coef_cov @ sens)
    return _make_df(v, u)

def evaluate_integral_integral_cr(x: float,
                                  order: int,
                                  coef: Iterable[float],
                                  coef_cov: Iterable[Iterable[float]]) -> pd.DataFrame:
    """
    Evaluates a olynomial.

    Parameters:
    -----------
    x : float
        The point whereto evaluate the integral.
    order : int
        Polynomial order to integrate
    coef : Iterable[float]
        Polynomial coefficients.
    coef_cov : Iterable[Iterable[float]]
        Polynomial coefficients covariance matrix.
    
    Returns:
    --------
        pd.DataFrame

    Notes:
    ------
    Used for integral control rods.
    """
    v = polynomial(order, coef, x)
    sens = np.array([x **(order - i) for i in range(order + 1)])
    u = np.sqrt(sens @ coef_cov @ sens)
    return _make_df(v, u)


@dataclass(slots=True)
class ControlRodCalibration:
    reaction_rates: dict[float, ReactionRate]  # height and corresponding RR
    critical_height: float
    name: int

    def _get_rhos(self,
                  delayed_data: EffectiveDelayedParams,
                  dtc_kwargs: dict=None,
                  ac_kwargs: dict=None) -> pd.DataFrame:
        """"
        Computes the reactivity associated with each reaction rate
        in self.reaction_rates.

        Parameters
        ---------
        delayed_data : EffectiveDelayedParams
            effective delayed  neutron data for control rod calibration.
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
            rho = ac.get_reactivity(delayed_data)
            rhos.append(rho.assign(h=h))
        rhos = pd.concat(rhos)
        return rhos.fillna(0)

    def get_reactivity_curve(self):
        # placeholder for methods of inheriting classes
        pass

    def _evaluate_integral(self):
        # placeholder for methods of inheriting classes
        pass

    def get_reactivity_worth(self,
                             x0: float,
                             x1: float,
                             delayed_data: EffectiveDelayedParams,
                             order: int,
                             dtc_kwargs: dict=None,
                             ac_kwargs: dict=None) -> pd.DataFrame:
            rho = self.get_reactivity_curve(delayed_data, dtc_kwargs, ac_kwargs)[['h', 'value', 'uncertainty']].copy()
            rho.columns = ['x', 'y', 'u']
            coef, coef_cov = polyfit(order, rho)
            i0 = self._evaluate_integral(x0, order, coef, coef_cov)
            i1 = self._evaluate_integral(x1, order, coef, coef_cov)
            return _make_df(i1.value - i0.value, np.sqrt(i1.uncertainty **2 + i0.uncertainty **2)
                            ).assign(VAR_PORT_X1=i1.uncertainty **2,
                                     VAR_PORT_X0=i0.uncertainty **2)


@dataclass(slots=True)
class DifferentialNoCompensation(ControlRodCalibration):

    def get_reactivity_curve(self,
                             delayed_data: EffectiveDelayedParams,
                             dtc_kwargs: dict=None,
                             ac_kwargs: dict=None) -> pd.DataFrame:
        """"
        Computes the differential reacitivty curve dr/dh for measurements without compensation.

        Parameters
        ---------
        delayed_data : EffectiveDelayedParams
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
        rho = self._get_rhos(delayed_data, dtc_kwargs, ac_kwargs)
        drho_v = (rho["value"].diff() / rho["h"].diff()).fillna(0).values

        VAR_PORT_T = rho["VAR_PORT_T"].rolling(2).sum() / rho["h"].diff() **2
        VAR_PORT_B = rho["VAR_PORT_B"].rolling(2).sum() / rho["h"].diff() **2
        VAR_PORT_L = rho["VAR_PORT_L"].rolling(2).sum() / rho["h"].diff() **2
        out = _make_df(drho_v, np.sqrt(VAR_PORT_T + VAR_PORT_B + VAR_PORT_L)).assign(VAR_PORT_T=VAR_PORT_T,
                                                                                     VAR_PORT_B=VAR_PORT_B,
                                                                                     VAR_PORT_L=VAR_PORT_L,
                                                                                     h=rho['h'].rolling(2).mean()  # each differential corrsponds to the average of two heights
                                                                                     )
        return out.dropna()  # dropping the first NaN of diff()

    @staticmethod
    def _evaluate_integral(x: float,
                           order: int,
                           coef: Iterable[float],
                           coef_cov: Iterable[Iterable[float]]):
        return evaluate_integral_differential_cr(x, order, coef, coef_cov)


@dataclass(slots=True)
class IntegralNoCompensation(ControlRodCalibration):

    def get_reactivity_curve(self,
                             delayed_data: EffectiveDelayedParams,
                             dtc_kwargs: dict=None,
                             ac_kwargs: dict=None) -> pd.DataFrame:
        """"
        Computes the integral reacitivty curve dr/dh for measurement without compensation.

        Parameters
        ---------
        delayed_data: EffectiveDelayedParams
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
        rho = self._get_rhos(delayed_data, dtc_kwargs, ac_kwargs)
        return rho
    
    @staticmethod
    def _evaluate_integral(x: float,
                           order: int,
                           coef: Iterable[float],
                           coef_cov: Iterable[Iterable[float]]):
        return evaluate_integral_integral_cr(x, order, coef, coef_cov)


@dataclass(slots=True)
class DifferentialCompensation(ControlRodCalibration):

    def get_reactivity_curve(self,
                             delayed_data: EffectiveDelayedParams,
                             dtc_kwargs: dict=None,
                             ac_kwargs: dict=None) -> pd.DataFrame:
        """"
        Computes the differential reacitivty curve dr/dh for measurements with compensation.

        Parameters
        ---------
        delayed_data: EffectiveDelayedParams
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
        rho = self._get_rhos(delayed_data, dtc_kwargs, ac_kwargs)
        # no need to differenciate rho as with compensation the measured reactivity
        # is already related with a differential movement of the control rod
        drho_v = rho.value / rho["h"].diff()
        
        VAR_PORT_T = rho["VAR_PORT_T"] / rho["h"].diff() **2
        VAR_PORT_B = rho["VAR_PORT_B"] / rho["h"].diff() **2
        VAR_PORT_L = rho["VAR_PORT_L"] / rho["h"].diff() **2
        out = _make_df(drho_v, np.sqrt(VAR_PORT_T + VAR_PORT_B + VAR_PORT_L)).assign(VAR_PORT_T=VAR_PORT_T,
                                                                                     VAR_PORT_B=VAR_PORT_B,
                                                                                     VAR_PORT_L=VAR_PORT_L,
                                                                                     h=rho['h'].rolling(2).mean()  # each differential corrsponds to the average of two heights
                                                                                     )
        return out.dropna()  # dropping the first NaN of diff()

    @staticmethod
    def _evaluate_integral(x: float,
                           order: int,
                           coef: Iterable[float],
                           coef_cov: Iterable[Iterable[float]]):
        return evaluate_integral_differential_cr(x, order, coef, coef_cov)


@dataclass(slots=True)
class IntegralCompensation(ControlRodCalibration):

    def get_reactivity_curve(self,
                             delayed_data: EffectiveDelayedParams,
                             dtc_kwargs: dict=None,
                             ac_kwargs: dict=None) -> pd.DataFrame:
        """"
        Computes the integral reacitivty curve dr/dh for measurement with compensation.

        Parameters
        ---------
        delayed_data: EffectiveDelayedParams
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
        rho = self._get_rhos(delayed_data, dtc_kwargs, ac_kwargs)
        return rho.loc[:, rho.columns != 'h'].cumsum().assign(h=rho['h'])

    @staticmethod
    def _evaluate_integral(x: float,
                           order: int,
                           coef: Iterable[float],
                           coef_cov: Iterable[Iterable[float]]):
        return evaluate_integral_integral_cr(x, order, coef, coef_cov)
