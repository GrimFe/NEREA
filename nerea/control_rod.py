from dataclasses import dataclass
import numpy as np
import pandas as pd

from .reaction_rate import ReactionRate
from .utils import _make_df

@dataclass(slots=True)
class ControlRodCalibration:
    reaction_rates: dict[float, ReactionRate]  # height and corresponding RR
    id: int

    def differential_curve_no_compensation(self, dtc_kwargs, ac_kwargs, rho_kwargs):
        rho = []
        for r in self.reaction_rates.values():
            rho.append(r.dead_time_correction(**dtc_kwargs
                                              ).get_asymptotic_counts(**ac_kwargs
                                                                      ).get_reactivity(**rho_kwargs))
        rho = pd.concat(rho, ignore_index=True).assign(h=self.reaction_rates.keys())

        drho_v = rho["value"].diff() / rho["h"].diff() # (...).fillna(0).values  # why fillna?
        VAR_FRAC_T = rho["VAR_FRAC_T"] / rho["h"].diff() **2
        VAR_FRAC_B = rho["VAR_FRAC_B"] / rho["h"].diff() **2
        VAR_FRAC_L = rho["VAR_FRAC_L"] / rho["h"].diff() **2
        out = _make_df(drho_v, np.sqrt(VAR_FRAC_T + VAR_FRAC_B + VAR_FRAC_L)).assign(VAR_FRAC_T=VAR_FRAC_T,
                                                                                     VAR_FRAC_B=VAR_FRAC_B,
                                                                                     VAR_FRAC_L=VAR_FRAC_L,
                                                                                     h=rho['h'])
        return out.set_index('h').reset_index()

    def integral_curve_no_compensation(self, dtc_kwargs, ac_kwargs, rho_kwargs):
        integral_curve = CR_1[differential_curve.columns]
        
        return cls({"differential": differential_curve.fillna(0),"integral":integral_curve.fillna(0)})

    def differential_curve_compensation(self, dtc_kwargs, ac_kwargs, rho_kwargs):
        pass

    def integral_curve_compensation(self, dtc_kwargs, ac_kwargs, rho_kwargs):
        pass

    def get_reactivity_worth(self, metod, **kwargs):
        pass
