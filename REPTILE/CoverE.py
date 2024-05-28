from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from REPTILE import Computable, SpectralIndex, Traverse
from REPTILE import Calculated, CalculatedSpectralIndex, CalculatedTraverse
from REPTILE.utils import ratio_v_u, _make_df


__all__ = ['CoverE']

@dataclass(slots=True)
class CoverE:
    c: Calculated
    e: Computable

    def __post_init__(self):
        self._check_consistency()

    def _check_consistency(self) -> None:
        self._check_class_consistency()
        if isinstance(self.e, Traverse):
            if not self.c.deposit_id == self.e.deposit_id:
                raise Exception("Inconsistent deposits between C and E.")
        if isinstance(self.e, SpectralIndex):
            if not self.c.deposit_ids == self.e.deposit_ids:
                raise Exception("Inconsistent deposits between C and E.")
        
    def _check_class_consistency(self):
        if ((isinstance(self.c, CalculatedTraverse) and not isinstance(self.e, Traverse)) or
            (isinstance(self.e, Traverse) and not isinstance(self.c, CalculatedTraverse))):
            raise Exception("Cannot compare Traverse and non-Traverse object.")
        if (isinstance(self.c, CalculatedSpectralIndex) and not isinstance(self.e, SpectralIndex) or
            isinstance(self.e, SpectralIndex) and not isinstance(self.c, CalculatedSpectralIndex)):
            raise Exception("Cannot compare SpectralIndex and non-SpectralIndex object.")

    @property
    def deposit_ids(self) -> list[str]:
        """
        The deposit IDs associated with the numerator and denominator.
        Consistency check performed in `self._check_consistency()`.

        Returns
        -------
        list[str]
            A list containing the deposit IDs of the numerator and denominator.
        """
        return self.c.deposit_ids

    def _compute_si(self, *args, **kwargs):
        v, u = ratio_v_u(self.c.calculate(), self.e.compute(*args, **kwargs))
        return _make_df(v, u)

    def _compute_traverse(self, *args, **kwargs):
        exp = self.e.compute(*args, **kwargs)
        cal = self.c.calculate()
        out = []
        for t in exp.traverse:
            v, u = ratio_v_u(cal.query("traverse == @t"),
                             exp.query("traverse == @t"))
            out.append(_make_df(v.iloc[0], u.iloc[0]).assign(traverse=t))
        return pd.concat(out, ignore_index=True)

    def compute(self, *args, **kwargs) -> pd.DataFrame:
        """
        Computes the cover-e value.

        Parameters
        ----------
        *args : Any
            Positional arguments to be passed to the `SpectralIndex.compute()` method.
        **kwargs : Any
            Keyword arguments to be passed to the `SpectralIndex.compute()` method.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the cover-e value.

        Examples
        --------
        >>> c_instance = C(data=pd.DataFrame({'value': [0.5]}), model_id='Model1', deposit_ids='Dep1')
        >>> ffs_num = ReactionRate(...)  # Assuming proper initialization
        >>> ffs_den = ReactionRate(...)  # Assuming proper initialization
        >>> spectral_index = SpectralIndex(numerator=ffs_num, denominator=ffs_den)
        >>> cover_e_instance = CoverE(c=c_instance, e=spectral_index)
        >>> cover_e_instance.compute()
           value  uncertainty
        0    ...          ...
        """
        if isinstance(self.e, SpectralIndex):
            out = self._compute_si(*args, **kwargs)
        if isinstance(self.e, Traverse):
            out = self._compute_traverse(*args, **kwargs)
        return out

    def minus_one_percent(self, *args, **kwargs):
        """
        Computes the cover-e value and subtracts 1%, adjusting the uncertainty accordingly.

        Parameters
        ----------
        *args : Any
            Positional arguments to be passed to the `SpectralIndex.compute()` method.
        **kwargs : Any
            Keyword arguments to be passed to the `SpectralIndex.compute()` method.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the adjusted cover-e value.

        Examples
        --------
        >>> c_instance = C(data=pd.DataFrame({'value': [1.05]}), model_id='Model1', deposit_ids='Dep1')
        >>> ffs_num = ReactionRate(...)  # Assuming proper initialization
        >>> ffs_den = ReactionRate(...)  # Assuming proper initialization
        >>> spectral_index = SpectralIndex(numerator=ffs_num, denominator=ffs_den)
        >>> cover_e_instance = CoverE(c=c_instance, e=spectral_index)
        >>> cover_e_instance.minus_one_percent()
           value  uncertainty
        0    5.0          ...
        """
        out = self.compute(*args, **kwargs)
        return _make_df((out['value'] - 1) * 100, out['uncertainty'] * 100)
