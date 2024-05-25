from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from REPTILE import Computable
from REPTILE import Calculated
from REPTILE.utils import ratio_v_u, _make_df


__all__ = ['CoverE']

@dataclass
class CoverE:
    c: Calculated
    e: Computable

    def __post_init__(self):
        self._check_consistency()

    def _check_consistency(self) -> None:
        if not self.c.deposit_ids == self.e.deposit_ids:
            raise Exception("Inconsistent deposits between C and E.")

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
        v, u = ratio_v_u(self.c.compute(), self.e.compute(*args, **kwargs))
        return _make_df(v, u)

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
