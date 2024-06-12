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

    def _compute_si(self, _minus_one_percent=False, **kwargs):
        v, u = ratio_v_u(self.c.calculate(), self.e.compute(**kwargs))
        return _make_df(v, u) if not _minus_one_percent else _make_df((v - 1) * 100, u * 100, relative=False)

    def _compute_traverse(self, _minus_one_percent=False, normalization=None, **kwargs):
        exp = self.e.compute(normalization=normalization, **kwargs)
        cal = self.c.calculate(normalization=normalization)
        out = []
        for t in exp.traverse:
            v, u = ratio_v_u(cal.query("traverse == @t").reset_index(),
                             exp.query("traverse == @t").reset_index())
            v, u = v.iloc[0], u.iloc[0]
            out.append(_make_df(v, u).assign(traverse=t) if not _minus_one_percent else
                       _make_df((v - 1) * 100 , u * 100, relative=False).assign(traverse=t))
        return pd.concat(out, ignore_index=True)

    def compute(self, _minus_one_percent=False, *args, normalization: str =None,
                **kwargs) -> pd.DataFrame:
        """
        Computes the cover-e value.

        Parameters
        ----------
        _minus_one_percent : bool, optional
            computes the C/E-1 [%]. Defaults to False.
        *args : Any
            Positional arguments to be passed to the `SpectralIndex.compute()` method.
        normalization : str, optional
            The detector name to normalize the traveres to.
            Defaults to None, normalizing to the one with the highest counts.
            Will be used to compute traverse C/E for normalization of both C and E.
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
            out = self._compute_si(_minus_one_percent, **kwargs)
        if isinstance(self.e, Traverse):
            out = self._compute_traverse(_minus_one_percent, normalization=normalization, **kwargs)
        return out

    def minus_one_percent(self, **kwargs):
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
        return self.compute(**kwargs, _minus_one_percent=True)
