from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from REPTILE import _Experimental, SpectralIndex, Traverse
from REPTILE import _Calculated, CalculatedSpectralIndex, CalculatedTraverse
from REPTILE.utils import ratio_v_u, _make_df


__all__ = ['CoverE', 'CoverC']

@dataclass(slots=True)
class _Comparison:
    num: _Calculated
    den: _Experimental | _Calculated

    def _check_consistency_attrs(self) -> None:
        if isinstance(self.den, Traverse):
            if not self.num.deposit_id == self.den.deposit_id:
                raise Exception("Inconsistent deposits between C and E.")
        if isinstance(self.den, SpectralIndex):
            if not self.num.deposit_ids == self.den.deposit_ids:
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
        return self.num.deposit_ids

    def _get_denominator(self, **kwargs):
        return self.den.process(**kwargs) if isinstance(self.den,
                                                        _Experimental) else self.den.calculate()

    def _compute_si(self, _minus_one_percent=False, **kwargs):
        num, den = self.num.calculate(), self._get_denominator(**kwargs)
        v, u = ratio_v_u(num, den)
        if num["VAR_C_n"].value is not None:
            vcn, vcd = num["VAR_C_n"] / den.value **2, num["VAR_C_d"] / den.value **2
        else:
            vcn, vcd = None, None
        if isinstance(self.den, _Experimental):
            if not _minus_one_percent:
                df = _make_df(v, u).assign(
                                        VAR_FFS_n=den["VAR_FFS_n"] * (num.value / den.value **2) **2,
                                        VAR_EM_n=den["VAR_EM_n"] * (num.value / den.value **2) **2,
                                        VAR_PM_n=den["VAR_PM_n"] * (num.value / den.value **2) **2,
                                        VAR_FFS_d=den["VAR_FFS_d"] * (num.value / den.value **2) **2,
                                        VAR_EM_d=den["VAR_EM_d"] * (num.value / den.value **2) **2,
                                        VAR_PM_d=den["VAR_PM_d"] * (num.value / den.value **2) **2,
                                        VAR_1GXS=den["VAR_1GXS"] * (num.value / den.value **2) **2,
                                        VAR_C_n=vcn,
                                        VAR_C_d=vcd,
                                        VAR_C=num["VAR_C"] / den.value **2)
            else:
                df = _make_df((v - 1) * 100, u * 100, relative=False).assign(
                                        VAR_FFS_n=den["VAR_FFS_n"] * (num.value / den.value **2 * 100) **2,
                                        VAR_EM_n=den["VAR_EM_n"] * (num.value / den.value **2 * 100) **2,
                                        VAR_PM_n=den["VAR_PM_n"] * (num.value / den.value **2 * 100) **2,
                                        VAR_FFS_d=den["VAR_FFS_d"] * (num.value / den.value **2 * 100) **2,
                                        VAR_EM_d=den["VAR_EM_d"] * (num.value / den.value **2 * 100) **2,
                                        VAR_PM_d=den["VAR_PM_d"] * (num.value / den.value **2 * 100) **2,
                                        VAR_1GXS=den["VAR_1GXS"] * (num.value / den.value **2 * 100) **2,
                                        VAR_C_n=vcn if vcn is None else vcn * 100 **2,
                                        VAR_C_d=vcd if vcd is None else vcd * 100 **2,
                                        VAR_C=num["VAR_C"] / den.value **2 * 100 **2)
        else:
            df = _make_df(v, u) if not _minus_one_percent else _make_df((v - 1) * 100, u * 100, relative=False)
        return df

    def _compute_traverse(self, _minus_one_percent=False, normalization=None, **kwargs):
        n = self.num.calculate(normalization=normalization)
        d = self.den.process(normalization=normalization, **kwargs)
        out = []
        for t in d.traverse:
            v, u = ratio_v_u(n.query("traverse == @t").reset_index(),
                             d.query("traverse == @t").reset_index())
            v, u = v.iloc[0], u.iloc[0]
            out.append(_make_df(v, u).assign(traverse=t) if not _minus_one_percent else
                       _make_df((v - 1) * 100 , u * 100, relative=False).assign(traverse=t))
        return pd.concat(out, ignore_index=True)

    def compute(self, _minus_one_percent=False, *args, normalization: str =None,
                **kwargs) -> pd.DataFrame:
        """
        Computes the comparison value.

        Parameters
        ----------
        _minus_one_percent : bool, optional
            computes the C/E-1 [%]. Defaults to False.
        *args : Any
            Positional arguments to be passed to the `SpectralIndex.process()` method.
        normalization : str, optional
            The detector name to normalize the traveres to.
            Defaults to None, normalizing to the one with the highest counts.
            Will be used to compute traverse C/E for normalization of both C and E.
        **kwargs : Any
            Keyword arguments to be passed to the `SpectralIndex.process()` method.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the comparison value.

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
        if isinstance(self.num, CalculatedSpectralIndex):
            out = self._compute_si(_minus_one_percent, **kwargs)
        if isinstance(self.num, CalculatedTraverse):
            out = self._compute_traverse(_minus_one_percent, normalization=normalization, **kwargs)
        return out

    def minus_one_percent(self, **kwargs):
        """
        Computes the comparison value and subtracts 1, adjusting the uncertainty accordingly.
        The result is in units of %.

        Parameters
        ----------
        *args : Any
            Positional arguments to be passed to the `SpectralIndex.process()` method.
        **kwargs : Any
            Keyword arguments to be passed to the `SpectralIndex.process()` method.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the adjusted comparison value.

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


@dataclass(slots=True)
class CoverE(_Comparison):
    num: _Calculated  # calculation
    den: _Experimental  # experiment

    def __post_init__(self):
        self._check_consistency()

    def _check_consistency(self) -> None:
        _Comparison(self.num, self.den)._check_consistency_attrs()
        if ((isinstance(self.num, CalculatedTraverse) and not isinstance(self.den, Traverse)) or
            (isinstance(self.den, Traverse) and not isinstance(self.num, CalculatedTraverse))):
            raise Exception("Cannot compare Traverse and non-Traverse object.")
        if (isinstance(self.num, CalculatedSpectralIndex) and not isinstance(self.den, SpectralIndex) or
            isinstance(self.den, SpectralIndex) and not isinstance(self.num, CalculatedSpectralIndex)):
            raise Exception("Cannot compare SpectralIndex and non-SpectralIndex object.")


@dataclass(slots=True)
class CoverC(_Comparison):
    num: _Calculated  # calculation
    den: _Calculated  # calculation

    def __post_init__(self):
        self._check_consistency()

    def _check_consistency(self) -> None:
        _Comparison(self.num, self.den)._check_consistency_attrs()
        if isinstance(self.num, CalculatedTraverse) and not isinstance(self.den, CalculatedTraverse):
            raise Exception("Cannot compare Traverse and non-Traverse object.")
        if isinstance(self.num, CalculatedSpectralIndex) and not isinstance(self.den, CalculatedSpectralIndex):
            raise Exception("Cannot compare SpectralIndex and non-SpectralIndex object.")
