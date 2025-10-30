from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import _Experimental, SpectralIndex, Traverse
from . import _Calculated, CalculatedSpectralIndex, CalculatedTraverse
from .utils import ratio_v_u, _make_df


__all__ = ['CoverE', 'CoverC', 'FrameCompare']


def _frame_comparison(num: pd.DataFrame, den: pd.DataFrame,
                      _minus_one_percent: bool=False):
    """
    Ratio comparison of `pd.DataFrame` objects.

    Parameters
    ----------
    num: pd.DataFrame
        the ratio numerator.
    den: pd.DataFrame
        the ratio denominator.
    _minus_one_percent: bool, optional
        flag to determine whether the comparison should be
        as N / D - 1 [%]. Defaults to False.
    
    Returns
    -------
    df: pd.DataFrame
        the result in the standard format of `nerea.utils._make_df()`.
    var_num: pd.DataFrame
        variance apportioning of the numerator.
    var_den: pd.DataFrame
        variance apportioning of the denominator.
    """
    v, u = ratio_v_u(num, den)
    df = _make_df((v - 1) * 100, u * 100, relative=False) if _minus_one_percent else _make_df(v, u)
    # sensitivities
    S_num, S_den = 1 / den.value, num.value / den.value **2
    factor = 100 if _minus_one_percent else 1
    # variances
    var_cols_num = [c for c in num.columns if c.startswith("VAR_PORT")]
    var_num = (num[var_cols_num] * (S_num.value * factor) **2).replace({np.nan: None})
    var_cols_den = [c for c in den.columns if c.startswith("VAR_PORT")]
    var_den = den[var_cols_den] * (S_den.value * factor) **2
    # Handling C where all variances are None
    if 'VAR_PORT_C_n' in var_cols_num and all([num[i].value is None for i in var_cols_num]):
        var_num['VAR_PORT_C'] = (num["uncertainty"].value * S_num.value * factor) **2
    if 'VAR_PORT_C_n' in var_cols_den and all([den[i].value is None for i in var_cols_den]):
        var_den['VAR_PORT_C'] = (den["uncertainty"].value * S_den.value * factor) **2
    # Disambiguation of the numerator and denominator column names if C/C
    if 'VAR_PORT_C_n' in num.columns and 'VAR_PORT_C_n' in den.columns:  # if C/C
        var_num.columns = [f'{c}_n' for c in var_num.columns]
        var_den.columns = [f'{c}_d' for c in var_den.columns]
        var_den = var_den.replace({np.nan: None})
    return df, var_num, var_den


@dataclass(slots=True)
class _Comparison:
    """
    Comparison superclass handling calculation of the
    comparison ratio.

    Attributes:
    -----------
    num: nerea._Calculated
        the calculated quantity to use as numerator.
    den: nerea._Calculated | nerea._Experimental
        the calculated or measured quantity to use
        as denominator.
    """
    num: _Calculated
    den: _Experimental | _Calculated

    def _check_consistency_attrs(self) -> None:
        """"
        Checks consistency of `deposit_id` or `deposit_ids`.
        """
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

    def _get_denominator(self, **kwargs) -> pd.DataFrame:
        """
        Processess the comparison denominator discrimiating
        between Experimental and Calculated types.

        Parameters
        ----------
        kwargs: any
            keyword arguments for experimental denominator
            processing.

        Returns
        -------
        pd.DataFrame
            The processed / calculated denominator.
        """
        return self.den.process(**kwargs) if isinstance(self.den,
                                                        _Experimental) else self.den.calculate()

    def _compute_si(self, _minus_one_percent: bool=False, **kwargs) -> pd.DataFrame:
        """
        Computes the C/E value for spectral indices.

        Parameters
        ----------
        _minus_one_percent: bool, optional
            flag to compute C/E - 1 [%]. Default is False.
        **kwargs: any
            keyword arguments for denominator (experimental)
            spectral index processing.

        Returns
        -------
        pd.DataFrame
            the C/E value and uncertainty
        """
        num, den = self.num.calculate(), self._get_denominator(**kwargs)
        df, var_num, var_den = _frame_comparison(num, den, _minus_one_percent)
        return pd.concat([df, var_num, var_den], axis=1)

    def _compute_traverse(self, _minus_one_percent: bool=False,
                          normalization: int|str=None, **kwargs) -> pd.DataFrame:
        """
        Computes the C/E value for traverses.

        Parameters
        ----------
        _minus_one_percent: bool, optional
            flag to compute C/E - 1 [%]. Default is False
        normalization: int|str, optional
            The point to normalize the traverse to. Default is
            None, normalizing to the one with the highest counts.
        **kwargs: any
            keyword arguments for denominator (experimental)
            traverse processing.

        Returns
        -------
        pd.DataFrame
            the C/E value and uncertainty
        """
        n = self.num.calculate(normalization=normalization).set_index('traverse')
        d = self.den.process(normalization=normalization, **kwargs).set_index('traverse')
        v, u = ratio_v_u(n, d)
        if not _minus_one_percent:
            out = _make_df(v, u, idx=v.index)
        else:
            out = _make_df((v - 1) * 100 , u * 100, relative=False, idx=v.index)
        return out.reset_index(names='traverse')[['value', 'uncertainty', 'uncertainty [%]', 'traverse']]

    def compute(self, _minus_one_percent: bool=False, *args, normalization: str =None,
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
        """
        if isinstance(self.num, CalculatedSpectralIndex):
            out = self._compute_si(_minus_one_percent, **kwargs)
        if isinstance(self.num, CalculatedTraverse):
            out = self._compute_traverse(_minus_one_percent, normalization=normalization, **kwargs)
        return out

    def minus_one_percent(self, **kwargs) -> pd.DataFrame:
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
        """
        return self.compute(**kwargs, _minus_one_percent=True)


@dataclass(slots=True)
class CoverE(_Comparison):
    """
    Calculates the C/E inheriting from `nerea._Comparison`.

    Attributes:
    -----------
    num: nerea._Calculated
        the calculated quantity to use as numerator.
    den: nerea._Experimental
        the measured quantity to use as denominator.
    _enable_checks: bool, optional
        flag to enable consistency checks.
        Default is `True`.
    """
    num: _Calculated  # calculation
    den: _Experimental  # experiment
    _enable_checks: bool = True

    def __post_init__(self) -> None:
        """
        Runs consistency checks.
        """
        if self._enable_checks:
            self._check_consistency()

    def _check_consistency(self) -> None:
        """
        Checks consistency of C and E types and runs _Comparison checks.
        """
        _Comparison(self.num, self.den)._check_consistency_attrs()
        if ((isinstance(self.num, CalculatedTraverse) and not isinstance(self.den, Traverse)) or
            (isinstance(self.den, Traverse) and not isinstance(self.num, CalculatedTraverse))):
            raise Exception("Cannot compare Traverse and non-Traverse object.")
        if (isinstance(self.num, CalculatedSpectralIndex) and not isinstance(self.den, SpectralIndex) or
            isinstance(self.den, SpectralIndex) and not isinstance(self.num, CalculatedSpectralIndex)):
            raise Exception("Cannot compare SpectralIndex and non-SpectralIndex object.")


@dataclass(slots=True)
class CoverC(_Comparison):
    """
    Calculates the C/C inheriting from `nerea._Comparison`.

    Attributes:
    -----------
    num: nerea._Calculated
        the calculated quantity to use as numerator.
    den: nerea._Calculated
        the calculated quantity to use as denominator.
    _enable_checks: bool, optional
        flag to enable consistency checks.
        Default is `True`.
    """
    num: _Calculated  # calculation
    den: _Calculated  # calculation
    _enable_checks: bool = True

    def __post_init__(self) -> None:
        """
        Runs consistency checks.
        """
        if self._enable_checks:
            self._check_consistency()

    def _check_consistency(self) -> None:
        """
        Checks consistency of C and E types and runs _Comparison checks.
        """
        _Comparison(self.num, self.den)._check_consistency_attrs()
        if isinstance(self.num, CalculatedTraverse) and not isinstance(self.den, CalculatedTraverse):
            raise Exception("Cannot compare Traverse and non-Traverse object.")
        if isinstance(self.num, CalculatedSpectralIndex) and not isinstance(self.den, CalculatedSpectralIndex):
            raise Exception("Cannot compare SpectralIndex and non-SpectralIndex object.")


@dataclass(slots=True)
class FrameCompare:
    """
    A class to compute the ratio comparison of two
    `pd.DataFrame` instances.
    Each data frame should be nerea-formatted, with value
    and uncertainty columns.

    Attributes:
    -----------
    num: pd.DataFrame
        the numerator.
    den: pd.DataFrame
        the denominator.
    """
    num: pd.DataFrame
    den: pd.DataFrame

    def compute(self, _minus_one_percent=False) -> pd.DataFrame:
        """
        Computes the comparison value as ratio of `num` and `den`.

        Parameters
        ----------
        _minus_one_percent : bool, optional
            computes the C/E-1 [%]. Defaults to False.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the comparison value.
        """
        return pd.concat(_frame_comparison(self.num, self.den, _minus_one_percent),
                         axis=1)
