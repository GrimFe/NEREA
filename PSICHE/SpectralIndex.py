import pandas as pd
import numpy as np
import serpentTools as sts  ## impurity correction
import warnings
from dataclasses import dataclass

from PSICHE.ReactionRate import ReactionRate, AverageReactionRate
from PSICHE.utils import ratio_v_u, _make_df

@dataclass
class SpectralIndex:
    numerator: ReactionRate | AverageReactionRate
    denominator: ReactionRate | AverageReactionRate

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
        >>> from PSICHE.ReactionRate import ReactionRate
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
        # normalize impurities and one group xs
        imp_v, imp_u = ratio_v_u(imp,
                                 imp.loc[self.numerator.deposit_id])
        xs_v, xs_u = ratio_v_u(one_g_xs,
                               one_g_xs.loc[self.denominator.deposit_id])
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
        return _make_df(correction, np.sqrt(correction_variance))

    def compute(self, one_g_xs: pd.DataFrame = None,
                one_g_xs_file: dict[str, tuple[str, str]] = None, *args, **kwargs) -> pd.DataFrame:
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
            Positional arguments to be passed to the `ReactionRate.compute()` method.
        **kwargs : Any
            Keyword arguments to be passed to the `ReactionRate.compute()` method.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the spectral index value and uncertainty.

        Examples
        --------
        >>> from PSICHE.ReactionRate import ReactionRate
        >>> ffs_num = ReactionRate(..., deposit_id='Dep1')
        >>> ffs_den = ReactionRate(..., deposit_id='Dep2')
        >>> spectral_index = SpectralIndex(numerator=ffs_num, denominator=ffs_den)
        >>> spectral_index.compute()
            value  uncertainty
        0  0.95   0.034139
        """
        v, u = ratio_v_u(self.numerator.compute(*args, **kwargs),
                         self.denominator.compute(*args, **kwargs))
        if (one_g_xs is None and one_g_xs_file is None
            and self.numerator.effective_mass.composition_.shape[0] > 1):
            warnings.warn("Impurities in the fission chambers require one group xs" +\
                          " to be accounted for.")
        if one_g_xs_file is not None:
            read = pd.DataFrame({nuc: sts.read(det[1]).detectors[det[0]].bins[0][-2:]
                                 } for nuc, det in one_g_xs_file).T
            read.columns = ['value', 'uncertainty']
            read.index.name = 'nuclide'
            # uncertainy is absolute
            read.uncertainty = read.uncertainty * read.value
        else:
            read = None
        one_g_xs_ = read if one_g_xs is None else one_g_xs
        if one_g_xs_ is not None:
            c = self._compute_correction(one_g_xs_)
            v = v - c.value
            u = np.sqrt(u **2 - c.uncertainty **2)
        return _make_df(v, u)
