import pandas as pd
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
            - experiment_id
            - location_id
        among `self.numerator` and `self.denominator`.

        Raises
        ------
        UserWarning
            If there are inconsistencies among the specified attributes.
        """
        should = ['campaign_id', 'experiment_id', 'location_id']
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

    def compute(self, *args, **kwargs) -> pd.DataFrame:
        """
        Computes the ratio of two reaction rates.

        Parameters
        ----------
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
        return _make_df(v, u)    
