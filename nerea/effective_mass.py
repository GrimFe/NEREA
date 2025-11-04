from typing import Self
from dataclasses import dataclass
from datetime import date
import pandas as pd

__all__ = ["EffectiveMass"]

@dataclass(slots=True)
class EffectiveMass:
    """
    ``nerea.EffectiveMass``
    =======================
    Class storing effective mass data from fission chamber calibration.

    Attributes
    ----------
    **deposit_id** : ``str``,
        metadata for fission chamber deposit.
    **detector_id** : ``str``
        metadata for detector identification.
    **data** : ``pd.DataFrame``
        effective mass data.
    **bins** : ``int``
        number of acquisition bins used in the calibration
        acquisition.
    **composition** : ``pd.DataFrame``, optional
        a nerea-formatted data frame with the fission chamber
        composition value and uncertainty. Default is ``None`` for
        monoisotopic chambers."""
    deposit_id: str
    detector_id: str
    data: pd.DataFrame
    bins: int
    composition: pd.DataFrame = None

    @property
    def composition_(self) -> pd.DataFrame:
        """
        `nerea.EffectiveMass.composition_()`
        ------------------------------------
        The material composition of the fission chamber.

        Returns
        -------
        ``pd.DataFrame``
            the material composition nuclide by nuclide (index) with
            ``'value'``  and ``'uncertainty'`` columns."""
        data = pd.DataFrame({'nuclide': [self.deposit_id],
                             'value': [1],
                             'uncertainty': [0]}).set_index('nuclide')
        return data if self.composition is None else self.composition.reset_index().set_index('nuclide')[['value', 'uncertainty']]

    @property
    def R_channel(self) -> int:
        """
        `nerea.EffectiveMass.R_channel()`
        ---------------------------------
        Calculates the channel where half maximum of the pulse height spectrum
        was found during the calibration.

        Returns
        -------
        ``int``
            The channel of the calibration half maximum."""
        return int(self.integral.channel.iloc[0] / 0.15)

    @property
    def integral(self) -> pd.DataFrame:
        """
        `nerea.EffectiveMass.integral()`
        ------------------------------------
        Computes the EffectiveMass values. Alias for self.data.

        Returns
        -------
        ``pd.DataFrame``
            dataframe with ``'value'`` and ``'uncertainty'``columns.

        Note
        ----
        - alias for ``self.data``
        """
        return self.data

    def to_xlsx(self, file_path: str) -> None:
        """
        `nerea.EffectiveMass.to_xslx()`
        -------------------------------
        Writes the effective mass to a formatted excel file.

        Parameters
        ----------
        **file** : ``str``
            the file name to write the instance to.

        Returns
        -------
            ``None``

        Note
        ----
        - for nerea read/write performance, filename should be {Deposit}_{Detector}.xlsx

        Example
        -------
        >>> em = EffectiveMass.from_xlsx(file_path)
        >>> em.to_xlsx(file_path1)"""
        slash = '' if file_path == '' else '\\'
        if '.xls' in file_path or '.xls' in file_path:
            file = file_path
        else:
            file = file_path + slash + f'Meff_{self.deposit_id}_{self.detector_id}.xlsx'
        with pd.ExcelWriter(file) as writer:
            self.data.to_excel(writer, index=False, sheet_name='Meff')
            self.composition_.to_excel(writer, index=True, sheet_name='Composition')
            pd.DataFrame({'R': [self.R_channel], 'bins': [self.bins]}
                         ).T.to_excel(writer, header=False, index=False, sheet_name='R')
            pd.DataFrame({'c': [f'Calibrated on {date.today()} using nerea.py']}
                         ).T.to_excel(writer, header=False, index=False, sheet_name='_CalData')

    @classmethod
    def from_xlsx(cls, file: str) -> Self:
        """
        `nerea.EffectiveMass.from_xlsx()`
        ---------------------------------
        Reads data from an Excel file and extracts deposit and detector ID from the file name.
        The filename is expected to be formatted as:
        {Deposit}_{Detector}.xlsx

        Parameters
        ----------
        **file** : ``str``
            File path of an Excel file containing the effective mass data.

        Returns
        -------
        ``nerea.EffectiveMass``
            Effective mass instance.

        Examples
        --------
        >>> eff_mass = EffectiveMass.from_xlsx('filename.xlsx')"""
        _, deposit_id, detector_id = file.split('\\')[-1].replace('.xlsx','').replace('.xls','').split('_')
        integral = pd.read_excel(file, sheet_name='Meff')
        bins = pd.read_excel(file, sheet_name='R', header=None).iloc[1][0]
        try:
            composition = pd.read_excel(file, sheet_name='Composition')
        except ValueError:
            composition = None
        return cls(deposit_id, detector_id, integral, bins=bins, composition=composition)
