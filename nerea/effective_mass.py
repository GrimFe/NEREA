from dataclasses import dataclass
from datetime import date
import pandas as pd

__all__ = ["EffectiveMass"]

@dataclass(slots=True)
class EffectiveMass:
    deposit_id: str
    detector_id: str
    data: pd.DataFrame
    bins: int
    composition: pd.DataFrame = None

    @property
    def composition_(self) -> pd.DataFrame:
        """
        The material composition of the fission chamber.

        Returns
        -------
        pd.DataFrame
            the material composition nuclide by nuclide with
            absolute uncertainty.
        """
        data = pd.DataFrame({'nuclide': [self.deposit_id],
                             'value': [1],
                             'uncertainty': [0]}).set_index('nuclide')
        return data if self.composition is None else self.composition.reset_index().set_index('nuclide')[['value', 'uncertainty']]

    @property
    def R_channel(self) -> int:
        """
        Calculates the channel where half maximum of the pulse height spectrum
        was found during the calibration.

        Returns
        -------
        int
            The channel of the calibration half maximum.

        Examples
        --------
        >>> eff_mass = EffectiveMass(...)
        >>> channel = eff_mass.R_channel
        """
        return int(self.integral.channel.iloc[0] / 0.15)

    @property
    def integral(self) -> pd.DataFrame:
        """
        Computes the EffectiveMass values. Alias for self.data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the EffectiveMass values.

        Examples
        --------
        """
        return self.data

    def to_xlsx(self, file_path: str) -> None:
        """
        Writes the effective mass to a formatted excel file.

        Parameters
        ----------
        file : str
            the file name to write the instance to.

        Returns
        -------
            None

        Example
        -------
        >>> em = EffectiveMass.from_xlsx(file_path)
        >>> em.to_xlsx(file_path1)
        """
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
    def from_xlsx(cls, file: str):
        """
        Reads data from an Excel file and extracts deposit and detector ID from the file name.
        The filename is expected to be formatted as:
        {Deposit}_{Detector}.xlsx

        Parameters
        ----------
        file : str
            File path of an Excel file containing the effective mass data.

        Returns
        -------
        EffectiveMass
            Effective mass instance.

        Examples
        --------
        >>> eff_mass = EffectiveMass.from_xlsx('filename.xlsx')
        """
        _, deposit_id, detector_id = file.split('\\')[-1].replace('.xlsx','').replace('.xls','').split('_')
        integral = pd.read_excel(file, sheet_name='Meff')
        bins = pd.read_excel(file, sheet_name='R', header=None).iloc[1][0]
        try:
            composition = pd.read_excel(file, sheet_name='Composition')
        except ValueError:
            composition = None
        return cls(deposit_id, detector_id, integral, bins=bins, composition=composition)
