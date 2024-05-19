from dataclasses import dataclass
import pandas as pd

__all__ = ["EffectiveMass"]

@dataclass
class EffectiveMass:
    deposit_id: str
    detector_id: str
    integral: pd.DataFrame
    bins: int

    @property
    def R_channel(self) -> int:
        """
        Calculates the channel where half maximum of the fission fragment spectrum
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
        return int(self.integral.channel[0] / 0.15)

    @classmethod
    def from_xls(cls, file: str):
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
        >>> eff_mass = EffectiveMass.from_xls('filename.xlsx')
        """
        _, deposit_id, detector_id = file.split('\\')[-1].replace('.xlsx','').replace('.xls','').split('_')
        integral = pd.read_excel(file)
        return cls(deposit_id, detector_id, integral, bins=4096)
