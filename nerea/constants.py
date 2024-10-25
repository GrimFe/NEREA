import pandas as pd
import numpy as np

from .utils import _make_df

AVOGADRO = 6.02214076e23

# he isotopic mass data is from G. Audi, A. H. Wapstra Nucl. Phys A. 1993, 565, 1-65 and G. Audi, A. H. Wapstra Nucl. Phys A. 1995,
# 595, 409-480. The percent natural abundance data is from the 1997 report of the IUPAC Subcommittee for Isotopic
# Abundance Measurements by K.J.R. Rosman, P.D.P. Taylor Pure Appl. Chem. 1999, 71, 1593-1607.
# WHEN AVAILABLE
ATOMIC_MASS = {"U235": 235.043923,
               "U238": 238.050783,
               "Np237": 237.048167,
               "Pu239": 239.0521634,
               "Pu240": 240.053807}
