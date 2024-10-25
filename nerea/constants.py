import pandas as pd
import numpy as np

from .utils import _make_df

AVOGADRO = 6.02214076e23

# he isotopic mass data is from G. Audi, A. H. Wapstra Nucl. Phys A. 1993, 565, 1-65 and G. Audi, A. H. Wapstra Nucl. Phys A. 1995,
# 595, 409-480. The percent natural abundance data is from the 1997 report of the IUPAC Subcommittee for Isotopic
# Abundance Measurements by K.J.R. Rosman, P.D.P. Taylor Pure Appl. Chem. 1999, 71, 1593-1607.
# WHEN AVAILABLE, nucleon number otherwise.
ATOMIC_MASS = pd.DataFrame(
    {"U233": [233.0, 0.0],
     "U234": [234.040916, 0.0],
     "U235": [235.043923, 0.0],
     "U236": [236.0, 0.0],
     "U238": [238.050783, 0.0],
     "Np237": [237.048167, 0.0],
     "Pu238": [238.0, 0.0],
     "Pu239": [239.0521634, 0.0],
     "Pu240": [240.053807, 0.0],
     "Pu241": [241.0, 0.0],
     "Pu242": [242.0, 0.0],
     "Am241": [241.0, 0.0]
     }, index=['value', 'uncertainty'])

KNBS = {"BR1-MARK3": _make_df(8703., 0.02 * 8703.),
        "BR1-EMPTY CAVITY": _make_df(25456., 0.021 * 25456.)}

## MISSING UNCERTAINTIES
XS_FAST = pd.DataFrame({"value": [72.88, 1133.12, 1489.03, 572.23, 284.95, 1264.48,
                                  1971.88, 2132.61, 1308.2, np.nan, 1115.87, 1321.81,
                                  1024.24],   ## fast xs JEFF-3.1.1 [b]
                        "uncertainty": [0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0.,
                                  0.]   ## to be computed [b]
                        }, index=["Th223", "U234", "U235", "U236", "U238", "Np237",
                                  "Pu238", "Pu239", "Pu240", "Pu241", "Pu242", "Am241",
                                  "Am243"])

XS_TH = pd.DataFrame({"value": [np.nan, 0.0669646, 584.977, 0.0613027, 2.65118E-05, 0.0180149,
                                17.8823, 746.995, 0.0591624, 1012.26, 2.55745E-03, 3.15064,
                                0.0813315],  ## thermal xs JEFF-3.1.1 [b]
                      "uncertainty": [0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0.,
                                      0.]   ## to be computed [b]
                     }, index=["Th223", "U234", "U235", "U236", "U238", "Np237",
                               "Pu238", "Pu239", "Pu240", "Pu241", "Pu242", "Am241",
                               "Am243"])

XS_MAXWELIAN = pd.DataFrame({"value": XS_TH.value * 
                             [np.nan, 0.99044, 0.97648, 1.00239, 1.00127, 0.98419,
                              0.95731, 1.05023, 1.033, 1.04697, 1.00605, 1.05867,
                              1.01289],  ## Westcott factor
                             "uncertainty": [0., 0., 0., 0., 0., 0.,
                                             0., 0., 0., 0., 0., 0.,
                                             0.]   ## to be computed [b]
                            }, index=["Th223", "U234", "U235", "U236", "U238", "Np237",
                                      "Pu238", "Pu239", "Pu240", "Pu241", "Pu242", "Am241",
                                      "Am243"])
import pandas as pd
import numpy as np

from .utils import _make_df

AVOGADRO = 6.02214076e23

# The isotopic mass data is from G. Audi, A. H. Wapstra Nucl. Phys A. 1993, 565, 1-65 and G. Audi, A. H. Wapstra Nucl. Phys A. 1995,
# 595, 409-480. The percent natural abundance data is from the 1997 report of the IUPAC Subcommittee for Isotopic
# Abundance Measurements by K.J.R. Rosman, P.D.P. Taylor Pure Appl. Chem. 1999, 71, 1593-1607.
# WHEN AVAILABLE, nucleon number otherwise.
ATOMIC_MASS = pd.DataFrame(
    {"U233": [233.0, 0.0],
     "U234": [234.040916, 0.0],
     "U235": [235.043923, 0.0],
     "U236": [236.0, 0.0],
     "U238": [238.050783, 0.0],
     "Np237": [237.048167, 0.0],
     "Pu238": [238.0, 0.0],
     "Pu239": [239.0521634, 0.0],
     "Pu240": [240.053807, 0.0],
     "Pu241": [241.0, 0.0],
     "Pu242": [242.0, 0.0],
     "Am241": [241.0, 0.0]
     }, index=['value', 'uncertainty'])
