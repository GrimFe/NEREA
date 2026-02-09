from .classes import *

from .pulse_height_spectrum import *
from .effective_mass import *
from .count_rate import *

from .experimental import *

from .calculated import *

from .comparisons import *

from .control_rod import *

from .logging_config import setup_logging
import logging
import os

setup_logging()  # sets up once
logger = logging.getLogger(__name__)

__version__ = '0.1.0'

logger.info(f"Starting NEREA version {__version__} from {__file__}")
