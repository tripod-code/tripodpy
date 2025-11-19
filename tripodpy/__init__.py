# from tripodpy import plot
from tripodpy.simulation import Simulation
from tripodpy import constants
from tripodpy import plot
from tripodpy import utils

from simframe.io.dump import readdump
from simframe.io.writers import hdf5writer

from importlib import metadata as _md

__name__ = "tripodpy"
__version__ = _md.version("tripodpy")

Simulation.__version__ = __version__
utils.__version__ = __version__

__all__ = [
    "constants",
    "hdf5writer",
    "plot",
    "readdump",
    "Simulation"
]
