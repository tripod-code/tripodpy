'''``TwoPopPy`` is a package for simulating the evolution of protoplanetary disks.

The class for performing simulations is ``twopoppy.Simulation``.

Furthermore, ``TwoPopPy`` contains a simple package for plotting data ``twopoppy.plot`` and a package
that contains pre-defined standard functions ``twopoppy.std`` that can be used in the simulation.

``twopoppy.utils`` contains some helper classes/functions that are used within the simulation.

``TwoPopPy`` is mainly written in ``Python``. Computation intensive calculations are written in ``Fortran``.

``TwoPopPy`` is using the ``simframe`` package for setting up scientific simulations.'''

from dustpy import constants
from twopoppy.simulation import Simulation
from dustpy import utils
from dustpy.utils import hdf5writer

from simframe.io.dump import readdump
from simframe.io.writers import hdf5writer

__name__ = "twopoppy"
__version__ = "3.0.0"

Simulation.__version__ = __version__

__all__ = ["constants",
           "hdf5writer",
           "readdump",
           "Simulation"]
