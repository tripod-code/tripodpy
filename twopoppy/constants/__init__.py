'''Package containing constants that are used in the simulations. All constants
are in cgs units.

The constants are defined in ``twopoppy/constants/constants.f90`` so they can be used in
the ``Fortran`` scripts.

Constants
---------
| ``k_B`` : Boltzmann constant
| ``pi`` : Ratio of circle's circumference to diameter
'''

import numpy as _np

import twopoppy.constants._constants_f as _c

k_B = float(_c.constants.k_B)
pi = float(_c.constants.pi)

__all__ = [
    "k_B",
    "pi"
]
