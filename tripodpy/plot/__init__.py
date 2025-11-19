from tripodpy.plot.plot import ipanel
from tripodpy.plot.plot import panel

from importlib import metadata as _md

__all__ = [
    "ipanel",
    "panel"
]
__version__ = _md.version("tripodpy")
