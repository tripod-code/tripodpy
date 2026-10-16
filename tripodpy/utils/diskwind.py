'''Helper functions for MHD disk wind-driven disk evolution (Tabone et al. 2022).'''

import numpy as np


def split_alpha(alpha_tilde, psi):
    """Function splits the total alpha parameter into the turbulent and the disk wind part
    (Tabone et al. 2022, Eqs. 12-13):

    alpha_tilde = alpha_SS + alpha_DW,  psi = alpha_DW / alpha_SS

    Parameters
    ----------
    alpha_tilde : float or array-like
        Total alpha parameter (turbulent + disk wind)
    psi : float or array-like
        Ratio of disk wind to turbulent alpha. ``psi = 0`` is a purely viscous disk,
        ``psi = np.inf`` is a purely wind-driven disk.

    Returns
    -------
    alpha_SS : float or array-like
        Turbulent (viscous) alpha parameter
    alpha_DW : float or array-like
        Disk wind alpha parameter

    Notes
    -----
    Typical usage before ``sim.initialize()``:

    ``sim.ini.gas.alpha, sim.ini.gas.alphaDiskwind = split_alpha(1.e-3, 1.)``

    The dust mixing parameters ``sim.dust.delta.*`` are initialized with
    ``sim.ini.gas.alpha``, i.e. with alpha_SS. For purely wind-driven disks
    (``psi = np.inf``) they are zero and have to be set explicitly."""
    alpha_tilde = np.asarray(alpha_tilde, dtype=float)
    psi = np.asarray(psi, dtype=float)
    if np.any(psi < 0.):
        raise ValueError("psi has to be non-negative.")
    if np.any(alpha_tilde < 0.):
        raise ValueError("alpha_tilde has to be non-negative.")
    # 1/(1+psi) is well defined for psi = inf
    f_SS = 1. / (1. + psi)
    alpha_SS = alpha_tilde * f_SS
    alpha_DW = alpha_tilde * (1. - f_SS)
    if alpha_SS.ndim == 0:
        return float(alpha_SS), float(alpha_DW)
    return alpha_SS, alpha_DW
