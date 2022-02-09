'''Module containing standard functions for the dust.'''


import dustpy.constants as c
from twopoppy.std import dust_f

import numpy as np
# import scipy.sparse as sp

# from simframe.integration import Scheme


def a(sim):
    """Function calculates the particle size from the specific particle sizes and the distribution exponent.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    a : Field
        Particle sizes"""
    return dust_f.a(sim.dust.s.amin, sim.dust.s.amax, sim.dust.s.aint, sim.dust.xi.calc, sim.grid._Nm_long)


def m(sim):
    """Function calculates the particle mass from the particle sizes.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    m : Field
        Particle masses"""
    return dust_f.m(sim.dust.a, sim.dust.rhos, sim.dust.fill)


def p_frag(sim):
    """Function calculates the fragmentation probability.
    It assumes a linear transition between sticking and
    fragmentation.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    pf : Field
        Fragmentation propability."""
    return dust_f.pfrag(sim.dust.v.rel.tot, sim.dust.v.frag)


def p_stick(sim):
    """Function calculates the sticking probability.
    The sticking probability is simply 1 minus the
    fragmentation probability.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    ps : Field
        Sticking probability"""
    p = 1. - sim.dust.p.frag
    p[0] = 0.
    p[-1] = 0.
    return p


def rho_midplane(sim):
    """Function calculates the midplane mass density.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    rho : Field
        Midplane mass density"""
    # The scale height H has a longer shape than Sigma and has to be adjusted
    return sim.dust.Sigma / (np.sqrt(2 * c.pi) * sim.dust.H[:, :2])


def Sigma_initial(sim):
    """Function calculates the initial condition fo the dust surface densities

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    Sigma : Field
        Initial dust surface density"""
    xi = sim.dust.xi.calc
    xip4 = xi + 4.
    sint = sim.dust.s.aint
    smax = sim.dust.s.amax
    smin = sim.dust.s.amin

    # Values for xi != -4
    S0 = (sint**xip4 - smin**xip4) / (smax**xip4 - smin**xi+4)
    S1 = 1. - S0
    S = np.array([S0, S1]).T

    # Values for xi == -4
    S0_4 = np.log(sint/smin) / np.log(smax/smin)
    S1_4 = 1. - S0_4
    S_4 = np.array([S0_4, S1_4]).T

    Sigma = sim.ini.dust.d2gRatio * sim.gas.Sigma[:, None] * \
        np.where(xi[:, None] == -4., S_4, S)

    return Sigma


def S_tot(sim, Sigma=None):
    """Function calculates the total source terms.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    Sigma : Field, optional, default : None
        Surface density to be used if not None

    Returns
    -------
    Stot : Field
        Total source terms of surface density"""
    Sext = sim.dust.S.ext
    if Sigma is None:
        Sigma = sim.dust.Sigma
        Shyd = sim.dust.S.hyd
    else:
        Shyd = sim.dust.S.hyd.updater.beat(sim, Sigma=Sigma)
        if Shyd is None:
            Shyd = sim.dust.S.hyd
    return Shyd + Sext


def vrel_brownian_motion(sim):
    """Function calculates the relative particle velocities due to Brownian motion.
    The maximum value is set to the sound speed.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    vrel : Field
        Relative velocities"""
    return dust_f.vrel_brownian_motion(sim.gas.cs, sim.dust.m, sim.gas.T)


def xicalc(sim):
    """Function calculates the exponent of the distribution.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    xicalc : Field
        Calculated exponent of distribution"""
    return dust_f.xicalc(sim.dust.Sigma, sim.dust.s.amax, sim.dust.s.aint)


def aint(sim):
    """Function calculates the intermediate particle size.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    aint : Field
        Intermediate particle size"""
    return dust_f.aint(sim.dust.s.amin, sim.dust.s.amax)
