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
    return dust_f.a(sim.dust.size.min, sim.dust.size.max, sim.dust.size.int, sim.dust.exp.calc)
  
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
  
  def expcalc(sim):
    """Function calculates the exponent of the distribution.
    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    Returns
    -------
    expcalc : Field
        Calculated exponent of distribution"""
    return dust_f.expcalc(sim.dust.Sigma, sim.dust.size.max, sim.dust.size.int)
  
  def sizeint(sim):
    """Function calculates the intermediate particle size.
    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    Returns
    -------
    sizeint : Field
        Intermediate particle size"""
    return dust_f.sizeint(sim.dust.size.min, sim.dust.size.max)
  
  def sizemean(sim):
    """Function calculates the mass-averaged particle size.
    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    Returns
    -------
    sizeint : Field
        Mass-averaged particle size"""
    return dust_f.sizemean(sim.dust.size.min, sim.dust.size.max, sim.dust.exp.calc)
  
  # TODO: Adjust function to Twopoppy
  def MRN_distribution(sim):
    """Function calculates the initial particle mass distribution. The parameters are taken from the
    ``Simulation.ini`` object.
    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    Returns
    -------
    Sigma : Field
        Initial dust surface density
    Notes
    -----
    ``sim.ini.dust.aIniMax`` : maximum initial particle size
    ``sim.ini.dust.d2gRatio`` : initial dust-to-gas ratio
    ``sim.ini.dust.distExp`` : initial particle size distribution ``n(a) da ∝ a^{distExp} da``
    ``sim.ini.dust.allowDriftingParticles`` : if ``True`` the particle size distribution
    will be filled up to ``aIniMax``, if ``False`` the maximum particle size will be chosen
    such that there are no drifting particles initially. This prevents a particle wave traveling
    though the simulation, that is already drifting initially."""
    exp = sim.ini.dust.distExp
    # Set maximum particle size
    if(sim.ini.dust.allowDriftingParticles):
        aIni = sim.ini.dust.aIniMax
    # TODO: Adjust routine for limiting max particle size to Twopoppy
    else:
        # Calculating pressure gradient
        P = sim.gas.P
        Pi = dust_f.interp1d(sim.grid.ri, sim.grid.r, P)
        gamma = (Pi[1:] - Pi[:-1]) / (sim.grid.ri[1:] - sim.grid.ri[:-1])
        gamma = np.abs(gamma)
        # Exponent of pressure gradient
        gamma *= sim.grid.r / P
        gamma = 1. / gamma
        # Maximum drift limited particle size with safety margin
        ad = 1.e-4 * 2./np.pi * sim.ini.dust.d2gRatio * sim.gas.Sigma \
            / sim.dust.fill[:, 0] * sim.dust.rhos[:, 0] * (sim.grid.OmegaK * sim.grid.r)**2. \
            / sim.gas.cs**2. / gamma
        aIni = np.minimum(sim.ini.dust.aIniMax, ad)[:, None]
    # Fill distribution
    ret = np.where(sim.dust.a <= aIni, sim.dust.a**(exp+4), 0.)
    s = np.sum(ret, axis=1)[..., None]
    s = np.where(s > 0., s, 1.)
    # Normalize to mass
    ret = ret / s * sim.gas.Sigma[..., None] * sim.ini.dust.d2gRatio
    return ret
