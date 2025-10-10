'''Module containing standard functions for the main simulation object.'''

from tripod import std
import dustpy.std as dp_std

import numpy as np


def dt(sim):
    """Function returns the timestep depending on the source terms.

    Paramters
    ---------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    dt : float
        Time step"""

    dt_gas = std.gas.dt_compo(sim) or 1.e100
    dt_dust = std.dust.dt(sim) or 1.e100
    if sim._dust_compo:
        dt_dust = min(dt_dust, std.dust.dt_compo(sim) or 1.e100)
    dt = np.minimum(dt_gas, dt_dust)
    return sim.t.cfl * dt


def prepare_implicit_dust(sim):
    """This function is the preparation function that is called
    before every integration step.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame"""
    #dp_std.gas.prepare(sim)
    std.gas.prepare(sim)
    std.dust.prepare(sim)
    std.compo.prepare(sim)


def finalize_implicit_dust(sim):
    """This function is the finalization function that is called
    after every integration step. It is managing the boundary
    conditions and is enforcing floor values.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame"""
    #dp_std.gas.finalize(sim)
    std.compo.finalize(sim)
    std.gas.finalize(sim)
    std.dust.finalize(sim)
 