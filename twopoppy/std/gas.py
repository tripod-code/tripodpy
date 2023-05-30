from dustpy.std import gas_f
import numpy as np


def boundary(sim):
    """Function set the boundary conditions of the gas.
    Not implemented, yet.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame"""
    for name, comp in sim.gas.components.__dict__.items():
        if name.startswith("_"):
            continue
        comp.boundary.inner.setboundary()
        comp.boundary.outer.setboundary()


def enforce_floor_value(sim):
    """Function enforces floor value to gas surface density.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame"""
    for name, comp in sim.gas.components.__dict__.items():
        if name.startswith("_"):
            continue
        comp.Sigma[:] = gas_f.enforce_floor(
            comp.Sigma,
            sim.gas.SigmaFloor
        )


def prepare(sim):
    """Function prepares gas integration step.
    It stores the current value of the surface density in a hidden field.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame"""
    # Storing current surface density
    sim.gas._SigmaOld[:] = sim.gas.Sigma[:]


def finalize(sim):
    """Function finalizes gas integration step.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame"""
    boundary(sim)
    enforce_floor_value(sim)
    sim.gas.v.update()
    sim.gas.Fi.update()
    sim.gas.S.hyd.update()
    set_implicit_boundaries(sim)


def set_implicit_boundaries(sim):
    """Function calculates the fluxes at the boundaries after the implicit integration step.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame"""
    ret = gas_f.implicit_boundaries(
        sim.t.prevstepsize,
        sim.gas.Fi,
        sim.grid.ri,
        sim.gas.Sigma,
        sim.gas._SigmaOld
    )

    # Source terms
    sim.gas.S.tot[0] = ret[0]
    sim.gas.S.hyd[0] = ret[0]
    sim.gas.S.tot[-1] = ret[1]
    sim.gas.S.hyd[-1] = ret[1]

    # Fluxes through boundaries
    sim.gas.Fi[0] = ret[2]
    sim.gas.Fi[-1] = ret[3]


def Sigma_tot(sim):
    """
    Function calculates the total gas surface density from
    the gas components.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    Sigma_tot : Field
        Total gas surface density
    """
    ret = np.zeros_like(sim.gas.Sigma)
    for key, comp in sim.gas.components.__dict__.items():
        if key.startswith("_"):
            continue
        if not comp.tracer:
            ret += comp.Sigma
    return ret


def mu(sim):
    """
    Function calculates the mean molecular weight from
    the gas components.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    mu : Field
        Mean molecular weight
    """
    ret = np.zeros_like(sim.gas.mu)
    for key, comp in sim.gas.components.__dict__.items():
        if key.startswith("_"):
            continue
        if not comp.tracer:
            ret += comp.Sigma / comp.mu

    return sim.gas.Sigma/ret
