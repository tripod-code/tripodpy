from dustpy.std import gas_f
import numpy as np


def boundary(sim):
    """Function set the boundary conditions of the gas.
    Not implemented, yet.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame"""
    for name, comp in sim.components.__dict__.items():
        if(name.startswith("_") or not comp.includegas):
            continue
        comp.gas.boundary.inner.setboundary()
        comp.gas.boundary.outer.setboundary()


def enforce_floor_value(sim):
    """Function enforces floor value to gas surface density.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame"""
    for name, comp in sim.components.__dict__.items():
        if (name.startswith("_") or not comp.includegas):
            continue
        comp.gas.Sigma[:] = gas_f.enforce_floor(
            comp.gas.Sigma,
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
    for name, comp in sim.components.__dict__.items():
        if name.startswith("_"):
            continue
        comp.gas._SigmaOld[:] = comp.gas.Sigma[:]



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
    #set_implicit_boundaries(sim)
    set_implicit_boundaries_compo(sim)

#Modify 
def set_implicit_boundaries_compo(sim):
    """Function calculates the fluxes at the boundaries after the implicit integration step.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame"""
    for name, comp in sim.components.__dict__.items():
        if name.startswith("_"):
            continue
        ret = gas_f.implicit_boundaries(
            sim.t.prevstepsize,
            comp.gas.Fi,
            sim.grid.ri,
            comp.gas.Sigma,
            comp.gas._SigmaOld
        )

        # Source terms
        comp.gas.S.tot[0] = ret[0]
        comp.gas.S.hyd[0] = ret[0]
        comp.gas.S.tot[-1] = ret[1]
        comp.gas.S.hyd[-1] = ret[1]

        # Fluxes through boundaries
        comp.gas.Fi[0] = ret[2]
        comp.gas.Fi[-1] = ret[3]

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
    for key, comp in sim.components.__dict__.items():
        if key.startswith("_"):
            continue
        if not comp.tracer:
            ret += comp.gas.Sigma
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
    for key, comp in sim.components.__dict__.items():
        if key.startswith("_"):
            continue
        if not comp.tracer:
            ret += comp.gas.Sigma / comp.gas.mu

    return sim.gas.Sigma/ret


def dt_compo(sim):
    """Function returns the timestep depending on the source terms.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    dt : float
        Time step"""
    dt = 1.e100
    for key,comp in sim.components.__dict__.items():
        if not key.startswith("_"):
            dt = min(dt,gas_f.timestep(comp.gas.S.tot,comp.gas.Sigma,sim.gas.SigmaFloor))

    return dt



def Fi_compo(sim,compkey = "default"):
    """Function returns the fluxes at the boundaries for each component.
    
    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    Returns
    -------
    Fi : Field
        Fluxes at the boundaries for each component"""
    comp = sim.components.__dict__.get(compkey)
    if comp is None:
        raise ValueError(f"Component {compkey} not found in gas components.")

    return gas_f.fi(comp.gas.Sigma, sim.gas.v.rad,sim.grid.r,sim.grid.ri)


def S_hyd_compo(sim, compkey="default"):
    """Function returns the hydrodynamical source terms for each component.
    
    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    compkey : str, optional
        Key of the component, by default "default"
        
    Returns
    -------
    S_hyd : Field
        Hydrodynamical source terms for each component"""
    comp = sim.components.__dict__.get(compkey)

    
    return gas_f.s_hyd(comp.gas.Fi,sim.grid.ri)


def S_tot_compo(sim, compkey="default"):
    """Function returns the external source terms for each component.
    
    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    compkey : str, optional
        Key of the component, by default "default"
        
    Returns
    -------
    S_ext : Field
        External source terms for each component"""
    comp = sim.components.__dict__.get(compkey)
    
    return  comp.gas.S.hyd + comp.gas.S.ext
    
def S_ext_total(sim):
    """Function returns the total external source terms for all components.
    
    Parameters
    ----------
    sim : Frame
        Parent simulation frame
        
    Returns
    -------
    S_ext_total : Field
        Total external source terms for all components"""
    ret = np.zeros_like(sim.gas.Sigma)
    for key, comp in sim.components.__dict__.items():
        if key.startswith("_"):
            continue
        ret += comp.gas.S.ext
        
    return ret