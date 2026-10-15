from math import e
from dustpy.std import gas_f
import numpy as np


def enforce_floor_value(sim):
    """Function enforces floor value to gas surface density.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame"""
    for name, comp in sim.components.__dict__.items():
        if (name.startswith("_")):
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
    Fi_bound = np.zeros(2)
    S_bound = np.zeros(2)
    for name, comp in sim.components.__dict__.items():
        if name.startswith("_"):
            continue
        Fi = comp.gas.Fi
        if comp.gas._active:
            # Fluxes at the new state (like dustpy.std.gas.finalize does before
            # setting the implicit boundaries)
            comp.gas.Fi.update()
            # The boundary fluxes are derived from the fluxes through the first
            # and last interior interfaces. Use the fluxes of the implicit scheme
            # there, the explicit fluxes deviate at steep gradients.
            Fi = np.array(comp.gas.Fi)
            Nr = Fi.shape[0] - 1
            Fi[1] = Fi_implicit_interface(sim, comp.gas.Sigma, 1)
            Fi[Nr-1] = Fi_implicit_interface(sim, comp.gas.Sigma, Nr-1)
        ret = gas_f.implicit_boundaries(
            sim.t.prevstepsize,
            Fi,
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

        if comp.gas._active:
            # Store the boundary fluxes and source terms, so the updaters do not
            # overwrite them with explicit values when the frame is updated
            comp.gas._Fi_boundary = np.array([ret[2], ret[3]])
            comp.gas._S_boundary = np.array([ret[0], ret[1]])
            Fi_bound += ret[2:]
            S_bound += ret[:2]

    # The total gas fluxes and source terms are not computed from the
    # components, so the boundary values have to be set here
    sim.gas.Fi[0] = Fi_bound[0]
    sim.gas.Fi[-1] = Fi_bound[1]
    sim.gas.S.hyd[0] = S_bound[0]
    sim.gas.S.hyd[-1] = S_bound[1]
    sim.gas.S.tot[0] = S_bound[0]
    sim.gas.S.tot[-1] = S_bound[1]


def Fi_implicit_interface(sim, Sigma, i):
    """Function returns the gas flux through a grid cell interface as it is
    discretized in the implicit gas Jacobian (dustpy.std.gas.jacobian).

    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    Sigma : array
        Gas surface density
    i : int
        Index of the interface between grid cells i-1 and i (0 < i < Nr)

    Returns
    -------
    Fi : float
        Flux through the interface [g/cm/s]"""
    r = sim.grid.r
    ri = sim.grid.ri

    # Same viscosity and velocity as in dustpy.std.gas.jacobian
    nu = sim.gas.nu * sim.dust.backreaction.A
    v = sim.dust.backreaction.B * 2. * sim.gas.eta * r * sim.grid.OmegaK
    v = v + sim.gas.torque.v

    # Velocity linearly interpolated onto the interface
    vi = v[i-1] + (v[i] - v[i-1]) * (ri[i] - r[i-1]) / (r[i] - r[i-1])

    # Advection, upwind
    F_adv = max(vi, 0.) * r[i-1] * Sigma[i-1] + min(vi, 0.) * r[i] * Sigma[i]
    # Viscous diffusion of nu * Sigma * sqrt(r)
    G = nu * Sigma * np.sqrt(r)
    F_visc = 3. * np.sqrt(ri[i]) * (G[i-1] - G[i]) / (r[i] - r[i-1])

    return (F_adv + F_visc) / ri[i]


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
    # divides the totoal suraced densyty by the sum of the numberdesities of all components -> mu = Sigma_tot / (sum_i Sigma_i/mu_i)
    ret = np.zeros_like(sim.gas.mu)
    for key, comp in sim.components.__dict__.items():
        if key.startswith("_"):
            continue
        ret += comp.gas.Sigma / comp.gas.pars.mu

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
        if not key.startswith("_") and comp.gas._active:
            dt = min(dt,gas_f.timestep(comp.gas.Sigma_dot,comp.gas.Sigma,sim.gas.SigmaFloor))

    return dt



def Fi_compo(sim,compkey = "default",group=None):
    """Function returns the fluxes at the boundaries for each component.
    
    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    Returns
    -------
    Fi : Field
        Fluxes at the boundaries for each component

    Notes
    -----
    The fluxes through the domain boundaries are taken from the last implicit
    integration step (set_implicit_boundaries_compo), if available."""
    if group is None:
        comp = sim.components.__dict__.get(compkey)
        if comp is None:
            raise ValueError(f"Component {compkey} not found in gas components.")
        group = comp.gas

    Fi = gas_f.fi(group.Sigma, sim.gas.v.rad,sim.grid.r,sim.grid.ri)
    Fi_boundary = getattr(group, "_Fi_boundary", None)
    if isinstance(Fi_boundary, np.ndarray):
        Fi[0] = Fi_boundary[0]
        Fi[-1] = Fi_boundary[1]
    return Fi


def S_hyd_compo(sim, compkey="default",group=None):
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
    
    if group is None:
        comp = sim.components.__dict__.get(compkey)
        group = comp.gas

    S_hyd = gas_f.s_hyd(group.Fi,sim.grid.ri)
    # Source terms in the boundary cells from the last implicit integration step
    S_boundary = getattr(group, "_S_boundary", None)
    if isinstance(S_boundary, np.ndarray):
        S_hyd[0] = S_boundary[0]
        S_hyd[-1] = S_boundary[1]
    return S_hyd


def S_tot_compo(sim, compkey="default",group=None):
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
    if group is not None:
        return group.S.ext + group.S.hyd
    else:
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
        ret += comp.gas.Sigma_dot
        
    return ret