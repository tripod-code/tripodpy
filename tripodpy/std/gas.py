from math import e
from dustpy.std import gas_f
import numpy as np
import scipy.sparse as sp


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
            # The disk wind sink is treated explicitly and has to limit the time step
            S = comp.gas.Sigma_dot + S_diskwind_compo(sim, group=comp.gas)
            dt = min(dt,gas_f.timestep(S,comp.gas.Sigma,sim.gas.SigmaFloor))

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
        Fluxes at the boundaries for each component"""
    if group is not None:
        return gas_f.fi(group.Sigma, sim.gas.v.rad,sim.grid.r,sim.grid.ri)
    else:
        comp = sim.components.__dict__.get(compkey)
        if comp is None:
            raise ValueError(f"Component {compkey} not found in gas components.")

        return gas_f.fi(comp.gas.Sigma, sim.gas.v.rad,sim.grid.r,sim.grid.ri)


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
        return gas_f.s_hyd(comp.gas.Fi,sim.grid.ri)
    else:
        return gas_f.s_hyd(group.Fi,sim.grid.ri)


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
    if group is None:
        group = sim.components.__dict__.get(compkey).gas
    ret = group.S.ext + group.S.hyd
    # Disk wind sink only in the interior as in the integration
    ret[1:-1] += group.S.diskwind[1:-1]
    return ret

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

def _S_diskwind(sim, Sigma):
    """Function calculates the disk wind mass loss for a given surface density
    (Tabone et al. 2022, Eq. 10, last term).

    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    Sigma : array-like
        Surface density [g/cm²]

    Returns
    -------
    S_diskwind : array-like
        Disk wind source terms [g/cm²/s]"""
    return -0.75 * sim.gas.diskwind.alpha * sim.gas.cs**2 * Sigma / \
        ((sim.gas.diskwind.Lambda - 1.) * sim.grid.r**2 * sim.grid.OmegaK)


def S_diskwind(sim):
    """Function calculates the source (loss) terms due to disk winds.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    S_diskwind : Field
        Disk wind source terms [g/cm²/s]"""
    return _S_diskwind(sim, sim.gas.Sigma)


def S_diskwind_compo(sim, compkey="default", group=None):
    """Function calculates the disk wind source (loss) terms of a gas component.
    The wind removes all gas components at the same rate per unit mass.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    compkey : str, optional
        Key of the component, by default "default"
    group : Group, optional
        Gas group of the component

    Returns
    -------
    S_diskwind : Field
        Disk wind source terms of the component [g/cm²/s]"""
    if group is None:
        group = sim.components.__dict__.get(compkey).gas
    return _S_diskwind(sim, group.Sigma)


def S_tot(sim):
    """Function calculates the total gas source terms including the disk wind.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    S_tot : Field
        Total surface density source terms [g/cm²/s]"""
    ret = np.array(sim.gas.S.hyd)
    ret[1:-1] += sim.gas.S.ext[1:-1] + sim.gas.diskwind.S[1:-1]
    return ret


def vdiskwind(sim):
    """Function calculates the velocity contribution from disk winds
    (Tabone et al. 2022, Eq. 10, second term).

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    vdiskwind : Field
        Velocity contribution from disk winds [cm/s]"""
    return -1.5 * sim.gas.diskwind.alpha * sim.gas.cs**2 / (sim.grid.r * sim.grid.OmegaK)


def vrad(sim):
    """Function calculates the radial gas velocity including torque and disk wind.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    vrad : Field
        Radial gas velocity [cm/s]"""
    vb = 2. * sim.gas.eta * sim.grid.r * sim.grid.OmegaK
    return sim.dust.backreaction.A * sim.gas.v.visc + sim.dust.backreaction.B * vb \
        + sim.gas.torque.v + sim.gas.diskwind.v


def jacobian(sim, x, *args, **kwargs):
    """Functions calculates the Jacobian for the gas including the disk wind.
    Same as ``dustpy.std.gas.jacobian`` with the disk wind velocity added
    to the advection velocity.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    x : IntVar
        Integration variable
    args : additional positional arguments
    kwargs : additional keyworda arguments

    Returns
    -------
    jac : Field
        Jacobi matrix for gas evolution

    Notes
    -----
    The boundaries need information about the time step, which is only available at
    the integration stage. The boundary values are therefore meaningless and should not
    be used to calculate the source terms via matrix-vector multiplication."""

    # Parameters
    nu = sim.gas.nu * sim.dust.backreaction.A
    # Velocity contribution from dust back reaction
    v = sim.dust.backreaction.B * 2. * sim.gas.eta * sim.grid.r * sim.grid.OmegaK
    # Velocity contribution from torque
    v += sim.gas.torque.v
    # Velocity contribution from disk wind
    v += sim.gas.diskwind.v

    # Helper variables for convenience
    r = sim.grid.r
    ri = sim.grid.ri
    area = sim.grid.A
    Nr = int(sim.grid.Nr)

    # Construct Jacobian
    A, B, C = gas_f.jac_abc(area, nu, r, ri, v)
    row_hyd = np.hstack(
        (np.arange(Nr-1)+1, np.arange(Nr), np.arange(Nr-1)))
    col_hyd = np.hstack(
        (np.arange(Nr-1), np.arange(Nr), np.arange(Nr-1)+1))
    dat_hyd = np.hstack((A.ravel()[1:], B.ravel(), C.ravel()[:-1]))

    # Right hand side
    sim.gas._rhs[:] = sim.gas.Sigma

    # Boundaries. This is only reserving space in the sparce matrix
    row_in = [0, 0, 0]
    col_in = [0, 1, 2]
    dat_in = [0., 0., 0.]
    row_out = [Nr-1, Nr-1, Nr-1]
    col_out = [Nr-3, Nr-2, Nr-1]
    dat_out = [0., 0., 0.]

    # Stitching together the generators
    row = np.hstack((row_hyd, row_in, row_out))
    col = np.hstack((col_hyd, col_in, col_out))
    dat = np.hstack((dat_hyd, dat_in, dat_out))
    gen = (dat, (row, col))
    # Building sparse matrix of gas Jacobian
    J = sp.csc_matrix(
        gen,
        shape=(Nr, Nr)
    )

    return J
