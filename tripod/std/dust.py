"""Module containing standard functions for the dust."""

import dustpy.constants as c
from dustpy.std import dust_f as dp_dust_f
import dustpy.std.dust as dp_dust
from tripod.std import dust_f

from simframe.integration import Scheme

import numpy as np

import scipy.sparse as sp

DEBUG = False


def dt(sim):
    """Function calculates the time step from dust.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    dt : float
        Dust time step"""
    dtSigma = dt_Sigma(sim) or 1e100
    dtsmax = dt_smax(sim)
    return np.minimum(dtSigma, dtsmax)

# TODO: Check if this is still needed since we now use the one from dustpy


def dt_Sigma(sim):
    """Function calculates the time step due to changes in Sigma.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    dt_Sigma : float
        Time step due to changes in Sigma"""
    if np.any(sim.dust.S.tot[1:-1, ...] < 0.):
        mask = np.logical_and(
            sim.dust.Sigma > sim.dust.SigmaFloor,
            sim.dust.S.tot < 0.)
        mask[0, :] = False
        mask[-1:, :] = False

        mask2 = sim.dust.S.tot[:, 1] < 0.
        mask2 = sim.dust.S.tot[:,1] * sim.dust.Sigma[:,0] - sim.dust.S.tot[:,0] * sim.dust.Sigma[:,1] < 0.
        f = sim.dust.Sigma[:,1]/sim.dust.Sigma.sum(-1)
        mask2 = np.logical_and(mask2, f<0.43)
        dsig_da = dsigda(sim)
        dt_pred = 10 * ((-0.1*  sim.dust.s.max[mask2] * dsig_da[mask2]) + sim.dust.Sigma[mask2,1] - sim.dust.f.crit* sim.dust.Sigma[mask2,:].sum(-1)) \
                /(sim.dust.S.tot[mask2, 1] *(sim.dust.f.crit - 1.) + sim.dust.f.crit * sim.dust.S.tot[mask2, 0] + dsig_da[mask2] * sim.dust.s.sdot_coag[mask2] +sim.dust.S.smax_hyd[mask2]*dsig_da[mask2])
        dt_pred = np.where(dt_pred != dt_pred, 0, dt_pred)  
        dt_pred = np.abs(dt_pred)
        dt_pred = np.where(dt_pred == np.inf, 0, dt_pred)

        dt = np.ones_like(sim.dust.Sigma)*1e100
        dt[mask] = np.abs(sim.dust.Sigma[mask] / sim.dust.S.tot[mask])
        dt[mask2,1] = np.maximum(dt[mask2,1],dt_pred)
        return dt.min()
    return 1e100


def S_smax_hyd(sim):
    """Function calculates the hydrodynamic source terms for s.max 
    -> assumes that I can callculate the hydrodynamic source term for the product and divide it by Sig1
    """

    Fi = Fi_sig1smax(sim)
    S_hyd = (dp_dust_f.s_hyd(Fi,sim.grid.ri)[:,0] - sim.dust.S.hyd[:,1]*sim.dust.s.max)/sim.dust.Sigma[:,1]

    return S_hyd

def Fi_sig1smax (sim):
    """
    Function that calculates the total flux of the Sigma[1] and s.max 
    used to solved the advection equation for s.max

    """
    Fi_diff = F_diff(sim,Sigma = sim.dust.Sigma*sim.dust.s.max[:,np.newaxis])
    Fi_adv = F_adv(sim,Sigma = sim.dust.Sigma*sim.dust.s.max[:,np.newaxis])
    Fi_tot = (Fi_diff + Fi_adv)[:,1]
    return Fi_tot



def dt_smax(sim):
    """Function calculates the time step due to changes in smax.
    Change of smax during one integration step bound by smin and maximum growth factor.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    dt_smax : float
        Particle growth time step"""
    # TODO: double check if this makes sense
    mask2 = sim.dust.S.tot[:, 1] < 0.
    mask2 = sim.dust.S.tot[:,1] * sim.dust.Sigma[:,0] - sim.dust.S.tot[:,0] * sim.dust.Sigma[:,1] < 0.
    f = sim.dust.Sigma[:,1]/sim.dust.Sigma.sum(-1)
    mask2 = np.logical_and(mask2, f<0.43)
    smax_dot_hyd = sim.dust.S.smax_hyd

    smax_dot = np.minimum(np.abs(sim.dust.s.sdot_coag[1:-1]) , np.abs(sim.dust.s.sdot_coag[1:-1]+smax_dot_hyd[1:-1]))
    dt = sim.dust.s.max[1:-1] / (smax_dot + 1e-100)
    dt[mask2[1:-1]] = 1e100
    return dt.min()


def prepare(sim):
    """Function prepares implicit dust integration step.
    It stores the current value of the surface density in a hidden field.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame"""
    Nm_s = int(sim.grid._Nm_short)
    Nr = int(sim.grid.Nr)

    # Copy values to state vector Y
    sim.dust._Y[:Nm_s * Nr] = sim.dust.Sigma.ravel()
    sim.dust._Y[Nm_s * Nr:] = sim.dust.s.max * sim.dust.Sigma[:, 1]

    # Setting coagulation sources and external sources at boundaries to zero
    sim.dust.S.coag[0] = 0.
    sim.dust.S.coag[-1] = 0.
    # Storing current surface density
    sim.dust._SigmaOld[...] = sim.dust.Sigma[...]
    sim.dust.s._maxOld = sim.dust.s.max
    sim.dust.s._prev_sdot_coag = sim.dust.s.sdot_coag
    s_max_deriv = sim.dust.s.max.derivative()
    sim.dust.S.shrink = S_shrink(sim)
    enforce_f(sim)
    sim.dust.S.ext.update()
    sim.dust.S.coag.update()
    sim.dust.S.hyd.update()
    sim.dust.S.coag[0] = 0.
    sim.dust.S.coag[-1] = 0.
    sim.dust.S.ext[0] = 0.
    sim.dust.S.ext[-1] = 0.


def finalize(sim):
    """Function finalizes implicit integration step.

    Parameters
    ----------
    sim : Frame
        Parent integration frame"""
    Nm_s = int(sim.grid._Nm_short)
    Nr = int(sim.grid.Nr)

    # Copy values from state vector to fields
    sim.dust.Sigma[...] = sim.dust._Y[:Nr * Nm_s].reshape(sim.dust.Sigma.shape)
    # Making sure smax is not smaller than 1.5 smin to ensure minimal distribution width])

    sim.dust.s.max = np.maximum(
        1.5 * sim.dust.s.min, sim.dust._Y[Nr * Nm_s:] / sim.dust.Sigma[..., 1])

    dp_dust.boundary(sim)
    dp_dust.enforce_floor_value(sim)
    enforce_f(sim)
    sim.dust.s.max.derivative()
    sim.dust.v.rad.update()
    sim.dust.Fi.update()
    sim.dust.S.coag.update()
    sim.dust.S.ext.update()
    sim.dust.S.hyd.update()
    dp_dust.set_implicit_boundaries(sim)


def smax_initial(sim):
    """Function calculates the initial maximum particle sizes

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    smax : Field
        initial maximum particle sizes"""

    # TODO: needs some check if aIniMax < smin

    # Helper variables
    smin = sim.dust.s.min

    # Routine for excluding initially drifting particles
    if not sim.ini.dust.allowDriftingParticles:
        # Calculating pressure gradient
        P = sim.gas.P
        Pi = dust_f.interp1d(sim.grid.ri, sim.grid.r, P)
        gamma = (Pi[1:] - Pi[:-1]) / (sim.grid.ri[1:] - sim.grid.ri[:-1])
        gamma = np.abs(gamma)
        # Exponent of pressure gradient
        gamma *= sim.grid.r / P
        # Maximum drift limited particle size with safety margin
        ad = 5e-3 * 2. / np.pi * sim.ini.dust.d2gRatio * sim.gas.Sigma / sim.dust.fill[:, 0] \
            * sim.dust.rhos[:, 0] * (sim.grid.OmegaK * sim.grid.r) ** 2. / sim.gas.cs ** 2. / gamma
        aIni = np.minimum(sim.ini.dust.aIniMax, ad)

        # TODO: Sandro used this; we need to check if this is necessary
        # Enforce initial drift limit
        # sim.dust.q.eff = np.where(
        #     aIni < sim.ini.dust.aIniMax, sim.dust.q.sweep, sim.dust.q.eff)

        return np.maximum(1.5 * smin, aIni)

    # Return the ini value if drifting particles are allowed
    return np.ones_like(smin) * sim.ini.dust.aIniMax


def Sigma_initial(sim):
    """Function calculates the initial condition of the dust surface densities

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    Sigma : Field
        Initial dust surface density"""

    # Helper variables
    q = sim.dust.q.eff
    qp4 = q + 4.
    smin = sim.dust.s.min
    smax = sim.dust.s.max
    sint = np.sqrt(smin * smax)
    SigmaFloor = sim.dust.SigmaFloor

    # Values for q != -4
    S0 = np.zeros_like(sim.grid.r)
    S1 = np.zeros_like(sim.grid.r)
    for i in range(int(sim.grid.Nr)):
        if smax[i] <= 1.5* smin[i]:
            S0[i] = SigmaFloor[i, 0]
            S1[i] = SigmaFloor[i, 1]
        else:
            S0[i] = (sint[i] ** qp4[i] - smin[i] ** qp4[i]) / \
                    (smax[i] ** qp4[i] - smin[i] ** qp4[i])
            S1[i] = 1. - S0[i]
    S = np.array([S0, S1]).T

    # Values for q == -4
    S0_4 = np.zeros_like(sim.grid.r)
    S1_4 = np.zeros_like(sim.grid.r)
    for i in range(int(sim.grid.Nr)):
        if smax[i] == smin[i]:
            S0_4[i] = SigmaFloor[i, 0]
            S1_4[i] = SigmaFloor[i, 1]
        else:
            S0_4[i] = np.log(sint[i] / smin[i]) / np.log(smax[i] / smin[i])
            S1_4[i] = 1. - S0_4[i]
    S_4 = np.array([S0_4, S1_4]).T

    Sigma = sim.ini.dust.d2gRatio * \
        sim.gas.Sigma[:, None] * np.where(q[:, None] == -4., S_4, S)

    return Sigma


def jacobian(sim, x, dx=None, *args, **kwargs):
    """Function calculates the Jacobian for implicit dust integration.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    x : IntVar
        Integration variable
    dx : float, optional, default : None
        stepsize
    args : additional positional arguments
    kwargs : additional keyword arguments

    Returns
    -------
    jac : Sparse matrix
        Dust Jacobian

    Notes
    -----
    Function returns the Jacobian for ``Simulation.dust.Sigma.ravel()``
    instead of ``Simulation.dust.Sigma``. The Jacobian is stored as
    sparse matrix."""

    # Helper variables for convenience
    if dx is None:
        dt = x.stepsize
    else:
        dt = dx
    r = sim.grid.r
    ri = sim.grid.ri
    area = sim.grid.A
    Nr = int(sim.grid.Nr)
    Nm_s = int(sim.grid._Nm_short)

    # Total problem size
    Ntot = int((Nr * Nm_s))

    # Getting data vector and coordinates for the coagulation sparse matrix
    dat_coag, row_coag, col_coag = dust_f.jacobian_coagulation_generator(
        # here we compute the cross section where the first entry is
        # the cross section of (a0, a1) and the second of (a1, fudge * a1)
        np.pi * (sim.dust.a[:, [0, 2]]+sim.dust.a[:, [2, 1]])**2,
        # same for the relative velocities
        sim.dust.v.rel.tot[:, [0, 2], [2, 1]],
        sim.dust.H[:, [0, 2]],
        sim.dust.m[:, [0, 2]],
        sim.dust.Sigma,
        sim.dust.s.min,
        sim.dust.s.max,
        sim.dust.q.eff
    )
        
    # Getting data vector and coordinates for the hydrodynamic sparse matrix
    A, B, C = dp_dust_f.jacobian_hydrodynamic_generator(
        area,
        sim.dust.D[:, [0, 2]],
        r,
        ri,
        sim.gas.Sigma,
        sim.dust.v.rad_flux[:, [0, 2]]
    )
    row_hyd = np.hstack(
        (np.arange(Ntot - Nm_s) + Nm_s, np.arange(Ntot), np.arange(Ntot - Nm_s)))
    col_hyd = np.hstack(
        (np.arange(Ntot - Nm_s), np.arange(Ntot), np.arange(Ntot - Nm_s) + Nm_s))
    dat_hyd = np.hstack((A.ravel()[Nm_s:], B.ravel(), C.ravel()[:-Nm_s]))

    # Right-hand side
    sim.dust._rhs[Nm_s:-Nm_s] = sim.dust.Sigma.ravel()[Nm_s:-Nm_s]

    # BOUNDARIES

    # Inner boundary

    # Initializing data and coordinate vectors for sparse matrix
    dat_in = np.zeros(int(3. * Nm_s))
    row0 = np.arange(int(Nm_s))
    col0 = np.arange(int(Nm_s))
    col1 = np.arange(int(Nm_s)) + Nm_s
    col2 = np.arange(int(Nm_s)) + 2. * Nm_s
    row_in = np.concatenate((row0, row0, row0))
    col_in = np.concatenate((col0, col1, col2))

    # Filling data vector depending on boundary condition
    if sim.dust.boundary.inner is not None:
        # Given value
        if sim.dust.boundary.inner.condition == "val":
            sim.dust._rhs[:Nm_s] = sim.dust.boundary.inner.value
        # Constant value
        elif sim.dust.boundary.inner.condition == "const_val":
            dat_in[Nm_s:2 * Nm_s] = 1. / dt
            sim.dust._rhs[:Nm_s] = 0.
        # Given gradient
        elif sim.dust.boundary.inner.condition == "grad":
            K1 = - r[1] / r[0]
            dat_in[Nm_s:2 * Nm_s] = -K1 / dt
            sim.dust._rhs[:Nm_s] = - ri[1] / r[0] * \
                (r[1] - r[0]) * sim.dust.boundary.inner.value
        # Constant gradient
        elif sim.dust.boundary.inner.condition == "const_grad":
            Di = ri[1] / ri[2] * (r[1] - r[0]) / (r[2] - r[0])
            K1 = - r[1] / r[0] * (1. + Di)
            K2 = r[2] / r[0] * Di
            dat_in[:Nm_s] = 0.
            dat_in[Nm_s:2 * Nm_s] = -K1 / dt
            dat_in[2 * Nm_s:] = -K2 / dt
            sim.dust._rhs[:Nm_s] = 0.
        # Given power law
        elif sim.dust.boundary.inner.condition == "pow":
            p = sim.dust.boundary.inner.value
            sim.dust._rhs[:Nm_s] = sim.dust.Sigma[1] * (r[0] / r[1]) ** p
        # Constant power law
        elif sim.dust.boundary.inner.condition == "const_pow":
            p = np.log(sim.dust.Sigma[2] /
                       sim.dust.Sigma[1]) / np.log(r[2] / r[1])
            K1 = - (r[0] / r[1]) ** p
            dat_in[Nm_s:2 * Nm_s] = -K1 / dt
            sim.dust._rhs[:Nm_s] = 0.

    # Outer boundary

    # Initializing data and coordinate vectors for sparse matrix
    dat_out = np.zeros(int(3. * Nm_s))
    row0 = np.arange(int(Nm_s))
    col0 = np.arange(int(Nm_s))
    col1 = np.arange(int(Nm_s)) - Nm_s
    col2 = np.arange(int(Nm_s)) - 2. * Nm_s
    offset = (Nr - 1) * Nm_s
    row_out = np.concatenate((row0, row0, row0)) + offset
    col_out = np.concatenate((col0, col1, col2)) + offset

    # Filling data vector depending on boundary condition
    if sim.dust.boundary.outer is not None:
        # Given value
        if sim.dust.boundary.outer.condition == "val":
            sim.dust._rhs[-Nm_s:] = sim.dust.boundary.outer.value
        # Constant value
        elif sim.dust.boundary.outer.condition == "const_val":
            dat_out[-2 * Nm_s:-Nm_s] = 1. / dt
            sim.dust._rhs[-Nm_s:] = 0.
        # Given gradient
        elif sim.dust.boundary.outer.condition == "grad":
            KNrm2 = -r[-2] / r[-1]
            dat_out[-2 * Nm_s:-Nm_s] = -KNrm2 / dt
            sim.dust._rhs[-Nm_s:] = ri[-2] / r[-1] * \
                (r[-1] - r[-2]) * sim.dust.boundary.outer.value
        # Constant gradient
        elif sim.dust.boundary.outer.condition == "const_grad":
            Do = ri[-2] / ri[-3] * (r[-1] - r[-2]) / (r[-2] - r[-3])
            KNrm2 = - r[-2] / r[-1] * (1. + Do)
            KNrm3 = r[-3] / r[-1] * Do
            dat_out[-2 * Nm_s:-Nm_s] = -KNrm2 / dt
            dat_out[-3 * Nm_s:-2 * Nm_s] = -KNrm3 / dt
            sim.dust._rhs[-Nm_s:] = 0.
        # Given power law
        elif sim.dust.boundary.outer.condition == "pow":
            p = sim.dust.boundary.outer.value
            sim.dust._rhs[-Nm_s:] = sim.dust.Sigma[-2] * (r[-1] / r[-2]) ** p
        # Constant power law
        elif sim.dust.boundary.outer.condition == "const_pow":
            p = np.log(sim.dust.Sigma[-2] /
                       sim.dust.Sigma[-3]) / np.log(r[-2] / r[-3])
            KNrm2 = - (r[-1] / r[-2]) ** p
            dat_out[-2 * Nm_s:-Nm_s] = -KNrm2 / dt
            sim.dust._rhs[-Nm_s:] = 0.

    # Stitching together the generators
    row = np.hstack((row_coag, row_hyd, row_in, row_out))
    col = np.hstack((col_coag, col_hyd, col_in, col_out))
    dat = np.hstack((dat_coag, dat_hyd, dat_in, dat_out))

    gen = (dat, (row, col))
    # Building sparse matrix of coagulation Jacobian
    J = sp.csc_matrix(
        gen,
        shape=(Ntot, Ntot)
    )

    # Adding and returning all matrix components
    return J


def a(sim):
    """Function calculates the particle sizes from the characteristic particle sizes and the distribution exponent.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    a : Field
        Particle sizes"""
    # interpolate between drift and turbulence dominated pre-factor
    return dust_f.calculate_a(sim.dust.s.min, sim.dust.s.max, sim.dust.qrec, sim.dust.f.dv, sim.grid._Nm_long)


def F_adv(sim, Sigma=None):
    """Function calculates the advective flux at the cell interfaces. It is linearly interpolating
    the velocities onto the grid cell interfaces and is assuming
    vi(0, :) = vi(1, :) and vi(Nr, :) = vi(Nr-1, :).

    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    Sigma : Field, optional, default : None
        Surface density to be used if not None

    Returns
    -------
    Fi : Field
        Advective mass fluxes through the grid cell interfaces"""
    if Sigma is None:
        Sigma = sim.dust.Sigma
    return dp_dust_f.fi_adv(Sigma, sim.dust.v.rad_flux[:, [0, 2]], sim.grid.r, sim.grid.ri)



def F_diff(sim, Sigma=None):
    """Function calculates the diffusive flux at the cell interfaces"""
    if Sigma is None:
        Sigma = sim.dust.Sigma

    Fi = dust_f.fi_diff_no_limit(sim.dust.D[:, [0, 2]],
                        Sigma,
                        sim.gas.Sigma,
                        sim.dust.St[:, [0, 2]]*sim.dust.f.drift,
                        np.sqrt(sim.dust.delta.rad * sim.gas.cs ** 2),
                        sim.grid.r,
                        sim.grid.ri)
    Fi[:1, :] = 0.
    Fi[-1:, :] = 0.

    return Fi


# TODO: This function is not needed after a DustPy update.

def F_tot(sim, Sigma=None):
    """Function calculates the total mass fluxes through grid cell interfaces.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    Sigma : Field, optional, default : None
        Surface density to be used if not None

    Returns
    -------
    Ftot : Field
        Total mass flux through interfaces"""
    Fi = np.zeros_like(sim.dust.Fi.tot)
    if Sigma is None:
        Fdiff = sim.dust.Fi.diff
        Fadv = sim.dust.Fi.adv
    else:
        Fdiff = sim.dust.Fi.diff.updater.beat(sim, Sigma=Sigma)
        Fadv = sim.dust.Fi.adv.updater.beat(sim, Sigma=Sigma)
    if Fdiff is not None:
        Fi += Fdiff
    if Fadv is not None:
        Fi += Fadv
    return Fi


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
    return dust_f.calculate_m(sim.dust.a, sim.dust.rhos, sim.dust.fill)


def p_frag(sim):
    """Function calculates the fragmentation probability.
    The type of assumed transition between sticking and
    fragmentation is given as an argument.

    This uses the relative velocities between 0.5 * amax and amax.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    pf : Field
        Fragmentation probability."""
    return dust_f.pfrag(sim.dust.v.rel.tot[:, -2, -1], sim.dust.v.frag)


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
    return p


def H(sim):
    """Computes the dust scale height by using the
    dustpy function, for all Nm_long dust sizes.

    Parameters
    ----------
    sim : the simulation frame
    """
    return dp_dust_f.h_dubrulle1995(
        sim.gas.Hp,
        sim.dust.St,
        sim.dust.delta.vert)


def rho_midplane(sim):
    """Function calculates the midplane mass density.

    Note that we have more particle sizes than surface densities,
    so we use the appropriate sigma for each size:
    [a0, fudge * a1, a1, 0.5 * amax, amax]
    ->
    [Sig0, Sig1, Sig1, Sig1, Sig1]

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    rho : Field
        Midplane mass density"""
    # The scale height H has a longer shape than Sigma and has to be adjusted
    return sim.dust.Sigma[:, [0, 1, 1, 1, 1]] / (np.sqrt(2 * c.pi) * sim.dust.H)


def smax_deriv(sim, t, smax):
    """Function calculates the derivative of smax.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    t : IntVar
        Current time
    smax : Field, optional, defaul : None
        Current smax to be used if not None

    Returns
    -------
    smax_dot: Field
        Derivative of smax"""

    if smax is None:
        smax = sim.dust.s.max

    ds_coag = dust_f.smax_deriv(
        sim.dust.v.rel.tot[:, -2, -1],
        sim.dust.rho[:, 2],
        sim.dust.rhos[:, 2],
        sim.dust.s.min,
        sim.dust.s.max,
        sim.dust.v.frag,
        sim.dust.Sigma,
        sim.dust.SigmaFloor)

    # Prevents unwanted growth of smax at the inner boundary Experimental
    if(True):
        mask = (sim.dust.v.rel.tot[:, -2, -1]/ sim.dust.v.frag) > 0.94
        damp_factor = 0.05
        r0 = 2.5e-1*c.au
        ir = np.argmin(abs(r0-sim.grid.r))
        width = 5e-2*c.au
        damp_coag = damp_factor + (1 - damp_factor) * (0.5 * (1 + np.tanh((sim.grid.r - r0) / width)))
        damp_coag = damp_factor + (1 - damp_factor) / (1 + np.exp(-(sim.grid.r - r0) / width))

        if(mask[0:ir].all()):
            ds_coag *= damp_coag

    sim.dust.s.sdot_shrink = np.zeros_like(sim.dust.s.max)
    sim.dust.s.sdot_coag = ds_coag

    return ds_coag


def S_coag(sim, Sigma=None):
    """Function calculates the source terms from dust growth

    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    Sigma : Field, optional, default : None
        Surface density to be used if not None

    Returns
    -------
    Scoag : Field
        Source terms from dust growth"""
    if Sigma is None:
        Sigma = sim.dust.Sigma

    s_coag = dust_f.s_coag(
        # here we compute the cross section where the first entry is
        # the cross section of (a0, a1) and the second of (a1, fudge * a1)
        np.pi * (sim.dust.a[:, [0, 2]]+sim.dust.a[:, [2, 1]])**2,
        # same for the relative velocities
        sim.dust.v.rel.tot[:, [0, 2], [2, 1]],
        sim.dust.H[:, [0, 2]],
        sim.dust.m[:, [0, 2]],
        Sigma,
        sim.dust.s.min,
        sim.dust.s.max,
        sim.dust.q.eff,
        sim.dust.SigmaFloor
    )

    # Prevents unwanted growth of smax
    return s_coag

# TODO: check if this is still needed after the dustpy update


def enforce_f(sim):

    delta = np.maximum( 0., sim.dust.f.crit * sim.dust.Sigma[...].sum(-1) - sim.dust.Sigma[:,1])
    sim.dust.s.max += delta * dadsig(sim) 
    sim.dust.Sigma[:,1] += delta
    sim.dust.Sigma[:,0] -= delta
    sim.dust.qrec.update()


def dadsig(sim):
    return dust_f.dadsig(sim.dust.s.lim, sim.dust.qrec,sim.dust.f.crit,  sim.dust.s.max, sim.dust.s.min, sim.dust.Sigma)

def dsigda(sim):
    return dust_f.dsigda(sim.dust.s.lim, sim.dust.qrec,sim.dust.f.crit,  sim.dust.s.max, sim.dust.s.min, sim.dust.Sigma)


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
        Scoag = sim.dust.S.coag
        Shyd = sim.dust.S.hyd
    else:
        Scoag = sim.dust.S.coag.updater.beat(sim, Sigma=Sigma)
        if Scoag is None:
            Scoag = sim.dust.S.coag
        Shyd = sim.dust.S.hyd.updater.beat(sim, Sigma=Sigma)
        if Shyd is None:
            Shyd = sim.dust.S.hyd
    return Scoag + Shyd + Sext


def S_shrink(sim, Sigma=None):
    """Function calculates the source terms from dust shrinkage

    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    Sigma : Field, optional, default : None
        Surface density to be used if not None

    Returns
    -------
    Sshrink : Field
        Source terms from dust shrinkage"""
    if Sigma is None:
        Sigma = sim.dust.Sigma
    return dust_f.sig_deriv_shrink(
        sim.dust.Sigma,
        sim.dust.s.min,
        sim.dust.s.max,
        sim.dust.s.lim,
        sim.dust.qrec,
        sim.t.stepsize,
        sim.dust.f.crit,
        sim.dust.s.sdot_shrink)

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
    return dust_f.vrel_brownian_motion(sim.gas.cs, sim.dust.m, sim.gas.T,sim.gas.mu)

def vrel_radial_drift(sim):
    """Function calculates the relative particle velocities due to Brownian motion.
    The maximum value is set to the sound speed.
    Different from dustpy to be cosistent with pluto version
    Only cosidders the bigger of the two sizes provided

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    vrel : Field
        Relative velocities"""
    return dust_f.vrel_radial_drift(sim.dust.v.driftmax, sim.dust.St)

def vrel_azimuthal_drift(sim):
    """Function calculates the relative particle velocities due to azimuthal drift.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    vrel : Field
        Relative velocities"""
    return dp_dust_f.vrel_azimuthal_drift(sim.dust.v.driftmax, sim.dust.St)



def vrel_vertical_settling(sim):
    """Function calculates the relative particle velocities due to vertical settling.
        Different from dustpy to be cosistent with pluto version
    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    vrel : Field
        Relative velocities"""
    return dust_f.vrel_vertical_settling(sim.dust.H, sim.grid.OmegaK, sim.dust.St)



def q_eff(sim):
    """Function calculates the equilibrium exponent of the distribution.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    q_eff : Field
        Calculated exponent of distribution"""
    return sim.dust.q.frag * sim.dust.p.frag + sim.dust.q.sweep * (1.0 - sim.dust.p.frag)


def q_frag(sim):
    """
    Calculate the effective fragmentation power-law of the size distribution.

    Parameters:
    sim (Simulation): The simulation object containing the dust and gas properties.

    Returns:
    float: The effective fragmentation power-law of the size distribution.
    """
    return dust_f.qfrag(
        sim.dust.p.driftfrag,
        sim.dust.v.rel.tot[:, -1, -2],
        sim.dust.v.frag,
        sim.dust.St[:, -1],
        sim.dust.q.turb1,
        sim.dust.q.turb2,
        sim.dust.q.drfrag,
        sim.gas.alpha,
        sim.gas.Sigma,
        sim.gas.mu)

def q_rec(sim):
    """
    Function computes the power law exponent of the size distribution
    n(a) da = a^q da

    Parameters
    ----------
    Sigma : array-like
        Dust surface densities
    smin : array-like
        Minimum particle sizes
    smax : array-like
        Maximum particle sizes

    Returns
    -------
    q : array-like
        Size distribution exponent
    """
    return np.log(sim.dust.Sigma[..., 1]/sim.dust.Sigma[..., 0]) / np.log(np.sqrt(sim.dust.s.max/sim.dust.s.min)) - 4.


def p_frag_trans(sim):
    """
    Calculate the transition between the two turbulent regime.

    Parameters:
    sim (Simulation): The simulation object containing the dust and gas properties.

    Returns:
    float: Transition between the two turbulent regime.
    """

    return dust_f.pfrag_trans(sim.dust.St[:, -1],
        sim.gas.alpha,
        sim.gas.Sigma,
        sim.gas.mu)


def p_drift_frag(sim):
    """Calculate the fudge factor for the relative velocities.

    Parameters
    ----------
    sim : simulation frame
    """

    """  # Pfeil+2024, Eq. A.3
    dv_drmax = np.sqrt(
        sim.dust.v.rel.rad[:, -1, -2]**2 + sim.dust.v.rel.azi[:, -1, -2]**2)
    # where does the 0.3 come from
    st_mx    = sim.dust.St[...,-1]
    st_mn    = 0.5*st_mx
    Re = 0.5 * sim.gas.alpha * 2e-15 * sim.gas.Sigma/ sim.gas.mu 
    vgas     = (1.5*sim.gas.alpha)**0.5*sim.gas.cs
    vsmall   = vgas * ((st_mx-st_mn)/(st_mx+st_mn) * (st_mx**2./(st_mx+Re**-0.5) - st_mn**2./(st_mn+Re**-0.5)))**0.5
    vinter   = vgas * (2.292*st_mx)**0.5
    pint = p_frag_trans(sim)
    psmall = 1. - pint
    vtr_simp = psmall*vsmall + pint*vinter

    f_dt = np.where(dv_drmax != 0, 0.3 * vtr_simp / dv_drmax, 1e100)

    # Eq. A.4
    pdr = np.where( f_dt < 1e100, 0.5 + 0.5 * ((1.0 - f_dt**6) / (f_dt**6 + 1.0)),0.0)"""

    return dust_f.pdriftfrag(sim.dust.v.rel.rad[:,-1,-2],sim.dust.v.rel.azi[:,-1,-2],sim.dust.St[:,-1],sim.gas.alpha,sim.gas.Sigma,sim.gas.mu,sim.gas.cs,sim.dust.p.fragtrans)

def D_mod(sim):
    """Function calculates the dust diffusivity.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    D : Field
        Dust diffusivity

    Notes
    -----
    The diffusivity at the first and last two radial
    grid cells will be set to zero to avoid unwanted
    behavior at the boundaries."""
    # warning this is only done beacuse opf the pluto code -> times gammma factor athe the end
    v2 = sim.dust.delta.rad * sim.gas.cs**2
    Diff = dp_dust_f.d(v2, sim.grid.OmegaK, sim.dust.St*sim.dust.f.drift)
    Diff[:2, ...] = 0.
    Diff[-2:, ...] = 0.
    return Diff

def vrad_mod(sim):
    """Function calculated the radial velocity of the dust.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    vrad : Field
        Radial dust velocity"""
    return dp_dust_f.vrad(sim.dust.St*sim.dust.f.drift, sim.dust.v.driftmax, sim.gas.v.rad)



def Y_jacobian(sim, x, dx=None, *args, **kwargs):
    # Helper variables for convenience
    if dx is None:
        dt = x.stepsize
    else:
        dt = dx
    r = sim.grid.r
    ri = sim.grid.ri
    area = sim.grid.A
    Nr = int(sim.grid.Nr)
    Nm_s = int(sim.grid._Nm_short)

    # Getting the Jacobian of Sigma
    J_Sigma = sim.dust.Sigma.jacobian(x, dx=dt)

    # Building the hydrodynamic Jacobian of smax

    # We are advecting smax*Sigma[1], which is stored in Y
    smaxSig = sim.dust._Y[Nm_s * Nr:]
    smaxSig_rhs = smaxSig[...]

    # TODO: double check if diffusion works as intended for amax.
    # Creating the sparse matrix
    sim.dust.v.rad_flux.update()
    A, B, C = dp_dust_f.jacobian_hydrodynamic_generator(
        area,
        sim.dust.D[:, 2],
        r,
        ri,
        sim.gas.Sigma,
        sim.dust.v.rad_flux[:, 2]
    )
    # Setting boundary conditions for the Jacobian of smax*Sigma
    # The boundary condition is constant value on both boundaries

    row_max_adv = np.hstack((np.arange(Nr-1)+1 , np.arange(Nr), np.arange(Nr - 1)))
    col_max_adv= np.hstack((np.arange(Nr - 1), np.arange(Nr), np.arange(Nr - 1) + 1))
    dat_max_adv = np.hstack((A.ravel()[1:], B.ravel(), C.ravel()[:-1]))

    dat_in = np.zeros(3)
    row_in = np.zeros(3)
    col_in = np.arange(3)


    if sim.dust.s.boundary.inner.condition is not None:
        # Given value    
        if sim.dust.s.boundary.inner.condition == "const_grad":
            Di = ri[1] / ri[2] * (r[1] - r[0]) / (r[2] - r[0])
            K1 = - r[1] / r[0] * (1. + Di)
            K2 = r[2] / r[0] * Di
            dat_in[1] = -K1 / dt
            dat_in[2] = -K2 / dt
            smaxSig_rhs[0] = 0.     
        elif sim.dust.s.boundary.inner.condition == "val":
            #dust value times the maximal size at the time
            smaxSig_rhs[0] =  sim.dust.s.boundary.inner.value

        elif sim.dust.s.boundary.inner.condition == "const_val":
            # const_val
            dat_in[1] = 1. / dt
            smaxSig_rhs[0] = 0.
        elif sim.dust.s.boundary.inner.condition == "pow":
            p = sim.dust.s.boundary.inner.value
            dat_in[1] =  (r[0]/r[1])**p/dt
            smaxSig_rhs[0] = 0.

        elif sim.dust.s.boundary.inner.condition == "const_pow":
            p = np.log(smaxSig[2] / smaxSig[1]) / \
                np.log(r[2]/r[1])
            K1 = -(r[0]/r[1])**p
            dat_in[1] = -K1/dt
            smaxSig_rhs[0] = 0.


    #outer boundary
    dat_out = np.zeros(3)
    row_out = np.zeros(3)+(Nr-1)
    col_out = np.arange(0, -3, -1)+(Nr-1)

    if sim.dust.boundary.outer is not None:
        # Given value    
        
        if sim.dust.s.boundary.outer.condition == "const_grad":
            Do = ri[-2] / ri[-3] * (r[-1] - r[-2]) / (r[-2] - r[-3])
            KNrm2 = - r[-2] / r[-1] * (1. + Do)
            KNrm3 = r[-3] / r[-1] * Do
            dat_out[1] = -KNrm2 / dt
            dat_out[2] = -KNrm3 / dt
            smaxSig_rhs[-1] = 0.    
        elif sim.dust.s.boundary.outer.condition == "val":
            #dust value times the maximal size at the time
            smaxSig_rhs[-1] =  sim.dust.s.boundary.outer.value
        elif sim.dust.s.boundary.outer.condition == "const_val":
            #const_val
            dat_out[1] = 1. / dt
            smaxSig_rhs[-1] = 0.
        # to do pow and const_pow

    # Stitching together the generators
    row = np.hstack((row_max_adv, row_in, row_out))
    col = np.hstack((col_max_adv, col_in, col_out))
    dat = np.hstack((dat_max_adv, dat_in, dat_out))

    gen = (dat, (row, col))
    # Building sparse matrix of coagulation Jacobian
    J_smax_hyd = sp.csc_matrix(
        gen,
        shape=(Nr, Nr)
    )

    # Stitching together both matrices
    # This is the fastest possibility I could find
    J = J_Sigma.copy()
    J.data = np.hstack((J_Sigma.data, J_smax_hyd.data))
    J.indices = np.hstack(
        (J_Sigma.indices, J_smax_hyd.indices + J_Sigma.shape[0]))
    J.indptr = np.hstack(
        (J_Sigma.indptr, J_smax_hyd.indptr[1:] + len(J_Sigma.data)))
    Ntot = J_Sigma.shape[0] + J_smax_hyd.shape[0]
    J._shape = (Ntot, Ntot)

    # Stitching together the right hand sides of the equations
    Sigma_rhs = sim.dust._rhs[...]
    sim.dust._Y_rhs[:] = np.hstack((Sigma_rhs, smaxSig_rhs))

    return J


def _f_impl_1_direct(x0, Y0, dx, jac=None, rhs=None, *args, **kwargs):
    """Implicit 1st-order integration scheme with direct matrix inversion
    Parameters
    ----------
    x0 : Intvar
        Integration variable at beginning of scheme
    Y0 : Field
        Variable to be integrated at the beginning of scheme
    dx : IntVar
        Stepsize of integration variable
    jac : Field, optional, defaul : None
        Current Jacobian. Will be calculated, if not set
    args : additional positional arguments
    kwargs : additional keyworda arguments
    Returns
    -------
    dY : Field
        Delta of variable to be integrated
    Butcher tableau
    ---------------
     1 | 1
    ---|---
       | 1
    """
    if jac is None:
        jac = Y0.jacobian(x0, dx)
    if rhs is None:
        rhs = np.array(Y0.ravel())

    # Add external/explicit source terms to right-hand side
    dust = Y0._owner.dust

    # Note: s.max.derivative also sets s.sdot_shrink which is used below
    s_max_deriv = dust.s.max.derivative()


    S_Sigma_ext = np.zeros_like(dust.Sigma)
    S_Sigma_ext[1:-1, ...] += dust.S.ext[1:-1, ...]
    

    # smax*Sigma (product rule)
    S_smax_expl = np.zeros_like(dust.s.max)
    S_smax_expl[1:-1] = s_max_deriv[1:-1] * dust.Sigma[1:-1, 1] \
        + (dust.S.ext[1:-1,1] + dust.S.coag[1:-1,1]) * dust.s.max[1:-1] 
    # Stitching both parts together
    S = np.hstack((S_Sigma_ext.ravel(), S_smax_expl))

    # Right hand side
    rhs[...] += dx * S 

    N = jac.shape[0]
    eye = sp.identity(N, format="csc")

    A = eye - dx * jac

    A_LU = sp.linalg.splu(A,
                          permc_spec="MMD_AT_PLUS_A",
                          diag_pivot_thresh=0.0,
                          options=dict(SymmetricMode=True))
    Y1_ravel = A_LU.solve(rhs)

    Y1 = Y1_ravel.reshape(Y0.shape)

    return Y1 - Y0


class impl_1_direct(Scheme):
    """Modified class for implicit dust integration."""

    def __init__(self):
        super().__init__(_f_impl_1_direct, description="Implicit 1st-order direct solver")
