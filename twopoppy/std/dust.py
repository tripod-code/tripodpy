"""Module containing standard functions for the dust."""

import dustpy.constants as c
from dustpy.std import dust_f as dp_dust_f
import dustpy.std.dust as dp_dust
from twopoppy.std import dust_f

from simframe.integration import Scheme

import numpy as np

import scipy.sparse as sp


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
    dtSigma = dt_Sigma(sim)
    dtsmax = dt_smax(sim)
    return np.minimum(dtSigma, dtsmax)


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
        dt = sim.dust.Sigma[mask] / sim.dust.S.tot[mask]
        return np.min(np.abs(dt))
    return 1.e100


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
    # TODO: Which factor for maximum growth makes sense here?
    max_growth_fact = 2.

    # Helper variables. Ignoring boundaries.
    smax_dot = sim.dust.s.max.derivative()[1:-1]
    smax = sim.dust.s.max[1:-1]
    smin = sim.dust.s.min[1:-1]

    # Time step if smax is shrinking.
    # Value must not drop below smin
    mask1 = np.logical_and(smax_dot < 0., smax > smin)
    if np.any(mask1):
        rate1 = (smin[mask1] - smax[mask1]) / smax_dot[mask1]
        dt1 = np.min(np.abs(rate1))
    else:
        dt1 = 1.e100

    # Time step if smax is growing.
    # Value must not grow by more than the maximum growth factor
    mask2 = smax_dot > 0.
    if np.any(mask2):
        rate2 = (max_growth_fact - 1.) * smax[mask2] / smax_dot[mask2]
        dt2 = np.min(np.abs(rate2))
    else:
        dt2 = 1.e100
    return np.minimum(dt1, dt2)


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
    sim.dust.S.ext[0] = 0.
    sim.dust.S.ext[-1] = 0.
    # Storing current surface density
    sim.dust._SigmaOld[...] = sim.dust.Sigma[...]


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
    # Making sure smax is not smaller than smin
    sim.dust.s.max[...] = np.maximum(
        sim.dust.s.min, sim.dust._Y[Nr * Nm_s:] / sim.dust.Sigma[:, 1])

    dp_dust.boundary(sim)
    dp_dust.enforce_floor_value(sim)
    sim.dust.v.rad.update()
    sim.dust.Fi.update()
    sim.dust.S.coag.update()
    sim.dust.S.hyd.update()
    dp_dust.set_implicit_boundaries(sim)
    # TODO: Doing this here for testing for now
    # This boundary condition keeps smax constant
    sim.dust.s.max[0] = sim.dust.s.max[1]
    sim.dust.s.max[-1] = sim.dust.s.max[-2]
    # This boundary condition keeps xi constant
    # xip4 = 2. * np.log(sim.dust.Sigma[1, 1]/sim.dust.Sigma[1, 0]) / \
    #    np.log(sim.dust.s.max[1]/sim.dust.s.min[1])
    # sim.dust.s.max[0] = sim.dust.s.min[0] * \
    #    np.exp(2.*np.log(sim.dust.Sigma[0, 1]/sim.dust.Sigma[0, 0]) / xip4)
    # xip4 = 2. * np.log(sim.dust.Sigma[-2, 1]/sim.dust.Sigma[-2, 0]) / \
    #    np.log(sim.dust.s.max[-2]/sim.dust.s.min[-2])
    # sim.dust.s.max[-1] = sim.dust.s.min[-1] * \
    #    np.exp(2.*np.log(sim.dust.Sigma[-1, 1]/sim.dust.Sigma[-1, 0]) / xip4)

    # print(repr(sim.dust.s.max))
    # print(repr(sim.dust.s.min))
    # print(repr(sim.dust.Sigma))


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
        gamma = 1. / gamma
        # Maximum drift limited particle size with safety margin
        ad = 1.e-4 * 2. / np.pi * sim.ini.dust.d2gRatio * sim.gas.Sigma / sim.dust.fill[:, 0] \
             * sim.dust.rhos[:, 0] * (sim.grid.OmegaK * sim.grid.r) ** 2. / sim.gas.cs ** 2. / gamma
        aIni = np.minimum(sim.ini.dust.aIniMax, ad)
        # Enforce initial drift limit
        sim.dust.xi.calc = np.where(
            aIni < sim.ini.dust.aIniMax, sim.dust.xi.stick, sim.dust.xi.calc)

        return np.maximum(smin, aIni)

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
    xi = sim.dust.xi.calc
    xip4 = xi + 4.
    smin = sim.dust.s.min
    smax = sim.dust.s.max
    sint = np.sqrt(smin * smax)
    SigmaFloor = sim.dust.SigmaFloor

    # Values for xi != -4
    S0 = np.zeros_like(sim.grid.r)
    S1 = np.zeros_like(sim.grid.r)
    for i in range(int(sim.grid.Nr)):
        if smax[i] == smin[i]:
            S0[i] = SigmaFloor[i, 0]
            S1[i] = SigmaFloor[i, 1]
        else:
            S0[i] = (sint[i] ** xip4[i] - smin[i] ** xip4[i]) / \
                    (smax[i] ** xip4[i] - smin[i] ** xip4[i])
            S1[i] = 1. - S0[i]
    S = np.array([S0, S1]).T

    # Values for xi == -4
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

    Sigma = sim.ini.dust.d2gRatio * sim.gas.Sigma[:, None] * np.where(xi[:, None] == -4., S_4, S)

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
        sim.dust.a[:, :2],
        sim.dust.v.rel.tot[:, :2, :2],
        sim.dust.H[:, :2],
        sim.dust.m[:, :2],
        sim.dust.p.frag[:, 2:, 2:],
        sim.dust.p.stick[:, 2:, 2:],
        sim.dust.Sigma,
        sim.dust.s.min,
        sim.dust.s.max,
        sim.dust.xi.frag,
        sim.dust.xi.stick
    )

    # Getting data vector and coordinates for the hydrodynamic sparse matrix
    A, B, C = dp_dust_f.jacobian_hydrodynamic_generator(
        area,
        sim.dust.D[:, :2],
        r,
        ri,
        sim.gas.Sigma,
        sim.dust.v.rad[:, :2]
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
            sim.dust._rhs[:Nm_s] = - ri[1] / r[0] * (r[1] - r[0]) * sim.dust.boundary.inner.value
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
            sim.dust._rhs[-Nm_s:] = ri[-2] / r[-1] * (r[-1] - r[-2]) * sim.dust.boundary.outer.value
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
    return dust_f.calculate_a(sim.dust.s.min, sim.dust.s.max, sim.dust.xi.calc, sim.grid._Nm_long)


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
    return dust_f.fi_adv(Sigma, sim.dust.v.rad, sim.grid.r, sim.grid.ri)


def F_diff(sim, Sigma=None):
    """Function calculates the diffusive flux at the cell interfaces"""
    if Sigma is None:
        Sigma = sim.dust.Sigma

    Fi = dust_f.fi_diff(sim.dust.D,
                        Sigma,
                        sim.gas.Sigma,
                        sim.dust.St,
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

    return dust_f.smax_deriv(
        sim.dust.v.rel.tot[:, 2, 3],
        sim.dust.rho[:, :2],
        sim.dust.rhos[:, :2],
        sim.dust.s.min,
        smax,
        sim.dust.v.frag
    )


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

    return dust_f.s_coag(
        sim.dust.a[:, :2],
        sim.dust.v.rel.tot[:, :2, :2],
        sim.dust.H[:, :2],
        sim.dust.m[:, :2],
        sim.dust.p.frag[:, 2:, 2:],
        sim.dust.p.stick[:, 2:, 2:],
        Sigma,
        sim.dust.s.min,
        sim.dust.s.max,
        sim.dust.xi.frag,
        sim.dust.xi.stick
    )


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
    return dust_f.calculate_xi(sim.dust.s.min, sim.dust.s.max, sim.dust.Sigma)


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

    # Creating the sparse matrix
    A, B, C = dp_dust_f.jacobian_hydrodynamic_generator(
        area,
        sim.dust.D[:, 1],
        r,
        ri,
        smaxSig,
        sim.dust.v.rad[:, 1]
    )
    # Setting boundary conditions for the Jacobian of smax*Sigma
    # The boundary condition is constant value on both boundaries
    B[0, 0] = 0.
    C[0, 0] = 1. / dt
    B[-1, 0] = 0.
    A[-1, 0] = 1. / dt
    # Building the matrix
    J_smax_hyd = sp.diags(
        (A[1:, 0], B[:, 0], C[:-1, 0]),
        offsets=(-1, 0, 1),
        shape=(Nr, Nr),
        format="csc"
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
    smaxSig_rhs = smaxSig[...]
    smaxSig_rhs[0] = 0.
    smaxSig_rhs[-1] = 0.
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
    # Sigma
    S_Sigma_ext = np.zeros_like(Y0._owner.dust.Sigma)
    S_Sigma_ext[1:-1, ...] = Y0._owner.dust.S.ext[1:-1, ...]
    # smax*Sigma (product rule)
    S_smax_expl = np.zeros_like(Y0._owner.dust.s.max)
    S_smax_expl[1:-1] = Y0._owner.dust.s.max.derivative()[1:-1] * Y0._owner.dust.Sigma[1:-1, 1] \
                        + S_Sigma_ext[1:-1, 1] * Y0._owner.dust.s.max[1:-1]
    # Stitching both parts together
    S = np.hstack((S_Sigma_ext.ravel(), S_smax_expl))

    # Right hand side
    rhs[...] += dx * S

    N = jac.shape[0]
    eye = sp.identity(N, format="csc")

    A = eye - dx[0] * jac

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
