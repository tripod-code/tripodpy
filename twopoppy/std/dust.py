"""Module containing standard functions for the dust."""


import dustpy.constants as c
from dustpy.std import dust_f as dp_dust_f
import dustpy.std.dust as dp_dust
from twopoppy.std import dust_f

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
    smax_dot = sim.dust.s.max.derivative()[1:-1]  # Ignoring boundaries
    smax = sim.dust.s.max[1:-1]  # Ignoring boundaries
    smin = sim.dust.s.min[1:-1]  # Ignoring boundaries
    # Time step if smax is shrinking.
    # Value must not drop below smin
    if np.any(smax_dot < 0.):
        mask1 = np.logical_and(smax > smin, smax_dot < 0.)
        rate1 = (smin[mask1] - smax[mask1]) / smax_dot[mask1]
        dt1 = np.min(np.abs(rate1))
    else:
        dt1 = 1.e100
    # Time step if smax is growing.
    # Value must not grow by more than the maximum growth factor
    if np.any(smax_dot > 0.):
        mask2 = np.where(smax_dot > 0.)
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
    dp_dust.boundary(sim)
    dp_dust.enforce_floor_value(sim)
    sim.dust.v.rad.update()
    sim.dust.Fi.update()
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


def SigmaFloor(sim):
    """Function calculates the floor value for the dust distribution. Floor value means that there is less than
    one particle in an annulus.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    Sigma_floor : Field
        Floor value of surface density"""
    area = c.pi * (sim.grid.ri[1:]**2. - sim.grid.ri[:-1]**2.)
    return sim.dust.m[:, :2] / area[..., None]


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
    xi = sim.dust.xi.calc
    xip4 = xi + 4.
    smax = sim.dust.s.max
    smin = sim.dust.s.min
    sint = np.sqrt(sim.dust.s.min * sim.dust.s.max)

    # Values for xi != -4
    S0 = (sint**xip4 - smin**xip4) / (smax**xip4 - smin**xip4)
    S1 = 1. - S0
    S = np.array([S0, S1]).T

    # Values for xi == -4
    S0_4 = np.log(sint / smin) / np.log(smax / smin)
    S1_4 = 1. - S0_4
    S_4 = np.array([S0_4, S1_4]).T

    Sigma = sim.ini.dust.d2gRatio * sim.gas.Sigma[:, None] \
            * np.where(xi[:, None] == -4., S_4, S)

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
        ad = 1.e-4 * 2./np.pi * sim.ini.dust.d2gRatio * sim.gas.Sigma \
            / sim.dust.fill[:, 0] * sim.dust.rhos[:, 0] * (sim.grid.OmegaK * sim.grid.r)**2. \
            / sim.gas.cs**2. / gamma
        aIni = np.minimum(sim.ini.dust.aIniMax, ad)[:, None]
        # Set surface densities of initially drifting particles to floor value
        Sigma = np.where(sim.dust.a[:, :2] > aIni, 0.1 * sim.dust.SigmaFloor, Sigma)

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

    # Building coagulation Jacobian

    # Total problem size
    Ntot = int((Nr*Nm_s))

    # Building the hydrodynamic Jacobian
    # TODO: Check this call
    A, B, C = dp_dust_f.jacobian_hydrodynamic_generator(
        area,
        sim.dust.D[:, :2],
        r,
        ri,
        sim.gas.Sigma,
        sim.dust.v.rad[:, :2]
    )
    J_hyd = sp.diags(
        (A.ravel()[Nm_s:], B.ravel(), C.ravel()[:-Nm_s]),
        offsets=(-Nm_s, 0, Nm_s),
        shape=(Ntot, Ntot),
        format="csc"
    )

    # Right-hand side
    sim.dust._rhs[Nm_s:-Nm_s] = sim.dust.Sigma.ravel()[Nm_s:-Nm_s]

    # BOUNDARIES

    # Inner boundary

    # Initializing data and coordinate vectors for sparse matrix
    dat = np.zeros(int(3.*Nm_s))
    row0 = np.arange(int(Nm_s))
    col0 = np.arange(int(Nm_s))
    col1 = np.arange(int(Nm_s)) + Nm_s
    col2 = np.arange(int(Nm_s)) + 2.*Nm_s
    row = np.concatenate((row0, row0, row0))
    col = np.concatenate((col0, col1, col2))

    # Filling data vector depending on boundary condition
    if sim.dust.boundary.inner is not None:
        # Given value
        if sim.dust.boundary.inner.condition == "val":
            sim.dust._rhs[:Nm_s] = sim.dust.boundary.inner.value
        # Constant value
        elif sim.dust.boundary.inner.condition == "const_val":
            dat[Nm_s:2*Nm_s] = 1./dt
            sim.dust._rhs[:Nm_s] = 0.
        # Given gradient
        elif sim.dust.boundary.inner.condition == "grad":
            K1 = - r[1]/r[0]
            dat[Nm_s:2*Nm_s] = -K1/dt
            sim.dust._rhs[:Nm_s] = - ri[1]/r[0] * \
                (r[1]-r[0])*sim.dust.boundary.inner.value
        # Constant gradient
        elif sim.dust.boundary.inner.condition == "const_grad":
            Di = ri[1]/ri[2] * (r[1]-r[0]) / (r[2]-r[0])
            K1 = - r[1]/r[0] * (1. + Di)
            K2 = r[2]/r[0] * Di
            dat[:Nm_s] = 0.
            dat[Nm_s:2*Nm_s] = -K1/dt
            dat[2*Nm_s:] = -K2/dt
            sim.dust._rhs[:Nm_s] = 0.
        # Given power law
        elif sim.dust.boundary.inner.condition == "pow":
            p = sim.dust.boundary.inner.value
            sim.dust._rhs[:Nm_s] = sim.dust.Sigma[1] * (r[0]/r[1])**p
        # Constant power law
        elif sim.dust.boundary.inner.condition == "const_pow":
            p = np.log(sim.dust.Sigma[2] /
                       sim.dust.Sigma[1]) / np.log(r[2]/r[1])
            K1 = - (r[0]/r[1])**p
            dat[Nm_s:2*Nm_s] = -K1/dt
            sim.dust._rhs[:Nm_s] = 0.

    # Creating sparce matrix for inner boundary
    gen = (dat, (row, col))
    J_in = sp.csc_matrix(
        gen,
        shape=(Ntot, Ntot)
    )

    # Outer boundary

    # Initializing data and coordinate vectors for sparse matrix
    dat = np.zeros(int(3.*Nm_s))
    row0 = np.arange(int(Nm_s))
    col0 = np.arange(int(Nm_s))
    col1 = np.arange(int(Nm_s)) - Nm_s
    col2 = np.arange(int(Nm_s)) - 2.*Nm_s
    offset = (Nr-1)*Nm_s
    row = np.concatenate((row0, row0, row0)) + offset
    col = np.concatenate((col0, col1, col2)) + offset

    # Filling data vector depending on boundary condition
    if sim.dust.boundary.outer is not None:
        # Given value
        if sim.dust.boundary.outer.condition == "val":
            sim.dust._rhs[-Nm_s:] = sim.dust.boundary.outer.value
        # Constant value
        elif sim.dust.boundary.outer.condition == "const_val":
            dat[-2*Nm_s:-Nm_s] = 1./dt
            sim.dust._rhs[-Nm_s:] = 0.
        # Given gradient
        elif sim.dust.boundary.outer.condition == "grad":
            KNrm2 = -r[-2]/r[-1]
            dat[-2*Nm_s:-Nm_s] = -KNrm2/dt
            sim.dust._rhs[-Nm_s:] = ri[-2]/r[-1] * \
                (r[-1]-r[-2])*sim.dust.boundary.outer.value
        # Constant gradient
        elif sim.dust.boundary.outer.condition == "const_grad":
            Do = ri[-2]/ri[-3] * (r[-1]-r[-2]) / (r[-2]-r[-3])
            KNrm2 = - r[-2]/r[-1] * (1. + Do)
            KNrm3 = r[-3]/r[-1] * Do
            dat[-2*Nm_s:-Nm_s] = -KNrm2/dt
            dat[-3*Nm_s:-2*Nm_s] = -KNrm3/dt
            sim.dust._rhs[-Nm_s:] = 0.
        # Given power law
        elif sim.dust.boundary.outer.condition == "pow":
            p = sim.dust.boundary.outer.value
            sim.dust._rhs[-Nm_s:] = sim.dust.Sigma[-2] * (r[-1]/r[-2])**p
        # Constant power law
        elif sim.dust.boundary.outer.condition == "const_pow":
            p = np.log(sim.dust.Sigma[-2] /
                       sim.dust.Sigma[-3]) / np.log(r[-2]/r[-3])
            KNrm2 = - (r[-1]/r[-2])**p
            dat[-2*Nm_s:-Nm_s] = -KNrm2/dt
            sim.dust._rhs[-Nm_s:] = 0.

    # Creating sparce matrix for outer boundary
    gen = (dat, (row, col))
    J_out = sp.csc_matrix(
        gen,
        shape=(Ntot, Ntot)
    )

    # Adding and returning all matrix components
    return J_in + J_hyd + J_out


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
                        np.sqrt(sim.dust.delta.rad*sim.gas.cs**2),
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
        Sigma = sim.dust.Sigma
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
    smax : Field
        Current smax

    Returns
    -------
    smax_dot: Field
        Derivative of smax"""
    vfrag = sim.dust.v.frag
    dv = sim.dust.v.rel.tot[:, 2, 3]
    exponent = 8.
    A = (dv / vfrag)**exponent
    B = (1. - A) / (1. + A)

    rho = sim.dust.rho[:, :2]
    rhos = sim.dust.rhos[:, :2]
    rhod = rho.sum(-1)
    rhos_mean = (rho * rhos).sum(-1) / rhod

    # additional modification of growth to ensure minimal distribution width is not surpassed
    smin = sim.dust.s.min
    smax = sim.dust.s.max
    minimum = 2. * smin
    threshold = 1.35 * minimum
    factor = np.where(smax <= threshold, np.exp(-100. * (smax/threshold-1.)**2.), 1.)
    factor = np.where(smax <= minimum, 0., factor)

    # apply transition factor only to reduce decline if growth is negative
    smax_dot = rhod / rhos_mean * dv * B
    smax_dot = np.where(smax_dot < 0., smax_dot * factor, smax_dot)

    return smax_dot


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

    # Helper variables
    sigma = np.pi * (sim.dust.a[:, :, None]**2 + sim.dust.a[:, None, :]**2)
    H = sim.dust.H
    dv = sim.dust.v.rel.tot
    m = sim.dust.m
    smax = sim.dust.s.max
    sint = np.sqrt(sim.dust.s.min * sim.dust.s.max)
    xifrag = sim.dust.xi.frag
    xistick = sim.dust.xi.stick
    pfrag = sim.dust.p.frag
    pstick = sim.dust.p.stick

    debug = True

    if debug:

        nan = False

        if (H == 0.).any():
            print("H")
            nan = True

        if (sigma == 0.).any():
            print("sigma")
            nan = True

        if (dv == 0.).any():
            print("dv")
            nan = True

        if (m == 0.).any():
            print("m")
            nan = True

        if (sint == 0.).any():
            print("sint")
            nan = True

        if nan:
            import sys
            sys.exit()

    xiprime = pfrag * xifrag[:, None, None] + pstick * xistick[:, None, None]

    F = H[:, 1] * np.sqrt(2. / (H[:, 0]**2 + H[:, 1]**2)) \
        * sigma[:, 0, 1] / sigma[:, 1, 1] * dv[:, 0, 1] / dv[:, 1, 1] \
        * (smax / sint)**(-xiprime[:, 1, 1] - 4.)

    dot01 = Sigma[:, 0] * Sigma[:, 1] * sigma[:, 0, 1] * dv[:, 0, 1] \
            / (sim.dust.m[:, 1] * np.sqrt(2. * np.pi * (H[:, 0]**2 + H[:, 1]**2)))

    dot10 = Sigma[:, 1]**2 * sigma[:, 1, 1] * dv[:, 1, 1] * F \
            / (2. * np.sqrt(np.pi) * m[:, 1] * H[:, 1])

    Scoag = np.empty_like(sim.dust.Sigma)
    Scoag[:, 0] = dot10 - dot01
    Scoag[:, 1] = - Scoag[:, 0]

    return Scoag


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
    sint = np.sqrt(sim.dust.s.min * sim.dust.s.max)
    # print(repr(np.log(sim.dust.Sigma[:, 1]/sim.dust.Sigma[:, 0])))
    # print(repr(sim.dust.s.max / sint))
    # print(repr(sim.dust.s.max / sint))
    return np.log(sim.dust.Sigma[:, 1] / sim.dust.Sigma[:, 0]) / np.log(sim.dust.s.max / sint) - 4.
