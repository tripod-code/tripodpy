from twopoppy.simulation import Simulation
from types import SimpleNamespace
import numpy as np
from dustpy.std import dust_f as dp_dust_f
from scipy.interpolate import interp1d
from simframe.io.writers import hdf5writer
import os


def _readdata_tpp(data, filename="data", extension="hdf5"):
    ret = {}

    if isinstance(data, Simulation):

        r = data.grid.r[None, ...]
        ri = data.grid.ri[None, ...]
        Nr = data.grid.Nr[None, ...]
        t = data.t[None, ...]
        Nt = np.array([1])[None, ...]

        SigmaDust = data.dust.Sigma[None, ...]
        SigmaGas = data.gas.Sigma[None, ...]
        eps = data.dust.eps[None, ...]

        cs = data.gas.cs[None, ...]
        delta = data.dust.delta.turb[None, ...]
        OmegaK = data.grid.OmegaK[None, ...]
        vK = OmegaK[None, ...] * r
        vFrag = data.dust.v.frag[None, ...]
        smax = data.dust.s.max[None, ...]
        xi = data.dust.xi.calc[None, ...]
        rhos = data.dust.rhos[None, ...]
        fill = data.dust.fill[None, ...]
        mfp = data.gas.mfp[None, ...]

        try:
            SigmaPlanet = data.planetesimals.Sigma[None, ...]
        except:
            pass

    elif os.path.isdir(data):

        writer = hdf5writer()

        # Setting up writer
        writer.datadir = data
        writer.extension = extension
        writer.filename = filename

        r = writer.read.sequence("grid.r")
        ri = writer.read.sequence("grid.ri")
        Nr = writer.read.sequence("grid.Nr")
        t = writer.read.sequence("t")
        Nt = np.array([len(t)])

        SigmaDust = writer.read.sequence("dust.Sigma")
        SigmaGas = writer.read.sequence("gas.Sigma")
        eps = writer.read.sequence("dust.eps")

        cs = writer.read.sequence("gas.cs")
        delta = writer.read.sequence("dust.delta.turb")
        OmegaK = writer.read.sequence("grid.OmegaK")
        vK = OmegaK * r
        vFrag = writer.read.sequence("dust.v.frag")
        smax = writer.read.sequence("dust.s.max")
        xi = writer.read.sequence("dust.xi.calc")
        rhos = writer.read.sequence("dust.rhos")
        fill = writer.read.sequence("dust.fill")
        mfp = writer.read.sequence("gas.mfp")

        try:
            SigmaPlanet = writer.read.sequence("planetesimals.Sigma")
        except:
            pass

    else:

        raise RuntimeError("Unknown data type.")

    # Masses
    Mgas = (np.pi * (ri[..., 1:] ** 2 - ri[..., :-1] ** 2) * SigmaGas[...]).sum(-1)
    Mdust = (np.pi * (ri[..., 1:] ** 2 - ri[..., :-1] ** 2)
             * SigmaDust[...].sum(-1)).sum(-1)
    try:
        Mplanet = (np.pi * (ri[..., 1:] ** 2 - ri[..., :-1] ** 2) * SigmaPlanet[...]).sum(-1)
    except:
        pass

    mi, SigmaDusti, Sti = powerlaw_extrapolation(SigmaDust, smax, xi, rhos, fill, SigmaGas, mfp,
                                                 nmbpd=7, mmin=1.e-12, mmax=1.e8)

    # Transformation of the density distribution
    a = np.array(np.mean(mi[..., 1:] / mi[..., :-1], axis=-1))
    dm = np.array(2. * (a - 1.) / (a + 1.))
    sigmaDusti = SigmaDusti / dm[..., None]

    Nmi = np.ones_like(Nr) * len(mi.sum(-1)[0])

    # Fragmentation limit
    b = vFrag ** 2 / (delta * cs ** 2)
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings(
            'ignore',
            r'invalid value encountered in sqrt')
        StFr = 1 / (2 * b) * (3 - np.sqrt(9 - 4 * b ** 2))

    # Drift limit
    p = SigmaGas * OmegaK * cs / np.sqrt(2. * np.pi)
    StDr = np.zeros_like(StFr)
    for i in range(int(Nt)):
        _f = interp1d(np.log10(r[i, ...]), np.log10(
            p[i, ...]), fill_value='extrapolate')
        pi = 10. ** _f(np.log10(ri[i, ...]))
        gamma = np.abs(r[i, ...] / p[i, ...] *
                       np.diff(pi) / np.diff(ri[i, ...]))
        StDr[i, ...] = eps[i, ...] / gamma * (vK[i, ...] / cs[i, ...]) ** 2

    # Drift fragmentation limit
    N = 0.5
    StDf = vFrag * vK / (gamma * cs ** 2 * (1 - N))

    # Stokes of maximum particle size
    StMax = np.zeros_like(StFr)
    rho = rhos * fill
    for i in range(int(Nt)):
        StMax[i] = dp_dust_f.st_epstein_stokes1(smax[i], mfp[i], rho[i, :, 0], SigmaGas[i])

    ret["mi"] = mi
    ret["Nmi"] = Nmi
    ret["r"] = r
    ret["ri"] = ri
    ret["Nr"] = Nr
    ret["t"] = t
    ret["Nt"] = Nt

    ret["SigmaDusti"] = SigmaDusti
    ret["sigmaDusti"] = sigmaDusti
    ret["SigmaGas"] = SigmaGas
    ret["eps"] = eps

    ret["Mdust"] = Mdust
    ret["Mgas"] = Mgas

    ret["cs"] = cs
    ret["delta"] = delta
    ret["OmegaK"] = OmegaK
    ret["Sti"] = Sti
    ret["StDr"] = StDr
    ret["StFr"] = StFr
    ret["StDf"] = StDf
    ret["StMax"] = StMax
    ret["vK"] = vK
    ret["vFrag"] = vFrag

    try:
        ret["SigmaPlanet"] = SigmaPlanet
        ret["Mplanet"] = Mplanet
    except:
        pass

    return SimpleNamespace(**ret)


def _readdata_dp(data, filename="data", extension="hdf5"):
    ret = {}

    if isinstance(data, Simulation):

        m = data.grid.m[None, ...]
        Nm = data.grid.Nm[None, ...]
        r = data.grid.r[None, ...]
        ri = data.grid.ri[None, ...]
        Nr = data.grid.Nr[None, ...]
        t = data.t[None, ...]
        Nt = np.array([1])[None, ...]

        SigmaDust = data.dust.Sigma[None, ...]
        SigmaGas = data.gas.Sigma[None, ...]
        eps = data.dust.eps[None, ...]

        cs = data.gas.cs[None, ...]
        delta = data.dust.delta.turb[None, ...]
        OmegaK = data.grid.OmegaK[None, ...]
        St = data.dust.St[None, ...]
        vK = OmegaK[None, ...] * r
        vFrag = data.dust.v.frag[None, ...]

        try:
            SigmaPlanet = data.planetesimals.Sigma[None, ...]
        except:
            pass

    elif os.path.isdir(data):

        writer = hdf5writer()

        # Setting up writer
        writer.datadir = data
        writer.extension = extension
        writer.filename = filename

        m = writer.read.sequence("grid.m")
        Nm = writer.read.sequence("grid.Nm")
        r = writer.read.sequence("grid.r")
        ri = writer.read.sequence("grid.ri")
        Nr = writer.read.sequence("grid.Nr")
        t = writer.read.sequence("t")
        Nt = np.array([len(t)])

        SigmaDust = writer.read.sequence("dust.Sigma")
        SigmaGas = writer.read.sequence("gas.Sigma")
        eps = writer.read.sequence("dust.eps")

        cs = writer.read.sequence("gas.cs")
        delta = writer.read.sequence("dust.delta.turb")
        OmegaK = writer.read.sequence("grid.OmegaK")
        St = writer.read.sequence("dust.St")
        vK = OmegaK * r
        vFrag = writer.read.sequence("dust.v.frag")

        try:
            SigmaPlanet = writer.read.sequence("planetesimals.Sigma")
        except:
            pass

    else:

        raise RuntimeError("Unknown data type.")

    # Masses
    Mgas = (np.pi * (ri[..., 1:] ** 2 - ri[..., :-1] ** 2) * SigmaGas[...]).sum(-1)
    Mdust = (np.pi * (ri[..., 1:] ** 2 - ri[..., :-1] ** 2)
             * SigmaDust[...].sum(-1)).sum(-1)
    try:
        Mplanet = (np.pi * (ri[..., 1:] ** 2 - ri[..., :-1] ** 2) * SigmaPlanet[...]).sum(-1)
    except:
        pass

    # Transformation of the density distribution
    a = np.array(np.mean(m[..., 1:] / m[..., :-1], axis=-1))
    dm = np.array(2. * (a - 1.) / (a + 1.))
    sigmaDust = SigmaDust[...] / dm[..., None, None]

    # Fragmentation limit
    b = vFrag ** 2 / (delta * cs ** 2)
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings(
            'ignore',
            r'invalid value encountered in sqrt')
        StFr = 1 / (2 * b) * (3 - np.sqrt(9 - 4 * b ** 2))

    # Drift limit
    p = SigmaGas * OmegaK * cs / np.sqrt(2. * np.pi)
    StDr = np.zeros_like(StFr)
    for i in range(int(Nt)):
        _f = interp1d(np.log10(r[i, ...]), np.log10(
            p[i, ...]), fill_value='extrapolate')
        pi = 10. ** _f(np.log10(ri[i, ...]))
        gamma = np.abs(r[i, ...] / p[i, ...] *
                       np.diff(pi) / np.diff(ri[i, ...]))
        StDr[i, ...] = eps[i, ...] / gamma * (vK[i, ...] / cs[i, ...]) ** 2

    ret["m"] = m
    ret["Nm"] = Nm
    ret["r"] = r
    ret["ri"] = ri
    ret["Nr"] = Nr
    ret["t"] = t
    ret["Nt"] = Nt

    ret["SigmaDust"] = SigmaDust
    ret["sigmaDust"] = sigmaDust
    ret["SigmaGas"] = SigmaGas
    ret["eps"] = eps

    ret["Mdust"] = Mdust
    ret["Mgas"] = Mgas

    ret["cs"] = cs
    ret["delta"] = delta
    ret["OmegaK"] = OmegaK
    ret["St"] = St
    ret["StDr"] = StDr
    ret["StFr"] = StFr
    ret["vK"] = vK
    ret["vFrag"] = vFrag

    try:
        ret["SigmaPlanet"] = SigmaPlanet
        ret["Mplanet"] = Mplanet
    except:
        pass

    return SimpleNamespace(**ret)


def powerlaw_extrapolation(sigma_d, s_max, xi, rhos, fill, sigma_g, mfp, nmbpd=7, mmin=1.e-12, mmax=1.e8):
    nt = len(s_max.sum(-1))
    nr = len(sigma_d.sum(-1)[0])
    logmmin = np.log10(mmin)
    logmmax = np.log10(mmax)
    decades = np.ceil(logmmax - logmmin)
    nm = int(decades * nmbpd) + 1

    sigma_d = sigma_d.sum(-1)
    sig_dm = np.zeros([nt, nr, nm]) + 1.e-100

    m_i = np.logspace(logmmin, logmmax, nm, base=10.)
    m_i = np.full((nt, nr, nm), m_i)

    rho = rhos * fill
    rho_i = np.full((nt, nr, nm), rho[0, 0, 0])

    m_max = 4. / 3. * np.pi * rho_i[:, :, 0] * s_max ** 3.

    for it in range(nt):
        for ir in range(nr):
            if m_max[it, ir] <= mmin:
                sig_dm[it, ir, 0] = 1.
            else:
                i_up = np.where(m_i[it, ir] < m_max[it, ir])[0][-1]
                # filling all bins that are strictly below m_max
                if xi[it, ir] == 4.0:
                    for im in range(i_up):
                        sig_dm[it, ir, im] = np.log(m_i[it, ir, im + 1] / m_i[it, ir, im])
                    # filling the bin that contains m_max
                    sig_dm[it, ir, i_up] = np.log(m_max[it, ir] / m_i[it, ir, i_up])
                else:
                    for im in range(i_up):
                        sig_dm[it, ir, im] = m_i[it, ir, im + 1] ** ((4. + xi[it, ir]) / 3.) - \
                                             m_i[it, ir, im] ** ((4. + xi[it, ir]) / 3.)
                    # filling the bin that contains m_max
                    sig_dm[it, ir, i_up] = m_max[it, ir] ** ((4. + xi[it, ir]) / 3.) - \
                                           m_i[it, ir, i_up] ** ((4. + xi[it, ir]) / 3.)
            # normalize
            sig_dm[it, ir, :] = sig_dm[it, ir, :] / sig_dm[it, ir, :].sum() * sigma_d[it, ir]

    a_i = (3. / (4. * np.pi * rho_i) * m_i) ** (1. / 3.)
    st_i = np.zeros((nt, nr, nm))
    for i in range(nt):
        st_i[i] = dp_dust_f.st_epstein_stokes1(a_i[i], mfp[i], rho_i[i], sigma_g[i])

    return m_i, sig_dm, st_i
