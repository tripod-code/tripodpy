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
        smin = data.dust.s.min[None, ...]
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
        smin = writer.read.sequence("dust.s.min")
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

    # Interpolation of the density distribution over mass grid
    # via distribution exponent
    Nmbpd = 7
    mmin_ini = 1.e-12
    mmax_ini = 1.e8
    logmmin = np.log10(mmin_ini)
    logmmax = np.log10(mmax_ini)
    decades = np.ceil(logmmax - logmmin)
    Nm = int(decades * Nmbpd) + 1
    Nmi = np.array([1])[None, ...] * Nm
    Nr_len = Nr[0]

    # Assumption: Particles of all sizes have same mass density
    rho = rhos * fill
    rho = np.full((int(Nt), int(Nr_len), int(Nmi)), rho[0, 0, 0])
    mmin = 4. / 3. * np.pi * rho[:, :, 0] * smin ** 3
    mmax = 4. / 3. * np.pi * rho[:, :, 0] * smax ** 3

    mi = np.logspace(np.log10(mmin_ini), np.log10(mmax_ini), num=Nm, base=10.)
    mi0 = mi[:-1]
    mi1 = mi[1:]
    mi_int = np.zeros(Nm + 1)
    for i in range(Nm - 1):
        j = int(i + 1)
        mi_int[j] = np.sqrt(mi0[i] * mi1[i])
    mi_int[0] = min(mi[0] - (mi_int[1] - mi[0]), mmin.min())
    mi_int[-1] = max(mi[-1] + (mi[-1] - mi_int[-2]), mmax.max())
    mi_int0 = mi_int[:-1]
    mi_int1 = mi_int[1:]
    mi = np.full(int(Nmi), mi)
    mi = np.full((int(Nt), int(Nmi)), mi)

    SigmaDustTot = SigmaDust[...].sum(-1)
    SigmaDusti = np.ones((int(Nt), int(Nr_len), int(Nmi))) * 1e-100
    for i in range(int(Nt)):
        for j in range(int(Nr_len)):
            for k in range(int(Nmi)):
                if mi_int1[k] <= mmax[i, j]:
                    if xi[i, j] != -4.:
                        expo = (xi[i, j] + 4.) / 3.
                        SigmaDusti[i, j, k] = SigmaDustTot[i, j] * \
                                              (mi_int1[k] ** expo - mi_int0[k] ** expo) / \
                                              (mmax[i, j] ** expo - mmin[i, j] ** expo)
                    else:
                        SigmaDusti[i, j, k] = SigmaDustTot[i, j] * \
                                              np.log(mi_int1[k] / mi_int0[k]) / \
                                              np.log(mmax[i, j] / mmin[i, j])
                else:
                    SigmaDusti[i, j, k] = np.maximum(1.e-100, SigmaDustTot[i, j] - SigmaDusti[i, j].sum())
    if np.abs(SigmaDustTot.sum() - SigmaDusti.sum()) / SigmaDustTot.sum() > 1.e-10:
        print("Error in surface density interpolation!")

    # Transformation of the density distribution
    a = np.array(np.mean(mi[..., 1:] / mi[..., :-1], axis=-1))
    dm = np.array(2. * (a - 1.) / (a + 1.))
    sigmaDusti = SigmaDusti[...] / dm[..., None, None]

    # Calculation of Stokes Number over mass grid
    ai = np.zeros((int(Nt), int(Nr_len), int(Nmi)))
    for i in range(int(Nt)):
        ai[i] = np.full((int(Nr_len), int(Nmi)), 3 / (4 * np.pi * rho[i, 0, 0]) * mi[i, :] ** (1 / 3))
    Sti = np.zeros((int(Nt), int(Nr_len), int(Nmi)))
    for i in range(int(Nt)):
        Sti[i] = dp_dust_f.st_epstein_stokes1(ai[i], mfp[i], rho[i], SigmaGas[i])

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
    ret["vK"] = vK
    ret["vFrag"] = vFrag
    ret["smin"] = smin
    ret["smax"] = smax
    ret["xi"] = xi

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
