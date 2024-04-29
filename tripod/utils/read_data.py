import numpy as np
from simframe.io.writers import hdf5writer
from tripod import Simulation
from tripod.utils.size_distribution import get_q
from tripod.utils.size_distribution import get_rhos_simple
from tripod.utils.size_distribution import get_size_distribution
from types import SimpleNamespace


def read_data(data, filename="data", extension="hdf5", Na=50):
    """
    Function returns a SimpleNamespace with the most useful
    data that can be used for plotting or other purposes.
    This avoids reading the entirety of data files.
    It also return data fields with the appendix ``_recon``
    that contains values related to the reconstructed size
    distributions.

    Parameters
    ----------
    data : str | tripod.Simulation
        Either a path to the data directory or a TriPoD
        simulation frame
    filename : str, optional, default: "data"
        Stem of the data files in the data directory
    extension : str, optional, default: "hdf5"
        File extension of the data file in the data directory
    Na : int, optional, default: 50
        Number of size bins used for the reconstructed
        size distribution

    Returns
    -------
    data : SimpleNamespace
        SimpleNamespace with the extracted and reconstructed
        data fields
    """

    if isinstance(data, Simulation):

        # Extract from simulation object

        # Simulation
        t = data.t.reshape((1))

        # Dust
        deltaTurb = data.dust.delta.turb.reshape((1, -1))
        eps = data.dust.eps.reshape((1, -1))
        fill = data.dust.fill.reshape((1, -1))
        rhos = data.dust.rhos.reshape((1, -1))
        smax = data.dust.s.max.reshape((1, -1))
        smin = data.dust.s.min.reshape((1, -1))
        SigmaDust = data.dust.Sigma.reshape((1, -1))
        vFrag = data.dust.v.frag.reshape((1, -1))

        # Gas
        cs = data.gas.cs.reshape((1, -1))
        mfp = data.gas.mfp.reshape((1, -1))
        SigmaGas = data.gas.Sigma.reshape((1, -1))

        # Grid
        OmegaK = data.grid.OmegaK.reshape((1, -1))
        r = data.grid.r.reshape((1, -1))
        ri = data.grid.ri.reshape((1, -1))

    else:

        # Read from data directory
        wrtr = hdf5writer(datadir=data, filename=filename, extension=extension)

        # Simulation
        t = wrtr.read.sequence("t")

        # Dust
        deltaTurb = wrtr.read.sequence("dust.delta.turb")
        eps = wrtr.read.sequence("dust.eps")
        fill = wrtr.read.sequence("dust.fill")
        rhos = wrtr.read.sequence("dust.rhos")
        smax = wrtr.read.sequence("dust.s.max")
        smin = wrtr.read.sequence("dust.s.min")
        SigmaDust = wrtr.read.sequence("dust.Sigma")
        vFrag = wrtr.read.sequence("dust.v.frag")

        # Gas
        cs = wrtr.read.sequence("gas.cs")
        mfp = wrtr.read.sequence("gas.mfp")
        SigmaGas = wrtr.read.sequence("gas.Sigma")

        # Grid
        OmegaK = wrtr.read.sequence("grid.OmegaK")
        r = wrtr.read.sequence("grid.r")
        ri = wrtr.read.sequence("grid.ri")

    # Calculations

    # Helper variables
    Nt, Nr = r.shape

    # Masses
    Mdust = np.pi * ((ri[:, 1:]**2-ri[:, :-1]**2) * SigmaDust.sum(-1)).sum(-1)
    Mgas = np.pi * ((ri[:, 1:]**2-ri[:, :-1]**2) * SigmaGas).sum(-1)

    # Compute the reconstructed quantities
    # Size distribution exponent
    q = get_q(SigmaDust, smin, smax)
    # Reconstructed sizes and size distribution
    SigmaDust_recon = np.empty((Nt, Nr, Na))
    a_recon = np.empty((Nt, Na))
    for i in range(Nt):
        a, _, sig_da = get_size_distribution(
            SigmaDust[i, ...].sum(-1), smax[i, :],
            q=q[i, ...], na=Na,
            agrid_min=smin.min(),
            agrid_max=2.*smax.max(),
        )
        a_recon[i, ...] = a
        SigmaDust_recon[i, ...] = sig_da
        imin = sig_da.argmin()
        idx = np.unravel_index(imin, sig_da.shape)
    # Reconstructed bulk density
    rhos_recon = get_rhos_simple(a_recon, rhos[..., [0, 2]], smin, smax)
    fill_recon = get_rhos_simple(a_recon, fill[..., [0, 2]], smin, smax)
    # Reconstructed sigma
    A = np.mean(a[1:]/a[:-1])
    B = 2 * (A-1) / (A+1)
    sigmaDust_recon = SigmaDust_recon / B

    # import sys
    # sys.exit()

    ret = {}
    # Simulation
    ret["t"] = t
    # Dust
    dust = {}
    dust["eps"] = eps
    dust["delta"] = SimpleNamespace(**{"turb": deltaTurb})
    dust["fill"] = fill
    dust["fill_recon"] = fill_recon
    dust["M"] = Mdust
    dust["rhos"] = rhos
    dust["rhos_recon"] = rhos_recon
    dust["s"] = SimpleNamespace(**{"max": smax, "min": smin})
    dust["Sigma"] = SigmaDust
    dust["Sigma_recon"] = SigmaDust_recon
    dust["sigma_recon"] = sigmaDust_recon
    dust["v"] = SimpleNamespace(**{"frag": vFrag})
    ret["dust"] = SimpleNamespace(**dust)
    # Gas
    gas = {}
    gas["cs"] = cs
    gas["M"] = Mgas
    gas["mfp"] = mfp
    gas["Sigma"] = SigmaGas
    ret["gas"] = SimpleNamespace(**gas)
    # Grid
    grid = {}
    grid["a_recon"] = a_recon
    grid["r"] = r
    grid["ri"] = ri
    grid["OmegaK"] = OmegaK
    ret["grid"] = SimpleNamespace(**grid)

    return SimpleNamespace(**ret)
