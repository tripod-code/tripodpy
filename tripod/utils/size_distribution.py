import numpy as np


def get_rhos_simple(a, rhos, smin, smax):
    """
    Function to compute the bulk density of the reconstructed particle sizes.
    This simple model assumes the bulk density to be constant within the
    two original particle size bins.

    Parameters
    ----------
    a : array-like
        Particle size for which bulk densities should be computed
    rhos : array-like
        Bulk densities in the two population model
    smin : array-like
        Minimum particle sizes
    smax : array-like
        Maximum particle sizes

    Returns
    -------
    rhos_recon : array-like
        Reconstructed particle bulk densities
    """
    sint = np.sqrt(smin*smax)
    rhos_recon = np.ones_like(a[:, None, :]) * rhos[..., 1, None]
    rhos_recon = np.where(
        a[:, None, :] < sint[..., None],
        rhos[..., 0, None],
        rhos_recon
    )
    return rhos_recon


def get_q(Sigma, smin, smax):
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
    sint = np.sqrt(smin*smax)
    return -(np.log(Sigma[..., 1]/Sigma[..., 0]) / np.log(smax/sint) - 4.)


def get_size_distribution(sigma_d, a_max, q=3.5, na=10, agrid_min=None, agrid_max=None):
    """
    Makes a power-law size distribution up to a_max, normalized to the given surface density
    where the power-law can be a single float or different in each radial bin.

    Arguments:
    ----------

    sigma_d : array
        dust surface density array (shape (nr) where nr is the number of radial bins)

    a_max : array
        maximum particle size array (shape (nr) where nr is the number of radial bins)

    Keywords:
    ---------

    q : float | array
        particle size index, n(a) propto a**-q
        if array, it has to have the same length as sigma_d

    na : int
        number of particle size bins

    agrid_min : float
        minimum particle size

    agrid_max : float
        maximum particle size of the grid

    Returns:
    --------

    a : array
        particle size grid (centers)

    a_i : array
        particle size grid (interfaces)

    sig_da : array
        particle size distribution of size (len(sigma_d), na),
        units of g/cm^2, so integrated over the bins.
    """

    if agrid_min is None:
        agrid_min = a_max.min()

    if agrid_max is None:
        agrid_max = 2 * a_max.max()

    nr = len(sigma_d)
    sig_da = np.zeros([nr, na]) + 1e-100

    a_i = np.logspace(np.log10(agrid_min), np.log10(agrid_max), na + 1)
    a = 0.5 * (a_i[1:] + a_i[:-1])

    # we want to turn q into an array if it isn't one already
    q = q * np.ones(nr)

    for ir in range(nr):

        if a_max[ir] <= agrid_min:
            sig_da[ir, 0] = 1
            i_up = 0
        else:
            i_up = np.where(a_i < a_max[ir])[0][-1]

            # filling all bins that are strictly below a_max

            if q[ir] == 4.0:
                for ia in range(i_up):
                    sig_da[ir, ia] = np.log(a_i[ia + 1] / a_i[ia])

                # filling the bin that contains a_max
                sig_da[ir, i_up] = np.log(a_max[ir] / a_i[i_up])
            else:
                for ia in range(i_up):
                    sig_da[ir, ia] = \
                        a_i[ia + 1]**(4 - q[ir]) - a_i[ia]**(4 - q[ir])

                # filling the bin that contains a_max
                sig_da[ir, i_up] = a_max[ir]**(4 - q[ir]) - \
                    a_i[i_up]**(4 - q[ir])

        # normalize
        sig_da[ir, :i_up+1] = sig_da[ir, :i_up+1] / \
            sig_da[ir, :i_up+1].sum() * sigma_d[ir]

    return a, a_i, sig_da
