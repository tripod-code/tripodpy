import numpy as np
from scipy.interpolate import interp1d
import dustpy
from dustpy import constants as c

au = dustpy.constants.au
year = dustpy.constants.year
amu = 1.66053886e-24

def refinegrid(ri, r0, num=3):
    """Function to refine the radial grid

    Parameters
    ----------
    ri : array
        Radial grid
    r0 : float
        Radial location around which grid should be refined
    num : int, option, default : 3
        Number of refinement iterations

    Returns
    -------
    ri : array
        New refined radial grid"""
    if num == 0:
        return ri
    ind = np.argmin(r0 > ri) - 1
    indl = ind-num
    indr = ind+num+1
    ril = ri[:indl]
    rir = ri[indr:]
    N = (2*num+1)*2
    rim = np.empty(N)
    for i in range(0, N, 2):
        j = ind-num+int(i/2)
        rim[i] = ri[j]
        rim[i+1] = 0.5*(ri[j]+ri[j+1])
    ri = np.concatenate((ril, rim, rir))
    return refinegrid(ri, r0, num=num-1)


def delta_arr(q=1e-3, alpha=0.01, Mach=20.):
    ''' My version for numpy arrays '''
    qNL = 1.04/Mach**3
    qW = 34*qNL*(alpha*Mach)**0.5
    delt = np.where(q > qNL, (qNL/q)**0.5, 1.0)
    delt += (q/qW)**3.
    return delt


def q_tilde(q=1e-3, alpha=0.01, Mach=20., r=1.0):
    D = 7.*Mach**1.5/alpha**0.25
    qt = q/(1 + D**3*(r**(1./6.) - 1.)**6)**(1./3.)
    return qt


def S_gap_arr(q=1e-3, alpha=0.01, Mach=20.):
    ''' My version for numpy arrays '''
    d = delta_arr(q, alpha, Mach)
    S = 1./(1. + (0.45/3./3.14159)*q**2*Mach**5/alpha*d)
    return S


def one_planet_arr(r=1.0, rp=1.0, q=1e-3, alpha=0.01, Mach=20.):
    ''' My version for numpy arrays '''
    x = r/rp
    qt = q_tilde(q, alpha, Mach, x)
    Sigma = S_gap_arr(qt, alpha, Mach)
    return Sigma


def get_sigma_arr(r, rs, qs, alphas, Machs, Sigma_bkgrd):
    ''' My version for numpy arrays '''
    Sigma = Sigma_bkgrd
    Sigma *= one_planet_arr(r, rs, qs, alphas, Machs)
    return Sigma
