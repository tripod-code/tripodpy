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


if __name__ == '__main__':
    ##########################################################################
    # Simulation with Mstar = 1 Msun
    ##########################################################################

    sim = dustpy.Simulation()

    # Grid Configuration
    ri = np.geomspace(2.0, 250, 400) * c.au
    #ri = refinegrid(ri, 5.2*c.au, num=2)
    #ri = refinegrid(ri, 8*c.au, num=4)
    sim.grid.ri = ri
    sim.ini.grid.Nmbpd = 7
    sim.ini.grid.mmin = (4./3.*np.pi*1.67*1e-5**3)

    # Gas Parameters
    sim.ini.gas.alpha = 0.001
    sim.ini.gas.SigmaRc = 60.0 * au
    sim.ini.gas.SigmaExp = -0.85

    # Dust Parameters
    sim.ini.dust.vFrag = 1000.0
    sim.ini.dust.d2gRatio = 0.01
    sim.ini.dust.allowDriftingParticles = False

    # Stellar Parameters (from Baraffe first snapshot)
    sim.ini.star.M = 1.0 * dustpy.constants.M_sun
    sim.ini.star.R = 3.096 * dustpy.constants.R_sun
    sim.ini.star.T = 4397.0

    sim.ini.gas.Mdisk = 0.05*sim.ini.star.M

    sim.initialize()

    sim.t.snapshots = np.logspace(0., 603.9912865, 300, base=1.025) * year

    sim.dust.kernel.save = False
    sim.dust.v.rel.azi.save = False
    sim.dust.v.rel.rad.save = False
    sim.dust.v.rel.brown.save = False
    sim.dust.v.rel.turb.save = False
    sim.dust.v.rel.vert.save = False
    sim.dust.Fi.adv.save = False
    sim.dust.Fi.diff.save = False
    sim.dust.coagulation.A.save = False
    sim.dust.coagulation.eps.save = False
    sim.dust.coagulation.phi.save = False
    sim.dust.coagulation.lf_ind.save = False
    sim.dust.coagulation.rm_ind.save = False
    sim.dust.coagulation.stick.save = False
    sim.dust.coagulation.stick_ind.save = False
    sim.dust.p.stick.save = False
    sim.dust.p.frag.save = False
    sim.dust.v.rel.tot.save = False

    # Turn off gas evolution
    sim.gas.nu[:] = 0.0
    sim.gas.nu.updater = None

    # Add Duffell Gap
    HR = interp1d(sim.grid.r, sim.gas.Hp/sim.grid.r)
    Mach = 1/HR(5.2*au)

    Sigma_bkgrd = sim.gas.Sigma.copy()
    sim.gas.Sigma = get_sigma_arr(
        sim.grid.r, 5.2*au, 1e-3, sim.ini.gas.alpha, Mach, sim.gas.Sigma)
    sim.dust.Sigma *= (sim.gas.Sigma/Sigma_bkgrd)[:, None]
    sim.gas.rho.update()
    sim.gas.P.update()

    # Write data
    sim.writer.datadir = "data_new"
    sim.writer.overwrite = True

    sim.run()
