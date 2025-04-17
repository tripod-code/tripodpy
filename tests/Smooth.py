# %%
import importlib.util
import sys
import numpy as np
from scipy.interpolate import interp1d
import dustpy
import tripod
from dustpy import constants as c
from dustpy.utils import Boundary

amu = 1.66053886e-24

au = dustpy.constants.au
year = dustpy.constants.year



if __name__ == '__main__':
    ##########################################################################
    # Simulation with Mstar = 1 Msun
    ##########################################################################

    sim = tripod.Simulation()

    # Grid Configuration
    ri = np.geomspace(2.0, 250, 400) * c.au
    sim.grid.ri = ri
    #sim.ini.grid.Nmbpd = 7
    sim.ini.grid.mmin = (4./3.*np.pi*1.67*1e-5**3)

    # Gas Parameters
    sim.ini.gas.alpha = 0.001
    sim.ini.gas.SigmaRc = 60.0 * au
    sim.ini.gas.SigmaExp = -0.85
    sim.ini.gas.mu = 2.33*amu

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

    sim.t.snapshots = np.insert(np.logspace(0., 603.9912865, 30, base=1.025) * year, 0, 0)
    #sim.t.snapshots = sim.t.snapshots[sim.t.snapshots < 1e5*year]

    sim.dust.s.lim = 1e-4 
    sim.dust.f.drift = 0.8 
    sim.dust.f.dv = 0.4

    # Turn off gas evolution
    sim.gas.nu[:] = 0.0
    sim.gas.nu.updater = None



    sim.gas.rho.update()
    sim.gas.P.update()

    sim.dust.s.boundary.inner = Boundary(
                sim.grid.r,
                sim.grid.ri,
                sim.dust.Sigma[...,1]*sim.dust.s.max,
                condition="const_pow")

    sim.dust.s.boundary.outer = Boundary(
                sim.grid.r,
                sim.grid.ri,
                sim.dust.Sigma[...,1]*sim.dust.s.max,
                condition="const_pow")


    sim.update()

    # Write data
    sim.writer.datadir = "data_smooth"
    sim.writer.overwrite = True

    sim.run()
