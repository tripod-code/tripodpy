import dustpy as dp
import dustpy.constants as c
import numpy as np
from simframe import Instruction
from simframe import Integrator
from simframe.frame import Field


class Simulation(dp.Simulation):

    # Exclude the following functions from the from DustPy inherited object
    _excludefromparent = [
        "checkmassconservation",
        "setdustintegrator"
    ]

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        # Deleting not needed entries from ini object
        del(self.ini.dust.crateringMassRatio)
        del(self.ini.dust.excavatedMass)
        del(self.ini.dust.fragmentDistribution)
        del(self.ini.grid.Nmbpd)
        del(self.ini.grid.mmin)
        del(self.ini.grid.mmax)

        # Deleting Fields that are not needed
        del(self.dust.coagulation)
        del(self.dust.kernel)
        del(self.dust.S.coag)
        del(self.grid.m)
        del(self.grid.Nm)

    # Note: the next two functions are to hide methods from DustPy that are not used in TwoPopPy
    # I have to check if there is a cleaner way of doing this.
    def __dir__(self):
        '''This function hides all attributes in _excludefromparten from inherited DustPy object. It is only hiding them. They can still be accessed.'''
        return sorted((set(dir(self.__class__)) | set(self.__dict__.keys())) - set(self._excludefromparent))

    def __getattribute__(self, __name):
        '''This function raises an attribute error if the hidden attributes are accessed.'''
        if __name in super(dp.Simulation, self).__getattribute__("_excludefromparent"):
            raise AttributeError(__name)
        return super(dp.Simulation, self).__getattribute__(__name)

    def __setattr__(self, name, value):
        '''This function removes attribute from list of hidden attributes if user manually set is.'''
        if name in self._excludefromparent:
            self._excludefromparent.remove(name)
        return super().__setattr__(name, value)
    
    def run(self):
        """This functions runs the simulation."""
        # Print welcome message
        if self.verbosity > 0:
            msg = ""
            msg += "\nTwoPopPy v{}".format(self.__version__)
            msg += "\n"
            print(msg)
        # Actually run the simulation
        super(dp.Simulation, self).run()

    def makegrids(self):
        '''Function creates radial grid.

        Notes
        -----
        The grids are set up with the parameters given in ``Simulation.ini``.
        If you want to have a custom radial grid you have to set the array of grid cell interfaces ``Simulation.grid.ri``,
        before calling ``Simulation.makegrids()``.'''

        # Number of mass species. Hard coded
        Nm = 2

        # The mass grid does not exist. But we store the size of the
        # particle dimension in a hidden variable.
        self.grid.addfield(
            "_Nm", Nm, description="# of particle species", constant=True)

        self._makeradialgrid()

    def _makeradialgrid(self):
        '''Function sets the mass grid using the parameters set in ``Simulation.ini``.'''
        if self.grid.ri is None:
            ri = np.logspace(np.log10(self.ini.grid.rmin), np.log10(
                self.ini.grid.rmax), num=self.ini.grid.Nr+1, base=10.)
            Nr = self.ini.grid.Nr
        else:
            ri = self.grid.ri
            Nr = ri.shape[0] - 1
        r = 0.5*(ri[:-1] + ri[1:])
        A = np.pi*(ri[1:]**2 - ri[:-1]**2)
        self.grid.addfield(
            "Nr", Nr, description="# of radial grid cells", constant=True)
        self.grid.addfield(
            "r", r, description="Radial grid cell centers [cm]", constant=True)
        self.grid.addfield(
            "ri", ri, description="Radial grid cell interfaces [cm]", constant=True)
        self.grid.addfield(
            "A", A, description="Radial grid annulus area [cm²]", constant=True)

    def initialize(self):
        '''Function initializes the simulation frame.

        Function sets all fields that are None with a standard value.
        If the grids are not set, it will call ``Simulation.makegrids()`` first.'''

        if not isinstance(self.grid.Nr, Field):
            self.makegrids()

        # Set integration variable
        if self.t is None:
            self.addintegrationvariable("t", 0., description="Time [s]")
            self.t.cfl = 0.1

            # Todo: Placeholder! This needs to be replaced with a TwoPopPy specific time step function
            self.t.updater = dp.std.sim.dt

            self.t.snapshots = np.logspace(3., 5., num=21, base=10.) * c.year

        # Initialize groups
        self._initializestar()
        self._initializegrid()
        self._initializegas()
        self._initializedust()

        # Set integrator
        if self.integrator is None:
            # Todo: Add instructions for dust quantities
            instructions = [
                # Instruction(std.dust.impl_1_direct,
                #            self.dust.Sigma,
                #            controller={"rhs": self.dust._rhs
                #                        },
                #            description="Dust: implicit 1st-order direct solver"
                #            ),
                Instruction(dp.std.gas.impl_1_direct,
                            self.gas.Sigma,
                            controller={"rhs": self.gas._rhs
                                        },
                            description="Gas: implicit 1st-order direct solver"
                            ),
            ]
            self.integrator = Integrator(
                self.t, description="Default integrator")
            self.integrator.instructions = instructions
            # Todo: Add preparator and finalizer?
            # self.integrator.preparator = dp.std.sim.prepare_implicit_dust
            # self.integrator.finalizer = dp.std.sim.finalize_implicit_dust

        # Set writer
        if self.writer is None:
            self.writer = dp.utils.hdf5writer()

    def _initializedust(self):
        '''Function to initialize dust quantities'''

        # Shapes needed to initialize arrays
        shape1 = (int(self.grid.Nr))
        shape2 = (int(self.grid.Nr), int(self.grid._Nm))
        shape2ravel = (int(self.grid.Nr*self.grid._Nm))
        shape2p1 = (int(self.grid.Nr)+1, int(self.grid._Nm))
        shape3 = (int(self.grid.Nr), int(
            self.grid._Nm), int(self.grid._Nm))

        # Particle size
        if self.dust.a is None:
            self.dust.addfield(
                "a", np.ones(shape2), description="Particle size [cm]")
            # Todo: Placeholder! This needs to be replaced with a TwoPopPy specific function
            self.dust.a.updater = dp.std.dust.a
        # Diffusivity
        if self.dust.D is None:
            self.dust.addfield(
                "D", np.zeros(shape2), description="Diffusivity [cm²/s]")
            self.dust.D.updater = dp.std.dust.D
        # Deltas
        if self.dust.delta.rad is None:
            delta = self.ini.gas.alpha * np.ones(shape1)
            self.dust.delta.addfield(
                "rad", delta, description="Radial mixing parameter")
        if self.dust.delta.turb is None:
            delta = self.ini.gas.alpha * np.ones(shape1)
            self.dust.delta.addfield(
                "turb", delta, description="Turbulent mixing parameter")
        if self.dust.delta.vert is None:
            delta = self.ini.gas.alpha * np.ones(shape1)
            self.dust.delta.addfield(
                "vert", delta, description="Vertical mixing parameter")
        # Vertically integrated dust to gas ratio
        if self.dust.eps is None:
            self.dust.addfield(
                "eps", np.zeros(shape1), description="Dust-to-gas ratio")
            self.dust.eps.updater = dp.std.dust.eps
        # Fluxes
        if self.dust.Fi.adv is None:
            self.dust.Fi.addfield(
                "adv", np.zeros(shape2p1), description="Advective flux [g/cm/s]")
            self.dust.Fi.adv.updater = dp.std.dust.F_adv
        if self.dust.Fi.diff is None:
            self.dust.Fi.addfield(
                "diff", np.zeros(shape2p1), description="Diffusive flux [g/cm/s]")
            self.dust.Fi.diff.updater = dp.std.dust.F_diff
        if self.dust.Fi.tot is None:
            self.dust.Fi.addfield(
                "tot", np.zeros(shape2p1), description="Total flux [g/cm/s]")
            self.dust.Fi.tot.updater = dp.std.dust.F_tot
        # Filling factor
        if self.dust.fill is None:
            self.dust.addfield(
                "fill", np.ones(shape2), description="Filling factor")
        # Scale height
        if self.dust.H is None:
            self.dust.addfield(
                "H", np.zeros(shape2), description="Scale heights [cm]")
            self.dust.H.updater = dp.std.dust.H
        # Midplane mass density
        if self.dust.rho is None:
            self.dust.addfield(
                "rho", np.zeros(shape2), description="Midplane mass density per mass bin [g/cm³]")
            self.dust.rho.updater = dp.std.dust.rho_midplane
        # Solid state density
        if self.dust.rhos is None:
            rhos = self.ini.dust.rhoMonomer * np.ones(shape2)
            self.dust.addfield(
                "rhos", rhos, description="Solid state density [g/cm³]")
        # Probabilities
        if self.dust.p.frag is None:
            self.dust.p.frag = Field(self, np.zeros(
                shape3), description="Fragmentation probability")
            # Todo: Placeholder! This needs to be replaced with a TwoPopPy specific function
            self.dust.p.frag.updater = dp.std.dust.p_frag
        if self.dust.p.stick is None:
            self.dust.p.stick = Field(self, np.zeros(
                shape3), description="Sticking probability")
            # Todo: Placeholder! This needs to be replaced with a TwoPopPy specific function
            self.dust.p.stick.updater = dp.std.dust.p_stick
        # Source terms
        if self.dust.S.ext is None:
            self.dust.S.addfield(
                "ext", np.zeros(shape2), description="External sources [g/cm²/s]")
        if self.dust.S.hyd is None:
            self.dust.S.addfield(
                "hyd", np.zeros(shape2), description="Hydrodynamic sources [g/cm²/s]")
            self.dust.S.hyd.updater = dp.std.dust.S_hyd
        if self.dust.S.tot is None:
            self.dust.S.addfield(
                "tot", np.zeros(shape2), description="Tot sources [g/cm²/s]")
            # Todo: Placeholder! This needs to be replaced with a TwoPopPy specific function
            self.dust.S.tot.updater = dp.std.dust.S_tot
        # Stokes number
        if self.dust.St is None:
            self.dust.addfield(
                "St", np.zeros(shape2), description="Stokes number")
            self.dust.St.updater = dp.std.dust.St_Epstein_StokesI
        # Velocities
        if self.dust.v.frag is None:
            vfrag = self.ini.dust.vfrag * np.ones(shape1)
            self.dust.v.addfield(
                "frag", vfrag, description="Fragmentation velocity [cm/s]")
        if self.dust.v.rel.azi is None:
            self.dust.v.rel.addfield(
                "azi", np.zeros(shape3), description="Relative azimuthal velocity [cm/s]")
            self.dust.v.rel.azi.updater = dp.std.dust.vrel_azimuthal_drift
        if self.dust.v.rel.brown is None:
            self.dust.v.rel.addfield(
                "brown", np.zeros(shape3), description="Relative Brownian motion velocity [cm/s]")
            # Todo: Placeholder! This needs to be replaced with a TwoPopPy specific function
            self.dust.v.rel.brown.updater = dp.std.dust.vrel_brownian_motion
        if self.dust.v.rel.rad is None:
            self.dust.v.rel.addfield(
                "rad", np.zeros(shape3), description="Relative radial velocity [cm/s]")
            self.dust.v.rel.rad.updater = dp.std.dust.vrel_radial_drift
        if self.dust.v.rel.turb is None:
            self.dust.v.rel.addfield(
                "turb", np.zeros(shape3), description="Relative turbulent velocity [cm/s]")
            self.dust.v.rel.turb.updater = dp.std.dust.vrel_turbulent_motion
        if self.dust.v.rel.vert is None:
            self.dust.v.rel.addfield(
                "vert", np.zeros(shape3), description="Relative vertical settling velocity [cm/s]")
            self.dust.v.rel.vert.updater = dp.std.dust.vrel_vertical_settling
        if self.dust.v.rel.tot is None:
            self.dust.v.rel.addfield(
                "tot", np.zeros(shape3), description="Total relative velocity [cm/s]")
            self.dust.v.rel.tot.updater = dp.std.dust.vrel_tot
        if self.dust.v.driftmax is None:
            self.dust.v.addfield(
                "driftmax", np.zeros(shape1), description="Maximum drift velocity [cm/s]")
            self.dust.v.driftmax.updater = dp.std.dust.vdriftmax
        if self.dust.v.rad is None:
            self.dust.v.addfield(
                "rad", np.zeros(shape2), description="Radial velocity [cm/s]")
            self.dust.v.rad.updater = dp.std.dust.vrad
        # Initialize dust quantities partly to calculate Sigma
        
        try:
            self.dust.update()
        except:
            pass
        
        # Floor value
        if self.dust.SigmaFloor is None:
            # Todo: What is a reasonable value for this in TwoPopPy
            SigmaFloor = 1.e-100 * np.ones(shape2)
            self.dust.addfield(
                "SigmaFloor", SigmaFloor, description="Floor value of surface density [g/cm²]")
        # Surface density, if not set
        if self.dust.Sigma is None:
            # Todo: This needs to be replaced with TwoPopPy specific functions
            Sigma = dp.std.dust.MRN_distribution(self)
            Sigma = np.where(Sigma <= self.dust.SigmaFloor,
                             0.1*self.dust.SigmaFloor,
                             Sigma)
            self.dust.addfield(
                "Sigma", Sigma, description="Surface density per mass bin [g/cm²]")
        # Todo: Differentiator and Jacobinator need to be modified for TwoPopPy
        self.dust.Sigma.differentiator = dp.std.dust.Sigma_deriv
        self.dust.Sigma.jacobinator = dp.std.dust.jacobian
        
        # Fully initialize dust quantities
        self.dust.update()
        
        # Hidden fields
        # We store the old values of the surface density in a hidden field
        # to calculate the fluxes through the boundaries in case of implicit integration.
        self.dust._SigmaOld = Field(
            self, self.dust.Sigma, description="Previous value of surface density [g/cm²]")
        # The right-hand side of the matrix equation is stored in a hidden field
        self.dust._rhs = Field(self, np.zeros(
            shape2ravel), description="Right-hand side of matrix equation [g/cm²]")
        # Boundary conditions
        if self.dust.boundary.inner is None:
            self.dust.boundary.inner = dp.utils.Boundary(
                self.grid.r,
                self.grid.ri,
                self.dust.Sigma,
                condition="const_grad"
            )
        if self.dust.boundary.outer is None:
            self.dust.boundary.outer = dp.utils.Boundary(
                self.grid.r[::-1],
                self.grid.ri[::-1],
                self.dust.Sigma[::-1],
                condition="val",
                value=0.1*self.dust.SigmaFloor[-1]
            )

        self.dust.update()
