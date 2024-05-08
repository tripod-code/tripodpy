import dustpy as dp
import dustpy.constants as c
import numpy as np
from simframe import Instruction
from simframe import Integrator
from simframe import schemes
from simframe.frame import Field
from . import std


class Simulation(dp.Simulation):
    """The main simulation class for running dust evolution simulations.
    `twopoppy.Simulation`` is a child of ``dustpy.Simulation``,
    which is in turn a child of ``simframe.Frame``.
    For setting simple initial conditions use ``Simulation.ini``,
    For making the simulation grids use ``Simulation.makegrids()``,
    For initialization use ``Simulation.initialize()``,
    For running simulations use ``Simulation.run()``.
    Please have a look at the documentation of ``simframe`` for further details."""

    __name__ = "TwoPopPy"

    # Exclude the following functions from the from DustPy inherited object
    _excludefromdustpy = [
        "checkmassconservation",
        "setdustintegrator"
    ]

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        # Add new fields
        self.dust.m = None
        self.dust.addgroup("q", description="Distribution exponents")
        self.dust.q.drfrag = None
        self.dust.q.eff = None
        self.dust.q.frag = None
        self.dust.q.sweep = None
        self.dust.q.turb1 = None
        self.dust.q.turb2 = None
        self.dust.q.updater = ["frag", "eff"]  # TODO: update the updater
        self.dust.addgroup("s", description="Characteristic particle sizes")
        self.dust.s.min = None
        self.dust.s.max = None
        self.dust.s.lim = None
        self.dust.s._sdot_shrink = None
        self.dust.addgroup(
            "f", description="Fudge factors")
        self.dust.f.crit = None
        self.dust.f.drift = None
        self.dust.f.dvdrift = None
        self.dust.f.dvturb = None
        self.dust.f.dv = None
        self.dust.f.updater = ["dv"]
        self.dust.p.drift = None
        self.dust.p.updater = ["frag", "stick", "drift"]

        # Adjusting update orders

        # Adding new elements to update order in a relative way
        def addelemtafter(lst, elem, after):
            idx = lst.index(after)
            lst.insert(idx + 1, elem)

        # Adjusting the updater of main simulation frame
        updtordr = self.dust.updateorder
        # Add "f" after "p"
        addelemtafter(updtordr, "f", "p")
        # move "a" after "f"
        updtordr.remove("a")
        addelemtafter(updtordr, "a", "f")
        # Add "m" after "a"
        addelemtafter(updtordr, "m", "a")
        # Add "q" after "m"
        addelemtafter(updtordr, "q", "m")
        # Add "SigmaFloor" after "m"
        addelemtafter(updtordr, "SigmaFloor", "m")
        # Removing elements that are not used
        updtordr.remove("kernel")
        # Assign updateorder
        self.dust.updater = updtordr

        # Deleting Fields that are not needed
        del self.ini.grid.Nmbpd
        del self.ini.grid.mmax
        del self.ini.dust.erosionMassRatio
        del self.ini.dust.excavatedMass
        del self.ini.dust.fragmentDistribution
        del self.dust.coagulation
        del self.dust.kernel
        del self.grid.m
        del self.grid.Nm

        # TODO: Managing the self.ini object

    # Note: the next two functions are to hide methods from DustPy that are not used in TwoPopPy
    def __dir__(self):
        '''This function hides all attributes in _excludefromparten from inherited DustPy object.
        It is only hiding them. They can still be accessed.'''
        exclude = set(self._excludefromdustpy) - set(self.__dict__.keys())
        return sorted((set(dir(self.__class__)) | set(self.__dict__.keys())) - exclude)

    def __getattribute__(self, name):
        '''This function raises an attribute error for elements that should not be inherited from DustPy if they
        were not manually set in TwoPopPy.'''
        in_tp = name in super(
            dp.Simulation, self).__getattribute__("__dict__")
        in_dp = name in dp.Simulation.__dict__
        in_ex = name in name in super(
            dp.Simulation, self).__getattribute__("_excludefromdustpy")
        if not in_tp and in_dp and in_ex:
            raise AttributeError(name)
        return super(dp.Simulation, self).__getattribute__(name)

    def run(self):
        '''This functions runs the simulation.'''
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
        If you want to have a custom radial grid you have to set the array of grid cell interfaces
        ``Simulation.grid.ri``, before calling ``Simulation.makegrids()``.'''

        # Number of mass species. Hard coded.
        # The surface densities have Nm_short particle species.
        # Other quantities need Nm_long particle species.
        # We store them in hidden variables.
        Nm_short = 2
        Nm_long = 5
        self.grid.addfield(
            "_Nm_short",
            Nm_short,
            description="# of particle species for surface densities",
            constant=True
        )
        self.grid.addfield(
            "_Nm_long",
            Nm_long,
            description="# of particle species for calculations",
            constant=True
        )

        self._makeradialgrid()

    def _makeradialgrid(self):
        '''Function sets the mass grid using the parameters set in ``Simulation.ini``.'''
        if self.grid.ri is None:
            ri = np.logspace(np.log10(self.ini.grid.rmin), np.log10(
                self.ini.grid.rmax), num=self.ini.grid.Nr + 1, base=10.)
            Nr = self.ini.grid.Nr
        else:
            ri = self.grid.ri
            Nr = ri.shape[0] - 1
        r = 0.5 * (ri[:-1] + ri[1:])
        A = np.pi * (ri[1:] ** 2 - ri[:-1] ** 2)
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

            self.t.updater = std.sim.dt

            self.t.snapshots = np.logspace(3., 5., num=21, base=10.) * c.year

        # Initialize groups
        self._initializestar()
        self._initializegrid()
        self._initializegas()
        self._initializedust()

        # Set integrator
        if self.integrator is None:
            instructions = [
                Instruction(
                    std.dust.impl_1_direct,
                    self.dust._Y,
                    controller={"rhs": self.dust._Y_rhs},
                    description="Dust (state vector): implicit 1st-order direct solver"
                ),
                Instruction(
                    dp.std.gas.impl_1_direct,
                    self.gas.Sigma,
                    controller={"rhs": self.gas._rhs},
                    description="Gas: implicit 1st-order direct solver"
                ),
            ]
            self.integrator = Integrator(
                self.t, description="Default integrator")
            self.integrator.instructions = instructions
            self.integrator.preparator = std.sim.prepare_implicit_dust
            self.integrator.finalizer = std.sim.finalize_implicit_dust

        # Set writer
        if self.writer is None:
            self.writer = dp.utils.hdf5writer()

    def _initializedust(self):
        '''Function to initialize dust quantities'''

        # Shapes needed to initialize arrays
        # Only radial grid
        shape1 = (int(self.grid.Nr))
        # Radial grid and long particle grid
        shape2 = (int(self.grid.Nr), int(self.grid._Nm_long))
        # Radial grid and short particle grid for surface densities
        shape2Sigma = (int(self.grid.Nr), int(self.grid._Nm_short))
        # Length of vector of implicit integration
        shape2Sigmaravel = (int(self.grid.Nr * self.grid._Nm_short))
        # Radial grid interfaces and short particle grid
        shape2p1Sigma = (int(self.grid.Nr) + 1, int(self.grid._Nm_short))
        # Radial grid and twice long mass grid for relative velocities etc.
        shape3 = (int(self.grid.Nr), int(
            self.grid._Nm_long), int(self.grid._Nm_long))

        # Particle size
        if self.dust.a is None:
            self.dust.addfield(
                "a", np.ones(shape2), description="Particle size [cm]"
            )
            self.dust.a.updater = std.dust.a
        # Particle mass
        if self.dust.m is None:
            self.dust.addfield(
                "m", np.ones(shape2), description="Particle mass [g]"
            )
            self.dust.m.updater = std.dust.m
        # Diffusivity
        if self.dust.D is None:
            self.dust.addfield(
                "D", np.zeros(shape2), description="Diffusivity [cm²/s]"
            )
            self.dust.D.updater = dp.std.dust.D
        # Deltas
        if self.dust.delta.rad is None:
            delta = self.ini.gas.alpha * np.ones(shape1)
            self.dust.delta.addfield(
                "rad", delta, description="Radial mixing parameter"
            )
        if self.dust.delta.turb is None:
            delta = self.ini.gas.alpha * np.ones(shape1)
            self.dust.delta.addfield(
                "turb", delta, description="Turbulent mixing parameter"
            )
        if self.dust.delta.vert is None:
            delta = self.ini.gas.alpha * np.ones(shape1)
            self.dust.delta.addfield(
                "vert", delta, description="Vertical mixing parameter"
            )
        # Vertically integrated dust to gas ratio
        if self.dust.eps is None:
            self.dust.addfield(
                "eps", np.zeros(shape1), description="Dust-to-gas ratio"
            )
            self.dust.eps.updater = dp.std.dust.eps
        # Fluxes
        if self.dust.Fi.adv is None:
            self.dust.Fi.addfield(
                "adv", np.zeros(shape2p1Sigma), description="Advective flux [g/cm/s]"
            )
            self.dust.Fi.adv.updater = std.dust.F_adv
        if self.dust.Fi.diff is None:
            self.dust.Fi.addfield(
                "diff", np.zeros(shape2p1Sigma), description="Diffusive flux [g/cm/s]"
            )
            self.dust.Fi.diff.updater = std.dust.F_diff
        if self.dust.Fi.tot is None:
            self.dust.Fi.addfield(
                "tot", np.zeros(shape2p1Sigma), description="Total flux [g/cm/s]"
            )
            # TODO: Use DustPy updater after DustPy update
            self.dust.Fi.tot.updater = std.dust.F_tot
        # Filling factor
        if self.dust.fill is None:
            self.dust.addfield(
                "fill", np.ones(shape2), description="Filling factor"
            )
        # Scale height
        if self.dust.H is None:
            self.dust.addfield(
                "H", np.zeros(shape2), description="Scale heights [cm]"
            )
            self.dust.H.updater = std.dust.H
        # Midplane mass density
        if self.dust.rho is None:
            self.dust.addfield(
                "rho", np.zeros(shape2), description="Midplane mass density per mass bin [g/cm³]"
            )
            self.dust.rho.updater = std.dust.rho_midplane
        # Solid state density
        if self.dust.rhos is None:
            rhos = self.ini.dust.rhoMonomer * np.ones(shape2)
            self.dust.addfield(
                "rhos", rhos, description="Solid state density [g/cm³]"
            )
        # Probabilities
        if self.dust.p.frag is None:
            self.dust.p.frag = Field(self, np.zeros(
                shape1), description="Fragmentation probability")
            self.dust.p.frag.updater = std.dust.p_frag
        if self.dust.p.stick is None:
            self.dust.p.stick = Field(self, np.zeros(
                shape1), description="Sticking probability")
            self.dust.p.stick.updater = std.dust.p_stick
        if self.dust.p.drift is None:
            self.dust.p.drift = Field(self, np.zeros(
                shape1), description="Transition function from drift to turbulence")
            self.dust.p.drift.updater = std.dust.p_drift

        # Source terms
        if self.dust.S.ext is None:
            self.dust.S.addfield(
                "ext", np.zeros(shape2Sigma), description="External sources [g/cm²/s]"
            )
        if self.dust.S.hyd is None:
            self.dust.S.addfield(
                "hyd", np.zeros(shape2Sigma), description="Hydrodynamic sources [g/cm²/s]"
            )
            self.dust.S.hyd.updater = dp.std.dust.S_hyd
        if self.dust.S.coag is None:
            self.dust.S.addfield(
                "coag", np.zeros(shape2Sigma), description="Coagulation sources [g/cm²/s]"
            )
            self.dust.S.coag.updater = std.dust.S_coag
        if self.dust.S.tot is None:
            self.dust.S.addfield(
                "tot", np.zeros(shape2Sigma), description="Total sources [g/cm²/s]"
            )
            self.dust.S.tot.updater = std.dust.S_tot
        # Stokes number
        if self.dust.St is None:
            self.dust.addfield(
                "St", np.zeros(shape2), description="Stokes number"
            )
            self.dust.St.updater = dp.std.dust.St_Epstein_StokesI
        # Velocities
        if self.dust.v.frag is None:
            vfrag = self.ini.dust.vfrag * np.ones(shape1)
            self.dust.v.addfield(
                "frag", vfrag, description="Fragmentation velocity [cm/s]"
            )
        if self.dust.v.rel.azi is None:
            self.dust.v.rel.addfield(
                "azi", np.zeros(shape3), description="Relative azimuthal velocity [cm/s]"
            )
            self.dust.v.rel.azi.updater = dp.std.dust.vrel_azimuthal_drift
        if self.dust.v.rel.brown is None:
            self.dust.v.rel.addfield(
                "brown", np.zeros(shape3), description="Relative Brownian motion velocity [cm/s]"
            )
            self.dust.v.rel.brown.updater = std.dust.vrel_brownian_motion
        if self.dust.v.rel.rad is None:
            self.dust.v.rel.addfield(
                "rad", np.zeros(shape3), description="Relative radial velocity [cm/s]"
            )
            self.dust.v.rel.rad.updater = dp.std.dust.vrel_radial_drift
        if self.dust.v.rel.turb is None:
            self.dust.v.rel.addfield(
                "turb", np.zeros(shape3), description="Relative turbulent velocity [cm/s]"
            )
            self.dust.v.rel.turb.updater = dp.std.dust.vrel_turbulent_motion
        if self.dust.v.rel.vert is None:
            self.dust.v.rel.addfield(
                "vert", np.zeros(shape3), description="Relative vertical settling velocity [cm/s]"
            )
            self.dust.v.rel.vert.updater = dp.std.dust.vrel_vertical_settling
        if self.dust.v.rel.tot is None:
            self.dust.v.rel.addfield(
                "tot", np.zeros(shape3), description="Total relative velocity [cm/s]"
            )
            self.dust.v.rel.tot.updater = dp.std.dust.vrel_tot
        if self.dust.v.driftmax is None:
            self.dust.v.addfield(
                "driftmax", np.zeros(shape1), description="Maximum drift velocity [cm/s]"
            )
            self.dust.v.driftmax.updater = dp.std.dust.vdriftmax
        if self.dust.v.rad is None:
            self.dust.v.addfield(
                "rad", np.zeros(shape2), description="Radial velocity [cm/s]"
            )
            self.dust.v.rad.updater = dp.std.dust.vrad
        # Distribution exponents
        if self.dust.q.eff is None:
            q = np.ones(shape1)  # will be computed in the updater
            self.dust.q.addfield(
                "eff", q, description="Calculated distribution exponent"
            )
            self.dust.q.eff.updater = std.dust.q_eff
        if self.dust.q.frag is None:
            self.dust.q.addfield(
                "frag", np.ones(shape1), description="Fragmentation distribution exponent"
            )
            self.dust.q.frag.updater = std.dust.q_frag
        if self.dust.q.turb1 is None:
            self.dust.q.addfield(
                "turb1", -3.75, description="Size distribution exponent in first turbulence regime"
            )
        if self.dust.q.turb2 is None:
            self.dust.q.addfield(
                "turb2", -3.5, description="Size distribution exponent in second turbulence regime"
            )
        if self.dust.q.drfrag is None:
            self.dust.q.addfield(
                "drfrag", -3.75, description="Size distribution exponent in drift-induced fragmentation regime"
            )
        if self.dust.q.sweep is None:
            self.dust.q.addfield(
                "sweep", -2.5, description="Size distribution exponent in the sweep-up regime"
            )
        # Specific particle sizes
        if self.dust.s.min is None:
            rho = self.dust.rhos[:, 0] * self.dust.fill[:, 0]
            smin = (3. * self.ini.grid.mmin / (4. * np.pi * rho)) ** (1. / 3.)
            self.dust.s.addfield(
                "min", smin, description="Minimum particle size"
            )
        # Particle size variation factors
        if self.dust.f.crit is None:
            self.dust.f.addfield(
                "crit", 0.475, description="Critical mass depletion coefficient for shrinking"
            )
        if self.dust.f.drift is None:
            self.dust.f.addfield(
                "drift", 0.7, description="Drift velocity calibration factor"
            )
        if self.dust.f.dvturb is None:
            self.dust.f.addfield(
                "dvturb", 0.1, description="Collision speed parameter in turb.-dom. regime"
            )
        if self.dust.f.dvdrift is None:
            self.dust.f.addfield(
                "dvdrift", 0.2, description="collision speed parameter in drift-dom. regime"
            )
        if self.dust.f.dv is None:
            self.dust.f.addfield(
                "dv", 0.15 * np.ones(shape1), description="effective collision speed parameter"
            )
            self.dust.f.dv.updater = std.dust.f_dv

        # Initialize dust quantities partly to calculate Sigma
        try:
            self.dust.update()
        except:
            pass

        # Initialize the initial maximum particle size
        # This needs the pressure gradiant. Therefore, the other quantities
        # need to be initialized previously
        if self.dust.s.max is None:
            smax = std.dust.smax_initial(self)
            self.dust.s.addfield(
                "max", smax, description="Maximum particle size"
            )
        self.dust.s.max.differentiator = std.dust.smax_deriv

        if self.dust.s._sdot_shrink is None:
            self.dust.s.addfield('_sdot_shrink', np.zeros(
                shape1), description="Shrinking rate of s_max via transport")

        if self.dust.s.lim is None:
            self.dust.s.addfield(
                'lim', 1e-4, description="Limiting size for shrinking")

        # Floor value
        if self.dust.SigmaFloor is None:
            # TODO: What is a reasonable value for this in TwoPopPy
            SigmaFloor = 1.e-50 * np.ones(shape2Sigma)
            self.dust.addfield(
                "SigmaFloor", SigmaFloor, description="Floor value of surface density [g/cm²]"
            )

        # Surface density, if not set
        if self.dust.Sigma is None:
            Sigma = std.dust.Sigma_initial(self)
            Sigma = np.where(Sigma <= self.dust.SigmaFloor,
                             0.1 * self.dust.SigmaFloor,
                             Sigma)
            self.dust.addfield(
                "Sigma", Sigma, description="Surface density per mass bin [g/cm²]"
            )
        self.dust.Sigma.jacobinator = std.dust.jacobian

        # Fully initialize dust quantities
        self.dust.update()

        # Hidden fields
        # We store the old values of the surface density in a hidden field
        # to calculate the fluxes through the boundaries in case of implicit integration.
        self.dust._SigmaOld = Field(
            self, self.dust.Sigma, description="Previous value of surface density [g/cm²]"
        )
        # The right-hand side of the matrix equation is stored in a hidden field
        self.dust._rhs = Field(self, np.zeros(
            shape2Sigmaravel), description="Right-hand side of matrix equation [g/cm²]"
        )
        # State vector
        self.dust.addfield("_Y", np.zeros((int(self.grid._Nm_short) + 1) * int(self.grid.Nr)),
                           description="Dust state vector (Sig1, Sig2, a_max * Sig1)")
        self.dust._Y.jacobinator = std.dust.Y_jacobian
        # The right-hand side of the state vector matrix equation is stored in a hidden field
        self.dust._Y_rhs = Field(self, np.zeros_like(
            self.dust._Y), description="Right-hand side of state vector matrix equation"
        )
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
                value=0.1 * self.dust.SigmaFloor[-1]
            )
