from calendar import c
from .component import Component
from ... import std 
from functools import partial
from simframe import Instruction
from dustpy.utils import Boundary



def addcomponent_c(self, name, Sigma_gas, mu, dust_active=False, gas_active=False, gas_tracer=False, description=""):
     

    if name in self.components.__dict__:
        raise RuntimeError(
                "Component with name {} already exists.".format(name))
    description = description + name
    comp = Component(self, dust_active=dust_active, gas_active=gas_active, gas_tracer=gas_tracer, description=description)
    comp._Y.jacobinator = partial(std.compo.c_jacobian,name=name,comp_type=comp._comp_type)


            #set updaters if active
    def updater(sim):
        comp.gas.Fi.update()
        comp.gas.S.update()


    if comp.gas._active:
        comp.gas.Fi.updater = partial(std.gas.Fi_compo,group=comp.gas)       
        comp.gas.S.tot.updater = partial(std.gas.S_tot_compo,group=comp.gas)
        comp.gas.S.hyd.updater = partial(std.gas.S_hyd_compo,group=comp.gas)

        comp.gas.updater = updater
        comp.gas.S.updater = ["ext","hyd","tot"]
    #Set initial conditions
    if gas_active or gas_tracer:
        comp.gas.Sigma = Sigma_gas
        comp.gas.pars.mu = mu
        comp.gas._SigmaOld = Sigma_gas

    #add boundary conditions



    # Adding component to updater
    self.components.__dict__[name] = comp
    if self.components.updater is None:
        self.components.updater = [name]
    else:
        lst = list(self.components.updateorder)
        lst.append(name)
        self.components.updater = lst

    
    #TODO proper handeling of boundary conditions for different component types
    comp.boundary.inner = Boundary(
                self.grid.r,
                self.grid.ri,
                self.components.__dict__[name].gas.Sigma,
                condition="val",
                value=Sigma_gas[0]
            )
    comp.boundary.outer = Boundary(
                self.grid.r[::-1],
                self.grid.ri[::-1],
                self.components.__dict__[name].gas.Sigma[::-1],
                condition="val",
                value=0.1*self.gas.SigmaFloor[-1]
            )


    inst = Instruction(
                    std.compo._f_impl_1_direct_compo,
                    self.components.__dict__[name]._Y,
                    description="{}: implicit 1st-order direct solver for tracers".format(name),
                    controller={"boundary": self.components.__dict__[name].boundary,
                    "Sext": self.components.__dict__[name]._S,
                    "rhs": self.components.__dict__[name]._Y_rhs})
    self.integrator.instructions.append(inst)

    # Set bounda cond 











def addcomponent(self, name, Sigma, mu, tracer=False,includegas = True, includedust=False, description=""):

        if name in self.components.__dict__:
            raise RuntimeError(
                "Component with name {} already exists.".format(name))

        if Sigma.shape != self.grid.r.shape:
            raise RuntimeError(
                "Sigma does not have the correct shape of {}".format(self.grid.r.shape))
        # Adding group and fields
        self.components.addgroup(name, description=description)
        self.components.__dict__[name].addfield(
            "tracer", tracer, description="Is this component tracer?")
        self.components.__dict__[name].addfield(
            "includedust", includedust, description="Is this component tracer?")
        self.components.__dict__[name].addfield(
            "includegas", includegas, description="Is this component tracer?")
        # Adding component to updater
        self.components.updater = self.components.updateorder + [name]
        self.components.__dict__[name].updater = []

        # Adding fields for gas component
        if includegas:
            self.components.__dict__[name].addgroup("gas", description="Gas phase of component")
            self.components.__dict__[name].updater = self.components.__dict__[name].updateorder + ["gas"]


            self.components.__dict__[name].gas.addfield(
                "mu", mu, description="Molecular weight [g]")
            self.components.__dict__[name].gas.addfield(
                "Sigma", Sigma, description="Surface density [g/cm²]")
            self.components.__dict__[name].gas.addfield(
                "_SigmaOld", Sigma, description="Surface density [g/cm²]")
            self.components.__dict__[name].gas.addfield(
                "Fi", np.zeros(self.grid.Nr + 1 ), description="Surface density [g/cm²]")
            self.components.__dict__[name].gas.Fi.updater = partial(std.gas.Fi_compo,compkey=name)

            # Adding S group 
            self.components.__dict__[name].gas.addgroup("S", description="Sources")

            self.components.__dict__[name].gas.S.addfield(
                "ext", np.zeros_like(Sigma), description="External sources [g/cm²/s]")
            self.components.__dict__[name].gas.S.addfield(
                "hyd", np.zeros_like(Sigma), description="External sources [g/cm²/s]")
            self.components.__dict__[name].gas.S.hyd.updater = partial(std.gas.S_hyd_compo,compkey=name)
            self.components.__dict__[name].gas.S.addfield(
                "tot", np.zeros_like(Sigma), description="External sources [g/cm²/s]")
            self.components.__dict__[name].gas.S.tot.updater = partial(std.gas.S_tot_compo,compkey=name)    
            # adding the updater for the gas component
            self.components.__dict__[name].gas.updater = ["Fi","S"]
            self.components.__dict__[name].gas.S.updater = ["ext","hyd","tot"]

            #add boundaries for gas component
            self.components.__dict__[name].gas.addgroup(
                "boundary", description="Boundary conditions")
            self.components.__dict__[name].gas.boundary.inner = Boundary(
                self.grid.r,
                self.grid.ri,
                self.components.__dict__[name].gas.Sigma,
                condition="const_grad"
            )

            self.components.__dict__[name].gas.boundary.outer = Boundary(
                self.grid.r[::-1],
                self.grid.ri[::-1],
                self.components.__dict__[name].gas.Sigma[::-1],
                condition="val",
                value=0.1*self.gas.SigmaFloor[-1]
            )

        if includedust:
            #add dust group
            self.components.__dict__[name].addgroup("dust", description="Dust component")
            self.components.__dict__[name].updater = self.components.__dict__[name].updateorder + ["dust"]



            shape = (int(self.grid.Nr), int(self.grid._Nm_short))
            self.components.__dict__[name].dust.addfield(
            "Sigma", np.zeros(shape), description="Surface density [g/cm²]")

            self.components.__dict__[name].dust.addfield(
            "_SigmaOld", np.zeros(shape), description="Surface density [g/cm²]")
            self.components.__dict__[name].dust.addgroup("S", description="Sources")
            self.components.__dict__[name].dust.S.addfield("ext", np.zeros(shape), description="source")
            self.components.__dict__[name].dust.S.addfield("hyd", np.zeros(shape), description="source")
            self.components.__dict__[name].dust.S.addfield("coag", np.zeros(shape), description="source")
            self.components.__dict__[name].dust.S.addfield("tot", np.zeros(shape), description="source")

            # adding the updater for the dust component
            self.components.__dict__[name].dust.updater = ["S"]
            self.components.__dict__[name].dust.S.updater = ["ext", "hyd", "coag", "tot"]

            # quntities for sublimation
            self.components.__dict__[name].addfield("Tsub", 0 , description="Sublimatio Temperature [K]")
            self.components.__dict__[name].addfield("nu", 0 , description="attempt frequency for sublimation")


        # define integrator instructions 
        if (includedust and includegas):

            # State vector
            self.components.__dict__[name].addfield("_Y", np.zeros((int(self.grid._Nm_short) + 1) * int(self.grid.Nr)),
                            description="Dust state vector (siggas , sig0, sig1)")
            self.components.__dict__[name]._Y.jacobinator = partial(std.compo.Y_jacobian,name=name)

            # The right-hand side of the state vector matrix equation is stored in a hidden field
            self.components.__dict__[name]._Y_rhs = Field(self, np.zeros_like(
                self.components.__dict__[name]._Y), description="Right-hand side of state vector matrix equation")
                        # Integrator
            inst = Instruction(
                std.compo._f_impl_1_direct_compo,
                self.components.__dict__[name]._Y,
                controller={"name": name },
                description="{}: implicit 1st-order direct solver for tracers".format(name))
            
            self.integrator.instructions.append(
                inst)
            
        elif includegas:
            self.components.__dict__[
                name].gas.Sigma.jacobinator = dp.std.gas.jacobian

            # Integrator
            inst = Instruction(
                dp.std.gas.impl_1_direct,
                self.components.__dict__[name].gas.Sigma,
                controller={
                    "boundary": self.components.__dict__[name].gas.boundary,
                    "Sext": self.components.__dict__[name].gas.S.ext,
                },
                description="{}: implicit 1st-order direct solver".format(name)
            )
            self.integrator.instructions.append(
                inst)

        elif includedust:
            print("Dust component {} added without gas component. This is not recommended.".format(name))
            exit()


