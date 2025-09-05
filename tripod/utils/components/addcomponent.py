from tripod.std import gas
from .component import Component
from ... import std 
from functools import partial
from simframe import Instruction
from dustpy.utils import Boundary
import numpy as np
import dustpy.constants as c



def addcomponent_c(self, name, gas_value, mu, dust_value = None , dust_active=False, gas_active=False, gas_tracer=False, description=""):
     
    #check if component with name already exists
    if name in self.components.__dict__:
        raise RuntimeError(
                "Component with name {} already exists.".format(name))
    

    #initalize gas and dust fields
    description = description + name
    comp = Component(self, dust_active=dust_active, gas_active=gas_active, gas_tracer=gas_tracer, description=description)

    #Jacobinator for state vector
    comp._Y.jacobinator = partial(std.compo.c_jacobian,component=comp)



    #TODO cleaner initalisation
    if comp.gas._active:
        comp.gas.Fi.updater = partial(std.gas.Fi_compo,group=comp.gas)   
        comp.gas.S.ext.updater = lambda sim: comp.gas.Sigma_dot    
        comp.gas.S.tot.updater = partial(std.gas.S_tot_compo,group=comp.gas)
        comp.gas.S.hyd.updater = partial(std.gas.S_hyd_compo,group=comp.gas)
        comp.gas.updater = ["Fi","S"]
        comp.gas.S.updater = ["ext","hyd","tot"]
        # set initial gas values
        comp.gas.Sigma = gas_value
        comp.gas.pars.mu = mu
        comp.gas._SigmaOld = gas_value

    if comp.gas._tracer:
        comp.gas._value = gas_value

    if comp.dust._active:
        if dust_value is None:
            raise RuntimeError("Dust value must be provided if dust_active is True.")
        comp.dust.value = dust_value


    # Adding component to updater
    self.components.__dict__[name] = comp
    if self.components.updater is None:
        self.components.updater = [name]
    else:
        lst = list(self.components.updateorder)
        lst.append(name)
        self.components.updater = lst


    # TODO cleaner way to set up boundaries and what values to use as default
    # Set boundary conditions values 
    if comp.dust._active and (comp.gas._active or comp.gas._tracer):
        val_inner = np.hstack((gas_value[0], dust_value[0,:]))
        val_outer = np.hstack((0.1*self.gas.SigmaFloor[-1], 0.1*self.dust.SigmaFloor[-1,:]))
    elif comp.gas._active or comp.gas._tracer:
        val_inner = gas_value[0]
        val_outer = 0.1*self.gas.SigmaFloor[-1]
    elif comp.dust._active:
        val_inner = dust_value[0,:]
        val_outer = 0.1*self.dust.SigmaFloor[-1,:]
    else:
        raise RuntimeError("Component must be either gas or dust active.")
    
    comp.boundary.inner = Boundary(
                self.grid.r,
                self.grid.ri,
                comp._Y,
                condition="val",
                value=val_inner
            )
    comp.boundary.outer = Boundary(
                self.grid.r[::-1],
                self.grid.ri[::-1],
                comp._Y[::-1],
                condition="val",
                value=val_outer
            )


    inst = Instruction(
                    std.compo._f_impl_1_direct_compo,
                    self.components.__dict__[name]._Y,
                    description="{}: implicit 1st-order direct solver for tracers".format(name),
                    controller={"boundary": self.components.__dict__[name].boundary,
                    "Sext": self.components.__dict__[name]._S,
                    "rhs": self.components.__dict__[name]._Y_rhs})
    self.integrator.instructions.append(inst)
