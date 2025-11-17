from tripod.std import compo as compo
from tripod.std import gas as gas
from tripod.std import dust as dust
from tripod.std.dust import Sigma_initial

from .component import Component
from functools import partial
from simframe import Instruction
from dustpy.utils import Boundary
import dustpy.std.dust as dp_dust
from simframe import schemes
import numpy as np
import dustpy.constants as c



def addcomponent(self, name, gas_value, mu, dust_value = None ,dust_active=False,dust_tracer=False, gas_active=False, gas_tracer=False, description="",rhos=1.0):
     
    #check if component with name already exists
    if name in self.components.__dict__:
        raise RuntimeError(
                "Component with name {} already exists.".format(name))
    

    #initalize gas and dust fields
    description = description + name
    comp = Component(self, dust_tracer=dust_tracer,dust_active=dust_active, gas_active=gas_active, gas_tracer=gas_tracer, description=description)

    #Jacobinator for state vector
    comp._Y.jacobinator = partial(compo.c_jacobian,component=comp)

    assert not (comp.dust._tracer and comp.dust._active), "Dust component cannot be both active and tracer."
    assert not (comp.gas._tracer and comp.gas._active), "Gas component cannot be both active and tracer."

    #TODO cleaner initalisation
    if comp.gas._active:
        comp.gas.Fi.updater = partial(gas.Fi_compo,group=comp.gas)   
        comp.gas.S.ext.updater = lambda sim: comp.gas.Sigma_dot    
        comp.gas.S.tot.updater = partial(gas.S_tot_compo,group=comp.gas)
        comp.gas.S.hyd.updater = partial(gas.S_hyd_compo,group=comp.gas)
        comp.gas.updater = ["Fi","S"]
        comp.gas.S.updater = ["ext","hyd","tot"]
        # set initial gas values
        comp.gas.Sigma = gas_value
        comp.gas.pars.mu = mu
        comp.gas._SigmaOld = gas_value

    if comp.gas._tracer:
        comp.gas._value = gas_value

    if comp.dust._tracer:
        if dust_value is None:
            raise RuntimeError("Dust value must be provided if dust_tracer is True.")
        comp.dust.value = dust_value

    if comp.dust._active:
        comp.dust.Sigma = dust_value
        comp.dust.Fi.updater = partial(dp_dust.F_tot,Sigma=comp.dust.Sigma)
        comp.dust.S.hyd.updater = partial(dust.S_hyd_compo,group=comp.dust)
        comp.dust.S.coag.updater = partial(dust.S_coag,Sigma=comp.dust.Sigma)
        comp.dust.S.tot.updater = partial(dust.S_tot_compo,group=comp.dust)
        comp.dust.S.ext.updater = lambda sim: comp.dust.Sigma_dot
        comp.dust.updater = ["Fi","S"]
        comp.dust.S.updater = ["ext","hyd","coag","tot"]
        comp.dust.pars.rhos = rhos
        comp.dust._SigmaOld = dust_value

        if not self._dust_compo:
            self.dust.rhos.updater = dust.rhos_compo
            self._dust_compo = True




    # Adding component to updater
    self.components.__dict__[name] = comp
    if self.components.updateorder is None:
        self.components.updater = [name]
    else:
        lst = list(self.components.updateorder)
        lst.append(name)
        self.components.updater = lst


    if gas_active:

        comp.gas.addgroup("boundary")
        
        comp.gas.boundary.inner = Boundary(self.grid.r, self.grid.ri, comp.gas.Sigma)
        comp.gas.boundary.inner.setcondition("const_grad")
        
        comp.gas.boundary.outer = Boundary(self.grid.r[::-1], self.grid.ri[::-1], comp.gas.Sigma[::-1])
        comp.gas.boundary.outer.setcondition("val", self.gas.SigmaFloor[-1])

    elif gas_tracer:
        comp.gas.addgroup("boundary")

        comp.gas.boundary.inner = Boundary(self.grid.r, self.grid.ri, comp.gas.value)
        comp.gas.boundary.inner.setcondition("val", gas_value[0])
        
        comp.gas.boundary.outer = Boundary(self.grid.r[::-1], self.grid.ri[::-1], comp.gas.value[::-1])
        comp.gas.boundary.outer.setcondition("val", gas_value[-1])



    if dust_tracer:
        comp.dust.addgroup("boundary")

        comp.dust.boundary.inner = Boundary(self.grid.r, self.grid.ri, comp.dust.value)
        comp.dust.boundary.inner.setcondition("val", dust_value[0,:])
        
        comp.dust.boundary.outer = Boundary(self.grid.r[::-1], self.grid.ri[::-1], comp.dust.value[::-1])
        comp.dust.boundary.outer.setcondition("val", dust_value[-1,:])

    if dust_active:
        comp.dust.addgroup("boundary")

        comp.dust.boundary.inner = Boundary(self.grid.r, self.grid.ri, comp.dust.Sigma)
        comp.dust.boundary.inner.setcondition("const_grad")
        
        comp.dust.boundary.outer = Boundary(self.grid.r[::-1], self.grid.ri[::-1], comp.dust.Sigma[::-1])
        comp.dust.boundary.outer.setcondition("val", self.dust.SigmaFloor[-1,:])



    inst = Instruction(
                    compo._f_impl_1_direct_compo,
                    self.components.__dict__[name]._Y,
                    description="{}: implicit 1st-order direct solver for components".format(name),
                    controller={"boundary": self.components.__dict__[name].boundary,
                    "Sext": self.components.__dict__[name]._S,
                    "rhs": self.components.__dict__[name]._Y_rhs,
                    "name": name},)
    

    if comp.gas._active or comp.dust._active:
        upd = Instruction(schemes.update, self.components.__dict__[name]._Y, description="Update {}".format(name))
        self.integrator.instructions.insert(2, upd)
        self.integrator.instructions.insert(2, inst)
        
    else:
        self.integrator.instructions.append(inst)
