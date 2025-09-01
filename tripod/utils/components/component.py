import numpy as np
from simframe.frame import Group
import dustpy.constants as c

class Component(Group):

    def __init__(self, owner, updater=None, description=""):
        #TODO: modify description with active, tracer flags
        super().__init__(owner, updater=updater, description=description)

        dust = DustComponent(owner, description="Dust related fields")
        self.dust = dust

        gas = GasComponent(owner, description="Gas related fields")
        self.gas = gas


class DustComponent(Group):

    def __init__(self, owner, updater=None, active=False, description=""):
        super().__init__(owner, updater=updater, description=description)
        
        self._active = active
        self.addfield("_value", np.zeros_like(owner.dust.Sigma), description="Gas surface density [g/cm²]")
    #TODO: Add sources for value and Sigma (S, Sigma_dot)
    @property
    def value(self):
        if self._active:
            return self._value
        return 0. * self._value
    
    @value.setter
    def value(self, value):
        if self._active:
            self._value = value
        else:
            raise RuntimeError("Do not set dust parameter for inactive dust species.")

class GasComponent(Group):
    #TODO: Add tracer flag -> Sigma_dot=0 when tracer
    def __init__(self, owner, updater=None, active=False, description=""):
        super().__init__(owner, updater=updater, description=description)

        self._active = active
        self.addfield("_Sigma", np.zeros_like(owner.gas.Sigma), description="Gas surface density [g/cm²]")
        self.pars = Group(owner, description="Gas parameters")
        self.pars.mu = 2.3 * c.m_p
    #TODO: Add sources for value and Sigma (S, Sigma_dot)
    @property
    def Sigma(self):
        if self._active:
            return self._Sigma
        return 0. * self._Sigma
    
    @Sigma.setter
    def Sigma(self, value):
        if self._active:
            self._Sigma = value
        else:
            raise RuntimeError("Do not set gas surface density for inactive gas species.")