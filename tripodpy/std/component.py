import numpy as np
from simframe.frame import Group
import dustpy.constants as c
class Component(Group):
    def __init__(self, owner, updater=None,dust_tracer= False,dust_active=False,gas_active=False,gas_tracer=False,description=""):
        super().__init__(owner, updater=updater, description=description)

        dust = DustComponent(owner,tracer=dust_tracer,active=dust_active, description="Dust related fields")
        self.dust = dust

        gas = GasComponent(owner,active=gas_active,tracer=gas_tracer, description="Gas related fields")
        self.gas = gas

        #add which components are active in the description
        desc = self.description
        desc += f" (dust_tracer={dust_tracer}, gas_active={gas_active}, gas_tracer={gas_tracer})"
        self.description = desc

        # Initalize combined state vector 2*Nr for dust, Nr for gas (added if poth are present)
        n = owner.grid.Nr * (gas_active or gas_tracer) + (dust_tracer or dust_active) * owner.grid.Nr*2

        self.addfield("_Y",np.zeros(n), description="Combined state vector for component [???]")
        self.addfield("_S",np.zeros(n), description="Combined source term for component [???/s]")
        self.addfield("_Y_rhs",np.zeros(n), description="Right-hand side for implicit solver [???/s]")
        self.addgroup("boundary", description="Boundary conditions for component")

        #Set updater 
        lst = []
        if dust_tracer or dust_active:
            lst.append("dust")
        if gas_active or gas_tracer:
            lst.append("gas")
        self.updater = lst



class DustComponent(Group):
    def __init__(self, owner, updater=None, tracer=False,active=False, description=""):
        super().__init__(owner, updater=updater, description=description)
        
        self._tracer = tracer
        self._active = active
        assert not (self._tracer and self._active), "Dust component cannot be both active and tracer."
        self.addfield("_Sigma", np.zeros_like(owner.dust.Sigma), description="Dust surface density [g/cm²]")
        self.addfield("_SigmaOld", np.zeros_like(owner.dust.Sigma), description="Dust surface density [g/cm²]")
        self.addfield("_value", np.zeros_like(owner.dust.Sigma), description="tracer value [???]")
        self.addfield("_value_dot" , np.zeros_like(owner.dust.Sigma), description="Gas parameter source term [???/s]")
        #self.addfield("_S", np.zeros_like(owner.dust.Sigma), description="tracer source term [???/s]")
        self.addfield("_Sigma_dot", np.zeros_like(owner.dust.Sigma), description="Gas surface density source term [g/cm²/s]")
        self.addfield("_S_Sigma", np.zeros_like(owner.dust.Sigma), description="Source term for dust surface density [g/cm²]")
    
        self.addgroup("S", description="Gas source terms")
        self.addfield("Fi", np.zeros_like(owner.dust.Fi.tot), description="Gas flux [g/cm²/s]")
        self.S.addfield("ext", np.zeros_like(owner.dust.Sigma), description="External source term [g/cm²/s]")
        self.S.addfield("hyd", np.zeros_like(owner.dust.Sigma), description="Hydrodynamical source term [g/cm²/s]")
        self.S.addfield("coag", np.zeros_like(owner.dust.Sigma), description="Hydrodynamical source term [g/cm²/s]")
        self.S.addfield("tot", np.zeros_like(owner.dust.Sigma), description="Total source term [g/cm²/s]")

        self.addgroup("pars", description="Dust parameters")
        self.pars.addfield("rhos", 1.0, description="Material density of dust grains [g/cm³]")
    @property
    def value(self):
        if self._tracer:
            return self._value
        return 0. * self._value
    
    @value.setter
    def value(self, value):
        if self._tracer:
            self._value = value
        else:
            raise RuntimeError("Do not set dust parameter for inactive dust species.")
            
    @property
    def S_Sigma(self):
        if self._active:
            return self._S_Sigma
        return 0. * self._S_Sigma
    
    @S_Sigma.setter
    def S_Sigma(self, value):
        if self._active:
            self._S_Sigma = value
        else:
            raise RuntimeError("Do not set Sigma source for inactive dust species.")
        
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
            raise RuntimeError("Do not set Sigma for inactive dust species.")       
        
    @property
    def Sigma_dot(self):
        if self._active:
            return self._Sigma_dot
        return 0. * self._Sigma_dot
    
    @Sigma_dot.setter
    def Sigma_dot(self, value):
        if self._active:
            self._Sigma_dot = value
        else:
            raise RuntimeError("Do not set Sigma source for inactive dust species.")
        
    @property
    def value_dot(self):
        if self._tracer:
            return self._value_dot
        return 0. * self._value_dot
    
    @value_dot.setter
    def value_dot(self, value):
        if self._tracer:
            self._value_dot = value
        else:
            raise RuntimeError("Do not set gas parameter source for inactive dust species.")

class GasComponent(Group):
    def __init__(self, owner, updater=None, active=False,tracer=True, description=""):
        super().__init__(owner, updater=updater, description=description)

        self._active = active
        self._tracer = tracer
        # flags are handled excluively -> tracer only or gas only never both
        assert not (self._active and self._tracer), "Gas component cannot be both active and tracer."

        #active gas component
        self.addfield("_Sigma", np.zeros_like(owner.gas.Sigma), description="Gas surface density [g/cm²]") 
        self.addfield("_SigmaOld", np.zeros_like(owner.gas.Sigma), description="Gas surface density [g/cm²]")  
        self.addfield("_Sigma_dot", np.zeros_like(owner.gas.Sigma), description="Gas surface density source term [g/cm²/s]")

        #tracer fields
        self.addfield("_value", np.zeros_like(owner.gas.Sigma), description="Gas parameter [???]")
        self.addfield("_value_dot" , np.zeros_like(owner.gas.Sigma), description="Gas parameter source term [???/s]")

        #add the extra parameters needed for the gas component
        self.addgroup("pars", description="Gas parameters")
        self.pars.addfield("mu", 2.3 * c.m_p, description="Mean molecular weight [g]")
        self.pars.addfield("nu", 1.4, description="trial frequency [1/s]")
        self.pars.addfield("Tsub", 10., description="evaporation temperatue [K]")

        #stuff needed for the timestep
        self.addfield("Fi", np.zeros(owner.grid.Nr + 1 ), description="Gas flux [g/cm²/s]")
        self.addgroup("S", description="Gas source terms")
        self.S.addfield("ext", np.zeros_like(owner.gas.Sigma), description="External source term [g/cm²/s]")
        self.S.addfield("hyd", np.zeros_like(owner.gas.Sigma), description="Hydrodynamical source term [g/cm²/s]")
        self.S.addfield("tot", np.zeros_like(owner.gas.Sigma), description="Total source term [g/cm²/s]")

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
            raise RuntimeError("Do not set Sigma for inactive gas species.")
        
    @property
    def Sigma_dot(self):
        if self._active:
            return self._Sigma_dot
        return 0. * self._Sigma_dot
    
    @Sigma_dot.setter
    def Sigma_dot(self, value):
        if self._active:
            self._Sigma_dot = value
        else:
            raise RuntimeError("Do not set Sigma source for inactive gas species.")
        
    @property
    def value(self):
        if self._tracer:
            return self._value
        return 0. * self._value
    
    @value.setter
    def value(self, value):
        if self._tracer:
            self._value = value
        else:
            raise RuntimeError("Do not set gas parameter for inactive gas species.")
        
    @property
    def value_dot(self):
        if self._tracer:
            return self._value_dot
        return 0. * self._value_dot
    
    @value_dot.setter
    def value_dot(self, value):
        if self._tracer:
            self._value_dot = value
        else:
            raise RuntimeError("Do not set gas parameter source for inactive gas species.")