import numpy as np
from simframe.frame import Group
import dustpy.constants as c
class Component(Group):
    def __init__(self, owner, updater=None,dust_active= False,gas_active=False,gas_tracer=False,description=""):
        super().__init__(owner, updater=updater, description=description)

        dust = DustComponent(owner,active=dust_active, description="Dust related fields")
        self.dust = dust

        gas = GasComponent(owner,active=gas_active,tracer=gas_tracer, description="Gas related fields")
        self.gas = gas

        #add which components are active in the description
        desc = self.description
        desc += f" (dust_active={dust_active}, gas_active={gas_active}, gas_tracer={gas_tracer})"
        self.description = desc

        # Initalize combined state vector 2*Nr for dust, Nr for gas (added if poth are present)
        comp_type = (dust_active, gas_active, gas_tracer)
        self.addfield("_comp_type", comp_type, description="Type of component (dust, gas)")
        n = owner.grid.Nr * (gas_active or gas_tracer) + dust_active * owner.grid.Nr*2
        self.addfield("_Y",np.zeros(n), description="Combined state vector for component [???]")
        self.addfield("_S",np.zeros(n), description="Combined source term for component [???/s]")
        self.addfield("_Y_rhs",np.zeros(n), description="Right-hand side for implicit solver [???/s]")
        self.addgroup("boundary", description="Boundary conditions for component")


        #Set updater 
        lst = []
        if dust_active:
            lst.append("dust")
        if gas_active or gas_tracer:
            lst.append("gas")
        self.updater = lst



class DustComponent(Group):
    _active = False
    def __init__(self, owner, updater=None, active=False, description=""):
        super().__init__(owner, updater=updater, description=description)
        
        self._active = active
        self.addfield("_value", np.zeros_like(owner.dust.Sigma), description="tracer value [???]")
        self.addfield("_S", np.zeros_like(owner.dust.Sigma), description="tracer source term [???/s]")
        self.addfield("_S_Sigma", np.zeros_like(owner.dust.Sigma), description="Source term for dust surface density [g/cm²]")

    def guarded_property(attr_name,error_msg,activator="_active"):
        """Return a property linked to a private attribute, with active check."""
        private_attr = f"_{attr_name}"
        def getter(self):
            if self.__dict__.get(activator,False):
                return getattr(self, private_attr)
            return 0. * getattr(self, private_attr)

        def setter(self, value):
            if self.__dict__.get(activator,False):
                setattr(self, private_attr, value)
            else:
                raise RuntimeError(error_msg)
        return property(getter, setter)
    #add properties

    value = guarded_property("value", "Do not set dust parameter for inactive dust species.",activator="_active")
    S = guarded_property("S", "Do not set source term for inactive dust species.",activator="_active")
    S_Sigma = guarded_property("S_Sigma", "Do not set Sigma source for inactive dust species.",activator="_active")



class GasComponent(Group):
    _active = False 
    _tracer = False

    def __init__(self, owner, updater=None, active=False,tracer=True, description=""):
        super().__init__(owner, updater=updater, description=description)

        self._active = active
        self._tracer = tracer
        # flags are handled excluively -> tracer only or gas only never both
        assert not (self._active and self._tracer), "Gas component cannot be both active and tracer."

        self.addfield("_Sigma", np.zeros_like(owner.gas.Sigma), description="Gas surface density [g/cm²]") 
        self.addfield("_SigmaOld", np.zeros_like(owner.gas.Sigma), description="Gas surface density [g/cm²]")  
       
        self.addfield("_Sigma_dot", np.zeros_like(owner.gas.Sigma), description="Gas surface density source term [g/cm²/s]")
        
    

        #
        self.addfield("_value", np.zeros_like(owner.gas.Sigma), description="Gas parameter [???]")
        

        self.addfield("_value_dot" , np.zeros_like(owner.gas.Sigma), description="Gas parameter source term [???/s]")

        #add the extra parameters needed for the gas component
        self.addgroup("pars", description="Gas parameters")
        self.pars.addfield("_mu", 2.3 * c.m_p, description="Mean molecular weight [g]")
        self.pars.addfield("mu", 2.3 * c.m_p, description="Mean molecular weight [g]")
        



        #stuff needed for the timestep
        self.addfield("_Fi", np.zeros(owner.grid.Nr + 1 ), description="Gas flux [g/cm²/s]")
        self.addgroup("S", description="Gas source terms")
        self.S.addfield("ext", np.zeros_like(owner.gas.Sigma), description="External source term [g/cm²/s]")
        self.S.addfield("hyd", np.zeros_like(owner.gas.Sigma), description="Hydrodynamical source term [g/cm²/s]")
        self.S.addfield("tot", np.zeros_like(owner.gas.Sigma), description="Total source term [g/cm²/s]")

    # add the guarded properties for the gas component 
    #TODo see if there is abetter way to do this
    def guarded_property(attr_name,error_msg,activator="_active "):
        """Return a property linked to a private attribute, with active check."""
        private_attr = f"_{attr_name}"
        def getter(self):
            if self.__dict__.get(activator,False):
                return getattr(self, private_attr)
            return 0. * getattr(self, private_attr)

        def setter(self, value):
            if self.__dict__.get(activator,False):
                setattr(self, private_attr, value)
            else:
                raise RuntimeError(error_msg)
        return property(getter, setter)

    #add properties
    Sigma = guarded_property("Sigma", "Do not set Sigma for inactive gas species.",activator="_active")
    Sigma_dot = guarded_property("Sigma_dot", "Do not set Sigma source for inactive gas species.",activator="_active")
    value =  guarded_property("value","do not set gas parameter for inactive gas species.",activator="_tracer")
    Fi = guarded_property("Fi", "Do not set flux for inactive gas species.",activator="_active")
    value_dot = guarded_property("value_dot","do not set gas parameter source for inactive gas species.",activator="_tracer")
    #pars.mu = guarded_property("mu", "Do not set mu for inactive gas species.",active=self._active or self._tracer)