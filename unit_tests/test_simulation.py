import pytest
import numpy as np
from tripod import Simulation
import dustpy.constants as c

class TestSimulation:
    def test_simulation_creation(self):
        """Test basic simulation creation"""
        sim = Simulation()
        assert sim is not None
    
    def test_simulation_initialization(self):
        """Test simulation initialization"""
        sim = Simulation()
        sim.ini.grid.rmin = 1 * c.au
        sim.ini.grid.rmax = 100 * c.au
        sim.ini.grid.Nr = 100
        sim.initialize()
        
        assert len(sim.grid.r) == 100
        assert sim.grid.r[0] >= 1 * c.au
        assert sim.grid.r[-1] <= 100 * c.au
    
    def test_add_component(self):
        """Test adding components"""
        sim = Simulation()
        sim.ini.grid.Nr = 50
        sim.initialize()
        
        tr = np.ones_like(sim.grid.r) * 1e-5
        sim.addcomponent_c("test_tracer", tr, 0.32, 
                          dust_active=False, gas_active=False, gas_tracer=True)
        
        assert hasattr(sim.components, "test_tracer")
        assert sim.components.test_tracer.gas._tracer == True