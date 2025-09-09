import pytest
import numpy as np
from tripod import Simulation
from tripod.std.component import Component, DustComponent, GasComponent
import dustpy.constants as c

class TestComponent:
    @pytest.fixture
    def sim(self):
        """Create a basic simulation for testing"""
        sim = Simulation()
        sim.ini.grid.Nr = 50
        sim.initialize()
        return sim
    
    def test_component_creation_gas_tracer(self, sim):
        """Test creating a gas tracer component"""
        comp = Component(sim, dust_active=False, gas_active=False, gas_tracer=True)
        
        assert comp.dust._active == False
        assert comp.gas._active == False
        assert comp.gas._tracer == True
        assert len(comp._Y) == sim.grid.Nr  # Only gas tracer
        assert "gas" in comp.updateorder
        assert "dust" not in comp.updateorder
    
    def test_component_creation_dust_active(self, sim):
        """Test creating a dust active component"""
        comp = Component(sim, dust_active=True, gas_active=False, gas_tracer=False)
        
        assert comp.dust._active == True
        assert comp.gas._active == False
        assert comp.gas._tracer == False
        assert len(comp._Y) == sim.grid.Nr * 2  # Dust has 2*Nr
        assert "dust" in comp.updateorder
        assert "gas" not in comp.updateorder
    
    def test_component_creation_both_active(self, sim):
        """Test creating component with both dust and gas active"""
        comp = Component(sim, dust_active=True, gas_active=True, gas_tracer=False)
        
        assert comp.dust._active == True
        assert comp.gas._active == True
        assert comp.gas._tracer == False
        assert len(comp._Y) == sim.grid.Nr * 3  # Gas + 2*Dust
        assert "dust" in comp.updateorder
        assert "gas" in comp.updateorder
    
    def test_component_invalid_gas_config(self, sim):
        """Test that gas cannot be both active and tracer"""
        with pytest.raises(AssertionError):
            Component(sim, dust_active=False, gas_active=True, gas_tracer=True)

class TestDustComponent:
    @pytest.fixture
    def sim(self):
        sim = Simulation()
        sim.ini.grid.Nr = 50
        sim.initialize()
        return sim
    
    def test_dust_component_inactive(self, sim):
        """Test inactive dust component behavior"""
        dust = DustComponent(sim, active=False)
        
        # Should return zeros when inactive
        assert np.all(dust.value == 0)
        assert np.all(dust.S == 0)
        assert np.all(dust.S_Sigma == 0)
    
    def test_dust_component_active(self, sim):
        """Test active dust component"""
        dust = DustComponent(sim, active=True)
        test_value = np.ones_like(sim.dust.Sigma) * 1e-4
        
        dust.value = test_value
        assert np.allclose(dust.value, test_value)
    
    def test_dust_component_set_inactive_raises(self, sim):
        """Test that setting values on inactive dust raises error"""
        dust = DustComponent(sim, active=False)
        
        with pytest.raises(RuntimeError, match="Do not set dust parameter"):
            dust.value = np.ones_like(sim.dust.Sigma)
        
        with pytest.raises(RuntimeError, match="Do not set source term"):
            dust.S = np.ones_like(sim.dust.Sigma)
        
        with pytest.raises(RuntimeError, match="Do not set Sigma source"):
            dust.S_Sigma = np.ones_like(sim.dust.Sigma)

class TestGasComponent:
    @pytest.fixture
    def sim(self):
        sim = Simulation()
        sim.ini.grid.Nr = 50
        sim.initialize()
        return sim
    
    def test_gas_component_tracer(self, sim):
        """Test gas tracer component"""
        gas = GasComponent(sim, active=False, tracer=True)
        
        assert gas._active == False
        assert gas._tracer == True
        assert hasattr(gas, '_value')
        assert hasattr(gas, '_value_dot')
    
    def test_gas_component_active(self, sim):
        """Test active gas component"""
        gas = GasComponent(sim, active=True, tracer=False)
        
        assert gas._active == True
        assert gas._tracer == False
        assert hasattr(gas, '_Sigma')
        assert hasattr(gas, '_Sigma_dot')
    
    def test_gas_component_both_raises(self, sim):
        """Test that gas cannot be both active and tracer"""
        with pytest.raises(AssertionError, match="cannot be both active and tracer"):
            GasComponent(sim, active=True, tracer=True)