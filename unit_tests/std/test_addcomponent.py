from unittest import mock
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call
from functools import partial
from simframe import Instruction
from dustpy.utils import Boundary
import dustpy.constants as c
from tripodpy import Simulation

from tripodpy.std.addcomponent import addcomponent

class TestAddComponentBasic:
    @pytest.fixture
    def mock_sim(self):
        sim = Simulation()
        sim.ini.grid.Nr = 3
        sim.initialize()
        
        return sim
    
    def test_addcomponent_gas_active_basic(self, mock_sim):
        """Test adding a basic gas active component"""
        gas_value = np.ones(3) * 100.0
        mu = 18.0
        

        addcomponent(mock_sim, 'water', gas_value, mu, gas_active=True)
        mock_comp = mock_sim.components.__dict__['water']
        
        assert 'water' in mock_sim.components.__dict__
        
        # Verify gas properties were set
        np.testing.assert_array_equal(mock_comp.gas.Sigma, gas_value)
        np.testing.assert_array_equal(mock_comp.gas._SigmaOld, gas_value)
        assert mock_comp.gas.pars.mu == mu
        np.testing.assert_array_equal(mock_comp.gas._SigmaOld, gas_value)
        
        #assert state vector length
        assert len(mock_comp._Y) == mock_sim.grid.Nr

        # Verify boundaries were set
        assert mock_comp.gas.boundary.inner.condition is not None
        assert mock_comp.gas.boundary.outer.condition is not None
    
    def test_addcomponent_gas_tracer_basic(self, mock_sim):
        """Test adding a basic gas tracer component"""
        gas_value = np.ones(3) * 0.01
        mu = 28.0
        
        
        addcomponent(mock_sim, 'CO', gas_value, mu, gas_tracer=True)
        mock_comp = mock_sim.components.__dict__['CO']
        
        # Verify tracer value was set
        np.testing.assert_array_equal(mock_comp.gas._value, gas_value)
        np.testing.assert_array_equal(mock_comp.gas.Sigma, np.zeros(len(mock_sim.grid.r)))

        #assert state vector length
        assert len(mock_comp._Y) == mock_sim.grid.Nr
        
        # Verify boundaries were set with correct values
        assert mock_comp.gas.boundary.inner.condition == "val"
        assert mock_comp.gas.boundary.outer.condition == "val"
    
    
    def test_addcomponent_dust_active_basic(self, mock_sim):
        """Test adding a basic dust active component"""
        gas_value = np.ones(3) * 100.0
        dust_value = np.ones((3, 2)) * 10.0
        mu = 18.0
        rhos = 1
        
        addcomponent(mock_sim, 'ice', gas_value, mu, 
                        dust_value=dust_value, dust_active=True, rhos=rhos)
        mock_comp = mock_sim.components.__dict__['ice']

        # Verify dust properties were set
        np.testing.assert_array_equal(mock_comp.dust.Sigma, dust_value)
        assert mock_comp.dust.pars.rhos == rhos
        np.testing.assert_array_equal(mock_comp.dust._SigmaOld, dust_value)
        
        #asser state vector length
        assert len(mock_comp._Y) == mock_sim.grid.Nr * mock_sim.grid._Nm_short
        
        # Verify boundaries were set
        assert mock_comp.dust.boundary.inner.condition is not None
        assert mock_comp.dust.boundary.outer.condition is not None

        # Verify _dust_compo was activated
        assert mock_sim._dust_compo == True
        assert mock_sim.dust.rhos.updater is not None
    
    def test_addcomponent_dust_tracer_basic(self, mock_sim):
        """Test adding a basic dust tracer component"""
        gas_value = np.ones(3) * 100.0
        dust_value = np.ones((3, 2)) * 0.02
        mu = 18.0
        
        
        addcomponent(mock_sim, 'water_ice', gas_value, mu,
                        dust_value=dust_value, dust_tracer=True)
        mock_comp = mock_sim.components.__dict__['water_ice']
        
        # Verify tracer value was set
        np.testing.assert_array_equal(mock_comp.dust.value, dust_value)
        
        # Verify boundaries were set
        assert mock_comp.dust.boundary.inner.condition == "val"
        assert mock_comp.dust.boundary.outer.condition == "val"

    def test_addcomponent_no_updaters(self, mock_sim):
        """Test adding a component when no updaters exist yet"""
        gas_value = np.ones(3) * 100.0
        mu = 18.0

        del mock_sim.components
        mock_sim.addgroup("components")
        print("Initial updater:", mock_sim.components.updateorder)
        addcomponent(mock_sim, 'tracer', gas_value, mu, gas_tracer=True)
        
        # Verify component was added to updater list
        assert 'tracer' in mock_sim.components.updateorder
        assert mock_sim.components.updateorder == ['tracer']

class TestAddComponentMixed:
    @pytest.fixture
    def mock_sim(self):
        """Create a real simulation for mixed component testing"""
        sim = Simulation()
        sim.ini.grid.Nr = 3
        sim.initialize()
        return sim
    
    def test_addcomponent_gas_active_dust_tracer(self, mock_sim):
        """Test adding component with active gas and tracer dust"""
        gas_value = np.ones(3) * 100.0
        dust_value = np.ones((3, 2)) * 0.02
        mu = 18.0
        num_inst = len(mock_sim.integrator.instructions)
                    
        addcomponent(mock_sim, 'water', gas_value, mu,
                        dust_value=dust_value, gas_active=True, dust_active=True)
        
        mock_comp = mock_sim.components.__dict__['water']
        
        # Verify both gas and dust properties were set
        np.testing.assert_array_equal(mock_comp.gas.Sigma, gas_value)
        np.testing.assert_array_equal(mock_comp.dust.Sigma, dust_value)
        
        # Verify instruction was added for active component
        # Real simulation may have more instructions, so we check that instructions were added
        assert len(mock_sim.integrator.instructions) == num_inst +2

        #check state vector length
        assert len(mock_comp._Y) == mock_sim.grid.Nr * (1 + mock_sim.grid._Nm_short)

        #check boundaries
        assert mock_comp.gas.boundary.inner.condition is not None
        assert mock_comp.gas.boundary.outer.condition is not None
        assert mock_comp.dust.boundary.inner.condition is not None
        assert mock_comp.dust.boundary.outer.condition is not None
        

class TestAddComponentErrorHandling:
    @pytest.fixture
    def mock_sim(self):
        """Create a real simulation for error testing"""
        sim = Simulation()
        sim.ini.grid.Nr = 3
        sim.initialize()
        return sim
    
    def test_duplicate_component_name(self, mock_sim):
        """Test error when component name already exists"""
        # Add existing component
        existing_comp = Mock()
        mock_sim.components.__dict__['water'] = existing_comp
        
        gas_value = np.ones(3) * 100.0
        mu = 18.0
        
        with pytest.raises(RuntimeError, match="Component with name water already exists"):
            addcomponent(mock_sim, 'water', gas_value, mu, gas_active=True)
    
    def test_dust_tracer_without_dust_value(self, mock_sim):
        """Test error when dust_tracer is True but dust_value is None"""
        gas_value = np.ones(3) * 100.0
        mu = 18.0
        
        with patch('tripodpy.std.addcomponent.Component') as MockComponent:
            mock_comp = Mock()
            mock_comp.dust._tracer = True
            mock_comp.dust._active = False
            mock_comp.gas._tracer = False
            mock_comp.gas._active = False
            MockComponent.return_value = mock_comp
            
            with pytest.raises(RuntimeError, match="Dust value must be provided if dust_tracer is True"):
                addcomponent(mock_sim, 'ice', gas_value, mu, 
                             dust_tracer=True, dust_value=None)
    
    def test_dust_tracer_and_active_conflict(self, mock_sim):
        """Test assertion error when dust is both tracer and active"""
        gas_value = np.ones(3) * 100.0
        dust_value = np.ones((3, 2)) * 10.0
        mu = 18.0
        
        with patch('tripodpy.std.addcomponent.Component') as MockComponent:
            mock_comp = Mock()
            mock_comp.dust._tracer = True
            mock_comp.dust._active = True  # Conflict!
            mock_comp.gas._tracer = False
            mock_comp.gas._active = False
            MockComponent.return_value = mock_comp
            
            with pytest.raises(AssertionError, match="Dust component cannot be both active and tracer"):
                addcomponent(mock_sim, 'conflicted', gas_value, mu,
                             dust_value=dust_value, dust_tracer=True, dust_active=True)
    
    def test_gas_tracer_and_active_conflict(self, mock_sim):
        """Test assertion error when gas is both tracer and active"""
        gas_value = np.ones(3) * 100.0
        mu = 18.0
        
        with patch('tripodpy.std.addcomponent.Component') as MockComponent:
            mock_comp = Mock()
            mock_comp.dust._tracer = False
            mock_comp.dust._active = False
            mock_comp.gas._tracer = True
            mock_comp.gas._active = True  # Conflict!
            MockComponent.return_value = mock_comp
            
            with pytest.raises(AssertionError, match="Gas component cannot be both active and tracer"):
                addcomponent(mock_sim, 'conflicted', gas_value, mu,
                             gas_tracer=True, gas_active=True)

class TestAddComponentUpdaters:
    @pytest.fixture
    def mock_sim(self):
        """Create a real simulation for updater testing"""
        sim = Simulation()
        sim.ini.grid.Nr = 3
        sim.initialize()
        return sim
    
    def test_gas_active_updaters_set(self, mock_sim):
        """Test that gas active component updaters are properly set"""
        gas_value = np.ones(3) * 100.0
        mu = 18.0
        
        mock_sim.addcomponent('water', gas_value, mu, gas_active=True)
        mock_comp = mock_sim.components.__dict__['water']

        # Verify updaters were set
        assert mock_comp.gas.Fi.updater is not None
        assert mock_comp.gas.S.ext.updater is not None
        assert mock_comp.gas.S.tot.updater is not None
        assert mock_comp.gas.S.hyd.updater is not None
        
        # Verify update lists
        assert mock_comp.gas.updateorder == ["Fi", "S"]
        assert mock_comp.gas.S.updateorder == ["ext", "hyd", "tot"]
    
    def test_dust_active_updaters_set(self, mock_sim):
        """Test that dust active component updaters are properly set"""
        gas_value = np.ones(3) * 100.0
        dust_value = np.ones((3, 2)) * 10.0
        mu = 18.0

        mock_sim.addcomponent('ice', gas_value, mu,dust_value=dust_value, dust_active=True)
        mock_comp = mock_sim.components.__dict__['ice']
        
        # Verify update lists
        assert mock_comp.dust.updateorder == ["Fi", "S"]
        assert mock_comp.dust.S.updateorder == ["ext", "hyd", "coag", "tot"]
    
    def test_components_updater_list(self, mock_sim):
        """Test that component is added to components updater list"""
        gas_value = np.ones(3) * 100.0
        mu = 18.0
        
        initial_updater = mock_sim.components.updater
        addcomponent(mock_sim, 'water', gas_value, mu, gas_tracer=True)
        assert 'water' in mock_sim.components.updateorder
        
        # Test second component (updater exists)
        addcomponent(mock_sim, 'co', gas_value, mu, gas_tracer=True)
        assert 'co' in mock_sim.components.updateorder
        assert 'water' in mock_sim.components.updateorder

class TestAddComponentJacobianAndInstructions:
    @pytest.fixture
    def mock_sim(self):
        """Create a real simulation for Jacobian and instruction testing"""
        sim = Simulation()
        sim.ini.grid.Nr = 3
        sim.initialize()
        return sim
    
    def test_jacobian_assignment(self, mock_sim):
        """Test that Jacobian is properly assigned to component"""
        gas_value = np.ones(3) * 100.0
        mu = 18.0
        
        mock_sim.addcomponent('tracer', gas_value, mu, gas_tracer=True)
        mock_comp = mock_sim.components.__dict__['tracer']
        # Verify jacobinator was assigned
        assert mock_comp._Y.jacobinator is not None
        assert callable(mock_comp._Y.jacobian)

    def test_instruction_creation_active_component(self, mock_sim):
        """Test instruction creation for active components"""
        gas_value = np.ones(3) * 100.0
        mu = 18.0

        integrator_len = len(mock_sim.integrator.instructions)        
        mock_sim.addcomponent('water', gas_value, mu, gas_active=True)
        

        # Check that instructions were added (real simulation may have existing instructions)
        assert len(mock_sim.integrator.instructions) == 2 + integrator_len
