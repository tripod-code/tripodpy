import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from tripod import Simulation
import dustpy.constants as c
import types

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

    def test_verbosity(self):
        """Test verbosity setting"""
        sim = Simulation()
        sim.ini.grid.Nr = 5
        sim.verbosity = 2
        sim.initialize()
        sim.writer = None 
        sim.t.snapshots = [1]
        sim.run()
        assert sim.verbosity == 2

class TestSimulationGrids:
    def test_makegrids_basic(self):
        """Test basic grid creation"""
        sim = Simulation()
        sim.ini.grid.rmin = 1.0 * c.au
        sim.ini.grid.rmax = 10.0 * c.au
        sim.ini.grid.Nr = 10
        
        sim.makegrids()
        
        # Check grid properties
        assert sim.grid.Nr == 10
        assert len(sim.grid.r) == 10
        assert len(sim.grid.ri) == 11
        assert len(sim.grid.A) == 10
        assert sim.grid._Nm_short == 2
        assert sim.grid._Nm_long == 5
        
        # Check grid bounds
        assert sim.grid.ri[0] == pytest.approx(1.0 * c.au, rel=1e-10)
        assert sim.grid.ri[-1] == pytest.approx(10.0 * c.au, rel=1e-10)
    
    def test_makegrids_custom_ri(self):
        """Test grid creation with custom radial interfaces"""
        sim = Simulation()
        custom_ri = np.logspace(0, 2, 21)  # 20 cells
        sim.grid.ri = custom_ri
        
        sim.makegrids()
        
        assert sim.grid.Nr == 20
        np.testing.assert_array_equal(sim.grid.ri, custom_ri)
        assert len(sim.grid.r) == 20
        assert len(sim.grid.A) == 20
    
    def test_makeradialgrid_area_calculation(self):
        """Test area calculation in radial grid"""
        sim = Simulation()
        sim.ini.grid.rmin = 1.0
        sim.ini.grid.rmax = 4.0
        sim.ini.grid.Nr = 3
        
        sim.makegrids()
        
        # Check that areas are calculated correctly
        expected_areas = np.pi * (sim.grid.ri[1:]**2 - sim.grid.ri[:-1]**2)
        np.testing.assert_array_almost_equal(sim.grid.A, expected_areas)

class TestSimulationInitialization:
    def test_initialize_integration_variable(self):
        """Test integration variable initialization"""
        sim = Simulation()
        sim.ini.grid.Nr = 5
        sim.initialize()
        
        assert sim.t is not None
        assert sim.t == 0.0
        assert sim.t.cfl == 0.1
        assert hasattr(sim.t, 'snapshots')
    
    def test_initialize_components_setup(self):
        """Test components group initialization"""
        sim = Simulation()
        sim.ini.grid.Nr = 5
        sim.initialize()
        
        assert hasattr(sim, 'components')
        assert hasattr(sim.components, '_gas_updated')
        assert hasattr(sim.components, '_dust_updated')
        assert hasattr(sim, '_dust_compo')
        assert sim._dust_compo == False  # Initially false
        
        # Check that Default component was added
        assert hasattr(sim.components, 'Default')
        assert sim.components.Default.gas._active == True
    
    def test_initialize_integrator_setup(self):
        """Test integrator initialization"""
        sim = Simulation()
        sim.ini.grid.Nr = 5
        sim.initialize()
        
        assert sim.integrator is not None
        assert len(sim.integrator.instructions) >= 2
        assert sim.integrator.preparator is not None
        assert sim.integrator.finalizer is not None


class TestSimulationDustInitialization:
    @pytest.fixture
    def initialized_sim(self):
        sim = Simulation()
        sim.ini.grid.Nr = 5
        sim.initialize()
        return sim
    
    def test_dust_fields_creation(self, initialized_sim):
        """Test that all dust fields are properly created"""
        sim = initialized_sim
        
        # Check particle properties
        assert sim.dust.a is not None
        assert sim.dust.m is not None
        assert sim.dust.H is not None
        assert sim.dust.rho is not None
        assert sim.dust.rhos is not None
        
        # Check velocities
        assert sim.dust.v.frag is not None
        assert sim.dust.v.driftmax is not None
        assert sim.dust.v.rad is not None
        assert sim.dust.v.rad_flux is not None
        
        # Check relative velocities
        assert sim.dust.v.rel.azi is not None
        assert sim.dust.v.rel.brown is not None
        assert sim.dust.v.rel.rad is not None
        assert sim.dust.v.rel.turb is not None
        assert sim.dust.v.rel.vert is not None
        assert sim.dust.v.rel.tot is not None
    
    def test_dust_source_terms(self, initialized_sim):
        """Test dust source term initialization"""
        sim = initialized_sim
        
        assert sim.dust.S.ext is not None
        assert sim.dust.S.hyd is not None
        assert sim.dust.S.coag is not None
        assert sim.dust.S.compo is not None
        assert sim.dust.S.tot is not None
        assert sim.dust.S.smax_hyd is not None
        
        # Check that compo is first in update order
        assert sim.dust.S.updateorder[0] == 'compo'
    
    def test_dust_size_distribution(self, initialized_sim):
        """Test dust size distribution initialization"""
        sim = initialized_sim
        
        assert sim.dust.s.min is not None
        assert sim.dust.s.max is not None
        assert sim.dust.s.lim is not None
        assert sim.dust.s._maxOld is not None
        
        # Check boundary conditions
        assert sim.dust.s.boundary.inner is not None
        assert sim.dust.s.boundary.outer is not None
    
    def test_dust_distribution_exponents(self, initialized_sim):
        """Test distribution exponent initialization"""
        sim = initialized_sim
        
        assert sim.dust.q.eff is not None
        assert sim.dust.q.frag is not None
        assert sim.dust.qrec is not None
        assert sim.dust.q.turb1 is not None
        assert sim.dust.q.turb2 is not None
        assert sim.dust.q.drfrag is not None
        assert sim.dust.q.sweep is not None
    
    def test_dust_probabilities(self, initialized_sim):
        """Test dust probability initialization"""
        sim = initialized_sim
        
        assert sim.dust.p.frag is not None
        assert sim.dust.p.stick is not None
        assert sim.dust.p.fragtrans is not None
        assert sim.dust.p.driftfrag is not None
    
    def test_dust_fudge_factors(self, initialized_sim):
        """Test fudge factor initialization"""
        sim = initialized_sim
        
        assert sim.dust.f.crit is not None
        assert sim.dust.f.drift is not None
        assert sim.dust.f.dv is not None
        
        assert sim.dust.f.crit == 0.425
        assert sim.dust.f.drift == 0.8
        assert sim.dust.f.dv == 0.4
    
    def test_dust_fluxes(self, initialized_sim):
        """Test dust flux initialization"""
        sim = initialized_sim
        
        assert sim.dust.Fi.adv is not None
        assert sim.dust.Fi.diff is not None
        assert sim.dust.Fi.tot is not None
        
        # Check shapes
        expected_shape = (sim.grid.Nr + 1, sim.grid._Nm_short)
        assert sim.dust.Fi.adv.shape == expected_shape
        assert sim.dust.Fi.diff.shape == expected_shape
        assert sim.dust.Fi.tot.shape == expected_shape
    
    def test_dust_hidden_fields(self, initialized_sim):
        """Test dust hidden field initialization"""
        sim = initialized_sim
        
        assert sim.dust._SigmaOld is not None
        assert sim.dust._rhs is not None
        assert sim.dust._Y is not None
        assert sim.dust._Y_rhs is not None
        assert sim.dust.s.sdot_coag is not None
        assert sim.dust.s.sdot_shrink is not None
        assert sim.dust.s._damp is not None
    
    def test_dust_boundary_conditions(self, initialized_sim):
        """Test dust boundary condition initialization"""
        sim = initialized_sim
        
        assert sim.dust.boundary.inner is not None
        assert sim.dust.boundary.outer is not None
        
        # Check boundary condition types
        assert sim.dust.boundary.inner.condition == "const_grad"
        assert sim.dust.boundary.outer.condition == "val"

class TestSimulationShapes:
    @pytest.fixture
    def sim_shapes(self):
        sim = Simulation()
        sim.ini.grid.Nr = 10
        sim.initialize()
        return sim
    
    def test_field_shapes_consistency(self, sim_shapes):
        """Test that field shapes are consistent"""
        sim = sim_shapes
        Nr = sim.grid.Nr
        Nm_short = sim.grid._Nm_short
        Nm_long = sim.grid._Nm_long
        
        # Radial-only fields
        assert sim.dust.eps.shape == (Nr,)
        assert sim.dust.s.max.shape == (Nr,)
        assert sim.dust.s.min.shape == (Nr,)
        
        # Radial + short mass grid
        assert sim.dust.Sigma.shape == (Nr, Nm_short)
        assert sim.dust.S.ext.shape == (Nr, Nm_short)
        assert sim.dust.S.hyd.shape == (Nr, Nm_short)
        assert sim.dust.S.coag.shape == (Nr, Nm_short)
        assert sim.dust.S.tot.shape == (Nr, Nm_short)
        
        # Radial + long mass grid
        assert sim.dust.a.shape == (Nr, Nm_long)
        assert sim.dust.m.shape == (Nr, Nm_long)
        assert sim.dust.H.shape == (Nr, Nm_long)
        assert sim.dust.rho.shape == (Nr, Nm_long)
        assert sim.dust.v.rad.shape == (Nr, Nm_long)
        
        # Radial + interfaces + short mass grid
        assert sim.dust.Fi.adv.shape == (Nr + 1, Nm_short)
        assert sim.dust.Fi.diff.shape == (Nr + 1, Nm_short)
        assert sim.dust.Fi.tot.shape == (Nr + 1, Nm_short)
        
        # Relative velocity tensors
        assert sim.dust.v.rel.tot.shape == (Nr, Nm_long, Nm_long)

class TestSimulationMethods:
    def test_run_method_exists(self):
        """Test that run method exists and has proper structure"""
        sim = Simulation()
        assert hasattr(sim, 'run')
        assert callable(sim.run)
    
    def test_addcomponent_c_method_binding(self):
        """Test that addcomponent_c method is properly bound"""
        sim = Simulation()
        sim.ini.grid.Nr = 5
        sim.initialize()
        
        assert hasattr(sim, 'addcomponent_c')
        assert callable(sim.addcomponent_c)
        assert isinstance(sim.addcomponent_c, types.MethodType)
    
    @patch('dustpy.std.gas.enforce_floor_value')
    def test_initialization_floor_enforcement(self, mock_enforce_floor):
        """Test that floor values are enforced during initialization"""
        sim = Simulation()
        sim.ini.grid.Nr = 5
        sim.initialize()
        
        mock_enforce_floor.assert_called_once_with(sim)
class TestSimulationFieldDeletion:
    def test_unused_fields_deleted(self):
        """Test that unused DustPy fields are properly deleted"""
        sim = Simulation()
        
        # These fields should not exist
        assert not hasattr(sim.ini.grid, 'Nmbpd')
        assert not hasattr(sim.ini.grid, 'mmax')
        assert not hasattr(sim.ini.dust, 'erosionMassRatio')
        assert not hasattr(sim.ini.dust, 'excavatedMass')
        assert not hasattr(sim.ini.dust, 'fragmentDistribution')
        
        # Grid fields that should be deleted
        assert not hasattr(sim.grid, 'm')
        assert not hasattr(sim.grid, 'Nm')
    
    def test_dust_fields_deleted(self):
        """Test that specific dust fields are deleted"""
        sim = Simulation()
        
        # These dust fields should not exist
        assert not hasattr(sim.dust, 'coagulation')
        assert not hasattr(sim.dust, 'kernel')

    def test_eclude_form_dustpy(self):
        """Test that exclude_from_dustpy works correctly"""
        sim = Simulation()
        sim.ini.grid.Nr = 5
        sim.initialize()

        for attr in sim._excludefromdustpy:
            with pytest.raises(AttributeError):
                getattr(sim, attr)
        


class TestSimulationUpdateOrders:
    def test_dust_update_order_modification(self):
        """Test that dust update order is properly modified"""
        sim = Simulation()
        sim.ini.grid.Nr = 5
        sim.initialize()
        
        updater = sim.dust.updateorder
        
        # Check that specific elements are in correct positions
        assert 'f' in updater
        assert 'qrec' in updater
        assert 'q' in updater
        
        # Check relative positions
        f_idx = updater.index('f')
        qrec_idx = updater.index('qrec')
        a_idx = updater.index('a')
        m_idx = updater.index('m')
        
        assert qrec_idx > f_idx
        assert a_idx > qrec_idx
        assert m_idx > a_idx
        
        # Check that removed elements are not present
        assert 'kernel' not in updater

class TestSimulationIntegration:
    def test_state_vector_setup(self):
        """Test state vector setup"""
        sim = Simulation()
        sim.ini.grid.Nr = 5
        sim.initialize()
        
        # Check state vector size
        expected_size = (sim.grid._Nm_short + 1) * sim.grid.Nr
        assert len(sim.dust._Y) == expected_size
        assert len(sim.dust._Y_rhs) == expected_size
        
        # Check that jacobinator is set
        assert sim.dust._Y.jacobinator is not None
        assert sim.dust.Sigma.jacobinator is not None
    
    def test_differentiator_setup(self):
        """Test that differentiators are properly set"""
        sim = Simulation()
        sim.ini.grid.Nr = 5
        sim.initialize()
        
        assert sim.dust.s.max.differentiator is not None

class TestSimulationComponentIntegration:
    def test_multiple_component_addition(self):
        """Test adding multiple components"""
        sim = Simulation()
        sim.ini.grid.Nr = 10
        sim.initialize()
        
        # Add multiple components
        gas_tracer = np.ones(10) * 0.01
        dust_tracer = np.ones((10, 2)) * 0.001
        
        sim.addcomponent_c("water_vapor", gas_tracer, 18.0, gas_tracer=True)
        sim.addcomponent_c("co_vapor", gas_tracer, 28.0, gas_tracer=True)
        sim.addcomponent_c("water_ice", gas_tracer, 18.0, 
                          dust_value=dust_tracer, dust_tracer=True)
        
        # Check all components exist
        assert hasattr(sim.components, "water_vapor")
        assert hasattr(sim.components, "co_vapor")
        assert hasattr(sim.components, "water_ice")
        
        # Check component types
        assert sim.components.water_vapor.gas._tracer == True
        assert sim.components.co_vapor.gas._tracer == True
        assert sim.components.water_ice.dust._tracer == True
    
    def test_component_updater_integration(self):
        """Test that components are properly integrated into update system"""
        sim = Simulation()
        sim.ini.grid.Nr = 5
        sim.initialize()
        
        # Check that components updater is set
        assert sim.components.updater is not None
        assert len(sim.components.updateorder) >= 1  # At least Default component
        
        # Check that Default component is in updater
        assert "Default" in sim.components.updateorder

