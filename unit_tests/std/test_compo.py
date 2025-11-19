from unittest import mock
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import scipy.sparse as sp
import dustpy.constants as c
from tripodpy import Simulation

from tripodpy.std.compo import (
    prepare, set_state_vector_components, finalize, Y_jacobian,
    _f_impl_1_direct_compo, jacobian_compo, A_grains, L_condensation,
    L_sublimation, c_jacobian, set_boundaries_component
)

class TestStateVectorManagement:

    @pytest.fixture
    def sim(self):
        sim = Simulation()
        sim.ini.grid.Nr = 3
        sim.initialize()
        return sim
    
    @pytest.fixture
    def Sigma_gas(self):
        return np.array([100.0, 200.0, 300.0])
    
    @pytest.fixture
    def Sigma_dust(self):
        return np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    

    def test_set_state_vector_components_gas_active(self,sim,Sigma_gas):
        """Test state vector setup for gas active component"""
        sim.addcomponent(name="water", gas_value=Sigma_gas, mu=18.0, gas_active=True)
        comp_gas = sim.components.__dict__["water"]
        set_state_vector_components(sim)
        
        # Verify state vector was set correctly
        np.testing.assert_array_equal(comp_gas._Y, comp_gas.gas.Sigma)
        np.testing.assert_array_equal(comp_gas._S, comp_gas.gas.Sigma_dot)
    
    def test_set_state_vector_components_dust_tracer(self,sim,Sigma_dust):
        """Test state vector setup for dust tracer component"""

        sim.addcomponent(name="silicate_tracer", gas_value=0.0, mu=0.0, dust_value=Sigma_dust, dust_tracer=True)
        comp_dust_tracer = sim.components.__dict__["silicate_tracer"]
        set_state_vector_components(sim)
        
        # Verify state vector calculation
        expected_Y = (comp_dust_tracer.dust.value * sim.dust.Sigma).ravel()
        np.testing.assert_array_equal(comp_dust_tracer._Y, expected_Y)
    
    def test_set_state_vector_components_dust_active(self,sim,Sigma_dust):
        """Test state vector setup for active dust component"""
        
        sim.addcomponent(name="carbon_dust", gas_value=0.0, mu=0.0, dust_value=Sigma_dust, dust_active=True)
        comp_dust = sim.components.__dict__["carbon_dust"]
        set_state_vector_components(sim)
        
        # Verify state vector was set correctly
        expected_Y = comp_dust.dust.Sigma.ravel()
        np.testing.assert_array_equal(comp_dust._Y, expected_Y)
        np.testing.assert_array_equal(comp_dust._S, comp_dust.dust.S.ext.ravel())
    
    def test_set_state_vector_components_mixed_dust_gas(self,sim,Sigma_gas,Sigma_dust):
        """Test state vector setup for mixed dust and gas component"""

        sim.addcomponent(name="mixed_compo", gas_value=Sigma_gas, mu=18.0, dust_value=Sigma_dust, dust_active=True, gas_active=True)
        comp_mixed = sim.components.__dict__["mixed_compo"]

        set_state_vector_components(sim)
        
        Nr = sim.grid.Nr
        # Verify gas part
        np.testing.assert_array_equal(comp_mixed._Y[:Nr], comp_mixed.gas.Sigma)
        np.testing.assert_array_equal(comp_mixed._S[:Nr], comp_mixed.gas.Sigma_dot)
        
        # Verify dust part
        expected_dust_Y = (comp_mixed.dust.Sigma).ravel()
        np.testing.assert_array_equal(comp_mixed._Y[Nr:], expected_dust_Y)

    def test_tracer_active_component(self,sim,Sigma_gas,Sigma_dust):
        """Test error raised when component is both tracer and active"""
        sim.addcomponent(name="comp_mixed", gas_value=Sigma_gas, mu=18.0, dust_value=Sigma_dust, dust_tracer=True, gas_active=True)
        comp_mixed = sim.components.__dict__["comp_mixed"]
        set_state_vector_components(sim)
        Nr = sim.grid.Nr

        np.testing.assert_array_equal(comp_mixed._Y[:Nr], comp_mixed.gas.Sigma)
        np.testing.assert_array_equal(comp_mixed._S[:Nr], comp_mixed.gas.Sigma_dot)

        np.testing.assert_array_equal(comp_mixed._Y[Nr:], (comp_mixed.dust.value * sim.dust.Sigma).ravel())
        np.testing.assert_array_equal(comp_mixed._S[Nr:],  comp_mixed.dust.value_dot.ravel()*sim.dust.Sigma.ravel() + comp_mixed.dust.value.ravel()*(sim.dust.S.ext.ravel() + sim.dust.S.compo.ravel()))
    
    def test_invalid_component(self,sim):
        """Test error raised for component with no active or tracer flags"""
        comp = Mock()
        comp.name = "invalid_compo"
        comp.gas._active = True
        comp.gas._tracer = True
        comp.dust._active = True
        comp.dust._tracer = True
        sim.components.__dict__["invalid_compo"] = comp
        with pytest.raises(RuntimeError, match="Component type not recognized"):
            set_state_vector_components(sim)

        comp.gas._active = False
        comp.gas._tracer = False
        comp.dust._active = False
        comp.dust._tracer = False
        with pytest.raises(RuntimeError, match="Component type not recognized"):
            set_state_vector_components(sim)
class TestFinalization:
    @pytest.fixture
    def sim(self):
        sim = Simulation()
        sim.ini.grid.Nr = 3
        sim.initialize()
        sim.t.snapshots = [1.0]
        sim.writer = None
        return sim


    def test_finalize_updates_tracer(self, sim):
        """Test finalization updates tracer components correctly"""
        sim.addcomponent(name="tracer_dust", gas_value=0.0, mu=0.0, dust_value=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), dust_tracer=True)
        sim.addcomponent(name="tracer_gas", gas_value=np.array([10.0, 20.0, 30.0]), mu=18.0, gas_tracer=True)
        sim.update()
        sim.run()

        # assert tracer values are updated correctly
        np.testing.assert_array_almost_equal(sim.components.tracer_dust.dust.value,(sim.components.tracer_dust._Y/sim.dust._Y[:sim.grid._Nm_short*sim.grid.Nr]).reshape(sim.components.tracer_dust.dust.value.shape))
        np.testing.assert_array_almost_equal(sim.components.tracer_gas.gas.value,sim.components.tracer_gas._Y/ sim.gas.Sigma)

    def test_finalize_updates_active(self, sim):
        """Test finalization updates active components correctly"""
        sim.addcomponent(name="active_dust", gas_value=0.0, mu=0.0, dust_value=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), dust_active=True)
        sim.addcomponent(name="active_gas", gas_value=np.array([10.0, 20.0, 30.0]), mu=18.0, gas_active=True)
        sim.update()
        sim.run()

        # assert active values are updated correctly
        np.testing.assert_array_almost_equal(sim.components.active_dust.dust.Sigma,sim.components.active_dust._Y.reshape(sim.components.active_dust.dust.Sigma.shape))
        np.testing.assert_array_almost_equal(sim.components.active_gas.gas.Sigma,sim.components.active_gas._Y)

        #assert flags are reset
        assert sim.components._gas_updated == False
        assert sim.components._dust_updated == False

    def test_finalize_tracer_mixed(self, sim):
        """Test finalization updates mixed components correctly"""
        sim.addcomponent(name="mixed_compo", gas_value=np.array([10.0, 20.0, 30.0]), mu=18.0, dust_value=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), dust_active=True, gas_active=True)
        sim.update()
        sim.run()

        Nr = sim.grid.Nr
        # assert gas part is updated correctly
        np.testing.assert_array_almost_equal(sim.components.mixed_compo.gas.Sigma,sim.components.mixed_compo._Y[:Nr])
        
        # assert dust part is updated correctly
        np.testing.assert_array_almost_equal(sim.components.mixed_compo.dust.Sigma,sim.components.mixed_compo._Y[Nr:].reshape(sim.components.mixed_compo.dust.Sigma.shape))

        #assert flags are reset
        assert sim.components._gas_updated == False
        assert sim.components._dust_updated == False

    def test_finalize_mixed(self, sim):
        """Test finalization updates mixed components correctly"""
        sim.addcomponent(name="mixed_compo", gas_value=np.array([10.0, 20.0, 30.0]), mu=18.0, dust_value=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), dust_tracer=True, gas_active=True)
        sim.update()
        sim.run()

        comp = sim.components.mixed_compo
        Nr = sim.grid.Nr
        # assert gas part is updated correctly
        np.testing.assert_array_almost_equal(sim.components.mixed_compo.gas.Sigma,sim.components.mixed_compo._Y[:Nr])
        
        # assert dust part is updated correctly
        np.testing.assert_array_almost_equal(sim.components.mixed_compo.dust.value, (comp._Y[sim.grid.Nr:]/sim.dust._Y[:sim.grid._Nm_short*sim.grid.Nr]).reshape(comp.dust.value.shape))

        #assert flags are reset
        assert sim.components._gas_updated == False
        assert sim.components._dust_updated == False

    def test_invalid_component(self,sim):
        """Test error raised for component with no active or tracer flags"""
        comp = Mock()
        comp.name = "invalid_compo"
        comp.gas._active = True
        comp.gas._tracer = True
        comp.dust._active = True
        comp.dust._tracer = True
        sim.components.__dict__["invalid_compo"] = comp
        with pytest.raises(RuntimeError, match="Component type not recognized"):
            finalize(sim)

        comp.gas._active = False
        comp.gas._tracer = False
        comp.dust._active = False
        comp.dust._tracer = False
        with pytest.raises(RuntimeError, match="Component type not recognized"):
            finalize(sim)
class TestJacobianCalculations:
    def test_Y_jacobian_basic(self, monkeypatch):
        """Test Y Jacobian calculation"""
        sim = Mock()
        sim.grid.r = np.array([1.0, 2.0, 3.0])
        sim.grid.ri = np.array([0.5, 1.5, 2.5, 3.5])
        sim.grid.A = np.ones(4)
        sim.grid.Nr = 3
        sim.grid._Nm_short = 2
        
        # Mock Jacobian matrices
        mock_gas_jac = sp.csc_matrix(np.eye(3))
        mock_dust_jac = sp.csc_matrix(np.eye(6))
        mock_compo_jac = sp.csc_matrix(np.eye(9))
        
        sim.dust.Sigma.jacobian = Mock(return_value=mock_dust_jac)
        
        monkeypatch.setattr('dustpy.std.gas.jacobian', lambda *args, **kwargs: mock_gas_jac)
        monkeypatch.setattr('tripodpy.std.compo.jacobian_compo', lambda *args, **kwargs: mock_compo_jac)
        
        x = Mock()
        x.stepsize = 0.1
        
        result = Y_jacobian(sim, x, component=Mock())
        
        assert isinstance(result, sp.csc_matrix)
        assert result.shape == (9, 9)  # 3 (gas) + 6 (dust)
    
    def test_Y_jacobian_with_nans(self, monkeypatch):
        """Test Y Jacobian with NaN values"""
        sim = Mock()
        sim.grid.Nr = 3
        sim.grid._Nm_short = 2
        
        # Create matrices with NaN values
        mock_gas_jac = sp.csc_matrix(np.array([[1, 0, 0], [0, np.nan, 0], [0, 0, 1]]))
        mock_dust_jac = sp.csc_matrix(np.eye(6))
        mock_compo_jac = sp.csc_matrix(np.eye(9))
        
        sim.dust.Sigma.jacobian = Mock(return_value=mock_dust_jac)
        
        monkeypatch.setattr('dustpy.std.gas.jacobian', lambda *args, **kwargs: mock_gas_jac)
        monkeypatch.setattr('tripodpy.std.compo.jacobian_compo', lambda *args, **kwargs: mock_compo_jac)
        
        x = Mock()
        x.stepsize = 0.1
        
        with pytest.raises(ValueError, match="Jacobian contains NaN values"):
            Y_jacobian(sim, x, component=Mock())
    
    def test_jacobian_compo(self, monkeypatch):
        """Test component Jacobian calculation"""
        sim = Mock()
        sim.grid.Nr = 3
        sim.grid._Nm_short = 2
        
        comp = Mock()
        
        # Mock sublimation and condensation functions
        monkeypatch.setattr('tripodpy.std.compo.L_sublimation', 
                           lambda sim, comp: np.ones((3, 2)) * 0.1)
        monkeypatch.setattr('tripodpy.std.compo.L_condensation', 
                           lambda sim, comp, **kwargs: np.ones((3, 2)) * 0.2)
        
        x = Mock()
        x.stepsize = 0.1
        
        result = jacobian_compo(sim, x, component=comp)
        
        assert isinstance(result, sp.csc_matrix)
        assert result.shape == (9, 9)  # (Nr * Nm_s) + Nr


    def test_impl_direct_basic(self):
        """Test implicit direct function"""
        Jac = sp.csc_matrix(np.eye(9))
        Y0 = np.ones(9)
        dx = 0.1
        x0 = 1 
        
        result = _f_impl_1_direct_compo(x0 = x0,Y0=Y0, dx=dx, jac=Jac)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (9,)
        expcted_result = np.ones(9)/9.
        np.testing.assert_array_almost_equal(result, expcted_result)

class TestPhysicalProcesses:
    def test_A_grains_q_minus_4(self):
        """Test grain surface area calculation with q = -4"""
        sim = Mock()
        sim.dust.Sigma = np.ones((3, 2)) * 10.0
        sim.dust.s.max = np.ones(3) * 1e-2
        sim.dust.s.min = np.ones(3) * 1e-4
        sim.dust.qrec = np.ones(3) * (-4.0)
        sim.dust.rhos = np.ones((3, 3)) * 1000.0  # Note: shape (3, 3) for indexing
        
        result = A_grains(sim)
        
        assert result.shape == (3, 2)
        assert np.all(result > 0)
    
    def test_A_grains_q_minus_3(self):
        """Test grain surface area calculation with q = -3"""
        sim = Mock()
        sim.dust.Sigma = np.ones((3, 2)) * 10.0
        sim.dust.s.max = np.ones(3) * 1e-2
        sim.dust.s.min = np.ones(3) * 1e-4
        sim.dust.qrec = np.ones(3) * (-3.0)
        sim.dust.rhos = np.ones((3, 3)) * 1000.0
        
        result = A_grains(sim)
        
        assert result.shape == (3, 2)
        assert np.all(result > 0)
    
    def test_A_grains_general_q(self):
        """Test grain surface area calculation with general q values"""
        sim = Mock()
        sim.dust.Sigma = np.ones((3, 2)) * 10.0
        sim.dust.s.max = np.ones(3) * 1e-2
        sim.dust.s.min = np.ones(3) * 1e-4
        sim.dust.qrec = np.ones(3) * (-3.5)
        sim.dust.rhos = np.ones((3, 3)) * 1000.0
        
        result = A_grains(sim)
        
        assert result.shape == (3, 2)
        assert np.all(result > 0)
    
    def test_L_condensation(self, monkeypatch):
        """Test condensation rate calculation"""
        sim = Mock()
        sim.gas.Hp = np.ones(3) * 0.1
        sim.gas.T = np.ones(3) * 100.0
        
        comp = Mock()
        comp.gas.pars.mu = 18.0  # Water molecular weight
        
        monkeypatch.setattr('tripodpy.std.compo.A_grains', 
                           lambda sim: np.ones((3, 2)) * 1e-4)
        
        result = L_condensation(sim, comp, Pstick=1.0)
        
        assert result.shape == (3, 2)
        assert np.all(result > 0)
    
    def test_L_condensation_with_pstick(self, monkeypatch):
        """Test condensation with different sticking probability"""
        sim = Mock()
        sim.gas.Hp = np.ones(3) * 0.1
        sim.gas.T = np.ones(3) * 100.0
        
        comp = Mock()
        comp.gas.pars.mu = 18.0
        
        monkeypatch.setattr('tripodpy.std.compo.A_grains', 
                           lambda sim: np.ones((3, 2)) * 1e-4)
        
        result_full = L_condensation(sim, comp, Pstick=1.0)
        result_half = L_condensation(sim, comp, Pstick=0.5)
        
        np.testing.assert_array_almost_equal(result_half, result_full * 0.5)
    
    def test_L_sublimation_active(self, monkeypatch):
        """Test sublimation rate for active dust component"""
        sim = Mock()
        sim.gas.T = np.ones(3) * 150.0
        
        comp = Mock()
        comp.dust._tracer = False
        comp.dust._active = True
        comp.dust.Sigma = np.ones((3, 2)) * 5.0
        comp.gas.pars.mu = 18.0
        comp.gas.pars.nu = 1e13
        comp.gas.pars.Tsub = 100.0
        
        monkeypatch.setattr('tripodpy.std.compo.A_grains', 
                           lambda sim: np.ones((3, 2)) * 1e-4)
        
        result = L_sublimation(sim, comp, N_bind=1e15)
        
        assert result.shape == (3, 2)
        assert np.all(result >= 0)
    
    def test_L_sublimation_invalid_component(self, monkeypatch):
        """Test sublimation with invalid component type"""
        sim = Mock()
        
        comp = Mock()
        comp.dust._tracer = False
        comp.dust._active = False  # Invalid combination
        
        monkeypatch.setattr('tripodpy.std.compo.A_grains', 
                           lambda sim: np.ones((3, 2)) * 1e-4)
        
        with pytest.raises(RuntimeError, match="Component dust type not recognized"):
            L_sublimation(sim, comp)
    
    def test_L_sublimation_thin_layer(self, monkeypatch):
        """Test sublimation with thin ice layer (N_layer < 1e-2)"""
        sim = Mock()
        sim.gas.T = np.ones(3) * 150.0
        sim.dust.Sigma = np.ones((3, 2)) * 10.0
        
        comp = Mock()
        comp.dust._tracer = True
        comp.dust._active = False
        comp.dust.value = np.ones((3, 2)) * 1e-6  # Very small value for thin layer
        comp.gas.pars.mu = 18.0
        comp.gas.pars.nu = 1e13
        comp.gas.pars.Tsub = 100.0
        
        monkeypatch.setattr('tripodpy.std.compo.A_grains', 
                           lambda sim: np.ones((3, 2)) * 1e-4)
        
        result = L_sublimation(sim, comp, N_bind=1e15)
        
        assert result.shape == (3, 2)
        assert np.all(result >= 0)

class TestComponentJacobians:
    def test_c_jacobian_gas_active(self, monkeypatch):
        """Test component Jacobian for gas active component"""
        sim = Mock()
        comp = Mock()
        comp.dust._active = False
        comp.dust._tracer = False
        comp.gas._active = True
        comp.gas._tracer = False
        
        mock_jac = sp.csc_matrix(np.eye(3))
        
        monkeypatch.setattr('dustpy.std.gas.jacobian', lambda *args, **kwargs: mock_jac)
        monkeypatch.setattr('tripodpy.std.compo.set_boundaries_component', 
                           lambda sim, J, dt, comp: J)
        
        x = Mock()
        x.stepsize = 0.1
        
        result = c_jacobian(sim, x, component=comp)
        
        assert isinstance(result, sp.csc_matrix)
        assert result.shape == (3, 3)
    
    def test_c_jacobian_gas_tracer(self, monkeypatch):
        """Test component Jacobian for gas tracer component"""
        sim = Mock()
        comp = Mock()
        comp.dust._active = False
        comp.dust._tracer = False
        comp.gas._active = False
        comp.gas._tracer = True
        
        mock_jac = sp.csc_matrix(np.eye(3))
        
        monkeypatch.setattr('dustpy.std.gas.jacobian', lambda *args, **kwargs: mock_jac)
        monkeypatch.setattr('tripodpy.std.compo.set_boundaries_component', 
                           lambda sim, J, dt, comp: J)
        
        x = Mock()
        x.stepsize = 0.1
        
        result = c_jacobian(sim, x, component=comp)
        
        assert isinstance(result, sp.csc_matrix)
    
    def test_c_jacobian_dust_tracer(self, monkeypatch):
        """Test component Jacobian for dust tracer component"""
        sim = Mock()
        comp = Mock()
        comp.dust._active = False
        comp.dust._tracer = True
        comp.gas._active = False
        comp.gas._tracer = False
        
        mock_jac = sp.csc_matrix(np.eye(6))
        
        monkeypatch.setattr('tripodpy.std.dust.jacobian', lambda *args, **kwargs: mock_jac)
        monkeypatch.setattr('tripodpy.std.compo.set_boundaries_component', 
                           lambda sim, J, dt, comp: J)
        
        x = Mock()
        x.stepsize = 0.1
        
        result = c_jacobian(sim, x, component=comp)
        
        assert isinstance(result, sp.csc_matrix)
    
    def test_c_jacobian_mixed_component(self, monkeypatch):
        """Test component Jacobian for mixed dust and gas component"""
        sim = Mock()
        comp = Mock()
        comp.dust._tracer = True
        comp.gas._active = True
        
        mock_jac = sp.csc_matrix(np.eye(9))
        
        monkeypatch.setattr('tripodpy.std.compo.Y_jacobian', lambda *args, **kwargs: mock_jac)
        monkeypatch.setattr('tripodpy.std.compo.set_boundaries_component', 
                           lambda sim, J, dt, comp: J)
        
        x = Mock()
        x.stepsize = 0.1
        
        result = c_jacobian(sim, x, component=comp)
        
        assert isinstance(result, sp.csc_matrix)
        assert result.shape == (9, 9)
    
    def test_c_jacobian_invalid_component(self):
        """Test component Jacobian with invalid component type"""
        sim = Mock()
        comp = Mock()
        comp.dust._active = False
        comp.dust._tracer = False
        comp.gas._active = False
        comp.gas._tracer = False  # Invalid combination
        
        x = Mock()
        x.stepsize = 0.1
        
        with pytest.raises(RuntimeError, match="Component type not recognized"):
            c_jacobian(sim, x, component=comp)

class TestBoundaryConditions:
    @pytest.fixture
    def mock_boundary_sim(self):
        """Create a mock simulation for boundary testing"""
        sim = Mock()
        sim.grid.Nr = 3
        sim.grid._Nm_short = 2
        sim.gas.Sigma = np.ones(3) * 100.0
        sim.dust.Sigma = np.ones((3, 2)) * 10.0
        sim.dust._Y = np.ones(6) * 5.0
        sim.components._dust_updated = False
        sim.components._gas_updated = False
        sim._dust_compo = True
        return sim
    
    def test_set_boundaries_gas_active_val_condition(self, mock_boundary_sim):
        """Test boundary conditions for gas active component with val condition"""
        sim = mock_boundary_sim
        J = sp.csc_matrix(np.eye(3))
        dt = 0.1
        
        # Mock gas active component
        comp = Mock()
        comp.dust._tracer = False
        comp.dust._active = False
        comp.gas._active = True
        comp.gas._tracer = False
        comp._Y_rhs = np.zeros(3)
        comp._S = np.zeros(3)
        
        # Mock boundary conditions
        comp.gas.boundary.inner = Mock()
        comp.gas.boundary.inner.condition = "val"
        comp.gas.boundary.inner.value = 150.0
        
        comp.gas.boundary.outer = Mock()
        comp.gas.boundary.outer.condition = "val"
        comp.gas.boundary.outer.value = 50.0
        
        result = set_boundaries_component(sim, J, dt, comp)
        
        assert comp._Y_rhs[0] == 150.0
        assert comp._Y_rhs[2] == 50.0  # Nr-1


class TestBoundaryComponents:
    @pytest.fixture
    def sim(self):
        sim = Simulation()
        sim.ini.grid.Nr = 5
        sim.initialize()

        return sim
    
    def test_set_boundaries_component_gas_active(self, sim):
        """Test boundary conditions for gas active component"""
        sim.addcomponent(name="water", gas_value=np.array([100.0, 200.0, 300.0,400.0,500.0]), mu=18.0, gas_active=True)
        comp = sim.components.__dict__["water"]
        J = sp.csc_matrix(np.eye(5))
        dt = 0.1

        #val
        comp.gas.boundary.inner.setcondition("val", value=150.0)
        comp.gas.boundary.outer.setcondition("val", value=50.0)

        result = set_boundaries_component(sim, J, dt, comp)
        assert comp._Y_rhs[0] == 150.0
        assert comp._Y_rhs[-1] == 50.0  # Nr-1

        #const_val
        comp._Y_rhs = comp.gas.Sigma
        comp.gas.boundary.inner.setcondition("const_val")
        comp.gas.boundary.outer.setcondition("const_val")

        result = set_boundaries_component(sim, J, dt, comp)
        assert comp._Y_rhs[0] == 0.0
        assert comp._Y_rhs[-1] == 0.0  # Nr-1
        assert result[0,1] == 10. # 1./dt

        #grad
        comp._Y_rhs = comp.gas.Sigma
        comp.gas.boundary.inner.setcondition("grad", value=1.0)
        comp.gas.boundary.outer.setcondition("grad", value=-1.0)

        result = set_boundaries_component(sim, J, dt, comp)
        K1 = -sim.grid.r[1]/sim.grid.r[0]
        Km1 = -sim.grid.r[-2]/sim.grid.r[-1]
        np.testing.assert_almost_equal(result[0,1],  (-K1/dt))
        np.testing.assert_almost_equal(result[-1,-2], (-Km1/dt))

        #const_grad
        comp.gas.boundary.inner.setcondition("const_grad")
        comp.gas.boundary.outer.setcondition("const_grad")

        result = set_boundaries_component(sim, J, dt, comp)

        assert result[0,0] == 0
        assert result[-1,-1] == 0
        assert comp._Y_rhs[0] == 0.0
        assert comp._Y_rhs[-1] == 0.0  # Nr-1
        assert result[0,1] != 0
        assert result[0,2] != 0
        assert result[-1,-2] != 0
        assert result[-1,-3] != 0


        #pow 
        comp._Y_rhs = comp.gas.Sigma
        p_inner = 2.
        p_outer = 0.5
        comp.gas.boundary.inner.setcondition("pow", value=p_inner)
        comp.gas.boundary.outer.setcondition("pow", value=p_outer)

        result = set_boundaries_component(sim, J, dt, comp)

        expected_inner = (sim.grid.r[0]/sim.grid.r[1])**p_inner
        expected_outer = (sim.grid.r[-1]/sim.grid.r[-2])**p_outer
        assert comp._Y_rhs[0] == expected_inner*comp._Y_rhs[1]
        assert comp._Y_rhs[-1] == expected_outer*comp._Y_rhs[-2]

        #const_pow
        comp._Y_rhs = comp.gas.Sigma
        comp.gas.boundary.inner.setcondition("const_pow")
        comp.gas.boundary.outer.setcondition("const_pow")

        result = set_boundaries_component(sim, J, dt, comp)
        assert result[0,0] == 0
        assert result[-1,-1] == 0
        assert comp._Y_rhs[0] == 0.0
        assert comp._Y_rhs[-1] == 0.0  # Nr-1
        assert result[0,1] != 0
        assert result[-1,-2] != 0


        


    def test_set_boundaries_component_dust_active(self, sim):
        """Test boundary conditions for dust active component"""
        sim.addcomponent("water",None,None, dust_value=np.array([[10., 20.],[30., 40.],[50. ,60.],[70.,80.],[90., 100.]]), dust_active=True)
        comp = sim.components.__dict__["water"]
        Nm = sim.grid._Nm_short
        Nr = sim.grid.Nr
        J = sp.csc_matrix(np.eye(Nm*Nr))
        dt = 0.1


        #val
        comp.dust.boundary.inner.setcondition("val", value=150.0)
        comp.dust.boundary.outer.setcondition("val", value=50.0)

        result = set_boundaries_component(sim, J, dt, comp)
        assert comp._Y_rhs[0] == 150.0
        assert comp._Y_rhs[-1] == 50.0  # Nr-1

        #const_val
        comp._Y_rhs = comp.dust.Sigma.ravel()
        comp.dust.boundary.inner.setcondition("const_val")
        comp.dust.boundary.outer.setcondition("const_val")
        result = set_boundaries_component(sim, J, dt, comp)
        assert all(comp._Y_rhs[:Nm] == 0.0)
        assert all(comp._Y_rhs[-Nm:] == 0.0)  # Nr-1
        for k in range(Nm):
            assert result[k,k+Nm] == 10. # 1./dt
            assert result[-Nm+k,-2*Nm+k] == 10. # 1./dt


        #grad
        comp._Y_rhs = comp.dust.Sigma.ravel()
        comp.dust.boundary.inner.setcondition("grad", value=1.0)
        comp.dust.boundary.outer.setcondition("grad", value=-1.0)
        result = set_boundaries_component(sim, J, dt, comp)
        K1 = -sim.grid.r[1]/sim.grid.r[0]
        Km1 = -sim.grid.r[-2]/sim.grid.r[-1]
        for k in range(Nm):
            np.testing.assert_almost_equal(result[k,k+Nm],  (-K1/dt))
            np.testing.assert_almost_equal(result[-Nm+k,-2*Nm+k], (-Km1/dt))
        
        #const_grad
        comp._Y_rhs = comp.dust.Sigma.ravel()
        comp.dust.boundary.inner.setcondition("const_grad")
        comp.dust.boundary.outer.setcondition("const_grad")
        result = set_boundaries_component(sim, J, dt, comp)
        for k in range(Nm):
            assert result[k,k] == 0
            assert result[-Nm+k,-Nm+k] == 0
            assert comp._Y_rhs[k] == 0.0
            assert comp._Y_rhs[-Nm+k] == 0.0  # Nr-1
            assert result[k,k+Nm] != 0
            assert result[k,k+2*Nm] != 0
            assert result[-Nm+k,-2*Nm+k] != 0
            assert result[-Nm+k,-3*Nm+k] != 0


        #pow
        comp._Y_rhs = comp.dust.Sigma.ravel()
        p_inner = 2.
        p_outer = 0.5
        comp.dust.boundary.inner.setcondition("pow", value=p_inner)
        comp.dust.boundary.outer.setcondition("pow", value=p_outer)
        result = set_boundaries_component(sim, J, dt, comp)
        for k in range(Nm):
            expected_inner = (sim.grid.r[0]/sim.grid.r[1])**p_inner
            expected_outer = (sim.grid.r[-1]/sim.grid.r[-2])**p_outer
            assert comp._Y_rhs[k] == expected_inner*comp._Y_rhs[k+Nm]
            assert comp._Y_rhs[-Nm+k] == expected_outer*comp._Y_rhs[-2*Nm+k]


        #const_pow
        comp._Y_rhs = comp.dust.Sigma.ravel()
        comp.dust.boundary.inner.setcondition("const_pow")
        comp.dust.boundary.outer.setcondition("const_pow")
        result = set_boundaries_component(sim, J, dt, comp)
        for k in range(Nm):
            assert result[k,k] == 0
            assert result[-Nm+k,-Nm+k] == 0
            assert comp._Y_rhs[k] == 0.0
            assert comp._Y_rhs[-Nm+k] == 0.0  # Nr-1
            assert result[k,k+Nm] != 0
            assert result[-Nm+k,-2*Nm+k] != 0

    def test_set_boundaries_component_mixed(self, sim):
        """Test boundary conditions for mixed dust and gas component"""
        sim.addcomponent(name="mixed_compo", gas_value=np.array([100.0, 200.0, 300.0,400.0,500.0]), mu=18.0, dust_value=np.array([[10., 20.],[30., 40.],[50. ,60.],[70.,80.],[90., 100.]]), dust_active=True, gas_active=True)
        comp = sim.components.__dict__["mixed_compo"]
        Nm = sim.grid._Nm_short
        Nr = sim.grid.Nr
        J = sp.csc_matrix(np.eye(Nr + Nm*Nr))
        dt = 0.1

        #val
        comp.gas.boundary.inner.setcondition("val", value=150.0)
        comp.gas.boundary.outer.setcondition("val", value=50.0)
        comp.dust.boundary.inner.setcondition("val", value=250.0)
        comp.dust.boundary.outer.setcondition("val", value=75.0)

        result = set_boundaries_component(sim, J, dt, comp)
        assert comp._Y_rhs[0] == 150.0
        assert comp._Y_rhs[Nr-1] == 50.0  # Nr-1
        assert comp._Y_rhs[Nr] == 250.0
        assert comp._Y_rhs[-1] == 75.0  # Nr-1

        #const_val
        comp._Y_rhs = np.concatenate((comp.gas.Sigma, comp.dust.Sigma.ravel()))
        comp.gas.boundary.inner.setcondition("const_val")
        comp.gas.boundary.outer.setcondition("const_val")
        comp.dust.boundary.inner.setcondition("const_val")
        comp.dust.boundary.outer.setcondition("const_val")
        result = set_boundaries_component(sim, J, dt, comp)
        assert (comp._Y_rhs[0] == 0.0)
        assert (comp._Y_rhs[Nr-1:Nr] == 0.0)  # Nr-1
        assert all(comp._Y_rhs[Nr:Nr+Nm] == 0.0)
        assert all(comp._Y_rhs[-Nm:] == 0.0)  # Nr-1
        for k in range(Nm):
            assert result[Nr+k,Nr+k+Nm] == 10. # 1./dt
            assert result[-Nm+k,-2*Nm+k] == 10. # 1./dt
        assert result[0,1] == 10. # 1./dt
        assert result[Nr-1,Nr-2] == 10. # 1./dt

        #grad
        comp._Y_rhs = np.concatenate((comp.gas.Sigma, comp.dust.Sigma.ravel()))
        comp.gas.boundary.inner.setcondition("grad", value=1.0)
        comp.gas.boundary.outer.setcondition("grad", value=-1.0)
        comp.dust.boundary.inner.setcondition("grad", value=1.0)
        comp.dust.boundary.outer.setcondition("grad", value=-1.0)
        result = set_boundaries_component(sim, J, dt, comp)
        K1 = -sim.grid.r[1]/sim.grid.r[0]
        Km1 = -sim.grid.r[-2]/sim.grid.r[-1]
        for k in range(Nm):
            np.testing.assert_almost_equal(result[Nr+k,Nr+k+Nm],  (-K1/dt))
            np.testing.assert_almost_equal(result[-Nm+k,-2*Nm+k], (-Km1/dt))

        np.testing.assert_almost_equal(result[0,1],  (-K1/dt))
        np.testing.assert_almost_equal(result[Nr-1,Nr-2], (-Km1/dt))

        #const_grad
        comp._Y_rhs = np.concatenate((comp.gas.Sigma, comp.dust.Sigma.ravel()))
        comp.gas.boundary.inner.setcondition("const_grad")
        comp.gas.boundary.outer.setcondition("const_grad")
        comp.dust.boundary.inner.setcondition("const_grad")
        comp.dust.boundary.outer.setcondition("const_grad")
        result = set_boundaries_component(sim, J, dt, comp)
        for k in range(Nm):
            assert result[Nr+k,Nr+k] == 0
            assert result[-Nm+k,-Nm+k] == 0
            assert comp._Y_rhs[Nr+k] == 0.0
            assert comp._Y_rhs[-Nm+k] == 0.0  # Nr-1
            assert result[Nr+k,Nr+k+Nm] != 0
            assert result[Nr+k,Nr+k+2*Nm] != 0
            assert result[-Nm+k,-2*Nm+k] != 0
            assert result[-Nm+k,-3*Nm+k] != 0
        assert result[0,0] == 0
        assert result[Nr-1,Nr-1] == 0
        assert comp._Y_rhs[0] == 0.0
        assert comp._Y_rhs[Nr-1] == 0.0  # Nr
        assert result[0,1] != 0
        assert result[0,2] != 0
        assert result[Nr-1,Nr-2] != 0
        assert result[Nr-1,Nr-3] != 0

        #pow
        comp._Y_rhs = np.concatenate((comp.gas.Sigma, comp.dust.Sigma.ravel()))
        p_inner = 2.
        p_outer = 0.5
        comp.gas.boundary.inner.setcondition("pow", value=p_inner)
        comp.gas.boundary.outer.setcondition("pow", value=p_outer)
        comp.dust.boundary.inner.setcondition("pow", value=p_inner)
        comp.dust.boundary.outer.setcondition("pow", value=p_outer)
        result = set_boundaries_component(sim, J, dt, comp)
        for k in range(Nm):
            expected_inner = (sim.grid.r[0]/sim.grid.r[1])**p_inner
            expected_outer = (sim.grid.r[-1]/sim.grid.r[-2])**p_outer
            assert comp._Y_rhs[Nr+k] == expected_inner*comp._Y_rhs[Nr+k+Nm]
            assert comp._Y_rhs[-Nm+k] == expected_outer*comp._Y_rhs[-2*Nm+k]
        expected_inner = (sim.grid.r[0]/sim.grid.r[1])**p_inner
        expected_outer = (sim.grid.r[-1]/sim.grid.r[-2])**p_outer
        assert comp._Y_rhs[0] == expected_inner*comp._Y_rhs[1]
        assert comp._Y_rhs[Nr-1] == expected_outer*comp._Y_rhs[Nr-2]

        #const_pow
        comp._Y_rhs = np.concatenate((comp.gas.Sigma, comp.dust.Sigma.ravel()))
        comp.gas.boundary.inner.setcondition("const_pow")
        comp.gas.boundary.outer.setcondition("const_pow")
        comp.dust.boundary.inner.setcondition("const_pow")
        comp.dust.boundary.outer.setcondition("const_pow")
        result = set_boundaries_component(sim, J, dt, comp)
        for k in range(Nm):
            assert result[Nr+k,Nr+k] == 0
            assert result[-Nm+k,-Nm+k] == 0
            assert comp._Y_rhs[Nr+k] == 0.0
            assert comp._Y_rhs[-Nm+k] == 0.0  # Nr-1
            assert result[Nr+k,Nr+k+Nm] != 0
            assert result[-Nm+k,-2*Nm+k] != 0
        assert result[0,0] == 0
        assert result[Nr-1,Nr-1] == 0
        assert comp._Y_rhs[0] == 0.0
        assert comp._Y_rhs[Nr-1] == 0.0  # Nr-1
        assert result[0,1] != 0
        assert result[Nr-1,Nr-2] != 0


    def test_boundary_gas_tracer(self, sim):
        """Test boundary conditions for gas tracer component"""
        sim.addcomponent(name="gas_tracer", gas_value=np.array([100.0, 200.0, 300.0,400.0,500.0]), mu=18.0, gas_tracer=True)
        comp = sim.components.__dict__["gas_tracer"]
        J = sp.csc_matrix(np.eye(5))
        dt = 0.1

        #val
        comp.gas.boundary.inner.setcondition("val", value=150.0)
        comp.gas.boundary.outer.setcondition("val", value=50.0)

        result = set_boundaries_component(sim, J, dt, comp)
        assert comp._Y_rhs[0] == 150.0 * sim.gas.Sigma[0]
        assert comp._Y_rhs[-1] == 50.0 * sim.gas.Sigma[-1] # Nr-1
    
    def test_boundary_dust_tracer_no_compo(self, sim):
        """Test boundary conditions for dust tracer component"""
        sim.addcomponent("dust_tracer",None,None, dust_value=np.array([[10., 20.],[30., 40.],[50. ,60.],[70.,80.],[90., 100.]]), dust_tracer=True)
        comp = sim.components.__dict__["dust_tracer"]
        Nm = sim.grid._Nm_short
        Nr = sim.grid.Nr
        J = sp.csc_matrix(np.eye(Nm*Nr))
        dt = 0.1

        #val
        comp.dust.boundary.inner.setcondition("val", value=150.0)
        comp.dust.boundary.outer.setcondition("val", value=50.0)

        result = set_boundaries_component(sim, J, dt, comp)
        for k in range(Nm):
            assert comp._Y_rhs[k] == 150.0 * sim.dust.Sigma[0,k]
            assert comp._Y_rhs[-Nm+k] == 50.0 * sim.dust.Sigma[-1,k] # Nr-1

    def test_boundary_dust_tracer_no_compo(self, sim):
        """Test boundary conditions for dust tracer component"""
        sim.addcomponent("dust_background",None,None, dust_value=sim.dust.Sigma*0.5, dust_active=True)
        sim.addcomponent("dust_tracer",None,None, dust_value=np.array([[10., 20.],[30., 40.],[50. ,60.],[70.,80.],[90., 100.]]), dust_tracer=True)
        comp = sim.components.__dict__["dust_tracer"]
        Nm = sim.grid._Nm_short
        Nr = sim.grid.Nr
        J = sp.csc_matrix(np.eye(Nm*Nr))
        dt = 0.1

        #val
        comp.dust.boundary.inner.setcondition("val", value=150.0)
        comp.dust.boundary.outer.setcondition("val", value=50.0)

        result = set_boundaries_component(sim, J, dt, comp)
        for k in range(Nm):
            assert comp._Y_rhs[k] == 150.0 * sim.dust.Sigma[0,k]
            assert comp._Y_rhs[-Nm+k] == 50.0 * sim.dust.Sigma[-1,k] # Nr-1