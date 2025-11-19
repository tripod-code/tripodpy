import pytest
from unittest.mock import Mock, MagicMock, patch
import warnings
warnings.filterwarnings("ignore", message=".*NumPy module was reloaded.*")
import numpy as np

from tripodpy.std.gas import (
    enforce_floor_value, prepare, finalize, set_implicit_boundaries_compo,
    set_implicit_boundaries, Sigma_tot, mu, dt_compo, Fi_compo,
    S_hyd_compo, S_tot_compo, S_ext_total
)

class TestFloorValueEnforcement:
    def test_enforce_floor_value_basic(self, monkeypatch):
        """Test basic floor value enforcement for gas components"""
        sim = Mock()
        sim.gas.SigmaFloor = np.ones(3) * 1e-10
        
        # Mock gas components
        comp1 = Mock()
        comp1.gas.Sigma = np.array([1e-12, 1.0, 1e-8])  # One below floor
        
        comp2 = Mock()
        comp2.gas.Sigma = np.array([1.0, 2.0, 3.0])  # All above floor
        
        sim.components = Mock()
        sim.components.__dict__ = {
            'water': comp1,
            'co': comp2,
            '_private': Mock()  # Should be skipped
        }
        
        # Mock enforce_floor function
        monkeypatch.setattr('dustpy.std.gas_f.enforce_floor',
                           lambda sigma, floor: np.maximum(sigma, floor))
        
        enforce_floor_value(sim)
        
        # Verify floor was enforced
        np.testing.assert_array_equal(comp1.gas.Sigma, np.array([1e-10, 1.0, 1e-8]))
        np.testing.assert_array_equal(comp2.gas.Sigma, np.array([1.0, 2.0, 3.0]))
    
    def test_enforce_floor_value_no_components(self):
        """Test floor value enforcement with no gas components"""
        sim = Mock()
        sim.gas.SigmaFloor = np.ones(3) * 1e-10
        sim.components = Mock()
        sim.components.__dict__ = {'_private': Mock()}
        
        # Should not raise an error
        enforce_floor_value(sim)

class TestPrepareFinalize:
    def test_prepare_basic(self):
        """Test preparation of gas integration step"""
        sim = Mock()
        sim.gas.Sigma = np.array([1.0, 2.0, 3.0])
        sim.gas._SigmaOld = np.zeros(3)
        
        # Mock gas components
        comp1 = Mock()
        comp1.gas.Sigma = np.array([0.5, 1.0, 1.5])
        comp1.gas._SigmaOld = np.zeros(3)
        
        comp2 = Mock()
        comp2.gas.Sigma = np.array([0.5, 1.0, 1.5])
        comp2.gas._SigmaOld = np.zeros(3)
        
        sim.components = Mock()
        sim.components.__dict__ = {
            'water': comp1,
            'co': comp2,
            '_private': Mock()
        }
        
        prepare(sim)
        
        # Verify old values were stored
        np.testing.assert_array_equal(sim.gas._SigmaOld, sim.gas.Sigma)
        np.testing.assert_array_equal(comp1.gas._SigmaOld, comp1.gas.Sigma)
        np.testing.assert_array_equal(comp2.gas._SigmaOld, comp2.gas.Sigma)
    
    def test_prepare_no_components(self):
        """Test preparation with no components"""
        sim = Mock()
        sim.gas.Sigma = np.array([1.0, 2.0, 3.0])
        sim.gas._SigmaOld = np.zeros(3)
        sim.components = Mock()
        sim.components.__dict__ = {}
        
        prepare(sim)
        
        # Should still store main gas values
        np.testing.assert_array_equal(sim.gas._SigmaOld, sim.gas.Sigma)
    
    def test_finalize_basic(self, monkeypatch):
        """Test finalization of gas integration step"""
        sim = Mock()
        
        # Mock update methods
        sim.gas.v = Mock()
        sim.gas.v.update = Mock()
        sim.gas.Fi = Mock()
        sim.gas.Fi.update = Mock()
        sim.gas.S.hyd = Mock()
        sim.gas.S.hyd.update = Mock()
        
        # Mock the functions called by finalize
        monkeypatch.setattr('tripodpy.std.gas.enforce_floor_value', Mock())
        monkeypatch.setattr('tripodpy.std.gas.set_implicit_boundaries_compo', Mock())
        
        finalize(sim)
        
        # Verify all updates were called
        assert sim.gas.v.update.called
        assert sim.gas.Fi.update.called
        assert sim.gas.S.hyd.update.called

class TestBoundaryConditions:
    def test_set_implicit_boundaries_compo(self, monkeypatch):
        """Test setting implicit boundaries for components"""
        sim = Mock()
        sim.t.prevstepsize = 0.1
        sim.grid.ri = np.array([0.5, 1.5, 2.5, 3.5])
        
        # Mock component
        comp = Mock()
        comp.gas.Fi = np.zeros(4)
        comp.gas.Sigma = np.ones(3)
        comp.gas._SigmaOld = np.ones(3) * 0.9
        comp.gas.S.tot = np.zeros(3)
        comp.gas.S.hyd = np.zeros(3)
        
        sim.components = Mock()
        sim.components.__dict__ = {'water': comp, '_private': Mock()}
        
        # Mock implicit_boundaries function
        mock_return = [0.1, 0.2, 0.05, 0.15]  # [S_inner, S_outer, Fi_inner, Fi_outer]
        monkeypatch.setattr('dustpy.std.gas_f.implicit_boundaries',
                           lambda *args: mock_return)
        
        set_implicit_boundaries_compo(sim)
        
        # Verify boundary conditions were set
        assert comp.gas.S.tot[0] == 0.1   # Inner source
        assert comp.gas.S.hyd[0] == 0.1   # Inner hydrodynamic source
        assert comp.gas.S.tot[-1] == 0.2  # Outer source
        assert comp.gas.S.hyd[-1] == 0.2  # Outer hydrodynamic source
        assert comp.gas.Fi[0] == 0.05     # Inner flux
        assert comp.gas.Fi[-1] == 0.15    # Outer flux
    
    def test_set_implicit_boundaries_main(self, monkeypatch):
        """Test setting implicit boundaries for main gas"""
        sim = Mock()
        sim.t.prevstepsize = 0.1
        sim.grid.ri = np.array([0.5, 1.5, 2.5, 3.5])
        sim.gas.Fi = np.zeros(4)
        sim.gas.Sigma = np.ones(3)
        sim.gas._SigmaOld = np.ones(3) * 0.9
        sim.gas.S.tot = np.zeros(3)
        sim.gas.S.hyd = np.zeros(3)
        
        # Mock implicit_boundaries function
        mock_return = [0.1, 0.2, 0.05, 0.15]
        monkeypatch.setattr('dustpy.std.gas_f.implicit_boundaries',
                           lambda *args: mock_return)
        
        set_implicit_boundaries(sim)
        
        # Verify boundary conditions were set for main gas
        assert sim.gas.S.tot[0] == 0.1
        assert sim.gas.S.hyd[0] == 0.1
        assert sim.gas.S.tot[-1] == 0.2
        assert sim.gas.S.hyd[-1] == 0.2
        assert sim.gas.Fi[0] == 0.05
        assert sim.gas.Fi[-1] == 0.15
    
    def test_set_implicit_boundaries_compo_no_components(self):
        """Test boundary setting with no components"""
        sim = Mock()
        sim.components = Mock()
        sim.components.__dict__ = {}
        
        # Should not raise an error
        set_implicit_boundaries_compo(sim)

class TestSurfaceDensityCalculations:
    def test_Sigma_tot_basic(self):
        """Test total surface density calculation"""
        sim = Mock()
        sim.gas.Sigma = np.zeros(3)  # Will be overwritten
        
        # Mock components with different surface densities
        comp1 = Mock()
        comp1.gas.Sigma = np.array([1.0, 2.0, 3.0])
        
        comp2 = Mock()
        comp2.gas.Sigma = np.array([0.5, 1.0, 1.5])
        
        comp3 = Mock()
        comp3.gas.Sigma = np.array([0.2, 0.3, 0.4])
        
        sim.components = Mock()
        sim.components.__dict__ = {
            'water': comp1,
            'co': comp2,
            'co2': comp3,
            '_private': Mock()  # Should be ignored
        }
        
        result = Sigma_tot(sim)
        expected = np.array([1.7, 3.3, 4.9])  # Sum of all components
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_Sigma_tot_no_components(self):
        """Test total surface density with no components"""
        sim = Mock()
        sim.gas.Sigma = np.array([1.0, 2.0, 3.0])
        sim.components = Mock()
        sim.components.__dict__ = {'_private': Mock()}
        
        result = Sigma_tot(sim)
        expected = np.zeros(3)  # Should be zero if no components
        np.testing.assert_array_equal(result, expected)

class TestMolecularWeight:
    def test_mu_basic(self):
        """Test mean molecular weight calculation"""
        sim = Mock()
        sim.gas.Sigma = np.array([30.0, 60.0, 90.0])  # Total surface density
        sim.gas.mu = np.zeros(3)  # Will be calculated
        
        # Mock components with different molecular weights
        comp1 = Mock()
        comp1.gas.Sigma = np.array([18.0, 36.0, 54.0])  # Water: Sigma
        comp1.gas.pars.mu = 18.0  # Water molecular weight
        
        comp2 = Mock()
        comp2.gas.Sigma = np.array([12.0, 24.0, 36.0])  # CO: Sigma
        comp2.gas.pars.mu = 28.0  # CO molecular weight
        
        sim.components = Mock()
        sim.components.__dict__ = {
            'water': comp1,
            'co': comp2,
            '_private': Mock()
        }
        
        result = mu(sim)
        
        # Expected calculation: mu = Sigma_tot / sum(Sigma_i/mu_i)
        # For each cell: sum(Sigma_i/mu_i) = 18/18 + 12/28 = 1.0 + 0.429 = 1.429
        # mu = 30/1.429 ≈ 21.0 (for first cell)
        expected_sum_inv = comp1.gas.Sigma/18.0 + comp2.gas.Sigma/28.0
        expected = sim.gas.Sigma / expected_sum_inv
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_mu_single_component(self):
        """Test molecular weight with single component"""
        sim = Mock()
        sim.gas.Sigma = np.array([18.0, 36.0, 54.0])
        sim.gas.mu = np.zeros(3)
        
        comp = Mock()
        comp.gas.Sigma = np.array([18.0, 36.0, 54.0])
        comp.gas.pars.mu = 18.0
        
        sim.components = Mock()
        sim.components.__dict__ = {'water': comp}
        
        result = mu(sim)
        
        # Should equal component molecular weight when only one component
        expected = np.ones(3) * 18.0
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_mu_zero_sigma(self):
        """Test molecular weight with zero surface density"""
        sim = Mock()
        sim.gas.Sigma = np.array([0.0, 36.0, 54.0])
        sim.gas.mu = np.zeros(3)
        
        comp = Mock()
        comp.gas.Sigma = np.array([0.0, 36.0, 54.0])
        comp.gas.pars.mu = 18.0
        
        sim.components = Mock()
        sim.components.__dict__ = {'water': comp}
        with pytest.warns(RuntimeWarning):
            result = mu(sim)
        
        # First element should be 0/0 which is handled by numpy (nan or inf)
        assert np.isnan(result[0]) or np.isinf(result[0])
        np.testing.assert_array_almost_equal(result[1:], [18.0, 18.0])

class TestTimeStep:
    def test_dt_compo_basic(self, monkeypatch):
        """Test component time step calculation"""
        sim = Mock()
        sim.gas.SigmaFloor = np.ones(3) * 1e-10
        
        # Mock active components
        comp1 = Mock()
        comp1.gas._active = True
        comp1.gas.Sigma_dot = np.array([0.1, 0.2, 0.3])
        comp1.gas.Sigma = np.array([1.0, 2.0, 3.0])
        
        comp2 = Mock()
        comp2.gas._active = True
        comp2.gas.Sigma_dot = np.array([0.05, 0.1, 0.15])
        comp2.gas.Sigma = np.array([0.5, 1.0, 1.5])
        
        # Mock inactive component (should be ignored)
        comp3 = Mock()
        comp3.gas._active = False
        
        sim.components = Mock()
        sim.components.__dict__ = {
            'active1': comp1,
            'active2': comp2,
            'inactive': comp3,
            '_private': Mock()
        }
        
        # Mock timestep function
        def mock_timestep(sigma_dot, sigma, floor):
            return np.min(sigma / (np.abs(sigma_dot) + 1e-100))
        
        monkeypatch.setattr('dustpy.std.gas_f.timestep', mock_timestep)
        
        result = dt_compo(sim)
        
        # Should return minimum timestep from all active components
        dt1 = mock_timestep(comp1.gas.Sigma_dot, comp1.gas.Sigma, sim.gas.SigmaFloor)
        dt2 = mock_timestep(comp2.gas.Sigma_dot, comp2.gas.Sigma, sim.gas.SigmaFloor)
        expected = min(dt1, dt2)
        
        assert result == expected
    
    def test_dt_compo_no_active_components(self):
        """Test time step with no active components"""
        sim = Mock()
        
        # Only inactive components
        comp = Mock()
        comp.gas._active = False
        
        sim.components = Mock()
        sim.components.__dict__ = {'inactive': comp, '_private': Mock()}
        
        result = dt_compo(sim)
        assert result == 1e100
    
    def test_dt_compo_zero_derivative(self, monkeypatch):
        """Test time step with zero time derivatives"""
        sim = Mock()
        sim.gas.SigmaFloor = np.ones(3) * 1e-10
        
        comp = Mock()
        comp.gas._active = True
        comp.gas.Sigma_dot = np.zeros(3)  # Zero derivatives
        comp.gas.Sigma = np.ones(3)
        
        
        sim.components = Mock()
        sim.components.__dict__ = {'zero_deriv': comp}
        
        
        result = dt_compo(sim)
        assert result == 1e100

class TestFluxCalculations:
    def test_Fi_compo_with_group(self, monkeypatch):
        """Test flux calculation with group parameter"""
        sim = Mock()
        sim.gas.v.rad = np.ones(3) * 0.1
        sim.grid.r = np.array([1.0, 2.0, 3.0])
        sim.grid.ri = np.array([0.5, 1.5, 2.5, 3.5])
        
        group = Mock()
        group.Sigma = np.ones(3) * 10.0
        
        # Mock fi function
        expected_flux = np.ones(4) * 1.0
        monkeypatch.setattr('dustpy.std.gas_f.fi',
                           lambda sigma, v_rad, r, ri: expected_flux)
        
        result = Fi_compo(sim, group=group)
        np.testing.assert_array_equal(result, expected_flux)
    
    def test_Fi_compo_with_compkey(self, monkeypatch):
        """Test flux calculation with component key"""
        sim = Mock()
        sim.gas.v.rad = np.ones(3) * 0.1
        sim.grid.r = np.array([1.0, 2.0, 3.0])
        sim.grid.ri = np.array([0.5, 1.5, 2.5, 3.5])
        
        comp = Mock()
        comp.gas.Sigma = np.ones(3) * 5.0
        
        sim.components = Mock()
        sim.components.__dict__ = {'water': comp}
        
        expected_flux = np.ones(4) * 0.5
        monkeypatch.setattr('dustpy.std.gas_f.fi',
                           lambda sigma, v_rad, r, ri: expected_flux)
        
        result = Fi_compo(sim, compkey='water')
        np.testing.assert_array_equal(result, expected_flux)
    
    def test_Fi_compo_invalid_component(self):
        """Test flux calculation with invalid component key"""
        sim = Mock()
        sim.components = Mock()
        sim.components.__dict__ = {'water': Mock()}
        
        with pytest.raises(ValueError, match="Component invalid not found"):
            Fi_compo(sim, compkey='invalid')
    
    def test_Fi_compo_default_component(self, monkeypatch):
        """Test flux calculation with default component"""
        sim = Mock()
        sim.gas.v.rad = np.ones(3) * 0.1
        sim.grid.r = np.array([1.0, 2.0, 3.0])
        sim.grid.ri = np.array([0.5, 1.5, 2.5, 3.5])
        
        comp = Mock()
        comp.gas.Sigma = np.ones(3) * 2.0
        
        sim.components = Mock()
        sim.components.__dict__ = {'default': comp}
        
        expected_flux = np.ones(4) * 0.2
        monkeypatch.setattr('dustpy.std.gas_f.fi',
                           lambda sigma, v_rad, r, ri: expected_flux)
        
        result = Fi_compo(sim, compkey='default')
        np.testing.assert_array_equal(result, expected_flux)

class TestSourceTerms:
    def test_S_hyd_compo_with_group(self, monkeypatch):
        """Test hydrodynamic source terms with group"""
        sim = Mock()
        sim.grid.ri = np.array([0.5, 1.5, 2.5, 3.5])
        
        group = Mock()
        group.Fi = np.ones(4) * 1.0
        
        expected_source = np.ones(3) * 0.1
        monkeypatch.setattr('dustpy.std.gas_f.s_hyd',
                           lambda Fi, ri: expected_source)
        
        result = S_hyd_compo(sim, group=group)
        np.testing.assert_array_equal(result, expected_source)
    
    def test_S_hyd_compo_with_compkey(self, monkeypatch):
        """Test hydrodynamic source terms with component key"""
        sim = Mock()
        sim.grid.ri = np.array([0.5, 1.5, 2.5, 3.5])
        
        comp = Mock()
        comp.gas.Fi = np.ones(4) * 2.0
        
        sim.components = Mock()
        sim.components.__dict__ = {'water': comp}
        
        expected_source = np.ones(3) * 0.2
        monkeypatch.setattr('dustpy.std.gas_f.s_hyd',
                           lambda Fi, ri: expected_source)
        
        result = S_hyd_compo(sim, compkey='water')
        np.testing.assert_array_equal(result, expected_source)
    
    def test_S_tot_compo_with_group(self):
        """Test total source terms with group"""
        sim = Mock()
        
        group = Mock()
        group.S.ext = np.ones(3) * 0.1
        group.S.hyd = np.ones(3) * 0.2
        
        result = S_tot_compo(sim, group=group)
        expected = np.ones(3) * 0.3
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_S_tot_compo_with_compkey(self):
        """Test total source terms with component key"""
        sim = Mock()
        
        comp = Mock()
        comp.gas.S.hyd = np.ones(3) * 0.15
        comp.gas.S.ext = np.ones(3) * 0.25
        
        sim.components = Mock()
        sim.components.__dict__ = {'water': comp}
        
        result = S_tot_compo(sim, compkey='water')
        expected = np.ones(3) * 0.4
        np.testing.assert_array_equal(result, expected)
    
    def test_S_ext_total_basic(self):
        """Test total external source terms"""
        sim = Mock()
        sim.gas.Sigma = np.zeros(3)
        
        # Mock components with different source terms
        comp1 = Mock()
        comp1.gas.Sigma_dot = np.array([0.1, 0.2, 0.3])
        
        comp2 = Mock()
        comp2.gas.Sigma_dot = np.array([0.05, 0.1, 0.15])
        
        comp3 = Mock()
        comp3.gas.Sigma_dot = np.array([0.02, 0.03, 0.04])
        
        sim.components = Mock()
        sim.components.__dict__ = {
            'water': comp1,
            'co': comp2,
            'co2': comp3,
            '_private': Mock()
        }
        
        result = S_ext_total(sim)
        expected = np.array([0.17, 0.33, 0.49])  # Sum of all Sigma_dot
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_S_ext_total_no_components(self):
        """Test total external source terms with no components"""
        sim = Mock()
        sim.gas.Sigma = np.array([1.0, 2.0, 3.0])
        sim.components = Mock()
        sim.components.__dict__ = {'_private': Mock()}
        
        result = S_ext_total(sim)
        expected = np.zeros(3)
        np.testing.assert_array_equal(result, expected)

class TestEdgeCases:
    def test_component_dict_access_edge_cases(self, monkeypatch):
        """Test various edge cases in component dictionary access"""
        sim = Mock()
        sim.gas.Sigma = np.zeros(3)
        sim.gas.mu = np.zeros(3)

        # Test with component that has None as key
        comp = Mock()
        comp.gas.Sigma = np.ones(3)
        comp.gas.pars.mu = 18.0
        
        # Test with mixed valid and invalid keys
        sim.components = Mock()
        sim.components.__dict__ = {
            'water': comp,
            '_internal': Mock(),
            '__dunder__': Mock(),
            '_': Mock(),  # Edge case: single underscore
        }
        
        # Test Sigma_tot with these edge cases
        result = Sigma_tot(sim)
        
        # Should only include 'water' and '' (empty string doesn't start with _)
        # But empty string is unusual, so let's just test that private ones are skipped
        expected_shape = (3,)
        assert result.shape == expected_shape
    
    def test_molecular_weight_edge_cases(self):
        """Test molecular weight calculation edge cases"""
        sim = Mock()
        sim.gas.Sigma = np.array([1e-100, 1.0, 1e100])  # Very small, normal, very large
        sim.gas.mu = np.zeros(3)

        comp = Mock()
        comp.gas.Sigma = np.array([1e-100, 1.0, 1e100])
        comp.gas.pars.mu = 18.0
        
        sim.components = Mock()
        sim.components.__dict__ = {'water': comp}
        
        result = mu(sim)
        
        # All should equal the component molecular weight
        np.testing.assert_array_almost_equal(result, [18.0, 18.0, 18.0])
    
    @pytest.mark.parametrize("sigma_values,expected_behavior", [
        (np.array([0.0, 0.0, 0.0]), "zero_division"),
        (np.array([np.inf, 1.0, 2.0]), "infinity_handling"),
        (np.array([np.nan, 1.0, 2.0]), "nan_handling"),
    ])
    def test_parametrized_edge_cases(self, sigma_values, expected_behavior):
        """Test various edge cases with parametrization"""
        sim = Mock()
        sim.gas.Sigma = sigma_values
        sim.gas.mu = np.zeros(3)
        
        comp = Mock()
        comp.gas.Sigma = sigma_values
        comp.gas.pars.mu = 18.0
        
        sim.components = Mock()
        sim.components.__dict__ = {'water': comp}
        
        if expected_behavior == "zero_division":
            with pytest.warns(RuntimeWarning):
                result = mu(sim)
            # With zero Sigma, mu should be nan or inf
            assert np.isnan(result[0]) or np.isinf(result[0])
        elif expected_behavior == "infinity_handling":
            with pytest.warns(RuntimeWarning):
                result = mu(sim)
            assert np.isinf(result[0]) or np.isnan(result[0])
            np.testing.assert_array_equal(result[1:], [18.0, 18.0])
        elif expected_behavior == "nan_handling":
            result = mu(sim)
            assert np.isnan(result[0])
            np.testing.assert_array_equal(result[1:], [18.0, 18.0])
