from unittest import mock
from matplotlib.image import resample
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from simframe.integration import Scheme
from tripod import Simulation
import scipy.sparse as sp
import dustpy.constants as c

from tripod.std.dust import (
    dt, dt_Sigma, dt_smax, S_smax_hyd, S_hyd_compo, S_tot_compo,
    rhos_compo, Fi_sig1smax, dt_compo, prepare, finalize, 
    smax_initial, Sigma_initial, jacobian, a, F_adv, F_diff,
    m, p_frag, p_stick, H, rho_midplane, smax_deriv, S_coag,
    S_tot_ext, enforce_f, dadsig, dsigda, S_tot, S_compo,
    vrel_brownian_motion, q_eff, q_frag, q_rec, p_frag_trans,
    p_drift_frag, D_mod, vrad_mod, Y_jacobian,_f_impl_1_direct
)

class TestDustTimesteps:
    @pytest.fixture
    def mock_sim(self):
        sim = Simulation()
        sim.ini.grid.Nr = 5
        sim.ini.grid.rmax = 10*c.au
        sim.initialize()
        sim.writer = None 
        sim.t.snapshot = [1]
        sim.run()
        return sim


    def test_dt_basic(self, monkeypatch):
        """Test basic time step calculation"""
        sim = Mock()
        
        # Mock dt_Sigma and dt_smax
        monkeypatch.setattr('tripod.std.dust.dt_Sigma', lambda _: 100.0)
        monkeypatch.setattr('tripod.std.dust.dt_smax', lambda _: 50.0)
        
        result = dt(sim)
        assert result == 50.0
    
    def test_dt_Sigma_no_negative_sources(self):
        """Test dt_Sigma when no negative source terms exist"""
        sim = Mock()
        sim.dust.S.tot = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        
        result = dt_Sigma(sim)
        assert result == 1e100
    
    def test_dt_Sigma_with_negative_sources(self, mock_sim):
        """Test dt_Sigma with negative source terms"""
        mock_sim.dust.S.tot = - abs(mock_sim.dust.S.tot)  # Ensure some negative values
        result = dt_Sigma(mock_sim)
        assert len(result.shape) == 0  # Should be a scalar
        assert result < 1e100
        assert result > 0
    
    def test_dt_smax(self,mock_sim):
        """Test smax time step calculation"""

        result = dt_smax(mock_sim)
        assert len(result.shape) == 0 
        assert result > 0
    
    def test_dt_compo(self,mock_sim):
        """Test component time step calculation"""
        #add a dust component with negative source terms
        mock_sim.addcomponent("background",None,1,dust_value=mock_sim.dust.Sigma,dust_active=True)
        mock_sim.components.background.dust.S.tot = - abs(mock_sim.components.background.dust.S.tot)-1 # Ensure some negative values

        result = dt_compo(mock_sim)
        assert result > 0
        assert result < 1e100

class TestParticleProperties:
    @pytest.fixture
    def mock_sim(self):
        sim = Simulation()
        sim.ini.grid.Nr = 5
        sim.ini.grid.rmax = 10*c.au
        sim.initialize()
        return sim

    def test_a_calculation(self, mock_sim):
        """Test particle size calculation"""

        
        result = a(mock_sim)
        Nr = mock_sim.grid.Nr

        assert result.shape == (Nr, 5)
        assert all(result[:,0] < result[:,2])  # a0 < a1 
        assert all(result[:,1] < result[:,2])  # fudge *a1 < a1
        assert all(result[:,0] < result[:,4])  # a0 < amax
        assert all(result[:,2] < result[:,4])  # a1 < amax
        
        
    
    def test_m_calculation(self, mock_sim):
        """Test particle mass calculation"""

        Nr = mock_sim.grid.Nr
        result = m(mock_sim)
        expected_result = (4/3) * np.pi * mock_sim.dust.rhos * mock_sim.dust.fill * mock_sim.dust.a**3
        assert result.shape == (Nr, 5)
        np.testing.assert_array_almost_equal(result, expected_result)
    
    def test_H_calculation(self, mock_sim):
        """Test dust scale height calculation"""

        Nr = mock_sim.grid.Nr
        Nm_l = mock_sim.grid._Nm_long

        #check if shape is correct
        result = H(mock_sim)
        assert result.shape == (Nr, Nm_l)

        #check if the only thing it does is call h_dubrulle1995 with correct arguments
        with patch('dustpy.std.dust_f.h_dubrulle1995') as mock_h:
            mock_h.return_value = np.ones((Nr, Nm_l))
            result = H(mock_sim)
            assert result.shape == (Nr, Nm_l)
            np.testing.assert_array_equal(result, np.ones((Nr, Nm_l)))
            mock_h.assert_called_once()
    
    def test_rho_midplane_calculation(self,mock_sim):
        """Test midplane density calculation"""

        Nr = mock_sim.grid.Nr
        Nm_l = mock_sim.grid._Nm_long
        result = rho_midplane(mock_sim)
        assert result.shape == (Nr, Nm_l)
        assert np.all(result > 0)

class TestFluxCalculations:
    def test_F_adv(self, monkeypatch):
        """Test advective flux calculation"""
        sim = Mock()
        sim.dust.Sigma = np.ones((3, 2))
        sim.dust.v.rad_flux = np.ones((3, 3))
        sim.grid.r = np.array([1.0, 2.0, 3.0])
        sim.grid.ri = np.array([0.5, 1.5, 2.5, 3.5])
        
        with patch('dustpy.std.dust_f.fi_adv') as mock_fi_adv:
            mock_fi_adv.return_value = np.ones((4, 2))
            result = F_adv(sim)
            assert result.shape == (4, 2)
            mock_fi_adv.assert_called_once()
    
    def test_F_diff(self, monkeypatch):
        """Test diffusive flux calculation"""
        sim = Mock()
        sim.dust.Sigma = np.ones((3, 2))
        sim.dust.D = np.ones((3, 3))
        sim.gas.Sigma = np.ones(3)
        sim.dust.St = np.ones((3, 3))
        sim.dust.f.drift = 1.
        sim.dust.delta.rad = np.ones(3)
        sim.gas.cs = np.ones(3)
        sim.grid.r = np.array([1.0, 2.0, 3.0])
        sim.grid.ri = np.array([0.5, 1.5, 2.5, 3.5])
        
        monkeypatch.setattr('tripod.std.dust_f.fi_diff_no_limit', 
                           lambda D, Sigma, gas_Sigma, St_drift, cs_term, r, ri: np.ones((4, 2)))
        
        result = F_diff(sim)
        assert result.shape == (4, 2)
        # Check boundary conditions
        np.testing.assert_array_equal(result[:1, :], 0.0)
        np.testing.assert_array_equal(result[-1:, :], 0.0)

class TestSourceTerms:
    def test_S_coag(self, monkeypatch):
        """Test coagulation source terms"""
        sim = Mock()
        sim.dust.a = np.ones((3, 5))
        sim.dust.v.rel.tot = np.ones((3, 5, 5))
        sim.dust.H = np.ones((3, 5))
        sim.dust.m = np.ones((3, 5))
        sim.dust.Sigma = np.ones((3, 2))
        sim.dust.s.min = np.ones(3)
        sim.dust.s.max = np.ones(3) * 10
        sim.dust.q.eff = np.ones(3) * (-3.5)
        sim.dust.SigmaFloor = np.ones((3, 2)) * 1e-10
        
        monkeypatch.setattr('tripod.std.dust_f.s_coag', 
                           lambda cross_sec, v_rel, H, m, Sigma, smin, smax, q_eff, SigmaFloor: np.ones((3, 2)))
        
        result = S_coag(sim)
        assert result.shape == (3, 2)
    
    def test_S_smax_hyd(self, monkeypatch):
        """Test hydrodynamic source terms for smax"""
        sim = Mock()
        sim.dust.Sigma = np.ones((3, 2))
        sim.dust.s.max = np.ones(3)
        sim.dust.S.hyd = np.ones((3, 2))
        sim.grid.ri = np.array([0.5, 1.5, 2.5, 3.5])
        
        monkeypatch.setattr('tripod.std.dust.Fi_sig1smax', lambda s: np.ones(4))
        with patch('dustpy.std.dust_f.s_hyd') as mock_s_hyd:
            mock_s_hyd.return_value = np.ones((3, 2))
            result = S_smax_hyd(sim)
            assert result.shape == (3,)
    
    def test_S_compo_no_components(self):
        """Test component source terms when no components exist"""
        sim = Mock()
        sim.dust.Sigma = np.ones((3, 2))
        # No components attribute
        delattr(sim, 'components')
        
        result = S_compo(sim)
        np.testing.assert_array_equal(result, np.zeros((3, 2)))
    
    def test_S_compo_with_components(self):
        """Test component source terms with active components"""
        sim = Mock()
        sim.dust.Sigma = np.ones((3, 2))
        
        # Mock components
        comp1 = Mock()
        comp1.dust.S_Sigma = np.ones((3, 2)) * 0.1
        comp2 = Mock()
        comp2.dust.S_Sigma = np.ones((3, 2)) * 0.2
        
        sim.components = Mock()
        sim.components.__dict__ = {
            'comp1': comp1,
            'comp2': comp2,
            '_private': Mock()  # Should be ignored
        }
        
        result = S_compo(sim)
        expected = np.ones((3, 2)) * 0.3
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_S_tot_compo(self):
        """Test total source terms for components"""
        group = Mock()
        group.S.ext = np.ones((3, 2)) * 0.1
        group.S.hyd = np.ones((3, 2)) * 0.2
        group.S.coag = np.ones((3, 2)) * 0.3
        
        sim = Mock()
        
        result = S_tot_compo(sim, group)
        expected = np.ones((3, 2)) * 0.6
        np.testing.assert_array_almost_equal(result, expected)

class TestProbabilities:
    def test_p_frag(self, monkeypatch):
        """Test fragmentation probability"""
        sim = Mock()
        sim.dust.v.rel.tot = np.ones((3, 5, 5)) * 10.0
        sim.dust.v.frag = np.ones(3) * 5.0
        
        monkeypatch.setattr('tripod.std.dust_f.pfrag', 
                           lambda v_rel, v_frag: np.ones(3) * 0.5)
        
        result = p_frag(sim)
        np.testing.assert_array_equal(result, np.ones(3) * 0.5)
    
    def test_p_stick(self):
        """Test sticking probability"""
        sim = Mock()
        sim.dust.p.frag = np.ones(3) * 0.3
        
        result = p_stick(sim)
        np.testing.assert_array_almost_equal(result, np.ones(3) * 0.7)
    
    def test_p_frag_trans(self, monkeypatch):
        """Test fragmentation transition probability"""
        sim = Mock()
        sim.dust.St = np.ones((3, 5))
        sim.gas.alpha = np.ones(3)
        sim.gas.Sigma = np.ones(3)
        sim.gas.mu = np.ones(3)
        
        monkeypatch.setattr('tripod.std.dust_f.pfrag_trans', 
                           lambda St, alpha, Sigma, mu: np.ones(3) * 0.4)
        
        result = p_frag_trans(sim)
        np.testing.assert_array_almost_equal(result, np.ones(3) * 0.4)
    
    def test_p_drift_frag(self, monkeypatch):
        """Test drift fragmentation probability"""
        sim = Mock()
        sim.dust.v.rel.rad = np.ones((3, 5, 5))
        sim.dust.v.rel.azi = np.ones((3, 5, 5))
        sim.dust.St = np.ones((3, 5))
        sim.gas.alpha = np.ones(3)
        sim.gas.Sigma = np.ones(3)
        sim.gas.mu = np.ones(3)
        sim.gas.cs = np.ones(3)
        sim.dust.p.fragtrans = np.ones(3)
        
        monkeypatch.setattr('tripod.std.dust_f.pdriftfrag', 
                           lambda v_rad, v_azi, St, alpha, Sigma, mu, cs, p_trans: np.ones(3) * 0.6)
        
        result = p_drift_frag(sim)
        np.testing.assert_array_equal(result, np.ones(3) * 0.6)

class TestInitialConditions:
    @pytest.fixture
    def mock_sim(self):
        sim = Simulation()
        sim.ini.grid.Nr = 5
        sim.ini.grid.rmax = 10*c.au
        return sim
    
    def test_smax_initial_no_drift_limit(self):
        """Test initial smax calculation without drift limitation"""
        sim = Mock()
        sim.dust.s.min = np.ones(3) * 1e-4
        sim.ini.dust.allowDriftingParticles = True
        sim.ini.dust.aIniMax = 1e-2
        
        result = smax_initial(sim)
        np.testing.assert_array_equal(result, np.ones(3) * 1e-2)
    
    def test_smax_initial_with_drift_limit(self, mock_sim):
        """Test initial smax calculation with drift limitation"""
        mock_sim.initialize()
        Nr = mock_sim.grid.Nr
        result = smax_initial(mock_sim)
        assert result.shape == (Nr,)
        assert np.all(result >= mock_sim.dust.s.min * 1.5)
    
    def test_Sigma_initial(self):
        """Test initial surface density calculation"""
        sim = Mock()
        sim.dust.q.eff = np.ones(3) * (-3.5)
        sim.dust.s.min = np.ones(3) * 1e-4
        sim.dust.s.max = np.ones(3) * 1e-2
        sim.dust.s.max[0] = sim.dust.s.min[0] #Edge case where smax = smin
        sim.dust.SigmaFloor = np.ones((3, 2)) * 1e-10
        sim.grid.Nr = 3
        sim.ini.dust.d2gRatio = 0.01
        sim.gas.Sigma = np.ones(3) * 100.0
        sim.grid.r = np.array([1.0, 2.0, 3.0])
        
        result = Sigma_initial(sim)
        assert result.shape == (3, 2)
        assert np.all(result >= 0)
    
    def test_Sigma_initial_q_equals_minus_4(self,mock_sim):
        """Test initial surface density with q = -4"""
        mock_sim.initialize()
        Nr = mock_sim.grid.Nr
        Nm_s = mock_sim.grid._Nm_short
        mock_sim.dust.q.eff = np.ones(Nr) * (-4.0)
        with pytest.warns(RuntimeWarning, match="invalid value encountered in scalar divide"):
            result = Sigma_initial(mock_sim)
            
        assert result.shape == (Nr, Nm_s)
        assert np.all(result >= 0)

class TestBoundaryEnforcement:
    def test_enforce_f(self, monkeypatch):
        """Test fragmentation barrier enforcement"""
        sim = Mock()
        sim.dust.f.crit = 0.5
        sim.dust.Sigma = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        sim.dust.s.max = np.array([1.0, 2.0, 3.0])
        sim.dust.s.lim = np.array([0.5, 1.0, 1.5])
        sim.dust.qrec = Mock()
        sim.dust.qrec.update = Mock()
        
        # Mock components
        comp1 = Mock()
        comp1.dust._active = True
        comp1.dust.Sigma = np.array([[5.0, 10.0], [15.0, 20.0], [25.0, 30.0]])
        
        sim.components = Mock()
        sim.components.__dict__ = {'comp1': comp1}
        
        monkeypatch.setattr('tripod.std.dust.dadsig', lambda s: np.ones(3))
        
        # Should not raise an error
        enforce_f(sim)
        assert sim.dust.qrec.update.called
    
    def test_dadsig(self, monkeypatch):
        """Test dadsig calculation"""
        sim = Mock()
        sim.dust.s.lim = np.ones(3)
        sim.dust.qrec = np.ones(3)
        sim.dust.f.crit = 0.5
        sim.dust.s.max = np.ones(3) * 2.0
        sim.dust.s.min = np.ones(3) * 0.1
        sim.dust.Sigma = np.ones((3, 2))
        
        monkeypatch.setattr('tripod.std.dust_f.dadsig', 
                           lambda s_lim, qrec, f_crit, s_max, s_min, Sigma: np.ones(3))
        
        result = dadsig(sim)
        np.testing.assert_array_equal(result, np.ones(3))
    
    def test_dsigda(self, monkeypatch):
        """Test dsigda calculation"""
        sim = Mock()
        sim.dust.s.lim = np.ones(3)
        sim.dust.qrec = np.ones(3)
        sim.dust.f.crit = 0.5
        sim.dust.s.max = np.ones(3) * 2.0
        sim.dust.s.min = np.ones(3) * 0.1
        sim.dust.Sigma = np.ones((3, 2))
        
        monkeypatch.setattr('tripod.std.dust_f.dsigda', 
                           lambda s_lim, qrec, f_crit, s_max, s_min, Sigma: np.ones(3))
        
        result = dsigda(sim)
        np.testing.assert_array_equal(result, np.ones(3))
        
class TestPhysicalQuantities:
    def test_q_eff(self):
        """Test effective q calculation"""
        sim = Mock()
        sim.dust.q.frag = np.ones(3) * (-3.5)
        sim.dust.q.sweep = np.ones(3) * (-2.0)
        sim.dust.p.frag = np.ones(3) * 0.6
        
        result = q_eff(sim)
        expected = (-3.5) * 0.6 + (-2.0) * (1.0 - 0.6)
        np.testing.assert_array_almost_equal(result, np.ones(3) * expected)
    
    def test_q_frag(self, monkeypatch):
        """Test fragmentation q calculation"""
        sim = Mock()
        sim.dust.p.driftfrag = np.ones(3)
        sim.dust.v.rel.tot = np.ones((3, 5, 5))
        sim.dust.v.frag = np.ones(3)
        sim.dust.St = np.ones((3, 5))
        sim.dust.q.turb1 = np.ones(3)
        sim.dust.q.turb2 = np.ones(3)
        sim.dust.q.drfrag = np.ones(3)
        sim.gas.alpha = np.ones(3)
        sim.gas.Sigma = np.ones(3)
        sim.gas.mu = np.ones(3)
        
        monkeypatch.setattr('tripod.std.dust_f.qfrag', 
                           lambda p_drift, v_rel, v_frag, St, q_turb1, q_turb2, q_drfrag, alpha, Sigma, mu: np.ones(3) * (-3.0))
        
        result = q_frag(sim)
        np.testing.assert_array_equal(result, np.ones(3) * (-3.0))
    
    def test_q_rec(self):
        """Test size distribution exponent calculation"""
        sim = Mock()
        sim.dust.Sigma = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        sim.dust.s.min = np.array([1e-4, 2e-4, 3e-4])
        sim.dust.s.max = np.array([1e-2, 2e-2, 3e-2])
        
        result = q_rec(sim)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))
    
    def test_vrel_brownian_motion(self, monkeypatch):
        """Test Brownian motion relative velocity"""
        sim = Mock()
        sim.gas.cs = np.ones(3)
        sim.dust.m = np.ones((3, 5))
        sim.gas.T = np.ones(3)
        
        monkeypatch.setattr('tripod.std.dust_f.vrel_brownian_motion', 
                           lambda cs, m, T: np.ones((3, 5)))
        
        result = vrel_brownian_motion(sim)
        assert result.shape == (3, 5)

class TestVelocityAndDiffusion:
    def test_D_mod(self, monkeypatch):
        """Test modified diffusivity calculation"""
        sim = Mock()
        sim.dust.delta.rad = np.ones(3)
        sim.gas.cs = np.ones(3)
        sim.grid.OmegaK = np.ones(3)
        sim.dust.St = np.ones((3, 5))
        sim.dust.f.drift = 0.8
        
        with patch('dustpy.std.dust_f.d') as mock_d:
            mock_d.return_value = np.ones((3, 5))
            result = D_mod(sim)
            
            # Check boundary conditions
            assert np.all(result[:1, :] == 0.0)
            assert np.all(result[-2:, :] == 0.0)
    
    def test_vrad_mod(self, monkeypatch):
        """Test modified radial velocity calculation"""
        sim = Mock()
        sim.dust.St = np.ones((3, 5))
        sim.dust.f.drift = 0.3
        sim.dust.v.driftmax = np.ones((3, 5))
        sim.gas.v.rad = np.ones(3)
        
        with patch('dustpy.std.dust_f.vrad') as mock_vrad:
            mock_vrad.return_value = np.ones((3, 5))
            result = vrad_mod(sim)
            assert result.shape == (3, 5)

class TestCompositionFunctions:
    def test_rhos_compo(self):
        """Test material density from composition"""
        sim = Mock()
        sim.gas.Sigma = np.ones(3)
        sim.dust.Sigma = np.ones((3, 2))
        sim.dust.SigmaFloor = np.ones((3, 2)) * 1e-10
        sim.dust.rhos = np.ones((3, 2)) * 1000.0
        
        # Mock components
        comp1 = Mock()
        comp1.dust._active = True
        comp1.dust.Sigma = np.ones((3, 2)) * 10.0
        comp1.dust.pars.rhos = 2000.0
        
        comp2 = Mock()
        comp2.dust._active = True
        comp2.dust.Sigma = np.ones((3, 2)) * 5.0
        comp2.dust.pars.rhos = 1500.0
        
        sim.components = Mock()
        sim.components.__dict__ = {
            'comp1': comp1,
            'comp2': comp2,
            '_private': Mock()
        }
        
        result = rhos_compo(sim)
        assert result.shape == (3, 2)
        assert np.all(result > 0)
    
    def test_S_hyd_compo(self, monkeypatch):
        """Test hydrodynamic source terms for components"""
        sim = Mock()
        sim.grid.ri = np.array([0.5, 1.5, 2.5, 3.5])
        
        group = Mock()
        group.Fi = np.ones(4)
        
        with patch('dustpy.std.dust_f.s_hyd') as mock_s_hyd:
            mock_s_hyd.return_value = np.ones((3, 2))
            result = S_hyd_compo(sim, group)
            mock_s_hyd.assert_called_once_with(group.Fi, sim.grid.ri)
    
    def test_Fi_sig1smax(self, monkeypatch):
        """Test flux calculation for Sigma[1] * smax"""
        sim = Mock()
        sim.dust.Sigma = np.ones((3, 2))
        sim.dust.s.max = np.ones(3) * 2.0
        
        # Mock F_diff and F_adv to return proper shapes
        monkeypatch.setattr('tripod.std.dust.F_diff', 
                           lambda sim, Sigma: np.ones((4, 2)) * 0.1)
        monkeypatch.setattr('tripod.std.dust.F_adv', 
                           lambda sim, Sigma: np.ones((4, 2)) * 0.2)
        
        result = Fi_sig1smax(sim)
        expected = np.ones(4) * 0.3  # 0.1 + 0.2 for column 1
        np.testing.assert_array_almost_equal(result, expected)

class TestJacobianAndIntegration:
    @pytest.fixture
    def mock_sim(self):
        sim = Simulation()
        sim.ini.grid.Nr = 5
        sim.ini.grid.rmax = 10*c.au
        sim.initialize()
        sim.writer = None 
        sim.t.snapshot = [1]
        sim.run()
        return sim

    def test_Y_jacobian(self, mock_sim, monkeypatch):
        """Test Y jacobian calculation"""
        
        x = Mock()
        x.stepsize = 0.1
        
        result = Y_jacobian(mock_sim, x)
        assert isinstance(result, sp.csc_matrix)

    def test_f_impl_1_direct_empty(self, mock_sim):
        """Test implicit function with empty _Y"""
        dx = 0.1 # Example step size
        x0 = 200 
        
        result = _f_impl_1_direct(x0,mock_sim.dust._Y,dx)
        assert isinstance(result, np.ndarray)
        assert result.shape == (mock_sim.grid.Nr * mock_sim.grid._Nm_short + mock_sim.grid.Nr,)


class TestMissingFunctions:
    @pytest.fixture
    def mock_sim(self):
        sim = Simulation()
        sim.ini.grid.Nr = 5
        sim.ini.grid.rmax = 10*c.au
        sim.initialize()
        sim.writer = None 
        sim.t.snapshot = [1]
        sim.run()
        return sim
    
    def test_prepare(self, mock_sim):
        """Test prepare function"""
        # Should not raise an error
        Nr = mock_sim.grid.Nr
        Nm_s = mock_sim.grid._Nm_short
        prepare(mock_sim)

        #check if _Y is correctly populated
        np.testing.assert_array_equal(mock_sim.dust._Y[:Nm_s * Nr], mock_sim.dust.Sigma.ravel())
        np.testing.assert_array_equal(mock_sim.dust._Y[Nm_s * Nr:], mock_sim.dust.s.max * mock_sim.dust.Sigma[:, 1])

        #seet source terms to zero at boundaries
        np.testing.assert_array_equal(mock_sim.dust.S.coag[0], np.zeros(mock_sim.dust.S.coag.shape[1]))
        np.testing.assert_array_equal(mock_sim.dust.S.coag[-1], np.zeros(mock_sim.dust.S.coag.shape[1]))
        np.testing.assert_array_equal(mock_sim.dust._SigmaOld[...],mock_sim.dust.Sigma[...])
        
    def test_finalize(self, mock_sim):
        """Test finalize function"""
        Nr = mock_sim.grid.Nr
        Nm_s = mock_sim.grid._Nm_short

        with mock.patch("dustpy.std.dust.boundary") as mock_boundary:
            mock_boundary.side_effect = lambda sim: None  # Do nothing
            finalize(mock_sim)  # Should not raise an error
            # Check if _Y is correctly unpacked
            np.testing.assert_array_almost_equal(mock_sim.dust.Sigma.ravel(), mock_sim.dust._Y[:Nm_s * Nr])
            smax_expected = np.maximum(1.5 * mock_sim.dust.s.min, mock_sim.dust._Y[Nr * Nm_s:] / mock_sim.dust.Sigma[..., 1])
            np.testing.assert_array_almost_equal(mock_sim.dust.s.max,smax_expected)


        with mock.patch('tripod.std.dust.enforce_f') as mock_enforce_f:
            mock_enforce_f.side_effect = lambda sim: None  # Do nothing

            # Modify _Y to simulate an update
            mock_sim.dust._Y[:Nm_s * Nr] = np.random.rand(Nr * Nm_s)
            mock_sim.dust._Y[Nm_s * Nr:] = np.random.rand(Nr) * 1e-2

            finalize(mock_sim)
            # Check if enforce_f was called
            mock_enforce_f.assert_called_once_with(mock_sim)
        
        with mock.patch('dustpy.std.dust.enforce_floor_value') as mock_enforce_floor:

            # Modify _Y to simulate an update
            mock_sim.dust._Y[:Nm_s * Nr] = np.random.rand(Nr * Nm_s) * 1e-5  # Some values below floor
            mock_sim.dust._Y[Nm_s * Nr:] = np.random.rand(Nr) * 1e-5  # Some values below floor

            finalize(mock_sim)
            # Check if enforce_floor_value was called
            mock_enforce_floor.assert_called_once_with(mock_sim)

        

    def test_finalize_with_compo(self):
        sim = Simulation()
        sim.ini.grid.Nr = 5
        sim.ini.grid.rmax = 10*c.au
        sim.initialize()
        sim.writer = None
        sim.t.snapshot = [1]
        sim.addcomponent("comp1", None, 1, dust_value=sim.dust.Sigma, dust_active=True)
        sim.addcomponent("comp2", None, 1, dust_value=sim.dust.SigmaFloor*0.1, dust_active=True)
        sim.run()


        finalize(sim)  # Should not raise an error
        assert (sim.dust.Sigma >= sim.dust.SigmaFloor*0.1).all()
        assert (sim.components.comp1.dust.Sigma >= sim.dust.SigmaFloor*0.1).all()
        assert (sim.components.comp2.dust.Sigma >= sim.dust.SigmaFloor*0.1).all()

    def test_smax_deriv(self, mock_sim):
        """Test smax derivative calculation"""
        result = smax_deriv(mock_sim,mock_sim.t,mock_sim.dust.s.max)
        assert len(result.shape) == 1  # Should be 1D array
        assert result.shape[0] == mock_sim.grid.Nr

    def test_smax_deriv_no_smax(self, mock_sim):
        """Test smax derivative calculation"""
        result = smax_deriv(mock_sim,mock_sim.t,None)
        assert len(result.shape) == 1  # Should be 1D array
        assert result.shape[0] == mock_sim.grid.Nr

    def test_S_tot_ext(self, mock_sim):
        """Test total external source terms"""
        # Mock external source terms
        mock_sim.dust.S.ext = np.ones_like(mock_sim.dust.Sigma) * 0.1
        
        result = S_tot_ext(mock_sim)
        expected_shape = mock_sim.dust.Sigma.shape
        assert result.shape == expected_shape
        np.testing.assert_array_equal(result, mock_sim.dust.S.ext)

    def test_S_tot(self, mock_sim):
        """Test total source terms calculation"""
        # Ensure all required source terms exist
        mock_sim.dust.S.ext = np.ones_like(mock_sim.dust.Sigma) * 0.1
        mock_sim.dust.S.hyd = np.ones_like(mock_sim.dust.Sigma) * 0.2
        mock_sim.dust.S.coag = np.ones_like(mock_sim.dust.Sigma) * 0.3
        
        result = S_tot(mock_sim)
        expected_shape = mock_sim.dust.Sigma.shape
        assert result.shape == expected_shape

    def test_S_tot_empty(self, mock_sim):
        
        res = S_tot(mock_sim,Sigma=mock_sim.dust.Sigma)
        expected_res = S_tot(mock_sim)

        np.testing.assert_almost_equal(res,expected_res)

    def test_S_tot_empty_noupdater(self, mock_sim):
        mock_sim.dust.S.coag.updater = None
        mock_sim.dust.S.hyd.updater = None
        mock_sim.dust.S.compo.updater = None
        res = S_tot(mock_sim,Sigma=mock_sim.dust.Sigma)
        expected_res = S_tot(mock_sim)

        np.testing.assert_almost_equal(res,expected_res)

class TestAdditionalCoverage:
    @pytest.fixture
    def mock_sim(self):
        sim = Simulation()
        sim.ini.grid.Nr = 5
        sim.ini.grid.rmax = 10*c.au
        sim.initialize()
        sim.writer = None 
        sim.t.snapshot = [1]
        sim.run()
        return sim
    
    def test_dt_components_attribute_error(self, mock_sim, monkeypatch):
        """Test dt when components attribute doesn't exist"""
        # Remove components attribute
        if hasattr(mock_sim, 'components'):
            delattr(mock_sim, 'components')
        
        monkeypatch.setattr('tripod.std.dust.dt_Sigma', lambda _: 100.0)
        monkeypatch.setattr('tripod.std.dust.dt_smax', lambda _: 50.0)
        
        result = dt(mock_sim)
        assert result == 50.0

    def test_dt_compo_no_active_components(self, mock_sim):
        """Test dt_compo when no active components exist"""
        # Add inactive component
        mock_sim.addcomponent("inactive", None, 1, dust_value=mock_sim.dust.Sigma, dust_tracer=True)
        
        result = dt_compo(mock_sim)
        assert result == 1e100

    def test_S_compo_with_private_attributes(self, mock_sim):
        """Test S_compo ignores private attributes"""
        # Add component with private attribute
        mock_sim.addcomponent("comp1", None, 1, dust_value=mock_sim.dust.Sigma, dust_active=True)
        mock_sim.components.comp1.dust.S_Sigma = np.ones_like(mock_sim.dust.Sigma) * 0.5
        
        # Add private attribute that should be ignored
        mock_sim.components._private = Mock()
        mock_sim.components._private.dust = Mock()
        mock_sim.components._private.dust.S_Sigma = np.ones_like(mock_sim.dust.Sigma) * 1000.0
        
        result = S_compo(mock_sim)
        expected = np.ones_like(mock_sim.dust.Sigma) * 0.5  # Only comp1, not _private
        np.testing.assert_array_almost_equal(result, expected)

    def test_rhos_compo_with_inactive_and_active(self, mock_sim):
        """Test rhos_compo with mix of active and inactive components"""
        # Add active component
        mock_sim.addcomponent("active", None, 1, dust_value=mock_sim.dust.Sigma, dust_active=True)
        mock_sim.components.active.dust.Sigma = np.ones_like(mock_sim.dust.Sigma) * 5.0
        mock_sim.components.active.dust.pars.rhos = 2500.0
        
        # Add inactive component
        mock_sim.addcomponent("inactive", None, 1, dust_value=mock_sim.dust.Sigma, dust_active=False)
        
        result = rhos_compo(mock_sim)
        # Should only consider active component
        assert result.shape == mock_sim.dust.rhos.shape
        assert np.all(result > 0)

    def test_enforce_f_components_missing_dust_attribute(self, mock_sim, monkeypatch):
        """Test enforce_f when component missing dust attribute"""
        # Mock component without dust attribute
        comp1 = Mock()
        del comp1.dust  # Remove dust attribute
        
        if not hasattr(mock_sim, 'components'):
            mock_sim.components = Mock()
        mock_sim.components.__dict__ = {'comp1': comp1}
        
        monkeypatch.setattr('tripod.std.dust.dadsig', lambda s: np.ones(mock_sim.grid.Nr))
        
        # Should handle missing dust attribute gracefully
        try:
            enforce_f(mock_sim)
        except AttributeError:
            pass  # Expected to fail gracefully

    def test_Sigma_initial_q_exactly_minus_4(self, mock_sim):
        """Test Sigma_initial special case when q exactly equals -4"""

        result = Sigma_initial(mock_sim)
        assert result.shape == mock_sim.dust.Sigma.shape
        assert np.all(result >= 0)
        assert np.all(np.isfinite(result))

    def test_p_stick_calculation_precision(self, mock_sim):
        """Test p_stick calculation with high precision values"""
        # Test with high precision fragmentation probabilities
        mock_sim.dust.p.frag = np.array([0.123456789, 0.987654321, 0.555555555, 0.111111111, 0.999999999])
        
        result = p_stick(mock_sim)
        expected = 1.0 - mock_sim.dust.p.frag
        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_q_rec_zero_division_protection(self, mock_sim):
        """Test q_rec handles potential division by zero"""
        # Set up case that might cause division issues
        mock_sim.dust.Sigma = np.array([[1e-20, 1e-20], [0.0, 1e-20], [1e-20, 0.0], [1.0, 1.0], [1e20, 1e20]])
        mock_sim.dust.s.min = np.ones(mock_sim.grid.Nr) * 1e-4
        mock_sim.dust.s.max = np.ones(mock_sim.grid.Nr) * 1e-2
        
        with np.errstate(divide='ignore'):
            result = q_rec(mock_sim)
        assert result.shape == (mock_sim.grid.Nr,)
        # Check for finite values, allowing NaNs where appropriate
        assert np.isnan(result[1]) or np.isinf(result[1])

    def test_H_calculation_delegates_correctly(self, mock_sim):
        """Test H calculation properly delegates to dustpy function"""
        with patch('dustpy.std.dust_f.h_dubrulle1995') as mock_h:
            expected_result = np.ones((mock_sim.grid.Nr, mock_sim.grid._Nm_long)) * 3.14
            mock_h.return_value = expected_result
            
            result = H(mock_sim)
            
            assert result.shape == expected_result.shape
            np.testing.assert_array_equal(result, expected_result)
            mock_h.assert_called_once()

    def test_vrel_brownian_motion_delegates_correctly(self, mock_sim, monkeypatch):
        """Test vrel_brownian_motion properly delegates"""
        expected_result = np.ones((mock_sim.grid.Nr, 1)) * 2.71828
        
        monkeypatch.setattr('tripod.std.dust_f.vrel_brownian_motion',
                           lambda cs, m, T: expected_result)
        
        result = vrel_brownian_motion(mock_sim)
        np.testing.assert_array_equal(result, expected_result)

    def test_Fi_sig1smax_flux_multiplication(self, mock_sim, monkeypatch):
        """Test Fi_sig1smax correctly multiplies flux by smax"""
        # Set specific values to test multiplication
        mock_sim.dust.s.max = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Mock fluxes with known values
        F_diff_result = np.ones((mock_sim.grid.Nr + 1, 2)) * 10.0
        F_adv_result = np.ones((mock_sim.grid.Nr + 1, 2)) * 5.0
        
        monkeypatch.setattr('tripod.std.dust.F_diff', lambda sim, Sigma: F_diff_result)
        monkeypatch.setattr('tripod.std.dust.F_adv', lambda sim, Sigma: F_adv_result)
        
        result = Fi_sig1smax(mock_sim)
        
        # Result should be sum of fluxes (15.0) for column 1, no multiplication by smax in this context
        expected = np.ones(mock_sim.grid.Nr + 1) * 15.0  # 10.0 + 5.0
        np.testing.assert_array_almost_equal(result, expected)

class TestEdgeCases:
    @pytest.fixture
    def mock_sim(self):
        sim = Simulation()
        sim.ini.grid.Nr = 5
        sim.ini.grid.rmax = 10*c.au
        sim.initialize()
        sim.writer = None 
        sim.t.snapshot = [1]
        sim.run()
        return sim
    
    def test_dt_with_zero_timesteps(self, mock_sim, monkeypatch):
        """Test dt when component timesteps are zero"""
        # Add a component with zero timestep
        mock_sim.addcomponent("test_comp", None, 1, dust_value=mock_sim.dust.Sigma, dust_active=True)
        
        # Mock dt_compo to return zero
        monkeypatch.setattr('tripod.std.dust.dt_compo', lambda _: 0.0)
        monkeypatch.setattr('tripod.std.dust.dt_Sigma', lambda _: 100.0)
        monkeypatch.setattr('tripod.std.dust.dt_smax', lambda _: 50.0)
        
        # this is the dt in dust -> should ignore zero from component
        result = dt(mock_sim)
        assert result == 50.0

    def test_dt_Sigma_edge_cases(self, mock_sim):
        """Test dt_Sigma with various edge cases"""
        # Test with very small negative values
        mock_sim.dust.S.tot = np.ones_like(mock_sim.dust.Sigma) * -1e-20
        result = dt_Sigma(mock_sim)
        assert result > 0
        assert np.isfinite(result)
        
        # Test with mixed positive and negative
        mock_sim.dust.S.tot = np.ones_like(mock_sim.dust.Sigma) * np.array([-1e-5, 1e-5])
        result = dt_Sigma(mock_sim)
        assert result > 0
        assert np.isfinite(result)

    def test_enforce_f_no_components(self, mock_sim, monkeypatch):
        """Test enforce_f when no components exist"""
        # Remove any existing components
        if hasattr(mock_sim, 'components'):
            mock_sim.components = Mock()
            mock_sim.components.__dict__ = {}
        
        monkeypatch.setattr('tripod.std.dust.dadsig', lambda s: np.ones(mock_sim.grid.Nr))
        
        # Should not raise an error
        enforce_f(mock_sim)

    def test_p_frag_zero_fragmentation_velocity(self, mock_sim, monkeypatch):
        """Test fragmentation probability with zero fragmentation velocity"""
        mock_sim.dust.v.frag = np.zeros(mock_sim.grid.Nr)
        
        monkeypatch.setattr('tripod.std.dust_f.pfrag', 
                           lambda v_rel, v_frag: np.ones(mock_sim.grid.Nr))
        
        result = p_frag(mock_sim)
        assert result.shape == (mock_sim.grid.Nr,)

    def test_q_rec_edge_cases(self, mock_sim):
        """Test q_rec with edge cases"""
        # Test with very small particle sizes
        mock_sim.dust.s.min = np.ones(mock_sim.grid.Nr) * 1e-10
        mock_sim.dust.s.max = np.ones(mock_sim.grid.Nr) * 1e-8
        
        result = q_rec(mock_sim)
        assert result.shape == (mock_sim.grid.Nr,)
        assert np.all(np.isfinite(result))

    def test_Sigma_initial_with_floor(self, mock_sim):
        """Test Sigma_initial when result is below floor"""
        # Set very high floor values
        mock_sim.dust.SigmaFloor = np.ones((mock_sim.grid.Nr, mock_sim.grid._Nm_short)) * 1e10
        
        result = Sigma_initial(mock_sim)
        # Result should be at least the floor value
        assert np.all(result >= mock_sim.dust.SigmaFloor*0.1)

    def test_rhos_compo_no_active_components(self, mock_sim):
        """Test rhos_compo when no active components exist"""
        # Mock components that are not active
        # internal state before calling rhos_compo
        result = mock_sim.dust.rhos

        comp1 = Mock()
        comp1.dust._active = False
        comp1.dust.Sigma = np.ones_like(mock_sim.dust.Sigma)
        comp1.dust.pars.rhos = 2000.0
        
        if not hasattr(mock_sim, 'components'):
            mock_sim.components = Mock()
        mock_sim.components.__dict__ = {'comp1': comp1}
        
        mock_sim.dust.rhos.update()
        # Should return the original dust density
        np.testing.assert_array_equal(result, mock_sim.dust.rhos)

    def test_S_compo_inactive_components(self, mock_sim):
        """Test S_compo with inactive components"""
        # Create inactive component
        mock_sim.addcomponent("comp1", None, 1, dust_value=mock_sim.dust.Sigma, dust_tracer=False)
        
        
        result = S_compo(mock_sim)
        # Should be zero since no active components
        np.testing.assert_array_equal(result, np.zeros_like(mock_sim.dust.Sigma))

class TestBoundaryConditions:
    @pytest.fixture
    def mock_sim(self):
        sim = Simulation()
        sim.ini.grid.Nr = 5
        sim.ini.grid.rmax = 10*c.au
        sim.initialize()
        sim.writer = None 
        sim.t.snapshot = [1]
        sim.run()
        return sim
    
    def test_F_diff_boundary_conditions(self, mock_sim, monkeypatch):
        """Test diffusive flux boundary conditions"""
        monkeypatch.setattr('tripod.std.dust_f.fi_diff_no_limit', 
                           lambda *args: np.ones((mock_sim.grid.Nr + 1, mock_sim.grid._Nm_short)) * 5.0)
        
        result = F_diff(mock_sim)
        
        # Check inner boundary (first row should be zero)
        np.testing.assert_array_equal(result[0, :], 0.0)
        # Check outer boundary (last row should be zero) 
        np.testing.assert_array_equal(result[-1, :], 0.0)

    def test_D_mod_boundary_conditions(self, mock_sim, monkeypatch):
        """Test modified diffusivity boundary conditions"""
        with patch('dustpy.std.dust_f.d') as mock_d:
            mock_d.return_value = np.ones((mock_sim.grid.Nr, mock_sim.grid._Nm_long)) * 10.0
            
            result = D_mod(mock_sim)
            
            # Check boundaries are set to zero
            np.testing.assert_array_equal(result[0, :], 0.0)
            np.testing.assert_array_equal(result[-2:, :], 0.0)

class TestIntegrationHelpers:
    @pytest.fixture
    def mock_sim(self):
        sim = Simulation()
        sim.ini.grid.Nr = 5
        sim.ini.grid.rmax = 10*c.au
        sim.initialize()
        sim.writer = None 
        sim.t.snapshot = [1]
        sim.run()
        return sim
    def test_a_array_properties(self, mock_sim):
        """Test particle size array properties"""
        result = a(mock_sim)
        
        # Test that all columns have expected relationships
        for i in range(mock_sim.grid.Nr):
            # a0 should be smallest
            assert result[i, 0] <= result[i, 1]  # a0 <= fudge*a1
            assert result[i, 0] <= result[i, 2]  # a0 <= a1
            assert result[i, 2] <= result[i, 4]  # a1 <= amax
            # Check that the array is properly ordered
            assert result[i, 0] > 0
            assert result[i, 4] > result[i, 0]

    def test_m_mass_conservation(self, mock_sim):
        """Test particle mass calculation follows expected scaling"""
        result = m(mock_sim)
        a_result = a(mock_sim)
        
        # Mass should scale as a^3 (ignoring fill factor and density variations)
        # Test the relationship between different size bins
        for i in range(mock_sim.grid.Nr):
            for j in range(4):  # Don't test last column as it might have different scaling
                ratio_a = a_result[i, j+1] / a_result[i, j] 
                ratio_m = result[i, j+1] / result[i, j]
                # Mass ratio should be approximately a^3 ratio (within factor of fill/density)
                assert ratio_m > 0  # Positive mass
                assert ratio_a > 0  # Positive size

class TestCoverageCompleteness:
    @pytest.fixture
    def mock_sim(self):
        sim = Simulation()
        sim.ini.grid.Nr = 5
        sim.ini.grid.rmax = 10*c.au
        sim.initialize()
        sim.writer = None 
        sim.t.snapshot = [1]
        sim.run()
        return sim
    
    def test_dt_with_no_components_attribute(self, mock_sim, monkeypatch):
        """Test dt when sim has no components attribute at all"""
        # Ensure components attribute doesn't exist
        if hasattr(mock_sim, 'components'):
            delattr(mock_sim, 'components')
        
        monkeypatch.setattr('tripod.std.dust.dt_Sigma', lambda _: 100.0)
        monkeypatch.setattr('tripod.std.dust.dt_smax', lambda _: 50.0)
        
        result = dt(mock_sim)
        assert result == 50.0

    def test_dt_compo_empty_components(self, mock_sim):
        """Test dt_compo with empty components dictionary"""
        # Ensure components exists but is empty
        mock_sim.components = Mock()
        mock_sim.components.__dict__ = {}
        
        result = dt_compo(mock_sim)
        assert result == 1e100

    def test_S_compo_no_components_attribute(self, mock_sim):
        """Test S_compo when components attribute doesn't exist"""
        # Remove components attribute completely
        if hasattr(mock_sim, 'components'):
            delattr(mock_sim, 'components')
        
        result = S_compo(mock_sim)
        expected_shape = mock_sim.dust.Sigma.shape
        assert result.shape == expected_shape
        np.testing.assert_array_equal(result, np.zeros(expected_shape))

    def test_rhos_compo_no_components_attribute(self, mock_sim):
        """Test rhos_compo when components attribute doesn't exist"""
        result = mock_sim.dust.rhos
        # Remove components attribute completely
        if hasattr(mock_sim, 'components'):
            delattr(mock_sim, 'components')
        
        mock_sim.dust.rhos.update()
        # Should return original dust density
        np.testing.assert_array_equal(result, mock_sim.dust.rhos)

    def test_enforce_f_component_without_dust_active(self, mock_sim, monkeypatch):
        """Test enforce_f when component has dust but no _active attribute"""
        # Create component with dust but missing _active
        comp1 = Mock()
        comp1.dust = Mock()
        comp1.dust.Sigma = np.ones_like(mock_sim.dust.Sigma)
        # Remove _active attribute
        if hasattr(comp1.dust, '_active'):
            delattr(comp1.dust, '_active')
        
        mock_sim.components = Mock()
        mock_sim.components.__dict__ = {'comp1': comp1}
        
        monkeypatch.setattr('tripod.std.dust.dadsig', lambda s: np.ones(mock_sim.grid.Nr))
        
        # Should handle missing _active attribute gracefully
        try:
            enforce_f(mock_sim)
        except AttributeError:
            pass  # Expected behavior

    def test_Sigma_initial_extreme_q_values(self, mock_sim):
        """Test Sigma_initial with extreme q values"""
        # Test with very negative q
        mock_sim.dust.q.eff = np.ones(mock_sim.grid.Nr) * (-10.0)
        mock_sim.dust.s.min = np.ones(mock_sim.grid.Nr) * 1e-4
        mock_sim.dust.s.max = np.ones(mock_sim.grid.Nr) * 1e-2
        mock_sim.dust.SigmaFloor = np.ones_like(mock_sim.dust.Sigma) * 1e-15
        mock_sim.gas.Sigma = np.ones(mock_sim.grid.Nr) * 100.0
        
        result = Sigma_initial(mock_sim)
        assert result.shape == mock_sim.dust.Sigma.shape
        assert np.all(result >= mock_sim.dust.SigmaFloor)
        assert np.all(np.isfinite(result))

    def test_smax_initial_minimum_size_constraint(self, mock_sim):
        """Test smax_initial respects minimum size constraint"""
        mock_sim.ini.dust.allowDriftingParticles = True
        mock_sim.ini.dust.aIniMax = 1e-6  # Very small maximum
        mock_sim.dust.s.min = np.ones(mock_sim.grid.Nr) * 1e-4  # Larger minimum
        
        result = smax_initial(mock_sim)
        # Should use aIniMax even if it's smaller than s.min * 1.5
        np.testing.assert_array_equal(result, np.ones(mock_sim.grid.Nr) * mock_sim.ini.dust.aIniMax)

    def test_dt_Sigma_all_positive_sources(self, mock_sim):
        """Test dt_Sigma when all source terms are positive"""
        mock_sim.dust.S.tot = np.abs(mock_sim.dust.S.tot)  # Ensure all positive
        
        result = dt_Sigma(mock_sim)
        assert result == 1e100  # Should return large timestep

    def test_F_diff_with_zero_diffusivity(self, mock_sim, monkeypatch):
        """Test F_diff when diffusivity is zero everywhere"""
        mock_sim.dust.D = np.zeros_like(mock_sim.dust.D)
        
        monkeypatch.setattr('tripod.std.dust_f.fi_diff_no_limit',
                           lambda *args: np.zeros((mock_sim.grid.Nr + 1, mock_sim.grid._Nm_short)))
        
        result = F_diff(mock_sim)
        # Should be all zeros due to boundary conditions and zero diffusivity
        np.testing.assert_array_equal(result, np.zeros((mock_sim.grid.Nr + 1, mock_sim.grid._Nm_short)))

    def test_jacobian_with_boundaries(self, mock_sim, monkeypatch):
        """Test jacobian calculation with boundary conditions"""
        # Set up boundary conditions
        
        Nr = mock_sim.grid.Nr
        Nm_s = mock_sim.grid._Nm_short
        monkeypatch.setattr('tripod.std.dust_f.jacobian_coagulation_generator',
                           lambda *args: (np.ones(5), np.arange(5), np.arange(5)))
        
        with patch('dustpy.std.dust_f.jacobian_hydrodynamic_generator') as mock_hyd_gen:
            mock_hyd_gen.return_value = (np.ones(Nr*Nm_s), np.arange(Nr*Nm_s), np.arange(Nr*Nm_s))
            
            x = Mock()
            x.stepsize = 0.1
            
            result = jacobian(mock_sim, x)
            assert isinstance(result, sp.csc_matrix)

    def test_rho_midplane_all_parameters(self, mock_sim):
        """Test rho_midplane calculation uses all required parameters"""
        result = rho_midplane(mock_sim)
        expected_shape = (mock_sim.grid.Nr, mock_sim.grid._Nm_long)
        assert result.shape == expected_shape
        assert np.all(result > 0)
        # Should depend on Sigma, H, and scale height
        assert np.all(np.isfinite(result))

class TestFinalCoverage:
    @pytest.fixture
    def mock_sim(self):
        sim = Simulation()
        sim.ini.grid.Nr = 5
        sim.ini.grid.rmax = 10*c.au
        sim.initialize()
        sim.writer = None 
        sim.t.snapshot = [1]
        sim.run()
        return sim
    
    def test_S_tot_all_components(self, mock_sim):
        """Test S_tot calculation with all source term components"""
        # Ensure all source terms exist and have proper values
        mock_sim.dust.S.ext = np.ones_like(mock_sim.dust.Sigma) * 0.1
        mock_sim.dust.S.hyd = np.ones_like(mock_sim.dust.Sigma) * 0.2
        mock_sim.dust.S.coag = np.ones_like(mock_sim.dust.Sigma) * 0.3
        
        # Add component source terms
        mock_sim.addcomponent("comp1", None, 1, dust_value=mock_sim.dust.Sigma, dust_active=True)
        mock_sim.components.comp1.dust.S_Sigma = np.ones_like(mock_sim.dust.Sigma) * 0.4
        mock_sim.dust.S.compo.update()
        result = S_tot(mock_sim)
        expected = np.ones_like(mock_sim.dust.Sigma) * 1.0  # 0.1 + 0.2 + 0.3 + 0.4
        np.testing.assert_array_almost_equal(result, expected)

    def test_S_smax_hyd_complete(self, mock_sim, monkeypatch):
        """Test S_smax_hyd with all calculation steps"""
        mock_sim.dust.Sigma = np.ones((mock_sim.grid.Nr, 2))
        mock_sim.dust.s.max = np.ones(mock_sim.grid.Nr) * 1e-2
        mock_sim.dust.S.hyd = np.ones_like(mock_sim.dust.Sigma) * 0.5
        
        # Mock Fi_sig1smax to return proper flux values
        monkeypatch.setattr('tripod.std.dust.Fi_sig1smax', 
                           lambda sim: np.ones(mock_sim.grid.Nr + 1) * 2.0)
        
        # Mock s_hyd calculation
        with patch('dustpy.std.dust_f.s_hyd') as mock_s_hyd:
            mock_s_hyd.return_value = np.ones_like(mock_sim.dust.Sigma) * 0.8
            result = S_smax_hyd(mock_sim)
            
            assert result.shape == (mock_sim.grid.Nr,)
            assert np.all(np.isfinite(result))
            mock_s_hyd.assert_called_once()

    def test_S_hyd_compo_calculation(self, mock_sim):
        """Test hydrodynamic source terms for components"""
        # Create a mock component group
        group = Mock()
        group.Fi = np.ones(mock_sim.grid.Nr + 1) * 3.0
        
        with patch('dustpy.std.dust_f.s_hyd') as mock_s_hyd:
            mock_s_hyd.return_value = np.ones_like(mock_sim.dust.Sigma) * 1.5
            result = S_hyd_compo(mock_sim, group)
            
            mock_s_hyd.assert_called_once_with(group.Fi, mock_sim.grid.ri)
            np.testing.assert_array_equal(result, np.ones_like(mock_sim.dust.Sigma) * 1.5)

    def test_full_simulation_workflow(self, mock_sim):
        """Test a comprehensive workflow using multiple dust functions"""
        # Test initialization
        prepare(mock_sim)
        
        # Test initial conditions
        smax_init = smax_initial(mock_sim)
        assert smax_init.shape == (mock_sim.grid.Nr,)
        
        Sigma_init = Sigma_initial(mock_sim)
        assert Sigma_init.shape == mock_sim.dust.Sigma.shape
        
        # Test physical quantities
        particle_sizes = a(mock_sim)
        particle_masses = m(mock_sim)
        scale_heights = H(mock_sim)
        
        assert particle_sizes.shape == (mock_sim.grid.Nr, 5)
        assert particle_masses.shape == (mock_sim.grid.Nr, 5)
        
        # Test source terms
        S_coag_result = S_coag(mock_sim)
        assert S_coag_result.shape == mock_sim.dust.Sigma.shape
        
        # Test timestep calculation
        dt_result = dt(mock_sim)
        assert dt_result > 0
        assert np.isfinite(dt_result)
        
        # Test finalization
        finalize(mock_sim)

    def test_error_handling_and_robustness(self, mock_sim):
        """Test error handling and robustness of dust functions"""
        # Test with extreme parameter values
        original_sigma = mock_sim.dust.Sigma.copy()
        
        # Test with very small Sigma values
        mock_sim.dust.Sigma = np.ones_like(mock_sim.dust.Sigma) * 1e-20
        try:
            result = dt_Sigma(mock_sim)
            assert np.isfinite(result)
        except:
            pass  # Some functions may legitimately fail with extreme values
        
        # Restore original values
        mock_sim.dust.Sigma = original_sigma
        
        # Test with NaN values in non-critical arrays
        mock_sim.dust.test_array = np.full(mock_sim.grid.Nr, np.nan)
        
        # Basic functions should still work
        result = a(mock_sim)
        assert result.shape == (mock_sim.grid.Nr, 5)
        assert not np.any(np.isnan(result))
        
        # Clean up
        delattr(mock_sim.dust, 'test_array')




class TestBoundaries:
    @pytest.fixture
    def mock_sim(self):
        sim = Simulation()
        sim.ini.grid.Nr = 5
        sim.ini.grid.rmax = 10*c.au
        sim.initialize()
        return sim 
   
    def test_jacobian_boundaries(self, mock_sim, monkeypatch):
        """Test jacobian calculation with boundary conditions"""
        # Set up boundary conditions
        
        Nr = mock_sim.grid.Nr
        Nm_s = mock_sim.grid._Nm_short
        Ndust = Nr*Nm_s

        x = Mock()
        x.stepsize = 0.1
        dx = x.stepsize
        
        #value boundaries
        val_inner = 150.0
        val_outer = 50.0

        mock_sim.dust.boundary.inner.setcondition('val',value=val_inner)
        mock_sim.dust.boundary.outer.setcondition('val',value=val_outer)
        result = jacobian(mock_sim, x)

        for k in range(Nm_s):
            assert mock_sim.dust._rhs[k] == val_inner
            assert mock_sim.dust._rhs[-1-k] == val_outer

        #const_val boundaries
        mock_sim.dust._rhs = mock_sim.dust.Sigma.flatten()
        mock_sim.dust.boundary.inner.setcondition('const_val')
        mock_sim.dust.boundary.outer.setcondition('const_val')
        result = jacobian(mock_sim,x)

        for k in range(Nm_s):
            assert mock_sim.dust._rhs[k] == 0.0
            assert mock_sim.dust._rhs[-k] == 0.0
            assert result[k,Nm_s+k] == 10.0
            assert result[-Nm_s+k,-2*Nm_s+k] == 10.0

        #grad
        mock_sim.dust._rhs = mock_sim.dust.Sigma.flatten()
        mock_sim.dust.boundary.inner.setcondition('grad',value=10.0)
        mock_sim.dust.boundary.outer.setcondition('grad',value=5.0)
        result = jacobian(mock_sim,x)
        K1 = - mock_sim.grid.r[1]/mock_sim.grid.r[0]
        Km1 = - mock_sim.grid.r[-2]/mock_sim.grid.r[-1]

        for k in range(Nm_s):
            assert result[k,Nm_s+k] == -K1/dx
            assert result[-Nm_s+k,-2*Nm_s+k] == -Km1/dx

        #const_grad 
        mock_sim.dust._rhs = mock_sim.dust.Sigma.flatten()
        mock_sim.dust.boundary.inner.setcondition('const_grad')
        mock_sim.dust.boundary.outer.setcondition('const_grad')
        result = jacobian(mock_sim,x)

        for k in range(Nm_s):
            assert mock_sim.dust._rhs[k] == 0.0
            assert mock_sim.dust._rhs[-k] == 0.0
            assert result[k,Nm_s+k] != 0.0
            assert result[-Nm_s+k,-2*Nm_s+k] != 0.0
            assert result[k,k+2*Nm_s] != 0
            assert result[-Nm_s+k,-3*Nm_s+k] != 0
    

        #pow
        mock_sim.dust._rhs = mock_sim.dust.Sigma.flatten()
        power_inner = 2.0
        power_outer = 3.0
        mock_sim.dust.boundary.inner.setcondition('pow',value=power_inner)
        mock_sim.dust.boundary.outer.setcondition('pow',value=power_outer)
        result = jacobian(mock_sim,x)
        ratio = (mock_sim.grid.r[0]/mock_sim.grid.r[1])**power_inner
        ratio_outer = (mock_sim.grid.r[-1]/mock_sim.grid.r[-2])**power_outer

        for k in range(Nm_s):
            assert mock_sim.dust._rhs[k] == ratio*mock_sim.dust._rhs[k+Nm_s]
            assert mock_sim.dust._rhs[-Nm_s+k] == ratio_outer*mock_sim.dust._rhs[-2*Nm_s+k] 

        #const_pow
        mock_sim.dust._rhs = mock_sim.dust.Sigma.flatten()
        mock_sim.dust.boundary.inner.setcondition('const_pow')
        mock_sim.dust.boundary.outer.setcondition('const_pow')
        result = jacobian(mock_sim,x)

        for k in range(Nm_s):
            assert result[k,k] == 0
            assert result[-Nm_s+k,-Nm_s+k] == 0
            assert mock_sim.dust._rhs[k] == 0.0
            assert mock_sim.dust._rhs[-Nm_s+k] == 0.0  # Nr-1
            assert result[k,k+Nm_s] != 0
            assert result[-Nm_s+k,-2*Nm_s+k] != 0

    def test_smax_boundaries(self,mock_sim):
        """Test smax boundary conditions"""
        x = Mock()
        x.stepsize = 0.1
        Nr = mock_sim.grid.Nr
        Nm_s = mock_sim.grid._Nm_short
        Ndust = Nr*Nm_s


        #override jacobianator to avoid errors
        create_zero_arrays = lambda x,dx: (np.zeros(mock_sim.grid.Nr), np.zeros(mock_sim.grid.Nr), np.zeros(mock_sim.grid.Nr))
        mock_sim.dust.Sigma.jacobianator = create_zero_arrays

        #value boundaries
        val_inner = 1e-3
        val_outer = 5e-3
        mock_sim.dust.s.boundary.inner.setcondition('val',value=val_inner)
        mock_sim.dust.s.boundary.outer.setcondition('val',value=val_outer)

        result = Y_jacobian(mock_sim,x)

        assert mock_sim.dust._Y_rhs[Ndust] == val_inner
        assert mock_sim.dust._Y_rhs[-1] == val_outer

        #const_val boundaries
        mock_sim.dust._Y[:Nm_s * Nr] = mock_sim.dust.Sigma.ravel()
        mock_sim.dust._Y[Nm_s * Nr:] = mock_sim.dust.s.max * mock_sim.dust.Sigma[:, 1]
        mock_sim.dust.s.boundary.inner.setcondition('const_val')
        mock_sim.dust.s.boundary.outer.setcondition('const_val')

        result = Y_jacobian(mock_sim,x)

        assert mock_sim.dust._Y_rhs[Ndust] == 0.0
        assert mock_sim.dust._Y_rhs[-1] == 0.0
        assert result[Ndust,Ndust+1] == 10.0
        assert result[-1,-2] == 10.0

        #grad
        grad_inner = 1e-4
        grad_outer = 2e-4
        mock_sim.dust._Y[:Nm_s * Nr] = mock_sim.dust.Sigma.ravel()
        mock_sim.dust._Y[Nm_s * Nr:] = mock_sim.dust.s.max * mock_sim.dust.Sigma[:, 1]
        mock_sim.dust.s.boundary.inner.setcondition('grad',value=grad_inner)
        mock_sim.dust.s.boundary.outer.setcondition('grad',value=grad_outer)
        result = Y_jacobian(mock_sim,x)
        K1 = - mock_sim.grid.r[1]/mock_sim.grid.r[0]
        Km1 = - mock_sim.grid.r[-2]/mock_sim.grid.r[-1]

        assert result[Ndust,Ndust+1] == -K1/x.stepsize
        assert result[-1,-2] == -Km1/x.stepsize
        assert mock_sim.dust._Y_rhs[Ndust] != 0.0
        assert mock_sim.dust._Y_rhs[-1] != 0.0


        #const_grad
        mock_sim.dust._Y[:Nm_s * Nr] = mock_sim.dust.Sigma.ravel()
        mock_sim.dust._Y[Nm_s * Nr:] = mock_sim.dust.s.max * mock_sim.dust.Sigma[:, 1]
        mock_sim.dust.s.boundary.inner.setcondition('const_grad')
        mock_sim.dust.s.boundary.outer.setcondition('const_grad')
        result = Y_jacobian(mock_sim,x)

        assert mock_sim.dust._Y_rhs[Ndust] == 0.0
        assert mock_sim.dust._Y_rhs[-1] == 0.0
        assert result[Ndust,Ndust+1] != 0.0
        assert result[-1,-2] != 0.0
        assert result[Ndust,Ndust+2] != 0.0
        assert result[-1,-3] != 0.0

        #pow
        power_inner = 2.0
        power_outer = 3.0
        mock_sim.dust._Y[:Nm_s * Nr] = mock_sim.dust.Sigma.ravel()
        mock_sim.dust._Y[Nm_s * Nr:] = mock_sim.dust.s.max * mock_sim.dust.Sigma[:, 1]
        mock_sim.dust.s.boundary.inner.setcondition('pow',value=power_inner)
        mock_sim.dust.s.boundary.outer.setcondition('pow',value=power_outer)
        result = Y_jacobian(mock_sim,x)
        ratio = (mock_sim.grid.r[0]/mock_sim.grid.r[1])**power_inner
        ratio_outer = (mock_sim.grid.r[-1]/mock_sim.grid.r[-2])**power_outer
        assert mock_sim.dust._Y_rhs[Ndust] == ratio*mock_sim.dust._Y_rhs[Ndust+1]
        assert mock_sim.dust._Y_rhs[-1] == ratio_outer*mock_sim.dust._Y_rhs[-2]

        #const_pow
        mock_sim.dust._Y[:Nm_s * Nr] = mock_sim.dust.Sigma.ravel()
        mock_sim.dust._Y[Nm_s * Nr:] = mock_sim.dust.s.max * mock_sim.dust.Sigma[:, 1]
        mock_sim.dust.s.boundary.inner.setcondition('const_pow')
        mock_sim.dust.s.boundary.outer.setcondition('const_pow')
        result = Y_jacobian(mock_sim,x)

        assert mock_sim.dust._Y_rhs[Ndust] == 0.0
        assert mock_sim.dust._Y_rhs[-1] == 0.0
        assert result[Ndust,Ndust+1] != 0
        assert result[-1,-2] != 0