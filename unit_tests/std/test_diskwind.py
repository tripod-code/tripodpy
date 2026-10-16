import warnings

import numpy as np
import pytest
import dustpy as dp
import dustpy.constants as c

from tripodpy import Simulation
from tripodpy.std import compo
from tripodpy.std import dust as tridust
from tripodpy.std import gas as trigas
from tripodpy.utils import split_alpha


def make_sim(alpha_dw=0., Lambda=None, Nr=20, rmin=1., rmax=100.):
    sim = Simulation()
    sim.ini.grid.Nr = Nr
    sim.ini.grid.rmin = rmin * c.au
    sim.ini.grid.rmax = rmax * c.au
    sim.ini.gas.alphaDiskwind = alpha_dw
    if Lambda is not None:
        sim.ini.gas.LambdaDiskwind = Lambda
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.initialize()
    sim.gas.v.update()
    sim.dust.v.rad.update()
    return sim


class TestSplitAlpha:
    def test_viscous(self):
        assert split_alpha(1e-3, 0.) == (1e-3, 0.)

    def test_hybrid(self):
        a_ss, a_dw = split_alpha(1e-3, 1.)
        assert np.isclose(a_ss, 5e-4) and np.isclose(a_dw, 5e-4)

    def test_pure_wind(self):
        assert split_alpha(1e-3, np.inf) == (0., 1e-3)

    def test_array(self):
        a_ss, a_dw = split_alpha(1e-3 * np.ones(3), np.array([0., 1., 3.]))
        np.testing.assert_allclose(a_ss + a_dw, 1e-3)
        np.testing.assert_allclose(a_dw[1:] / a_ss[1:], [1., 3.])

    def test_negative_psi(self):
        with pytest.raises(ValueError):
            split_alpha(1e-3, -1.)


class TestDiskwindSetup:
    def test_fields_exist(self):
        sim = make_sim(alpha_dw=1e-3, Lambda=2.)
        for name in ["alpha", "Lambda", "v", "S", "dustcoupling"]:
            assert getattr(sim.gas.diskwind, name) is not None
        np.testing.assert_allclose(sim.gas.diskwind.alpha, 1e-3)
        np.testing.assert_allclose(sim.gas.diskwind.Lambda, 2.)
        np.testing.assert_allclose(sim.gas.diskwind.dustcoupling, 1.)

    def test_updateorder(self):
        sim = make_sim()
        order = sim.gas.updateorder
        assert order.index("diskwind") == order.index("eta") + 1
        assert order.count("diskwind") == 1

    def test_default_off(self):
        sim = make_sim()
        assert sim.ini.gas.alphaDiskwind == 0.
        assert sim.ini.gas.LambdaDiskwind == 3.5
        assert np.all(sim.gas.diskwind.v == 0.)
        assert np.all(sim.gas.diskwind.S == 0.)

    def test_zero_turbulence_warns(self):
        sim = Simulation()
        sim.ini.grid.Nr = 10
        sim.ini.gas.alpha = 0.
        with pytest.warns(UserWarning, match="delta.turb"):
            sim.initialize()


class TestDiskwindFormulas:
    def test_velocity_and_sink(self):
        sim = make_sim(alpha_dw=1e-3, Lambda=3.)
        a, cs, r, Om = 1e-3, sim.gas.cs, sim.grid.r, sim.grid.OmegaK
        np.testing.assert_allclose(sim.gas.diskwind.v, -1.5 * a * cs**2 / (r * Om))
        np.testing.assert_allclose(sim.gas.diskwind.S,
                                   -0.75 * a * cs**2 * sim.gas.Sigma / (2. * r**2 * Om))

    def test_vrad_contains_wind(self):
        sim = make_sim(alpha_dw=1e-3)
        # DustPy's vrad (installed version) does not contain the disk wind
        np.testing.assert_allclose(sim.gas.v.rad - dp.std.gas.vrad(sim), sim.gas.diskwind.v)

    def test_vrad_without_wind_equals_dustpy(self):
        sim = make_sim()
        np.testing.assert_allclose(sim.gas.v.rad, dp.std.gas.vrad(sim))

    def test_S_tot_interior(self):
        sim = make_sim(alpha_dw=1e-3)
        S = trigas.S_tot(sim)
        expected = np.array(sim.gas.S.hyd)
        expected[1:-1] += sim.gas.S.ext[1:-1] + sim.gas.diskwind.S[1:-1]
        np.testing.assert_allclose(S, expected)

    def test_component_sinks_sum_to_total(self):
        sim = make_sim(alpha_dw=1e-3)
        co = 1e-2 * sim.gas.Sigma
        sim.addcomponent("CO", co, 28 * c.m_p, gas_active=True)
        sim.components.Default.gas.Sigma = sim.components.Default.gas.Sigma - co
        sim.update()
        total = trigas.S_diskwind_compo(sim, group=sim.components.Default.gas) \
            + trigas.S_diskwind_compo(sim, group=sim.components.CO.gas)
        np.testing.assert_allclose(total, trigas.S_diskwind(sim), rtol=1e-12)


class TestDiskwindJacobian:
    def test_equals_dustpy_without_wind(self):
        sim = make_sim()
        J_tp = trigas.jacobian(sim, sim.t).toarray()
        J_dp = dp.std.gas.jacobian(sim, sim.t).toarray()
        np.testing.assert_allclose(J_tp, J_dp)

    def test_wind_enters_advection(self):
        sim = make_sim(alpha_dw=1e-3)
        J_tp = trigas.jacobian(sim, sim.t).toarray()
        # DustPy's Jacobian with the wind velocity added to the torque velocity
        vtorque = np.array(sim.gas.torque.v)
        sim.gas.torque.v[:] = vtorque + sim.gas.diskwind.v
        J_dp = dp.std.gas.jacobian(sim, sim.t).toarray()
        sim.gas.torque.v[:] = vtorque
        np.testing.assert_allclose(J_tp, J_dp)


class TestDiskwindStateVector:
    def test_gas_active_and_tracer_sources(self):
        sim = make_sim(alpha_dw=1e-3)
        tr = np.linspace(0., 1., int(sim.grid.Nr))
        sim.addcomponent("tr", tr, 1., gas_tracer=True)
        compo.set_state_vector_components(sim)
        default = sim.components.Default
        np.testing.assert_allclose(default._S,
                                   default.gas.Sigma_dot + trigas.S_diskwind_compo(sim, group=default.gas))
        np.testing.assert_allclose(sim.components.tr._S,
                                   tr * (sim.gas.S.ext + sim.gas.diskwind.S))

    def test_timestep_limited_by_wind(self):
        sim = make_sim(alpha_dw=1e-3)
        Sdw = trigas.S_diskwind(sim)
        expected = np.min(-sim.gas.Sigma[1:-1] / Sdw[1:-1])
        assert np.isclose(trigas.dt_compo(sim), expected)

    def test_sink_only_decay(self):
        """Without viscosity and wind velocity the surface density decays exponentially."""
        sim = make_sim(alpha_dw=1e-2, Lambda=1.5, Nr=12, rmin=1., rmax=3.)
        sim.gas.nu.updater = None
        sim.gas.nu[:] = 0.
        sim.gas.diskwind.v.updater = None
        sim.gas.diskwind.v[:] = 0.
        sim.update()
        Sigma0 = np.array(sim.gas.Sigma)
        k = -np.array(sim.gas.diskwind.S) / Sigma0
        sim.writer = None
        sim.verbosity = 0
        sim.t.snapshots = [1e3 * c.year]
        sim.run()
        np.testing.assert_allclose(sim.gas.Sigma[2:-2] / Sigma0[2:-2],
                                   np.exp(-k[2:-2] * float(sim.t)), rtol=2e-3)


class TestDustCoupling:
    def test_coupling_switch(self):
        sim = make_sim(alpha_dw=1e-3)
        np.testing.assert_allclose(tridust.vgas_dust(sim), sim.gas.v.rad)
        vdust_full = np.array(tridust.vrad(sim))
        sim.gas.diskwind.dustcoupling[:] = 0.
        np.testing.assert_allclose(tridust.vgas_dust(sim), sim.gas.v.rad - sim.gas.diskwind.v)
        vdust_none = np.array(tridust.vrad(sim))
        St = sim.dust.St
        np.testing.assert_allclose(vdust_full - vdust_none,
                                   sim.gas.diskwind.v[:, None] / (St**2 + 1.), atol=1e-10)
