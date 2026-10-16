### Things I have changed in the python version to make it consistent with PLUTO

- changed the shrinkage tertm to force the solution to f = fcrit at each timestep -> makes the results te same as in pluto 

- changed the power law to be consistent with the tripod code sent to me -> different than in the paper -> switch the two turbulent regimes

- changed how the transition berween driftfag and turb frag is calcualted (now same as in tripod)

- changed the way the drift fuge factor is implemented -> modify the stokes number in the calculation of the drift and diffusion (to be consistent with tripod)

- for now coagualtion is added as implicit term and added to the matrix inversion 

- added the reconstructed power law qeff used in the calcualtion to compute the representative sizes 

#### things beyond the PLUTO version that have been changhed 

- added tracers and components that can sublimate and condensate between gas and dust

#### Ideas what should be adapted/Future directions

- Solve all the components in the same matrix to make chemistry possible

- find a way to implictly calcualte the growth of amax
    

#### MHD disk winds (Tabone et al. 2022)

- added the MHD disk wind formalism of Tabone et al. (2022) to the gas evolution, following the DustPy develop branch (same names, but computed in TriPoDPy itself so it works with released DustPy versions)
  - new parameters `sim.ini.gas.alphaDiskwind` (default 0, wind switched off) and `sim.ini.gas.LambdaDiskwind` (default 3.5)
  - new group `sim.gas.diskwind` with `alpha`, `Lambda`, `v` (wind-driven accretion velocity, added to `sim.gas.v.rad` and to the gas Jacobian) and `S` (mass loss, explicit sink added to the sources of every gas component and gas tracer, limits the time step)
  - new per-component source term `sim.components.<name>.gas.S.diskwind`
- added `sim.gas.diskwind.dustcoupling` (default 1): fraction of the wind-driven gas velocity felt by the dust (0: accretion in the disk surface layers only)
- added `tripodpy.utils.split_alpha(alpha_tilde, psi)` to split a total alpha into turbulent and wind parts (Tabone et al. 2022, Eqs. 12-13)
- the TriPoD fragmentation regimes (`q_frag`, `p_frag_trans`, `p_drift_frag`) now use `sim.dust.delta.turb` instead of `sim.gas.alpha`, so the gas viscosity can be zero in purely wind-driven disks (identical results by default, since `delta.turb = ini.gas.alpha`)
