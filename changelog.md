### Things I have changed in the python vesrsion to make it consistent with PLUTO

- fixed the eternal source term of the shrinkage -> not fully cosnistent yet 
    -same as in Pluto on the timestep but limitng the maximal deviation

- changed the power law to be consistent wit hthe tripod code sent to me -> different than in the paper -> switch the two turbulent regimes
-> changed how the transition berween driftfag and turb frag is calcualted (now same as in tripod)

- changed the relative velocity terms for the browninan motion (only considders the larger size), radial drift (uses vmax *.. instead of vdrift of the code) and vertical settleing (st/1+st**2 instead of cutoff) to be consistent with PLuto 

- changed the implementation of Re to use the same definiton as in Pluto -> use rhoGas

- changed the differences due to the use of the adiabatic vs the isothermal sound speed -> results in a factor of 1/gamma**0.5 for ST and1/gamma for vdriftmax

-changed the way the drift fuge factor is implemented -> modify the stokes number in the calculation of the drift and diffusion (to be consistent with tripod)

-for now coagualtion has beeen set as an external term to be cosistent with tripod

#### things beyond the PLUTO version that have been changhed 


#### Ideas what should be adapted 
-timescale for the reduction of amax seems odd 
