
"""Module containing standard functions for the composition"""
import dustpy.constants as c
from dustpy.std import dust_f as dp_dust_f
import dustpy.std.dust as dp_dust
from matplotlib.pylab import f
from tripod.std import dust_f
from tripod.std import dust as tridust
import dustpy as dp
import numpy as np
import scipy.sparse as sp


def prepare(sim):
    """Function prepares implicit dust integration step.
    It stores the current value of the surface density in a hidden field.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame"""
    
    set_state_vector_components(sim)


def set_state_vector_components(sim):
    #iterate over all components and set the state vector
    for name, comp in sim.components.__dict__.items():
        if(name.startswith("_")):
            continue

        #gas component
        if comp.dust._active == False and (comp.dust._tracer == False) and (comp.gas._active == True) and (comp.gas._tracer == False):
            comp._Y = comp.gas.Sigma
            comp._S = comp.gas.Sigma_dot
        #gas tracer
        elif comp.dust._active == False and (comp.dust._tracer == False) and (comp.gas._tracer == True) and (comp.gas._active == False):
            comp._Y = comp.gas.value*sim.gas.Sigma # tracer int variable = tracer * Sigma
            comp._S = comp.gas.value_dot*sim.gas.Sigma + comp.gas.value*sim.gas.S.ext
        #dust tracer
        elif (comp.dust._tracer == True) and (comp.gas._active == False) and (comp.gas._tracer == False):
            comp._Y = comp.dust.value.ravel()*sim.dust.Sigma.ravel() # tracer int variable = tracer * Sigma
            comp._S = comp.dust.value_dot.ravel()*sim.dust.Sigma.ravel() + comp.dust.value.ravel()*(sim.dust.S.ext.ravel() + sim.dust.S.compo.ravel())
        #dust and gas
        elif (comp.dust._tracer == True) and (comp.dust._active == False) and (comp.gas._active == True) and (comp.gas._tracer == False):
            Nr = int(sim.grid.Nr)
            comp._Y[:Nr] = comp.gas.Sigma.ravel()
            comp._Y[Nr:] = comp.dust.value.ravel()*sim.dust.Sigma.ravel()
            comp._S[:Nr] = comp.gas.Sigma_dot
            comp._S[Nr:] = comp.dust.value_dot.ravel()*sim.dust.Sigma.ravel() + comp.dust.value.ravel()*(sim.dust.S.ext.ravel() + sim.dust.S.compo.ravel())

        elif (comp.dust._active == True) and (comp.gas._active == False) and (comp.gas._tracer == False):
            comp._Y = comp.dust.Sigma.ravel()
            comp._S = comp.dust.S.ext.ravel()
        elif (comp.dust._active == True) and (comp.dust._tracer == False)  and (comp.gas._active == True) and (comp.gas._tracer == False):
            Nr = int(sim.grid.Nr)
            comp._Y[:Nr] = comp.gas.Sigma.ravel()
            comp._Y[Nr:] = comp.dust.Sigma.ravel()
            comp._S[:Nr] = comp.gas.Sigma_dot.ravel()
            comp._S[Nr:] = comp.dust.S.ext.ravel()
        else:
            raise RuntimeError("Component type not recognized")
        
        # set rhs to state vector
        comp._Y_rhs = comp._Y

def finalize(sim):
    """Function finalizes implicit integration step.

    Parameters
    ----------
    sim : Frame
        Parent integration frame"""

    Nr = int(sim.grid.Nr)
    #iterate over all components and get back variable from state vector
    for name, comp in sim.components.__dict__.items():
        if(name.startswith("_")):
            continue

        if (comp.dust._tracer == False) and (comp.dust._tracer == False) and (comp.gas._active == True) and (comp.gas._tracer == False):
            comp.gas.Sigma[...] = comp._Y[:Nr]

    for name, comp in sim.components.__dict__.items():
        if(name.startswith("_") or ((comp.dust._active == False) and (comp.dust._tracer == False) and (comp.gas._active == True))):
            continue
        elif (comp.dust._active == False) and (comp.dust._tracer == False) and (comp.gas._active == False) and (comp.gas._tracer == True):
            comp.gas.value[...] = (comp._Y/ sim.gas.Sigma)
        elif (comp.dust._active == False) and (comp.dust._tracer == True) and (comp.gas._active == False) and (comp.gas._tracer == False):
            comp.dust.value[...] = (comp._Y/sim.dust._Y[:sim.grid._Nm_short*sim.grid.Nr]).reshape(comp.dust.value.shape)
        elif (comp.dust._active == False) and (comp.dust._tracer == True) and (comp.gas._active == True) and (comp.gas._tracer == False):
            comp.gas.Sigma[...] = comp._Y[:sim.grid.Nr].reshape(comp.gas.Sigma.shape)
            comp.dust.value[...] = (comp._Y[sim.grid.Nr:]/sim.dust._Y[:sim.grid._Nm_short*sim.grid.Nr]).reshape(comp.dust.value.shape)
        elif (comp.dust._active == True) and (comp.dust._tracer == False) and (comp.gas._active == False) and (comp.gas._tracer == False):
            comp.dust.Sigma[...] = comp._Y.reshape(comp.dust.Sigma.shape)
        elif (comp.dust._active == True) and (comp.dust._tracer == False) and (comp.gas._active == True) and (comp.gas._tracer == False):
            comp.gas.Sigma[...] = comp._Y[:sim.grid.Nr].reshape(comp.gas.Sigma.shape)
            comp.dust.Sigma[...] = comp._Y[sim.grid.Nr:].reshape(comp.dust.Sigma.shape)
        else:
            raise RuntimeError("Component type not recognized")
        
    sim.dust.S.compo.update()
    sim.dust.S.tot.update()
        
    sim.components._gas_updated = False
    sim.components._dust_updated = False



def Y_jacobian(sim, x, dx=None, *args, **kwargs):
    # Helper variables for convenience
    if dx is None:
        dt = x.stepsize
    else:
        dt = dx

    comp = kwargs.get("component", None)
    r = sim.grid.r
    ri = sim.grid.ri
    area = sim.grid.A
    Nr = int(sim.grid.Nr)
    Nm_s = int(sim.grid._Nm_short)

    # Getting the Jacobian of Sigma
    J_Sigma = sim.dust.Sigma.jacobian(x, dx=dt)

    # Getting the Jacobian of Gas
    J_Gas =  dp.std.gas.jacobian(sim,x, dx= dt)

    # Jacopbian for evaporation and condensation
    J_compo = jacobian_compo(sim, x, dx=dt,component=comp)

    # Convert to COO format for easier manipulation
    J_Gas_coo = J_Gas.tocoo()
    J_Sigma_coo = J_Sigma.tocoo()
    J_compo_coo = J_compo.tocoo()

    # Combine row indices (offset for block structure)
    rows = np.hstack([
        J_Gas_coo.row,
        J_Sigma_coo.row + J_Gas.shape[0],
        J_compo_coo.row
    ])
    
    # Combine column indices (offset for block structure)
    cols = np.hstack([
        J_Gas_coo.col,
        J_Sigma_coo.col + J_Gas.shape[1],
        J_compo_coo.col
    ])
    #J_compo_coo.data *= 0
    # Combine data
    data = np.hstack([
        J_Gas_coo.data,
        J_Sigma_coo.data,
        J_compo_coo.data
    ])


    # Total size
    Ntot = J_Gas.shape[0] + J_Sigma.shape[0]

    # check for nans 
    if np.isnan(data).any():
        #prind indices of NaNs
        nan_indices = np.where(np.isnan(data))[0]
        print(f"NaN values found at indices: {nan_indices}")
        print("Rows:", rows[nan_indices])
        print("Cols:", cols[nan_indices])
        raise ValueError("Jacobian contains NaN values. Please check the input data.")
    
    
    # Create the combined matrix
    J = sp.csc_matrix((data, (rows, cols)), shape=(Ntot, Ntot))

    return J

def _f_impl_1_direct_compo(x0, Y0, dx, jac=None, rhs=None, *args, **kwargs):
    """Implicit 1st-order integration scheme with direct matrix inversion
    Parameters
    ----------
    x0 : Intvar
        Integration variable at beginning of scheme
    Y0 : Field
        Variable to be integrated at the beginning of scheme
    dx : IntVar
        Stepsize of integration variable
    jac : Field, optional, defaul : None
        Current Jacobian. Will be calculated, if not set
    args : additional positional arguments
    kwargs : additional keyworda arguments
    Returns
    -------
    dY : Field
        Delta of variable to be integrated
    Butcher tableau
    ---------------
     1 | 1
    ---|---
       | 1
    """
    if jac is None:
        jac = Y0.jacobian(x0, dx)
    if rhs is None:
        rhs = np.array(Y0.ravel())

    # Source term for the integration variable
    S = kwargs.get("Sext", np.zeros_like(Y0))
    
    name = kwargs.get("name", "component")
    # Right hand side
    rhs[...] += dx * S 

    N = jac.shape[0]
    eye = sp.identity(N, format="csc")

    A = eye - dx * jac

    A_LU = sp.linalg.splu(A,
                          permc_spec="MMD_AT_PLUS_A",
                          diag_pivot_thresh=0.0,
                          options=dict(SymmetricMode=True))
    Y1_ravel = A_LU.solve(rhs)

    Y1 = Y1_ravel.reshape(Y0.shape)
    return Y1 - Y0



def jacobian_compo(sim, x, dx=None, *args, **kwargs):
    """Function calculates the Jacobian for implicit integration caused by sublimation.

    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    x : IntVar
        Integration variable
    dx : float, optional, default : None
        stepsize
    args : additional positional arguments
    kwargs : additional keyword arguments

    Returns
    -------
    jac : Sparse matrix
        Dust Jacobian

    Notes
    -----
    Function returns the Jacobian for ``Simulation.dust.Sigma.ravel()``
    instead of ``Simulation.dust.Sigma``. The Jacobian is stored as
    sparse matrix."""
    # Helper variables for convenience
    if dx is None:
        dt = x.stepsize
    else:
        dt = dx

    Nr = int(sim.grid.Nr)
    Nm_s = int(sim.grid._Nm_short)
    comp = kwargs.get("component", None)

    # Total problem size
    Ntot = int((Nr * Nm_s) + Nr)


    # Insert source terms here 
    sublimation = L_sublimation(sim, comp).ravel("F")
    condensation = L_condensation(sim,comp, Pstick=kwargs.get("Pstick", 1.0)).ravel("F")

    # set source terms to zero at the boundaries
    sublimation[0] = 0
    sublimation[Nr] = 0

    sublimation[-1] = 0
    sublimation[-Nr] = 0


    condensation[-Nr] = 0
    condensation[-1] = 0

    condensation[0] = 0
    condensation[Nr] = 0



    #Gas affecting terms 
    row_con = np.hstack((np.arange(Nr),np.arange(Nr)))
    col_con = np.hstack((np.arange(Nr),np.arange(Nr)))
    #dust affecting terms
    row_cond = np.hstack((np.arange(Nr,Ntot,2), np.arange(Nr+1,Ntot,2)))
    col_cond = np.hstack((np.arange(Nr),np.arange(Nr)))

    # sublimation
    row_sub = np.hstack((np.arange(Nr), np.arange(Nr)))
    col_sub = np.hstack((np.arange(Nr,Ntot,2), np.arange(Nr+1,Ntot,2)))
    row_subd = np.hstack((np.arange(Nr,Ntot,2), np.arange(Nr+1,Ntot,2)))
    col_subd = np.hstack((np.arange(Nr,Ntot,2), np.arange(Nr+1,Ntot,2)))

    dat = np.hstack((sublimation,-sublimation,-condensation,condensation))
    row = np.hstack((row_sub, row_subd, row_con, row_cond))
    col = np.hstack((col_sub, col_subd, col_con, col_cond))


    gen = (dat, (row, col))
    # Building sparse matrix of coagulation Jacobian
    J = sp.csc_matrix(
        gen,
        shape=(Ntot, Ntot)
    )
    return J



def A_grains(sim):
    """ returns total surface area of all dust grains in the simulation

    Parameters
    ----------
    sim : Frame
        Parent simulation frame

    Returns
    -------
    A_grains : float
        Total surface area of all dust grains in the simulation for each bin
    """

    I = np.zeros_like(sim.dust.Sigma)
    a_int = (sim.dust.s.max*sim.dust.s.min)**0.5
    mask4 = sim.dust.qrec == -4
    [mask4]
    I[mask4,1] = 1./(np.log(sim.dust.s.max[mask4]) -np.log(a_int[mask4])) * (1./a_int[mask4] - 1./sim.dust.s.max[mask4])
    I[mask4,0] = 1./(np.log(a_int[mask4]) -np.log(sim.dust.s.min[mask4])) * (1./sim.dust.s.min[mask4] - 1./a_int[mask4])
    mask3 = sim.dust.qrec == -3
    I[mask3,1] = (np.log(sim.dust.s.max[mask3]) -np.log(a_int[mask3])) / (sim.dust.s.max[mask3] - a_int[mask3])
    I[mask3,0] = (np.log(a_int[mask3]) - np.log(sim.dust.s.min[mask3])) / (a_int[mask3] - sim.dust.s.min[mask3])
    mask = np.logical_and(~mask4 , ~mask3)
    I[mask,1] =  (sim.dust.qrec[mask] +4) / (sim.dust.qrec[mask] +3) * (sim.dust.s.max[mask]**(sim.dust.qrec[mask] + 3) - a_int[mask]**(sim.dust.qrec[mask] + 3))/(sim.dust.s.max[mask]**(sim.dust.qrec[mask] + 4) - a_int[mask]**(sim.dust.qrec[mask] + 4))
    I[mask,0] = (sim.dust.qrec[mask] +4) / (sim.dust.qrec[mask] +3) * (a_int[mask]**(sim.dust.qrec[mask] + 3) - sim.dust.s.min[mask]**(sim.dust.qrec[mask] + 3))/(a_int[mask]**(sim.dust.qrec[mask] + 4) - sim.dust.s.min[mask]**(sim.dust.qrec[mask] + 4))

    A_grains = sim.dust.Sigma * 3./sim.dust.rhos[:,[0,2]] * I 
    return A_grains



# L * u = S where S is the source term
def L_condensation(sim,comp,Pstick=1):
    """Function calculates the condensation source term for a given component.
    
    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    name : str, optional, default : None
        Name of the component. If None, the first component is used.
    Pstick : float, optional, default : 1
        Sticking probability
        
    Returns
    -------
    S_condensation : Field
        Condensation source term for the given component
    """

    A = A_grains(sim)

    L_con = A/4./(2*np.pi)**0.5 /  sim.gas.Hp[:,None] *((8.*c.k_B*sim.gas.T[:,None])/(np.pi*comp.gas.pars.mu))**0.5 * Pstick

    return L_con


def L_sublimation(sim,comp,N_bind=1e15):
    """Function calculates the sublimation source term for a given component.
    
    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    name : str, optional, default : None
        Name of the component. If None, the first component is used.
    N_bind : float, optional, default : 1e15
        number of binding sites per cm² on the dust grain surface
        
    Returns
    -------
    L_sublimation : Field
        Sublimation source term for the given component
    """

    # Calculate the total surface area of all dust grains
    A = A_grains(sim)
            
    if(comp.dust._tracer):
        Sig = comp.dust.value * sim.dust.Sigma
    elif(comp.dust._active):
        Sig = comp.dust.Sigma
    else:
        raise RuntimeError("Component dust type not recognized for sublimation calculation")

    N_layer = Sig/(A * N_bind*comp.gas.pars.mu)

    mask = N_layer < 1e-2

    L_sub = np.where(mask, comp.gas.pars.nu * np.exp(-comp.gas.pars.Tsub/sim.gas.T[:,None]), \
                    A / Sig *  N_bind * comp.gas.pars.nu * comp.gas.pars.mu * (1. - np.exp(-N_layer)) * np.exp(-comp.gas.pars.Tsub/sim.gas.T[:,None]))

    return L_sub


def c_jacobian(sim, x, dx=None, *args, **kwargs):
    """Function calculates the Jacobian for implicit integration of components

    Parameters
    ----------
    sim : Frame
        Parent simulation frame
    x : IntVar
        Integration variable
    dx : float, optional, default : None
        stepsize
    args : additional positional arguments
    kwargs : additional keyword arguments
        component : Component
            Component to calculate the Jacobian for
    Returns
    -------
    jac : Sparse matrix
        Component Jacobian
    Notes
    -----
    also setts the boudary conditions for the Jacobian including rhs
    """
    if dx is None:
        dt = x.stepsize
    else:
        dt = dx

    #get component type
    comp = kwargs.get("component", None)
    #call the correct jacobian depending on the component type (dust_active, gas_active, gas_tracer)
    
    #gas component only or tracrer share jacobian
    if comp.dust._active == False and comp.dust._tracer == False and (comp.gas._active == True or comp.gas._tracer == True):
        J = dp.std.gas.jacobian(sim,x, dx=dt, *args, **kwargs)
        J_new = set_boundaries_component(sim,J,dt,comp)
        return J_new
    #dust tracer
    elif comp.dust._tracer == True and comp.gas._active == False and comp.gas._tracer == False:
        J = tridust.jacobian(sim,x, dx=dt, *args, **kwargs)
        J_new = set_boundaries_component(sim,J,dt,comp)
        return J_new
    #dust and ga
    elif comp.dust._tracer == True and comp.gas._active == True:
        J = Y_jacobian(sim, x, dx=dt, *args, **kwargs)
        J_new = set_boundaries_component(sim,J,dt,comp)
        return J_new
    elif comp.dust._active == True and comp.gas._active == False and comp.gas._tracer == False:
        J = tridust.jacobian(sim,x, dx=dt, *args, **kwargs)
        J_new = set_boundaries_component(sim,J,dt,comp)
        return J_new
    elif comp.dust._active == True and comp.gas._active == True:
        J = Y_jacobian(sim, x, dx=dt, *args, **kwargs)
        J_new = set_boundaries_component(sim,J,dt,comp)
        return J_new
    else:    
        raise RuntimeError("Component type not recognized")
    
def set_boundaries_component(sim,J,dx,comp):
    """Function sets the boundary conditions for the Jacobian of a component.
    
    Parameters
    ----------
    J : Sparse matrix
        Jacobian matrix to set the boundary conditions for
    comp : Component
        Component to set the boundary conditions for
    """
    Nr = int(sim.grid.Nr)
    Nm_s = int(sim.grid._Nm_short)

    if(comp.dust._tracer):
        if not sim.components._dust_updated and sim._dust_compo:
            # still need to update the dust surface density for the boundary conditions
            sim.dust.Sigma[...] = np.zeros_like(sim.dust.Sigma)
            for nm, cint in sim.components.__dict__.items():
                if(nm.startswith("_")):
                    continue
                if cint.gas._active or cint.gas._tracer:
                    offset = Nr
                else:
                    offset = 0

                if cint.dust._active:
                    cint.dust.Sigma[...] += cint._Y[offset:].reshape(sim.dust.Sigma.shape)
            sim.components._dust_updated = True
        # Given value
        if comp.gas._active or comp.gas._tracer:
            offset = Nr
        else:
            offset = 0
        
        #assure jacobian is empty for boundaries
        comp._Y_rhs[offset:offset+Nm_s] = comp.dust.boundary.inner.value * sim.dust.Sigma[:Nr * Nm_s].reshape(sim.dust.Sigma.shape)[0,:]
        comp._Y_rhs[offset+Nm_s*(Nr-1):offset+Nm_s*Nr] = comp.dust.boundary.outer.value * sim.dust.Sigma[:Nr * Nm_s].reshape(sim.dust.Sigma.shape)[-1,:]

        # Set source term to zero at the boundaries
        comp._S[offset] = 0.
        comp._S[offset+Nm_s*Nr-1] = 0.

        # Set rows of jacobian to zero at the boundaries
        row_indices = np.arange(offset, offset + Nm_s)
        row_mask = np.isin(J.indices, row_indices)
        J.data[row_mask] = 0.0

        #set rows of jacobian to zero at the boundaries
        row_indices = np.arange(offset+Nm_s*(Nr-1), offset+Nm_s*(Nr))
        row_mask = np.isin(J.indices, row_indices)
        J.data[row_mask] = 0.0



    if(comp.gas._tracer):
        if not sim.components._gas_updated:
            for nm, cint in sim.components.__dict__.items():
                if(nm.startswith("_")):
                    continue
                if cint.gas._active:
                    cint.gas.Sigma[...] = cint._Y[:sim.grid.Nr].reshape(sim.gas.Sigma.shape)

            sim.gas.Sigma.update()
            sim.components._gas_updated = True
        # Given value
        comp._Y_rhs[0] = comp.gas.boundary.inner.value * sim.gas.Sigma[0]
        comp._Y_rhs[Nr-1] = comp.gas.boundary.outer.value * sim.gas.Sigma[-1]

        # Set source term to zero at the boundaries
        comp._S[0] = 0.
        comp._S[Nr-1] = 0.


    if(comp.gas._active):

        # Set source term to zero at the boundaries
        comp._S[0] = 0.
        comp._S[Nr-1] = 0.
        if comp.gas.boundary.inner is not None:
            # Given value
            if comp.gas.boundary.inner.condition == "val":
                comp._Y_rhs[0] = comp.gas.boundary.inner.value
            # Constant value
            elif comp.gas.boundary.inner.condition == "const_val":
                J[0, 1] = 1./dx
                comp._Y_rhs[0] = 0.
            # Given gradient
            elif comp.gas.boundary.inner.condition == "grad":
                K1 = - comp.gas.boundary.inner._r[1]/comp.gas.boundary.inner._r[0]
                J[0, 1] = -K1/dx
                comp._Y_rhs[0] = - comp.gas.boundary.inner._ri[1]/comp.gas.boundary.inner._r[0] * \
                    (comp.gas.boundary.inner._r[1]-comp.gas.boundary.inner._r[0]) * \
                    comp.gas.boundary.inner.value
            # Constant gradient
            elif comp.gas.boundary.inner.condition == "const_grad":
                Di = comp.gas.boundary.inner._ri[1]/comp.gas.boundary.inner._ri[2] * (
                    comp.gas.boundary.inner._r[1]-comp.gas.boundary.inner._r[0]) / (comp.gas.boundary.inner._r[2]-comp.gas.boundary.inner._r[0])
                K1 = - comp.gas.boundary.inner._r[1]/comp.gas.boundary.inner._r[0] * (1. + Di)
                K2 = comp.gas.boundary.inner._r[2]/comp.gas.boundary.inner._r[0] * Di
                J[0, :3] = 0.
                J[0, 1] = -K1/dx
                J[0, 2] = -K2/dx
                comp._Y_rhs[0] = 0.
            # Given power law
            elif comp.gas.boundary.inner.condition == "pow":
                p = comp.gas.boundary.inner.value
                comp._Y_rhs[0] = comp._Y_rhs[1] * (comp.gas.boundary.inner._r[0]/comp.gas.boundary.inner._r[1])**p
            # Constant power law
            elif comp.gas.boundary.inner.condition == "const_pow":
                p = np.log(comp._Y_rhs[2] / comp._Y_rhs[1]) / \
                    np.log(comp.gas.boundary.inner._r[2]/comp.gas.boundary.inner._r[1])
                K1 = - (comp.gas.boundary.inner._r[0]/comp.gas.boundary.inner._r[1])**p
                J[0, 1] = -K1/dx
                comp._Y_rhs[0] = 0.

    # Outer boundary
        if comp.gas.boundary.outer is not None:
            # Given value
            if comp.gas.boundary.outer.condition == "val":
                comp._Y_rhs[Nr-1] = comp.gas.boundary.outer.value
            # Constant value
            elif comp.gas.boundary.outer.condition == "const_val":
                J[Nr-1,Nr-2] = (1./dx)
                comp._Y_rhs[Nr-1] = 0.
            # Given gradient
            elif comp.gas.boundary.outer.condition == "grad":
                KNrm2 = - comp.gas.boundary.outer._r[1]/comp.gas.boundary.outer._r[0]
                J[Nr-1,Nr-2] = -(KNrm2/dx)
                comp._Y_rhs[Nr-1] = comp.gas.boundary.outer._ri[1]/comp.gas.boundary.outer._r[0] * \
                    (comp.gas.boundary.outer._r[0]-comp.gas.boundary.outer._r[1]) * \
                    comp.gas.boundary.outer.value
            # Constant gradient
            elif comp.gas.boundary.outer.condition == "const_grad":
                Do = comp.gas.boundary.outer._ri[1]/comp.gas.boundary.outer._ri[2] * (
                    comp.gas.boundary.outer._r[0]-comp.gas.boundary.outer._r[1]) / (comp.gas.boundary.outer._r[1]-comp.gas.boundary.outer._r[2])
                KNrm2 = - comp.gas.boundary.outer._r[1]/comp.gas.boundary.outer._r[0] * (1. + Do)
                KNrm3 = comp.gas.boundary.outer._r[2]/comp.gas.boundary.outer._r[0] * Do
                J[Nr-1,Nr-3:Nr] = 0.
                J[Nr-1,Nr-2] = -KNrm2/dx
                J[Nr-1,Nr-3] = -KNrm3/dx
                comp._Y_rhs[Nr-1] = 0.
            # Given power law
            elif comp.gas.boundary.outer.condition == "pow":
                p = comp.gas.boundary.outer.value
                comp._Y_rhs[Nr-1] = comp._Y_rhs[Nr-2] * (comp.gas.boundary.outer._r[-0]/comp.gas.boundary.outer._r[1])**p
            # Constant power law
            elif comp.gas.boundary.outer.condition == "const_pow":
                p = np.log(comp._Y_rhs[Nr-2] / comp._Y_rhs[Nr-3]) / \
                    np.log(comp.gas.boundary.outer._r[1]/comp.gas.boundary.outer._r[2])
                KNrm2 = - (comp.gas.boundary.outer._r[0]/comp.gas.boundary.outer._r[1])**p
                J[Nr-1,Nr-2] = -KNrm2/dx
                comp._Y_rhs[Nr-1] = 0.

    if(comp.dust._active):
        # Given value
        if comp.gas._active or comp.gas._tracer:
            offset = Nr
        else:
            offset = 0

        # Filling data vector depending on boundary condition
        if comp.dust.boundary.inner is not None:
            # Given value
            if comp.dust.boundary.inner.condition == "val":
                comp._Y_rhs[offset:offset+Nm_s] = comp.dust.boundary.inner.value
            # Constant value
            elif comp.dust.boundary.inner.condition == "const_val":
                for k in range(Nm_s):
                    J[offset+k,offset+Nm_s+k] = 1. / dx
                comp._Y_rhs[offset:offset+Nm_s] = 0.
            # Given gradient
            elif comp.dust.boundary.inner.condition == "grad":
                K1 = - comp.dust.boundary.inner._r[1] / comp.dust.boundary.inner._r[0]
                for k in range(Nm_s):
                    J[offset+k,offset+Nm_s+k] = -K1 / dx
                comp._Y_rhs[offset:offset+Nm_s] = - comp.dust.boundary.inner._ri[1] / comp.dust.boundary.inner._r[0] * \
                    (comp.dust.boundary.inner._r[1] - comp.dust.boundary.inner._r[0]) * comp.dust.boundary.inner.value
            # Constant gradient
            elif comp.dust.boundary.inner.condition == "const_grad":
                Di = comp.dust.boundary.inner._ri[1] / comp.dust.boundary.inner._ri[2] * (comp.dust.boundary.inner._r[1] - comp.dust.boundary.inner._r[0]) / (comp.dust.boundary.inner._r[2] - comp.dust.boundary.inner._r[0])
                K1 = - comp.dust.boundary.inner._r[1] / comp.dust.boundary.inner._r[0] * (1. + Di)
                K2 = comp.dust.boundary.inner._r[2] / comp.dust.boundary.inner._r[0] * Di
                for k in range(Nm_s):
                    J[offset+k,offset+k] = 0.
                    J[offset+k,offset+Nm_s+k] =-K1 / dx
                    J[offset+k,offset+2*Nm_s+k] = -K2 / dx
                comp._Y_rhs[offset:offset+Nm_s] = 0.
            # Given power law
            elif comp.dust.boundary.inner.condition == "pow":
                p = comp.dust.boundary.inner.value
                comp._Y_rhs[offset:offset+Nm_s] = comp.dust.Sigma[1] * (comp.dust.boundary.inner._r[0] /comp.dust.boundary.inner._r[1]) ** p
            # Constant power law
            elif comp.dust.boundary.inner.condition == "const_pow":
                p = np.log(comp.dust.Sigma[2] /
                        comp.dust.Sigma[1]) / np.log(comp.dust.boundary.inner._r[2] / comp.dust.boundary.inner._r[1])
                K1 = - (comp.dust.boundary.inner._r[0] / comp.dust.boundary.inner._r[1]) ** p
                for k in range(Nm_s):
                    J[offset+k,offset+Nm_s+k] = -K1[k] / dx
                comp._Y_rhs[offset:offset+Nm_s] = 0.

        if comp.dust.boundary.outer is not None:
            # Given value
            if comp.dust.boundary.outer.condition == "val":
                comp._Y_rhs[-Nm_s:]= comp.dust.boundary.outer.value
            # Constant value
            elif comp.dust.boundary.outer.condition == "const_val":
                for k in range(Nm_s):
                    J[-Nm_s+k,-2 * Nm_s+k] = 1. / dx
                comp._Y_rhs[-Nm_s:]= 0.
            # Given gradient
            elif comp.dust.boundary.outer.condition == "grad":
                KNrm2 = -comp.dust.boundary.outer._r[1] / comp.dust.boundary.outer._r[0]
                for k in range(Nm_s):
                    J[-Nm_s+k,-2 * Nm_s+k] = -KNrm2 / dx
                comp._Y_rhs[-Nm_s:]= comp.dust.boundary.outer._ri[1] / comp.dust.boundary.outer._r[0] * \
                    (comp.dust.boundary.outer._r[0] - comp.dust.boundary.outer._r[1]) * comp.dust.boundary.outer.value
            # Constant gradient
            elif comp.dust.boundary.outer.condition == "const_grad":
                Do = comp.dust.boundary.outer._r[1] / comp.dust.boundary.outer._ri[2] * (comp.dust.boundary.outer._r[0] - comp.dust.boundary.outer._r[1]) / (comp.dust.boundary.outer._r[1] - comp.dust.boundary.outer._r[2])
                KNrm2 = - comp.dust.boundary.outer._r[1] / comp.dust.boundary.outer._r[0] * (1. + Do)
                KNrm3 = comp.dust.boundary.outer._r[2] / comp.dust.boundary.outer._r[0] * Do
                for k in range(Nm_s):
                    J[-Nm_s+k,-Nm_s+k] = 0.
                    J[-Nm_s+k,-2 * Nm_s+k] = -KNrm2 / dx
                    J[-Nm_s+k,-3 * Nm_s+k] = -KNrm3 / dx
                comp._Y_rhs[-Nm_s:]= 0.
            # Given power law
            elif comp.dust.boundary.outer.condition == "pow":
                p = comp.dust.boundary.outer.value
                comp._Y_rhs[-Nm_s:]= comp.dust.Sigma[-2] * (comp.dust.boundary.outer._r[0] / comp.dust.boundary.outer._r[1]) ** p
            # Constant power law
            elif comp.dust.boundary.outer.condition == "const_pow":
                p = np.log(comp.dust.Sigma[-2] /
                        comp.dust.Sigma[-3]) / np.log(comp.dust.boundary.outer._r[1] / comp.dust.boundary.outer._r[2])
                KNrm2 = - (comp.dust.boundary.outer._r[0] / comp.dust.boundary.outer._r[1]) ** p
                for k in range(Nm_s):
                    J[-Nm_s+k,-2 * Nm_s+k] = -KNrm2[k] / dx
                comp._Y_rhs[-Nm_s:]= 0.


    return J
