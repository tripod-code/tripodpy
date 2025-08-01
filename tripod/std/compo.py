
"""Module containing standard functions for the composition"""
import dustpy.constants as c
from dustpy.std import dust_f as dp_dust_f
import dustpy.std.dust as dp_dust
from tripod.std import dust_f
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
    Nr = int(sim.grid.Nr)

    # Copy values to state vector Y
    for name, comp in sim.components.__dict__.items():
        if(name.startswith("_") or not (comp.includedust and comp.includegas)):
            continue

        comp._Y[:Nr] = comp.gas.Sigma.ravel()
        comp._Y[Nr:] = comp.dust.Sigma.ravel()
    
def finalize(sim):
    """Function finalizes implicit integration step.

    Parameters
    ----------
    sim : Frame
        Parent integration frame"""
    Nr = int(sim.grid.Nr)

    for name, comp in sim.components.__dict__.items():
        if(name.startswith("_") or not (comp.includedust and comp.includegas)):
            continue

        comp.gas.Sigma = comp._Y[:Nr].reshape(comp.gas.Sigma.shape)
        comp.dust.Sigma = comp._Y[Nr:].reshape(comp.dust.Sigma.shape)



def Y_jacobian(sim, x, dx=None, *args, **kwargs):
    # Helper variables for convenience
    if dx is None:
        dt = x.stepsize
    else:
        dt = dx

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
    J_compo = jacobian_compo(sim, x, dx=dt,name=kwargs.get("name", None))

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
    J = sp.coo_matrix((data, (rows, cols)), shape=(Ntot, Ntot))

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


    # Add external/explicit source terms to right-hand side
    name = kwargs.get("name")
    comp = Y0._owner.components.__dict__.get(name)

    r = Y0._owner.grid.r
    ri = Y0._owner.grid.ri
    area = Y0._owner.grid.A
    Nr = int(Y0._owner.grid.Nr)
    Nm_s = int(Y0._owner.grid._Nm_short)

    #set first row of jacobian to zero 
    jac.data[jac.row == 0] = 0.0


    # Set the right-hand side to 0 for the dust to be handeled like the global dust
    if Y0._owner.dust.boundary.inner.condition.startswith("const"):
        rhs[Nr:Nr+Nm_s] = 0.

    if Y0._owner.dust.boundary.outer.condition.startswith("const"):
        rhs[-Nm_s:] = 0.

    
    if Y0._owner.dust.boundary.outer.condition == "val":
        rhs[-Nm_s:] =  Y0[-Nm_s:]

    

    S = np.hstack((comp.gas.S.ext.ravel(), comp.dust.S.ext.ravel()))

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

    # Total problem size
    Ntot = int((Nr * Nm_s) + Nr)


    # Insert source terms here 
    sublimation = L_sublimation(sim, name=kwargs.get("name", None)).ravel("F")
    condensation = L_condensation(sim, name=kwargs.get("name", None), Pstick=kwargs.get("Pstick", 1.0)).ravel("F")

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
def L_condensation(sim,name=None,Pstick=1):
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
    if name is None:
        name = list(sim.components.__dict__.keys())[0]
            
    comp = sim.components.__dict__[name]
    if not (comp.includedust and comp.includegas):
        raise ValueError(f"Component {name} does not include evaporation and sublimation.")
    
    # Calculate the total surface area of all dust grains
    A = A_grains(sim)

    L_con = A/4./(2*np.pi)**0.5 /  sim.gas.Hp[:,None] *((8.*c.k_B*sim.gas.T[:,None])/(np.pi*comp.gas.mu))**0.5 * Pstick

    return L_con


def L_sublimation(sim,name=None,N_bind=1e15):
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

    if name is None:
        name = list(sim.components.__dict__.keys())[0]
            
    comp = sim.components.__dict__[name]
    if not (comp.includedust and comp.includegas):
        raise ValueError(f"Component {name} does not include evaporation and sublimation.")
    

    N_layer = comp.dust.Sigma/(A * N_bind*comp.gas.mu)

    mask = N_layer < 1e-2

    L_sub = np.where(mask, comp.nu * np.exp(-comp.Tsub/sim.gas.T[:,None]), \
                    A / comp.dust.Sigma *  N_bind * comp.nu * comp.gas.mu * (1. - np.exp(-N_layer)) \
                    * np.exp(-comp.Tsub/sim.gas.T[:,None]))

    return L_sub