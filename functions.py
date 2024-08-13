import numpy as np
import scipy.sparse as sc
import pandas as pd
import pypardiso as ps
from numba import jit, njit

def shift_array(arr, n):
    if n >= 0:
        return np.concatenate((arr[-n:], arr[:-n]))
    else:
        return np.concatenate((arr[-n:], arr[:-n]))

def is_nan(q):
    r = 0
    for i in range(0, len(q)):
        if np.isnan(q[i]):
            r = 1
            break
    if r==1:
        return True
    else:
        return False

@jit
def avg_perm(k):
    ##Function to return the averaged permeabilities required for heat diffusion in an anisotropic medium
    kiph = k
    kimh = k
    kjph = k
    kjmh = k

    for i in range(0, len(k[:,1])-1):
        for j in range(0, len(k[1,:])-1):
            kiph[i,j] = (k[i,j]+k[i+1,j])/2
            kjph[i,j] = (k[i,j]+k[i,j+1])/2

    for i in range(1,len(k[:,1])-1):
        for j in range(1,len(k[1,:])-1):
            kimh[i,j] = (k[i,j]+k[i+1,j])/2
            kjmh[i,j] = (k[i,j]+k[i,j+1])/2

    return kiph, kimh, kjph, kjmh

def cheat_solver(Tf, a):
    """
    Matrix taking too much time to solve?
    Here is a cheat solver for linear gradients. Can't promise it will be the same as the regular equilibrium solve, but it should be accurate
    """
    T_top = Tf[0,0]
    grad = (Tf[-1,0]-Tf[0,0])/a
    for i in range(0,a):
        Tf[i,:] = T_top + i*grad
    return Tf

def straight_solver(Ab,dee, a, b):
    """
    Inverse matrix multiplication for small sized matrices - slow and memory consuming for larger matrices
    Ab = LHS weight matrix MxN
    dee = RHS matrix M*Nx1
    a = number of rows - M int
    b = number of columns N int
    """
    Tf = np.array(ps.spsolve(sc.csc_matrix(Ab), dee))
    Tf = Tf.reshape((a,b), order = 'F')
    return Tf

def JacobianIt(Ab, dee, a, b):
    """
    Iterative solver for heat flux equation based on Jacobian iterative method - Slowest convergence rate, but sure to eventually converge
    Ab = LHS weight matrix MxN
    dee = RHS matrix M*Nx1
    a = number of rows - M int
    b = number of columns N int
    """
    sc.linalg.use_solver(useUmfpack=True)
    D = sc.csc_matrix(sc.diags(Ab.diagonal(),0))#np.diag(np.diag(Ab))
    E = -(sc.tril(Ab, k=-1))
    F = -(sc.triu(Ab, k=1))
    do = sc.csc_matrix(sc.linalg.spsolve(D, sc.csc_matrix((E+F))))
    T = dee
    err = 1e10
    iter = 10000
    c = 0
    while c<iter and err>1e-3:
        c = c+1
        T_new = do.dot(T)+ sc.linalg.spsolve(D, dee)
        err = np.max(T_new - T)
        print(err)
        T = T_new

    T = T.reshape((a,b), order = 'F')
    print('Jacobian Iterations completed: ', c)
    return T

def GSIt(Ab, dee, a, b):
    """
    Iterative solver for heat flux equation based on Gauss-Seidell iterative method - Faster convergence rate per iteration than J and sure to eventually converge
    Ab = LHS weight matrix MxN
    dee = RHS matrix M*Nx1
    a = number of rows - M int
    b = number of columns N int
    """
    D = sc.diags(Ab.diagonal(),0)
    E = -(sc.tril(Ab, k=-1))
    F = -(sc.triu(Ab, k=1))
    tre = sc.csc_matrix(D - E)
    do = sc.csc_matrix(ps.spsolve(tre, sc.csc_array(F)))
    T = dee
    err = 1e10
    iter = 10000
    c = 0
    while c<iter and err>1e-3:
        c = c+1
        T_new = do.dot(T)+ ps.spsolve(tre, dee)
        err = np.max(T_new - T)
        print(err)
        T = T_new

    T = T.reshape((a,b), order = 'F')
    print('GS Iterations completed: ', c)
    return T


def heat_flux(k, a, b, dx, dy, Tnow, method, q = np.nan):
    """
    Fourier's law solver
    method = straight/ Jacobian
    k = Diffusivity field (anisotropic) MxN matrix
    a = number of rows - M int
    b = number of columns N int
    dx = spacing in x direction int
    dy = spacing in y direction int
    Tnow = temperature field at current time step MxN matrix - Note that you need to set the Dirichlet boundary values
    """
    if method == 'cheat':
            return cheat_solver(Tnow,a)
    else:
        if ~np.isnan(q).any():
            Tnow[-1,:] = q*dy/k[-1,:]
        bee = Tnow.reshape((a*b), order = 'F')

        main_diag = np.zeros(a*b)
        p1_diag = np.zeros(a*b - 1)
        m1_diag = np.zeros(a*b-1)
        pa_diag = np.zeros(a*b-a)
        ma_diag = np.zeros(a*b-a)
        index = 0

        for j in range(0,b):
            for i in range(0,a):
                if i>0 and i<a-1 and j>0 and j<b-1:
                    main_diag[index] = -(((2*k[i,j])/dx**2)+((2*k[i,j])/dy**2))
                    p1_diag[index] = ((k[i,j]/dx**2)+((k[i+1,j]-k[i-1,j])/(4*dx**2)))
                    m1_diag[index-1] = ((k[i,j]/dx**2)-((k[i+1,j]-k[i-1,j])/(4*dx**2)))
                    pa_diag[index] = ((k[i,j]/dy**2)+((k[i,j+1]-k[i,j-1])/(4*dy**2)))
                    ma_diag[index-a] = ((k[i,j]/dy**2)-((k[i,j+1]-k[i,j-1])/(4*dy**2)))
                if i==0:
                    main_diag[index] = 1
                if j==0 and i!=0 and i!=a-1:
                    main_diag[index] = -1
                    pa_diag[index] = 1
                if j==b-1 and i!=0 and i!=a-1:
                    main_diag[index] = 1
                    ma_diag[index-a] = -1
                if i==a-1:
                    if np.isnan(q).any():
                        main_diag[index] = 1
                    else:
                        main_diag[index] = 1
                        m1_diag[index-1] = -1
                index = index+1
        #Af = sc.spdiags([main_diag, p1_diag, m1_diag, pb_diag, mb_diag], [0,1,-1,a,-a])
        Af = sc.lil_matrix((a*b,a*b))
        print('Weight matrix:', Af.shape)
        Af.setdiag(main_diag, k=0)
        Af.setdiag(p1_diag, k=1)
        Af.setdiag(m1_diag,k=-1)
        #Af.setdiag(p2_diag,k=2*a)
        #Af.setdiag(m2_diag,k=-2*a)
        Af.setdiag(pa_diag, k=a)
        Af.setdiag(ma_diag, k=-a)

        #pd.DataFrame(Af.toarray()).to_csv('Af_new.csv')
        #exit()

        if method =='straight':
            return straight_solver(Af, bee, a, b)
        elif method == 'Jacobian':
            return JacobianIt(Af, bee, a, b)
        elif method == 'GS':
            return GSIt(Af, bee, a, b)





def perm_smoothed_solve(k, a, b, dx, dy, dt, Tnow, q, Af, H):
    """
    Solver for time varying heat diffusion equation building an averaged permeability field (for anisotropic heat diffusion)
    k = Diffusivity field (anisotropic) MxN matrix
    a = number of rows - M int
    b = number of columns N int
    dx = spacing in x direction int
    dy = spacing in y direction int
    Tnow = temperature field at current time step MxN matrix
    q = Heat flux at the bottom boundary - N int If q is left as nan, the boundary condition changes to Dirichlet i.e., constant temp
    """
    if np.isnan(Af):
        kiph, kimh, kjph, kjmh = avg_perm(k)
        main_diag = np.zeros(a*b)
        p1_diag = np.zeros(a*b - 1)
        m1_diag = np.zeros(a*b-1)
        pa_diag = np.zeros(a*b-a)
        ma_diag = np.zeros(a*b-a)
        index = 0



        for j in range(0,b):
            for i in range(0,a):
                if i>0 and i<a-1 and j>0 and j<b-1:
                    main_diag[index] = -(((kiph[i,j]+kimh[i,j])*(dt/(dx**2)))+((kjph[i,j]+kjmh[i,j])*(dt/(dy**2)))+ (H[i,j]*dt))+1
                    pa_diag[index] = kiph[i,j]*(dt/(dx**2))
                    ma_diag[index-a] = kimh[i,j]*(dt/(dx**2))
                    p1_diag[index] = kjph[i,j]*(dt/(dy**2))
                    m1_diag[index-1] = kjmh[i,j]*(dt/(dy**2))
                if i==0 or i==a-1:
                    main_diag[index] = 1
                if j==0 and i!=0 and i!=a-1:
                    main_diag[index] = 1
                if j==b-1 and i!=0 and i!=a-1:
                    main_diag[index] = 1
                index = index+1
        Af = sc.lil_matrix((a*b,a*b))
        Af.setdiag(main_diag, k=0)
        Af.setdiag(p1_diag, k=1)
        Af.setdiag(m1_diag,k=-1)
        Af.setdiag(pa_diag, k=a)
        Af.setdiag(ma_diag, k=-a)

    bee = Tnow.reshape((a*b), order = 'F')
    Tret = np.array(Af.dot(bee))
    Tret = Tret.reshape((a,b), order = 'F')
    Tret[:,0] = Tret[:,2]
    Tret[:,-1] = Tret[:,-3]
    if ~np.isnan(q).any():
        Tret[-1,:] = Tret[-2,:]+ (q*dy/k[-1,:])
    return Tret

def perm_chain_solve(k, a, b, dx, dy, dt, Tnow, q, Af, H):
        """
        Solver for time varying heat diffusion equation building an chain rule (for anisotropic heat diffusion)
        k = Diffusivity field (anisotropic) MxN matrix
        a = number of rows - M int
        b = number of columns N int
        dx = spacing in x direction int
        dy = spacing in y direction int
        Tnow = temperature field at current time step MxN matrix
        q = Heat flux at the bottom boundary - N int If q is left as nan, the boundary condition changes to Dirichlet i.e., constant temp
        """
        if np.isnan(Af):
            main_diag = np.zeros(a*b)
            p1_diag = np.zeros(a*b - 1)
            m1_diag = np.zeros(a*b-1)
            pa_diag = np.zeros(a*b-a)
            ma_diag = np.zeros(a*b-a)

            index = 0

            for j in range(0,b):
                for i in range(0,a):
                    if i>0 and i<a-1 and j>0 and j<b-1:
                        main_diag[index] = -((((2*k[i,j])/dx**2)+((2*k[i,j])/dy**2)+ H[i,j])*dt)+1 
                        pa_diag[index] = ((k[i,j]/dx**2)+((k[i+1,j]-k[i-1,j])/(4*dx**2)))*dt
                        ma_diag[index-a] = ((k[i,j]/dx**2)-((k[i+1,j]-k[i-1,j])/(4*dx**2)))*dt
                        p1_diag[index] = ((k[i,j]/dy**2)+((k[i+1,j]-k[i-1,j])/(4*dy**2)))*dt
                        m1_diag[index-1] = ((k[i,j]/dy**2)-((k[i+1,j]-k[i-1,j])/(4*dy**2)))*dt
                    if i==0 or i==a-1:
                        main_diag[index] = 1
                    if j==0 and i!=0 and i!=a-1:
                        main_diag[index] = 1
                    if j==b-1 and i!=0 and i!=a-1:
                        main_diag[index] = 1
                    index = index+1
            Af = sc.lil_matrix((a*b,a*b))
            Af.setdiag(main_diag, k=0)
            Af.setdiag(p1_diag, k=1)
            Af.setdiag(m1_diag,k=-1)
            Af.setdiag(pa_diag, k=a)
            Af.setdiag(ma_diag, k=-a)

        bee = Tnow.reshape((a*b), order = 'F')
        Tret = np.array(Af.dot(bee))
        Tret = Tret.reshape((a,b), order = 'F')
        Tret[:,0] = Tret[:,2]
        Tret[:,-1] = Tret[:,-3]
        if ~np.isnan(q).any():
            Tret[-1,:] = Tret[-2,:]+ (q*dy/k[-1,:])
        return Tret

@jit
def conv_chain_solve(k, a, b, dx, dy, dt, Tf, H, q = np.nan):
    """
    Solver for the heat diffusion equation (expanded via the chain rule) based on convolution method - faster when inhomogenous time varying permeability is used
    k = Diffusivity field (anisotropic) MxN matrix
    a = number of rows - M int
    b = number of columns - N int
    dx = spacing in x direction - int
    dy = spacing in y direction - int
    dt = time step - int
    Tf = temperature field at current time step - MxN matrix
    q = Heat flux at the bottom boundary - N int If q is left as nan, the boundary condition changes to Dirichlet i.e., constant temp
    """
    
    Tnow = np.zeros((a,b))
    T_surf = Tf[0,0]
    T_bot = Tf[-1,0]
    for i in range(1,a-1):
        for j in range(1,b-1):
            Tnow[i,j] =  (-((((2*k[i,j])/dx**2)+((2*k[i,j])/dy**2))*dt+(H[i,j]*dt))+1)*Tf[i,j] + (((k[i,j]/dx**2)+((k[i+1,j]-k[i-1,j])/(4*dx**2)))*dt)*Tf[i+1,j] + (((k[i,j]/dx**2)-((k[i+1,j]-k[i-1,j])/(4*dx**2)))*dt)*Tf[i-1,j] + (((k[i,j]/dy**2)+((k[i+1,j]-k[i-1,j])/(4*dy**2)))*dt)*Tf[i,j+1] + (((k[i,j]/dy**2)-((k[i+1,j]-k[i-1,j])/(4*dy**2)))*dt)*Tf[i,j-1]
    for i in range(1,a-2):
        Tnow[i,0] = Tnow[i,2]
        Tnow[i,b-1] = Tnow[i,b-3]
    Tnow[0,:] = T_surf
    if (np.isnan(np.array(q)).any()):
        #Tnow[-1,:] = Tnow[-2,:]+ (0.02857*dy)
        Tnow[-1,:] = T_bot
    else:
        Tnow[-1,:] = Tnow[-2,:]+ (q*dy/k[-1,:])
    return Tnow

@jit
def conv_smooth_solve(k, a, b, dx, dy, dt, Tf, H, q = np.nan):
    """
    Solver for the heat diffusion equation (expanded via averaging permeability) based on convolution method - faster when inhomogenous time varying permeability is used
    k = Diffusivity field (anisotropic) MxN matrix
    a = number of rows - M int
    b = number of columns - N int
    dx = spacing in x direction - int
    dy = spacing in y direction - int
    dt = time step int
    Tf = temperature field at current time step - MxN matrix
    q = Heat flux at the bottom boundary - N int If q is left as nan, the boundary condition changes to Dirichlet i.e., constant temp
    """
    kiph, kimh, kjph, kjmh = avg_perm(k)
    Tnow = np.zeros((a,b))
    T_surf = Tf[0,0]
    T_bot = Tf[-1,0]
    for i in range(1,a-1):
        for j in range(1,b-1):
            Tnow[i,j] =  (-(((kiph[i,j]+kimh[i,j])*(dt/(dx**2)))+((kjph[i,j]+kjmh[i,j])*(dt/(dy**2)))+(H[i,j]*dt))+1)*Tf[i,j] + (kiph[i,j]*(dt/(dx**2)))*Tf[i+1,j] + (kimh[i,j]*(dt/(dx**2)))*Tf[i-1,j] + (kjph[i,j]*(dt/(dy**2)))*Tf[i,j+1] + (kjmh[i,j]*(dt/(dy**2)))*Tf[i,j-1]
    for i in range(1,a-2):
        Tnow[i,0] = Tnow[i,2]
        Tnow[i,b-1] = Tnow[i,b-3]
    Tnow[0,:] = T_surf
    if (np.array(np.isnan(np.array(q))).any()):
        Tnow[-1,:] = T_bot
    else:
        Tnow[-1,:] = Tnow[-2,:]+ (q*dy/k[-1,:])
    return Tnow

def diff_solve(k, a, b, dx, dy, dt, Tnow, q, method, H, k_const=False):
    """
    Switch function to call particular solver and pass parameters. This functions also constructs the weight matrix for the linear algebra solvers if the diffusivity is kept constant and not time varying
    k = Diffusivity field (anisotropic) MxN matrix
    a = number of rows - M int
    b = number of columns - N int
    dx = spacing in x direction - int
    dy = spacing in y direction - int
    dt = time step - int
    Tnow = temperature field at current time step - MxN matrix
    q = Heat flux at the bottom boundary - N int If q is left as nan, the boundary condition changes to Dirichlet i.e., constant temp
    """
    if method=='conv chain':
        Tnow=conv_chain_solve(k, a, b, dx, dy, dt, Tnow, H, q)
        return Tnow
    elif method=='conv smooth':
        Tnow=conv_smooth_solve(k, a, b, dx, dy, dt, Tnow, H, q)
        return Tnow
    elif method=='smooth':
        if k_const:
            kiph, kimh, kjph, kjmh = avg_perm(k)
            main_diag = np.zeros(a*b)
            p1_diag = np.zeros(a*b - 1)
            m1_diag = np.zeros(a*b-1)
            pa_diag = np.zeros(a*b-a)
            ma_diag = np.zeros(a*b-a)
            index = 0



            for j in range(0,b):
                for i in range(0,a):
                    if i>0 and i<a-1 and j>0 and j<b-1:
                        main_diag[index] = -(((kiph[i,j]+kimh[i,j])*(dt/(dx**2)))+((kjph[i,j]+kjmh[i,j])*(dt/(dy**2))))+1
                        pa_diag[index] = kiph[i,j]*(dt/(dx**2))
                        ma_diag[index-a] = kimh[i,j]*(dt/(dx**2))
                        p1_diag[index] = kjph[i,j]*(dt/(dy**2))
                        m1_diag[index-1] = kjmh[i,j]*(dt/(dy**2))
                    if i==0 or i==a-1:
                        main_diag[index] = 1
                    if j==0 and i!=0 and i!=a-1:
                        main_diag[index] = 1
                    if j==b-1 and i!=0 and i!=a-1:
                        main_diag[index] = 1
                    index = index+1

            Af = sc.lil_matrix((a*b,a*b))
            Af.setdiag(main_diag, k=0)
            Af.setdiag(p1_diag, k=1)
            Af.setdiag(m1_diag,k=-1)
            Af.setdiag(pa_diag, k=a)
            Af.setdiag(ma_diag, k=-a)
        else:
            Af = np.nan

        return perm_smoothed_solve(k, a, b, dx, dy, dt, Tnow, q, Af, H)
    elif method=='chain':
        if k_const:
            main_diag = np.zeros(a*b)
            p1_diag = np.zeros(a*b - 1)
            m1_diag = np.zeros(a*b-1)
            pa_diag = np.zeros(a*b-a)
            ma_diag = np.zeros(a*b-a)

            index = 0

            for j in range(0,b):
                for i in range(0,a):
                    if i>0 and i<a-1 and j>0 and j<b-1:
                        main_diag[index] = -((((2*k[i,j])/dx**2)+((2*k[i,j])/dy**2))*dt)+1
                        pa_diag[index] = ((k[i,j]/dx**2)+((k[i+1,j]-k[i-1,j])/(4*dx**2)))*dt
                        ma_diag[index-a] = ((k[i,j]/dx**2)-((k[i+1,j]-k[i-1,j])/(4*dx**2)))*dt
                        p1_diag[index] = ((k[i,j]/dy**2)+((k[i+1,j]-k[i-1,j])/(4*dy**2)))*dt
                        m1_diag[index-1] = ((k[i,j]/dy**2)-((k[i+1,j]-k[i-1,j])/(4*dy**2)))*dt
                    if i==0 or i==a-1:
                        main_diag[index] = 1
                    if j==0 and i!=0 and i!=a-1:
                        main_diag[index] = 1
                    if j==b-1 and i!=0 and i!=a-1:
                        main_diag[index] = 1
                    index = index+1
            #Af = sc.spdiags([main_diag, p1_diag, m1_diag, pb_diag, mb_diag], [0,1,-1,a,-a])
            Af = sc.lil_matrix((a*b,a*b))
            Af.setdiag(main_diag, k=0)
            Af.setdiag(p1_diag, k=1)
            Af.setdiag(m1_diag,k=-1)
            Af.setdiag(pa_diag, k=a)
            Af.setdiag(ma_diag, k=-a)
        else:
            Af = np.nan
        return perm_chain_solve(k, a, b, dx, dy, dt, Tnow, q, Af, H)
    else:
        print('Error method not supported')
        exit()

## Side - Add side flux BC later
## New list -
## 1 - Add functions for temperature and lithology dependent density and diffusivity