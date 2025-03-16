import numpy as np
from numba import jit
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sc
import pypardiso as ps
from scipy.special import erf, erfinv
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.spatial import KDTree
import pandas as pd
import pyvista as pv
import os
import h5py
import warnings
from autograd import elementwise_grad
import autograd.numpy as anp
import pdb
import utilities as util


class cool:
    def __init__(self):
        pass
    ###Functions to cool magma bodies###

    @staticmethod
    def is_nan(q):
        '''
        Alternative implementation of numpy's numpy.isnan. Originally created to allow for a nopython implementation in numba.
        '''
        r = 0
        for i in range(0, len(q)):
            if np.isnan(q[i]):
                r = 1
                break
        if r==1:
            return True
        else:
            return False

    @staticmethod
    @jit
    def avg_perm(k):
        ##Function to return the averaged permeabilities required for heat diffusion in an anisotropic medium
        kiph = k.copy()
        kimh = k.copy()
        kjph = k.copy()
        kjmh = k.copy()

        for i in range(0, len(k[:,1])-1):
            for j in range(0, len(k[1,:])-1):
                kiph[i,j] = (k[i,j]+k[i+1,j])/2 #Averaging permeabilities beterrn i and i+1
                kjph[i,j] = (k[i,j]+k[i,j+1])/2 #Averaging permeabilities between j and j+1

        for i in range(1,len(k[:,1])-1):
            for j in range(1,len(k[1,:])-1):
                kimh[i,j] = (k[i,j]+k[i+1,j])/2 #Averaging permeabilities between i and i-1
                kjmh[i,j] = (k[i,j]+k[i,j+1])/2 #Averaging permeabilities between j and j-1

        return kiph, kimh, kjph, kjmh

    def cheat_solver(self, Tf, a, q):
        """
        Quick approximate alternative to set up initial temperature grid while testing code to avoid lengthy set up times for large grids
        """
        if self.is_nan(q):
            T_top = Tf[0,0]
            grad = (Tf[-1,0]-Tf[0,0])/a #Solve for temperature based on thermal gradient and depth
            for i in range(0,a):
                Tf[i,:] = T_top + i*grad
        else:
            for i in range(1, a):
                Tf[i,:] = Tf[i-1,:]+(q/31.532) #This implementation is not recommended. Please use other solvers for custom qs
        return Tf

    def straight_solver(self, Ab,dee, a, b):
        """
        Inverse matrix multiplication for small sized matrices - fast implementation due to parallelization in pypardiso
        Ab = LHS weight matrix MxN
        dee = RHS matrix M*Nx1
        a = number of rows - M int
        b = number of columns N int
        """
        Tf = np.array(ps.spsolve(sc.csc_matrix(Ab), dee)) #Matrix multiplication for solving for temperature
        Tf = Tf.reshape((a,b), order = 'F')
        return Tf
    ###Alternative functions to get initial thermal state. it is important to note that the straight solver will always be the fastest since there is only one matrix multiplication to perform###
    def JacobianIt(self, Ab, dee, a, b):
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
        do = sc.csc_matrix(ps.spsolve(D, sc.csc_matrix((E+F)))) #If ps.spsolve does not work, change to sp.linalg.spsolve (the scipy non-parallelized implementation)
        T = dee
        err = 1e10
        iter = 10000
        c = 0
        while c<iter and err>1e-3:
            c = c+1
            T_new = do.dot(T)+ ps.spsolve(D, dee) #Same here: Change to ps.spsolve
            err = np.max(T_new - T)
            print(err, 'ln 109')
            T = T_new

        T = T.reshape((a,b), order = 'F')
        print('Jacobian Iterations completed: ', c)
        return T

    def GSIt(self, Ab, dee, a, b):
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


    def heat_flux(self, k, a, b, dx, dy, Tnow, method, q = np.nan):
        """
        Fourier's law solver switch function. Choos between the straight, Jacobian, Gauss-Seidell and cheat implementations
        method = straight/ Jacobian
        k = Diffusivity field (anisotropic) MxN matrix
        a = number of rows - M int
        b = number of columns N int
        dx = spacing in x direction int
        dy = spacing in y direction int
        Tnow = temperature field at current time step MxN matrix - Note that you need to set the Dirichlet boundary values
        """
        if method == 'cheat':
                return self.cheat_solver(Tnow,a, q)
        else:
            if ~np.isnan(q).any():
                Tnow[-1,:] = q*dy/k[-1,:]
            bee = Tnow.reshape((a*b), order = 'F')
            #Set up the weight matrix for each node that takes into account the boundary conditions and the temperature of the adjacenet nodes
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
            Af.setdiag(pa_diag, k=a)
            Af.setdiag(ma_diag, k=-a)

            if method =='straight':
                return self.straight_solver(Af, bee, a, b)
            elif method == 'Jacobian':
                return self.JacobianIt(Af, bee, a, b)
            elif method == 'GS':
                return self.GSIt(Af, bee, a, b)
            else:
                raise ValueError(f'method must be one of straight, Jacobian, GS or cheat, but is {method}')





    def perm_smoothed_solve(self, k, a, b, dx, dy, dt, Tnow, q, Af, H):
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
        H_rad = H[0]
        H_lat = H[1] #This is the latent heat solution that should be divided and not the actual latent heat of crystalization. See get_latH for details
        if np.isnan(Af):
            #Generate the weight matrix if the thermal diffusivity is not constant and needs to be changed at every time step
            kiph, kimh, kjph, kjmh = self.avg_perm(k)
            main_diag = np.zeros(a*b)
            p1_diag = np.zeros(a*b - 1)
            m1_diag = np.zeros(a*b-1)
            pa_diag = np.zeros(a*b-a)
            ma_diag = np.zeros(a*b-a)
            index = 0

            for j in range(0,b):
                for i in range(0,a):
                    if i>0 and i<a-1 and j>0 and j<b-1:
                        main_diag[index] = -((((kiph[i,j]+kimh[i,j])*(dt/(dx**2)))+((kjph[i,j]+kjmh[i,j])*(dt/(dy**2)))+ (H_rad[i,j]))+1)/H_lat[i,j]
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
        Tret = np.array(Af.dot(bee)) #Perform the matrix multiplication
        Tret = Tret.reshape((a,b), order = 'F')
        Tret[:,0] = Tret[:,2]
        Tret[:,-1] = Tret[:,-3]
        if ~np.isnan(q).any():
            Tret[-1,:] = Tret[-2,:]+ (q*dy/k[-1,:])
        return Tret

    def perm_chain_solve(self, k, a, b, dx, dy, dt, Tnow, q, Af, H):
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
        H_rad = H[0]
        H_lat = H[1] #This is the latent heat solution that should be divided and not the actual latent heat of crystalization. See get_latH for details
        k = k/H_lat
        if np.isnan(Af):
            #Generate the weight matrix if the thermal diffusivity is not constant and needs to be changed at every time step
            main_diag = np.zeros(a*b)
            p1_diag = np.zeros(a*b - 1)
            m1_diag = np.zeros(a*b-1)
            pa_diag = np.zeros(a*b-a)
            ma_diag = np.zeros(a*b-a)

            index = 0

            for j in range(0,b):
                for i in range(0,a):
                    if i>0 and i<a-1 and j>0 and j<b-1:
                        main_diag[index] = -(((((2*k[i,j])/dx**2)+((2*k[i,j])/dy**2)+ H_rad[i,j]))+1)/H_lat[i,j] 
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
        Tret = np.array(Af.dot(bee)) #Perform matrix multiplication
        Tret = Tret.reshape((a,b), order = 'F')
        Tret[:,0] = Tret[:,2]
        Tret[:,-1] = Tret[:,-3]
        if ~np.isnan(q).any():
            Tret[-1,:] = Tret[-2,:]+ (q*dy/k[-1,:])
        return Tret

    @jit(forceobj = True)
    def conv_chain_solve(self, k, a, b, dx, dy, dt, Tf, H, q = np.nan):
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
        H_rad = H[0]
        H_lat = H[1]
        Tnow = np.zeros((a,b))
        T_surf = Tf[0,0]
        T_bot = Tf[-1,0]
        for i in range(1,a-1):
            for j in range(1,b-1):
                Tnow[i,j] =  ((-((((2*k[i,j])/dx**2)+((2*k[i,j])/dy**2))*dt)+1)*Tf[i,j] + (((k[i,j]/dx**2)+((k[i+1,j]-k[i-1,j])/(4*dx**2)))*dt)*Tf[i+1,j] + (((k[i,j]/dx**2)-((k[i+1,j]-k[i-1,j])/(4*dx**2)))*dt)*Tf[i-1,j] + (((k[i,j]/dy**2)+((k[i+1,j]-k[i-1,j])/(4*dy**2)))*dt)*Tf[i,j+1] + (((k[i,j]/dy**2)-((k[i+1,j]-k[i-1,j])/(4*dy**2)))*dt)*Tf[i,j-1]+(H_rad[i,j]))
        for i in range(1,a-1):
            Tnow[i,0] = Tnow[i,2]
            Tnow[i,b-1] = Tnow[i,b-3]
        Tnow[0,:] = T_surf
        if (np.isnan(np.array(q)).any()):
            #Tnow[-1,:] = Tnow[-2,:]+ (0.02857*dy)
            Tnow[-1,:] = T_bot
        else:
            Tnow[-1,:] = Tnow[-2,:]+ (q*dy/k[-1,:])
        return Tnow

    #@jit(forceobj = True)
    def conv_smooth_solve(self, k, a, b, dx, dy, dt, Tf, H, q = np.nan):
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
        H_rad = H[0]
        H_lat = H[1]
        kiph, kimh, kjph, kjmh = self.avg_perm(k)/H_lat
        Tnow = np.zeros((a,b))
        T_surf = Tf[0,0]
        T_bot = Tf[-1,0]
        for i in range(1,a-1):
            for j in range(1,b-1):
                Tnow[i,j] =  ((-(((kiph[i,j]+kimh[i,j])*(dt/(dx**2)))+((kjph[i,j]+kjmh[i,j])*(dt/(dy**2))))+1)*Tf[i,j] + (kiph[i,j]*(dt/(dx**2)))*Tf[i+1,j] + (kimh[i,j]*(dt/(dx**2)))*Tf[i-1,j] + (kjph[i,j]*(dt/(dy**2)))*Tf[i,j+1] + (kjmh[i,j]*(dt/(dy**2)))*Tf[i,j-1]+(H_rad[i,j]))
        for i in range(1,a-1):
            Tnow[i,0] = Tnow[i,2]
            Tnow[i,b-1] = Tnow[i,b-3]
        Tnow[0,:] = T_surf
        if (np.array(np.isnan(np.array(q))).any()):
            Tnow[-1,:] = T_bot
        else:
            Tnow[-1,:] = Tnow[-2,:]+ (q*dy/k[-1,:])
        return Tnow
    
    @staticmethod
    def func_assigner(func, *args, **kwargs):
        '''
        Function that dynamically calls a given function with specified positional and keyword arguments, returning the result of the function call.
        func: The function to be called.
        *args: Positional arguments to be passed to the function.
        **kwargs: Keyword arguments to be passed to the function.
        '''
        result = func(*args,**kwargs)
        return result
    
    @staticmethod
    def calcF(T_field, T_solidus, T_liquidus):
        '''
        Arbitrary method to calculate the fraction of melt remaining based on temperature
        T_field: A 2D numpy array representing the temperature field.
        T_liquidus: An optional float representing the liquidus temperature, default is 1250.
        T_solidus: An optional float representing the solidus temperature, default is 800.
        '''
        def safe_tanh(x):
            """Clip inputs to tanh to avoid overflow in cosh during gradients."""
            x_clipped = anp.clip(x, -50, 50)  # Prevents overflow in cosh
            return anp.tanh(x_clipped)

        def smooth_step(x, lower_bound, upper_bound, steepness=20):
            """Smoothly transitions between 0 and 1 using tanh."""
            return 0.5 * (safe_tanh((x - lower_bound) / (upper_bound - lower_bound + 1e-12) * steepness) + 1)
        # Avoid division-by-zero by ensuring T_liquidus > T_solidus
        delta = max(0.05 * (T_liquidus - T_solidus), 1e-6)  # Minimum delta to avoid collapse
        
        # Smooth masks for transitions around T_solidus and T_liquidus
        mask_solidus = smooth_step(T_field, T_solidus - delta, T_solidus + delta)
        delta_liquidus = max(0.25 * (T_liquidus - T_solidus), 1e-6)  # Wider transition (5% vs. 1%)
        mask_liquidus = smooth_step(T_field, T_liquidus - delta_liquidus, T_liquidus + delta_liquidus, steepness=20)
        
        # Safe computation of p_values (non-negative input for exponentiation)
        k = 1  # Controls steepness of softplus
        smoothed_term = anp.log(1 + anp.exp(k * (T_field - T_solidus))) / k  # Always ≥ 0
        smoothed_term_lower = anp.log(1 + anp.exp(k*(T_liquidus - T_solidus)))
        p_values = (smoothed_term / smoothed_term_lower) ** 2.5
        p_values = p_values * mask_solidus  # Suppress values below T_solidus
        
        # Combine masks to compute F
        F = (1 - mask_liquidus) * p_values + mask_liquidus *safe_tanh((T_field-T_solidus)/2)
        return F 
        
    @staticmethod
    def calcF_from_csv(T_field, dir_csv, temp_col: str, fraction_column: str, T_liquidus, T_solidus):
        def smooth_step(val, lower_bound, upper_bound, steepness=10):
            """Smoothly transitions between 0 and 1 using tanh."""
            return 0.5 * (anp.tanh((val - lower_bound) / (upper_bound - lower_bound + 1e-12) * steepness) + 1)
        # Avoid division-by-zero by ensuring T_liquidus > T_solidus
        delta = max(0.01 * (T_liquidus - T_solidus), 1e-6)  # Minimum delta to avoid collapse
        
        # Smooth masks for transitions around T_solidus and T_liquidus
        mask_solidus = smooth_step(T_field, T_solidus - delta, T_solidus + delta)
        mask_liquidus = smooth_step(T_field, T_liquidus - delta, T_liquidus + delta)
        data = pd.read_csv(dir_csv)
        temp = data[temp_col]
        fraction_melt = data[fraction_column]
        p = interp1d(temp, fraction_melt)
        p_values = p(T_field)* mask_solidus  # Suppress values below T_solidus
        F = (1 - mask_liquidus) * p_values + mask_liquidus 
        return F



    @staticmethod
    def get_latH(T_field, lithology, melt='basalt', specific_heat = 850, L = 4e5, T_liquidus=1100, T_solidus=800, curve_func = None, args = None):
        '''
        Get the latent heat of crystallization term for the ehat diffusion equation based onthe model of Karakas et al. (2017)
        T_field: A 2D numpy array representing the temperature field.
        lithology: A 2D numpy array representing the lithology types.
        melt: A string specifying the type of melt, default is 'basalt'.
        rho_melt: A float representing the density of the melt, default is 2850 kg/m³.
        T_liquidus: A float representing the liquidus temperature, default is 1250.
        T_solidus: A float representing the solidus temperature, default is 800.
        '''
        H_lat = np.ones_like(T_field)
        if args is None:
            args = (T_liquidus, T_solidus)
        if curve_func is None:
            curve_func = cool.calcF
        phi_cr = elementwise_grad(curve_func)
        H_lat[lithology==melt] = (1 + (phi_cr(T_field[lithology==melt] , *args)*L/specific_heat))
        H_lat[lithology!=melt] = 1
        return H_lat
    
    @staticmethod
    def convert_latH_to_J(H_lat, specific_heat, cooling_rate):
        '''
        Function to get actual latent heat if needed
        '''
        a,b = H_lat.shape
        for i in range(a):
            for j in range(b):
                if H_lat[i,j]!=0:
                    H_lat[i,j] = (1-H_lat[i,j]*cooling_rate)*specific_heat
        return H_lat
    
    @staticmethod
    def get_radH(T_field, rho, dx):
        '''
        Function to get radioactive heat release
        T_field: A 2D numpy array representing the temperature field.
        rho: A 2D numpy array representing the density at each point in the field.
        dx: A scalar representing the grid spacing in the vertical direction.

        '''
        a, b = T_field.shape
        Ho = 8e-10 #W/kg
        Lc = 12000 #m
        depth = np.array([i*dx for i in range(a)])
        H = np.zeros((a,b))
        for i in range(a):
            for j in range(b):
                H[i,j] = Ho*rho[i,j]*np.exp(-depth[i]/Lc)
        return H
    
    @staticmethod
    def get_conductivity(T_field, rock, density, dy):
        '''
        Stand-in funciton to get diffusivity based on properties
        '''
        diffusivity = 1/(0.24094+(4.6019e-4*(T_field+273.15)))
        return diffusivity
    
    @staticmethod
    def get_specific_heat(T_field, rock, density, dy):
        a,b = density.shape
        pressure = np.zeros((a,b))
        for i in range(1,a):
            for j in range(b):
                pressure[i,j] = pressure[i-1,j] + density[i,j]*9.8*dy

        a_ign = 1.02*1e3
        b_ign = 4.26*1e3
        c_ign = -49.8*1e3
        d_ign = 1429.42*1e3
        e_ign = -17692.69*1e3

        a_sed = 1.49*1e3
        b_sed = -9.14*1e3
        c_sed = 58.03*1e3
        d_sed = -1465*1e3
        e_sed = 17691.52*1e3

        A = np.zeros_like(T_field)
        B = np.zeros_like(T_field)
        C = np.zeros_like(T_field)
        D = np.zeros_like(T_field)
        E = np.zeros_like(T_field)

        igneous = rock=='granite' | rock=='basalt' | rock=='peridotite'
        
        A[igneous] = a_ign
        B[igneous] = b_ign
        C[igneous] = c_ign
        D[igneous] = d_ign
        E[igneous] = e_ign

        sedimentary = rock=='sandstone' | rock=='shale' | rock=='limestone' | rock=='marl' | rock=='evaporite'

        A[sedimentary] = a_sed
        B[sedimentary] = b_sed
        C[sedimentary] = c_sed
        D[sedimentary] = d_sed
        E[sedimentary] = e_sed

        Cp = A + (B*np.sqrt(T_field)) + (C/T_field) + (D/(T_field**2)) + (E/(T_field**3))
        return Cp    
    @staticmethod
    def get_k(thermal_conductivity, density, specific_heat):
        k = thermal_conductivity/density/specific_heat
        return k

    def diff_solve(self, k, a, b, dx, dy, dt, Tnow, q, method, H, k_const=False):
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
            Tnow=self.conv_chain_solve(k, a, b, dx, dy, dt, Tnow, H, q)
            return Tnow
        elif method=='conv smooth':
            Tnow=self.conv_smooth_solve(k, a, b, dx, dy, dt, Tnow, H, q)
            return Tnow
        elif method=='smooth':
            if k_const:
                kiph, kimh, kjph, kjmh = self.avg_perm(k)/H[1]
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

            return self.perm_smoothed_solve(k/H[1], a, b, dx, dy, dt, Tnow, q, Af, H)
        elif method=='chain':
            if k_const:
                k = k/H[1]
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
            return self.perm_chain_solve(k/H[1], a, b, dx, dy, dt, Tnow, q, Af, H)
        else:
            raise ValueError('diff_solve solution method not supported')

class emit:
    def __init__(self):
        #Initilize the class. No variables needed
        pass
    
    @staticmethod
    def get_init_CO2_percentages(T_field, lithology, density, dy):
        '''
        The get_init_CO2_percentages method calculates the initial exsolved CO2 percentages for different lithologies based on temperature and pressure conditions 
        i.e., ensuring the CO2 is in equilibrium with the PT conditions. 
        It uses interpolation to estimate CO2 values from pre-loaded data for specific rock types.
        T_field: 2D array representing the temperature field.
        lithology: 2D array indicating the type of rock at each location.
        density: 2D array of rock densities.
        dy: Scalar representing the vertical distance between layers.
        '''
        a,b = lithology.shape
        break_parser = (lithology=='dolostone') | (lithology=='limestone') | (lithology=='marl') | (lithology=='evaporite')
        #Read in the data for temeprature pressure stability
        dolo = loadmat('dat/Dolostone.mat')
        evap = loadmat('dat/DolostoneEvaporite.mat')
        marl = loadmat('dat/Marl.mat')
        T = np.array(dolo['Dolo']['T'][0][:][0])
        P = np.array(dolo['Dolo']['P'][0][0][0])
        T = T[:,0]
        dolo_CO2 = np.array(dolo['Dolo']['CO2'][0][0])
        evap_CO2 = np.array(evap['Dol_ev']['CO2'][0][0])
        marl_CO2 = np.array(marl['Marl']['CO2'][0][0])
        #Create the regular grid only if the rock type exists. This is to speed up the operation
        if lithology.any()=='dolostone' or lithology.any()=='limestone':
            dolo_inter = RegularGridInterpolator((T,P), dolo_CO2)
        if lithology.any()=='evaporite':
            evap_inter = RegularGridInterpolator((T,P), evap_CO2)
        if lithology.any()=='marl':
            marl_inter = RegularGridInterpolator((T,P), marl_CO2)
        init_CO2 = np.zeros_like(T_field)
        for i in range(a):
            for j in range(b):
                if break_parser[i,j]:
                    pressure = 0
                    for l in range(0,i):
                        pressure = pressure + (density[l,j]*9.8*dy) #Getting lithostatic pressure upto this point
                    pressure = pressure*1e-5 #conversion from Pa to bar
                    pressure = 1 if pressure==0 else pressure
                    if lithology[i,j]=='dolostone' or lithology[i,j]=='limestone':
                        try:
                            init_CO2[i,j] = dolo_inter([T_field[i,j],pressure])
                        except ValueError:
                            init_CO2[i,j] = 0
                            print('Warning: Limestone or dolostone pressure out of bounds. Skipping')

                    elif lithology[i,j] == 'evaporite':
                        init_CO2[i,j] = evap_inter([T_field[i,j],pressure])
                    elif lithology[i,j]=='marl':
                        init_CO2[i,j]== marl_inter([T_field[i,j],pressure])
        return init_CO2
    @staticmethod
    def get_breakdown_CO2(T_field, lithology, density, breakdownCO2, dy, dt):
        '''
        Calculate the amount of CO2 exsolved from the breakdown of carbonate rocks
        T_field: 2D array representing the temperature field.
        lithology: 2D array indicating the type of rock at each location.
        density: 2D array of rock densities.
        breakdownCO2: 2D array of initial CO2 breakdown values.
        dy: Scalar representing the vertical distance between layers.
        dt: Scalar representing the time step.
        '''
        break_parser = (lithology=='dolostone') | (lithology=='limestone') | (lithology=='marl') | (lithology=='evaporite')
        a, b = T_field.shape
        #Read in the data for percentage CO2 released at the temeprature and pressure grid
        dolo = loadmat('dat/Dolostone.mat')
        evap = loadmat('dat/DolostoneEvaporite.mat')
        marl = loadmat('dat/Marl.mat')
        T = np.array(dolo['Dolo']['T'][0][:][0])
        P = np.array(dolo['Dolo']['P'][0][0][0])
        T = T[:,0]
        dolo_CO2 = np.array(dolo['Dolo']['CO2'][0][0])
        evap_CO2 = np.array(evap['Dol_ev']['CO2'][0][0])
        marl_CO2 = np.array(marl['Marl']['CO2'][0][0])
        #Create the P, T grid for for interpolation
        dolo_inter = RegularGridInterpolator((T,P), dolo_CO2)
        evap_inter = RegularGridInterpolator((T,P), evap_CO2)
        marl_inter = RegularGridInterpolator((T,P), marl_CO2)
        curr_breakdown_CO2 = np.zeros_like(T_field)
        for i in range(a):
            for j in range(b):
                if break_parser[i,j]:
                    #Calculate breakdown CO2, only for carbonate lithology nodes
                    pressure = 0
                    for l in range(0,i):
                        pressure = pressure + (density[l,j]*9.8*dy) #Getting lithostatic pressure upto this point
                    pressure = pressure*1e-5 #conversion from Pa to bar
                    pressure = 1 if pressure==0 else pressure #Change pressure on the surface to a small non-zero value
                    if lithology[i,j]=='dolostone' or lithology[i,j]=='limestone':
                        try:
                            curr_breakdown_CO2[i,j] = dolo_inter([T_field[i,j],pressure]) #Get breakdown CO2 using interpolation
                        except ValueError:
                            curr_breakdown_CO2[i,j] = 0
                            print(f'Warning: Limestone pressure out of bounds at {i} {j}. Skipping') #Change value to zero if the T and P are out of the grid range. This usually happens at deeper conditions, where limestone is not stable.

                    elif lithology[i,j] == 'evaporite':
                        try:
                            curr_breakdown_CO2[i,j] = evap_inter([T_field[i,j],pressure])
                        except ValueError:
                            curr_breakdown_CO2[i,j] = 0
                            print(f'Warning: Evaporite pressure out of bounds at {i} {j}. Skipping') #Change value to zero if the T and P are out of the grid range. This usually happens at deeper conditions, where evaporite is not stable.
                    elif lithology[i,j]=='marl':
                        try:
                            curr_breakdown_CO2[i,j]== marl_inter([T_field[i,j],pressure])
                        except ValueError:
                            curr_breakdown_CO2[i,j] = 0
                            print(f'Warning: Marl pressure out of bounds at {i} {j}. Skipping') #Change value to zero if the T and P are out of the grid range. This usually happens at deeper conditions, where marl is not stable.
        max_breakdown_co2 = np.maximum(breakdownCO2, curr_breakdown_CO2)
        try:
            for i in range(a):
                for j in range(b):
                    curr_breakdown_CO2[i, j] = np.minimum((curr_breakdown_CO2[i, j] - breakdownCO2[i, j]), 0) if curr_breakdown_CO2[i, j] > breakdownCO2[i, j] else 0
        except TypeError as e:
            print(e)
            print('Function outputs two arrays')
        RCO2_breakdown = (curr_breakdown_CO2)/dt
        return RCO2_breakdown, max_breakdown_co2

    
    @staticmethod
    @jit(forceobj=True)
    def SILLi_emissions(T_field, density, lithology, porosity, TOC_prev, dt, TOCo=np.nan, W=np.nan):
        '''
        Python implementation of SILLi (Iyer et al. 2018) based on the EasyRo% method of Sweeney and Burnham (1990)
        T_field - temperature field (array)
        dT = Rate of cooloing array
        density - Rock density array
        lithology - Lithology array
        porosity - porosity array
        '''
        calc_parser = (lithology=='shale') | (lithology=='sandstone')
        break_parser = (lithology=='dolostone') | (lithology=='limestone') | (lithology=='marl') | (lithology=='evaporite')
        calc_parser = calc_parser | break_parser
        a,b = T_field.shape
            
        A = 1e13
        R = 8.314 #J/K/mol
        E = np.array([34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72])*4184 #J/mole
        f = np.array([0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06, 0.04, 0.04, 0.07, 0.06, 0.06, 0.06, 0.05, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01])
        T_field = T_field + 273.15
        if np.isnan(W).all():
            W = np.ones((len(E),a, b))
            TOCo = TOC_prev
        if np.isnan(np.array(TOCo, dtype = float)).any():
            raise ValueError('TOCo cannot be NaN after the first time step')
        fl = np.zeros_like(W)
        for l in range(0, len(E)):
            k = A*np.exp(-E[l]/(R*T_field))
            W[l] = np.maximum(W[l]*np.exp(-k*dt),0)
            fl[l,:,:] = f[l]*(1-W[l,:,:])
        Frac = np.sum(fl, axis = 0)
        percRo = np.exp(-1.6+3.7*Frac) #vitrinite reflectance
        TOC = TOCo*(1-Frac)*calc_parser
        dTOC = (TOC_prev-TOC)/dt
        Rom = (1-porosity)*density*dTOC
        RCO2 = Rom*3.67/100
        return RCO2, Rom, percRo, TOC, W

    def analytical_Ro(T_field, dT, density, lithology, porosity, I_prev, TOC_prev, dt, TOCo, W):
        '''
        Untested function for the analytical solution of the Easy Ro model
        '''
        calc_parser = (lithology=='shale') | (lithology=='sandstone')
        a1 = 2.334733
        a2 = 0.250621
        b1 = 3.330657
        b2 = 1.681534
        A = 1e13
        R = 8.314 #J/K/mol
        E = [34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72]*4184 #J/mole
        f = [0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06, 0.04, 0.04, 0.07, 0.06, 0.06, 0.06, 0.05, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01]
        I_curr = np.empty_like(I_prev)
        del_I = np.empty_like(I_prev)
        w_ratio = np.empty_like(E)
        fl = np.empty_like(E)
        for l in range(len(I_prev[:,0,0])):
            Ert = E[l]/(R*T_field)
            I_curr[l,:,:] = T_field*A*np.exp(Ert)*(1-((Ert**2+(a1*Ert)+a2)/(Ert**2+(b1*Ert)+b2)))
            del_I[l] = (I_curr[l]-I_prev[l])/dT
            w_ratio[l] = np.maximum(np.exp(-del_I[l]),0)
            fl[l] = (1 - w_ratio[l])*f[l]
        Frac = 1 - np.sum(fl, axis = 0)
        percRo = np.exp(-1.6+3.7*Frac) #vitrinite reflectance
        TOC = TOCo*Frac*calc_parser
        dTOC = (TOC_prev-TOC)/dt
        Rom = (1-porosity)*density*dTOC
        RCO2 = Rom*3.67/100
        return RCO2, Rom, percRo, I_curr, TOC

    def analyticalRo_I(T_field):
        '''
        Initialization of I for the analytical solution of the EasyRo model
        '''

        a = len(T_field[:,0])
        b = len(T_field[0,:])
        A = 1e13
        a1 = 2.334733
        a2 = 0.250621
        b1 = 3.330657
        b2 = 1.681534
        R = 1.9872036e-3 #kcal/K/mol
        E = [34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72]*4184 #J/mole
        f = [0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06, 0.04, 0.04, 0.07, 0.06, 0.06, 0.06, 0.05, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01]
        I = np.empty(len(E),a,b)
        for l in range(len(E)):
            Ert = E[l]/(R*T_field)
            I[l,:,:] = T_field*A*np.exp(Ert)*(1-((Ert**2+(a1*Ert)+a2)/(Ert**2+(b1*Ert)+b2)))
        return I

    @staticmethod
    def get_sillburp_reaction_energies():
        '''
        Function to create the reaction energies required for sillburp model for thermogenic carbon emissions. 
        These generate normal distributions of activation energies for each component (Jonees et al. 2019)
        '''
        sqrt_2pi = np.sqrt(2 * np.pi)
        n_reactions = 4
        mean_E = [208e3, 279e3, 242e3, 230e3]  # mean activation energies for the reactions
        sd_E = [5e3, 13e3, 41e3, 5e3]  # Standard deviation of the normal distributions of the activation energies
        no_reactions = [7, 21, 55, 7]  # Number of reactions for each kerogen type
        reaction_energies = np.zeros((n_reactions, max(no_reactions)))
        
        for i in range(n_reactions):
            s_r2 = sd_E[i] * np.sqrt(2)
            N = no_reactions[i]
            fraction = 2 / N
            E_0 = 0
            E_1 = 0
            n_middle = N // 2
            
            for i_approx in range(n_middle, N):
                if i_approx == n_middle:
                    reaction_energies[i, i_approx] = mean_E[i]
                    if N != 1:
                        E_0 = mean_E[i] - s_r2 * erfinv(-1.0 / N)
                    continue
                if i_approx == N - 1:
                    reaction_energies[i, i_approx] = N * (sd_E[i] / sqrt_2pi * np.exp(-(mean_E[i] - E_0)**2 / (2.0 * sd_E[i]**2)) + mean_E[i] / 2.0 * (1.0 + erf((mean_E[i] - E_0) / s_r2)))
                else:
                    right_side = erf((mean_E[i] - E_0) / s_r2) - fraction
                    erf_inv = erfinv(right_side)
                    E_1 = mean_E[i] - erf_inv * s_r2
                    reaction_energies[i, i_approx] = N * (-sd_E[i] / sqrt_2pi *
                        (np.exp(-(mean_E[i] - E_1)**2 / (2.0 * sd_E[i]**2)) - np.exp(-(mean_E[i] - E_0)**2 / (2.0 * sd_E[i]**2))) -
                        mean_E[i] / 2.0 * (erf((mean_E[i] - E_1) / s_r2) - erf((mean_E[i] - E_0) / s_r2)))
                
                reaction_energies[i, N - i_approx - 1] = 2.0 * mean_E[i] - reaction_energies[i, i_approx]
                E_0 = E_1
        return reaction_energies
    @staticmethod
    @jit(forceobj = True)
    def sillburp(T_field, TOC_prev, density, lithology, porosity, dt, reaction_energies, TOCo=None, oil_production_rate=0, progress_of_reactions=np.nan, rate_of_reactions = np.nan, weights = None):
        '''
        Python implementation of the sillburp model of Jones et al. (2019)
        T_field - Temperature field array
        TOC_prev - Total organic content at the previous time step. If the first time step, leave TOCo blank and make this the TOC content
        density - Density array of the slice
        lithology - Lithology array of the slice
        porosity - Porosity field of the slice
        dt - Time step
        reaction_energies - Reaction energy arrays. These are most easily generated from the get_sillburp_reaction_energies() function. The model is designed to be used with these energies
        TOCo - Total organic carbon content of the slice at the initial time
        oil_production_rate - Rate of oil production as calculated by this model
        progress_reactions - The progress of reactions at the current time step. Leave blank at the initial time step
        rate_of_reactions - The rate of reaction progress at the current time step
        weights - Array containing the distribution of different components of reaction groups - The order of weights is ['LABILE', 'REFRACTORY', 'VITRINITE', 'OIL']
        '''
        #TOCo = np.array(TOCo, dtype=float)

        if TOCo is None:
            TOCo = TOC_prev
        
        a, b = T_field.shape
        calc_parser = (lithology == 'shale') | (lithology == 'sandstone')
        n_reactions = 4
        reactants = ['LABILE', 'REFRACTORY', 'VITRINITE', 'OIL']
        OIL = reactants.index('OIL')
        no_reactions = [7, 21, 55, 7]  # Number of reactions for each kerogen type
        
        As = [1.58e13, 1.83e18, 4e10, 1e13]  # pre-exponential constants for the different reactions
        if np.isnan(progress_of_reactions).all():
            progress_of_reactions = np.zeros((n_reactions, max(no_reactions), a, b))
            progress_of_reactions_old = np.zeros_like(progress_of_reactions)
            rate_of_reactions = np.zeros_like(progress_of_reactions)
            #oil_production_rate = np.zeros((a,b))
        else:
            progress_of_reactions_old = progress_of_reactions.copy()
        #S_over_k = np.zeros((a,b))
        
        do_labile_reaction = progress_of_reactions[reactants.index('LABILE'), :, :, :] < 1
        do_refractory_reaction = progress_of_reactions[reactants.index('REFRACTORY'), :, :, :] < 1
        do_oil_reaction = progress_of_reactions[reactants.index('OIL'), :, :, :] < 1
        do_vitrinite_reaction = progress_of_reactions[reactants.index('VITRINITE'), :, :, :] < 1
        do_reaction = np.array([do_labile_reaction, do_refractory_reaction, do_vitrinite_reaction, do_oil_reaction])
        mass_frac_labile_to_gas = 0.2
        
        for i in range(a):
            for j in range(b):
                for i_reaction in range(n_reactions):
                    for i_approx in range(no_reactions[i_reaction]):
                        if ~do_reaction[i_reaction, i_approx, i, j]:#.all():
                            continue
                        initial_product_conc = progress_of_reactions[i_reaction, i_approx, i, j]
                        activation_energy = reaction_energies[i_reaction, i_approx]
                        reaction_rate = As[i_reaction] * np.exp(-activation_energy / 8.314 / (T_field[i, j] + 273.15))
                        
                        if reactants[i_reaction] != 'OIL':
                            progress_of_reactions[i_reaction, i_approx, i, j] = min((1.0 - (1.0 - initial_product_conc) * np.exp(-reaction_rate * dt)), 1)
                            rate_of_reactions[i_reaction, i_approx, i, j] = (1.0 - initial_product_conc) * (1.0 - np.exp(-reaction_rate * dt)) / dt
                        else:
                            S_over_k = 0.0 if reaction_rate == 0 else oil_production_rate / reaction_rate / no_reactions[OIL]
                            progress_of_reactions[i_reaction, i_approx, i, j] = min(1.0 - S_over_k - (1.0 - initial_product_conc - S_over_k) * np.exp(-reaction_rate * dt), 1)
                            rate_of_reactions[i_reaction, i_approx, i, j] = (1.0 - initial_product_conc - S_over_k) * (1.0 - np.exp(-reaction_rate * dt)) / dt
                        
                        if i_reaction == reactants.index('LABILE'):
                            if i_approx == 0:
                                oil_production_rate = 0.0
                            oil_production_rate += -reaction_rate * (1.0 - progress_of_reactions[i_reaction, i_approx, i, j]) / no_reactions[reactants.index('LABILE')]
                            if i_approx == no_reactions[reactants.index('LABILE')] - 1:
                                oil_production_rate *= (1.0 - mass_frac_labile_to_gas)
        products_progress = np.zeros((n_reactions, a, b))
        if weights is None:
            for i_reaction in range(0,n_reactions):
                products_progress[i_reaction,:, :] = np.mean(progress_of_reactions[i_reaction,0:no_reactions[i_reaction],:,:], axis  = 0)
            products_progress = np.mean(products_progress, axis=0)
            #time_step_summarized = np.mean(time_step_progress, axis=0)
            #time_step_summarized = np.mean(time_step_summarized, axis=0)
            TOC = TOCo * (1-products_progress) * calc_parser
            dTOC = (TOC_prev - TOC)/dt
            Rom = (1 - porosity) * density * dTOC
            RCO2 = Rom * 3.67
        else:
            if weights.shape!=products_progress.shape:
                print(weights)
                raise IndexError(f'Shape of weights must be {products_progress.shape}')
            for i_reaction in range(0,n_reactions):
                products_progress[i_reaction,:,:] = np.mean(progress_of_reactions[i_reaction,0:no_reactions[i_reaction],:,:], axis = 0)
            products_progress = np.average(products_progress, axis=0, weights = weights)
            TOC = TOCo * (1-products_progress) * calc_parser
            dTOC = (TOC_prev - TOC)/dt
            Rom = (1 - porosity) * density * dTOC
            RCO2 = Rom * 3.67/100
        
        return RCO2, Rom, progress_of_reactions, oil_production_rate, TOC, rate_of_reactions


class rules:
    def __init__(self):
        pass
    @staticmethod
    def to_emplace(t_now, t_thresh):
        '''
        Boolean function that decides whether or not to emplace sills based on if the time since the last sill was emplaced exceeds the threshold
        t_now - time since last sill was emplaced
        t_thresh - threhold time for sill emplacement
        '''
        if (t_now<t_thresh):
            return False
        elif t_now>=t_thresh:
            return True

    @staticmethod
    def build_lith_dict(lithology):
        '''
        Function to build lithology dictionary (str to int) based on the lithology array provided. Values are assigned in order of appearance of new rock types
        '''
        a,b = lithology.shape
        lith_dict = {0:str(lithology[0,0])}
        n = 1
        for i in range(a):
            for j in range(b):
                if not (lithology[i,j] in lith_dict.values()):
                    lith_dict.update({n:lithology[i,j]})
                    n = n+1
        return lith_dict

    @staticmethod
    def build_prop_dict(prop, lithology):
        '''
        Function to build dictionary assigning the invariant properties with lithology based on the two arrays provided
        prop - 2D array containing the values of the particular proerty
        lithology - 2D array containing the rock at each position
        '''
        a,b = lithology.shape
        prop_dict = {lithology[0,0]: prop[0,0]}
        for i in range(a):
            for j in range(b):
                if not lithology[i,j] in prop_dict:
                    prop_dict.update({lithology[i,j]:prop[i,j]})
        return prop_dict
    @staticmethod
    def single_sill(T_field, x_space, height, width, thick, T_mag):
        """
        Emplacing a simple rectangular sill without a dike tail
        T_field: A 2D numpy array representing the temperature field.
        x_space: The x-coordinate for the center of the sill in nodes.
        height: The y-coordinate for the center of the sill in nodes.
        width: The width of the sill in nodes.
        thick: The thickness of the sill in nodes.
        T_mag: The temperature of the intruding magma.
        """
        T_field[int(height-(thick//2)):int(height+(thick//2)), int(x_space-(width//2)):int(x_space+(width//2))] = T_mag
        return T_field
    @staticmethod
    def circle_sill(T_field, x_space, height, r, T_mag, a, b, dx, dy):
        """
        Emplacing a simple circular sill without the dike tail
        T_field: 2D numpy array representing the temperature field.
        x_space: x-coordinate for the center of the sill in nodes.
        height: y-coordinate for the center of the sil in nodes.
        r: radius of the sill in nodes.
        T_mag: temperature magnitude of the sill.
        a: number of nodes in the y-direction.
        b: number of nodes in the x-direction.
        dx: spacing in the x-direction in m.
        dy: spacing in the y-direction in m.
        """
        x = np.arange(0,b*dx, dx)
        y = np.arange(0, a*dy, dy)
        for m in range(0, len(T_field[1,:])-1):
                for n in range(0, len(T_field[:,1])-1):
                    x_dist = ((x[m]-x[int(x_space)])**2)/(((r)//2)**2)
                    y_dist = ((y[n]-y[int(height)])**2)/(((r)//2)**2)
                    if (x_dist+y_dist)<=1:
                        T_field[n,m]=T_mag
        #T_field[int(height):-1,int(x_space)] = T_mag
        return T_field

    def randn_heights(self, n_sills, l_sill, h_sill, sd, dy):
        """
        Get random emplacement heights over a normal distribution within the specified range. Output is in nodes.
        n_sills - number of sills int
        l_sills - Depth of lowest sill emplacement range (m) int
        h_sills - Depth of shallowest depth emplacement range (m) int
        sd = Standard Deviation of the heights distribution
        dy - Grid spacing in the y direction (m) int
        """
        if h_sill<l_sill:
            pass
        else:
            print('l_sill:', l_sill)
            print('h_sill:', h_sill)
            raise ValueError('l_sill should be greater than h_sill')
        bean = np.mean([l_sill/dy, h_sill/dy])
        heights = np.round((sd/dy)*np.random.randn(n_sills) + bean)
        while ((heights>l_sill/dy).any() or (heights<h_sill/dy).any()):
            if (heights>l_sill/dy).any():
                heights[heights>l_sill/dy] = self.randn_heights(np.sum(heights>l_sill/dy), l_sill, h_sill, sd, dy)
            if (heights<h_sill/dy).any():
                heights[heights<h_sill/dy] = self.randn_heights(np.sum(heights<(h_sill/dy)), l_sill, h_sill, sd, dy)
        return heights

    def x_spacings(self, n_sills, x_min, x_max, sd, dx):
        """
        Nodes for x-coordinate space chosen as a random normal distribution
        n_sills - number of sills int
        x_min - The lower range (left side) (m) int
        x_max - The upper range (right side) (m) int
        sd - Standard deviation of the distribution. For the entire distribution to fit within the range, a maximum of 10% of the distribution is recommended. 
        """
        space = np.round((sd/dx)*np.random.randn(n_sills)+ np.mean([x_min/dx, x_max/dx]))
        while ((space>x_max/dx).any() or (space<x_min/dx).any()):
                if (space>x_max/dx).any():
                    space[space>x_max/dx] = self.x_spacings(np.sum(space>x_max/dx), x_min, x_max, sd, dx)
                if (space<x_min/dx).any():
                    space[space<x_min/dx] = self.x_spacings(np.sum((space<x_min/dx)), x_min, x_max, sd, dx)
        return space

    @staticmethod
    def uniform_heights(n_sills, l_sill, h_sill, dy):
        """
        Get heights spacing for sill emplacement randomly picked from a uniform distribution
        n_sills: Number of heights to generate.
        l_sill: Lower bound of the height range in m.
        h_sill: Upper bound of the height range in m.
        dy: Grid spacing in the y-direction.
        Returns emplacement depths in node spacings
        """
        heights = np.round(np.random.uniform(l_sill, h_sill, n_sills)/dy)
        return heights

    @staticmethod
    def uniform_x(n_sills, x_min, x_max, dx):
        """
        Nodes for x-coordinate space chosen as a random normal distribution
        n_sills: Number of x-coordinate nodes to generate.
        x_min: Minimum value of the x-coordinate range.
        x_max: Maximum value of the x-coordinate range.
        dx: Grid spacing in the x-direction.
        Returns x-coordinates in node spacings
        """
        space = np.round(np.random.uniform(x_min, x_max, n_sills)/dx)
        return space
    @staticmethod
    def empirical_CDF(n_sills, xarray, cdf):
        """
        Function to give random numbers from a specific empirical distribution
        n_sills - number of sills needed int
        xarray - array of domain for empirical CDF
        cdf - array of CDF for the x array
        Returns distribution in units of inputs
        """
        why = np.zeros(n_sills)
        for k in range(0,n_sills):
            a = np.random.uniform(0,1)
            gee = np.argmax(cdf>=a)-1
            why[k] = xarray[gee+1]
        return why
        
    @staticmethod
    def get_scaled_dims(min_min, min_max, mar, sar, heights, n_sills):
        """
        Linearly scaled with height (inversely) plus noise for both aspect ratio and shape
        Returns the width and height respectively in the number of nodes
        min_min = Minimum value for the thickness (m)
        min_max = Maximum value for the thickness (m)
        mar = Mean aspect ratio (Width/Thickness)
        sar = Standard deviation for the distribution of the aspect ratios
        n_sills = Number of sills
        dx = Node spacing in the x-direction
        dy = Node spacing in the y-direction
        Returns dims in length units
        """
        fact_min = ((min_max-min_min)/min_max)*((np.max(heights)-np.min(heights))/np.max(heights))
        major = np.zeros(n_sills)
        minor = np.zeros(n_sills)
        aspect_ratio = sar*np.random.randn(n_sills) + mar
        for i in range(0, n_sills):
            minor[i] = min_min + fact_min*heights[i] + np.round(2*np.random.randn())
            major[i] = minor[i]*aspect_ratio[i]
        return major, minor
    @staticmethod
    def randn_dims(min_min, min_max, sd_min, mar, sar, n_sills):
        """
        Random normal distribution of dims for aspect ratio and shape
        min_min = Minimum value for the thickness (m)
        min_max = Maximum value for the thickness (m)
        mar = Mean aspect ratio (Width/Thickness)
        sar = Standard deviation for the distribution of the aspect ratios
        n_sills = Number of sills
        Returns dims in length units
        """
        aspect_ratio = sar*np.random.randn(n_sills) + mar
        for i in range(len(aspect_ratio)):
            if aspect_ratio[i]<0:
                aspect_ratio[i] = np.abs(aspect_ratio[i])
        minor = np.round(sd_min*np.random.randn(n_sills)+np.mean([min_min, min_max]))
        while ((minor>min_max).any() or (minor<min_min).any()):
                if (minor>min_max).any():
                    minor[(minor>min_max)] = np.round(sd_min*np.random.randn(np.sum(minor>min_max))+np.mean([min_min, min_max]))
                if (minor<min_min).any():
                    minor[minor<min_min] = np.round(sd_min*np.random.randn(np.sum(minor<min_min))+np.mean([min_min, min_max]))

        major = np.multiply(minor, aspect_ratio)
        return major, minor
    @staticmethod
    def uniform_dims(min_min, min_max, min_ar, max_ar, n_sills):
        """
        Random uniform distribution of dims for aspect ratio and shape
        min_min = Minimum value for the thickness (m)
        min_max = Maximum value for the thickness (m)
        min_ar = Minimum aspect ratio (Width/Thickness)
        max_ar =  Maximum aspect ratio
        n_sills = Number of sills
        Returns dims in length units
        """
        aspect_ratio = np.random.uniform(min_ar, max_ar, n_sills)
        minor = np.round(np.random.randn(min_min, min_max, n_sills))
        major = np.multiply(minor, aspect_ratio)
        return major, minor
    @staticmethod
    def value_pusher(array, new_value, push_index, push_value):
        '''
        Insert values in specified index and pushed the values down by a specified amount. 
        This function works, but is a less efficient version of rules.value_pusher2D
        array: A 2D numpy array to be modified.
        new_value: The value to be inserted into the array.
        push_index: A tuple indicating the (x, y) index where the new value will be inserted.
        push_value: An integer indicating how many positions to shift the existing values downwards.
        '''
        x,y = push_index
        if push_value<=0:
            raise ValueError("push_value must be greater than 0")
        # Ensure the push operation does not exceed the array bounds
        if x + push_value >= len(array[0,:]):
            push_value = len(array[0,:])-x-1
        # Shift the values down
        array[x+push_value:x-1:-1, y] = array[x:x-push_value-1:-1, y]
        array[x:x+push_value,y] = new_value
        return array
    @staticmethod
    def prop_updater(lithology, lith_dict: dict, prop_dict: dict, property: str):
        '''
        This function updates the associated rock properties once everything has shifted.
        lithology: A 2D numpy array representing different rock types.
        lith_dict: A dictionary mapping integer keys to rock type names.
        prop_dict: A dictionary mapping rock type names to their properties.
        property: A string indicating the specific property to update.
        Returns a 2D array.
        '''
        prop = np.zeros_like(lithology)
        for rock in lith_dict.keys():
            prop[lithology==rock] = prop_dict[rock][property]
        return prop
    

    @staticmethod
    def value_pusher2D(array, new_value, row_index, push_amount):
        '''
        Modify a 2D numpy array by inserting a new value at specified row indices and shifting existing values downwards by a specified amount for each column.
        array: A 2D numpy array to be modified.
        new_value: The value to be inserted into the array.
        row_index: A list of row indices for each column where the new value will be inserted.
        push_amount: A list of integers indicating how many positions to shift the existing values downwards for each column.
        '''
        a,b = array.shape
        if len(row_index) != b or len(push_amount) != b:
            raise ValueError("row_index and push_values must have the same length as the number of columns")
        for j in range(b):
            if row_index[j] + push_amount[j] >= a:
                    raise ValueError(f"Push value for column {j} exceeds array bounds")
            array[row_index[j]+push_amount[j]:,j] = array[row_index[j]:a-push_amount[j], j]
            array[row_index[j]:row_index[j]+push_amount[j],j] = new_value
        return array


    @staticmethod
    def index_finder(array, string):
        '''
        Function to check and return the indices of an array that contains the specified string
        array: A numpy array of strings or elements that can be converted to strings. Prefered dtype is object.
        string: A string to search for within each element of the array.
        '''
        if not np.issubdtype(array.dtype, np.str_):
            array = array.astype(str)
        
        def contains_string(element):
            return string in element
        
        string_index = np.vectorize(contains_string)(array)
        return string_index


    def mult_sill(self, T_field,  majr, minr, height, x_space, dx, dy, rock = np.array([]), emplace_rock = 'basalt', T_mag = 1000, shape = 'elli', dike_empl = True, push = False):
        '''
        Emplace sills in a 2D temperature array and optionallu update the lithology array with the sills. 
        Optionally push the rocks downward, or overwrite the rocks in the field.
        T_field: A 2D numpy array representing the temperature field.
        majr: Major axis length of the sill.
        minr: Minor axis length of the sill.
        height: Y-coordinate for the center of the sill.
        x_space: X-coordinate for the center of the sill.
        dx: Spacing in the x-direction.
        dy: Spacing in the y-direction.
        rock: Optional 2D numpy array representing rock types.
        emplace_rock: Type of rock to emplace, default is 'basalt'.
        T_mag: Temperature magnitude of the sill.
        shape: Shape of the sill, either 'rect' or 'elli'.
        dike_empl: Boolean indicating if a dike tail should be emplaced.
        push: Boolean indicating if the sill should be pushed into the field.
        Returns the temperature field and an array indicating where the sill was emplaced and optionally the lithology array
        '''
        a,b = T_field.shape
        new_dike = np.zeros_like(T_field)
        majr = majr//dx
        minr = minr//dy
        if not push:
            if shape == 'rect':
                T_field[int(height-(minr//2)):int(height+(minr//2)), int(x_space-(majr//2)):int(x_space+(majr//2))] = T_mag
                new_dike[int(height-(minr//2)):int(height+(minr//2)), int(x_space-(majr//2)):int(x_space+(majr//2))] = 1
                if rock.size>0:
                    rock[int(height-(minr//2)):int(height+(minr//2)), int(x_space-(majr//2)):int(x_space+(majr//2))] = emplace_rock
            elif shape == 'elli':
                x = np.arange(0,b)
                y = np.arange(0, a)
                for m in range(0, a):
                    for n in range(0, b):
                        x_dist = ((x[n]-x[int(x_space)])**2)/(((majr)//2)**2)
                        y_dist = ((y[m]-y[int(height)])**2)/(((minr)//2)**2)
                        if (x_dist+y_dist)<=1:
                            T_field[m,n]=T_mag
                            new_dike[m,n] = 1
                            if rock.size>0:
                                rock[m,n] = emplace_rock
            if dike_empl:
                T_field[int(height):-1,int(x_space)] = T_mag
                new_dike[int(height):-1,int(x_space)] = 1
                if rock.size>0:
                    rock.loc[int(height):-1,int(x_space)] = 'basalt'
        elif push:
            if shape == 'rect':
                new_dike[int(height-(minr//2)):int(height+(minr//2)), int(x_space-(majr//2)):int(x_space+(majr//2))] = 1
                if dike_empl:
                    new_dike[int(height):-1,int(x_space)] = 1
            elif shape=='elli':
                x = np.arange(0,b)
                y = np.arange(0, a)
                for m in range(0, a):
                    for n in range(0, b):
                        x_dist = ((x[n]-x[int(x_space)])**2)/(((majr)//2)**2)
                        y_dist = ((y[m]-y[int(height)])**2)/(((minr)//2)**2)
                        if (x_dist+y_dist)<=1:
                            new_dike[m,n] = 1
                if dike_empl:
                    new_dike[int(height):-1,int(x_space)] = 1
            columns_push = np.sum(new_dike, axis = 0, dtype=int)
            row_push_start = np.zeros(b, dtype = int)
            for n in range(b):
                for m in range(a):
                    if new_dike[m,n]==1:
                        row_push_start[n] = m
                        break
            T_field = self.value_pusher2D(T_field, T_mag, row_push_start, columns_push)
            if rock.size>0:
                rock = self.value_pusher2D(rock, emplace_rock, row_push_start, columns_push)
        if rock.size>0:
            return T_field, rock, new_dike
        else:
            return T_field, new_dike


    @staticmethod
    def get_chemH(T_field, rho, CU, CTh, CK, T_sol, dike_net, a, b):
        """
        Untested function to calculate external heat sources generated through latent heat of crystallization and radiactive heat generation from Rybach and Cermack (1982) based on geochemistry
        T_field = Temp field, int
        rho = Density kg/m3
        CU, CTh = U, Th concentrations in ppm, array
        CK = K conc in wt %, array
        T_sol = Solidus temperature
        """
        J = 4e5 #J/kg latent heat of crystallization
        Cp = 850 #J/kgK specific heat capacity
        H = np.zeros_like(T_field)
        for i in range(0,a):
            for j in range(0, b):
                if T_field[i,j]>T_sol and dike_net[i,j]!=0:
                    H[i,j] = J/(rho*Cp)
        A = rho*1e-5*(9.52*CU + 2.56*CTh + 3.48*CK) #Formula from Rybach and Cermack 1982 - Radioactive heat generation in rocks
        H = H+A
        return H

    

    @staticmethod
    def rotate_nodes(coords_array, theta, center = (0,0)):
        '''
        Rotate the given coordinates by specified angle clockwise
        coords_array: array containing the row and column indices'''
        # Convert theta to radians and adjust for clockwise rotation
        theta_rad = np.radians(theta + 90)  # Add 90 for N orientation along z-axis

        # Extract coordinates
        x, y = coords_array

        # Translate coordinates so that the center is at the origin
        x_translated = x - center[0]
        y_translated = y - center[1]

        # Perform rotation
        x_rotated = x_translated * np.cos(theta_rad) + y_translated * np.sin(theta_rad)
        y_rotated = -x_translated * np.sin(theta_rad) + y_translated * np.cos(theta_rad)
        # Translate coordinates back to the original reference frame
        x1 = x_rotated + center[0]
        y1 = y_rotated + center[1]

        return x1, y1
            



    def sill_3Dcube(self, x, y, z, dx, dy, n_sills, x_coords, y_coords, z_coords, maj_dims, min_dims, empl_times, shape='elli', dike_tail=False, orientations = True, dike_width = None):
        '''
        Function to generate sills in 3D space to employ fluxes as a control for sill emplacement.
        Choose any 1 slice for a 2D cooling model, or multiple slices for multiple cooling models
        x, y, z = width, height and third dimension extension of the crustal slice (m)
        n_sills = Number of sills to be emplaced
        dx, dy = Node spacing
        x_coords = x coordinates for the center of the sills
        y_coords = y coordinates for the center of the sills
        z_coords = z coordinates for the center of the sills
        maj_dims, minor dims = dimensions of the 2D sills. Implicit assumption of circularity in the z-direction is present in the code (m)
        '''
        a = int(y // dy)
        b = int(x // dx)
        c = int(z // dx)
        sillcube = np.empty([c, a, b], dtype=object)
        sillcube[:, :, :] = ''
        z_len, y_len, x_len = np.mgrid[:c, :a, :b]

        if dike_width is None:
            dike_width = dx

        shift_nodes = np.max([int(np.round(dike_width/dx)), 1])
        maj_dims = maj_dims/dx
        min_dims = min_dims/dy

        if dike_tail:
            if orientations is None:
                orientations = np.random.randn(n_sills)*360

        if shape == 'elli':
            for l in trange(n_sills):
                mask = ((((z_len-z_coords[l])**2)/maj_dims[l]**2)+(((y_len-y_coords[l])**2)/min_dims[l]**2)
                +(((x_len-x_coords[l])**2)/maj_dims[l]**2))<=1
                sillcube[mask] += '_' + str(l) + 's' + str(empl_times[l])
                if dike_tail:
                    dike_zcoords = np.arange((z_coords[l]-maj_dims[l]//2), (z_coords[l]+maj_dims[l]//2))
                    dike_xcoords = np.arange((x_coords[l]-maj_dims[l]//2), (x_coords[l]+maj_dims[l]//2))
                    rot_zcoords = np.zeros_like(dike_zcoords)
                    rot_xcoords = np.zeros_like(dike_xcoords)
                    for i_len in range(len(dike_zcoords)):
                        rot_zcoords[i_len], rot_xcoords[i_len] = self.rotate_nodes([dike_zcoords[i_len], dike_xcoords[i_len]], orientations[l], [z_coords[l], x_coords[l]])
                    rot_zcoords = np.round(rot_zcoords).astype(int)
                    rot_xcoords = np.round(rot_xcoords).astype(int)
                    mask_z = (rot_zcoords >= 0) & (rot_zcoords < c)
                    rot_zcoords = rot_zcoords[mask_z]
                    rot_xcoords = rot_xcoords[mask_z]
                    #pdb.set_trace()
                    sillcube[rot_zcoords, int(y_coords[l]):-1, rot_xcoords] += '_' + str(l) + 's' + str(empl_times[l])
        
        elif shape == 'rect':
            for l in trange(n_sills):
                z_start = int(z_coords[l] - (maj_dims[l] // 2))
                z_end = int(z_coords[l] + (maj_dims[l] // 2))
                y_start = int(y_coords[l] - (maj_dims[l] // 2))
                y_end = int(y_coords[l] + (maj_dims[l] // 2))
                x_start = int(x_coords[l] - (min_dims[l] // 2))
                x_end = int(x_coords[l] + (min_dims[l] // 2))

                sillcube[z_start:z_end, y_start:y_end, x_start:x_end] += '_' + str(l) + 's' + str(empl_times[l])
                if dike_tail:
                    dike_zcoords = np.arange(z_start, z_end)
                    dike_xcoords = np.arange(x_start, x_end)
                    rot_zcoords = np.zeros_like(dike_zcoords)
                    rot_xcoords = np.zeros_like(dike_xcoords)
                    for i_len in range(len(dike_zcoords)):
                        rot_zcoords[i_len], rot_xcoords[i_len] = self.rotate_nodes([dike_zcoords[i_len], dike_xcoords[i_len]], orientations[l], [z_coords[l], x_coords[l]])
                    rot_zcoords = np.round(rot_zcoords)
                    rot_xcoords = np.round(rot_xcoords)
                    for shift_node_val in range(0,shift_nodes):
                        sillcube[int(rot_zcoords)-shift_node_val, int(y_coords[l]):-1, int(rot_xcoords)] += '_' + str(l) + 's' + str(empl_times[l])

        return sillcube
    def emplace_3Dsill(self, T_field, sillcube, n_rep, T_mag, z_index, curr_empl_time):
        '''
        Function to empalce a sill into the 2D slice T_field
        T_field = 2D temperature array
        sillcube = 3D sill array
        n_rep = the number of the sill being emplaced
        z_index = The 2D slice from the 3D sill array being considered
        '''
        string_finder = str(n_rep)+'s'+str(curr_empl_time)
        if len(sillcube.shape)!=3:
            raise IndexError('sillcube array must be three-dimensional')
        if T_field.size==0:
            raise IndexError("T_field cannot be empty")
        T_field[self.index_finder(sillcube[z_index], string_finder)] = T_mag
        return T_field

    @staticmethod
    def lithology_3Dsill(rock, sillcube, nrep, z_index, rock_type = 'basalt'):
        '''
        Function to keep track of lithology changes in the 2D array
        rock = 2D lithology array
        sillcube = 3D sill array
        nrep = number of the sill being emplaced
        z_index = The 2D slice from the 3D sill array being considered
        rock_type = the type of rock formed by the magma
        '''
        sillcube[sillcube==nrep] = rock_type
        rock_2d = sillcube[z_index,:,:]
        rock[rock_2d==rock_type]==rock_type
        return rock

    def sill3D_pushy_emplacement(self, props_array, props_dict, sillsquare, n_rep, mag_props_dict, curr_empl_time):
        '''
        Function used to modify a 3D properties array including temperature by emplacing a sill and shifting existing values downwards. 
        It identifies the emplacement location using a string identifier, calculates the necessary shifts, and updates the properties array accordingly.
        props_array: A 3D NumPy array containing property values.
        props_dict: A dictionary mapping property names to their indices in props_array.
        sillsquare: A 2D NumPy array containing string identifiers for emplacement.
        n_rep: An integer representing the sill number.
        mag_props_dict: A dictionary mapping property names to new values for emplacement.
        curr_empl_time: An integer representing the current emplacement time.
        '''
        string_finder = str(n_rep)+'s'
        T_field_index = props_dict['Temperature']
        T_field = props_array[T_field_index]
        a,b = T_field.shape
        if len(sillsquare.shape)!=2:
            raise IndexError(f'sillsquare array must be two-dimensional but is {len(sillsquare.shape)}')
        if T_field.size==0:
            raise ValueError("Temperature values in props_array cannot be empty")
        new_dike = np.zeros_like(T_field)
        new_dike[self.index_finder(sillsquare, string_finder)] = 1
        columns_pushed = np.sum(new_dike, axis =0, dtype=int)
        #print(f'Total columns pushed{columns_pushed[columns_pushed!=0]}')
        #columns_pushed  = columns_pushed.astype(int)
        row_push_start = np.zeros(b, dtype = int)
        for n in range(b):
            for m in range(a):
                if new_dike[m,n]==1:
                    #if row_push_start[n]==0:
                    row_push_start[n] = m
                    break
        #row_push_start = np.array(row_push_start, dtype=int)
        #row_push_start[row_push_start==np.nan] = 0
        #print(f'rows are {row_push_start}')
        if np.sum(self.index_finder(sillsquare, string_finder))==0:
            #print(f'Sill {n_rep} was NOT emplaced in this slice')
            pass
        else:
            print(f'Sill {n_rep} was emplaced in this slice')
            reverse_prop_dict = {value: key for key, value in props_dict.items()}
            for i in [listy for listy in props_dict.values()]:
                props_array[i] = self.value_pusher2D(props_array[i],mag_props_dict[reverse_prop_dict[i]],row_push_start, columns_pushed)
        return props_array, row_push_start, columns_pushed

class sill_controls:
    #Class that contains functions that make functions from the other classes easier to use for specific use cases.
    def __init__(self, x, y, dx, dy, 
                 T_liquidus = 1250, T_solidus = 800, include_external_heat = True,
                 k_const = True, kc_val = 7.884e7, cp = 1, cp_const = True,
                 calculate_closest_sill = False, calculate_all_sills_distances = False, calculate_at_all_times = False, 
                 rock_prop_dict = None, magma_prop_dict = None,lith_plot_dict=None,
                 sill_cube_dir='sillcubes/',k_func=None, cp_func = None):
        '''
        x - Horizontal extent of the crust m
        y - Vertical extent of the crust m
        T_liquidus - Liquidus of thr magma
        T_solidus - Solidus of the magma
        kc_val = Thermal conductivity W/mK
        cp = Initial specific heat capacity value for initialization 
        calculate_closest_sill = False (is done at the time of sill emplacement)
        calculate_all_sills_distances = False (is not used if calculate_closest_sill = False)
        calculate_at_all_times = False (calculate this at each timestep in the simulation)
        rock_prop_dict - Properties dictionary of ythe rocks in the crust
        magma_prop_dict - Properties dictionary of the magma
        Properties to include in the rock properties dictionary - Each dictionary entry should be the name of the rock, with further dictionary entries including the density, porosity, and total organic content (labelled as TOC)
        lith_plot_dict - Translation of rock valuies from strings to arbitrary integers for plotting purposes.
        '''
        ###
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.k_const = k_const
        a = int(y//dy)
        b = int(x//dx)
        if k_const:
            self.thermal_conductivity = np.ones((a,b))*kc_val
        else:
            self.thermal_conductivity = np.ones((a,b)) # default initialization to be overwritten later by chosen k_func

        
        self.calculate_closest_sill = calculate_closest_sill
        self.calculate_all_sill_distances = calculate_all_sills_distances
        self.calculate_at_all_times = calculate_at_all_times
        self.include_heat = include_external_heat
        
        
        self.cool = cool()
        self.rool = rules() 

        if k_func is None:
            self.k_func = self.cool.get_conductivity
        else:
            self.k_func = k_func
        if cp_func is None:
            self.cp_func = self.cool.get_specific_heat
        else:
            self.cp_func = cp_func
        self.cp_const = cp_const
        ###Setting up the properties dictionary to translate properties to indices for 3D array###
        self.prop_dict = {'Temperature':0,
                    'Lithology':1,
                    'Porosity':2,
                    'Density':3,
                    'TOC':4,
                    'Specific Heat':5}
        
        self.rev_prop_dict = {0:'Temperature',
                        1:'Lithology',
                        2:'Porosity',
                        3:'Density',
                        4:'TOC',
                        5:'Specific Heat'}

        #Lithology dictionary to translate rock types into numerical codes for numpy arrays
        if lith_plot_dict is None:
            self.lith_plot_dict = {'granite':0,
                            'shale':1,
                            'sandstone':2,
                            'peridotite':3,
                            'basalt':4}
        else:
            self.lith_plot_dict = lith_plot_dict

        if rock_prop_dict is None:
            self.rock_prop_dict = {
                "shale":{
                    'Porosity':0.1,
                    'Density':2500,
                    'TOC':2,
                    'Specific Heat': 820
                },
                "sandstone":{
                    'Porosity':0.2,
                    'Density':2600,
                    'TOC':2.5,
                    'Specific Heat': 800
                },
                "limestone":{
                    'Porosity':0.2,
                    'Density':2600,
                    'TOC':2.5,
                    'Specific Heat': 800
                },
                "granite":{
                    'Porosity':0.05,
                    'Density':2700,
                    'TOC':0,
                    'Specific Heat': 800
                },
                "basalt":{
                    'Porosity': 0.0,
                    'Density': 2850, #kg/m3
                    'TOC':0,
                    'Specific Heat': 850
                },
                "peridotite":{
                    'Porosity': 0.05,
                    'Density': 3100, #kg/m3
                    'TOC':0,
                    'Specific Heat': 1200
                }
            }
        else:
            self.rock_prop_dict = rock_prop_dict

        if magma_prop_dict is None:
            self.magma_prop_dict = {'Temperature': 1100,
                        'Lithology': 'basalt',
                        'Porosity': 0, #Porosity of the rock for calculation of carbon emissions
                        'Density': 2850, #kg/m3
                        'Specific Heat': 850, 
                        'Latent Heat': 4e5,
                        'TOC':0} #wt%
        else:
            self.magma_prop_dict = magma_prop_dict

        self.sill_cube_dir = sill_cube_dir
        ### Assigning some useful variables for usage later
        self.Temp_index = self.prop_dict['Temperature']
        self.rock_index = self.prop_dict['Lithology']
        self.poros_index = self.prop_dict['Porosity']
        self.dense_index = self.prop_dict['Density']
        self.TOC_index = self.prop_dict['TOC']
        self.sph_index = self.prop_dict['Specific Heat']

        #Set up melt properties#
        self.T_liquidus = T_liquidus
        self.T_solidus = T_solidus
        self.melt = self.magma_prop_dict['Lithology']



    def generate_sill_2D_slices(self,fluxy_list,iter_list,z_index_list):
        '''
        Function to generate 2D slices from the 3D cube
        fluxy_list - List of fluxes
        iter_list - Integer list of volumes of sills from the n_sills file
        z_index_list - List of z indices of the cubes to extract
        '''
        for flux in fluxy_list:
            load_dir = self.sill_cube_dir+str(format(flux, '.3e'))
            os.makedirs(load_dir+'/slice_volumes', exist_ok=True)
            for filename in os.listdir(os.path.join(load_dir, 'slice_volumes')):
                file_path = os.path.join(load_dir, 'slice_volumes', filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            n_sills_dataframe = pd.read_csv(load_dir+'/n_sills.csv')
            for iters in iter_list:
                volumes = float(n_sills_dataframe['volumes'][iters])

                sillcube = np.load(load_dir+'/sillcube'+str(volumes)+'.npy', allow_pickle=True)
                for z_indexs in z_index_list:
                    sillsquare = sillcube[z_indexs]
                    np.save(load_dir+'/slice_volumes/sillcube'+str(volumes)+'_'+str(z_indexs)+'.npy', sillsquare)
        print(f"Done generate_sill_2D_slices with {z_index_list} slices")
    
    def sill_controls_get_k(self, T_field, rock, density, dy, return_all = False):
        if not self.k_const:
            thermal_conductivity = self.cool.get_conductivity(T_field, rock, density, dy)
        else:
            thermal_conductivity = self.thermal_conductivity
        if not self.cp_const:
            specific_heat = self.cool.get_specific_heat(T_field, rock, density, dy)
        else:
            specific_heat = self.rool.prop_updater(rock, self.lith_plot_dict, self.rock_prop_dict, 'Specific Heat')
        k = np.array(thermal_conductivity/density/specific_heat, dtype = float)
        if return_all:
            return k, np.array(specific_heat, dtype = float), np.array(thermal_conductivity, dtype = float)
        else:
            return k

    def get_physical_properties(self, rock, rock_prop_dict):
        '''
        Function to return arrays of physical properties of density, porosity, and TOC based on rock type at a given node
        rock: A 2D numpy array representing the types of rocks at different nodes.
        rock_prop_dict: An optional dictionary mapping rock types to their properties.
        '''
        a,b = rock.shape
        porosity = np.zeros_like(rock)
        density = np.zeros_like(rock)
        TOC = np.zeros_like(rock)

        for i in range(a):
            for j in range(b):
                porosity[i,j] = rock_prop_dict[rock[i,j]]['Porosity']
                density[i,j] = rock_prop_dict[rock[i,j]]['Density']
                TOC[i,j] = rock_prop_dict[rock[i,j]]['TOC']
        return porosity, density, TOC
    
    @staticmethod
    def func_assigner(func, *args, **kwargs):
        '''
        Function that dynamically calls a given function with specified positional and keyword arguments, returning the result of the function call.
        func: The function to be called.
        *args: Positional arguments to be passed to the function.
        **kwargs: Keyword arguments to be passed to the function.
        '''
        result = func(*args,**kwargs)
        # If the result is a tuple or list, enumerate it
        #if isinstance(result, (tuple, list)):
        #    enumerated_result = list(enumerate(result))
        #    return enumerated_result
        # If the result is a single value, return it as is
        return result
    
    @staticmethod
    def check_closest_sill_temp(T_field, sills_array, curr_sill, dx, time, T_solidus, no_sill = '', calculate_all = False, save_file = None):
        '''
        Function that calculates the closest sill to a given sill based on temperature data and spatial arrangement. 
        It uses a KDTree to find the nearest sill that is hotter than a specified solidus temperature and optionally saves the results to a CSV file.
        T_field: 2D numpy array representing temperature values.
        sills_array: 2D numpy array representing sill identifiers.
        curr_sill: Integer representing the current sill identifier.
        T_solidus: Float representing the solidus temperature threshold (default is 800).
        no_sill: Value representing no sill in the array (default is an empty string).
        calculate_all: Boolean indicating whether to calculate for all sills (default is False).
        save_file: String representing the file path to save results (default is None).
        '''
        def get_width_and_thickness(bool_array):
            '''
            Function to get the width and thickness of a sill inside the check closest sill temp function
            '''
            max_row = np.max(np.where(bool_array==True)[1])
            min_row = np.min(np.where(bool_array==True)[1])
            width = max_row - min_row + 1
            max_col = np.max(np.where(bool_array==True)[0])
            min_col = np.min(np.where(bool_array==True)[0])
            thickness = max_col - min_col + 1
            center_row = np.mean([max_row, min_row])
            center_col = np.mean([max_col, min_col])
            center = str([center_row, center_col])
            return width, thickness, center
    
        is_sill = np.array((sills_array!=no_sill)) #Boolean of whether or not the node is a sill
        is_curr_sill = sills_array==curr_sill #Boolean to separate the nodes of the current sill
        boundary_finder = np.array(is_sill, dtype=int) #Array that counts the boundaries of the sills
        boundary_finder[1:-1, 1:-1] = (
        boundary_finder[:-2, 1:-1] +  # Above
        boundary_finder[2:, 1:-1] +   # Below
        boundary_finder[1:-1, :-2] +  # Left
        boundary_finder[1:-1, 2:])     # Right
        sills_number = sills_array.copy()
        sills_data = pd.DataFrame({'sills':curr_sill}, index = [0]) #Dataframe containing sill data for the current sill
        a,b = T_field.shape
        rows = np.arange(a)
        columns = np.arange(b)
        rows_grid, columns_grid = np.meshgrid(rows, columns, indexing='ij')
        points = np.column_stack((rows_grid.ravel(), columns_grid.ravel())) #Create a list of point coordinates to filter based on econditions
        points = points.reshape(-1,2)
        if calculate_all:
            #Code block to calculate the closest sills to all currently emplaced sills
            tot_sills = curr_sill #The last sill emplaced is the currently emplaced sill
            all_sills_data = pd.DataFrame() #Initialize the dataframe to save the computed information
            for curr_sill in range(tot_sills):
                condition = (T_field>T_solidus) & (sills_number!=curr_sill) #Select the nodes in the grid that are sills and are liquid
                query_condition = (sills_number == curr_sill) & (boundary_finder > 0) & (boundary_finder < 4) #Select the nodes that are at the boundary of the currently emplaced sill
                query_points = points[query_condition.ravel()] #Filter the points based on the condition
                #Initialize the variables to save in the dataframe
                saved_distance = 1e10 #Distance to the closest sill
                saved_index = -1 #Index of the current sill to which the sill is the closest
                saved_temperature = -1 #Temperature of the closest sill node
                saved_sill = -1 #The closest sill
                closest_curr_sill = 'N/A' #Index of the closest sill
                closest_sill_width_curr = 0 #Current width of the closest sill
                closest_sill_width = 0 #Original width of the closest sill
                closest_sill_thickness = 0  #Closest sill thickness
                closest_sill_thickness_curr = -1 #Current thickness of the closest sill
                closest_sill_center_curr = -1 #Current center of the closest sill
                closest_sill_center = -1 #Original center of the closest sill
                filtered_points = points[condition.ravel()] #Filter the points to query distances of current sill from
                tree = KDTree(filtered_points) #KDTree to query and find the least distance from a given point
                if len(query_points)>0:
                    for curr_point in query_points:
                        distance, index = tree.query(curr_point)
                        curr_sill_width, curr_sill_thickness, curr_sill_center = get_width_and_thickness(is_curr_sill)
                        if distance<saved_distance:
                            index1 = filtered_points[index]
                            saved_distance = distance
                            saved_index = str(index1)
                            saved_temperature = T_field[index1[0], index1[1]]
                            saved_sill = sills_array[index1[0], index1[1]]
                            closest_curr_sill = str(curr_point)
                            is_closest_sill_curr = (sills_array == saved_sill) & (T_field>T_solidus)
                            closest_sill_width_curr, closest_sill_thickness_curr, closest_sill_center_curr = get_width_and_thickness(is_closest_sill_curr)
                            is_closest_sill = (sills_array == saved_sill)
                            closest_sill_width, closest_sill_thickness, closest_sill_center = get_width_and_thickness(is_closest_sill)
                            
                sills_data['closest_sill'] = saved_sill
                sills_data['distance'] = saved_distance*dx
                sills_data['index of closest sill'] = saved_index
                sills_data['temperature'] = saved_temperature
                sills_data['index of current sill'] = closest_curr_sill
                sills_data['width of current sill'] = curr_sill_width*dx
                sills_data['thickness of current sill'] = curr_sill_thickness*dx
                sills_data['index of current sill center'] = curr_sill_center
                sills_data['width of closest sill'] = closest_sill_width_curr*dx
                sills_data['thickness of closest sill'] = closest_sill_thickness_curr*dx
                sills_data['current center of closest sill'] = closest_sill_center_curr
                sills_data['original width of closest sill'] = closest_sill_width*dx
                sills_data['original thickness of closest sill'] = closest_sill_thickness*dx
                sills_data['original center of closest sill'] = closest_sill_center
                sills_data['current time'] = time
                
            if all_sills_data.columns is None:
                all_sills_data.columns==sills_data.columns
            pd.concat([all_sills_data, sills_data], reset_index = True)
            if save_file is None:
                return all_sills_data
            else:
                all_sills_data.to_csv(save_file+'.csv')
                return all_sills_data
        else:
            #Code block to only calculate the closest sill to the specified sill
            condition = (T_field>T_solidus) & (sills_number!=curr_sill) #Select nodes that nor the current sill and are above liquidus
            query_condition = (sills_number == curr_sill) & (boundary_finder > 0) & (boundary_finder < 4) #Select nodes that are the boundary of the current sill
            query_points = points[query_condition.ravel()] #Use the points
            #Initialize variables to save the dataframe. See above for description
            saved_distance = 1e30
            saved_index = -1
            saved_temperature = -1
            saved_sill = -1
            closest_curr_sill = -1
            closest_sill_width_curr = 0
            closest_sill_width = 0
            closest_sill_thickness = 0
            closest_sill_width = -1
            closest_sill_thickness = -1
            closest_sill_width_curr = -1
            closest_sill_thickness_curr = -1
            closest_sill_center_curr = -1
            closest_sill_center = -1
            filtered_points = points[condition.ravel()]
            tree = KDTree(filtered_points) #KDTree to query and find shortes distance
            if len(query_points)>0:
                for curr_point in query_points:
                    distance, index = tree.query(curr_point)
                    curr_sill_width, curr_sill_thickness, curr_sill_center = get_width_and_thickness(is_curr_sill)
                    if distance<saved_distance:
                        index1 = filtered_points[index]
                        saved_distance = distance
                        saved_index = str(index1)
                        saved_temperature = T_field[index1[0], index1[1]]
                        saved_sill = sills_array[index1[0], index1[1]]
                        closest_curr_sill = str(curr_point)
                        is_closest_sill_curr = (sills_array == saved_sill) & (T_field>T_solidus)
                        closest_sill_width_curr, closest_sill_thickness_curr, closest_sill_center_curr = get_width_and_thickness(is_closest_sill_curr)
                        is_closest_sill = (sills_array == saved_sill)
                        closest_sill_width, closest_sill_thickness, closest_sill_center = get_width_and_thickness(is_closest_sill)
                        
                sills_data['closest_sill'] = saved_sill
                sills_data['distance'] = saved_distance*dx
                sills_data['index of closest sill'] = saved_index
                sills_data['temperature'] = saved_temperature
                sills_data['index of current sill'] = closest_curr_sill
                sills_data['width of current sill'] = curr_sill_width*dx
                sills_data['thickness of current sill'] = curr_sill_thickness*dx
                sills_data['index of current sill center'] = curr_sill_center
                sills_data['width of closest sill'] = closest_sill_width_curr*dx
                sills_data['thickness of closest sill'] = closest_sill_thickness_curr*dx
                sills_data['current center of closest sill'] = closest_sill_center_curr
                sills_data['original width of closest sill'] = closest_sill_width*dx
                sills_data['original thickness of closest sill'] = closest_sill_thickness*dx
                sills_data['original center of closest sill'] = closest_sill_center
                sills_data['current time'] = time
        return sills_data

    def build_sillcube(self, z, dt, thickness_range, aspect_ratio, depth_range, z_range, lat_range, phase_times, tot_volume, flux, n_sills, shape = 'elli', depth_function = None, lat_function = None, dims_function = None, emplace_dike = False, orientations = None):
        '''
        generates a 3D representation of sills in a geological model. It calculates emplacement heights, lateral spacings, and dimensions of sills based on specified distributions and parameters. 
        The method calculates the emplacement times for each sill based on specified flux rates, considering thermal maturation and cooling phases, and returns the constructed sill cube along with relevant parameters.
        Inputs - 
        z: The third dimension extension of the crustal slice.
        dt: Time step for the simulation.
        thickness_range: Range for sill thickness.
        aspect_ratio: Mean and standard deviation for aspect ratio.
        depth_range: Range for sill emplacement depth - 1D array with 2 elements for normal and uniform distributions. For an emperical distribution, a 2D array with the arrays of values and the corresponding cumulative distribution values are required 
        z_range: Range for z-coordinate - 1D array with 2 elements for normal and uniform distributions. For an emperical distribution, a 2D array with the arrays of values and the corresponding cumulative distribution values are required
        lat_range: Range for lateral coordinates - 1D array with 2 elements for normal and uniform distributions. For an emperical distribution, a 2D array with the arrays of values and the corresponding cumulative distribution values are required
        phase_times: Times for thermal maturation and cooling.
        tot_volume: Total volume of sills to be emplaced.
        flux: Emplacement flux.
        n_sills: Number of sills to be emplaced.
        shape: Shape of the sills, default is 'elli'.
        depth_function, lat_function, dims_function: Functions to determine distributions.
        Returns - 
        sillcube: A 3D numpy array representing the sill emplacement.
        n_sills: The number of sills emplaced.
        params: An array containing emplacement times, heights, lateral spacings, widths, and thicknesses.

        '''
        x = self.x
        y = self.y
        dx = self.dx
        dy = self.dy
        dims_empirical = False
        min_thickness = thickness_range[0] #m
        max_thickness = thickness_range[1] #m
        if len(thickness_range)>2:
            sd_min = thickness_range[2]
        mar = aspect_ratio[0]
        sar = aspect_ratio[1]

        min_emplacement = depth_range[0] #m
        max_emplacement = depth_range[1] #m
        if len(depth_range)>2:
            sd_empl = depth_range[2]

        x_min = lat_range[0]
        x_max = lat_range[1]
        if len(lat_range)>2:
            sd_x = lat_range[2]

        z_min = z_range[0]
        z_max = z_range[1]
        if len(z_range)>2:
            sd_z = z_range[2]

        if depth_function==None or depth_function=='normal':
            depth_function = self.rool.randn_heights
            depth_input_params = (n_sills, max_emplacement, min_emplacement, sd_empl, dy)
        elif depth_function == 'uniform':
            depth_function = self.rool.uniform_heights
            depth_input_params = (n_sills, min_emplacement, max_emplacement, dy)
        elif depth_function=='empirical':
            depth_function = self.rool.empirical_CDF
            depth_input_params = (n_sills, depth_range[0], depth_range[1])
        else:
            raise ValueError('depth_function should be either normal or uniform or empirical')

        empl_heights = self.func_assigner(depth_function, *depth_input_params)
        
        if lat_function==None or lat_function=='normal':
            lat_function = self.rool.x_spacings
            lat_input_params = (n_sills, x_min, x_max, sd_x, dx)
            z_params = (n_sills, z_min, z_max, sd_z, dx)
        elif lat_function=='uniform':
            lat_function = self.rool.uniform_x
            lat_input_params = (n_sills, x_min, x_max, dx)
            z_params = (n_sills, z_min, z_max, dx)
        elif lat_function == 'empirical':
            lat_function = self.rool.empirical_CDF
            lat_input_params = (n_sills, lat_range[0], lat_range[1])
            z_params = (n_sills, z_range[0], z_range[1])
        else:
            raise ValueError('lat_function should be either normal or uniform or emipirical')
        
        if dims_function==None or dims_function== 'normal':
            dims_function = self.rool.randn_dims
            dims_input_params = (min_thickness, max_thickness, sd_min, mar, sar, n_sills)
        elif dims_function == 'uniform':
            dims_function = self.rool.uniform_dims
            dims_input_params = (min_thickness, max_thickness, aspect_ratio[0], aspect_ratio[1], n_sills)
        elif dims_function == 'scaled':
            dims_function = self.rool.get_scaled_dims
            dims_input_params = (min_thickness, max_thickness, mar, sar, empl_heights, n_sills)
        elif dims_function == 'empirical':
            dims_empirical = True
            dims_function = self.rool.empirical_CDF
            dims_input_params = (n_sills, aspect_ratio[0], aspect_ratio[1])
        else:
            raise ValueError('dims_function should be either normal or uniform or scaled or empirical')
        

        '''
        #Checking to see if there are any assignments outside the distribution#
        n = 0
        while ((empl_heights>max_emplacement/dy).any() or (empl_heights<min_emplacement/dy).any()):
            print('Heights')
            print((len(empl_heights[empl_heights>max_emplacement/dy])+len(empl_heights[empl_heights<(min_emplacement/dy)]))*100/n_sills, '%')
            n = int(n+1)
            print('Cycle', n ,'reassigning')
            if (empl_heights>max_emplacement/dy).any():
                empl_heights[empl_heights>max_emplacement/dy] = depth_function(np.sum(empl_heights>max_emplacement/dy), max_emplacement, min_emplacement, sd_empl, dy)
            if (empl_heights<min_emplacement/dy).any():
                empl_heights[empl_heights<min_emplacement/dy] = depth_function(np.sum(empl_heights<(min_emplacement/dy)), max_emplacement, min_emplacement, sd_empl, dy)
        '''
        sns.kdeplot(empl_heights*dy/1000, label = 'Depth distribution', color = 'red', linewidth = 1.75)
        plt.ylabel('Depth distribution (km)')
        plt.savefig('plots/Depth'+str(format(flux, '.3e'))+'_'+str(format(tot_volume, '.3e'))+'.png', format = 'png', bbox_inches = 'tight')
        plt.close()
        x_space = self.func_assigner(lat_function, *lat_input_params)
        '''
        n = 0
        while ((x_space>x_max/dx).any() or (x_space<(x_min/dx)).any()):
            print((len(x_space[x_space>x_max/dx])+len(x_space[x_space<(x_min/dx)]))*100/n_sills, '%')
            print('Horizontal space')
            n = int(n+1)
            print('Cycle', n ,'reassigning')
            if (x_space>x_max/dx).any():
                x_space[x_space>x_max/dx] = lat_function(np.sum(x_space>x_max/dx), x_min, x_max, sd_x, dx)
            if (x_space<(x//(3*dx))).any():
                x_space[x_space<(x_min/dx)] = lat_function(np.sum(x_space<(x_min/dx)), x_min, x_max, sd_x, dx)
        '''
        width, thickness = self.func_assigner(dims_function, *dims_input_params) if not dims_empirical else (None, None)
        if (width==None).all():
            aspect_ratios = self.func_assigner(dims_function, dims_input_params)
            thickness = self.func_assigner(dims_function, *(thickness_range[0], thickness_range[1], n_sills))
            width = thickness*aspect_ratios

        sns.kdeplot(width, label = 'Width distribution', linewidth = 1.75)
        sns.kdeplot(thickness, label = 'Thickness Distribution', linewidth = 1.75, color = 'red')
        plt.xlim(left = 0)
        plt.xlabel('Length units (m)')
        plt.legend()
        plt.savefig('plots/WidthThickness'+str(format(tot_volume, '.3e'))+'_'+str(format(flux, '.3e'))+'.png', format = 'png', bbox_inches = 'tight')
        plt.close()
        
        thermal_maturation_time = phase_times[0]
        total_empl_time = tot_volume/flux
        cooling_time = phase_times[1]
        model_time = total_empl_time+thermal_maturation_time+cooling_time
        time_steps = np.arange(0, model_time,dt)
        saving_time_step_index = np.min(np.where(time_steps>=thermal_maturation_time)[0])
        print(saving_time_step_index, time_steps[saving_time_step_index])
        empl_times = []
        plot_time = []
        cum_volume = []

        if shape == 'elli':
            volume = (4*np.pi/3)*width*width*thickness
        elif shape=='rect':
            volume = width*width*thickness

        unemplaced_volume = 0
        #print(f'{np.sum(volume):.5e}, {float(tot_volume):.5e}, {np.sum(volume)<tot_volume}')
        n = 0
        for l in range(len(time_steps)):
            if time_steps[l]<thermal_maturation_time+dt:
                continue
            else:
                if n>0:
                    mean_flux = np.sum(volume[0:n])/(time_steps[l]-thermal_maturation_time)
                else:
                    mean_flux = 0
                unemplaced_volume += flux*dt
                if unemplaced_volume>=volume[n] and mean_flux<0.95*flux:
                    empl_times.append(time_steps[l])
                    plot_time.append(time_steps[l])
                    cum_volume.append(volume[n])
                    unemplaced_volume -= volume[n]
                    print(f'Emplaced sill {n} at time {time_steps[l]}')
                    print(f'Remaining volume to emplace: {tot_volume-np.sum(volume[:n]):.4e}')
                    mean_flux = np.sum(volume[0:n])/(time_steps[l]-thermal_maturation_time)
                    n+=1
                    
                    while unemplaced_volume>volume[n] and mean_flux<(0.95*flux if np.sum(volume[0:n+1])<=tot_volume else 1.05*flux) and np.sum(volume[0:n])<=tot_volume:
                        empl_times.append(time_steps[l])
                        unemplaced_volume -= volume[n]
                        cum_volume[-1]+=volume[n]
                        print(f'Emplaced sill {n} at time {time_steps[l]}')
                        print(f'Remaining volume to emplace: {tot_volume-np.sum(volume[:n]):.4e}')
                        mean_flux = np.sum(volume[0:n])/(time_steps[l]-thermal_maturation_time)
                        n+=1

                if (n>0) and (np.sum(volume[0:n-1])>tot_volume):
                    print('Total sills emplaced:', n)
                    n_sills = int(n)
                    empl_heights = empl_heights[0:n_sills]
                    x_space = x_space[0:n_sills]
                    width = width[0:n_sills]
                    thickness = thickness[0:n_sills]
                    break
        cum_volume = np.cumsum(cum_volume)
        plt.plot(plot_time, cum_volume/int(1e9), color = 'red', linewidth = 1.75, label = 'Cumulative volume emplaced')
        lol = (np.array(empl_times)-thermal_maturation_time)*flux
        plt.plot(empl_times, lol/int(1e9), color = 'black', linewidth = 1.75, label = 'Mean cumulative volume')
        plt.ylabel(r'Volume emplacemed ($km^3$)')
        plt.xlabel(r'Time (Ma)')
        plt.legend()
        plt.savefig('plots/VolumeTime'+str(format(tot_volume, '.3e'))+'_'+str(format(flux, '.3e'))+'.png', format = 'png', bbox_inches = 'tight')
        plt.close()

        z_coords = self.func_assigner(lat_function, *z_params)
        sillcube = self.rool.sill_3Dcube(x,y,z,dx,dy,n_sills, x_space, empl_heights, z_coords, width, thickness, empl_times,shape, dike_tail=emplace_dike, orientations=orientations)
        params = np.array([empl_times, empl_heights, x_space, width, thickness])
        return sillcube, n_sills, params
    
    def get_silli_initial_thermogenic_state(self, props_array, dt, method, time = np.nan, lith_plot_dict = None, rock_prop_dict = None):
        '''
        Function to get the background CO2 release for the silli model over the thermal maturation time before the emplacement fo sills begins
        Inputs - 
        props_array: A 3D numpy array containing properties like temperature, lithology, porosity, density, and TOC.
        dt: A scalar representing the time step for the simulation.
        method: A string indicating the method used for solving the diffusion equation.
        time: Optional scalar for the total simulation time.
        lith_plot_dict: Optional dictionary mapping lithology names to indices.
        rock_prop_dict: Optional dictionary mapping rock types to their properties.

        Returns - 
        current_time: The current simulation time.
        tot_RCO2: A list of total CO2 emissions over time.
        props_array: The updated properties array.
        RCO2_silli: CO2 emissions from SILLi.
        Rom_silli: Rate of organic matter conversion.
        percRo_silli: Vitrinite reflectance percentage.
        curr_TOC_silli: Current TOC values.
        W_silli: Weight fractions of remaining reactants.
        '''
        dx = self.dx
        dy = self.dy

        if not lith_plot_dict or all(value is None for value in lith_plot_dict.values()):
            lith_plot_dict = self.lith_plot_dict
        
        if not rock_prop_dict or all(value is None for value in rock_prop_dict.values()):
            rock_prop_dict = self.rock_prop_dict


        density = props_array[self.dense_index]
        porosity = props_array[self.poros_index]
        rock = props_array[self.rock_index]
        T_field = props_array[self.Temp_index]

        k, specific_heat, _ = self.sill_controls_get_k(T_field, rock, density,return_all=True)

        breakdown_CO2 = np.zeros_like(T_field)
        specific_heat = np.zeros_like(T_field)
        specific_heat = np.vectorize(
            lambda rt: self.rock_prop_dict[rt]['Specific Heat'], 
            otypes=[float]  # Ensure output is float
        )(rock)
        if self.include_heat:
            H_lat = self.cool.get_latH(T_field, rock, self.melt, self.magma_prop_dict['Density'], self.T_liquidus, self.T_solidus)
            H_rad = self.cool.get_radH(T_field, density,dx)/density/specific_heat/H_lat
            H = np.array([H_rad, H_lat])
            #H = H/self.magma_prop_dict['Density']/magma_prop_dict['Specific Heat']
        else:
            H_rad = np.zeros_like(T_field)
            H_lat = np.ones_like(T_field)
            H = np.array([H_rad, H_lat])

        t = 0
        a, b = props_array[0].shape
        TOC = self.rool.prop_updater(props_array[self.rock_index], lith_plot_dict, rock_prop_dict, 'TOC')
        if np.isnan(time):
            T_field = self.cool.diff_solve(k, a, b, dx, dy, dt, T_field, np.nan, method, H)
            props_array[self.Temp_index] = T_field
            curr_TOC_silli = props_array[self.TOC_index]
            RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli = emit.SILLi_emissions(T_field, density, rock, porosity, TOC, dt)
            if (rock=='limestone').any():
                breakdown_CO2 = emit.get_init_CO2_percentages(T_field, rock, density, dy)
            props_array[self.TOC_index] = curr_TOC_silli
            diff = 1e6
            iter = 0
            thresh = 1e-3
            t+=dt
            tot_RCO2 = []
            dV = dx*dx*dy
            tot_RCO2.append(np.sum(RCO2_silli)+np.sum(breakdown_CO2))
            iter_thresh = int(1e7//dt)
            current_time = 0
            with tqdm(total = iter_thresh, desc = 'Processing') as pbar:
                while iter<iter_thresh and diff>thresh:
                    T_field = self.cool.diff_solve(k, a, b, dx, dy, dt, T_field, np.nan, method, H)
                    props_array[self.Temp_index] = T_field
                    curr_TOC_silli = props_array[self.TOC_index]
                    RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli = emit.SILLi_emissions(T_field, density, rock, porosity, curr_TOC_silli, dt, TOC, W_silli)
                    if (rock=='limestone').any():    
                        breakdown_CO2, _ = emit.get_breakdown_CO2(T_field, rock, density, breakdown_CO2, dy, dt)
                    props_array[self.TOC_index] = curr_TOC_silli
                    RCO2_silli = RCO2_silli*density*dV/100
                    breakdown_CO2 = breakdown_CO2*density*dV/100
                    tot_RCO2.append(np.sum(RCO2_silli)+np.sum(breakdown_CO2))
                    if iter>0:
                        diff = tot_RCO2[-2]-tot_RCO2[-1]
                    iter+=1
                    current_time += dt
                    pbar.update(1)
                    pbar.set_postfix({"Change": diff})
        else:
            t_steps = np.arange(0, time, dt)
            tot_RCO2 = []
            dV = dx*dx*dy
            for l in trange(0, len(t_steps)):
                T_field = self.cool.diff_solve(k, a, b, dx, dy, dt, T_field, np.nan, method, H)
                props_array[self.Temp_index] = T_field
                curr_TOC_silli = props_array[self.TOC_index]
                if l==0:
                    #TOC = props_array[self.TOC_index]
                    RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli = emit.SILLi_emissions(T_field, density, rock, porosity, TOC, dt)
                    if (rock=='limestone').any():
                        breakdown_CO2 = emit.get_init_CO2_percentages(T_field, rock, density, dy)
                else:
                    RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli = emit.SILLi_emissions(T_field, density, rock, porosity, curr_TOC_silli, dt, TOC, W_silli)
                    if (rock=='limestone').any():    
                        breakdown_CO2, _ = emit.get_breakdown_CO2(T_field, rock, density, breakdown_CO2, dy, dt)
                props_array[self.TOC_index] = curr_TOC_silli
                RCO2_silli = RCO2_silli*density*dV/100
                breakdown_CO2 = breakdown_CO2*density*dV/100
                tot_RCO2.append(np.sum(RCO2_silli)+np.sum(breakdown_CO2))
                current_time = t_steps[l]
        props_array[self.Temp_index] = T_field
        props_array[self.dense_index] = density
        props_array[self.rock_index] = rock
        props_array[self.poros_index] = porosity
        props_array[self.TOC_index] = curr_TOC_silli
        return current_time, tot_RCO2, props_array, RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli

    def get_sillburp_initial_thermogenic_state(self, props_array, dt, method, sillburp_weights = None, time = np.nan, lith_plot_dict = None, rock_prop_dict = None):
        '''
        Function to get the background CO2 release for the sillburp model over the thermal maturation time before the emplacement fo sills begins
        Inputs - 
        props_array: A 3D numpy array containing properties like temperature, lithology, porosity, density, and TOC.
        dt: A scalar representing the time step for the simulation.
        method: A string indicating the method used for solving the diffusion equation.
        sillburp_weights: Optional weights for the sillburp model.
        time: Optional scalar for the total simulation time.
        lith_plot_dict: Optional dictionary mapping lithology names to indices.
        rock_prop_dict: Optional dictionary mapping rock types to their properties.

        Returns - 
        current_time: The current simulation time.
        tot_RCO2: A list of total CO2 emissions over time.
        props_array: The updated properties array.
        RCO2: CO2 emissions from the sillburp model.
        Rom: Rate of organic matter conversion.
        progress_of_reactions: Progress of reactions in the sillburp model.
        oil_production_rate: Rate of oil production.
        curr_TOC: Current TOC values.
        rate_of_reactions: Rate of reactions.
        sillburp_weights: Weights used in the sillburp model.
        '''
        dx = self.dx
        dy = self.dy
        try:
            if not lith_plot_dict or all(value is None for value in lith_plot_dict.values()):
                lith_plot_dict = self.lith_plot_dict
            if not rock_prop_dict or all(value is None for value in rock_prop_dict.values()):
                rock_prop_dict = self.rock_prop_dict
        except AttributeError:
            pass

        density = np.array(props_array[self.dense_index], dtype = float)
        porosity = np.array(props_array[self.poros_index], dtype = float)
        rock = props_array[self.rock_index]
        T_field = np.array(props_array[self.Temp_index], dtype = float)

        k, specific_heat, _ = self.sill_controls_get_k(T_field, rock, density,return_all=True)

        breakdown_CO2 = np.zeros_like(T_field)
        dV = dx*dx*dy
        t = 0
        a, b = props_array[0].shape
        specific_heat = np.vectorize(
            lambda rt: self.rock_prop_dict[rt]['Specific Heat'], 
            otypes=[float]  # Ensure output is float
        )(rock)
        if self.include_heat:
            H_lat = self.cool.get_latH(T_field, rock, self.melt, self.magma_prop_dict['Density'], self.T_liquidus, self.T_solidus)
            H_rad = self.cool.get_radH(T_field, density,dx)/density/specific_heat/H_lat
            H = np.array([H_rad, H_lat])
            #H = H/self.magma_prop_dict['Density']/magma_prop_dict['Specific Heat']
        else:
            H_rad = np.zeros_like(T_field)
            H_lat = np.ones_like(T_field)
            H = np.array([H_rad, H_lat])

        TOC = self.rool.prop_updater(props_array[self.rock_index], lith_plot_dict, rock_prop_dict, 'TOC')
        reaction_energies = emit.get_sillburp_reaction_energies()
        if np.isnan(time):
            T_field = self.cool.diff_solve(k, a, b, dx, dy, dt, T_field, np.nan, method, H)
            props_array[self.Temp_index] = T_field
            curr_TOC = props_array[self.TOC_index]
            RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions = emit.sillburp(T_field, TOC, density, rock, porosity, dt, reaction_energies, weights=sillburp_weights)
            if (rock=='limestone').any():
                breakdown_CO2 = emit.get_init_CO2_percentages(T_field, rock, density, dy)
            props_array[self.TOC_index] = curr_TOC
            diff = 1e6
            iter = 0
            thresh = 1e-3
            t+=dt
            tot_RCO2 = []
            iter_thresh = int(1e7//dt)
            RCO2 = RCO2*density*dV/100
            breakdown_CO2 = breakdown_CO2*density*dV/100 if (rock=='limestone').any() else np.zeros_like(T_field)
            tot_RCO2.append(np.sum(RCO2)+np.sum(breakdown_CO2))
            current_time = 0
            with tqdm(total = iter_thresh, desc = 'Processing') as pbar:
                while iter<iter_thresh and diff>thresh:
                    curr_TOC = np.array(props_array[self.TOC_index], dtype = float)
                    RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions = emit.sillburp(T_field, curr_TOC, density, rock, porosity, dt, reaction_energies, TOC, oil_production_rate, progress_of_reactions, rate_of_reactions, weights=sillburp_weights)
                    if (rock=='limestone').any():    
                        breakdown_CO2, _ = emit.get_breakdown_CO2(T_field, rock, density, breakdown_CO2, dy, dt)
                    props_array[self.TOC_index] = curr_TOC
                    RCO2 = RCO2*density*dV/100
                    breakdown_CO2 = breakdown_CO2*density*dV/100
                    tot_RCO2.append(np.sum(RCO2)+np.sum(breakdown_CO2))
                    if iter>0:
                        diff = tot_RCO2[-2]-tot_RCO2[-1]
                    iter+=1
                    current_time += dt

                    pbar.update(1)
                    pbar.set_postfix({"Change": diff})
        else:
            print('Entered  else loop')
            t_steps = np.arange(0, time, dt)
            tot_RCO2 = []
            for l in trange(0, len(t_steps)):
                T_field = self.cool.diff_solve(k, a, b, dx, dy, dt, T_field, np.nan, method, H)
                props_array[self.Temp_index] = T_field
                curr_TOC_silli = np.array(props_array[self.TOC_index], dtype = float)
                TOC = np.array(self.rool.prop_updater(rock, lith_plot_dict, rock_prop_dict, 'TOC'), dtype = float)
                if l==0:
                    print('First time step')
                    RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions = emit.sillburp(T_field, TOC, density, rock, porosity, dt, reaction_energies, weights=sillburp_weights)
                    print('Carbon burped')
                    if (rock=='limestone').any():
                        breakdown_CO2 = emit.get_init_CO2_percentages(T_field, rock, density, dy)
                else:
                    print('Other steps')
                    RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions = emit.sillburp(T_field, curr_TOC, density, rock, porosity, dt, reaction_energies, TOC, oil_production_rate, progress_of_reactions, rate_of_reactions, weights=sillburp_weights)
                    print('Carbon burped')
                    if (rock=='limestone').any():    
                        breakdown_CO2, _ = emit.get_breakdown_CO2(T_field, rock, density, breakdown_CO2, dy, dt)
                props_array[self.TOC_index] = curr_TOC_silli
                RCO2 = RCO2*density*dV/100
                breakdown_CO2 = breakdown_CO2*density*dV/100
                tot_RCO2.append(np.sum(RCO2)+np.sum(breakdown_CO2))
                current_time = t_steps[l]
        props_array[self.Temp_index] = T_field
        props_array[self.dense_index] = density
        props_array[self.rock_index] = rock
        props_array[self.poros_index] = porosity
        return current_time, tot_RCO2, props_array, RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions, sillburp_weights

    def emplace_sills(self,props_array, n_sills, cool_method, time_steps, current_time, sillsquare, carbon_model_params, empl_times, volume_params, z_index, saving_factor = None, save_dir = None, model=None,dt = None, q= np.nan, rock_prop_dict = None, lith_plot_dict = None, prop_dict = None, magma_prop_dict = None):
        '''
        Function to simulate cooling and associated thermogenic carbon release (optional) and save the properties at the specified intervals optionally
        Inputs - 
        props_array: 3D numpy array of geological properties.
        n_sills: Number of sills to emplace.
        cool_method: Method for cooling calculation.
        time_steps: Array of time steps for simulation.
        current_time: Current simulation time.
        sillsquare: 2D array representing sill geometry.
        carbon_model_params: Parameters for carbon emission models.
        emplacement_params: Parameters for sill emplacement.
        volume_params: Volume and flux parameters.
        z_index: Index for vertical dimension.
        saving_factor: How often should the properties be saved i.e. 1 in how many time steps 
        save_dir: Directory to save the results in. 
        model: Which carbon model to use silli or sillburp?
        dt - Time step, 
        q - Bottom heat flux, 
        H - External heat
        rock_prop_dict, lith_plot_dict, prop_dict, magma_prop_dict: Optional dictionary parameters.
        '''
        dx = self.dx
        dy = self.dy
        saving_time_step_index = np.where(time_steps==current_time)[0]
        if len(saving_time_step_index)>0:
            saving_time_step_index = saving_time_step_index[0]
        print(f'Current time: {current_time}')
        print(f'saving_time_step_index: {saving_time_step_index}')
        #shape_index = [len(time_steps[saving_time_step_index:])]+list(props_array.shape)
        #props_total_array = np.empty(shape_index, dtype = object)
        if lith_plot_dict==None:
            lith_plot_dict = self.lith_plot_dict
        if rock_prop_dict==None:
            rock_prop_dict = self.rock_prop_dict
        if prop_dict==None:
            prop_dict = self.prop_dict
        if magma_prop_dict==None:
            magma_prop_dict = self.magma_prop_dict
        if dt is None:
            dts = np.array([time_steps[i]-time_steps[i-1] for i in range(1, len(time_steps))])
            dts = np.append(dts[0],dts)
        else:
            dts = np.repeat(dt,len(time_steps))
        rock = np.array(props_array[self.rock_index])
        density = np.array(props_array[self.dense_index])
        porosity = np.array(props_array[self.poros_index])
        T_field = np.array(props_array[self.Temp_index], dtype = float)
        specific_heat = np.array(props_array[self.sph_index])
        TOC1 = self.rool.prop_updater(rock, lith_plot_dict, rock_prop_dict, 'TOC')
        a,b = T_field.shape
        k = self.sill_controls_get_k(T_field, rock, density, dy)
        breakdown_CO2 = np.zeros_like(T_field)
        if model=='silli':
            tot_RCO2, props_array_unused, RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli = carbon_model_params
        elif model =='sillburp':
           tot_RCO2, props_array_unused, RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions, sillburp_weights = carbon_model_params
           reaction_energies = emit.get_sillburp_reaction_energies()
        elif model==None:
            pass
        else:
            raise ValueError(f'model is {model}, but must be either silli or sillburp')
        empl_times = np.array(empl_times, dtype=float)
        curr_sill = 0
        dV = dx*dx*dy

        flux = volume_params[0]
        tot_volume = volume_params[1]
        props_array_vtk = pv.ImageData(dimensions = [props_array.shape[2], props_array.shape[1],1])
        
        if save_dir is None:
            save_dir = 'sillcubes/'+str(format(flux, '.3e'))+'/'+str(format(tot_volume, '.3e'))+'/'+str(z_index)
        os.makedirs(save_dir, exist_ok = True)
        sillnet = np.zeros((a,b), dtype = object)
        sillnet[:] = ''
        if self.calculate_closest_sill and not self.calculate_at_all_times:
            all_sills_data = pd.DataFrame()
        sills_emplaced = np.zeros((a,b))
        tot_melt10 = []
        tot_melt50 = []
        tot_solidus = []
        area_sills = []
        TOC1 = self.rool.prop_updater(rock, lith_plot_dict, rock_prop_dict, 'TOC')
        for l in trange(saving_time_step_index, len(time_steps)):
            #curr_time = time_steps[l]
            dt = dts[l]          
            T_field = np.array(props_array[self.Temp_index], dtype = float)
            specific_heat = np.array(props_array[self.sph_index])
            density = np.array(props_array[self.dense_index])
            rock = np.array(props_array[self.rock_index])
            porosity = np.array(props_array[self.poros_index])
            if self.include_heat:
                H_lat = self.cool.get_latH(T_field, rock, self.melt, self.magma_prop_dict['Density'], self.T_liquidus, self.T_solidus)
                H_rad = self.cool.get_radH(T_field, density,dx)/density/specific_heat/H_lat
                H = np.array([H_rad, H_lat])
            else:
                H_rad = np.zeros_like(T_field)
                H_lat = np.ones_like(T_field)
                H = np.array([H_rad, H_lat])
            k, specific_heat, _ = self.sill_controls_get_k(T_field, rock, density, dy, return_all=True)
            T_field = self.cool.diff_solve(k, a, b, dx, dy, dt, T_field, q, cool_method, H)
            if np.max(T_field)>1.05*1100:
                warnings.warn(f'Too much latent heat: {np.min(H_lat)}. Maximum temperature is now {np.max(T_field)}', RuntimeWarning)
                #pdb.set_trace()
            if self.calculate_closest_sill and self.calculate_at_all_times:
                save_file = save_dir+'/sill_distances'+str(time_steps[l])
                sills_data = self.check_closest_sill_temp(props_array[self.Temp_index], sillnet, curr_sill,dx, time_steps[l], T_solidus=self.T_solidus, calculate_all=self.calculate_all_sill_distances, save_file=save_file)
            props_array[self.Temp_index] = T_field
            curr_TOC_silli = props_array[self.TOC_index]
            rock = props_array[self.rock_index]
            

            if model=='silli':
                if l!=saving_time_step_index:
                    RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli = emit.SILLi_emissions(T_field, density, rock, porosity, curr_TOC_silli, dt, TOC1, W_silli)
                    RCO2_model = RCO2_silli*dV
                    
            elif model=='sillburp':
                if l!=saving_time_step_index:
                    RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions = emit.sillburp(T_field, curr_TOC, density, rock, porosity, dt, reaction_energies, TOC1, oil_production_rate, progress_of_reactions, rate_of_reactions, weights=sillburp_weights)
                    RCO2_model = RCO2*dV
            if (rock=='limestone').any():    
                breakdown_CO2, _ = emit.get_breakdown_CO2(T_field, rock, density, breakdown_CO2, dy, dt)
            if l!=saving_time_step_index:
                RCO2_model += breakdown_CO2*dV
                tot_RCO2.append(np.sum(RCO2_model))
            props_array[self.TOC_index] = curr_TOC_silli
            if isinstance(n_sills, str):
                n_sills = n_sills.strip('[]')
            n_sills = int(n_sills)
            while time_steps[l]==empl_times[curr_sill] and curr_sill<int(n_sills):
                #print(f'Now emplacing sill {curr_sill}')
                props_array, row_start, col_pushed = self.rool.sill3D_pushy_emplacement(props_array, prop_dict, sillsquare, curr_sill, magma_prop_dict, empl_times[curr_sill])
                curr_TOC_silli = props_array[self.TOC_index]
                sillnet = self.rool.value_pusher2D(sillnet, curr_sill, row_start, col_pushed)
                if self.calculate_closest_sill and not self.calculate_at_all_times:
                    if len(col_pushed[col_pushed!=0]>0):
                        print(f'Checking closest sills for {curr_sill}')
                        sills_data = self.check_closest_sill_temp(props_array[self.Temp_index], sillnet, curr_sill, dx, time_steps[l], T_solidus=self.T_solidus, calculate_all=self.calculate_all_sill_distances)
                        if all_sills_data.columns.empty:
                            all_sills_data = pd.DataFrame(columns = sills_data.columns)
                        all_sills_data = pd.concat([all_sills_data, sills_data], ignore_index = True)
                if model=='silli':
                    if (col_pushed!=0).any():
                        sills_emplaced = self.rool.value_pusher2D(sills_emplaced, curr_sill, row_start, col_pushed)
                        RCO2_silli = self.rool.value_pusher2D(RCO2_silli,0, row_start, col_pushed)
                        Rom_silli = self.rool.value_pusher2D(Rom_silli,0, row_start, col_pushed)
                        percRo_silli =self.rool.value_pusher2D(percRo_silli, 0, row_start, col_pushed)
                        
                        for m in range(W_silli.shape[0]):
                            W_silli[m] = self.rool.value_pusher2D(W_silli[m],0, row_start, col_pushed)
                        col_pushed = np.zeros_like(row_start)
                elif model=='sillburp':
                    if (col_pushed!=0).all():
                        RCO2 = self.rool.value_pusher2D(RCO2,0, row_start, col_pushed)
                        Rom = self.rool.value_pusher2D(Rom,0, row_start, col_pushed)
                        curr_TOC = self.rool.value_pusher2D(curr_TOC,0, row_start, col_pushed)
                        for huh in range(progress_of_reactions.shape[0]):
                            for bruh in range(progress_of_reactions.shape[1]):
                                progress_of_reactions[huh][bruh] = self.rool.value_pusher2D(progress_of_reactions[huh][bruh],1, row_start, col_pushed)
                                progress_of_reactions[huh][bruh] = self.rool.value_pusher2D(progress_of_reactions[huh][bruh],1, row_start, col_pushed)
                        col_pushed = np.zeros_like(row_start)
                if (curr_sill+1)<n_sills:
                    curr_sill +=1
                else:
                    break
            Frac_melt = cool.calcF(np.array(props_array[self.Temp_index], dtype = float), self.T_solidus, self.T_liquidus)*(props_array[self.rock_index]==magma_prop_dict['Lithology'])
            melt_50 = np.sum(Frac_melt<=0.5)*dx*dy*dx
            melt_10 = np.sum(Frac_melt<=0.1)*dx*dy*dx
            tot_melt10.append(melt_10)
            tot_melt50.append(melt_50)
            tot_solidus.append(np.sum(np.array(props_array[self.Temp_index], dtype = float)>self.T_solidus)*dx*dy)
            area_sills.append(np.sum(sills_emplaced>0)*dx*dy)
            if saving_factor is not None:
                if type(saving_factor)== int:
                    saving_factor = [saving_factor]
                if len(saving_factor)>1 and len(saving_factor)<3:
                    if curr_sill<n_sills:
                        if l%saving_factor[0]==0:
                            props_array_vtk.point_data['Sills'] = np.array(sills_emplaced, dtype = float).flatten()
                            props_array_vtk.point_data['Temperature'] = np.array(props_array[self.Temp_index], dtype = float).flatten()
                            props_array_vtk.point_data['Density'] = np.array(props_array[self.dense_index], dtype = float).flatten()
                            props_array_vtk.point_data['Porosity'] = np.array(props_array[self.poros_index],dtype = float).flatten()
                            props_array_vtk.point_data['TOC'] = np.array(props_array[self.TOC_index], dtype = float).flatten()
                            props_array_vtk.point_data['Lithology'] = np.array(props_array[self.rock_index]).flatten()
                            props_array_vtk.point_data['Rate of CO2'] = np.array(RCO2_silli, dtype = float).flatten()
                            props_array_vtk.point_data['Rate of organic matter'] = np.array(Rom_silli, dtype = float).flatten()
                            props_array_vtk.point_data['Vitrinite reflectance'] = np.array(percRo_silli, dtype = float).flatten()
                            props_array_vtk.save(save_dir+'/'+'Properties_'+str(l)+'.vtk')
                        else:
                            if l%saving_factor[1]==0:
                                props_array_vtk.point_data['Sills'] = np.array(sills_emplaced, dtype = float).flatten()
                                props_array_vtk.point_data['Temperature'] = np.array(props_array[self.Temp_index], dtype = float).flatten()
                                props_array_vtk.point_data['Density'] = np.array(props_array[self.dense_index], dtype = float).flatten()
                                props_array_vtk.point_data['Porosity'] = np.array(props_array[self.poros_index],dtype = float).flatten()
                                props_array_vtk.point_data['TOC'] = np.array(props_array[self.TOC_index], dtype = float).flatten()
                                props_array_vtk.point_data['Lithology'] = np.array(props_array[self.rock_index]).flatten()
                                props_array_vtk.point_data['Rate of CO2'] = np.array(RCO2_silli, dtype = float).flatten()
                                props_array_vtk.point_data['Rate of organic matter'] = np.array(Rom_silli, dtype = float).flatten()
                                props_array_vtk.point_data['Vitrinite reflectance'] = np.array(percRo_silli, dtype = float).flatten()
                                props_array_vtk.save(save_dir+'/'+'Properties_'+str(l)+'.vtk')
                elif len(saving_factor)==1:
                    if l%saving_factor[0]==0:
                        print('Saving cube')
                        props_array_vtk.point_data['Sills'] = np.array(sills_emplaced, dtype = float).flatten()
                        props_array_vtk.point_data['Temperature'] = np.array(props_array[self.Temp_index], dtype = float).flatten()
                        props_array_vtk.point_data['Density'] = np.array(props_array[self.dense_index], dtype = float).flatten()
                        props_array_vtk.point_data['Porosity'] = np.array(props_array[self.poros_index],dtype = float).flatten()
                        props_array_vtk.point_data['TOC'] = np.array(props_array[self.TOC_index], dtype = float).flatten()
                        props_array_vtk.point_data['Lithology'] = np.array(props_array[self.rock_index]).flatten()
                        props_array_vtk.point_data['Rate of CO2'] = np.array(RCO2_silli, dtype = float).flatten()
                        props_array_vtk.point_data['Rate of organic matter'] = np.array(Rom_silli, dtype = float).flatten()
                        props_array_vtk.point_data['Vitrinite reflectance'] = np.array(percRo_silli, dtype = float).flatten()
                        props_array_vtk.save(save_dir+'/'+'Properties_'+str(l)+'.vtk')
                else:
                    raise ValueError('saving_factor should have either one or two values')
        if self.calculate_closest_sill and not self.calculate_at_all_times:
            all_sills_data.to_csv(save_dir+'/sill_distances.csv')
        if model=='silli':
            if self.calculate_closest_sill:
                carbon_model_params = tot_RCO2, tot_melt10, tot_melt50, tot_solidus, area_sills, props_array, RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli, all_sills_data
            else:
                carbon_model_params = tot_RCO2, tot_melt10, tot_melt50, tot_solidus, area_sills, props_array, RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli
        elif model=='sillburp':
            if self.calculate_closest_sill:
                carbon_model_params = tot_RCO2, tot_melt10, tot_melt50, tot_solidus, area_sills, props_array_unused, RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions, all_sills_data
            else:
                carbon_model_params = tot_RCO2, tot_melt10, tot_melt50, tot_solidus, area_sills, props_array_unused, RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions 
        else:
            carbon_model_params = props_array
        return carbon_model_params

    ##########################################################################
    # STORAGE IN HDF5
    ##########################################################################
    @staticmethod
    def store_objlist_as_hd5f(fileName,cls):
        '''
        Function to save class parameters into an HDF5 file
        '''
        with h5py.File(fileName, 'w') as f:
                for item in vars(cls).items():
                    #print(item, type(item))
                    if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes,tuple,tuple,list,bool,float,int)) & isinstance(item[1], (np.ndarray, np.int64, np.float64, str, bytes,tuple,list,bool,float,int)):
                            #print(item[0])
                            f.create_dataset(item[0], data = item[1])
                    else:
                        if isinstance(item[1], (dict)):
                            for item_dict in item[1].items():
                                f.create_dataset('params_fixed_'+item_dict[0], data = item_dict[1])
                        else :
                            print(item,' is skipped',type(item),type(item[1]))
                        #raise ValueError('Cannot save %s type'%type(item))
        print('Saved the HDF5 file - ',fileName)
        return "Done"
    def load_objlist_from_hd5f_into_class(self,fileName):
        '''
        Load class parameters from a saved HDF5 file
        '''
        ans = {}
        with h5py.File(fileName, 'r') as h5file:
            for key, item in h5file.items():
                #print(key,item)
                if isinstance(item, h5py._hl.dataset.Dataset):
                    ans[key] = item[()]
        for key, value in ans.items():
            if isinstance(value,bytes):
                value = value.decode("utf-8")
                #print(key,value)
            setattr(self, key, value)
        self.update_material_prop()
        self.model_setup()
        print('Read the HDF5 file - ',fileName,' into the class')

class examples:

    def __init__(self, x = 300000, y = 35000, dx = 250, dy = 250):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        a = int(y//dy) #Number of rows
        b = int(x//dx) #Number of columns
        self.k = np.ones((a,b))*31.536
        self.cool = cool()
        self.rool = rules()
        self.sc = sill_controls(self.x, self.y, self.dx, self.dy, self.k)

    def example_run(self):
        #Dimensions of the 2D grid

        a = int(self.y//self.dy) #Number of rows
        b = int(self.x//self.dx) #Number of columns

        #Temp at the surface
        T_surf = 0 #deg C

        #Magmatic temperature
        T_mag = 1000 #deg C

        #Initializing diffusivity field

        dt = (min(self.dx,self.dy)**2)/(5*np.max(self.k))

        #Shape of the sills
        shape = 'elli'

        #Initializing the temp field
        T_field = np.zeros((a,b))
        T_field[-1,:] = T_mag
        T_field = self.cool.heat_flux(self.k, a, b, self.dx, self.dy, T_field, 'straight')
        rock = np.empty((a,b), dtype = object)

        rock[:] = 'granite'
        rock[0:int(5000/self.dy),:] = 'shale'
        rock[int((5000/self.dy)+1):int(15000/self.dy),:] = 'sandstone'
        rock[int((30000/self.dy)+1):,:] = 'peridotite'

        plot_rock = np.zeros((a,b), dtype = int)

        for i in range(a):
            for j in range(b):
                plot_rock[i,j] = self.sc.lith_plot_dict[rock[i,j]]


        labels = [key for key in self.sc.lith_plot_dict]

        os.makedirs('plots', exist_ok = True)

        # Visualize the rock array
        plt.imshow(plot_rock, cmap='viridis', extent = [0, self.x/1000, self.y/1000, 0])
        plt.ylabel('Depth (km)')
        plt.xlabel('Lateral extent (km)')
        cbar = plt.colorbar(ticks=list(self.sc.lith_plot_dict.values()), orientation = 'horizontal')
        cbar.set_ticklabels(list(labels))
        cbar.set_label('Rock Type')
        plt.title('Bedrock Composition')
        plt.savefig('plots/bedrock_distribution.png', format = 'png')
        plt.close()

        #Setting up the remaining property arrays#
        porosity = np.zeros_like(rock)
        density = np.zeros_like(rock)
        TOC = np.zeros_like(rock)

        for i in range(a):
            for j in range(b):
                porosity[i,j] = self.sc.rock_prop_dict[rock[i,j]]['Porosity']
                density[i,j] = self.sc.rock_prop_dict[rock[i,j]]['Density']
                TOC[i,j] = self.sc.rock_prop_dict[rock[i,j]]['TOC'] 
        ###Building the 3d properties array###
        props_array = np.empty((len((self.prop_dict.keys())),a,b), dtype = object)

        props_array[self.sc.Temp_index] = T_field
        props_array[self.sc.rock_index] = rock
        props_array[self.sc.poros_index] = porosity
        props_array[self.sc.dense_index] = density
        props_array[self.sc.TOC_index] = TOC

        ###Setting up sill dimensions and locations###
        min_thickness = 900 #m
        max_thickness = 3500 #m

        mar = 7
        sar = 2.5

        min_emplacement = 1000 #m
        max_emplacement = 15500 #m
        n_sills = 20000

        tot_volume = int(0.5e6*1e9)
        flux = int(30e9)

        thermal_mat_time = int(3e6)
        model_time = tot_volume/flux
        cooling_time = int(1e4)

        phase_times = [thermal_mat_time, model_time, cooling_time]
        time_steps = np.arange(0, np.sum(phase_times), dt)
        print(f'Length of time_steps:{len(time_steps)}')

        sillcube, n_sills, emplacement_params = self.build_sillcube(dt, [min_thickness, max_thickness, 500], [mar, sar], [min_emplacement, max_emplacement, 5000], [self.x//3, 2*self.x//3, self.x//6], phase_times, tot_volume, flux, n_sills)
        print('sillcube built')
        params = self.sc.get_silli_initial_thermogenic_state(props_array, dt, 'conv smooth', time = thermal_mat_time)
        current_time = params[0]
        print(f'Current time before function: {current_time}')
        carbon_model_params = params[1:]
        print('Got initial emissions state')
        props_total_array, carbon_model_params = self.sc.emplace_sills(props_array, dt, n_sills, b//2, 'conv smooth', time_steps, current_time, sillcube, carbon_model_params, emplacement_params, model = 'silli')
        print('Model Run complete')
        tot_RCO2 = carbon_model_params[0]
        plt.plot(time_steps[np.where(time_steps==current_time)[0][0]:], np.log10(tot_RCO2[np.where(time_steps==current_time)[0][0]:]))
        plt.savefig('plots/CarbonEmisisons.png', format = 'png')
