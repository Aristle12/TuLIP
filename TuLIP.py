import numpy as np
from numba import jit, int32, float64, boolean, void, types, experimental
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sc
import pypardiso as ps
from scipy.special import erf, erfinv
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator

class cool:
    def __init__(self):
        pass
    ###Functions to cool magma bodies###
    @staticmethod
    def shift_array(arr, n):
        if n >= 0:
            return np.concatenate((arr[-n:], arr[:-n]))
        else:
            return np.concatenate((arr[-n:], arr[:-n]))

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def cheat_solver(Tf, a):
        """
        Matrix taking too much time to solve?
        Here is a cheat solver for linear gradients. Can't promise it will be the same as the regular equilibrium solve, but it should be approximately accurate
        """
        T_top = Tf[0,0]
        grad = (Tf[-1,0]-Tf[0,0])/a
        for i in range(0,a):
            Tf[i,:] = T_top + i*grad
        return Tf

    def straight_solver(self, Ab,dee, a, b):
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
                return self.cheat_solver(Tnow,a)
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
                return self.straight_solver(Af, bee, a, b)
            elif method == 'Jacobian':
                return self.JacobianIt(Af, bee, a, b)
            elif method == 'GS':
                return self.GSIt(Af, bee, a, b)





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
        if np.isnan(Af):
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

    @jit(forceobj = True)
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
        kiph, kimh, kjph, kjmh = self.avg_perm(k)
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

            return self.perm_smoothed_solve(k, a, b, dx, dy, dt, Tnow, q, Af, H)
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
            return self.perm_chain_solve(k, a, b, dx, dy, dt, Tnow, q, Af, H)
        else:
            raise ValueError('diff_solve solution method not supported')

class emit:
    def __init__(self):
        pass
    
    @staticmethod
    def get_init_CO2_percentages(T_field, lithology, density, dy):
        a,b = lithology.shape
        break_parser = (lithology=='dolostone') | (lithology=='limestone') | (lithology=='marl') | (lithology=='evaporite')
        dolo = loadmat('dat/Dolostone.mat')
        evap = loadmat('dat/DolostoneEvaporite.mat')
        marl = loadmat('dat/Marl.mat')
        T = np.array(dolo['Dolo']['T'][0][:][0])
        P = np.array(dolo['Dolo']['P'][0][0][0])
        T = T[:,0]
        dolo_CO2 = np.array(dolo['Dolo']['CO2'][0][0])
        evap_CO2 = np.array(evap['Dol_ev']['CO2'][0][0])
        marl_CO2 = np.array(marl['Marl']['CO2'][0][0])

        dolo_inter = RegularGridInterpolator((T,P), dolo_CO2)
        evap_inter = RegularGridInterpolator((T,P), evap_CO2)
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
                            print('Warning: Limestone pressure out of bounds. Skipping')

                    elif lithology[i,j] == 'evaporite':
                        init_CO2[i,j] = evap_inter([T_field[i,j],pressure])
                    elif lithology[i,j]=='marl':
                        init_CO2[i,j]== marl_inter([T_field[i,j],pressure])
        return init_CO2
    @staticmethod
    def get_breakdown_CO2(T_field, lithology, density, breakdownCO2, dy, dt):
        break_parser = (lithology=='dolostone') | (lithology=='limestone') | (lithology=='marl') | (lithology=='evaporite')
        a, b = T_field.shape
        dolo = loadmat('dat/Dolostone.mat')
        evap = loadmat('dat/DolostoneEvaporite.mat')
        marl = loadmat('dat/Marl.mat')
        T = np.array(dolo['Dolo']['T'][0][:][0])
        P = np.array(dolo['Dolo']['P'][0][0][0])
        T = T[:,0]
        dolo_CO2 = np.array(dolo['Dolo']['CO2'][0][0])
        evap_CO2 = np.array(evap['Dol_ev']['CO2'][0][0])
        marl_CO2 = np.array(marl['Marl']['CO2'][0][0])

        dolo_inter = RegularGridInterpolator((T,P), dolo_CO2)
        evap_inter = RegularGridInterpolator((T,P), evap_CO2)
        marl_inter = RegularGridInterpolator((T,P), marl_CO2)
        curr_breakdown_CO2 = np.zeros_like(T_field)
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
                            curr_breakdown_CO2[i,j] = dolo_inter([T_field[i,j],pressure])
                        except ValueError:
                            curr_breakdown_CO2[i,j] = 0
                            print(f'Warning: Limestone pressure out of bounds at {i} {j}. Skipping')

                    elif lithology[i,j] == 'evaporite':
                        try:
                            curr_breakdown_CO2[i,j] = evap_inter([T_field[i,j],pressure])
                        except ValueError:
                            curr_breakdown_CO2[i,j] = 0
                            print(f'Warning: Evaporite pressure out of bounds at {i} {j}. Skipping')
                    elif lithology[i,j]=='marl':
                        try:
                            curr_breakdown_CO2[i,j]== marl_inter([T_field[i,j],pressure])
                        except ValueError:
                            curr_breakdown_CO2[i,j] = 0
                            print(f'Warning: Marl pressure out of bounds at {i} {j}. Skipping')
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
        RCO2 = Rom*3.67
        return RCO2, Rom, percRo, TOC, W

    def analytical_Ro(T_field, dT, density, lithology, porosity, I_prev, TOC_prev, dt, TOCo, W):
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
        RCO2 = Rom*3.67
        return RCO2, Rom, percRo, I_curr, TOC

    def analyticalRo_I(T_field):
        '''
        Initialization of I for the SILLi carbon model
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
    @jit(forceobj=True)
    def sillburp(T_field, TOC_prev, density, lithology, porosity, dt, reaction_energies, TOCo=np.nan, oil_production_rate=0, progress_of_reactions=np.nan, rate_of_reactions = np.nan, weights = None):
        if np.isnan(TOCo).all():
            TOCo = TOC_prev
        
        a, b = T_field.shape
        calc_parser = (lithology == 'shale') | (lithology == 'sandstone')
        sqrt_2pi = np.sqrt(2 * np.pi)
        n_reactions = 4
        reactants = ['LABILE', 'REFRACTORY', 'VITRINITE', 'OIL']
        OIL = reactants.index('OIL')
        no_reactions = [7, 21, 55, 7]  # Number of reactions for each kerogen type
        
        As = [1.58e13, 1.83e18, 4e10, 1e13]  # pre-exponential constants for the different reactions
        '''
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
        '''
        if np.isnan(progress_of_reactions).all():
            progress_of_reactions = np.zeros((n_reactions, max(no_reactions), a, b))
            progress_of_reactions_old = np.zeros_like(progress_of_reactions)
            rate_of_reactions = np.zeros_like(progress_of_reactions)
            oil_production_rate = np.zeros((a,b))
        
        else:
            progress_of_reactions_old = progress_of_reactions.copy()
        S_over_k = np.zeros((a,b))
        
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
                            S_over_k[i,j] = 0.0 if reaction_rate == 0 else oil_production_rate[i,j] / reaction_rate / no_reactions[OIL]
                            progress_of_reactions[i_reaction, i_approx, i, j] = min(1.0 - S_over_k[i,j] - (1.0 - initial_product_conc - S_over_k[i,j]) * np.exp(-reaction_rate * dt), 1)
                            rate_of_reactions[i_reaction, i_approx, i, j] = (1.0 - initial_product_conc - S_over_k[i,j]) * (1.0 - np.exp(-reaction_rate * dt)) / dt
                        
                        if i_reaction == reactants.index('LABILE'):
                            if i_approx == 0:
                                oil_production_rate[i,j] = 0.0
                            oil_production_rate += -reaction_rate * (1.0 - progress_of_reactions[i_reaction, i_approx, i, j]) / no_reactions[reactants.index('LABILE')]
                            if i_approx == no_reactions[reactants.index('LABILE')] - 1:
                                oil_production_rate[i,j] *= (1.0 - mass_frac_labile_to_gas)
        
        time_step_progress = progress_of_reactions - progress_of_reactions_old
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
                raise IndexError(f'Shape of weights must be {products_progress.shape}')
            for i_reaction in range(0,n_reactions):
                products_progress[i_reaction,:,:] = np.mean(progress_of_reactions[i_reaction,0:no_reactions[i_reaction],:,:], axis = 0)
            products_progress = np.average(products_progress, axis=0, weights = weights)
            TOC = TOCo * (1-products_progress) * calc_parser
            dTOC = (TOC_prev - TOC)/dt
            Rom = (1 - porosity) * density * dTOC
            RCO2 = Rom * 3.67
        
        return RCO2, Rom, progress_of_reactions, oil_production_rate, TOC, rate_of_reactions


class rules:
    def __init__(self):
        pass
    @staticmethod
    def to_emplace(t_now, t_thresh):
        if (t_now<t_thresh):
            return False
        elif t_now>=t_thresh:
            return True

    @staticmethod
    def build_lith_dict(lithology):
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
        Emplacing a simple sill without a dike tail
        """
        T_field[int(height-(thick//2)):int(height+(thick//2)), int(x_space-(width//2)):int(x_space+(width//2))] = T_mag
        return T_field
    @staticmethod
    def circle_sill(T_field, x_space, height, r, T_mag, a, b, dx, dy):
        """
        Emplacing a simple circular sill without the dike tail
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
        Get heights spacing randomly picked from a uniform distribution
        """
        heights = np.round(np.random.uniform(l_sill, h_sill, n_sills)/dy)
        return heights

    @staticmethod
    def uniform_x(n_sills, x_min, x_max, dx):
        """
        Nodes for x-coordinate space chosen as a random normal distribution
        """
        space = np.round(np.random.uniform(x_min, x_max, n_sills)/dx)
        return space
    @staticmethod
    def empirical_CDF(n_sills, xarray, cdf):
        """Function to give random numbers from a specific empirical distribution
        n_sills - number of sills needed int
        xarray - array of domain for empirical CDF
        cdf - array of CDF for the x array"""
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
        This function updates the associated rock properties once everything has shiftes. This is done to avoid thermopgenic carbon generation from popints that are now pure magma'''
        prop = np.zeros_like(lithology)
        for rock in lith_dict.keys():
            prop[lithology==rock] = prop_dict[rock][property]
        return prop
    @staticmethod
    def value_pusher2D(array, new_value, row_index, push_amount):
        a,b = array.shape
        if len(row_index) != b or len(push_amount) != b:
            raise ValueError("row_index and push_values must have the same length as the number of columns")
        for j in range(b):
            if row_index[j] + push_amount[j] >= a:
                    raise ValueError(f"Push value for column {j} exceeds array bounds")
            array[row_index[j]+push_amount[j]:,j] = array[row_index[j]:a-push_amount[j], j]
            array[row_index[j]:row_index[j]+push_amount[j],j] = new_value
        return array


    '''
    def index_finder(array, string):
        a,b = array.shape
        string_index = np.empty((a,b), dtype=bool)
        for i in range(a):
            for j in range(b):
                if string in array[i,j]:
                    string_index[i,j] = True
                else:
                    string_index[i,j] = False
        return string_index
        '''
    @staticmethod
    def index_finder(array, string):
        if not np.issubdtype(array.dtype, np.str_):
            array = array.astype(str)
        
        def contains_string(element):
            return string in element
        
        string_index = np.vectorize(contains_string)(array)
        return string_index


    def mult_sill(self, T_field,  majr, minr, height, x_space, dx, dy, rock = np.array([]), emplace_rock = 'basalt', T_mag = 1000, shape = 'elli', dike_empl = True, push = False):
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




    '''
    Broken function
    #@jit
    def mult_sill(T_field, majr, minr, height, x_space, dx, dy, dike_net, cm_array = [], cmb = [], rock = np.array([]), T_mag = 1000, shape = 'rect', dike_empl = True, cmb_exists = False):
        a,b = T_field.shape
        if dike_empl:
            T_field[int(height):-1,int(x_space)] = T_mag
            if rock.size>0:
                rock.loc[int(height):-1,int(x_space)] = 'basalt'

        if cmb_exists:
            new_dike = np.zeros_like(T_field)
            if shape == 'rect':
                new_dike[int(height-(minr//2)):int(height+(minr//2)), int(x_space-(majr//2)):int(x_space+(majr//2))] = 1
            elif shape=='elli':
                x = np.arange(0,b*dx, dx)
                y = np.arange(0, a*dy, dy)
                majr = majr*dx
                minr = minr*dy
                for m in range(0, a):
                    for n in range(0, b):
                        if (x_dist+y_dist)<=1:
                            T_field[m,n]=T_mag
                            new_dike[m,n] = 1
            cm_mov = np.sum(new_dike, axis = 0)
            cmb = cmb + cm_mov
            for l in range(0, b):
                if cm_mov[l]!=0:
                    for m in range(0,a):
                        if new_dike[m,l]==1:
                            T_field = value_pusher(T_field, T_mag,[m,l], cm_mov[l])
                            rock = value_pusher(rock,'basalt',[m,l],cm_mov[l])
                            continue
            for i in range(0, a):
                for j in range(0, b):
                    if i>cmb[i]:
                        cm_array.loc[i,j] = 'mantle'
            return T_field, dike_net, rock, cm_array
        else:
            new_dike = np.zeros_like(T_field)
            if shape == 'rect':
                T_field[int(height-(minr//2)):int(height+(minr//2)), int(x_space-(majr//2)):int(x_space+(majr//2))] = T_mag
                new_dike[int(height-(minr//2)):int(height+(minr//2)), int(x_space-(majr//2)):int(x_space+(majr//2))] = 1
            elif shape == 'elli':
                x = np.arange(0,b*dx, dx)
                y = np.arange(0, a*dy, dy)
                majr = majr*dx
                minr = minr*dy
                for m in range(0, a):
                    for n in range(0, b):
                        x_dist = ((x[m]-x[int(x_space)])**2)/(((majr)//2)**2)
                        y_dist = ((y[n]-y[int(height)])**2)/(((minr)//2)**2)
                        if (x_dist+y_dist)<=1:
                            T_field[m,n]=T_mag
                            new_dike[m,n] = 1
                            if rock.size>0:
                                rock.loc[n,m] = 'basalt'
            dike_net = dike_net + new_dike
            return T_field, dike_net, rock
    '''
    def get_H(T_field, rho, CU, CTh, CK, T_sol, dike_net, a, b):
        """
        Function to calculate external heat sources generated through latent heat of crystallization and radiactive heat generation
        T_field = Temp field, int
        rho = Density kg/m3
        CU, CTh = U, Th concentrations in ppm, array
        CK = K conc in wt %, array
        T_sol = Solidus temperature
        """
        J = 0.25 #J/kg latent heat of crystallization
        Cp = 1450 #J/kgK specific heat capacity
        H = np.zeros_like(T_field)
        for i in range(0,a):
            for j in range(0, b):
                if T_field[i,j]>T_sol and dike_net[i,j]!=0:
                    H[i,j] = J/(rho*Cp)
        A = rho*1e-5*(9.52*CU + 2.56*CTh + 3.48*CK) #Formula from Rybach and Cermack 1982 - Radioactive heat generation in rocks
        H = H+A
        return H
    @staticmethod
    def get_diffusivity(T_field, lithology):
        K = 31.536*np.ones_like(T_field)
        return K


    def sill_3Dcube(self, x, y, z, dx, dy, n_sills, x_coords, y_coords, z_coords, maj_dims, min_dims, empl_times, shape='elli', dike_tail=False):
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



        maj_dims = maj_dims/dx
        min_dims = min_dims/dy

        if shape == 'elli':
            for l in trange(n_sills):
                mask = ((((z_len-z_coords[l])**2)/maj_dims[l]**2)+(((y_len-y_coords[l])**2)/min_dims[l]**2)
                +(((x_len-x_coords[l])**2)/maj_dims[l]**2))<=1
                sillcube[mask] += '_' + str(l) + 's' + str(empl_times[l])
                if dike_tail:
                    sillcube[int(z_coords[l]), int(y_coords[l]):-1, int(x_coords[l])] += '_' + str(l) + 's' + str(empl_times[l])

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
                    sillcube[int(z_coords[l]), int(y_coords[l]):-1, int(x_coords[l])] += '_' + str(l) + 's' + str(empl_times[l])

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
    @staticmethod
    def cmb_3Dsill(cm_array, cmb, sillcube, nrep, z_index):
        new_sill = sillcube==nrep
        new_sill = new_sill[z_index,:,:]
        cm_mov = np.sum(new_sill, axis = 0)
        cmb = cmb+cm_mov
        for i in range(0, cm_array[:,0]):
                for j in range(0, cm_array[0,:]):
                    if i>cmb[i]:
                        cm_array.loc[i,j] = 'mantle'
        return cm_array

    '''Broken function
    def array_shifter(array_old,array_new, sillcube_z, n_rep, curr_empl_time):
        string_finder = str(n_rep)+'s'+str(curr_empl_time)
        y_shifts = np.sum(np.array([string_finder in sillcube_z]).astype(int), axis = 0)
        for i in range(0, len(array_new[:,0])):
            if y_shifts[i]!=0:
                for j in range(0, len(array_new[0,:])):
                    if array_old[i,j]!= array_new[i,j]:
                        array_new[i,j+y_shifts::-1] = array_old[i,::-y_shifts]
                        continue
        return array_new
        '''

    def sill3D_pushy_emplacement(self, props_array, props_dict, sillcube, n_rep, mag_props_dict, z_index, curr_empl_time):
        string_finder = str(n_rep)+'s'+str(curr_empl_time)
        T_field_index = props_dict['Temperature']
        T_field = props_array[T_field_index]
        a,b = T_field.shape
        if len(sillcube.shape)!=3:
            raise IndexError('sillcube array must be three-dimensional')
        if T_field.size==0:
            raise ValueError("Temperature values in props_array cannot be empty")
        new_dike = np.zeros_like(T_field)
        new_dike[self.index_finder(sillcube[z_index], string_finder)] = 1
        columns_pushed = np.sum(new_dike, axis =0, dtype=int)
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
        if np.sum(self.index_finder(sillcube[z_index], string_finder))==0:
            #print(f'Sill {n_rep} was NOT emplaced in this slice')
            pass
        else:
            print(f'Sill {n_rep} was emplaced in this slice')
            reverse_prop_dict = {value: key for key, value in props_dict.items()}
            for i in [listy for listy in props_dict.values()]:
                props_array[i] = self.value_pusher2D(props_array[i],mag_props_dict[reverse_prop_dict[i]],row_push_start, columns_pushed)
        return props_array, row_push_start, columns_pushed

class sill_controls:
    def __init__(self):
        self.cool = cool()
        self.rool = rules() 
        ###Setting up the properties dictionary to translate properties to indices for 3D array###
        self.prop_dict = {'Temperature':0,
                    'Lithology':1,
                    'Porosity':2,
                    'Density':3,
                    'TOC':4}
        
        self.rev_prop_dict = {0:'Temperature',
                        1:'Lithology',
                        2:'Porosity',
                        3:'Density',
                        4:'TOC'
        }

        #Lithology dictionary to translate rock types into numerical codes for numpy arrays
        self.lith_plot_dict = {'granite':0,
                        'shale':1,
                        'sandstone':2,
                        'peridotite':3,
                        'basalt':4}
        self.Temp_index = self.prop_dict['Temperature']
        self.rock_index = self.prop_dict['Lithology']
        self.poros_index = self.prop_dict['Porosity']
        self.dense_index = self.prop_dict['Density']
        self.TOC_index = self.prop_dict['TOC']

        self.magma_prop_dict = {'Temperature': 1000,
                    'Lithology': 'basalt',
                    'Porosity': 0.2,
                    'Density': 2850, #kg/m3
                    'TOC':0} #wt%

        self.rock_prop_dict = {
            "shale":{
                'Porosity':0.1,
                'Density':2500,
                'TOC':7
            },
            "sandstone":{
                'Porosity':0.2,
                'Density':2600,
                'TOC':2.5
            },
            "limestone":{
                'Porosity':0.2,
                'Density':2600,
                'TOC':2.5
            },
            "granite":{
                'Porosity':0.05,
                'Density':2700,
                'TOC':0
            },
            "basalt":{
                'Porosity': 0.2,
                'Density': 2850, #kg/m3
                'TOC':0
            },
            "peridotite":{
                'Porosity': 0.05,
                'Density': 3100, #kg/m3
                'TOC':0
            }
        }
        


    #Setting up the remaining property arrays#
    def get_physical_properties(self, rock, rock_prop_dict = None):
        if rock_prop_dict==None:
            rock_prop_dict = self.rock_prop_dict
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
    def func_assigner(self, func, *args, **kwargs):
        result = func(*args,**kwargs)
        # If the result is a tuple or list, enumerate it
        #if isinstance(result, (tuple, list)):
        #    enumerated_result = list(enumerate(result))
        #    return enumerated_result
        # If the result is a single value, return it as is
        return result
    def build_sillcube(self, x, y, dx, dy, dt, thickness_range, aspect_ratio, depth_range, lat_range, phase_times, tot_volume, flux, n_sills, shape = 'elli', depth_function = None, lat_function = None, dims_function = None):
        dims_empirical = False
        min_thickness = thickness_range[0] #m
        max_thickness = thickness_range[1] #m
        sd_min = thickness_range[2]
        mar = aspect_ratio[0]
        sar = aspect_ratio[1]

        min_emplacement = depth_range[0] #m
        max_emplacement = depth_range[1] #m
        sd_empl = depth_range[2]

        x_min = lat_range[0]
        x_max = lat_range[1]
        sd_x = lat_range[2]

        if depth_function==None or depth_function=='normal':
            depth_function = self.rool.randn_heights
            depth_input_params = (n_sills, max_emplacement, min_emplacement, sd_empl, dy)
        elif depth_function == 'uniform':
            depth_function = self.rool.uniform_heights
            depth_input_params = (n_sills, min_emplacement, max_emplacement, dy)
        elif depth_function=='empirical':
            depth_function = self.rool.empirical_CDF
            depth_input_params = (n_sills, depth_range[0], depth_range[1])

        empl_heights = self.func_assigner(depth_function, *depth_input_params)
        
        if lat_function==None or lat_function=='normal':
            lat_function = self.rool.x_spacings
            lat_input_params = (n_sills, x_min, x_max, sd_x, dx)
        elif lat_function=='uniform':
            lat_function = self.rool.uniform_x
            lat_input_params = (n_sills, x_min, x_max, dx)
        elif lat_function == 'empirical':
            lat_function = self.rool.empirical_CDF
            lat_input_params = (n_sills, lat_range[0], lat_range[1])
        
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
        plt.savefig('plots/Depth.png', format = 'png', bbox_inches = 'tight')
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
        plt.savefig('plots/WidthThickness.png', format = 'png', bbox_inches = 'tight')
        plt.close()
        
        thermal_maturation_time = phase_times[0]
        total_empl_time = tot_volume/flux
        cooling_time = phase_times[1]
        model_time = total_empl_time+thermal_maturation_time+cooling_time
        time_steps = np.arange(model_time,step=dt)
        saving_time_step_index = np.min(np.where(time_steps>=thermal_maturation_time)[0])
        print(saving_time_step_index, time_steps[saving_time_step_index])
        empl_times = []
        plot_time = []
        cum_volume = []

        if shape == 'elli':
            print(width, thickness)
            volume = (4*np.pi/3)*width*width*thickness
        elif shape=='rect':
            volume = width*width*thickness

        unemplaced_volume = 0
        #print(f'{np.sum(volume):.5e}, {float(tot_volume):.5e}, {np.sum(volume)<tot_volume}')
        n = 0
        for l in range(len(time_steps)):
            if time_steps[l]<thermal_maturation_time:
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
                    n_sills = n
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
        plt.savefig('plots/VolumeTime.png', format = 'png', bbox_inches = 'tight')
        plt.close()

        z_coords = lat_function(n_sills, x_min, x_max, sd_x, dx)
        sillcube = self.rool.sill_3Dcube(x,y,x,dx,dy,n_sills, x_space, empl_heights, z_coords, width, thickness, empl_times,shape)

        params = np.array([empl_times, empl_heights, x_space, width, thickness])
        return sillcube, n_sills, params
    
    def get_silli_initial_thermogenic_state(self, props_array, dx, dy, dt, method, k=np.nan, time = np.nan, lith_plot_dict = None, rock_prop_dict = None):
        try:
            if not lith_plot_dict or all(value is None for value in lith_plot_dict.values()):
                lith_plot_dict = self.lith_plot_dict
            if not rock_prop_dict or all(value is None for value in rock_prop_dict.values()):
                rock_prop_dict = self.rock_prop_dict
        except AttributeError:
            pass
        density = props_array[self.dense_index]
        porosity = props_array[self.poros_index]
        rock = props_array[self.rock_index]
        T_field = props_array[self.Temp_index]
        breakdown_CO2 = np.zeros_like(T_field)
        t = 0
        a, b = props_array[0].shape
        H = np.zeros((a,b))
        TOC = self.rool.prop_updater(props_array[self.rock_index], lith_plot_dict, rock_prop_dict, 'TOC')
        if np.isnan(k).all():
            k = self.rool.get_diffusivity(props_array[self.Temp_index], props_array[self.rock_index])
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
            tot_RCO2.append(np.sum(RCO2_silli)+np.sum(breakdown_CO2))
            iter_thresh = int(1e7//dt)
            current_time = 0
            with tqdm(total = iter_thresh, desc = 'Processing') as pbar:
                while iter<iter_thresh and diff>thresh:
                    curr_TOC_silli = props_array[self.TOC_index]
                    RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli = emit.SILLi_emissions(T_field, density, rock, porosity, curr_TOC_silli, dt, TOC, W_silli)
                    if (rock=='limestone').any():    
                        breakdown_CO2, _ = emit.get_breakdown_CO2(T_field, rock, density, breakdown_CO2, dy, dt)
                    props_array[self.TOC_index] = curr_TOC_silli
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
            for l in trange(0, len(t_steps)):
                T_field = self.cool.diff_solve(k, a, b, dx, dy, dt, T_field, np.nan, method, H)
                props_array[self.Temp_index] = T_field
                curr_TOC_silli = props_array[self.TOC_index]
                TOC = self.rool.prop_updater(rock, lith_plot_dict, rock_prop_dict, 'TOC')
                if l==0:
                    RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli = emit.SILLi_emissions(T_field, density, rock, porosity, TOC, dt)
                    if (rock=='limestone').any():
                        breakdown_CO2 = emit.get_init_CO2_percentages(T_field, rock, density, dy)
                else:
                    RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli = emit.SILLi_emissions(T_field, density, rock, porosity, curr_TOC_silli, dt, TOC, W_silli)
                    if (rock=='limestone').any():    
                        breakdown_CO2, _ = emit.get_breakdown_CO2(T_field, rock, density, breakdown_CO2, dy, dt)
                props_array[self.TOC_index] = curr_TOC_silli
                tot_RCO2.append(np.sum(RCO2_silli)+np.sum(breakdown_CO2))
                current_time = t_steps[l]
        props_array[self.Temp_index] = T_field
        props_array[self.dense_index] = density
        props_array[self.rock_index] = rock
        props_array[self.poros_index] = porosity
        return current_time, tot_RCO2, props_array, RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli

    def get_sillburp_initial_thermogenic_state(self, props_array, dx, dy, dt, method, sillburp_weights = None, k=np.nan, time = np.nan, lith_plot_dict = None, rock_prop_dict = None):
        try:
            if not lith_plot_dict or all(value is None for value in lith_plot_dict.values()):
                lith_plot_dict = self.lith_plot_dict
            if not rock_prop_dict or all(value is None for value in rock_prop_dict.values()):
                rock_prop_dict = self.rock_prop_dict
        except AttributeError:
            pass
        density = props_array[self.dense_index]
        porosity = props_array[self.poros_index]
        rock = props_array[self.rock_index]
        T_field = props_array[self.Temp_index]
        breakdown_CO2 = np.zeros_like(T_field)
        t = 0
        a, b = props_array[0].shape
        H = np.zeros((a,b))
        TOC = self.rool.prop_updater(props_array[self.rock_index], lith_plot_dict, rock_prop_dict, 'TOC')
        if np.isnan(k).all():
            k = self.rool.get_diffusivity(props_array[self.Temp_index], props_array[self.rock_index])
        if np.isnan(time):
            T_field = self.cool.diff_solve(k, a, b, dx, dy, dt, T_field, np.nan, method, H)
            props_array[self.Temp_index] = T_field
            curr_TOC = props_array[self.TOC_index]
            reaction_energies = emit.get_sillburp_reaction_energies()
            RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions = emit.sillburp(T_field, TOC, density, rock, porosity, dt, reaction_energies, weights=sillburp_weights)
            if (rock=='limestone').any():
                breakdown_CO2 = emit.get_init_CO2_percentages(T_field, rock, density, dy)
            props_array[self.TOC_index] = curr_TOC_silli
            diff = 1e6
            iter = 0
            thresh = 1e-3
            t+=dt
            tot_RCO2 = []
            iter_thresh = int(1e7//dt)
            tot_RCO2.append(np.sum(RCO2)+np.sum(breakdown_CO2))
            current_time = 0
            with tqdm(total = iter_thresh, desc = 'Processing') as pbar:
                while iter<iter_thresh and diff>thresh:
                    curr_TOC = props_array[self.TOC_index]
                    RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions = emit.sillburp(T_field, curr_TOC, density, rock, porosity, dt, reaction_energies, TOC, oil_production_rate, progress_of_reactions, rate_of_reactions, weights=sillburp_weights)
                    if (rock=='limestone').any():    
                        breakdown_CO2, _ = emit.get_breakdown_CO2(T_field, rock, density, breakdown_CO2, dy, dt)
                    props_array[self.TOC_index] = curr_TOC_silli
                    tot_RCO2.append(np.sum(RCO2)+np.sum(breakdown_CO2))
                    if iter>0:
                        diff = tot_RCO2[-2]-tot_RCO2[-1]
                    iter+=1
                    current_time += dt

                    pbar.update(1)
                    pbar.set_postfix({"Change": diff})
        else:
            t_steps = np.linspace(0, time, dt)
            tot_RCO2 = []
            for l in trange(0, len(t_steps)):
                T_field = self.cool.diff_solve(k, a, b, dx, dy, dt, T_field, np.nan, method, H)
                props_array[self.Temp_index] = T_field
                curr_TOC_silli = props_array[self.TOC_index]
                TOC = self.rool.prop_updater(rock, lith_plot_dict, rock_prop_dict, 'TOC')
                if l==0:
                    RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions = emit.sillburp(T_field, TOC, density, rock, porosity, dt, reaction_energies, weights=sillburp_weights)
                    if (rock=='limestone').any():
                        breakdown_CO2 = emit.get_init_CO2_percentages(T_field, rock, density, dy)
                else:
                    RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions = emit.sillburp(T_field, curr_TOC, density, rock, porosity, dt, reaction_energies, TOC, oil_production_rate, progress_of_reactions, rate_of_reactions, weights=sillburp_weights)
                    if (rock=='limestone').any():    
                        breakdown_CO2, _ = emit.get_breakdown_CO2(T_field, rock, density, breakdown_CO2, dy, dt)
                props_array[self.TOC_index] = curr_TOC_silli
                tot_RCO2.append(np.sum(RCO2)+np.sum(breakdown_CO2))
                current_time = t_steps[l]
        props_array[self.Temp_index] = T_field
        props_array[self.dense_index] = density
        props_array[self.rock_index] = rock
        props_array[self.poros_index] = porosity
        return current_time, tot_RCO2, props_array, RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions

    def emplace_sills(self,props_array, k, dx, dy, dt, n_sills, z_index, cool_method, time_steps, current_time, sillcube, carbon_model_params, emplacement_params, model=None, H = np.nan, rock_prop_dict = None, lith_plot_dict = None, prop_dict = None, magma_prop_dict = None):
        saving_time_step_index = np.where(time_steps==current_time)[0][0]
        print(f'Current time: {current_time}')
        print(f'saving_time_step_index: {saving_time_step_index}')
        shape_index = [len(time_steps[saving_time_step_index:])]+list(props_array.shape)
        props_total_array = np.empty(shape_index, dtype = object)
        if lith_plot_dict==None:
            lith_plot_dict = self.lith_plot_dict
        if rock_prop_dict==None:
            rock_prop_dict = self.rock_prop_dict
        if prop_dict==None:
            prop_dict = self.prop_dict
        if magma_prop_dict==None:
            magma_prop_dict = self.magma_prop_dict
        rock = props_array[self.rock_index]
        density = props_array[self.dense_index]
        porosity = props_array[self.poros_index]
        T_field = props_array[self.Temp_index]
        a,b = sillcube.shape[1], sillcube.shape[2]
        breakdown_CO2 = np.zeros_like(T_field)
        if np.isnan(H):
            H = np.zeros((a,b))
        if model=='silli':
            tot_RCO2, props_array_unused, RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli = carbon_model_params
        elif model =='sillburp':
           tot_RCO2, props_array_unused, RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions, sillburp_weights = carbon_model_params
           reaction_energies = emit.get_sillburp_reaction_energies()
        elif model==None:
            pass
        else:
            raise ValueError(f'model is {model}, but must be either silli or sillburp')
        empl_times, empl_heights, x_space, width, thickness = emplacement_params
        curr_sill = 0
        dV = dx*dx*dy
        for l in trange(saving_time_step_index, len(time_steps)):
            curr_time = time_steps[l]
            T_field = self.cool.diff_solve(k, a, b, dx, dy, dt, T_field, np.nan, cool_method, H)
            curr_TOC_silli = props_array[self.TOC_index]
            TOC = self.rool.prop_updater(rock, lith_plot_dict, rock_prop_dict, 'TOC')
            if model=='silli':
                if l!=saving_time_step_index:
                    RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli = emit.SILLi_emissions(T_field, density, rock, porosity, curr_TOC_silli, dt, TOC, W_silli)
                    RCO2_silli = RCO2_silli*density*dV/100
                    tot_RCO2.append(np.sum(RCO2_silli))
            elif model=='sillburp':
                if l!=saving_time_step_index:
                    RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions = emit.sillburp(T_field, curr_TOC, density, rock, porosity, dt, reaction_energies, TOC, oil_production_rate, progress_of_reactions, rate_of_reactions, weights=sillburp_weights)
                    RCO2 = RCO2*density*dV/100
                    tot_RCO2.append(np.sum(RCO2))
            if (rock=='limestone').any():    
                breakdown_CO2, _ = emit.get_breakdown_CO2(T_field, rock, density, breakdown_CO2, dy, dt)
            props_array[self.TOC_index] = curr_TOC_silli
            while time_steps[l]==empl_times[curr_sill] and curr_sill<n_sills:
                #print(f'Now emplacing sill {curr_sill}')
                props_array, row_start, col_pushed = self.rool.sill3D_pushy_emplacement(props_array, prop_dict, sillcube, curr_sill, magma_prop_dict, z_index, empl_times[curr_sill])
                if model=='silli':
                    if (col_pushed!=0).all():
                        RCO2_silli = self.rool.value_pusher2D(RCO2_silli,0, row_start, col_pushed)
                        Rom_silli = self.rool.value_pusher2D(Rom_silli,0, row_start, col_pushed)
                        percRo_silli =self.rool.value_pusher2D(percRo_silli, 0, row_start, col_pushed)
                        curr_TOC_silli = self.rool.value_pusher2D(curr_TOC_silli,0, row_start, col_pushed)
                        for m in range(W_silli.shape[0]):
                            W_silli[m] = self.rool.value_pusher2D(W_silli[m],0, row_start, col_pushed)
                elif model=='sillburp':
                    if (col_pushed!=0).all():
                        RCO2 = self.rool.value_pusher2D(RCO2,0, row_start, col_pushed)
                        Rom = self.rool.value_pusher2D(Rom,0, row_start, col_pushed)
                        oil_production_rate = self.rool.value_pusher2D(oil_production_rate,0, row_start, col_pushed)
                        curr_TOC = self.rool.value_pusher2D(curr_TOC,0, row_start, col_pushed)
                        for huh in range(progress_of_reactions.shape[0]):
                            for bruh in range(progress_of_reactions.shape[1]):
                                progress_of_reactions[huh][bruh] = self.rool.value_pusher2D(progress_of_reactions[huh][bruh],1, row_start, col_pushed)
                                progress_of_reactions[huh][bruh] = self.rool.value_pusher2D(progress_of_reactions[huh][bruh],1, row_start, col_pushed)

                if (curr_sill+1)<n_sills:
                    curr_sill +=1
                else:
                    break
                dV = dx*dx*dy
            props_total_array[l-saving_time_step_index] = props_array
        if model=='silli':
            carbon_model_params = tot_RCO2, props_array, RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli
        elif model=='sillburp':
            carbon_model_params = tot_RCO2, props_array_unused, RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions
        else:
            carbon_model_params = None
        return props_total_array, carbon_model_params

    def example_run(self):
        #Dimensions of the 2D grid
        x = 300000 #m - Horizontal extent of the crust
        y = 35000 #m - Vertical thickness of the crust

        dx = 250 #m node spacing in x-direction
        dy = 250 #m node spacing in y-direction

        a = int(y//dy) #Number of rows
        b = int(x//dx) #Number of columns

        #Temp at the surface
        T_surf = 0 #deg C

        #Magmatic temperature
        T_mag = 1000 #deg C

        #Initializing diffusivity field
        k = np.ones((a,b))*31.536 #m2/yr

        dt = (min(dx,dy)**2)/(5*np.max(k))

        #Shape of the sills
        shape = 'elli'

        #Initializing the temp field
        T_field = np.zeros((a,b))
        T_field[-1,:] = T_mag
        T_field = self.cool.heat_flux(k, a, b, dx, dy, T_field, 'straight')
        rock = np.empty((a,b), dtype = object)

        rock[:] = 'granite'
        rock[0:int(5000/dy),:] = 'shale'
        rock[int((5000/dy)+1):int(15000/dy),:] = 'sandstone'
        rock[int((30000/dy)+1):,:] = 'peridotite'

        plot_rock = np.zeros((a,b), dtype = int)

        for i in range(a):
            for j in range(b):
                plot_rock[i,j] = self.lith_plot_dict[rock[i,j]]


        labels = [key for key in self.lith_plot_dict]

        # Visualize the rock array
        plt.imshow(plot_rock, cmap='viridis', extent = [0, x/1000, y/1000, 0])
        plt.ylabel('Depth (km)')
        plt.xlabel('Lateral extent (km)')
        cbar = plt.colorbar(ticks=list(self.lith_plot_dict.values()), orientation = 'horizontal')
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
                porosity[i,j] = self.rock_prop_dict[rock[i,j]]['Porosity']
                density[i,j] = self.rock_prop_dict[rock[i,j]]['Density']
                TOC[i,j] = self.rock_prop_dict[rock[i,j]]['TOC'] 
        ###Building the 3d properties array###
        props_array = np.empty((len((self.prop_dict.keys())),a,b), dtype = object)

        props_array[self.Temp_index] = T_field
        props_array[self.rock_index] = rock
        props_array[self.poros_index] = porosity
        props_array[self.dense_index] = density
        props_array[self.TOC_index] = TOC

        ###Setting up sill dimensions and locations###
        min_thickness = 900 #m
        max_thickness = 3500 #m

        mar = 7
        sar = 2.5

        min_emplacement = 1000 #m
        max_emplacement = 15500 #m
        n_sills = 20000

        tot_volume = int(0.03e6*1e9)
        flux = int(30e9)

        thermal_mat_time = int(3e4)
        model_time = tot_volume/flux
        cooling_time = int(1e4)

        phase_times = [thermal_mat_time, model_time, cooling_time]
        time_steps = np.arange(0, np.sum(phase_times), dt)
        print(f'Length of time_steps:{len(time_steps)}')

        sillcube, n_sills, emplacement_params = self.build_sillcube(x, y, dx, dy, dt, [min_thickness, max_thickness, 500], [mar, sar], [min_emplacement, max_emplacement, 5000], [x//3, 2*x//3, x//6], phase_times, tot_volume, flux, n_sills)
        print('sillcube built')
        params = self.get_silli_initial_thermogenic_state(props_array, dx, dy, dt, 'conv smooth', k, time = thermal_mat_time)
        current_time = params[0]
        print(f'Current time before function: {current_time}')
        carbon_model_params = params[1:]
        print('Got initial emissions state')
        props_total_array, carbon_model_params = self.emplace_sills(props_array, k, dx, dy, dt, n_sills, b//2, 'conv smooth', time_steps, current_time, sillcube, carbon_model_params, emplacement_params, model = 'silli')
        print('Model Run complete')
        tot_RCO2 = carbon_model_params[0]
        plt.plot(time_steps, tot_RCO2)
        plt.savefig('plots/CarbonEmisisons.png', format = 'png')
