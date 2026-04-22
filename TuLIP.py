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
from scipy.linalg import solve_banded
import pandas as pd
import pyvista as pv
import os
import h5py
import warnings
from autograd import elementwise_grad
import autograd.numpy as anp
import pdb
import utilities as util
import re
try:
    import networkx as nx
except ImportError:
    nx = None
    print("Warning: networkx not found")



"""
TuLIP: Thermally understanding Large Igneous Provinces
==============================

TuLIP is a Python library for simulating heat diffusion and thermal maturation in geological settings.
It is designed to model the emplacement of sills (magma intrusions) and their thermal effects on surrounding rocks,
including the generation of carbon emissions from organic-rich host rocks.

This file contains the core physics solvers and utility classes for the simulation:
- `cool`: Handles heat diffusion solvers (Implicit, ADI, Explicit).
- `emit`: Handles chemical kinetics and emission calculations (SILLi, sillburp).
- `rules`: Handles geometric rules for sill emplacement and mesh manipulation.
- `sill_controls`: Integration class that orchestrates the simulation.

Authors: [Aristle Monteiro, Tushar Mittal]
Optimized by: Google DeepMind Agent
"""


class cool:
    """
    The `cool` class contains all the numerical solvers for heat diffusion.
    
    It supports multiple solving methods:
    1.  **ADI (Alternating Direction Implicit)**: Optimized for speed and stability (`conv_smooth_solve_adi`).
    2.  **Implicit Direct**: Solves the full matrix system using `pypardiso` or `scipy` (`straight_solver`).
    3.  **Iterative**: Jacobian (`JacobianIt`) and Gauss-Seidell (`GSIt`) solvers (slower, for educational use).
    4.  **Explicit**: Convolution-based explicit solvers (`conv_chain_solve`, `conv_smooth_solve`).
    
    It also handles the construction of the linear system (weight matrices) for the implicit solvers.
    """
    def __init__(self):
        pass
    ###Functions to cool magma bodies###

    @staticmethod
    def is_nan(q):
        """
        Checks if an array contains any NaN (Not a Number) values.
        
        This is a JIT-friendly alternative to `np.isnan().any()` used for Numba compatibility.
        
        Parameters
        ----------
        q : array-like
            The input array to check.
            
        Returns
        -------
        bool
            True if any element is NaN, False otherwise.
        """
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
    def avg_perm(k):
        """
        Calculates averaged permeability (thermal diffusivity) at grid cell boundaries.
        
        In the Finite Volume Method, properties are defined at cell centers, but fluxes occur
        across cell faces. We need the diffusivity at the face (harmonic or arithmetic mean).
        Here, we compute the arithmetic mean of neighbors.
        
        Parameters
        ----------
        k : numpy.ndarray
            The 2D diffusivity field (MxN array) at cell centers.
            
        Returns
        -------
        kiph : numpy.ndarray
            Diffusivity at the "i plus half" face (between Row i and Row i+1).
        kimh : numpy.ndarray
            Diffusivity at the "i minus half" face (between Row i and Row i-1).
        kjph : numpy.ndarray
            Diffusivity at the "j plus half" face (between Column j and Column j+1).
        kjmh : numpy.ndarray
            Diffusivity at the "j minus half" face (between Column j and Column j-1).
        """
        kiph = k.copy()
        kimh = k.copy()
        kjph = k.copy()
        kjmh = k.copy()

        # Averaging permeability in i (Row) and j (Column) directions.
        # k[i, :] refers to Row i.
        
        # kiph[i] = (k[i] + k[i+1]) / 2  -> Valid for i=0 to a-2
        kiph[:-1, :] = (k[:-1, :] + k[1:, :]) / 2
        
        # kjph[j] = (k[j] + k[j+1]) / 2  -> Valid for j=0 to b-2
        kjph[:, :-1] = (k[:, :-1] + k[:, 1:]) / 2
        
        # kimh[i] = (k[i] + k[i-1]) / 2  -> Valid for i=1 to a-1 (Shifted view)
        kimh[1:, :] = (k[1:, :] + k[:-1, :]) / 2
        
        # kjmh[j] = (k[j] + k[j-1]) / 2  -> Valid for j=1 to b-1 (Shifted view)
        kjmh[:, 1:] = (k[:, 1:] + k[:, :-1]) / 2
        
        return kiph, kimh, kjph, kjmh

    def cheat_solver(self, Tf, a, q):
        """
        Approximates the initial thermal gradient quickly without running a full simulation.
        Warning: This is a rough approximation for testing and should be replaced by a proper equilibrium solve.
        
        Parameters
        ----------
        Tf : numpy.ndarray
            Temperature field to initialize.
        a : int
            Number of rows (depth).
        q : float or nan
            Heat flux at the bottom. If NaN, a fixed gradient based on T_top and T_bot is used.
            
        Returns
        -------
        Tf : numpy.ndarray
            The initialized temperature field.
        """
        if self.is_nan(np.array([q])): # Wrap in array for is_nan compatibility
            # Dirichlet-Dirichlet case: Linear interpolation from Top to Bottom
            T_top = Tf[0,0]
            T_bot = Tf[-1,0]
            grad = (T_bot - T_top)/a 
            for i in range(0,a):
                Tf[i,:] = T_top + i*grad
        else:
            # Neumann case: Fixed flux q
            for i in range(1, a):
                Tf[i,:] = Tf[i-1,:]+(q/31.532) # Note: 31.532 assumes specific conductivity hardcoded?
        return Tf

    def straight_solver(self, Ab, dee, a, b):
        """
        Solves the linear system Ax = b directly for obtaining the temperature field.
        This uses `pypardiso` (parallel sparse solver) if available, or `scipy.sparse.linalg` otherwise.
        
        Parameters
        ----------
        Ab : scipy.sparse.csc_matrix
            The Coefficient Matrix (A) of shape (M*N, M*N). 
            This represents the discretized heat equations for all nodes.
        dee : numpy.ndarray
            The Right-Hand Side Vector (b) of shape (M*N).
            This contains the knowns (old temperature, source terms, boundary conditions).
        a : int
            Number of rows in the 2D grid.
        b : int
            Number of columns in the 2D grid.
            
        Returns
        -------
        Tf : numpy.ndarray
            The solved temperature field reshaped back to (a, b).
        """
        # Solve Ax = b
        solution_flat = ps.spsolve(sc.csc_matrix(Ab), dee)
        Tf = np.array(solution_flat)
        # Reshape the flat solution vector back to 2D grid (Fortran order column-major)
        Tf = Tf.reshape((a,b), order = 'F')
        return Tf
    ###Alternative functions to get initial thermal state. it is important to note that the straight solver will always be the fastest since there is only one matrix multiplication to perform###
    def JacobianIt(self, Ab, dee, a, b):
        """
        Solves the linear system using the Jacobian Iterative Method.
        
        A x = b
        (D + R) x = b
        x_new = D^-1 (b - R * x_old)
        
        Parameters
        ----------
        Ab : scipy.sparse.csc_matrix
            LHS Coefficient Matrix.
        dee : numpy.ndarray
            RHS Vector.
        a, b : int
            Grid dimensions.
            
        Returns
        -------
        T : numpy.ndarray
            Solved temperature field.
        """
        # Extract Diagonal
        d = Ab.diagonal()
        
        # Calculate Inverse Diagonal (Vectorized)
        # Avoid division by zero if any (though PDE matrix usually well-behaved)
        with np.errstate(divide='ignore'):
            d_inv = 1.0 / d
            d_inv[np.isinf(d_inv)] = 0.0 # handle zero diagonals if any
        
        # R = L + U (Strictly Lower + Upper) = A - D
        # Efficiently: Ab - Diag
        R = Ab - sc.diags(d, 0)
        
        # Precompute D^-1 * b
        constant_term = d_inv * dee
        
        # Initial Guess (Use RHS)
        T = dee.copy()
        
        iter_max = 10000
        tol = 1e-3
        
        # Iteration Loop
        for c in range(iter_max):
            # x_new = D^-1 * (b - R*x)
            #       = D^-1*b - D^-1*(R*x)
            #       = constant_term - d_inv * (R.dot(T))
            
            # Compute R*T (Sparse Mat-Vec mult)
            RT = R.dot(T)
            
            # Update
            T_new = constant_term - (d_inv * RT)
            
            # Error Check (Infinity Norm)
            err = np.max(np.abs(T_new - T))
            
            if err < tol:
                print(f'Jacobian Converged: {c} iterations, Error: {err:.2e}')
                return T_new.reshape((a,b), order = 'F')
            
            T = T_new

        print(f'Jacobian Max Iterations ({iter_max}) Reached. Final Error: {err:.2e}')
        return T.reshape((a,b), order = 'F')

    def GSIt(self, Ab, dee, a, b):
        """
        Solves the linear system using the Gauss-Seidel Iterative Method.
        
        Splitting: A = (D + L) + U
        Iteration: (D + L) x_new = b - U x_old
        x_new = (D + L)^-1 * (b - U x_old)
        
        Parameters
        ----------
        Ab : scipy.sparse.csc_matrix
            LHS Coefficient Matrix.
        dee : numpy.ndarray
            RHS Vector.
        a, b : int
            Grid dimensions.
            
        Returns
        -------
        T : numpy.ndarray
            Solved temperature field.
        """
        # Decomposition
        # D = Diag, E = -Lower (Strict), F = -Upper (Strict)
        # Original code definitions:
        # A = D - E - F
        # D = Diag
        # E = - Lower
        # F = - Upper
        # A = D + Lower + Upper
        #
        # GS Split: (D + Lower) x_new = b - Upper x_old
        # M = D + Lower
        # N = - Upper
        
        # Consistent Extraction
        D = sc.diags(Ab.diagonal(), 0)
        Lower = sc.tril(Ab, k=-1)
        Upper = sc.triu(Ab, k=1)
        
        # M = D + L (Lower Triangular)
        M = D + Lower
        M = sc.csc_matrix(M)
        
        # N = - U (RHS term)
        # We need term -U * x. So N = -Upper.
        N = -Upper
        
        # Pre-solve constant part: M^-1 * b
        # Using pypardiso (ps) as requested.
        try:
            constant_term = ps.spsolve(M, dee)
        except Exception as e:
            # Fallback if ps fails (though redundant given imports, good for safety)
            print(f"Pypardiso failed: {e}. Using scipy.")
            import scipy.sparse.linalg as spla
            constant_term = spla.spsolve(M, dee)

        T = dee.copy()
        iter_max = 10000
        tol = 1e-3
        
        for c in range(iter_max):
            # x_new = M^-1 b  + M^-1 (-U x_old)
            #       = constant_term + M^-1 (N * x)
            
            # 1. N * x (Sparse Mat-Vec)
            rhs_part = N.dot(T)
            
            # 2. M^-1 * (N*x)
            try:
                term_update = ps.spsolve(M, rhs_part)
            except:
                import scipy.sparse.linalg as spla
                term_update = spla.spsolve(M, rhs_part)
            
            T_new = constant_term + term_update
            
            err = np.max(np.abs(T_new - T))
            if err < tol:
                print(f'GS Converged: {c} iterations, Error: {err:.2e}')
                return T_new.reshape((a,b), order = 'F')
            
            T = T_new

        print(f'GS Max Iterations ({iter_max}) Reached. Final Error: {err:.2e}')
        return T.reshape((a,b), order = 'F')




    def heat_flux(self, k, a, b, dx, dy, Tnow, method, q = np.nan):
        """
        Switch function to solve the heat diffusion equation using the selected method (Implicit).
        
        This function sets up the Linear System Ax = b for the implicit time step.
        It handles boundary conditions (Dirichlet at top, Neumann at sides/bottom).
        
        **Important Coordinate System Note:**
        In this implementation:
        - Axis 0 (Rows, size `a`) corresponds to the `dy` spatial step (Y-axis/Depth).
        - Axis 1 (Columns, size `b`) corresponds to the `dx` spatial step (X-axis/Width).
        
        Parameters
        ----------
        k : numpy.ndarray
            Thermal diffusivity field (MxN).
        a : int
            Number of rows (Axis 0).
        b : int
            Number of columns (Axis 1).
        dx : float
            Grid spacing along Axis 1 (Columns).
        dy : float
            Grid spacing along Axis 0 (Rows).
        Tnow : numpy.ndarray
            Current temperature field. Dirichlet BCs should already be set.
        method : str
            Solver method: 'straight', 'Jacobian', 'GS', or 'cheat'.
        q : float or nan
            Heat flux at bottom boundary. 
            - If nan: Dirichlet condition (fixed temp).
            - If float: Neumann condition (fixed flux).
            
        Returns
        -------
        T_new : numpy.ndarray
            Solved temperature field for the next time step.
        """
        if method == 'cheat':
                return self.cheat_solver(Tnow,a, q)
        else:
            if ~np.isnan(q).any():
                Tnow[-1,:] = q*dy/k[-1,:]
            bee = Tnow.reshape((a*b), order = 'F')
            
            # Vectorized Matrix Construction
            main_diag = np.zeros((a, b))
            p1_diag = np.zeros((a, b))
            m1_diag = np.zeros((a, b))
            pa_diag = np.zeros((a, b))
            ma_diag = np.zeros((a, b))
            
            # Precompute terms
            # Interior indices: 1 to a-1 (rows), 1 to b-1 (cols)
            # Use views for concise code
            k_ipv = k[2:, 1:-1]   # i+1 (Row Neighbor)
            k_imv = k[:-2, 1:-1]  # i-1 (Row Neighbor)
            k_jpv = k[1:-1, 2:]   # j+1 (Col Neighbor)
            k_jmv = k[1:-1, :-2]  # j-1 (Col Neighbor)
            k_cv = k[1:-1, 1:-1]  # center
            
            # Terms
            # Row Gradients (Axis 0) -> dy
            term_row = (k_ipv - k_imv) / (4 * dy**2)
            base_row = k_cv / dy**2

            # Col Gradients (Axis 1) -> dx
            term_col = (k_jpv - k_jmv) / (4 * dx**2)
            base_col = k_cv / dx**2
            
            # Fill Diagonals (Interior)
            main_diag[1:-1, 1:-1] = -(2 * base_row + 2 * base_col)
            
            # p1/m1: Row Neighbors (Axis 0) -> Use Row terms/dy
            p1_diag[1:-1, 1:-1]   = base_row + term_row
            m1_diag[1:-1, 1:-1]   = base_row - term_row
            
            # pa/ma: Col Neighbors (Axis 1) -> Use Col terms/dx
            pa_diag[1:-1, 1:-1]   = base_col + term_col
            ma_diag[1:-1, 1:-1]   = base_col - term_col
            
            # Boundaries
            # i=0 (Top) - Dirichlet
            main_diag[0, :] = 1
            
            # j=0 (Left), i!=0, i!=a-1 - Neumann
            main_diag[1:-1, 0] = -1
            pa_diag[1:-1, 0] = 1
            
            # j=b-1 (Right), i!=0, i!=a-1 - Neumann
            main_diag[1:-1, -1] = 1
            ma_diag[1:-1, -1] = -1
            
            # i=a-1 (Bottom)
            if np.isnan(q).any():
                main_diag[-1, :] = 1
            else:
                main_diag[-1, :] = 1
                m1_diag[-1, :] = -1
                
            # Flatten in Fortran order
            md = main_diag.flatten(order='F')
            p1 = p1_diag.flatten(order='F')[:-1] # Remove last element (invalid for superdiagonal)
            m1 = m1_diag.flatten(order='F')[1:]  # Remove first element (invalid for subdiagonal)
            pa = pa_diag.flatten(order='F')[:-a]
            ma = ma_diag.flatten(order='F')[a:]
            
            Af = sc.lil_matrix((a*b,a*b))
            print('Weight matrix:', Af.shape)
            Af.setdiag(md, k=0)
            Af.setdiag(p1, k=1)
            Af.setdiag(m1, k=-1)
            Af.setdiag(pa, k=a)
            Af.setdiag(ma, k=-a)

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
        Explicit solver for anisotropic, time-varying diffusivity using Averaged Permeability.
        
        This solver builds (or reuses) the coefficient matrix `Af` based on the arithmetic mean 
        of diffusivity at cell interfaces. It is more accurate for heterogeneous media.
        
        The equation solved is:
        (1/H_lat) * dT/dt = Div(k * Grad(T)) + H_rad
        
        Discretized as:
        (1 - Term_x - Term_y + H_rad)*T_new = T_old

        Parameters
        ----------
        k : numpy.ndarray
            Diffusivity field.
        a, b : int
            Grid dimensions (Rows, Cols).
        dx, dy : float
            Grid spacing (dx=Rows, dy=Cols).
        dt : float
            Time step.
        Tnow : numpy.ndarray
            Current temperature field (RHS vector source).
        q : float or nan
            Bottom boundary heat flux.
        Af : scipy.sparse.lil_matrix or nan
            Pre-computed weight matrix. If nan, it is built.
        H : tuple (H_rad, H_lat)
            H_rad: Radiogenic heat production term.
            H_lat: Latent heat scaling factor (Effective Heat Capacity/Latent Heat ratio).
            
        Returns
        -------
        Tret : numpy.ndarray
            New temperature field.
        """
        H_rad = H[0]
        H_lat = H[1] #This is the latent heat solution that should be divided and not the actual latent heat of crystalization. See get_latH for details
        
        # Determine if Af is actually checking for NaN (it might be a matrix object)
        build_matrix = False
        if isinstance(Af, (float, int)) and np.isnan(Af):
            build_matrix = True
        elif isinstance(Af, np.ndarray) and np.isnan(Af).any():
             # Array of nans?
            build_matrix = True
            
        if build_matrix:
            #Generate the weight matrix if the thermal diffusivity is not constant and needs to be changed at every time step
            kiph, kimh, kjph, kjmh = self.avg_perm(k)
            
            # Vectorized Matrix Build
            main_diag = np.ones((a, b))
            p1_diag = np.zeros((a, b)) # p1 is superdiag (k=1)
            m1_diag = np.zeros((a, b)) # m1 is subdiag (k=-1)
            pa_diag = np.zeros((a, b)) # pa is k=a
            ma_diag = np.zeros((a, b)) # ma is k=-a
            
            # Constants
            cx = dt / (dx**2)
            cy = dt / (dy**2)
            
            # Interior views
            kiph_in = kiph[1:-1, 1:-1]
            kimh_in = kimh[1:-1, 1:-1]
            kjph_in = kjph[1:-1, 1:-1]
            kjmh_in = kjmh[1:-1, 1:-1]
            H_rad_in = H_rad[1:-1, 1:-1]
            H_lat_in = H_lat[1:-1, 1:-1]
            
            # Term: Sum of outgoing fluxes
            # Row neighbors (kiph, kimh) -> Y-direction -> Cy
            # Col neighbors (kjph, kjmh) -> X-direction -> Cx
            term = (kiph_in + kimh_in) * cy + (kjph_in + kjmh_in) * cx
            main_diag[1:-1, 1:-1] = 1 - (term / H_lat_in)
            
            # Off Diags for Fortran Flattening (Column-Major):
            # +/- 1 indices correspond to same Column, adjacent Row -> Y-axis neighbors (Depth) -> Uses cy
            # +/- a indices correspond to same Row, adjacent Column -> X-axis neighbors (Width) -> Uses cx
            
            # pa (k=a): Right neighbor in Matrix (Column + 1) -> X-axis neighbors -> Cx
            pa_diag[1:-1, 1:-1] = (kjph_in * cx) / H_lat_in
            
            # ma (k=-a): Left neighbor in Matrix (Column - 1) -> X-axis neighbors -> Cx
            ma_diag[1:-1, 1:-1] = (kjmh_in * cx) / H_lat_in
            
            # p1 (k=1): Bottom neighbor (Row + 1) -> Y-axis neighbors (Depth) -> Cy
            p1_diag[1:-1, 1:-1] = (kiph_in * cy) / H_lat_in
            
            # m1 (k=-1): Top neighbor (Row - 1) -> Y-axis neighbors (Depth) -> Cy
            m1_diag[1:-1, 1:-1] = (kimh_in * cy) / H_lat_in
            
            # Flatten
            md = main_diag.flatten(order='F')
            p1 = p1_diag.flatten(order='F')[:-1]
            m1 = m1_diag.flatten(order='F')[1:]
            pa = pa_diag.flatten(order='F')[:-a]
            ma = ma_diag.flatten(order='F')[a:]
            
            Af = sc.lil_matrix((a*b,a*b))
            Af.setdiag(md, k=0)
            Af.setdiag(p1, k=1)
            Af.setdiag(m1, k=-1)
            Af.setdiag(pa, k=a)
            Af.setdiag(ma, k=-a)

        bee = Tnow.reshape((a*b), order = 'F')
        Tret = np.array(Af.dot(bee)) #Perform the matrix multiplication
        Tret = Tret.reshape((a,b), order = 'F')
        Tret[1:-1, 1:-1] += (H_rad[1:-1, 1:-1] * dt) / H_lat[1:-1, 1:-1]
        Tret[:,0] = Tret[:,2]
        Tret[:,-1] = Tret[:,-3]
        if ~np.isnan(q).any():
            Tret[-1,:] = Tret[-2,:]+ (q*dy/k[-1,:])
        return Tret

    def perm_chain_solve(self, k, a, b, dx, dy, dt, Tnow, q, Af, H):
        """
        Implicit solver using the Chain Rule expansion of the diffusion equation.
        
        Expands Div(k * Grad(T)) into:
        k * Laplacian(T) + Grad(k) * Grad(T)
        
        This form accounts for gradients in diffusivity (`dk/dx`, `dk/dy`) explicitly.
        
        Parameters
        ----------
        k : numpy.ndarray
            Diffusivity field.
        a, b : int
            Grid dimensions.
        dx, dy : float
            Spacing.
        dt : float
            Time step.
        Tnow : numpy.ndarray
            Current temp.
        q : float
            Bottom flux.
        Af : matrix or nan
            Weight matrix.
        H : tuple
            (H_rad, H_lat).
            
        Returns
        -------
        Tret : numpy.ndarray
            New temp field.
        """
        H_rad = H[0]
        H_lat = H[1] #This is the latent heat solution that should be divided and not the actual latent heat of crystalization. See get_latH for details
        # Initialize matrix build flag

        build_matrix = False
        if isinstance(Af, (float, int)) and np.isnan(Af):
            build_matrix = True
        elif isinstance(Af, np.ndarray) and np.isnan(Af).any():
            build_matrix = True
            
        if build_matrix:
            #Generate the weight matrix if the thermal diffusivity is not constant and needs to be changed at every time step
            main_diag = np.ones((a, b))
            p1_diag = np.zeros((a, b))
            m1_diag = np.zeros((a, b))
            pa_diag = np.zeros((a, b))
            ma_diag = np.zeros((a, b))

            cx = dt / (dx**2)
            cy = dt / (dy**2)
            
            # Views
            k_in = k[1:-1, 1:-1]
            k_ip1 = k[2:, 1:-1]
            k_im1 = k[:-2, 1:-1]
            k_jp1 = k[1:-1, 2:]
            k_jm1 = k[1:-1, :-2]
            H_rad_in = H_rad[1:-1, 1:-1]
            H_lat_in = H_lat[1:-1, 1:-1]
            
            # Terms
            # main = (1 - ( 2*k*cx + 2*k*cy) + H_rad ) / H_lat
            # Note: k is already k/H_lat.
            # Terms
            # main = (1 - ( 2*k*cy + 2*k*cx) + H_rad ) / H_lat
            # Note: k is already k/H_lat.
            # Row (Y) -> cy. Col (X) -> cx.
            term = 2*k_in*cy + 2*k_in*cx
            main_diag[1:-1, 1:-1] = 1 - (term / H_lat_in)
            
            # Gradient terms
            # k_row_grad (Axis 0 - Y) -> Uses dy
            k_ip1 = k[2:, 1:-1]
            k_im1 = k[:-2, 1:-1]
            k_row_grad = (k_ip1 - k_im1) / (4 * dy**2)
            
            # k_col_grad (Axis 1 - X) -> Uses dx
            k_jp1 = k[1:-1, 2:]
            k_jm1 = k[1:-1, :-2]
            k_col_grad = (k_jp1 - k_jm1) / (4 * dx**2)
            
            # Diagonals
            # p1/m1: Row Neighbors (Axis 0, Y) -> Use cy
            p1_diag[1:-1, 1:-1] = ((k_in/(dy**2) + k_row_grad) * dt) / H_lat_in
            m1_diag[1:-1, 1:-1] = ((k_in/(dy**2) - k_row_grad) * dt) / H_lat_in
            
            # pa/ma: Col Neighbors (Axis 1, X) -> Use cx
            pa_diag[1:-1, 1:-1] = ((k_in/(dx**2) + k_col_grad) * dt) / H_lat_in
            ma_diag[1:-1, 1:-1] = ((k_in/(dx**2) - k_col_grad) * dt) / H_lat_in
            
            # Flatten
            md = main_diag.flatten(order='F')
            p1 = p1_diag.flatten(order='F')[:-1]
            m1 = m1_diag.flatten(order='F')[1:]
            pa = pa_diag.flatten(order='F')[:-a]
            ma = ma_diag.flatten(order='F')[a:]

            Af = sc.lil_matrix((a*b,a*b))
            Af.setdiag(md, k=0)
            Af.setdiag(p1, k=1)
            Af.setdiag(m1, k=-1)
            Af.setdiag(pa, k=a)
            Af.setdiag(ma, k=-a)
            
        bee = Tnow.reshape((a*b), order = 'F')
        Tret = np.array(Af.dot(bee)) #Perform matrix multiplication
        Tret = Tret.reshape((a,b), order = 'F')
        Tret[1:-1, 1:-1] += (H_rad[1:-1, 1:-1] * dt) / H_lat[1:-1, 1:-1]
        Tret[:,0] = Tret[:,2]
        Tret[:,-1] = Tret[:,-3]
        if ~np.isnan(q).any():
            Tret[-1,:] = Tret[-2,:]+ (q*dy/k[-1,:])
        return Tret

    @jit(forceobj = True)
    def conv_chain_solve(self, k, a, b, dx, dy, dt, Tf, H, q = np.nan):
        """
        Explicit solver using Chain Rule logic (Convolution/Stencil-based).
        
        Optimized for JIT compilation (currently forceobj=True).
        Solves: Div(k Grad T) = k*Laplacian(T) + Grad(k)*Grad(T) + Source
        
        Parameters
        ----------
        k : numpy.ndarray
            Diffusivity field.
        a, b : int
            Grid dimensions.
        dx, dy : float
            Spacing.
        dt : float
            Time step.
        Tf : numpy.ndarray
            Current temperature field.
        H : tuple
            (H_rad, H_lat).
        q : float or nan
            Bottom flux.
            
        Returns
        -------
        Tnow : numpy.ndarray
            Updated temperature.
        """
        H_rad = H[0]
        H_lat = H[1]
        
        # Precompute constants
        cx = dt / (dx**2)
        cy = dt / (dy**2)
        
        # Slices
        T_cent = Tf[1:-1, 1:-1]
        T_ip1 = Tf[2:, 1:-1]
        T_im1 = Tf[:-2, 1:-1]
        T_jp1 = Tf[1:-1, 2:]
        T_jm1 = Tf[1:-1, :-2]
        
        k_cent = k[1:-1, 1:-1]
        H_rad_in = H_rad[1:-1, 1:-1]
        
        H_lat_in = H_lat[1:-1, 1:-1]
        k_cent_eff = k_cent / H_lat_in

        # Gradients
        # Row Gradient (dK/dy) -> Axis 0 -> Y
        dk_dy = (k[2:, 1:-1] - k[:-2, 1:-1]) / (4 * dy**2) * dt / H_lat_in
        
        # Column Gradient (dK/dx) -> Axis 1 -> X
        dk_dx = (k[1:-1, 2:] - k[1:-1, :-2]) / (4 * dx**2) * dt / H_lat_in
        
        # Coefficients
        # Main term: 1 - 2*k*cy - 2*k*cx
        
        term_main = 1.0 - (2 * k_cent_eff * cy + 2 * k_cent_eff * cx)
        
        # Neighbors
        # Row (i+/-1, Y): (k/dy2 +/- dk/dy ) * dt
        c_ip1 = (k_cent_eff * cy) + dk_dy
        c_im1 = (k_cent_eff * cy) - dk_dy
        
        # Col (j+/-1, X): (k/dx2 +/- dk/dx ) * dt
        c_jp1 = (k_cent_eff * cx) + dk_dx
        c_jm1 = (k_cent_eff * cx) - dk_dx
        
        # Update Interior
        Tnow = np.copy(Tf)
        Tnow[1:-1, 1:-1] = (
            term_main * T_cent +
            c_ip1 * T_ip1 +
            c_im1 * T_im1 +
            c_jp1 * T_jp1 +
            c_jm1 * T_jm1 +
            (H_rad_in * dt) / H_lat_in
        )
        
        # Boundaries
        T_surf = Tf[0,0]
        T_bot = Tf[-1,0]
        
        # Left/Right (Columns 0 and -1) -> Copy neighbor
        Tnow[1:-1, 0] = Tnow[1:-1, 2]
        Tnow[1:-1, -1] = Tnow[1:-1, -3]
        
        # Top (Row 0)
        Tnow[0, :] = T_surf
        
        # Bottom (Row -1)
        if (np.isnan(np.array(q)).any()):
            Tnow[-1, :] = T_bot
        else:
            # Gradient condition
            # Original: Tnow[-2,:] + (q * dy / k[-1,:])
            Tnow[-1, :] = Tnow[-2, :] + (q * dy / k[-1, :])
            
        return Tnow

    @jit(forceobj = True)
    def conv_smooth_solve(self, k, a, b, dx, dy, dt, Tf, H, q = np.nan):
        """
        Explicit solver using Averaged Permeability logic (Convolution/Stencil-based).
        
        Uses arithmetic mean of diffusivity at cell faces. Less sensitive to K-gradients than chain rule.
        Solves: (T_new - T_old)/dt = (1/H_lat) * [ Div(k_avg * Grad(T)) + H_rad ]
        
        Parameters
        ----------
        k : numpy.ndarray
            Diffusivity field.
        a, b : int
            Grid dimensions.
        dx, dy : float
            Spacing.
        dt : float
            Time step.
        Tf : numpy.ndarray
            Current temp.
        H : tuple
            (H_rad, H_lat).
        q : float or nan
            Bottom flux.
            
        Returns
        -------
        Tnow : numpy.ndarray
            Updated temperature.
        """
        H_rad = H[0]
        H_lat = H[1]
        # --- 1. GET DIFFUSIVITIES ---
        kiph, kimh, kjph, kjmh = self.avg_perm(k)
        
        Tnow = np.array(Tf)

        # Pre-calculate constants
        Cx = dt / (dx**2)
        Cy = dt / (dy**2)
        # Slicing the arrays for the interior block
        T_C = Tf[1:-1, 1:-1]  # T(i, j)
        T_N = Tf[:-2, 1:-1]   # T(i-1, j)
        T_S = Tf[2:, 1:-1]    # T(i+1, j)
        T_W = Tf[1:-1, :-2]   # T(i, j-1)
        T_E = Tf[1:-1, 2:]    # T(i, j+1)

        # Slicing the k-arrays
        kiph_s = kiph[1:-1, 1:-1]
        kimh_s = kimh[1:-1, 1:-1]
        kjph_s = kjph[1:-1, 1:-1]
        kjmh_s = kjmh[1:-1, 1:-1]

        Tnow[1:-1, 1:-1] = T_C + (
        -(kiph_s + kimh_s) * Cy * T_C - (kjph_s + kjmh_s) * Cx * T_C
        + (kiph_s * Cy) * T_S
        + (kimh_s * Cy) * T_N
        + (kjph_s * Cx) * T_E
        + (kjmh_s * Cx) * T_W
        + (H_rad[1:-1, 1:-1] * dt)
        ) / H_lat[1:-1, 1:-1]
    
        # --- Apply Boundary Conditions (also vectorized) ---
        T_surf = Tf[0,0]
        T_bot = Tf[-1,0]
        
        # Left/Right Neumann
        Tnow[1:-1, 0] = Tnow[1:-1, 2]
        Tnow[1:-1, -1] = Tnow[1:-1, -3]

        # Top Dirichlet
        Tnow[0, :] = T_surf
        
        if (np.array(np.isnan(np.array(q))).any()):
            Tnow[-1,:] = T_bot
        else:
            Tnow[-1,:] = Tnow[-2,:]+ (q*dy/k[-1,:])
        return Tnow
    
    def conv_smooth_solve_adi(self, k, a, b, dx, dy, dt, Tf, H, q=np.nan):
        """
        Solves the Heat Diffusion Equation using the Alternating Direction Implicit (ADI) method.
        Specifically, the Peaceman-Rachford ADI scheme.
        
        Stability: Unconditionally stable (though accuracy degrades with large dt).
        Complexity: O(N) due to tridiagonal matrix solves.
        
        The method splits the time step `dt` into two halves `dt/2`:
        
        **Step 1:** Implicit in Axis 0 (Rows/Y), Explicit in Axis 1 (Cols/X).
        Solves for intermediate `T*`.
        System: (I - Ay/2) * T* = (I + Ax/2) * Tn + Source/2
        
        **Step 2:** Implicit in Axis 1 (Cols/X), Explicit in Axis 0 (Rows/Y).
        Solves for final `T_new`.
        System: (I - Ax/2) * T_new = (I + Ay/2) * T* + Source/2
        
        **Axis Mapping:**
        - Axis 0 (Rows i): Corresponds to `dy` spacing. Diffusivities `kiph`, `kimh`.
        - Axis 1 (Cols j): Corresponds to `dx` spacing. Diffusivities `kjph`, `kjmh`.
        
        Parameters
        ----------
        k : numpy.ndarray
            Diffusivity field.
        a : int
            Number of rows (Axis 0).
        b : int
            Number of columns (Axis 1).
        dx : float
            Spacing along Axis 1 (Cols).
        dy : float
            Spacing along Axis 0 (Rows).
        dt : float
            Time step.
        Tf : numpy.ndarray
            Current temperature field.
        H : tuple
            (H_rad, H_lat).
        q : float or nan
            Heat flux at bottom (Neumann condition).
        
        Returns
        -------
        Tnow : numpy.ndarray
            Updated temperature field.
        """
        H_rad = np.array(H[0], dtype = float)
        H_lat = np.array(H[1], dtype = float)
        
        # --- 1. GET DIFFUSIVITIES ---
        kiph, kimh, kjph, kjmh = self.avg_perm(k)
        kiph = kiph / H_lat
        kimh = kimh / H_lat
        kjph = kjph / H_lat
        kjmh = kjmh / H_lat
        
        # Normalize Source Term (consistent with Explicit solver)
        # Explicit solver: Main term divided by H_lat. H_rad*dt also divided by H_lat.
        # So Q_source should be H_rad * dt / H_lat
        Q_source = (H_rad * dt) / H_lat
        
        T_star = np.array(Tf) 
        Tnow = np.zeros_like(Tf)

        dt2 = dt / 2.0
        Cx2 = dt2 / (dx**2)
        Cy2 = dt2 / (dy**2)
        
        T_surf = Tf[0, 0]
        T_bot = Tf[-1, 0]

        # ===============================================================
        # STEP 1: Implicit in Y (Rows - Axis 0), Explicit in X (Cols - Axis 1)
        # ===============================================================
        
        # --- RHS Explicit X ---
        # (I + Ax * dt/2) * Tn
        # We process INTERIOR columns j=1 to b-2
        RHS_step1_base = np.zeros_like(Tf)
        
        # Storage for the ACTUAL RHS used (including boundary additions)
        RHS_step1_used = np.zeros_like(Tf)
        
        # Interior Nodes (j=1 to b-2, i=1 to a-2)
        # Uses kjph/kjmh (X-diffusivity? Wait. avg_perm docs: kjph is Col boundary (X). kiph is Row boundary (Y).)
        # RHS Explicit should clearly use X-Diffusivity (kjph, kjmh) and Cx.
        # Check avg_perm usage:
        # kiph/kimh -> Row Faces (i+1/2). Correct for Implicit Row loop.
        # kjph/kjmh -> Col Faces (j+1/2). Correct for Explicit Col neighbors.
        
        RHS_step1_base[1:-1, 1:-1] = (
            (1.0 - (kjph[1:-1, 1:-1] + kjmh[1:-1, 1:-1]) * Cx2) * Tf[1:-1, 1:-1]
            + (kjph[1:-1, 1:-1] * Cx2) * Tf[1:-1, 2:]  # T[i, j+1]
            + (kjmh[1:-1, 1:-1] * Cx2) * Tf[1:-1, :-2] # T[i, j-1]
            + Q_source[1:-1, 1:-1] / 2.0 
        )
        
        # RHS for Bottom Row (if Neumann)
        # i = a-1.
        if not np.isnan(np.array(q)).any():
            # Bottom row indices [ -1, 1:-1 ]
            # Neighbors: T[-1, 2:] (East), T[-1, :-2] (West) -> X neighbors
            
            row_idx = -1
            RHS_step1_base[row_idx, 1:-1] = (
                (1.0 - (kjph[row_idx, 1:-1] + kjmh[row_idx, 1:-1]) * Cx2) * Tf[row_idx, 1:-1]
                + (kjph[row_idx, 1:-1] * Cx2) * Tf[row_idx, 2:]  # T[i, j+1]
                + (kjmh[row_idx, 1:-1] * Cx2) * Tf[row_idx, :-2] # T[i, j-1]
                + Q_source[row_idx, 1:-1] / 2.0 
            )

        # --- LHS Implicit Y (Rows) ---
        # Solve for T_star[:, j] for j in 1..b-2
        
        for j in range(1, b - 1): # Loop INTERIOR columns only
            num_unknowns = a - 2 # Rows 1 to a-2
            if not np.isnan(np.array(q)).any():
                num_unknowns = a - 1 # Rows 1 to a-1 (Bottom is variable)

            if num_unknowns <= 0: continue
            
            kiph_c = kiph[:, j]
            kimh_c = kimh[:, j]
            
            # Tridiagonal Matrix construction
            ab = np.zeros((3, num_unknowns))
            
            # Indices relative to unknowns [0..N-1] mapping to rows [1..N]
            
            # Main Diagonal: 1 + (kiph + kimh) * Cy2  (Y-Implicit uses Cy)
            ab[1, :] = 1.0 + (kiph_c[1:num_unknowns+1] + kimh_c[1:num_unknowns+1]) * Cy2
            
            # Upper Diagonal (k=1, connects i to i+1): -kiph * Cy2
            ab[0, 1:] = -kiph_c[1:num_unknowns] * Cy2
            
            # Lower Diagonal (k=-1, connects i to i-1): -kimh * Cy2
            ab[2, :-1] = -kimh_c[2:num_unknowns+1] * Cy2
            
            # --- RHS Logic ---
            rhs_vec = RHS_step1_base[1:num_unknowns+1, j].copy()
            
            # --- Boundaries (X-direction) ---
            # Top (i=0): Dirichlet T_surf
            # Term was -kimh[1] * T[0]. Move to RHS.
            rhs_vec[0] += (kimh_c[1] * Cy2) * T_surf
            
            # Bottom (i=a-1 or i=a):
            if np.isnan(np.array(q)).any():
                # Dirichlet T_bot at i=a-1
                rhs_vec[-1] += (kiph_c[num_unknowns] * Cy2) * T_bot
            else:
                # Neumann (Flux q) at i=a-1
                # Modify Matrix for last node (index -1 in system, i=a-1 in grid)
                # T[a] (ghost) = T[a-2] + Flux
                # See derivation in previous step:
                # Coeff of T[a-2] (which is T[system_last-1]) gets added -kiph term.
                # ab[2, -2] corresponds to coeff of T[system_last-1] in equ for T[system_last].
                # Wait, ab[2, -1]? No ab[2] is lower diag. 
                # Last element of lower diag is ab[2, -1] ?? No.
                # solve_banded `ab` shape (3, N).
                # ab[2] is lower. entries ab[2, 1]...ab[2, N-1].
                # ab[2, j] is A[j, j-1].
                # The last row `N-1`. entry A[N-1, N-2]. This is ab[2, N-1]. (using column index of A)
                # Wait, scipy docs: "ab[u + i - j, j] == a[i,j]"
                # Lower diag: i = j+1. index = 1 + j+1 - j = 2. Correct.
                # We want A[N-1, N-2]. j=N-2.
                # So ab[2, N-2].
                # Yes, ab[2, -2] in python slice (second to last element of array).
                
                # Substitute T[a] = T[a-2] + Flux
                # T[a-2] term change. Coeff of T[a-2] is Lower Diagonal.
                # ab[2, -2] corresponds to lower diagonal element of last row.
                ab[2, -2] -= kiph_c[num_unknowns] * Cy2 
                
                flux_val = q[j] * dy / k[a-1, j] # Approx. Flux q -> Q = -k * dT/dy
                rhs_vec[-1] += (kiph_c[num_unknowns] * Cy2) * flux_val

            # SAVE the final rhs_vec for Step 2 stability trick
            RHS_step1_used[1:num_unknowns+1, j] = rhs_vec

            # Solve
            T_star[1:num_unknowns+1, j] = solve_banded((1, 1), ab, rhs_vec)

        # Apply Y-Boundaries to T_star (Slave Condition)
        T_star[:, 0] = T_star[:, 2]
        T_star[:, -1] = T_star[:, -3]
        
        # Apply X-Boundaries to T_star (Explicit/Dirichlet)
        T_star[0, :] = T_surf
        if np.isnan(np.array(q)).any():
            T_star[-1, :] = T_bot
        else:
             T_star[-1, :] = T_star[-2, :] + (q * dy / k[-1, :])

        # ===============================================================
        # STEP 2: Implicit in X (Cols - Axis 1), Explicit in Y (Rows - Axis 0)
        # ===============================================================
        
        # --- RHS Explicit Y (Rows) ---
        # (I + Ay * dt/2) * T*
        
        RHS_step2_base = np.zeros_like(T_star)

        # Interior Nodes (j=1 to b-2, i=1 to a-2)
        # Explicitly calculating Y-direction fluxes (Rows) -> Uses Cy2
        # kiph: i+1/2 (Bottom face). kimh: i-1/2 (Top face).
        
        RHS_step2_base[1:-1, 1:-1] = (
            (1.0 - (kiph[1:-1, 1:-1] + kimh[1:-1, 1:-1]) * Cy2) * T_star[1:-1, 1:-1]
            + (kiph[1:-1, 1:-1] * Cy2) * T_star[2:, 1:-1]  # T[i+1, j] (Row neighbor)
            + (kimh[1:-1, 1:-1] * Cy2) * T_star[:-2, 1:-1] # T[i-1, j] (Row neighbor)
            + Q_source[1:-1, 1:-1] / 2.0 
        )
        
        # --- LHS Implicit X (Cols) ---
        # Solve for T_now[i, :] for i in 1..a-2
        # Implicitly solving X-direction (Cols) -> Uses Cx2
        
        for i in range(1, a - 1): # Loop ROWS
            num_unknowns = b - 2 # Cols 1 to b-2
            if num_unknowns <= 0: continue
            
            # Row slice of diffusivities (X-direction limits)
            kjph_r = kjph[i, :]
            kjmh_r = kjmh[i, :]
            
            # Tridiagonal Construction
            ab = np.zeros((3, num_unknowns))
            
            # Main Diag: 1 + (kjph + kjmh) * Cx2
            ab[1, :] = 1.0 + (kjph_r[1:b-1] + kjmh_r[1:b-1]) * Cx2
            
            # Upper Diag (k=1, j to j+1): -kjph * Cx2
            ab[0, 1:] = -kjph_r[1:b-2] * Cx2
            
            # Lower Diag (k=-1, j to j-1): -kjmh * Cx2
            ab[2, :-1] = -kjmh_r[2:b-1] * Cx2
            
            # RHS
            rhs_vec = RHS_step2_base[i, 1:b-1].copy()
            
            # Boundaries (Y-direction / Cols)
            # Left (j=0): Ghost T[i, 0] = T[i, 2] (Reflecting)
            # Equ for j=1:
            # Coeffs: -kjmh[1]*T[0] + Main*T[1] - kjph[1]*T[2] = RHS
            # T[0] = T[2].
            # -kjmh[1]*T[2] + Main*T[1] - kjph[1]*T[2]
            # Combine T[2] coeffs: -(kjph[1] + kjmh[1])*Cx2.
            # Upper Diag term for T[2] (rel index 1) corresponds to j=1 eqn.
   
            ab[0, 1] -= kjmh_r[1] * Cx2
            
            # Right (j=b-1): Ghost T[i, b-1] = T[i, b-3] (Reflecting)
            # Last unknown is index num-1 (j=b-2).
            # Equ for j=b-2:
            # -kjmh*T[b-3] + Main*T[b-2] - kjph*T[b-1] = RHS
            # T[b-1] = T[b-3].
            # Combine T[b-3]. Coeff of T[b-3] is Lower Diag.
            # ab[2, -2] (second to last of array, which is last valid lower diag).
            
            ab[2, -2] -= kjph_r[b-2] * Cx2
            
            # Solve
            Tnow[i, 1:b-1] = solve_banded((1, 1), ab, rhs_vec)

        # Apply Y-Boundaries to Tnow
        Tnow[:, 0] = Tnow[:, 2]
        Tnow[:, -1] = Tnow[:, -3]
        
        # Apply X-Boundaries
        Tnow[0, :] = T_surf
        if np.isnan(np.array(q)).any():
             Tnow[-1, :] = T_bot
        else:
             Tnow[-1, :] = Tnow[-2, :] + (q * dy / k[-1, :])
             
        return Tnow
    
    @staticmethod
    def func_assigner(func, *args, **kwargs):
        """
        Dynamically calls a function with provided arguments.
        
        This wrapper allows for flexible function calls, useful when the specific
        function to be executed is determined at runtime (e.g., specific phase behavior).
        
        Parameters
        ----------
        func : callable
            Function to execute.
        *args
            Positional arguments.
        **kwargs
            Keyword arguments.
            
        Returns
        -------
        result : Any
            Return value of the function.
        """
        result = func(*args,**kwargs)
        return result
    
    @staticmethod
    def calcF(T_field, T_solidus, T_liquidus):
        """
        Calculates melt fraction (F) based on temperature using a smoothed step function.
        
        Uses a tanh-based smoothing to transition between Solid (F=0) and Liquid (F=1).
        This smooth transition is crucial for numerical stability when taking derivatives (latent heat).
        
        Parameters
        ----------
        T_field : numpy.ndarray
            Temperature field.
        T_solidus : float
            Solidus temperature (F starts increasing). Default usually 800.
        T_liquidus : float
            Liquidus temperature (F reaches 1). Default usually 1250.
            
        Returns
        -------
        F : numpy.ndarray
            Melt fraction field (0 to 1).
        """
        def safe_tanh(x):
            """Clip inputs to tanh to avoid overflow in cosh during gradients."""
            x_clipped = anp.clip(x, -50, 50)  # Prevents overflow in cosh
            return anp.tanh(x_clipped)

        def smooth_step(x, lower_bound, upper_bound, steepness=40):
            """Smoothly transitions between 0 and 1 using tanh."""
            return 0.5 * (safe_tanh((x - lower_bound) / (upper_bound - lower_bound + 1e-8) * steepness) + 1)
        # Avoid division-by-zero by ensuring T_liquidus > T_solidus
        
        T_liquidus = anp.maximum(T_liquidus, T_solidus + 1e-6)
        delta = max(0.05 * (T_liquidus - T_solidus), 1e-6)  # Minimum delta to avoid collapse
        delta_liquidus = max(0.1 * (T_liquidus - T_solidus), 1e-6)

        # Smooth masks for transitions around T_solidus and T_liquidus
        mask_solidus = smooth_step(T_field, T_solidus - delta, T_solidus + delta)
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
        """
        Calculates melt fraction (F) by interpolating valid phase diagram data from a CSV.
        
        Parameters
        ----------
        T_field : numpy.ndarray
            Temperature field.
        dir_csv : str
            Path to the CSV file containing phase data.
        temp_col : str
            Column name for temperature.
        fraction_column : str
            Column name for melt fraction.
        T_liquidus : float
            Liquidus temp (used for smoothing bounds).
        T_solidus : float
            Solidus temp (used for smoothing bounds).
            
        Returns
        -------
        F : numpy.ndarray
            Melt fraction.
        """
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
        """
        Calculates the Effective Specific Heat term to account for Latent Heat of Crystallization.
        
        The heat equation with phase change is:
        (rho*Cp + rho*L*dF/dT) * dT/dt = ...
        
        We define H_lat = (Cp + L*dF/dT) / Cp = 1 + (L/Cp)*dF/dT.
        Then solving for T corresponds to diffusion with a modified heat capacity.
        
        Parameters
        ----------
        T_field : numpy.ndarray
            Temperature field.
        lithology : numpy.ndarray
            Lithology map (unused in default implementation, but kept for interface).
        melt : str
            Melt type (e.g. 'basalt'). Unused in current logic.
        specific_heat : float
            Cp (J/kg/K).
        L : float
            Latent heat of fusion (J/kg).
        T_liquidus, T_solidus : float
            Phase change bounds.
        curve_func : callable
            Function F(T) to calculate melt fraction.
        args : tuple
            Arguments for curve_func.
            
        Returns
        -------
        H_lat : numpy.ndarray
            Scaling factor for the heat equation LHS.
        """
        H_lat = np.ones_like(T_field)
        if args is None:
            args = (T_solidus, T_liquidus)
        if curve_func is None:
            curve_func = cool.calcF
        phi_cr = elementwise_grad(curve_func)
        phi_vals = phi_cr(T_field[lithology==melt], *args)
        if np.isnan(phi_vals).any():
           raise ValueError('Invalid value (NaN) encountered in melt fraction function. You should either change the melt fraction function or the liquidus temperature to solve this issue...')
        H_lat[lithology==melt] = (1 + (phi_vals*L/specific_heat))
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
        Stand-in funciton to get conductivity based on properties
        '''
        conductivity = 1/(0.24094+(4.6019e-4*(T_field+273.15)))
        return conductivity
    
    @staticmethod
    def get_specific_heat(T_field, rock, density, dy):
        '''
        Function to get specfic heat as a function of temperature for specific rock types.
        '''
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
        elif method =='adi':
            Tnow = np.array(Tnow, dtype = 'float')
            Tnow = self.conv_smooth_solve_adi(k, a, b, dx, dy, dt, Tnow, H, q)
            return Tnow
        elif method=='smooth':
            if k_const:
                res = cool.avg_perm(k)
                kiph, kimh, kjph, kjmh = [x/H[1] for x in res]
                
                # Vectorized Matrix Build
                main_diag = np.ones((a, b))
                p1_diag = np.zeros((a, b))
                m1_diag = np.zeros((a, b))
                pa_diag = np.zeros((a, b))
                ma_diag = np.zeros((a, b))
                
                cx = dt / (dx**2)
                cy = dt / (dy**2)
                
                # Interior views
                kiph_in = kiph[1:-1, 1:-1]
                kimh_in = kimh[1:-1, 1:-1]
                kjph_in = kjph[1:-1, 1:-1]
                kjmh_in = kjmh[1:-1, 1:-1]
                
                # Main
                term = (kiph_in + kimh_in) * cx + (kjph_in + kjmh_in) * cy
                main_diag[1:-1, 1:-1] = -term + 1
                
                # Off Diags
                # p1/m1 (Row Neighbors, k=1) use kiph/kimh (Row Avg) and cx (dx)
                p1_diag[1:-1, 1:-1] = kiph_in * cx
                m1_diag[1:-1, 1:-1] = kimh_in * cx
                
                # pa/ma (Col Neighbors, k=a) use kjph/kjmh (Col Avg) and cy (dy)
                pa_diag[1:-1, 1:-1] = kjph_in * cy
                ma_diag[1:-1, 1:-1] = kjmh_in * cy
                
                # Flatten
                md = main_diag.flatten(order='F')
                p1 = p1_diag.flatten(order='F')[:-1]
                m1 = m1_diag.flatten(order='F')[1:]
                pa = pa_diag.flatten(order='F')[:-a]
                ma = ma_diag.flatten(order='F')[a:]

                Af = sc.lil_matrix((a*b,a*b))
                Af.setdiag(md, k=0)
                Af.setdiag(p1, k=1)
                Af.setdiag(m1, k=-1)
                Af.setdiag(pa, k=a)
                Af.setdiag(ma, k=-a)
            else:
                Af = np.nan

            return self.perm_smoothed_solve(k/H[1], a, b, dx, dy, dt, Tnow, q, Af, H)
        elif method=='chain':
            if k_const:
                k_scaled = k/H[1] # Local copy
                
                # Vectorized Matrix Build
                main_diag = np.ones((a, b))
                p1_diag = np.zeros((a, b))
                m1_diag = np.zeros((a, b))
                pa_diag = np.zeros((a, b))
                ma_diag = np.zeros((a, b))

                cx = dt / (dx**2)
                cy = dt / (dy**2)
                
                # Views
                k_in = k_scaled[1:-1, 1:-1]
                k_ip1 = k_scaled[2:, 1:-1]
                k_im1 = k_scaled[:-2, 1:-1]
                # k_jp1 = k_scaled[1:-1, 2:]
                # k_jm1 = k_scaled[1:-1, :-2]
                
                # Terms
                term = 2*k_in*cx + 2*k_in*cy
                main_diag[1:-1, 1:-1] = -term + 1
                
                # Gradients
                # k_row_grad (Axis 0)
                k_ip1 = k_scaled[2:, 1:-1]
                k_im1 = k_scaled[:-2, 1:-1]
                k_row_grad = (k_ip1 - k_im1) / (4 * dx**2)
                
                # k_col_grad (Axis 1)
                k_jp1 = k_scaled[1:-1, 2:]
                k_jm1 = k_scaled[1:-1, :-2]
                k_col_grad = (k_jp1 - k_jm1) / (4 * dy**2)
                
                # Diagonals
                # p1/m1: Row Neighbors (Axis 0, dx)
                p1_diag[1:-1, 1:-1] = (k_in/(dx**2) + k_row_grad) * dt
                m1_diag[1:-1, 1:-1] = (k_in/(dx**2) - k_row_grad) * dt
                
                # pa/ma: Col Neighbors (Axis 1, dy)
                pa_diag[1:-1, 1:-1] = (k_in/(dy**2) + k_col_grad) * dt
                ma_diag[1:-1, 1:-1] = (k_in/(dy**2) - k_col_grad) * dt

                # Flatten
                md = main_diag.flatten(order='F')
                p1 = p1_diag.flatten(order='F')[:-1]
                m1 = m1_diag.flatten(order='F')[1:]
                pa = pa_diag.flatten(order='F')[:-a]
                ma = ma_diag.flatten(order='F')[a:]

                Af = sc.lil_matrix((a*b,a*b))
                Af.setdiag(md, k=0)
                Af.setdiag(p1, k=1)
                Af.setdiag(m1, k=-1)
                Af.setdiag(pa, k=a)
                Af.setdiag(ma, k=-a)
            else:
                Af = np.nan
            return self.perm_chain_solve(k/H[1], a, b, dx, dy, dt, Tnow, q, Af, H)
        else:
            raise ValueError('diff_solve solution method not supported')

@jit(nopython=True, cache=True)
def _sillburp_core(T_field, progress_of_reactions, rate_of_reactions,
                   reaction_energies, no_reactions, As, dt, n_reactions,
                   oil_production_rate_accum):
    """
    JIT-compiled core logic for the Sillburp organic matter degradation model.
    calculate the degradation of organic matter and production of gases/oil.
    
    This function handles the kinetics for multiple reaction channels (Labile, Refractory, Inert, Oil).
    It updates the progress of reactions (P) and reaction rates (R) in place.
    
    Parameters
    ----------
    T_field : numpy.ndarray
        Temperature field (degree C).
    progress_of_reactions : numpy.ndarray
        Accumulated fraction of reaction completed (4, N_approx, a, b).
    rate_of_reactions : numpy.ndarray
        Current rate of reaction (4, N_approx, a, b).
    reaction_energies : numpy.ndarray
        Activation energies for each approximated reaction channel.
    no_reactions : array-like
        Number of approximations for each kerogen type.
    As : array-like
        Arrhenius pre-exponential factors.
    dt : float
        Time step.
    n_reactions : int
        Number of reaction types (usually 4).
    oil_production_rate_accum : numpy.ndarray
        Accumulator for oil production (modified in place).
        
    Returns
    -------
    progress_of_reactions, rate_of_reactions, oil_production_rate_accum
    """
    
    a, b = T_field.shape
    
    # Constants
    reactants_LABILE = 0
    reactants_REFRACTORY = 1
    reactants_VITRINITE = 2
    reactants_OIL = 3
    
    mass_frac_labile_to_gas = 0.2
    
    # Temperature setup
    T_K = T_field + 273.15
    RT = 8.314 * T_K
    
    # Iterate reactions
    for i_reac in range(n_reactions):
        n_approx = no_reactions[i_reac]
        A_val = As[i_reac]
        
        # Slices
        # reaction_energies: (n_reactions, max_approx)
        E_slice = reaction_energies[i_reac, :n_approx]
        
        # Calculate reaction rates for this group: (n_approx, a, b)
        reaction_rates = np.empty((n_approx, a, b), dtype=np.float64)
        for k in range(n_approx):
            reaction_rates[k] = A_val * np.exp(-E_slice[k] / RT)
            
        exp_rate_dt = np.exp(-reaction_rates * dt)
        
        # P_old view (read-only for calculations)
        P_old_slice = progress_of_reactions[i_reac, :n_approx]
        
        # Mask
        do_mask = P_old_slice < 1.0
        
        if i_reac != reactants_OIL:
            # Standard Kinetics
            for k in range(n_approx):
                for ia in range(a):
                    for ib in range(b):
                        if do_mask[k, ia, ib]:
                            p_old = P_old_slice[k, ia, ib]
                            exp_term = exp_rate_dt[k, ia, ib]
                            
                            val = 1.0 - (1.0 - p_old) * exp_term
                            if val > 1.0: val = 1.0
                            P_new = val
                            
                            # Update Progress
                            progress_of_reactions[i_reac, k, ia, ib] = P_new
                            
                            # Update Rate
                            R_new = (1.0 - p_old) * (1.0 - exp_term) / dt
                            rate_of_reactions[i_reac, k, ia, ib] = R_new
                            
                            # Labile -> Oil Accumulation
                            if i_reac == reactants_LABILE:
                                rate = reaction_rates[k, ia, ib]
                                contrib = -rate * (1.0 - P_new) / n_approx
                                oil_production_rate_accum[ia, ib] += contrib
            
            # Post-reaction adjustment for Labile
            if i_reac == reactants_LABILE:
                oil_production_rate_accum *= (1.0 - mass_frac_labile_to_gas)

        else:
            # OIL Kinetics (depends on accumulated oil)
            for k in range(n_approx):
                for ia in range(a):
                    for ib in range(b):
                        # S_over_k = oil_accum / (rate * n_approx)
                        rate = reaction_rates[k, ia, ib]
                        denom = rate * n_approx
                        
                        s_k = 0.0
                        if denom != 0.0:
                            s_k = oil_production_rate_accum[ia, ib] / denom
                        
                        if do_mask[k, ia, ib]:
                            p_old = P_old_slice[k, ia, ib]
                            exp_term = exp_rate_dt[k, ia, ib]
                            
                            # P_new = 1 - S/k - (1 - P_old - S/k) * exp
                            term_inner = 1.0 - p_old - s_k
                            val = 1.0 - s_k - term_inner * exp_term
                            
                            if val > 1.0: val = 1.0
                            P_new = val
                            
                            # Update Progress
                            progress_of_reactions[i_reac, k, ia, ib] = P_new
                            
                            # Update Rate
                            R_new = (1.0 - p_old - s_k) * (1.0 - exp_term) / dt
                            rate_of_reactions[i_reac, k, ia, ib] = R_new

    return progress_of_reactions, rate_of_reactions, oil_production_rate_accum

@jit(nopython=True, cache=True)
def _SILLi_core(T_field, W, calc_parser, dt, E, f, A, R):
    """
    JIT-compiled core logic for the SILLi (EasyRo) vitrinite reflectance model.
    
    This function computes the 'Easy%Ro' type maturation by iterating through 
    parallel reaction channels with different activation energies.
    
    Parameters
    ----------
    T_field : numpy.ndarray
        Temperature field (degree C).
    W : numpy.ndarray
        Array keeping track of remaining reactant fraction for each channel (N_E, a, b).
    calc_parser : numpy.ndarray
        Boolean mask of where to calculate (shale/sandstone).
    dt : float
        Time step (seconds).
    E : numpy.ndarray
        Activation energies (J/mol).
    f : numpy.ndarray
        Stoichiometric factors weighting each channel.
    A : float
        Frequency factor (1/s).
    R : float
        Gas constant (J/mol/K).
        
    Returns
    -------
    Frac : numpy.ndarray
        Total fraction of conversion (Sum of f_i * (1-W_i)).
    W : numpy.ndarray
        Updated W array.
    """
    a, b = T_field.shape
    n_E = len(E)
    
    # Pre-calculate constants
    T_K = T_field + 273.15
    RT = R * T_K
    
    Frac = np.zeros((a, b), dtype=np.float64)
    
    # To avoid allocating (n_E, a, b) for fl, we accumulate Frac directly
    # Frac = sum(fl) = sum(f[l] * (1 - W[l]))
    
    for l in range(n_E):
        val_E = E[l]
        val_f = f[l]
        
        # Calculate k
        # k = A * exp(-E/RT)
        # Using explicit loops to avoid temporary array allocation if possible,
        # but vectorized math within the block is fine for Numba
        
        # Vectorized operations (Numba handles these efficiently)
        k_slice = A * np.exp(-val_E / RT)
        
        # Update W in place
        # W_new = max(W_old * exp(-k*dt), 0)
        # We can read/write W[l]
        w_slice = W[l]
        exp_kdt = np.exp(-k_slice * dt)
        
        for i in range(a):
            for j in range(b):
                val_w = w_slice[i, j] * exp_kdt[i, j]
                if val_w < 0.0: val_w = 0.0
                w_slice[i, j] = val_w
                
                # fl contribution
                # fl = f * (1 - W)
                fl_val = val_f * (1.0 - val_w)
                Frac[i, j] += fl_val
                
    return Frac, W

class emit:
    """
    The `emit` class handles chemical kinetics and carbon emission models.
    
    It includes methods for:
    1.  **SILLi**: Simulating thermal maturation and vitrinite reflectance (Easy%Ro).
    2.  **Sillburp**: Simulating kinetic breakdown of organic matter types (Labile, Refractory, etc.).
    3.  **Carbonate Breakdown**: Calculating CO2 release from decarbonation reactions.
    4.  **Initial CO2**: estimating initial dissolved CO2 in equilibrium.
    """
    def __init__(self):
        pass
    
    @staticmethod
    def get_init_CO2_percentages(T_field, lithology, density, dy):
        """
        Calculates the initial exsolved CO2 percentages for carbonate rocks in equilibrium.
        
        Uses phase diagrams (interpolated from loaded .mat files) to determine the equilibrium 
        CO2 content based on Temperature and Pressure (derived from density/depth).
        
        Parameters
        ----------
        T_field : numpy.ndarray
            Temperature field (degree C).
        lithology : numpy.ndarray
            Lithology map (strings).
        density : numpy.ndarray
            Density field (kg/m3) for lithostatic pressure calculation.
        dy : float
            Layer thickness (m).
            
        Returns
        -------
        init_CO2 : numpy.ndarray
            Equilibrium CO2 percentage field.
        """
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
        """
        Calculates the instantaneous CO2 release rate from decarbonation reactions.
        
        It compares current equilibrium CO2 capacity (based on P-T) with the previous tracked capacity.
        If the rock can hold less CO2 than before, the difference is released.
        
        Parameters
        ----------
        T_field : numpy.ndarray
            Temperature field.
        lithology : numpy.ndarray
            Lithology map.
        density : numpy.ndarray
            Density field.
        breakdownCO2 : numpy.ndarray
            Previous cumulative internal CO2 capacity (or state).
        dy : float
            Depth increment.
        dt : float
            Time step.
            
        Returns
        -------
        RCO2_breakdown : numpy.ndarray
            Rate of CO2 release (kg/m3/s or similar units depending on calibration).
        max_breakdown_co2 : numpy.ndarray
            Updated cumulative breakdown state to be passed to next step.
        """
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
    def SILLi_emissions(T_field, density, lithology, porosity, TOC_prev, dt, TOCo=np.nan, W=np.nan):
        """
        Calculates thermal maturation and carbon emissions using the Easy%Ro model (Sweeney & Burnham, 1990).
        
        This implementation is JIT-optimized ("SILLi_core") to handle large grids efficiently.
        It models the degradation of Vitrinite (proxy for all organic matter) via 
        parallel first-order Arrhenius reactions.
        
        Parameters
        ----------
        T_field : numpy.ndarray
            Temperature field (degree C).
        density : numpy.ndarray
            Rock density (kg/m3).
        lithology : numpy.ndarray
            Lithology map.
        porosity : numpy.ndarray
            Porosity map (fraction).
        TOC_prev : numpy.ndarray
            Total Organic Carbon from previous step (fraction).
        dt : float
            Time step (seconds).
        TOCo : numpy.ndarray
            Initial Total Organic Carbon (at t=0). Required after first step.
        W : numpy.ndarray
            State variable tracking remaining potential for each reaction channel. 
            Shape: (N_reactions, a, b).
            
        Returns
        -------
        RCO2 : numpy.ndarray
            Diffusive CO2 flux (kg/m3/s converted to percent/s scale? Depends on usage).
        Rom : numpy.ndarray
             mass rate of organic matter loss (kg/m3/s).
        percRo : numpy.ndarray
            Calculated Vitrinite Reflectance (%Ro).
        TOC : numpy.ndarray
            Updated Total Organic Carbon.
        W : numpy.ndarray
            Updated state variable.
        """
        calc_parser = (lithology=='shale') | (lithology=='sandstone')
        break_parser = (lithology=='dolostone') | (lithology=='limestone') | (lithology=='marl') | (lithology=='evaporite')
        calc_parser = calc_parser | break_parser
        a,b = T_field.shape
            
        A = np.float64(1e13)
        R = np.float64(8.314) #J/K/mol
        E = np.array([34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72], dtype = float)*4184 #J/mole
        f = np.array([0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06, 0.04, 0.04, 0.07, 0.06, 0.06, 0.06, 0.05, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01], dtype = float)
        T_field = T_field.astype(np.float64)
        
        # Handle Initialization
        if np.isnan(W).all():
            W = np.ones((len(E),a, b), dtype=np.float64)
            TOCo = TOC_prev
            
        if np.isnan(np.array(TOCo, dtype = float)).any():
            raise ValueError('TOCo cannot be NaN after the first time step')
        
        # Ensure types for JIT
        dt = float(dt)
        
        # Call JIT Core
        Frac, W = _SILLi_core(T_field, W, calc_parser, dt, E, f, A, R)
        
        # Post-process results (Vectorized)
        percRo = np.exp(-1.6+3.7*Frac) #vitrinite reflectance
        TOC = TOCo*(1-Frac)*calc_parser
        dTOC = (TOC_prev-TOC)/dt
        Rom = (1-porosity)*density*dTOC
        RCO2 = Rom*3.67/100
        return RCO2, Rom, percRo, TOC, W

    def analytical_Ro(T_field, dT, density, lithology, porosity, I_prev, TOC_prev, dt, TOCo, W):
        """
        Analytical solution for the Easy%Ro model.
        
        Note: This function is currently marked as 'Untested' and serves as a reference 
        or alternative implementation to the numerical integration in `SILLi_emissions`.
        
        Parameters
        ----------
        (Similar to SILLi_emissions, with 'I_prev' as the integration state variable)
        """
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
        """
        Initializes the integral state variable 'I' for the analytical Easy%Ro solution.
        """

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
        """
        Generates the Activation Energy distributions for the Sillburp model.
        
        Based on Jones et al. (2019). Creates normal distributions of activation energies
        for Labile, Refractory, and Inert Kerogen components.
        
        Returns
        -------
        reaction_energies : numpy.ndarray
             Array of activation energies for each approximated reaction channel.
        """
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
    def sillburp(T_field, TOC_prev, density, lithology, porosity, dt, reaction_energies, TOCo=None, oil_production_rate=0, progress_of_reactions=np.nan, rate_of_reactions = np.nan, weights = None):
        """
        Calculates thermogenic carbon emissions using the Sillburp model (Jones et al., 2019).
        
        This model is more detailed than Easy%Ro, accounting for different kerogen types 
        (Labile, Refractory, Inert) and the intermediate generation of oil before cracking to gas.
        
        This implementation uses a JIT-compiled core (`_sillburp_core`) for performance.
        
        Parameters
        ----------
        T_field : numpy.ndarray
            Temperature field.
        TOC_prev : numpy.ndarray
            Total Organic Carbon from previous step.
        density, lithology, porosity: numpy.ndarray
            Rock properties.
        dt : float
            Time step.
        reaction_energies : numpy.ndarray
            Pre-calculated activation energies (from `get_sillburp_reaction_energies`).
        TOCo : numpy.ndarray
            Initial TOC (required after first step).
        progress_of_reactions : numpy.ndarray
            State variable: Fraction of reaction completed for each channel.
        rate_of_reactions : numpy.ndarray
            State variable: Current reaction rates.
        weights : numpy.ndarray or None
            Optional weights for averaging reaction products.
            
        Returns
        -------
        RCO2 : numpy.ndarray
             CO2 generation rate.
        Rom : numpy.ndarray
             Organic Matter loss rate.
        progress_of_reactions : numpy.ndarray
             Updated state.
        oil_production_rate : numpy.ndarray
             Rate of oil generation/cracking.
        TOC : numpy.ndarray
             Updated TOC.
        rate_of_reactions : numpy.ndarray
             Updated rates.
        """
        if TOCo is None:
            TOCo = TOC_prev
        
        a, b = T_field.shape
        calc_parser = (lithology == 'shale') | (lithology == 'sandstone')
        n_reactions = 4
        
        # Constants for JIT
        no_reactions = np.array([7, 21, 55, 7], dtype=np.int64)
        As = np.array([1.58e13, 1.83e18, 4e10, 1e13], dtype=np.float64)
        
        # Initialize arrays if needed
        if np.isnan(progress_of_reactions).all():
            progress_of_reactions = np.zeros((n_reactions, max(no_reactions), a, b), dtype=np.float64)
            rate_of_reactions = np.zeros_like(progress_of_reactions)
        
        # Ensure correct types for JIT
        T_field = T_field.astype(np.float64)
        reaction_energies = reaction_energies.astype(np.float64)
        dt = float(dt)
        
        # Oil production accumulator
        oil_production_rate_accum = np.zeros_like(T_field)
        
        # Call JIT core
        progress_of_reactions, rate_of_reactions, oil_production_rate_accum = _sillburp_core(
            T_field, progress_of_reactions, rate_of_reactions, 
            reaction_energies, no_reactions, As, dt, n_reactions, 
            oil_production_rate_accum
        )
        
        # Final Outputs
        oil_production_rate = oil_production_rate_accum
        products_progress = np.zeros((n_reactions, a, b))
        
        if weights is None:
            for i_reaction in range(0,n_reactions):
                products_progress[i_reaction,:, :] = np.mean(progress_of_reactions[i_reaction,0:no_reactions[i_reaction],:,:], axis  = 0)
            products_progress = np.mean(products_progress, axis=0)
            
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
            RCO2 = Rom * 3.67/100
        
        return RCO2, Rom, progress_of_reactions, oil_production_rate, TOC, rate_of_reactions


class rules:
    """
    The `rules` class contains heuristic rules and geometric utilities for the simulation.
    
    It handles:
    1.  **Sill Emplacement**: Deciding when and where to emplace sills.
    2.  **Geometry**: Generating random heights, widths, and positions for sills based on specified distributions and parameters.
    3.  **Mesh Manipulation**: Pushing strata down to accommodate new sills.
    4.  **Properties**: Managing lithology and property dictionaries.
    """
    def __init__(self):
        pass
    @staticmethod
    def to_emplace(t_now, t_thresh):
        """
        Determines if a new sill should be emplaced based on time threshold.
        
        Parameters
        ----------
        t_now : float
            Time elapsed since last event. 
        t_thresh : float
            Time threshold triggering emplacement.
            
        Returns
        -------
        bool
            True if t_now >= t_thresh i.e., if enough time has lapsed since last event.
        """
        if (t_now<t_thresh):
            return False
        elif t_now>=t_thresh:
            return True

    @staticmethod
    def build_lith_dict(lithology):
        """
        Builds a dictionary mapping integer IDs to lithology names.
        
        This allows optimizing storage/calculation by handling integers in the core logic 
        while maintaining string labels for reference.
        
        Parameters
        ----------
        lithology : numpy.ndarray
            2D array of lithology strings (or objects).
            
        Returns
        -------
        lith_dict : dict
            Mapping {int_id: 'rock_name'}.
        """
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
        """
        Builds a dictionary mapping lithology names to constant properties (like density/porosity).
        Assumes property is constant for a given lithology type in the input arrays.
        
        Parameters
        ----------
        prop : numpy.ndarray
            Property field.
        lithology : numpy.ndarray
            Lithology field.
            
        Returns
        -------
        prop_dict : dict
            Mapping {'rock_name': property_value}.
        """
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
        Emplaces a single rectangular sill at a specified location.
        
        Parameters
        ----------
        T_field : numpy.ndarray
            Temperature field (modified in place).
        x_space : int
            Center Column index.
        height : int
            Center Row index.
        width : int
            Width in nodes.
        thick : int
            Thickness in nodes.
        T_mag : float
            Magma temperature.
            
        Returns
        -------
        T_field : numpy.ndarray
            Updated temperature field.
        """
        T_field[int(height-(thick//2)):int(height+(thick//2)), int(x_space-(width//2)):int(x_space+(width//2))] = T_mag
        return T_field
    @staticmethod
    def circle_sill(T_field, x_space, height, r, T_mag, a, b, dx, dy):
        """
        Emplaces a circular sill.
        
        Parameters
        ----------
        T_field : numpy.ndarray
             Modified in place.
        x_space, height : int
             Center indices (col, row).
        r : float
             Radius in meters.
        T_mag : float
             Magma temp.
        a, b : int
             Grid dimensions.
        dx, dy : float
             Grid spacing (m).
             
        Returns
        -------
        T_field
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
        Generates random emplacement depths (heights) from a Normal Distribution.
        
        The distribution is centered between `l_sill` (low/deep) and `h_sill` (high/shallow).
        Values outside the [h_sill, l_sill] range are resampled recursively.
        
        Parameters
        ----------
        n_sills : int
            Number of sills.
        l_sill : float
            Lower bound (Deepest depth in m). Note: In geological plotting dependent on y-axis, 'lower' often means deeper (higher y-index).
        h_sill : float
             Upper bound (Shallowest depth in m).
        sd : float
             Standard deviation of the distribution (m).
        dy : float
             Grid spacing (m).
             
        Returns
        -------
        heights : numpy.ndarray
             Array of Row indices (nodes) for sill centers.
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
        Generates random horizontal positions (Column indices) from a Normal Distribution.
        
        Centered between x_min and x_max.
        
        Parameters
        ----------
        n_sills : int
             Number of sills.
        x_min, x_max : float
             Range bounds (m).
        sd : float
             Standard deviation (m).
        dx : float
             Grid spacing (m).
             
        Returns
        -------
        space : numpy.ndarray
             Column indices.
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
        Generates random emplacement depths from a Uniform Distribution.
        
        Parameters
        ----------
        n_sills : int
            Number of sills.
        l_sill, h_sill : float
            Depth bounds (m).
        dy : float
            Grid spacing (m).
            
        Returns
        -------
        heights : numpy.ndarray
            Row indices.
        """
        heights = np.round(np.random.uniform(l_sill, h_sill, n_sills)/dy)
        return heights

    @staticmethod
    def uniform_x(n_sills, x_min, x_max, dx):
        """
        Generates random horizontal positions from a Uniform Distribution.
        
        Parameters
        ----------
        n_sills : int
            Number of sills.
        x_min, x_max : float
            Range bounds (m).
        dx : float
            Grid spacing (m).
            
        Returns
        -------
        space : numpy.ndarray
            Column indices.
        """
        space = np.round(np.random.uniform(x_min, x_max, n_sills)/dx)
        return space
    @staticmethod
    def empirical_CDF(n_sills, xarray, cdf):
        """
        Samples random values from a user-provided Empirical Cumulative Distribution Function (CDF).
        
        Parameters
        ----------
        n_sills : int
            Number of samples needed.
        xarray : numpy.ndarray
            Domain values (x-axis of CDF).
        cdf : numpy.ndarray
            Cumulative probability values (y-axis of CDF, 0 to 1).
            
        Returns
        -------
        why : numpy.ndarray
            Sampled values.
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
        Generates sill dimensions (width, height) scaling linearly with depth (heights).
        
        Assumes deeper sills might be larger/smaller based on the scaling logic.
        Also adds random noise to aspect ratio and dimensions.
        
        Parameters
        ----------
        min_min, min_max : float
            Minimum/Maximum thickness bounds (m).
        mar : float
            Mean Aspect Ratio (Width/Thickness).
        sar : float
            Standard Deviation of Aspect Ratio.
        heights : numpy.ndarray
            Emplacement depths (nodes). Used to scale dimensions.
        n_sills : int
            Number of sills.
            
        Returns
        -------
        major : numpy.ndarray
            Widths (major axis) in length units (m).
        minor : numpy.ndarray
            Thicknesses (minor axis) in length units (m).
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
        Generates sill dimensions using a Normal Distribution.
        
        Parameters
        ----------
        min_min, min_max : float
             Thickness bounds (m).
        sd_min : float
             Standard deviation for thickness (m).
        mar, sar : float
             Mean and SD for Aspect Ratio.
        n_sills : int
             Count.
             
        Returns
        -------
        major, minor : numpy.ndarray
             Dimensions in meters.
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
        Generates sill dimensions using a Uniform Distribution.
        
        Parameters
        ----------
        min_min, min_max : float
            Thickness bounds (m).
        min_ar, max_ar : float
            Aspect Ratio bounds.
        n_sills : int
            Count.
            
        Returns
        -------
        major, minor : numpy.ndarray
            Dimensions in meters.
        """
        aspect_ratio = np.random.uniform(min_ar, max_ar, n_sills)
        minor = np.round(np.random.randn(min_min, min_max, n_sills))
        major = np.multiply(minor, aspect_ratio)
        return major, minor
    @staticmethod
    def value_pusher(array, new_value, push_index, push_value):
        """
        Inserts a single value into a 2D array and shifts column values down.
        
        Note: Less efficient than `value_pusher2D`. Operations are column-specific.
        
        Parameters
        ----------
        array : numpy.ndarray
             Target array.
        new_value : float/int
             Value to insert.
        push_index : tuple (row, col)
             Insertion point.
        push_value : int
             Number of rows to shift down (thickness of insertion).
             
        Returns
        -------
        array : numpy.ndarray
             Modified array.
        """
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
    def prop_updater(lithology, prop_dict: dict, properrty: str):
        """
        Updates a property array based on the lithology map and a property dictionary.
        
        Used after mesh manipulation (shifting/pushing) to ensure properties 
        (density, porosity, etc.) match the new lithology positions.
        
        Parameters
        ----------
        lithology : numpy.ndarray
             Lithology map.
        prop_dict : dict
             Mapping of {lithology: {property_name: value}}.
        property : str
             Key of the property to returning (e.g., 'Density').
             
        Returns
        -------
        prop : numpy.ndarray
             Updated property field.
        """
        prop = np.zeros_like(lithology)
        for rock in prop_dict.keys():
            prop[lithology==rock] = prop_dict[rock][properrty]
        return prop
    

    @staticmethod
    def value_pusher2D(array, new_value, row_index, push_amount):
        """
        Vectorized shifting of 2D array columns to simulate intrusion of sills.
        
        Moves values downwards in each column by `push_amount` starting from `row_index`.
        Fills the gap with `new_value`.
        
        Parameters
        ----------
        array : numpy.ndarray
            Target array.
        new_value : float/int
            Value to fill in the opened space (e.g., magma property).
        row_index : numpy.ndarray (1D, length=cols)
            Start row for shift in each column.
        push_amount : numpy.ndarray (1D, length=cols)
            Amount to shift in each column.
        """
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
        """
        Finds indices where a string matches or contains a substring in an object array.
        
        Used for identifying specific sill IDs in the 3D sill cube.
        Custom logic: 'Only check if the character BEFORE `string` is a digit' suggests avoiding partial matches like '10s' when searching for '1s'.
        
        Parameters
        ----------
        array : numpy.ndarray
            Array of strings/objects.
        string : str
            Target substring (e.g., '1s100').
            
        Returns
        -------
        boolean mask : numpy.ndarray
        """
        if not np.issubdtype(array.dtype, np.str_):
            array = array.astype(str)
        
        def contains_string(element):
            s = str(element)
            idx = s.find(string)
            if idx == -1:
                return False  # `string` not found
            # Only check if the character BEFORE `string` is a digit
            return (idx == 0) or (not s[idx-1].isdigit())
                
        string_index = np.vectorize(contains_string)(array)
        return string_index


    def mult_sill(self, T_field,  majr, minr, height, x_space, dx, dy, rock = np.array([]), emplace_rock = 'basalt', T_mag = 1000, shape = 'elli', dike_empl = True, push = False):
        """
        Emplaces a single sill with advanced options.
        
        Options include:
        - Rectangular ('rect') or Elliptical ('elli') shape.
        - Updating a lithology ('rock') map simultaneously.
        - 'Pushing' existing strata down (`push=True`) vs Overwriting (`push=False`).
        - Adding a vertical feeder dike (`dike_empl=True`).
        
        Parameters
        ----------
        T_field : numpy.ndarray
             Temperature field to modify.
        majr : float
             Major axis length (width) in m. (Code converts to nodes).
        minr : float
             Minor axis length (thickness) in m. (Code converts to nodes).
        height, x_space : float
             Center coordinates (indices).
        dx, dy : float
             Grid spacing.
        rock : numpy.ndarray
             Lithology map (optional).
        emplace_rock : str
             Rock type string for the new sill.
        T_mag : float
             Magma temperature.
        shape : str
             'rect' or 'elli'.
        dike_empl : bool
             If True, adds a vertical channel from the sill down to the bottom boundary.
        push : bool
             If True, uses `value_pusher2D` to displace existing rocks.
             
        Returns
        -------
        if rock provided: T_field, rock, new_dike (mask)
        else: T_field, new_dike
        """
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
        Calculates radiogenic heat production and latent heat of crystallization.
        
        Based on Rybach and Cermack (1982).
        
        Parameters
        ----------
        T_field : numpy.ndarray
            Temperature field.
        rho : float or numpy.ndarray
            Density field (kg/m3).
        CU, CTh, CK : numpy.ndarray
            Concentrations of Uranium (ppm), Thorium (ppm), Potassium (%).
        T_sol : float
            Solidus temperature.
        dike_net : numpy.ndarray
             Mask indicating location of magmatic intrusions (for latent heat).
             
        Returns
        -------
        H : numpy.ndarray
             Heat generation term (W/m3).
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
        """
        Rotates 2D coordinates clockwise by angle `theta`.
        
        Parameters
        ----------
        coords_array : list or array
            [row_index, col_index] i.e. [y, x].
        theta : float
            Rotation angle in degrees.
        center : tuple
            Center of rotation (y, x).
            
        Returns
        -------
        x1, y1 : numpy.ndarray
             Rotated coordinates.
        """
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
        """
        Generates a 3D "Sill Cube" containing strings identifying sills at each voxel.
        
        This cube is pre-calculated to map where sills will be at any given time/slice.
        Strings like "3s1000" means Sill #3 emplaced at time 1000.
        
        Parameters
        ----------
        x, y, z : float
             Physical dimensions of the crustal block (m).
        dx, dy : float
             Grid spacing (m). Note: z assumed to have dx spacing? code uses c = int(z//dx).
        n_sills : int
             Number of sills.
        x_coords, y_coords, z_coords : numpy.ndarray
             Center coordinates (Indices? Or meters? Code suggests z_coords are checked against z_len (indices?)).
             Note: Variable naming in `mult_sill` used meters for inputs but code converted. Here usually inputs are pre-converted or matching logic. 
             Looking at `sillcube` generation `z_coords[l] - maj_dims[l]`, these are likely indices.
        maj_dims, min_dims : numpy.ndarray
             Dimensions (m).
        empl_times : numpy.ndarray
             Emplacement times.
             
        Returns
        -------
        sillcube : numpy.ndarray (dtype=object)
             3D array of strings.
        """
        a = int(y // dy)
        b = int(x // dx)
        c = int(z // dx)
        sillcube = np.empty([c, a, b], dtype=object)
        sillcube[:, :, :] = ''
        z_len, y_len, x_len = np.ogrid[:c, :a, :b]

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
                '''
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
                '''
                z_min = max(0, int(z_coords[l] - maj_dims[l]))
                z_max = min(c, int(z_coords[l] + maj_dims[l] + 1))
                
                y_min = max(0, int(y_coords[l] - min_dims[l]))
                y_max = min(a, int(y_coords[l] + min_dims[l] + 1))
                
                x_min = max(0, int(x_coords[l] - maj_dims[l]))
                x_max = min(b, int(x_coords[l] + maj_dims[l] + 1))
                z_local, y_local, x_local = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]
                dist_sq = (
                            ((z_local - z_coords[l])**2 / maj_dims[l]**2) + 
                            ((y_local - y_coords[l])**2 / min_dims[l]**2) + 
                            ((x_local - x_coords[l])**2 / maj_dims[l]**2)
                        )
                local_mask = dist_sq <= 1
                target_region = sillcube[z_min:z_max, y_min:y_max, x_min:x_max]
                target_region[local_mask] += '_' + str(l) + 's' + str(empl_times[l])
                sillcube[z_min:z_max, y_min:y_max, x_min:x_max] = target_region
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
        """
        Emplaces a specific sill from the 3D sillcube into a 2D Temperature slice.
        
        Searches the 2D slice of `sillcube` (at `z_index`) for the string key corresponding 
        to the sill (`n_rep`) and time (`curr_empl_time`). Sets T_field to T_mag there.
        
        Parameters
        ----------
        T_field : numpy.ndarray
            2D Temperature slice.
        sillcube : numpy.ndarray
            3D Sill Cube.
        n_rep : int
            Sill ID to emplace.
        T_mag : float
            Magma temperature.
        z_index : int
            Which Z-slice of the cube corresponds to the current 2D model plane.
        
        Returns
        -------
        T_field : numpy.ndarray
            Updated temperature.
        """
        string_finder = str(n_rep)+'s'+str(curr_empl_time)
        if len(sillcube.shape)!=3:
            raise IndexError('sillcube array must be three-dimensional')
        if T_field.size==0:
            raise IndexError("T_field cannot be empty")
        T_field[self.index_finder(sillcube[z_index], string_finder)] = T_mag
        return T_field


    def sill3D_pushy_emplacement(self, props_array, props_dict, sillsquare, n_rep, mag_props_dict, curr_empl_time):
        """
        Emplaces a sill into a set of 2D property arrays using the "Push" mechanism.
        
        Instead of overwriting, it inserts the sill and shifts the existing rock/properties downwards.
        
        Parameters
        ----------
        props_array : numpy.ndarray
             3D Array containing stacked 2D property fields (Temp, Lith, Porosity, etc.).
             Shape: (N_properties, Rows, Cols).
        props_dict : dict
             Mapping {Property_Name: Index_in_props_array}.
        sillsquare : numpy.ndarray
             2D extracted slice from sillcube containing emplacement keys.
        n_rep : int
             Sill ID.
        mag_props_dict : dict
             Properties of the magma to insert.
        curr_empl_time : int
             Time of emplacement (used for key generation).
             
        Returns
        -------
        props_array, row_push_start, columns_pushed
        """
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
    #Class that contains functions that make functions from the other classes easier to use for specific use cases
    """
    The `sill_controls` class orchestrates the model setup, execution, and property management.
    
    It serves as a "Controller" in the MVC sense, linking the "Cooling" physics (`cool`), 
    the "Rules" logic (`rules`), and the state management of the simulation.
    
    It handles:
    1.  **Initialization**: Setting up grids, properties, and constants.
    2.  **Property Management**: Translating strings (rock names) to physical properties (rho, Cp, k).
    3.  **High-Level Workflows**: Generating sill swarms (`build_sillcube`), calculating distances, and getting initial states.
    """
    def __init__(self, x, y, dx, dy, 
                 T_liquidus = 1250, T_solidus = 800, include_external_heat = True,
                 k_const = True, kc_val = 7.884e7, cp = 1, cp_const = True,
                 calculate_closest_sill = False, calculate_all_sills_distances = False, calculate_at_all_times = False, 
                 rock_prop_dict = None, magma_prop_dict = None,lith_plot_dict=None,
                 sill_cube_dir='sillcubes/',k_func=None, cp_func = None, melt_rock = 'basalt',
                 melt_fraction_function = None, melt_function_args = None):
        """
        Initializes the simulation controller.
        
        Parameters
        ----------
        x, y : float
            Physical dimensions of the crustal block (meters).
        dx, dy : float
            Grid spacing (meters).
        T_liquidus, T_solidus : float
            Magma phase change temperatures (C).
        include_external_heat : bool
            If True, calculates Latent Heat and Radiogenic Heat.
        k_const : bool
            If True, uses constant thermal conductivity `kc_val`.
        kc_val : float
            Constant thermal conductivity value.
        cp : float
            Initial specific heat capacity (reference).
        cp_const : bool
            If True, uses constant or simple lookup for Cp.
        calculate_closest_sill : bool
            Enable proximity analysis for thermal interaction.
        rock_prop_dict, magma_prop_dict : dict
            Custom property dictionaries.
        k_func, cp_func : callable
            Custom functions for temperature/pressure dependent properties.
        melt_rock : str
            Type of rock melting.
        """
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

        
        if magma_prop_dict is None:
            self.magma_prop_dict = {'Temperature': 1100,
                        'Lithology': 'basalt_melt',
                        'Porosity': 0, #Porosity of the rock for calculation of carbon emissions
                        'Density': 2850, #kg/m3
                        'Specific Heat': 850, 
                        'Latent Heat': 4e5,
                        'TOC':0} #wt%
        else:
            self.magma_prop_dict = magma_prop_dict

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
                },
                self.magma_prop_dict['Lithology']:{
                    'Porosity':self.magma_prop_dict['Porosity'],
                    'Density':self.magma_prop_dict['Density'],
                    'TOC':self.magma_prop_dict['TOC'],
                    'Specific Heat': self.magma_prop_dict['Specific Heat']
                }
            }
        else:
            self.rock_prop_dict = rock_prop_dict
        #Lithology dictionary to translate rock types into numerical codes for numpy arrays
        if lith_plot_dict is None:
            all_liths = {
                **self.rock_prop_dict, 
                self.magma_prop_dict['Lithology']: self.magma_prop_dict
                }
            self.lith_plot_dict = {name: i for i, name in enumerate(all_liths.keys())}
        else:
            self.lith_plot_dict = lith_plot_dict
        
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
        self.melt = melt_rock
        self.melt_fraction_function = melt_fraction_function
        self.args = melt_function_args

    def generate_sill_2D_slices(self, fluxy_list,iter_list,z_index_list, lat_range = None, file_dir = None):
        """
        Extracts 2D slices from pre-calculated 3D sill cubes for 2D simulation usage.
        
        This allows running multiple 2D cross-sections from a single 3D stochastic generation.
        
        Parameters
        ----------
        fluxy_list : list of float
            List of magma fluxes to process.
        iter_list : list of int
            List of iteration indices (indices in the n_sills_dataframe).
        z_index_list : list of int
            List of Z-indices (slices) to extract from the cube.
        lat_range : list, optional
            Lateral range filter.
        file_dir : str, optional
            Root directory for sill cubes.
        """
        file_dir = self.sill_cube_dir if file_dir is None else file_dir
        for flux in fluxy_list:
            load_dir = file_dir+str(format(flux, '.3e'))+'/'+str(lat_range) if lat_range is not None  else file_dir+str(format(flux, '.3e'))
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
        """
        Retreives or calculates thermal properties (Conductivity, Specific Heat) for the grid.
        
        Handles both constant and temperature-dependent property logic based on initialization flags.
        
        Parameters
        ----------
        T_field : numpy.ndarray
            Temperature field.
        rock : numpy.ndarray
            Lithology field.
        density : numpy.ndarray
            Density field.
        dy : float
            Grid spacing.
        return_all : bool
            If True, returns (Diffusivity 'k', Cp, Thermal_Conductivity).
            If False, returns only Diffusivity 'k'. (Note: variable `k` here usually denotes diffusivity = cond / (rho*cp)).
            
        Returns
        -------
        k (diffusivity) [and Cp, Conductivity if return_all=True]
        """
        magma_name = self.magma_prop_dict['Lithology']
        if magma_name not in self.rock_prop_dict:
            self.rock_prop_dict[magma_name] = {
                'Porosity': self.magma_prop_dict['Porosity'],
                'Density': self.magma_prop_dict['Density'],
                'TOC': self.magma_prop_dict['TOC'],
                'Specific Heat': self.magma_prop_dict['Specific Heat']
            }
        if not self.k_const:
            thermal_conductivity = np.array(self.k_func(T_field, rock, density, dy))
        else:
            thermal_conductivity = np.array(self.thermal_conductivity)
        if not self.cp_const:
            specific_heat = self.cp_func(T_field, rock, density, dy)
        else:
            specific_heat = self.rool.prop_updater(rock, self.rock_prop_dict, 'Specific Heat')
            specific_heat[rock==self.magma_prop_dict['Lithology']] = self.magma_prop_dict['Specific Heat']
        specific_heat = np.array(specific_heat, dtype = float)
        try:
            k = np.array(thermal_conductivity/density/specific_heat, dtype = float)
        except:
            target = self.magma_prop_dict['Lithology']
            actual_unique = np.unique(list(self.rock_prop_dict.keys()))
            print(target, actual_unique)
            if target not in actual_unique:
                print(f"--- INTERMITTENT ERROR DETECTED ---")
                print(f"Looking for: '{target}'")
                print(f"Found in array: {actual_unique}")
                # This check helps see if the name changed slightly
                import difflib
                matches = difflib.get_close_matches(target, [str(x) for x in actual_unique])
                print(f"Close matches: {matches}")
                print("Did you add solid rock properties for the melt in the rock properties dictionary?")
                raise ValueError(f"Target lithology '{target}' not found in rock array. Check for typos or missing entries in rock_prop_dict.")
            else:
                print(f"Rock shape: {rock.shape}, Density shape: {density.shape}, Specific Heat shape: {specific_heat.shape}, Thermal Conductivity shape: {thermal_conductivity.shape}")
                raise ValueError("Array division failed. You might have a mismatch in array shapes")
        if return_all:
            return k, np.array(specific_heat, dtype = float), np.array(thermal_conductivity, dtype = float)
        else:
            return k

    def get_physical_properties(self, rock, rock_prop_dict):
        """
        Converts a Lithology map (strings) into physical property arrays using a dictionary lookup.
        
        Parameters
        ----------
        rock : numpy.ndarray
            2D array of rock types (str or object).
        rock_prop_dict : dict
            Mapping of rock types to properties.
            
        Returns
        -------
        porosity : numpy.ndarray
        density : numpy.ndarray
        TOC : numpy.ndarray
        """
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
        """
        Wrapper to dynamically call a function with unpacking arguments.
        
        Mostly a helper to keep code clean when switching between different distribution functions 
        (normal, uniform, empirical) that take different arguments.
        
        Parameters
        ----------
        func : callable
        *args, **kwargs : arguments
        
        Returns
        -------
        result : return value of func
        """
        result = func(*args,**kwargs)
        # If the result is a tuple or list, enumerate it
        #if isinstance(result, (tuple, list)):
        #    enumerated_result = list(enumerate(result))
        #    return enumerated_result
        # If the result is a single value, return it as is
        return result
    
    @staticmethod
    def check_closest_sill_temp(T_field, sills_array, curr_sill, dx, time, T_solidus, no_sill='', calculate_all=False, save_file=None):
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
        #print(f'DEBUG: unique sill values are {np.unique(sills_array.astype(str))}')
        def get_width_and_thickness(bool_array):
            '''
            Function to get the width and thickness of a sill inside the check closest sill temp function
            '''
            rows, columns = np.where(bool_array)

            if rows.size == 0 or columns.size == 0:
                # Return default values if the sill is empty/cold
                return 0, 0, 'N/A'
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

        # Prepare grid points
        a, b = T_field.shape
        rows = np.arange(a)
        columns = np.arange(b)
        rows_grid, columns_grid = np.meshgrid(rows, columns, indexing='ij')
        points = np.column_stack((rows_grid.ravel(), columns_grid.ravel()))
        sills_number = sills_array.copy()

        if calculate_all:
            # Calculate for all sills
            all_sills_data = pd.DataFrame()
            all_sill_ids = np.unique(sills_array[sills_array!=no_sill])
            
            # Optimization: Build Global KDTree of all hot points once
            hot_mask = (T_field > T_solidus) & (sills_array != no_sill)
            if np.any(hot_mask):
                hot_points_flat = points[hot_mask.ravel()]
                hot_sill_ids_flat = sills_array.ravel()[hot_mask.ravel()]
                global_tree = KDTree(hot_points_flat)
            else:
                global_tree = None
                hot_sill_ids_flat = np.array([])

            for sill_id in all_sill_ids: 
                # Initialize data for this sill
                sills_data = pd.DataFrame({'sills': sill_id}, index=[0])
                is_curr_sill = (sills_array == sill_id)
                
                # Find edges of current sill
                is_edge = np.zeros_like(is_curr_sill, dtype=bool)
                is_edge[1:-1, 1:-1] = (
                    (is_curr_sill[:-2, 1:-1] != is_curr_sill[1:-1, 1:-1]) |  # Above
                    (is_curr_sill[2:, 1:-1] != is_curr_sill[1:-1, 1:-1]) |   # Below
                    (is_curr_sill[1:-1, :-2] != is_curr_sill[1:-1, 1:-1]) |  # Left
                    (is_curr_sill[1:-1, 2:] != is_curr_sill[1:-1, 1:-1])     # Right
                )
                query_points = points[is_edge.ravel()]
                
                # Default values
                saved_distance = 1e30
                saved_index = 'N/A'
                saved_temperature = -1
                saved_sill = 'N/A'
                closest_curr_sill = 'N/A'
                
                # Dimensions
                if np.any(is_curr_sill):
                    curr_sill_width, curr_sill_thickness, curr_sill_center = get_width_and_thickness(is_curr_sill)
                else:
                    curr_sill_width = curr_sill_thickness = 0
                    curr_sill_center = 'N/A'
                
                # Find closest sill using Global Tree
                if len(query_points) > 0 and global_tree is not None:
                    # Query k neighbors to skip self
                    k_neighbors = 10 
                    # Clip k if fewer points exist
                    k_safe = min(k_neighbors, len(hot_sill_ids_flat))
                    
                    distances, indices = global_tree.query(query_points, k=k_safe)
                    
                    # Ensure 2D arrays even if k=1
                    if k_safe == 1:
                        distances = distances[:, None]
                        indices = indices[:, None]
                        
                    # Filter self
                    found_min = False
                    
                    # indices map to hot_sill_ids_flat
                    neighbor_ids = hot_sill_ids_flat[indices] # (N_query, k)
                    
                    # Mask of valid neighbors (id != curr)
                    valid_mask = (neighbor_ids != sill_id)
                    
                    if np.any(valid_mask):
                        # Set invalid distances to inf
                        valid_dists = np.where(valid_mask, distances, np.inf)
                        
                        # Min over neighbors
                        min_dist_per_point = np.min(valid_dists, axis=1)
                        # Min over all query points
                        min_idx_query = np.argmin(min_dist_per_point)
                        min_val = min_dist_per_point[min_idx_query]
                        
                        if min_val < 1e29:
                            # Retrieve the neighbor index that gave this
                            # valid_dists[min_idx_query] is array of k distances
                            k_idx = np.argmin(valid_dists[min_idx_query])
                            global_idx = indices[min_idx_query, k_idx]
                            
                            saved_distance = min_val
                            index1 = hot_points_flat[global_idx]
                            closest_curr_sill = str(query_points[min_idx_query])
                            
                            saved_index = str(index1)
                            saved_temperature = T_field[index1[0], index1[1]]
                            saved_sill = sills_array[index1[0], index1[1]]
                            found_min = True
                    
                    if not found_min:
                        # Fallback: If 10 neighbors were all self, or something tricky.
                        # Rebuild tree excluding self (Slow Path, but rare if k=10 is checked)
                        # Only if current sill is HOT (otherwise it wouldn't be in global tree)
                        # Explicitly filter points
                        # Original Logic:
                        condition = (T_field > T_solidus) & (~is_curr_sill) & (sills_array != no_sill)
                        filtered_points = points[condition.ravel()]
                        
                        if len(filtered_points) > 0:
                            tree_fallback = KDTree(filtered_points)
                            d, i = tree_fallback.query(query_points)
                            min_idx = np.argmin(d)
                            index1 = filtered_points[i[min_idx]]
                            
                            saved_distance = d[min_idx]
                            saved_index = str(index1)
                            saved_temperature = T_field[index1[0], index1[1]]
                            saved_sill = sills_array[index1[0], index1[1]]
                            closest_curr_sill = str(query_points[min_idx])
                    
                    # Get closest sill dimensions
                    is_closest_sill_curr = (sills_array == saved_sill) & (T_field > T_solidus)
                    closest_sill_width_curr, closest_sill_thickness_curr, closest_sill_center_curr = get_width_and_thickness(is_closest_sill_curr)
                    
                    is_closest_sill = (sills_array == saved_sill)
                    closest_sill_width, closest_sill_thickness, closest_sill_center = get_width_and_thickness(is_closest_sill)
                
                # Populate data
                sills_data['closest_sill'] = saved_sill
                sills_data['distance'] = saved_distance * dx
                sills_data['index of closest sill'] = saved_index
                sills_data['temperature'] = saved_temperature
                sills_data['index of current sill'] = closest_curr_sill
                sills_data['width of current sill'] = curr_sill_width * dx
                sills_data['thickness of current sill'] = curr_sill_thickness * dx
                sills_data['index of current sill center'] = curr_sill_center
                sills_data['width of closest sill'] = closest_sill_width_curr * dx
                sills_data['thickness of closest sill'] = closest_sill_thickness_curr * dx
                sills_data['current center of closest sill'] = closest_sill_center_curr
                sills_data['original width of closest sill'] = closest_sill_width * dx
                sills_data['original thickness of closest sill'] = closest_sill_thickness * dx
                sills_data['original center of closest sill'] = closest_sill_center
                sills_data['current time'] = time
                
                # Append to results
                all_sills_data = pd.concat([all_sills_data, sills_data], ignore_index=True)
            
            # Save or return results
            if save_file is not None:
                all_sills_data.to_csv(save_file + '.csv')
            return all_sills_data
        
        else:
            # Calculate for single sill
            sills_data = pd.DataFrame({'sills': curr_sill}, index=[0])
            is_curr_sill = (sills_array == curr_sill)
            
            # Find edges of current sill
            is_edge = np.zeros_like(is_curr_sill, dtype=bool)
            is_edge[1:-1, 1:-1] = (
                (is_curr_sill[:-2, 1:-1] != is_curr_sill[1:-1, 1:-1]) |  # Above
                (is_curr_sill[2:, 1:-1] != is_curr_sill[1:-1, 1:-1]) |   # Below
                (is_curr_sill[1:-1, :-2] != is_curr_sill[1:-1, 1:-1]) |  # Left
                (is_curr_sill[1:-1, 2:] != is_curr_sill[1:-1, 1:-1])     # Right
            )
            query_points = points[is_edge.ravel()]
            
            # Find other hot sills
            condition = (T_field > T_solidus) & (sills_number != curr_sill) & (sills_array != no_sill)
            filtered_points = points[condition.ravel()]
            
            # Initialize default values
            saved_distance = 1e30
            saved_index = 'N/A'
            saved_temperature = -1
            saved_sill = -1
            closest_curr_sill = 'N/A'
            closest_sill_width_curr = 0
            closest_sill_width = 0
            closest_sill_thickness = 0
            closest_sill_thickness_curr = -1
            closest_sill_center_curr = 'N/A'
            closest_sill_center = 'N/A'
            
            # Get current sill dimensions
            if np.any(is_curr_sill):
                curr_sill_width, curr_sill_thickness, curr_sill_center = get_width_and_thickness(is_curr_sill)
            else:
                curr_sill_width = curr_sill_thickness = 0
                curr_sill_center = 'N/A'
            
            # Find closest sill if edges exist
            if len(query_points) > 0 and len(filtered_points) > 0:
                tree = KDTree(filtered_points)
                distances, indices = tree.query(query_points)  # Batch query
                min_idx = np.argmin(distances)
                index1 = filtered_points[indices[min_idx]]
                
                saved_distance = distances[min_idx]
                saved_index = str(index1)
                saved_temperature = T_field[index1[0], index1[1]]
                saved_sill = sills_array[index1[0], index1[1]]
                closest_curr_sill = str(query_points[min_idx])
                
                # Get closest sill dimensions
                is_closest_sill_curr = (sills_array == saved_sill) & (T_field > T_solidus)
                closest_sill_width_curr, closest_sill_thickness_curr, closest_sill_center_curr = get_width_and_thickness(is_closest_sill_curr)
                
                is_closest_sill = (sills_array == saved_sill)
                closest_sill_width, closest_sill_thickness, closest_sill_center = get_width_and_thickness(is_closest_sill)
            #print(f'Closest sill for sill {curr_sill} is sill {str(saved_sill)}')
            # Populate data
            sills_data['closest_sill'] = saved_sill
            sills_data['distance'] = saved_distance * dx
            sills_data['index of closest sill'] = str(saved_index)
            sills_data['temperature'] = saved_temperature
            sills_data['index of current sill'] = closest_curr_sill
            sills_data['width of current sill'] = curr_sill_width * dx
            sills_data['thickness of current sill'] = curr_sill_thickness * dx
            sills_data['index of current sill center'] = curr_sill_center
            sills_data['width of closest sill'] = closest_sill_width_curr * dx
            sills_data['thickness of closest sill'] = closest_sill_thickness_curr * dx
            sills_data['current center of closest sill'] = closest_sill_center_curr
            sills_data['original width of closest sill'] = closest_sill_width * dx
            sills_data['original thickness of closest sill'] = closest_sill_thickness * dx
            sills_data['original center of closest sill'] = closest_sill_center
            sills_data['current time'] = time
            return sills_data

    def build_sillcube(self, z, dt, thickness_range, aspect_ratio, depth_range, z_range, lat_range, phase_times, tot_volume, flux, n_sills, shape = 'elli', depth_function = None, lat_function = None, dims_function = None, emplace_dike = False, orientations = None):
        """
        Generates a 3D stochastic model of sill emplacement over time.
        
        This is a complex high-level function that:
        1.  Samples random distributions for sill geometry (depth, location, dimensions).
        2.  Calculates emplacement timing based on magma flux rates.
        3.  Checks for volume constraints and iterates until the target total volume is reached.
        4.  Generates visualization plots (Depth, Width/Thickness distributions, Volume vs Time).
        5.  Calls `rules.sill_3Dcube` to construct the final 3D string array.
        
        Parameters
        ----------
        z : float
            Z-dimension extent (m).
        dt : float
            Time step.
        thickness_range : list
            [min, max, sd] for thickness.
        aspect_ratio : list
            [mean, sd] for aspect ratio.
        depth_range, z_range, lat_range : list
            [min, max, sd] for spatial bounds.
        phase_times : list
            [thermal_maturation_time, cooling_time]. Used to offset emplacement start.
        tot_volume : float
            Target total volume of magma (m3).
        flux : float
            Magma supply rate (m3/s).
        n_sills : int
            Estimated number of sills (used for initial array sizing).
        shape : str
            'elli' or 'rect'.
        depth_function, lat_function, dims_function : str or callable
            Distribution types ('normal', 'uniform', 'empirical').
            
        Returns
        -------
        sillcube : numpy.ndarray
             3D string array.
        n_sills : int
             Actual number of sills emplaced.
        params : numpy.ndarray
             [empl_times, empl_heights, x_space, width, thickness]
        """
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
        n_sills = None
        for l in range(len(time_steps)):
            if time_steps[l]<thermal_maturation_time+dt:
                continue
            else:
                if n>0:
                    mean_flux = np.sum(volume[0:n])/(time_steps[l]-thermal_maturation_time)
                else:
                    mean_flux = 0
                unemplaced_volume += flux*dt
                if unemplaced_volume>=volume[n] and mean_flux<1.1*flux:
                    empl_times.append(time_steps[l])
                    plot_time.append(time_steps[l])
                    cum_volume.append(volume[n])
                    unemplaced_volume -= volume[n]
                    print(f'Emplaced sill {n} at time {time_steps[l]}')
                    print(f'Remaining volume to emplace: {tot_volume-np.sum(volume[0:n]):.4e}')
                    print(f'n_sills is now {n}')
                    mean_flux = np.sum(volume[0:n])/(time_steps[l]-thermal_maturation_time)
                    n+=1
                    
                    while (unemplaced_volume>volume[n] and 
                        mean_flux<(0.95*flux if np.sum(volume[0:n+1])<=tot_volume else 1.05*flux) and 
                        np.sum(volume[0:n])<=tot_volume):
                        empl_times.append(time_steps[l])
                        unemplaced_volume -= volume[n]
                        cum_volume[-1]+=volume[n]
                        print(f'Emplaced sill {n} at time {time_steps[l]}')
                        print(f'Remaining volume to emplace: {tot_volume-np.sum(volume[0:n]):.4e}')
                        print(f'n_sills is now {n}')
                        mean_flux = np.sum(volume[0:n])/(time_steps[l]-thermal_maturation_time)
                        n+=1

                if (n>0) and (np.sum(volume[0:n])>tot_volume):
                    print('Total sills emplaced:', n)
                    n_sills = int(n)
                    break
        if n_sills is None:
            print(f"Warning: Loop finished without reaching tot_volume.")
            print(f"Current n is {n}. Arrays will be sliced to this value.")
        empl_heights = empl_heights[0:n_sills]
        x_space = x_space[0:n_sills]
        width = width[0:n_sills]
        thickness = thickness[0:n_sills]
                    
        if n_sills!=n:
            n_sills = int(n)
            print(f"n_sills:, {n_sills}; n, {n}")
            print(f"Warning: Final if not entered")
            print(f'Volume debug {unemplaced_volume:.3e}, {volume[n]:.3e}')
            print(f'Flux debug ({mean_flux:.3e}, {volume[n]/(time_steps[l]-thermal_maturation_time):.3e})')
            print(f'Volume debug 2{ np.sum(volume[0:n]):.3e}, {tot_volume:.3e}')
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
        """
        Runs a "background" thermal maturation model for the initial condition.
        
        Calculates the thermal maturation and CO2 release that occurs *during the thermal maturation phase* 
        (before sills start emplacing). This establishes the baseline TOC and vitrinite reflectance 
        of the crust before the magmatic event.
        
        Parameters
        ----------
        props_array : numpy.ndarray
            3D array of property fields.
        dt : float
            Time step.
        method : str
            Solver method name.
        time : float
            Total duration of this initialization phase.
            
        Returns
        -------
        current_time : float
        tot_RCO2 : list
             Time series of CO2 release.
        props_array : numpy.ndarray
             Updated properties (TOC lowered, etc.).
        RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli
             Final states of the SILLi variables.
        """
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

        k, specific_heat, _ = self.sill_controls_get_k(T_field, rock, density, self.dy,return_all=True)
        if dt>np.round((min(dx,dy)**2)/(np.max(k)),3):
            print(f'Warning: Given time step {dt} is larger than stable...')
            dt = np.round((min(dx,dy)**2)/(2*np.max(k)),3)
            print(f'dt changed to {dt}')
            print(f'Maximum thermal conductivity is {np.max(k)} for rock type {props_array[self.rock_index][np.where(k==np.max(k))[0][0]][0]}')
            #method = 'adi'
        breakdown_CO2 = np.zeros_like(T_field)
        #specific_heat = np.zeros_like(T_field)
        #specific_heat = np.vectorize(
        #    lambda rt: self.rock_prop_dict[rt]['Specific Heat'], 
        #    otypes=[float]  # Ensure output is float
        #)(rock)
        if self.include_heat:
                if self.melt_fraction_function is None:
                    H_rad = self.cool.get_radH(T_field, density,dx)/density/specific_heat
                    H_lat = self.cool.get_latH(T_field, rock, self.magma_prop_dict['Lithology'], self.magma_prop_dict['Specific Heat'], self.magma_prop_dict['Latent Heat'], self.T_liquidus, self.T_solidus, curve_func=self.melt_fraction_function)
                    H = np.array([H_rad, H_lat])
                    #H = H/self.magma_prop_dict['Density']/magma_prop_dict['Specific Heat']
                else:
                    H_rad = self.cool.get_radH(T_field, density,dx)/density/specific_heat
                    H_lat = self.cool.get_latH(T_field, rock, self.magma_prop_dict['Lithology'], self.magma_prop_dict['Specific Heat'], self.magma_prop_dict['Latent Heat'], self.T_liquidus, self.T_solidus, curve_func=self.melt_fraction_function)                    
                    H = np.array([H_rad, H_lat])
        else:
            H_rad = self.cool.get_radH(T_field, density,dx)/density/specific_heat
            H_lat = np.ones_like(T_field)
            H = np.array([H_rad, H_lat])

        t = 0
        a, b = props_array[0].shape
        TOC = self.rool.prop_updater(props_array[self.rock_index], rock_prop_dict, 'TOC')
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
            max_T = np.max(T_field)
            for l in trange(0, len(t_steps)):
                T_field = self.cool.diff_solve(k, a, b, dx, dy, dt, T_field, np.nan, method, H)
                if (T_field>max_T).any():
                    plt.imshow(np.array(T_field, dtype = float))
                    plt.colorbar()
                    plt.title(t_steps[l])
                    plt.show()
                    print("Warning: T_field is not stable!")
                if np.isnan(np.array(T_field, dtype = float)).any():
                    plt.imshow(T_field)
                    plt.show()
                    raise(ValueError('T_field is nan'))
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
        props_array[self.sph_index] = specific_heat
        return current_time, tot_RCO2, props_array, RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli

    def get_sillburp_initial_thermogenic_state(self, props_array, dt, method, sillburp_weights = None, time = np.nan, lith_plot_dict = None, rock_prop_dict = None):
        '''
        Function to get the background CO2 release for the sillburp model over the thermal maturation time before the emplacement of sills begins
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

        k, specific_heat, _ = self.sill_controls_get_k(T_field, rock, density, dy, return_all=True)
        if dt>np.round((min(dx,dy)**2)/(5*np.max(k)),3):
            print(f'Warning: Given time step {dt} is larger than stable. Changing method from {method} to adi')
            print(f'Maximum thermal conductivity is {np.max(k)} for rock type {props_array[self.rock_index][np.where(k==np.max(k))[0][0]][0]}')
            method = 'adi'
        breakdown_CO2 = np.zeros_like(T_field)
        dV = dx*dx*dy
        t = 0
        a, b = props_array[0].shape
        specific_heat = np.vectorize(
            lambda rt: self.rock_prop_dict[rt]['Specific Heat'], 
            otypes=[float]  # Ensure output is float
        )(rock)
        if self.include_heat:
                if self.melt_fraction_function is None:
                    H_rad = self.cool.get_radH(T_field, density,dx)/density/specific_heat
                    H_lat = self.cool.get_latH(T_field, rock, self.magma_prop_dict['Lithology'], self.magma_prop_dict['Specific Heat'], self.magma_prop_dict['Latent Heat'], self.T_liquidus, self.T_solidus, curve_func=self.melt_fraction_function)
                    H = np.array([H_rad, H_lat])
                    #H = H/self.magma_prop_dict['Density']/magma_prop_dict['Specific Heat']
                else:
                    H_rad = self.cool.get_radH(T_field, density,dx)/density/specific_heat
                    H_lat = self.cool.get_latH(T_field, rock, self.magma_prop_dict['Lithology'], self.magma_prop_dict['Specific Heat'], self.magma_prop_dict['Latent Heat'], self.T_liquidus, self.T_solidus, curve_func=self.melt_fraction_function)                    
                    H = np.array([H_rad, H_lat])
        else:
            H_rad = self.cool.get_radH(T_field, density,dx)/density/specific_heat
            H_lat = np.ones_like(T_field)
            H = np.array([H_rad, H_lat])

        TOC = self.rool.prop_updater(props_array[self.rock_index], rock_prop_dict, 'TOC')
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
                TOC = np.array(self.rool.prop_updater(rock, rock_prop_dict, 'TOC'), dtype = float)
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
        magma_name = self.magma_prop_dict['Lithology']
        if magma_name not in self.rock_prop_dict:
            self.rock_prop_dict[magma_name] = {
                'Porosity': self.magma_prop_dict['Porosity'],
                'Density': self.magma_prop_dict['Density'],
                'TOC': self.magma_prop_dict['TOC'],
                'Specific Heat': self.magma_prop_dict['Specific Heat']
            }
        if dt is None:
            dts = np.array([time_steps[i]-time_steps[i-1] for i in range(1, len(time_steps))])
            dts = np.append(dts[0],dts)
        else:
            dts = np.repeat(dt,len(time_steps))
        rock = np.array(props_array[self.rock_index])
        density = np.array(props_array[self.dense_index], dtype = float)
        porosity = np.array(props_array[self.poros_index], dtype = float) 
        T_field = np.array(props_array[self.Temp_index], dtype = float)
        specific_heat = np.array(props_array[self.sph_index])
        TOC1 = self.rool.prop_updater(rock, rock_prop_dict, 'TOC')
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
        sills_emplaced = np.ones((a,b))*(-1)
        tot_melt10 = []
        tot_melt50 = []
        tot_solidus = []
        area_sills = []
        for l in trange(saving_time_step_index, len(time_steps)):
            #curr_time = time_steps[l]
            dt = dts[l]          
            T_field = np.array(props_array[self.Temp_index], dtype = float)
            specific_heat = np.array(props_array[self.sph_index], dtype = float)
            density = np.array(props_array[self.dense_index])
            rock = np.array(props_array[self.rock_index])
            porosity = np.array(props_array[self.poros_index])
            if model=='silli':
                curr_TOC_silli = props_array[self.TOC_index]
            elif model=='sillburp':
                curr_TOC = props_array[self.TOC_index]
            if self.include_heat:
                if self.melt_fraction_function is None:
                    H_rad = np.array(self.cool.get_radH(T_field, density,dx)/density/specific_heat, dtype = float)
                    H_lat = np.array(self.cool.get_latH(T_field, rock, self.magma_prop_dict['Lithology'], magma_prop_dict['Specific Heat'], magma_prop_dict['Latent Heat'], self.T_liquidus, self.T_solidus, curve_func=self.melt_fraction_function), dtype = float)
                    H = np.array([H_rad, H_lat])
                    #H = H/self.magma_prop_dict['Density']/magma_prop_dict['Specific Heat']
                else:
                    H_rad = np.array(self.cool.get_radH(T_field, density,dx)/density/specific_heat, dtype = float)
                    H_lat = self.cool.get_latH(T_field, rock, self.magma_prop_dict['Lithology'], magma_prop_dict['Specific Heat'], magma_prop_dict['Latent Heat'], self.T_liquidus, self.T_solidus, curve_func=self.melt_fraction_function)                    
                    H = np.array([H_rad, H_lat])
            else:
                H_rad = np.array(self.cool.get_radH(T_field, density,dx)/density/specific_heat, dtype = float)
                H_lat = np.ones_like(T_field, dtype = float)
                H = np.array([H_rad, H_lat])
            k = self.sill_controls_get_k(T_field, rock, density, dy, return_all=False)
            T_field = self.cool.diff_solve(k, a, b, dx, dy, dt, T_field, q, cool_method, H)
            is_magma = rock==self.magma_prop_dict['Lithology']
            mask = np.logical_and(T_field<self.T_solidus, is_magma)
            rock[mask] = self.melt
            if (T_field[rock==self.magma_prop_dict['Lithology']]<self.T_solidus).any():
                pdb.set_trace()
            if np.max(T_field)>1.05*1100:
                warnings.warn(f'Too much latent heat: {np.min(H_lat)}. Maximum temperature is now {np.max(T_field)}', RuntimeWarning)

            if self.calculate_closest_sill and self.calculate_at_all_times:
                save_file = save_dir+'/sill_distances'+str(time_steps[l])
                sills_data = self.check_closest_sill_temp(props_array[self.Temp_index], sillnet, curr_sill,dx, time_steps[l], T_solidus=self.T_solidus, calculate_all=self.calculate_all_sill_distances, save_file=save_file)
            props_array[self.Temp_index] = T_field
            props_array[self.rock_index] = rock
            

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
            if model=='silli':
                props_array[self.TOC_index] = curr_TOC_silli
            elif model == 'sillburp':
                props_array[self.TOC_index] = curr_TOC
            else:
                props_array[self.TOC_index] = np.zeros_like(props_array[self.TOC_index])
            TOC1 = self.rool.prop_updater(rock, rock_prop_dict, 'TOC')
            if isinstance(n_sills, str):
                n_sills = n_sills.strip('[]')
            n_sills = int(n_sills)
            while time_steps[l]==empl_times[curr_sill] and curr_sill<int(n_sills):
                #print(f'Now emplacing sill {curr_sill}')
                props_array, row_start, col_pushed = self.rool.sill3D_pushy_emplacement(props_array, prop_dict, sillsquare, curr_sill, magma_prop_dict, empl_times[curr_sill])
                rock = props_array[self.rock_index]
                TOC1 = self.rool.prop_updater(rock, rock_prop_dict, 'TOC')
                    #if (props_array[self.Temp_index][props_array[self.rock_index] == magma_prop_dict['Lithology']] < self.T_solidus).any():
                        #print(f'Warning: Magma temperature is {props_array[props_array[self.rock_index]==magma_prop_dict['Lithology']].max()}')
                        #pdb.set_trace()
                if model=='silli':
                    curr_TOC_silli = props_array[self.TOC_index]
                elif model=='sillburp':
                    curr_TOC = props_array[self.TOC_index]
                sillnet = self.rool.value_pusher2D(sillnet, curr_sill, row_start, col_pushed)
                #pdb.set_trace()
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
                            props_array_vtk.point_data['Specific Heat'] = np.array(props_array[self.sph_index], dtype = float).flatten()
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
                                props_array_vtk.point_data['Specific Heat'] = np.array(props_array[self.sph_index], dtype = float).flatten()
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
                        props_array_vtk.point_data['Specific Heat'] = np.array(props_array[self.sph_index], dtype = float).flatten()
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
    
    def plot_matrix(self,matrix=None,save_me:bool=False, name_file_dir = None, vmax_perct=99,vmin_perct=1,cmap = 'Reds'):
        plt.figure()
        if matrix is None :
            # Calculate the percentile values
            q1 = np.percentile(self.k, vmin_perct)
            q3 = np.percentile(self.k, vmax_perct)
            plt.imshow(self.k,cmap=cmap,vmin=q1, vmax=q3)
        else :
            q1 = np.percentile(matrix, vmin_perct)
            q3 = np.percentile(matrix, vmax_perct)
            plt.imshow(matrix,cmap=cmap,vmin=q1, vmax=q3)
        plt.xlabel('Network Node Number')
        plt.ylabel('Network Node Number')
        plt.title('Node connection network')
        plt.colorbar()
        if save_me:
            if name_file_dir is None:
                name_file_dir = os.getcwd()
            plt.savefig(name_file_dir+'Network_Connection_Matrix.png')
            plt.close()
        else :
            plt.show()
    def plot_Full_Graph(self,G_full, graph_layout='spring',save_me:bool=False, name_file_dir = None):
        if graph_layout == 'spring':
            pos = nx.spring_layout(G_full, k=0.5, iterations = 100)
        elif graph_layout == 'circular':
            pos = nx.circular_layout(G_full)
        elif graph_layout == 'shell':
            pos = nx.shell_layout(G_full)
        elif graph_layout == 'spectral':
            pos = nx.spectral_layout(G_full)
        else:
            raise ValueError(f"Invalid graph type: {graph_layout}")
        plt.figure(figsize=(10, 10))  # Set the figure size to 10x10 inches
        nx.draw(G_full, pos, with_labels=True, node_color='lightblue', edge_color='black', node_size = 800, font_size = 16)
        if save_me:
            if name_file_dir is None:
                name_file_dir = os.getcwd()
            plt.savefig(name_file_dir+'Network_Plot.png')
            plt.close()
        else :
            plt.show()

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


