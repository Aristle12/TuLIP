import numpy as np
from TuLIP import sill_controls
from numba import set_num_threads
from joblib import Parallel, delayed
import itertools
import utilities as util

x = 300000 #m - Horizontal extent of the crust
y = 12000 #m - Vertical thickness of the crust
dx = dz = 50 #m node spacing in x-direction
dy = 50 #m node spacing in y-direction

k_val = 31.536 # Average diffusivity value
temp_grad_base = 30/1000

sc = sill_controls(x,y, dx, dy, T_liquidus=1175,
                   k_val=k_val,
                   include_external_heat=True,
                   calculate_closest_sill=True)

file_path_dir = sc.sill_cube_dir


fluxy_list = [int(3e9)]#, int(3e8), int(3e7), int(3*10**(7.5)), int(3*10**(8.5))]
iter_list = [0]#, 1, 2, 3, 4]
z_index_list = [191, 284, 300, 493, 506]
pairs = itertools.product(iter_list, z_index_list, fluxy_list)

'''
redo_flux = [int(3e9), int(3*10**8.5), int(3e8), int(3e8)]
redo_iter = [3, 0, 1, 2]
redo_flux = np.repeat(redo_flux, 5)
redo_iter = np.repeat(redo_iter, 5)
z_index = [191, 284, 300, 493, 506]
tiled_z = np.tile(z_index, 5)
pairs = zip(redo_iter, tiled_z, redo_flux)
'''

### Run this only once for a model set
sc.generate_sill_2D_slices(fluxy_list,iter_list,z_index_list)

Parallel(n_jobs = 30)(delayed(util.cooler)(iter_1, z_index_1, fluxy_1,sc=sc,k_val=k_val,
                                      temp_grad_base = temp_grad_base,
                                      file_path_dir=file_path_dir,
                                      x=x,y=y,dx=dx,dy=dy) for iter_1, z_index_1, fluxy_1 in pairs)
#Parallel(n_jobs = 30)(delayed(cooler)(iter2, z_indexs, fluxy) for z_indexs in z_index)

#cooler(0, 191, int(3e9))