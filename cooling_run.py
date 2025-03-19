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

diff_val = 31.536 # Average diffusivity value
temp_grad_base = 30/1000

magma_prop_dict = {'Temperature': 1100,
                        'Lithology': 'basalt_melt',
                        'Porosity': 0, #Porosity of the rock for calculation of carbon emissions
                        'Density': 2850, #kg/m3
                        'Specific Heat': 850, 
                        'Latent Heat': 4e5,
                        'TOC':0} #wt%
magma_prop_dict2 = {'Temperature': 1100,
                        'Lithology': 'basalt_melt',
                        'Porosity': 0, #Porosity of the rock for calculation of carbon emissions
                        'Density': 2850, #kg/m3
                        'Specific Heat': 850+ (4e5/(1175-800)), 
                        'Latent Heat': 3.8e5,
                        'TOC':0} #wt%
sc_latent_heat = sill_controls(x,y, dx, dy, T_liquidus=1175,T_solidus=800,
                   magma_prop_dict=magma_prop_dict,
                   include_external_heat=True,
                   calculate_closest_sill=False)

sc_nolatent_heat = sill_controls(x,y, dx, dy, T_liquidus=1175,T_solidus=800,
                   magma_prop_dict=magma_prop_dict,
                   include_external_heat=False,
                   calculate_closest_sill=False)
sc_nolatent_heat2 = sill_controls(x,y, dx, dy, T_liquidus=1175,T_solidus=800,
                   magma_prop_dict=magma_prop_dict2,
                   include_external_heat=False,
                   calculate_closest_sill=False)

file_path_dir = 'sillcubes/'
footnote1 = 'LH_2'
footnote2 = 'noLH_2'
footnote3 = 'noLHCP2_2'
post_cooling_time = 30000 #years
fluxy_list = [int(3e9), int(3e8), int(3e7), int(3*10**(7.5)), int(3*10**(8.5))]
iter_list = [0, 1, 2, 3, 4]
z_index_list = [191, 284, 300, 493, 506]
pairs = itertools.product(iter_list, z_index_list, fluxy_list)
saving_factor = [100]

'''
redo_flux = [int(3e9), int(3*10**8.5), int(3e8), int(3e8)]
redo_iter = [3, 0, 1, 2]
redo_flux = np.repeat(redo_flux, 5)
redo_iter = np.repeat(redo_iter, 5)
z_index = [191, 284, 300, 493, 506]
tiled_z = np.tile(z_index, 5)
pairs = zip(redo_iter, tiled_z, redo_flux)
'''
Parallel(n_jobs = 30)(delayed(util.cooler)(iter_1, z_index_1, fluxy_1,sc=sc_latent_heat,diff_val=diff_val,
                                      temp_grad_base = temp_grad_base,
                                      file_path_dir=file_path_dir, post_cooling_time = post_cooling_time,
                                      x=x,y=y,dx=dx,dy=dy, saving_factor = saving_factor) for iter_1, z_index_1, fluxy_1 in pairs)


### Run this only once for a model set###
#sc.generate_sill_2D_slices(fluxy_list,iter_list,z_index_list)
'''

Parallel(n_jobs = 30)(delayed(util.cooler)(iter_1, z_index_1, fluxy_1,sc=sc_latent_heat,diff_val=diff_val,
                                      temp_grad_base = temp_grad_base,
                                      file_path_dir=file_path_dir, post_cooling_time = post_cooling_time,
                                      x=x,y=y,dx=dx,dy=dy, save_dir_footnote=footnote1, saving_factor = saving_factor) for iter_1, z_index_1, fluxy_1 in pairs)


Parallel(n_jobs = 30)(delayed(util.cooler)(iter_1, z_index_1, fluxy_1,sc=sc_nolatent_heat,diff_val=diff_val,
                                      temp_grad_base = temp_grad_base,
                                      file_path_dir=file_path_dir, post_cooling_time = post_cooling_time,
                                      x=x,y=y,dx=dx,dy=dy, save_dir_footnote=footnote2, saving_factor = saving_factor) for iter_1, z_index_1, fluxy_1 in pairs)


Parallel(n_jobs = 30)(delayed(util.cooler)(iter_1, z_index_1, fluxy_1,sc=sc_nolatent_heat2,diff_val=diff_val,
                                      temp_grad_base = temp_grad_base,
                                      file_path_dir=file_path_dir, post_cooling_time = post_cooling_time,
                                      x=x,y=y,dx=dx,dy=dy, save_dir_footnote=footnote3, saving_factor = saving_factor) for iter_1, z_index_1, fluxy_1 in pairs)
'''
#Parallel(n_jobs = 30)(delayed(cooler)(iter2, z_indexs, fluxy) for z_indexs in z_index)

#cooler(0, 191, int(3e9))

#Paraview command - 
#inputs[0].PointData['Temperature']-inputs[1].PointData['Temperature']