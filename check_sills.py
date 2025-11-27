import numpy as np
from TuLIP import sill_controls
import pandas as pd
import utilities
from joblib import Parallel, delayed
import itertools

x = 300000 #m - Horizontal extent of the crust
y = 12000 #m - Vertical thickness of the crust
dx = dz = 50 #m node spacing in x-direction
dy = 50 #m node spacing in y-direction

diff_val = 31.536 # Average diffusivity value
temp_grad_base = 30/1000

post_cooling_time = 30 #years

magma_prop_dict = {'Temperature': 1100,
                        'Lithology': 'basalt_melt',
                        'Porosity': 0, #Porosity of the rock for calculation of carbon emissions
                        'Density': 2850, #kg/m3
                        'Specific Heat': 850, 
                        'Latent Heat': 4e5,
                        'TOC':0} #wt%

sc_latent_heat = sill_controls(x,y, dx, dy, T_liquidus=1175,T_solidus=800,
                   magma_prop_dict=magma_prop_dict,
                   include_external_heat=True,
                   calculate_closest_sill=True)

file_path_dir = 'sillcubes/'

rock_prop_dict = {
                "shale":{
                    'Porosity':0.1,
                    'Density':2500,
                    'TOC':2,
                    'Specific Heat': 800
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

flux = int(3e9)
vol = pd.read_csv(file_path_dir+str(format(flux, '.3e')+'/n_sills.csv'))
volumes = vol['volumes']
iter = np.where(volumes==3.42e12)[0]
z_index_list2 = [191, 300]#[191, 284, 300, 493, 506]
fluxs = [int(3e9)]
pairs = itertools.product(iter, z_index_list2, fluxs)
saving_factor = [100]

Parallel(n_jobs = 2)(delayed(utilities.cooler)(iter_1, z_index_1, fluxy_1, sc=sc_latent_heat,diff_val=diff_val,
                                      temp_grad_base = temp_grad_base,
                                      file_path_dir=file_path_dir, post_cooling_time = post_cooling_time,
                                      x=x,y=y,dx=dx,dy=dy, rock_prop_dict = rock_prop_dict, save_dir_footnote = '', saving_factor = saving_factor) for iter_1, z_index_1, fluxy_1 in pairs)
