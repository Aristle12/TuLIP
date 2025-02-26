import numpy as np
from TuLIP import sill_controls
import pandas as pd
from joblib import Parallel, delayed
import utilities as util
import os


flux = int(3e9) #km3/yr
save_dir = 'sillcubes/'+str(format(flux, '.3e'))+'/'

os.makedirs(save_dir, exist_ok = True)

maturation_time = int(3e6)

#Dimensions of the 2D grid
x = 300000 #m - Horizontal extent of the crust
y = 12000 #m - Vertical thickness of the crust
z = 30000 #m - Third dimension for cube

dx = 50 #m node spacing in x-direction
dy = 50 #m node spacing in y-direction


volume = x*4000*z



tot_volume_start = 0.05*volume
tot_volume_end = 0.175*volume
tot_volumes = np.arange(tot_volume_start, tot_volume_end, 0.025*volume)
print(tot_volumes)
sc = sill_controls(x =x,
    y = y,
    dx = dx, 
    dy = dy)


n_sills_array = Parallel(n_jobs = 2)(delayed(util.cubemaker)(tot_volume, flux=flux, x=x, y=y, z=z, dx=dx, dy=dy, maturation_time = maturation_time, save_dir = save_dir, sc = sc) for tot_volume in tot_volumes)
#n_sills_total = pd.read_csv(save_dir+'/n_sills.csv')
# Append new data to the DataFrame
#new_data = pd.DataFrame({
#    'volumes': tot_volumes,
#    'n_sills': n_sills_array
#})

## Concatenate the existing DataFrame with the new data
#n_sills_total = pd.concat([n_sills_total, new_data], ignore_index=True)
#n_sills_total.to_csv(save_dir+'/n_sills.csv')
pd.DataFrame({'n_sills': n_sills_array, 'volumes': tot_volumes}).to_csv(save_dir+'/n_sills.csv')
