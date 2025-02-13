import matplotlib.pyplot as plt
import numpy as np
from TuLIP import sill_controls
import pandas as pd
from tqdm import trange
import pdb
from numba import jit
import pyvista as pv
import os
from joblib import Parallel, delayed

#@jit
n_sills_array = []

flux = int(3e9)
save_dir = 'sillcubes/'+str(format(flux, '.3e'))+'/'

os.makedirs(save_dir, exist_ok = True)

def cubemaker(tot_volume):
    def int_maker(sillcube):
        for i in trange(sillcube.shape[0]):
            for j in range(sillcube.shape[1]):
                for g in range(sillcube.shape[2]):
                    if sillcube[i,j,g] != 0:
                        ele = str(sillcube[i,j,g])
                        sillcube[i,j,g] = ele[1:3] if 's' not in ele[1:3] else ele[1]
        return sillcube
    

    x = 300000 #m - Horizontal extent of the crust
    y = 12000 #m - Vertical thickness of the crust
    dx = 50 #m node spacing in x-direction
    dy = 50 #m node spacing in y-direction
    a = int(y//dy) #Number of rows
    b = int(x//dx) #Number of columns
    k = np.ones((a,b))*31.536 #m2/yr
    

    sc = sill_controls(x =x,
    y = y,
    dx = 50, 
    dy = 50, 
    )

    #Dimensions of the 2D grid

    z = 30000 #m - Third dimension for cube

    #Initializing diffusivity field
    k = np.ones((a,b))*31.536 #m2/yr
    dt = np.round((min(dx,dy)**2)/(5*np.max(k)),3)
    #Shape of the sills
    shape = 'elli'

    ###Setting up sill dimensions and locations###
    min_thickness = 100 #m
    max_thickness = 600 #m

    mar = 19.23
    sar = 9.74

    min_emplacement = 500 #m
    max_emplacement = 4000 #m
    n_sills = 2000


    
    

    thermal_mat_time = (int(3e6//dt)+1)*(dt)
    print(f'Thermal maturation time is {thermal_mat_time}')
    model_time = tot_volume/flux
    cooling_time = int(1e6//dt)*dt

    
    z_range = [0, z, z//3]



    #print(f'Building cube for {tot_volume[l]}: Cube {l} of {len(tot_volume)}')
    phase_times = np.array([thermal_mat_time, model_time, cooling_time])
    time_steps = np.arange(0, np.sum(phase_times), dt)
    print(f'Length of time_steps:{len(time_steps)}')
    lat_range = [x//3, 2*x//3, x//6]
    sillcube, n_sills1, emplacement_params = sc.build_sillcube(z, dt, [min_thickness, max_thickness, 500], [mar, sar], [min_emplacement, max_emplacement, 5000], z_range, lat_range, phase_times, tot_volume, flux, n_sills)
    print('sillcube built')
    #pdb.set_trace()
    n_sills_array.append(int(n_sills1))

    np.save(save_dir+'/sillcube'+str(np.round(tot_volume, 2)), sillcube)
    sillcube[sillcube==''] = 0
    sillcube = int_maker(sillcube)
    sillcube = sillcube.astype(int)
    #pdb.set_trace()
    grid = pv.ImageData()
    grid.dimensions = sillcube.shape
    grid.point_data["sillcube"] = sillcube.flatten(order="F")
    grid.save(save_dir+'/sillcube'+str(tot_volume)+'.vtk')
    emplace_frame = pd.DataFrame(np.transpose(emplacement_params), columns=['empl_times', 'empl_heights', 'x_space', 'width', 'thickness'])
    emplace_frame.to_csv(save_dir+'/emplacement_params'+str(tot_volume)+'.csv')
    return n_sills_array





#Dimensions of the 2D grid
x = 300000 #m - Horizontal extent of the crust
y = 12000 #m - Vertical thickness of the crust
z = 30000 #m - Third dimension for cube

dx = 50 #m node spacing in x-direction
dy = 50 #m node spacing in y-direction

a = int(y//dy) #Number of rows
b = int(x//dx) #Number of columns


volume = x*4000*z



tot_volume_start = 0.05*volume
tot_volume_end = 0.175*volume
tot_volumes = np.arange(tot_volume_start, tot_volume_end, 0.025*volume)
print(tot_volumes)


n_sills_array = Parallel(n_jobs = 2)(delayed(cubemaker)(tot_volume) for tot_volume in tot_volumes)
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
