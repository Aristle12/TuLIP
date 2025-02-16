from TuLIP import sill_controls
import os
import numpy as np
import pandas as pd
from tqdm import trange
import pyvista as pv


def int_maker(sillcube):
        for i in trange(sillcube.shape[0]):
            for j in range(sillcube.shape[1]):
                for g in range(sillcube.shape[2]):
                    if sillcube[i,j,g] != 0:
                        ele = str(sillcube[i,j,g])
                        sillcube[i,j,g] = ele[1:3] if 's' not in ele[1:3] else ele[1]
        return sillcube
    
n_sills_array = []

flux = int(3e9)
save_dir = 'sillcubes/'+str(format(flux, '.3e'))+'/'



os.makedirs(save_dir, exist_ok = True)


x = 300000
y = 4000
z = x//10
dx = 50
dy = 50

a = int(y//dy) #Number of rows
b = int(x//dx) #Number of columns
k = np.ones((a,b))*31.536 #m2/yr
dt = np.round((min(dx,dy)**2)/(5*np.max(k)),3)

min_thickness = 100 #m
max_thickness = 600 #m

mar = 19.23
sar = 9.74

min_emplacement = 500 #m
max_emplacement = 4000 #m
n_sills = 2000


tot_volume = 0.1*x*y*z  
    

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


sc = sill_controls(x, y, dx, dy)

sillcube , n_sills1, emplacement_params = sc.build_sillcube(z, dt, [min_thickness, max_thickness, 500], [mar, sar], [min_emplacement, max_emplacement, 5000], z_range, lat_range, phase_times, tot_volume, flux, n_sills, emplace_dike=True)



np.save(save_dir+'/sillcube'+str(np.round(tot_volume, 2)), sillcube)

emplace_frame = pd.DataFrame(np.transpose(emplacement_params), columns=['empl_times', 'empl_heights', 'x_space', 'width', 'thickness'])
emplace_frame.to_csv(save_dir+'/emplacement_params'+str(tot_volume)+'.csv')

sillcube[sillcube==''] = 0
sillcube = int_maker(sillcube)
sillcube = sillcube.astype(int)
#pdb.set_trace()
grid = pv.ImageData()
grid.dimensions = sillcube.shape
grid.point_data["sillcube"] = sillcube.flatten(order="F")
grid.save(save_dir+'/sillcube'+str(tot_volume)+'.vtk')