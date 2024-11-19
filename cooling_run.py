import numpy as np
import pyvista as pv
from TuLIP import sill_controls
import pandas as pd
import os
sc = sill_controls()

iter = 0

#Dimensions of the 2D grid
x = 300000 #m - Horizontal extent of the crust
y = 8000 #m - Vertical thickness of the crust
z = 30000 #m - Third dimension for cube



dx = 50 #m node spacing in x-direction
dy = 50 #m node spacing in y-direction

a = int(y//dy) #Number of rows
b = int(x//dx) #Number of columns


#Temp at the surface
T_surf = 0 #deg C

#Magmatic temperature
T_mag = 1000 #deg C

#Initializing diffusivity field
k = np.ones((a,b))*31.536 #m2/yr

dt = (min(dx,dy)**2)/(5*np.max(k))/10
print(f'Time step: {dt} years')
#Shape of the sills
shape = 'elli'

q = k[-1,:]*(30/1000)

data = pv.read('sillcubes/initial_silli_state_carbon.vtk')
carbon_params = data.point_data['data'].reshape(data.dimensions)

props_array_vtk = pv.read('sillcubes/initial_silli_state_properties.vtk')
props_array = props_array_vtk.point_data['data'].reshape(props_array_vtk.dimensions)
props_array = np.array(props_array, dtype = object)
props_array[sc.Temp_index] = np.array(props_array[sc.Temp_index], dtype = float)
props_array[sc.dense_index] = np.array(props_array[sc.dense_index], dtype = float)
props_array[sc.poros_index] = np.array(props_array[sc.poros_index], dtype = float)
props_array[sc.TOC_index] = np.array(props_array[sc.TOC_index], dtype = float)

W_vtk = pv.read('sillcubes/W_data.vtk')
W_silli = W_vtk.point_data['data'].reshape(W_vtk.dimensions)

RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli = carbon_params

n_sills_dataframe = pd.read_csv('sillcubes/n_sills.csv')
current_time = n_sills_dataframe['curr_time'][iter]
n_sills = n_sills_dataframe['n_sills'][iter]
volumes = n_sills_dataframe['volumes'][iter]

sillcube = np.load('sillcubes/sillcube'+str(volumes)+'.npy', allow_pickle=True)
emplacement_params = pd.read_csv('sillcubes/emplacement_params'+str(volumes)+'.csv')
empl_times = emplacement_params['empl_times']
empl_heights = emplacement_params['empl_heights']
x_space = emplacement_params['x_space']
width = emplacement_params['width']
thickness = emplacement_params['thickness']

emplacement_params = [empl_times, empl_heights, x_space, width, thickness]
#RCO2_vtk = pv.read('sillcubes/RCO2.vtk')
tot_RCO2 = []# RCO2_vtk.point_data['data'].reshape(RCO2_vtk.dimensions)
carbon_model_params = [tot_RCO2, props_array, RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli]

end_time = np.array(empl_times)[-1]+(50*dt)
print(f'End time is {end_time}')
time_steps = np.arange(current_time,end_time,dt)

props_total_array, carbon_model_params = sc.emplace_sills(props_array, k, dx, dy, dt, n_sills, b//2, 'conv smooth', time_steps, current_time, sillcube, carbon_model_params, emplacement_params, model = 'silli', q=q)


factor = 10
saving_props_array = np.empty((props_total_array.shape[0]/factor,a,b))
saving_time_steps = np.empty(len(time_steps)/10)
T_field = np.empty((len(saving_time_steps,a,b)))
density_field = np.empty_like(T_field)
porosity_field = np.empty_like(T_field)
TOC_field = np.empty_like(T_field)
rock_field = np.empty_like(T_field)

for i in range(len(saving_time_steps)):
    if i%factor ==0:
        saving_time_steps[i/factor] = time_steps[i]
        saving_props_array[i/factor] = props_total_array[i]
        T_field[i/factor] = props_total_array[i,sc.Temp_index]
        density_field[i/factor] = props_total_array[i,sc.dense_index]
        porosity_field[i/factor] = props_total_array[i,sc.poros_index]
        TOC_field[i/factor] = props_total_array[i,sc.TOC_index]
        rock_field[i/factor] = props_total_array[i, sc.rock_index]

T_data = pv.ImageData(dims=(x,y), spacing = (dx,dy))
density_data = pv.ImageData(dims=(x,y), spacing = (dx,dy))
porosity_data = pv.ImageData(dims=(x,y), spacing = (dx,dy))
TOC_data = pv.ImageData(dims=(x,y), spacing = (dx,dy))
rock_data = pv.ImageData(dims=(x,y), spacing = (dx,dy))

for t in saving_time_steps:
    T_data.point_data[f'data_{t}'] = T_field[t].flatten
    density_data.point_data[f'data_{t}'] = density_field[t].flatten
    porosity_data.point_data[f'data_{t}'] = porosity_field[t].flatten
    TOC_data.point_data[f'data_{t}'] = TOC_field[t].flatten
    rock_data.point_data[f'data_{t}'] = rock_field[t].flatten


T_data.time_values = saving_time_steps
density_data.time_values = saving_time_steps
porosity_data.time_values = saving_time_steps
TOC_data.time_values = saving_time_steps
rock_data.time_values = saving_time_steps

dir = 'sillcube/'+str(volumes)
os.makedirs(dir, exist_ok=True)
T_data.save(dir+'T_data.vtk')
density_data.save(dir+'density_data.vtk')
porosity_data.save(dir+'porosity_data.vtk')
TOC_data.save(dir+'TOC_data.vtk')
rock_data.save(dir+'rock_data.vtk')
