import numpy as np
import pyvista as pv
from TuLIP import sill_controls
import pandas as pd
import os
from numba import set_num_threads
from joblib import Parallel, delayed
import itertools
import pdb


x = 300000 #m - Horizontal extent of the crust
y = 12000 #m - Vertical thickness of the crust
dx = dz = 50 #m node spacing in x-direction
dy = 50 #m node spacing in y-direction

a = int(y//dy) #Number of rows
b = int(x//dx) #Number of columns
k = np.ones((a,b))*31.536 #m2/yr


sc = sill_controls(x,y, dx, dy, T_liquidus=1175,
                   include_external_heat=True,
                   calculate_closest_sill=True)



def cooler(iter, z_index, flux):
    def truncate(number):
        # Convert the number to a string
        number_str = str(number)
        
        # Find the position of the decimal point
        decimal_index = number_str.find('.')
        
        # If there's no decimal point, return the number as is
        if decimal_index == -1:
            return float(number_str)
        
        # Truncate the string to three decimal places
        truncated_str = number_str[:decimal_index + 4]
        
        # Convert the truncated string back to a float
        return float(truncated_str)
    #Dimensions of the 2D grid
   
    z = 30000 #m - Third dimension for cube


    #Initializing diffusivity field


    dt = np.round((min(dx,dy)**2)/(5*np.max(k)),3)
    print(f'Time step: {dt} years')
    #Shape of the sills
    shape = 'elli'

    q = k[-1,:]*(30/1000)

    load_dir = 'sillcubes/'+str(format(flux, '.3e'))

    data = pv.read('sillcubes/initial_silli_state_carbon.vtk')
    

    props_array_vtk = pv.read('sillcubes/initial_silli_state_properties.vtk')
    props_array = props_array_vtk.point_data['data'].reshape(props_array_vtk.dimensions)
    props_array = np.array(props_array, dtype = object)
    props_array[sc.Temp_index] = np.array(props_array[sc.Temp_index], dtype = float)
    props_array[sc.dense_index] = np.array(props_array[sc.dense_index], dtype = float)
    props_array[sc.poros_index] = np.array(props_array[sc.poros_index], dtype = float)
    props_array[sc.TOC_index] = np.array(props_array[sc.TOC_index], dtype = float)

    W_vtk = pv.read('sillcubes/W_data.vtk')
    W_silli = W_vtk.point_data['data'].reshape(W_vtk.dimensions)

    RCO2_silli = data.point_data['RCO2_silli'].reshape(data.dimensions[0], data.dimensions[1])
    RCO2_silli = RCO2_silli.reshape(data.dimensions[0], data.dimensions[1])/props_array[sc.dense_index]
    Rom_silli = data.point_data['Rom_silli'].reshape(data.dimensions[0], data.dimensions[1])
    percRo_silli = data.point_data['percRo_silli'].reshape(data.dimensions[0], data.dimensions[1])
    curr_TOC_silli = data.point_data['curr_TOC_silli'].reshape(data.dimensions[0], data.dimensions[1])

    n_sills_dataframe = pd.read_csv(load_dir+'/n_sills.csv')
    current_time = np.load('sillcubes/curr_time.npy')
    n_sills = n_sills_dataframe['n_sills'][iter]
    volumes = float(n_sills_dataframe['volumes'][iter])

    sillsquare = np.load(load_dir+'/slice_volumes/sillcube'+str(volumes)+'_'+str(z_index)+'.npy', allow_pickle=True)
    
    emplacement_params = pd.read_csv(load_dir+'/emplacement_params'+str(volumes)+'.csv')
    empl_times = np.array(emplacement_params['empl_times'])



    #RCO2_vtk = pv.read('sillcubes/RCO2.vtk')
    tot_RCO2 = []#list(pd.read_csv('sillcubes/tot_RCO2.csv'))
    carbon_model_params = [tot_RCO2, props_array, RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli]
    post_cooling_time = 60000 #years
    end_time = np.array(empl_times)[-1]+post_cooling_time+dt
    print(f'End time is {end_time}')
    time_steps1 = np.arange(current_time,np.array(empl_times)[-1],dt)
    time_steps2 = np.arange(np.array(empl_times)[-1], end_time+dt, dt)
    time_steps = np.append(time_steps1, time_steps2)

    if time_steps[-1]<np.array(empl_times)[-1]:
        raise ValueError(f'Time history stops at {time_steps[-1]} before last emplacement {empl_times[-1]} for {flux, volumes} and {z_index}')
    volume_params = [flux, volumes]

    #empl_times = np.array([truncate(number) for number in empl_times])
    #time_steps = np.array([truncate(number) for number in time_steps])
    for j in range(len(empl_times)):
        if len(np.where(time_steps==empl_times[j])[0])==0:
            min_diff = np.min(np.abs(time_steps - empl_times[j]))
            print(f'Sill {j} is not in time_steps and difference to closest time step is {min_diff:.2e}')
            time_index = np.where(min_diff==np.abs(time_steps - empl_times[j]))[0]
            if min_diff<dt:
                empl_times[j] = time_steps[time_index]
                print(f'Emplacement time for sill {j} has been readjusted and is now at {np.where(time_steps==empl_times[j])[0]}')
        else:
            print(f'Sill {j} is at {np.where(time_steps==empl_times[j])[0]}')
    dir_save = 'sillcubes/'+str(format(flux, '.3e'))+'/'+str(format(volumes, '.3e'))+'/'+str(z_index)
    os.makedirs(dir_save, exist_ok = True)
    timeframe = pd.DataFrame(time_steps[1:], columns=['time_steps'])
    timeframe.to_csv(dir_save+'/times.csv')

    current_time = np.round(current_time, 3)
    carbon_model_params = sc.emplace_sills(props_array, n_sills, 'conv smooth', time_steps, current_time, sillsquare, carbon_model_params, empl_times, volume_params, z_index, saving_factor=[100],model = 'silli', q=q)
    tot_RCO2 = carbon_model_params[0]
    timeframe['tot_RCO2'] = tot_RCO2
    timeframe['melt10'] = carbon_model_params[1][1:]
    timeframe['melt50'] = carbon_model_params[2][1:]
    timeframe['area_sills'] = carbon_model_params[4][1:]
    timeframe['tot_solidus'] = carbon_model_params[3][1:]
    timeframe.to_csv(dir_save+'/times.csv')


x = 300000 #m - Horizontal extent of the crust
y = 8000 #m - Vertical thickness of the crust
z = 30000 #m - Third dimension for cube

fluxy = [int(3e9)]#, int(3e8), int(3e7), int(3*10**(7.5)), int(3*10**(8.5))]
#flux2 = [int(3*10**7.5), int(3*10**8.5)]

dx = dz = 50 #m node spacing in x-direction
dy = 50 #m node spacing in y-direction

a = int(y//dy) #Number of rows
b = int(x//dx) #Number of columns
c = int(z//dz) # Number of columns in z direction

iter = [3, 4]

iter2 = [0, 1, 2, 3, 4]
'''
redo_flux = [int(3e9), int(3*10**8.5), int(3e8), int(3e8)]
redo_iter = [3, 0, 1, 2]
redo_flux = np.repeat(redo_flux, 5)
redo_iter = np.repeat(redo_iter, 5)
z_index = [191, 284, 300, 493, 506]
tiled_z = np.tile(z_index, 5)
pairs = zip(redo_iter, tiled_z, redo_flux)
'''
factor = np.random.randint(1, int(0.9*c//2), 4)
#z_index = [c//2, c//2+factor[0], c//2+factor[1], c//2-factor[2], c//2-factor[3]]
z_index = [191, 284, 300, 493, 506]
pairs = itertools.product(iter2, z_index, fluxy)
'''
#pairs = pairs+pairs2
for flux in fluxy:
    load_dir = 'sillcubes/'+str(format(flux, '.3e'))
    os.makedirs(load_dir+'/slice_volumes', exist_ok=True)
    for filename in os.listdir(os.path.join(load_dir, 'slice_volumes')):
        file_path = os.path.join(load_dir, 'slice_volumes', filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    n_sills_dataframe = pd.read_csv(load_dir+'/n_sills.csv')
    current_time = np.load('sillcubes/curr_time.npy')
    for iters in iter2:
        volumes = float(n_sills_dataframe['volumes'][iters])

        sillcube = np.load(load_dir+'/sillcube'+str(volumes)+'.npy', allow_pickle=True)
        for z_indexs in z_index:
            sillsquare = sillcube[z_indexs]
            np.save(load_dir+'/slice_volumes/sillcube'+str(volumes)+'_'+str(z_indexs)+'.npy', sillsquare)
print(f'slices are {z_index}')
'''
Parallel(n_jobs = 30)(delayed(cooler)(iter, z_indexs, fluxy) for iter, z_indexs, fluxy in pairs)
#Parallel(n_jobs = 30)(delayed(cooler)(iter2, z_indexs, fluxy) for z_indexs in z_index)

#cooler(0, 191, int(3e9))