import numpy as np
import pyvista as pv
from TuLIP import sill_controls
import pandas as pd
import os
#from numba import set_num_threads
from joblib import Parallel, delayed
import itertools
import pdb

sc = sill_controls()

def cooler(load_dir, z_index, iter):
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
    #iter = 1
    flux = int(3e7)

    #Dimensions of the 2D grid
    x = 300000 #m - Horizontal extent of the crust
    y = 12000 #m - Vertical thickness of the crust
    z = 30000 #m - Third dimension for cube



    dx = dz = 50 #m node spacing in x-direction
    dy = 50 #m node spacing in y-direction

    a = int(y//dy) #Number of rows
    b = int(x//dx) #Number of columns


    #Temp at the surface
    T_surf = 0 #deg C

    #Magmatic temperature
    T_mag = 1000 #deg C

    #Initializing diffusivity field
    k = np.ones((a,b))*31.536 #m2/yr
    if flux==int(3e9):
        dt_factor = 10
    elif flux==int(3e8):
        dt_factor = 5
    elif flux==int(3e7):
        dt_factor = 1

    dt = (min(dx,dy)**2)/(5*np.max(k))/dt_factor
    print(f'Time step: {dt} years')
    #Shape of the sills
    shape = 'elli'

    q = k[-1,:]*(30/1000)

    

    data = pv.read('sillcubes/initial_silli_state_carbon.vtk')
    Hello

    props_array_vtk = pv.read('sillcubes/initial_silli_state_properties.vtk')
    props_array = props_array_vtk.point_data['data'].reshape(props_array_vtk.dimensions)
    props_array = np.array(props_array, dtype = object)
    props_array[sc.Temp_index] = np.array(props_array[sc.Temp_index], dtype = float)
    props_array[sc.dense_index] = np.array(props_array[sc.dense_index], dtype = float)
    props_array[sc.poros_index] = np.array(props_array[sc.poros_index], dtype = float)
    props_array[sc.TOC_index] = np.array(props_array[sc.TOC_index], dtype = float)

    W_vtk = pv.read('sillcubes/W_data.vtk')
    W_silli = W_vtk.point_data['data'].reshape(W_vtk.dimensions)

    RCO2_silli = data.point_data['RCO2_silli'].reshape(data.dimensions)
    Rom_silli = data.point_data['Rom_silli'].reshape(data.dimensions)
    percRo_silli = data.point_data['percRo_silli'].reshape(data.dimensions)
    curr_TOC_silli = data.point_data['curr_TOC_silli'].reshape(data.dimensions)

    n_sills_dataframe = pd.read_csv(load_dir+'/n_sills.csv')
    current_time = np.load('sillcubes/curr_time.npy')
    n_sills = n_sills_dataframe['n_sills'][iter]
    volumes = float(n_sills_dataframe['volumes'][iter])

    sillsquare = np.load(load_dir+'/slice_volumes/sillcube'+str(volumes)+'_'+str(z_index)+'.npy', allow_pickle=True)
    emplacement_params = pd.read_csv(load_dir+'/emplacement_params'+str(volumes)+'.csv')
    empl_times = emplacement_params['empl_times']
    empl_heights = emplacement_params['empl_heights']
    x_space = emplacement_params['x_space']
    width = emplacement_params['width']
    thickness = emplacement_params['thickness']



    #RCO2_vtk = pv.read('sillcubes/RCO2.vtk')
    tot_RCO2 = []#list(pd.read_csv('sillcubes/tot_RCO2.csv'))
    carbon_model_params = [tot_RCO2, props_array, RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli]
    post_cooling_time = 30000 #years
    end_time = np.array(empl_times)[-1]+post_cooling_time
    print(f'End time is {end_time}')
    time_steps1 = np.arange(current_time,np.array(empl_times)[-1],dt)
    time_steps2 = np.arange(np.array(empl_times)[-1]+(dt_factor*dt), end_time, dt_factor*dt)
    time_steps = np.append(time_steps1, time_steps2)

    if flux==int(3e7):
        print(f'Total time steps: {len(time_steps)}')

    volume_params = [flux, volumes]

    switch = False
    empl_times = np.array([truncate(number) for number in empl_times])
    emplacement_params = [empl_times, empl_heights, x_space, width, thickness]
    time_steps = np.array([truncate(number) for number in time_steps])
    for j in range(len(empl_times)):
        if len(np.where(time_steps==empl_times[j])[0])==0:
            print(f'Emplacement time step {j} is not in time_steps')
        else:
            print(f'Emplacement time step {j} is at {np.where(time_steps==empl_times[j])[0]}')

    dir_save = load_dir+str(format(volumes,'.3e'))+'/'+str(z_index)
    print(f'Saving at {dir_save}')
    #dir_save = load_dir+'/'+str(z_index)
    os.makedirs(dir_save, exist_ok = True)
    timeframe = pd.DataFrame(time_steps[1:], columns=['time_steps'])
    print(len(timeframe['time_steps']))
    timeframe.to_csv(dir_save+'/times.csv')

    current_time = np.round(current_time, 3)
    carbon_model_params = sc.emplace_sills(props_array, k, dx, dy, n_sills, 'conv smooth', time_steps, current_time, sillsquare, carbon_model_params, emplacement_params, volume_params, z_index, saving_factor=100,model = 'silli', q=q, save_dir=dir_save)
    tot_RCO2 = carbon_model_params[0]
    timeframe['tot_RCO2'] = tot_RCO2
    timeframe.to_csv(dir_save+'/times.csv')


x = 300000 #m - Horizontal extent of the crust
y = 8000 #m - Vertical thickness of the crust
z = 30000 #m - Third dimension for cube

dx = dz = 50 #m node spacing in x-direction
dy = 50 #m node spacing in y-direction

a = int(y//dy) #Number of rows
b = int(x//dx) #Number of columns
c = int(z//dz) # Number of columns in z direction

load_dirs = ['sillcubes/3.000e+07/middle60/']

#factor = np.random.randint(1, int(0.9*c//2), 4)
#z_index = [c//2, c//2+factor[0], c//2+factor[1], c//2-factor[2], c//2-factor[3]]
z_index = [89, 280, 300, 483, 555]
#z_index = [89,300]
itera = [0, 1, 2]

for load_dir in load_dirs:
    os.makedirs(load_dir+'/slice_volumes', exist_ok=True)
    for filename in os.listdir(os.path.join(load_dir, 'slice_volumes')):
        file_path = os.path.join(load_dir, 'slice_volumes', filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    n_sills_dataframe = pd.read_csv(load_dir+'/n_sills.csv')
    for iters in itera:
        volumes = float(n_sills_dataframe['volumes'][iters])
        sillcube = np.load(load_dir+'/sillcube'+str(volumes)+'.npy', allow_pickle=True)
        for z_indexs in z_index:
            sillsquare = sillcube[z_indexs]
            np.save(load_dir+'/slice_volumes/sillcube'+str(volumes)+'_'+str(z_indexs)+'.npy', sillsquare)

print(f'slices are {z_index}')

pairs = itertools.product(load_dirs, z_index, itera)



Parallel(n_jobs = 30)(delayed(cooler)(load_dir, z_indexs, iters) for load_dir, z_indexs, iters in pairs)
