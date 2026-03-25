import numpy as np
import scipy as scp
import pyvista as pv
import pandas as pd
import os
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
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
def validator (n_sills, emplace_params):
    return bool(int(n_sills) == len(emplace_params[0,:]))

def cubemaker(tot_volume, flux, x, y, z, dx, dy, maturation_time, save_dir, sc = None, lat_range = None, thickness_range = None, aspect_ratio = None, depth_range = None, shape = None, depth_function = None, lat_function = None, dims_function = None, emplace_dike = False, orientations = None, k = 31.536):
    def int_maker(sillcube):
        for i in tqdm(range(sillcube.shape[0]), desc = "Creating ints"):
            layer = sillcube[i]
            mask = (layer!=-1) & (layer!='-1')
            #if not np.any(mask):
            #    continue
            layer_vals = layer[mask]
            new_vals = []
            for val in layer_vals:
                ele = str(val)
                clean = ele[1:]
                clean = clean.split('s')[0]
                clean = clean.split('.')[0]
                #print(clean)
                try:

                    new_vals.append(int(clean))
                except ValueError:
                    new_vals.append(0)
            layer[mask] = new_vals
            sillcube[i] = layer
        return sillcube

    n_sills_array = []
    #Initializing time_steps
    
    dt = np.round((min(dx,dy)**2)/(5*k),3)

    if thickness_range is None:
        ###Setting up sill dimensions and locations###
        min_thickness = 100 #m
        max_thickness = 600 #m
        sd_thickness = 500
        thickness_range = [min_thickness, max_thickness, sd_thickness]

    if aspect_ratio is None:
        mar = 19.23
        sar = 9.74
        aspect_ratio = [mar, sar]

    if depth_range is None:
        min_emplacement = 500 #m
        max_emplacement = 4000 #m
        sd_emplacement = 5000
        depth_range = [min_emplacement, max_emplacement, sd_emplacement]
    n_sills = 200000


    shape = 'elli' if shape is None else shape
    

    thermal_mat_time = (maturation_time//dt+1)*(dt)
    print(f'Thermal maturation time is {thermal_mat_time}')
    model_time = tot_volume/flux
    cooling_time = int(1e6//dt)*dt

    
    z_range = [0, z, z//3]



    #print(f'Building cube for {tot_volume[l]}: Cube {l} of {len(tot_volume)}')
    phase_times = np.array([thermal_mat_time, model_time, cooling_time])
    time_steps = np.arange(0, np.sum(phase_times), dt)
    print(f'Length of time_steps:{len(time_steps)}')
    print(f'Total model time: {time_steps[-1]}')
    if lat_range is None:
        lat_range = [x//3, 2*x//3, x//6] #Min, max, sd
    n_val = 0
    sillcube, n_sills1, emplacement_params = sc.build_sillcube(z, dt, thickness_range, aspect_ratio, depth_range, z_range, lat_range, phase_times, tot_volume, flux, n_sills, shape, depth_function, lat_function, dims_function, emplace_dike, orientations)
    if validator(n_sills1, emplacement_params):
        print('sillcube built')
    else:
        while not validator(n_sills1, emplacement_params) or n_val<5:
            print(f'Sillcube validation failed for {flux:.3e} and {tot_volume:.3e}... Rebuilding')
            print(f'n_sills is {n_sills} while emplace_params is {len(emplacement_params[:,0])}')
            del sillcube
            sillcube, n_sills1, emplacement_params = sc.build_sillcube(z, dt, thickness_range, aspect_ratio, depth_range, z_range, lat_range, phase_times, tot_volume, flux, n_sills, shape, depth_function, lat_function, dims_function, emplace_dike, orientations)
            n_val+=1
        #pdb.set_trace()
    n_sills_array.append(int(n_sills1))
    os.makedirs(save_dir+'/'+str(format(flux,'.3e')), exist_ok=True)
    np.save(save_dir+'/'+str(format(flux,'.3e'))+'/sillcube'+str(np.round(tot_volume, 2)), sillcube)
    print('Saved numpy binary')
    sillcube[sillcube==''] = -1
    sillcube = int_maker(sillcube)
    sillcube = sillcube.astype(int)
    #pdb.set_trace()
    grid = pv.ImageData()
    grid.dimensions = sillcube.shape
    grid.point_data["sillcube"] = sillcube.flatten(order="F")
    grid.save(save_dir+'/'+str(format(flux,'.3e'))+'/sillcube'+str(tot_volume)+'.vtk')
    emplace_frame = pd.DataFrame(np.transpose(emplacement_params), columns=['empl_times', 'empl_heights', 'x_space', 'width', 'thickness'])
    emplace_frame.to_csv(save_dir+'/'+str(format(flux,'.3e'))+'/emplacement_params'+str(tot_volume)+'.csv')
    return n_sills_array


def cooler(iter, z_index, flux, lat_range = None, sc=None,diff_val=31.536,temp_grad_base = 30/1000,
           file_path_dir='sillcubes/', post_cooling_time = 30000,saving_factor = [100], x=None,y=None,dx=None,dy=None, rock_prop_dict = None, save_dir_footnote = None):
    '''
    temp_grad_base == 30/1000 # Temperature gradient at the base of the modeled section (in C/m)
    k_val = Typical value of thermal diffusivity in the model
    post_cooling_time = 60000 yr ## Cooling time past the last sill emplacement

    x = 300000 #m - Horizontal extent of the crust
    y = 12000 #m - Vertical thickness of the crust
    dx = dz = 50 #m node spacing in x-direction
    dy = 50 #m node spacing in y-direction
    sc=None -- Sill control class
    '''
    a = int(y//dy) #Number of rows
    b = int(x//dx) #Number of columns
    props_array_vtk = pv.read(file_path_dir+'initial_silli_state_properties.vtk')
    props_array = props_array_vtk.point_data['data'].reshape(props_array_vtk.dimensions)
    props_array = np.array(props_array, dtype = object)
    props_array[sc.Temp_index] = np.array(props_array[sc.Temp_index], dtype = float)

    props_array[sc.dense_index] = np.array(props_array[sc.dense_index], dtype = float)
    props_array[sc.poros_index] = np.array(props_array[sc.poros_index], dtype = float)
    props_array[sc.TOC_index] = np.array(props_array[sc.TOC_index], dtype = float)
    try:
        props_array[sc.sph_index] = np.array(props_array[sc.sph_index], dtype = float)
    except:
        print("Properties Array does not have a specific heat index. Creating one from properties dictionary...")
        specific_heat = np.vectorize(
                        lambda rt: rock_prop_dict[rt]['Specific Heat'], 
                        otypes=[float]  # Ensure output is float
                    )(props_array[sc.rock_index])
        #specific_heat = np.zeros((a,b), dtype = float)
        #for i in range(a):
        #    for j in range(b):
        #        specific_heat[i,j] = rock_prop_dict[props_array[sc.rock_index][i,j]]['Specific Heat']
        #        if specific_heat[i,j] == "None":
        #            raise ValueError("specific_heat is None")
        try:
            props_array[sc.sph_index] = specific_heat
        except:
            print("Exception occurred and properties array needs to have sepcific heat index added. Adding...")
            props_array =  np.append(props_array, specific_heat[np.newaxis,:,:], axis = 0)
            
    #Initializing diffusivity field
    k = sc.sill_controls_get_k(props_array[sc.Temp_index], props_array[sc.rock_index], props_array[sc.dense_index], dy)
    dt = np.round((min(dx,dy)**2)/(5*np.max(k)),3)
    dt_change_factor = 1 ## For change in dt after emplacement of the last sill
    print(f'Time step: {dt} years')

    #Initializing boundary heat flux
    q = k[-1,:]*temp_grad_base
    load_dir = file_path_dir+str(format(flux, '.3e'))+'/'+str(lat_range) if lat_range is not None else file_path_dir+str(format(flux, '.3e'))

    data = pv.read(file_path_dir+'/initial_silli_state_carbon.vtk')
    
    if rock_prop_dict is None:
        rock_prop_dict = sc.rock_prop_dict


    props_array_vtk = pv.read(file_path_dir+'initial_silli_state_properties.vtk')
    props_array = props_array_vtk.point_data['data'].reshape(props_array_vtk.dimensions)
    props_array = np.array(props_array, dtype = object)
    props_array[sc.Temp_index] = np.array(props_array[sc.Temp_index], dtype = float)

    props_array[sc.dense_index] = np.array(props_array[sc.dense_index], dtype = float)
    props_array[sc.poros_index] = np.array(props_array[sc.poros_index], dtype = float)
    props_array[sc.TOC_index] = np.array(props_array[sc.TOC_index], dtype = float)
    try:
        props_array[sc.sph_index] = np.array(props_array[sc.sph_index], dtype = float)
    except:
        print("Properties Array does not have a specific heat index. Creating one from properties dictionary...")
        specific_heat = np.vectorize(
                        lambda rt: rock_prop_dict[rt]['Specific Heat'], 
                        otypes=[float]  # Ensure output is float
                    )(props_array[sc.rock_index])
        #specific_heat = np.zeros((a,b), dtype = float)
        #for i in range(a):
        #    for j in range(b):
        #        specific_heat[i,j] = rock_prop_dict[props_array[sc.rock_index][i,j]]['Specific Heat']
        #        if specific_heat[i,j] == "None":
        #            raise ValueError("specific_heat is None")
        try:
            props_array[sc.sph_index] = specific_heat
        except:
            print("Exception occurred and properties array needs to have sepcific heat index added. Adding...")
            props_array =  np.append(props_array, specific_heat[np.newaxis,:,:], axis = 0)
    W_vtk = pv.read(file_path_dir+'W_data.vtk')
    W_silli = W_vtk.point_data['data'].reshape(W_vtk.dimensions)

    RCO2_silli = data.point_data['RCO2_silli'].reshape(data.dimensions[0], data.dimensions[1])
    RCO2_silli = RCO2_silli.reshape(data.dimensions[0], data.dimensions[1])/props_array[sc.dense_index]
    Rom_silli = data.point_data['Rom_silli'].reshape(data.dimensions[0], data.dimensions[1])
    percRo_silli = data.point_data['percRo_silli'].reshape(data.dimensions[0], data.dimensions[1])
    curr_TOC_silli = data.point_data['curr_TOC_silli'].reshape(data.dimensions[0], data.dimensions[1])

    n_sills_dataframe = pd.read_csv(load_dir+'/n_sills.csv')
    current_time = np.load(file_path_dir+'curr_time.npy')
    n_sills = n_sills_dataframe['n_sills'][iter]
    volumes = float(n_sills_dataframe['volumes'][iter])

    sillsquare = np.load(load_dir+'/slice_volumes/sillcube'+str(volumes)+'_'+str(z_index)+'.npy', allow_pickle=True)
    
    emplacement_params = pd.read_csv(load_dir+'/emplacement_params'+str(volumes)+'.csv')
    empl_times = np.array(emplacement_params['empl_times'])



    #RCO2_vtk = pv.read('sillcubes/RCO2.vtk')
    tot_RCO2 = []#list(pd.read_csv('sillcubes/tot_RCO2.csv'))
    carbon_model_params = [tot_RCO2, props_array, RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli]
    end_time = np.array(empl_times)[-1]+post_cooling_time+dt
    print(f'End time is {end_time}')
    time_steps1 = np.arange(current_time,np.array(empl_times)[-1]+100*dt,dt)
    time_steps2 = np.arange(np.array(empl_times)[-1]+100*dt, end_time+dt, dt*dt_change_factor)
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
            time_index = np.where(min_diff==np.abs(time_steps - empl_times[j]))[0][0]
            if min_diff<dt:
                empl_times[j] = time_steps[time_index]
                print(f'Emplacement time for sill {j} has been readjusted and is now at {np.where(time_steps==empl_times[j])[0]}')
        else:
            print(f'Sill {j} is at {np.where(time_steps==empl_times[j])[0]}')
    if save_dir_footnote is None:
        dir_save = file_path_dir+str(format(flux, '.3e'))+'/'+str(lat_range)+'/'+str(format(volumes, '.3e'))+'/'+str(z_index) if lat_range is not None else file_path_dir+str(format(flux, '.3e'))+'/'+str(format(volumes, '.3e'))+'/'+str(z_index)
    else:
        dir_save = file_path_dir+str(format(flux, '.3e'))+'/'+str(lat_range)+'/'+str(format(volumes, '.3e'))+str(save_dir_footnote)+'/'+str(z_index) if lat_range is not None else file_path_dir+str(format(flux, '.3e'))+'/'+str(format(volumes, '.3e'))+str(save_dir_footnote)+'/'+str(z_index)
    os.makedirs(dir_save, exist_ok = True)
    timeframe = pd.DataFrame(time_steps[1:], columns=['time_steps'])
    #timeframe.to_csv(dir_save+'/times.csv')

    current_time = np.round(current_time, 3)
    carbon_model_params = sc.emplace_sills(props_array, n_sills, 'conv smooth', time_steps, current_time, sillsquare, carbon_model_params, empl_times, volume_params, z_index, saving_factor=saving_factor,model = 'silli', q=q, save_dir = dir_save)
    tot_RCO2 = carbon_model_params[0]
    timeframe['tot_RCO2'] = tot_RCO2
    timeframe['melt10'] = carbon_model_params[1][1:]
    timeframe['melt50'] = carbon_model_params[2][1:]
    timeframe['area_sills'] = carbon_model_params[4][1:]
    timeframe['tot_solidus'] = carbon_model_params[3][1:]
    timeframe.to_csv(dir_save+'/times.csv')


###Write a custom diffusivity function here####