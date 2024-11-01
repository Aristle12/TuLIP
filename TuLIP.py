import numpy as np
import rule_functions as rool
import functions as cool
from tqdm import trange
import matplotlib.pyplot as plt
import carbon_emissions as emit
import seaborn as sns

class sill_controls:
    def __init__(self):
        ###Setting up the properties dictionary to translate properties to indices for 3D array###
        self.prop_dict = {'Temperature':0,
                    'Lithology':1,
                    'Porosity':2,
                    'Density':3,
                    'TOC':4}
        
        self.rev_prop_dict = {0:'Temperature',
                        1:'Lithology',
                        2:'Porosity',
                        3:'Density',
                        4:'TOC'
        }

        #Lithology dictionary to translate rock types into numerical codes for numpy arrays
        self.lith_plot_dict = {'granite':0,
                        'shale':1,
                        'sandstone':2,
                        'peridotite':3,
                        'basalt':4}
        self.Temp_index = self.prop_dict['Temperature']
        self.rock_index = self.prop_dict['Lithology']
        self.poros_index = self.prop_dict['Porosity']
        self.dense_index = self.prop_dict['Density']
        self.TOC_index = self.prop_dict['TOC']

        self.magma_prop_dict = {'Temperature': 1000,
                    'Lithology': 'basalt',
                    'Porosity': 0.2,
                    'Density': 2850, #kg/m3
                    'TOC':0} #wt%

        self.rock_prop_dict = {
            "shale":{
                'Porosity':0.1,
                'Density':2500,
                'TOC':7
            },
            "sandstone":{
                'Porosity':0.2,
                'Density':2600,
                'TOC':2.5
            },
            "limestone":{
                'Porosity':0.2,
                'Density':2600,
                'TOC':2.5
            },
            "granite":{
                'Porosity':0.05,
                'Density':2700,
                'TOC':0
            },
            "basalt":{
                'Porosity': 0.2,
                'Density': 2850, #kg/m3
                'TOC':0
            },
            "peridotite":{
                'Porosity': 0.05,
                'Density': 3100, #kg/m3
                'TOC':0
            }
        }
        


    #Setting up the remaining property arrays#
    def get_physical_properties(self, rock, rock_prop_dict = None):
        if rock_prop_dict==None:
            rock_prop_dict = self.rock_prop_dict
        a,b = rock.shape
        porosity = np.zeros_like(rock)
        density = np.zeros_like(rock)
        TOC = np.zeros_like(rock)

        for i in range(a):
            for j in range(b):
                porosity[i,j] = rock_prop_dict[rock[i,j]]['Porosity']
                density[i,j] = rock_prop_dict[rock[i,j]]['Density']
                TOC[i,j] = rock_prop_dict[rock[i,j]]['TOC']
        return porosity, density, TOC
    def func_assigner(self, func, *args, **kwargs):
        result = func(*args,**kwargs)
        print(f'Result: {result}')
        # If the result is a tuple or list, enumerate it
        #if isinstance(result, (tuple, list)):
        #    enumerated_result = list(enumerate(result))
        #    return enumerated_result
        # If the result is a single value, return it as is
        return result
    def build_sillcube(self, x, y, dx, dy, dt, thickness_range, aspect_ratio, depth_range, lat_range, phase_times, tot_volume, flux, n_sills, shape = 'elli', depth_function = None, lat_function = None, dims_function = None):
        dims_empirical = False
        min_thickness = thickness_range[0] #m
        max_thickness = thickness_range[1] #m
        sd_min = thickness_range[2]
        mar = aspect_ratio[0]
        sar = aspect_ratio[1]

        min_emplacement = depth_range[0] #m
        max_emplacement = depth_range[1] #m
        sd_empl = depth_range[2]

        x_min = lat_range[0]
        x_max = lat_range[1]
        sd_x = lat_range[2]

        if depth_function==None or depth_function=='normal':
            depth_function = rool.randn_heights
            depth_input_params = (n_sills, max_emplacement, min_emplacement, sd_empl, dy)
        elif depth_function == 'uniform':
            depth_function = rool.uniform_heights
            depth_input_params = (n_sills, min_emplacement, max_emplacement, dy)
        elif depth_function=='empirical':
            depth_function = rool.empirical_CDF
            depth_input_params = (n_sills, depth_range[0], depth_range[1])

        empl_heights = self.func_assigner(depth_function, *depth_input_params)
        
        if lat_function==None or lat_function=='normal':
            lat_function = rool.x_spacings
            lat_input_params = (n_sills, x_min, x_max, sd_x, dx)
        elif lat_function=='uniform':
            lat_function = rool.uniform_x
            lat_input_params = (n_sills, x_min, x_max, dx)
        elif lat_function == 'empirical':
            lat_function = rool.empirical_CDF
            lat_input_params = (n_sills, lat_range[0], lat_range[1])
        
        if dims_function==None or dims_function== 'normal':
            dims_function = rool.randn_dims
            dims_input_params = (min_thickness, max_thickness, sd_min, mar, sar, n_sills)
        elif dims_function == 'uniform':
            dims_function = rool.uniform_dims
            dims_input_params = (min_thickness, max_thickness, aspect_ratio[0], aspect_ratio[1], n_sills)
        elif dims_function == 'scaled':
            dims_function = rool.get_scaled_dims
            dims_input_params = (min_thickness, max_thickness, mar, sar, empl_heights, n_sills)
        elif dims_function == 'empirical':
            dims_empirical = True
            dims_function = rool.empirical_CDF
            dims_input_params = (n_sills, aspect_ratio[0], aspect_ratio[1])
        

        '''
        #Checking to see if there are any assignments outside the distribution#
        n = 0
        while ((empl_heights>max_emplacement/dy).any() or (empl_heights<min_emplacement/dy).any()):
            print('Heights')
            print((len(empl_heights[empl_heights>max_emplacement/dy])+len(empl_heights[empl_heights<(min_emplacement/dy)]))*100/n_sills, '%')
            n = int(n+1)
            print('Cycle', n ,'reassigning')
            if (empl_heights>max_emplacement/dy).any():
                empl_heights[empl_heights>max_emplacement/dy] = depth_function(np.sum(empl_heights>max_emplacement/dy), max_emplacement, min_emplacement, sd_empl, dy)
            if (empl_heights<min_emplacement/dy).any():
                empl_heights[empl_heights<min_emplacement/dy] = depth_function(np.sum(empl_heights<(min_emplacement/dy)), max_emplacement, min_emplacement, sd_empl, dy)
        '''
        sns.kdeplot(empl_heights*dy/1000, label = 'Depth distribution', color = 'red', linewidth = 1.75)
        plt.ylabel('Depth distribution (km)')
        plt.savefig('plots/Depth.png', format = 'png')
        plt.close()
        x_space = self.func_assigner(lat_function, *lat_input_params)
        '''
        n = 0
        while ((x_space>x_max/dx).any() or (x_space<(x_min/dx)).any()):
            print((len(x_space[x_space>x_max/dx])+len(x_space[x_space<(x_min/dx)]))*100/n_sills, '%')
            print('Horizontal space')
            n = int(n+1)
            print('Cycle', n ,'reassigning')
            if (x_space>x_max/dx).any():
                x_space[x_space>x_max/dx] = lat_function(np.sum(x_space>x_max/dx), x_min, x_max, sd_x, dx)
            if (x_space<(x//(3*dx))).any():
                x_space[x_space<(x_min/dx)] = lat_function(np.sum(x_space<(x_min/dx)), x_min, x_max, sd_x, dx)
        '''
        width, thickness = self.func_assigner(dims_function, *dims_input_params) if not dims_empirical else (None, None)
        print(f"Width: {width}, Thickness: {thickness}")
        if (width==None).all():
            aspect_ratios = self.func_assigner(dims_function, dims_input_params)
            thickness = self.func_assigner(dims_function, *(thickness_range[0], thickness_range[1], n_sills))
            width = thickness*aspect_ratios

        sns.kdeplot(width, label = 'Width distribution', linewidth = 1.75)
        sns.kdeplot(thickness, label = 'Thickness Distribution', linewidth = 1.75, color = 'red')
        plt.xlim(left = 0)
        plt.xlabel('Length units (m)')
        plt.legend()
        plt.savefig('plots/WidthThickness.png', format = 'png')
        plt.close()
        
        thermal_maturation_time = phase_times[0]
        total_empl_time = tot_volume/flux
        cooling_time = phase_times[1]
        model_time = total_empl_time+thermal_maturation_time+cooling_time
        time_steps = np.arange(model_time,step=dt)
        saving_time_step_index = np.min(np.where(time_steps>=thermal_maturation_time)[0])
        print(saving_time_step_index, time_steps[saving_time_step_index])
        empl_times = []
        plot_time = []
        cum_volume = []

        if shape == 'elli':
            print(width, thickness)
            volume = (4*np.pi/3)*width*width*thickness
        elif shape=='rect':
            volume = width*width*thickness

        unemplaced_volume = 0
        #print(f'{np.sum(volume):.5e}, {float(tot_volume):.5e}, {np.sum(volume)<tot_volume}')
        print('Time steps:', len(time_steps))
        n = 0
        for l in range(len(time_steps)):
            if time_steps[l]<thermal_maturation_time:
                continue
            else:
                if n>0:
                    mean_flux = np.sum(volume[0:n])/(time_steps[l]-thermal_maturation_time)
                else:
                    mean_flux = 0
                unemplaced_volume += flux*dt
                if unemplaced_volume>=volume[n] and mean_flux<0.95*flux:
                    empl_times.append(time_steps[l])
                    plot_time.append(time_steps[l])
                    cum_volume.append(volume[n])
                    unemplaced_volume -= volume[n]
                    print(f'Emplaced sill {n} at time {time_steps[l]}')
                    print(f'Remaining volume to emplace: {tot_volume-np.sum(volume[:n]):.4e}')
                    mean_flux = np.sum(volume[0:n])/(time_steps[l]-thermal_maturation_time)
                    n+=1
                    
                    while unemplaced_volume>volume[n] and mean_flux<(0.95*flux if np.sum(volume[0:n+1])<=tot_volume else 1.05*flux) and np.sum(volume[0:n])<=tot_volume:
                        empl_times.append(time_steps[l])
                        unemplaced_volume -= volume[n]
                        cum_volume[-1]+=volume[n]
                        print(f'Emplaced sill {n} at time {time_steps[l]}')
                        print(f'Remaining volume to emplace: {tot_volume-np.sum(volume[:n]):.4e}')
                        mean_flux = np.sum(volume[0:n])/(time_steps[l]-thermal_maturation_time)
                        n+=1

                if (n>0) and (np.sum(volume[0:n-1])>tot_volume):
                    print('Total sills emplaced:', n)
                    n_sills = n
                    empl_heights = empl_heights[0:n_sills]
                    x_space = x_space[0:n_sills]
                    width = width[0:n_sills]
                    thickness = thickness[0:n_sills]
                    break
        cum_volume = np.cumsum(cum_volume)
        plt.plot(plot_time, cum_volume, color = 'red', linewidth = 1.75, label = 'Cumulative volume emplaced')
        lol = (np.array(empl_times)-thermal_maturation_time)*flux
        plt.plot(empl_times, lol, color = 'black', linewidth = 1.75, label = 'Mean cumulative volume')
        plt.ylabel(r'Volume emplacemed ($km^3$)')
        plt.xlabel(r'Time (Ma)')
        plt.legend()
        plt.savefig('plots/VolumeTime.png', format = 'png')
        plt.close()

        z_coords = lat_function(n_sills, x_min, x_max, sd_x, dx)
        sillcube = rool.sill_3Dcube(x,y,x,dx,dy,n_sills, x_space, empl_heights, z_coords, width, thickness, empl_times,shape)

        params = np.array([empl_times, empl_heights, x_space, width, thickness])
        return sillcube, n_sills, params
    
    def get_silli_initial_thermogenic_state(self, props_array, dx, dy, dt, method, k=None, time = None, lith_plot_dict = None, rock_prop_dict = None):
        if lith_plot_dict==None:
            lith_plot_dict = self.lith_plot_dict
        if rock_prop_dict==None:
            rock_prop_dict = self.rock_prop_dict
        density = props_array[self.dense_index]
        porosity = props_array[self.poros_index]
        rock = props_array[self.rock_index]
        t = 0
        a, b = props_array[0].shape
        H = np.zeros((a,b))
        TOC = rool.prop_updater(props_array[self.rock_index], lith_plot_dict, rock_prop_dict, 'TOC')
        if k == None:
            k = rool.get_diffusivity(props_array[self.Temp_index], props_array[self.rock_index])
        if time==None:
            T_field = cool.diff_solve(k, a, b, dx, dy, dt, T_field, np.nan, method, H)
            props_array[self.Temp_index] = T_field
            curr_TOC_silli = props_array[self.TOC_index]
            RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli = emit.SILLi_emissions(T_field, density, rock, porosity, TOC, dt)
            if (rock=='limestone').any():
                breakdown_CO2 = emit.get_init_CO2_percentages(T_field, rock, density, dy)
            props_array[self.TOC_index] = curr_TOC_silli
            diff = 1e6
            iter = 0
            thresh = 1e-5
            t+=dt
            tot_RCO2 = []
            tot_RCO2.append(np.sum(RCO2_silli)+np.sum(breakdown_CO2))
            iter_thresh = int(1e7//dt)
            while iter<iter_thresh and diff>thresh:
                curr_TOC_silli = props_array[self.TOC_index]
                RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli = emit.SILLi_emissions(T_field, density, rock, porosity, curr_TOC_silli, dt, TOC, W_silli)
                if (rock=='limestone').any():    
                    breakdown_CO2, _ = emit.get_breakdown_CO2(T_field, rock, density, breakdown_CO2, dy, dt)
                props_array[self.TOC_index] = curr_TOC_silli
                tot_RCO2.append(np.sum(RCO2_silli)+np.sum(breakdown_CO2))
                if iter>0:
                    diff = tot_RCO2[-2]-tot_RCO2[-1]
                iter+=1
        else:
            t_steps = np.linspace(0, time, dt)
            tot_RCO2 = []
            for l in range(0, len(t_steps)):
                T_field = cool.diff_solve(k, a, b, dx, dy, dt, T_field, np.nan, method, H)
                props_array[self.Temp_index] = T_field
                curr_TOC_silli = props_array[self.TOC_index]
                TOC = rool.prop_updater(rock, lith_plot_dict, rock_prop_dict, 'TOC')
                if l==0:
                    RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli = emit.SILLi_emissions(T_field, density, rock, porosity, TOC, dt)
                    if (rock=='limestone').any():
                        breakdown_CO2 = emit.get_init_CO2_percentages(T_field, rock, density, dy)
                else:
                    RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli = emit.SILLi_emissions(T_field, density, rock, porosity, curr_TOC_silli, dt, TOC, W_silli)
                    if (rock=='limestone').any():    
                        breakdown_CO2, _ = emit.get_breakdown_CO2(T_field, rock, density, breakdown_CO2, dy, dt)
                props_array[self.TOC_index] = curr_TOC_silli
                tot_RCO2.append(np.sum(RCO2_silli)+np.sum(breakdown_CO2))
        return tot_RCO2, props_array, RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli

    def get_sillburp_initial_thermogenic_state(self, props_array, dx, dy, dt, method, sillburp_weights = None, k=None, time = None, lith_plot_dict = None, rock_prop_dict = None):
        if lith_plot_dict==None:
            lith_plot_dict = self.lith_plot_dict
        if rock_prop_dict==None:
            rock_prop_dict = self.rock_prop_dict
        density = props_array[self.dense_index]
        porosity = props_array[self.poros_index]
        rock = props_array[self.rock_index]
        t = 0
        a, b = props_array[0].shape
        H = np.zeros((a,b))
        TOC = rool.prop_updater(props_array[self.rock_index], lith_plot_dict, rock_prop_dict, 'TOC')
        if k == None:
            k = rool.get_diffusivity(props_array[self.Temp_index], props_array[self.rock_index])
        if time==None:
            T_field = cool.diff_solve(k, a, b, dx, dy, dt, T_field, np.nan, method, H)
            props_array[self.Temp_index] = T_field
            curr_TOC = props_array[self.TOC_index]
            reaction_energies = emit.get_sillburp_reaction_energies()
            RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions = emit.sillburp(T_field, TOC, density, rock, porosity, dt, reaction_energies, weights=sillburp_weights)
            if (rock=='limestone').any():
                breakdown_CO2 = emit.get_init_CO2_percentages(T_field, rock, density, dy)
            props_array[self.TOC_index] = curr_TOC_silli
            diff = 1e6
            iter = 0
            thresh = 1e-5
            t+=dt
            tot_RCO2 = []
            iter_thresh = int(1e7//dt)
            tot_RCO2.append(np.sum(RCO2)+np.sum(breakdown_CO2))
            while iter<iter_thresh and diff>thresh:
                curr_TOC = props_array[self.TOC_index]
                RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions = emit.sillburp(T_field, curr_TOC, density, rock, porosity, dt, reaction_energies, TOC, oil_production_rate, progress_of_reactions, rate_of_reactions, weights=sillburp_weights)
                if (rock=='limestone').any():    
                    breakdown_CO2, _ = emit.get_breakdown_CO2(T_field, rock, density, breakdown_CO2, dy, dt)
                props_array[self.TOC_index] = curr_TOC_silli
                tot_RCO2.append(np.sum(RCO2)+np.sum(breakdown_CO2))
                if iter>0:
                    diff = tot_RCO2[-2]-tot_RCO2[-1]
                iter+=1
        else:
            t_steps = np.linspace(0, time, dt)
            tot_RCO2 = []
            for l in range(0, len(t_steps)):
                T_field = cool.diff_solve(k, a, b, dx, dy, dt, T_field, np.nan, method, H)
                props_array[self.Temp_index] = T_field
                curr_TOC_silli = props_array[self.TOC_index]
                TOC = rool.prop_updater(rock, lith_plot_dict, rock_prop_dict, 'TOC')
                if l==0:
                    RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions = emit.sillburp(T_field, TOC, density, rock, porosity, dt, reaction_energies, weights=sillburp_weights)
                    if (rock=='limestone').any():
                        breakdown_CO2 = emit.get_init_CO2_percentages(T_field, rock, density, dy)
                else:
                    RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions = emit.sillburp(T_field, curr_TOC, density, rock, porosity, dt, reaction_energies, TOC, oil_production_rate, progress_of_reactions, rate_of_reactions, weights=sillburp_weights)
                    if (rock=='limestone').any():    
                        breakdown_CO2, _ = emit.get_breakdown_CO2(T_field, rock, density, breakdown_CO2, dy, dt)
                props_array[self.TOC_index] = curr_TOC_silli
                tot_RCO2.append(np.sum(RCO2)+np.sum(breakdown_CO2))
        return tot_RCO2, props_array, RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions

    def emplace_sills(self,props_array, k, dx, dy, dt, n_sills, z_index, cool_method, model, time_steps, current_time, sillcube, carbon_model_params, emplacement_params, H = np.nan, rock_prop_dict = None, lith_plot_dict = None, prop_dict = None, magma_prop_dict = None):
        shape_index = [len(time_steps[time_steps.index(current_time):])]+list(props_array.shape)
        props_total_array = np.empty(shape_index, dtype = object)
        if lith_plot_dict==None:
            lith_plot_dict = self.lith_plot_dict
        if rock_prop_dict==None:
            rock_prop_dict = self.rock_prop_dict
        if prop_dict==None:
            prop_dict = self.prop_dict
        if magma_prop_dict==None:
            magma_prop_dict = self.magma_prop_dict
        rock = props_array[self.rock_index]
        density = props_array[self.dense_index]
        porosity = props_array[self.poros_index]
        a,b = sillcube.shape[1], sillcube.shape[2]
        if np.isnan(H):
            H = np.zeros((a,b))
        if model=='silli':
            tot_RCO2, props_array_unused, RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli = carbon_model_params
        elif model =='sillburp':
           tot_RCO2, props_array_unused, RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions, sillburp_weights = carbon_model_params
           reaction_energies = emit.get_sillburp_reaction_energies()
        else:
            raise ValueError(f'model is {model}, but must be either silli or sillburp')
        empl_times, empl_heights, x_space, width, thickness = emplacement_params
        curr_sill = 0
        for l in range(time_steps.index(current_time), len(time_steps)):
            curr_time = time_steps[l]
            T_field = cool.diff_solve(k, a, b, dx, dy, dt, T_field, np.nan, cool_method, H)
            curr_TOC_silli = props_array[self.TOC_index]
            TOC = rool.prop_updater(rock, lith_plot_dict, rock_prop_dict, 'TOC')
            if model=='silli':
                RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli = emit.SILLi_emissions(T_field, density, rock, porosity, curr_TOC_silli, dt, TOC, W_silli)
            elif model=='sillburp':
                RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions = emit.sillburp(T_field, curr_TOC, density, rock, porosity, dt, reaction_energies, TOC, oil_production_rate, progress_of_reactions, rate_of_reactions, weights=sillburp_weights)
            if (rock=='limestone').any():    
                breakdown_CO2, _ = emit.get_breakdown_CO2(T_field, rock, density, breakdown_CO2, dy, dt)
            props_array[self.TOC_index] = curr_TOC_silli
            while time_steps[l]==empl_times[curr_sill] and curr_sill<n_sills:
                #print(f'Now emplacing sill {curr_sill}')
                props_array, row_start, col_pushed = rool.sill3D_pushy_emplacement(props_array, prop_dict, sillcube, curr_sill, magma_prop_dict, z_index, empl_times[curr_sill])
                if model=='silli':
                    if (col_pushed!=0).all():
                        RCO2_silli = rool.value_pusher2D(RCO2_silli,0, row_start, col_pushed)
                        Rom_silli = rool.value_pusher2D(Rom_silli,0, row_start, col_pushed)
                        percRo_silli = rool.value_pusher2D(percRo_silli, 0, row_start, col_pushed)
                        curr_TOC_silli = rool.value_pusher2D(curr_TOC_silli,0, row_start, col_pushed)
                        for m in range(W_silli.shape[0]):
                            W_silli[m] = rool.value_pusher2D(W_silli[m],0, row_start, col_pushed)
                elif model=='sillburp':
                    if (col_pushed!=0).all():
                        RCO2 = rool.value_pusher2D(RCO2,0, row_start, col_pushed)
                        Rom = rool.value_pusher2D(Rom,0, row_start, col_pushed)
                        oil_production_rate = rool.value_pusher2D(oil_production_rate,0, row_start, col_pushed)
                        curr_TOC = rool.value_pusher2D(curr_TOC,0, row_start, col_pushed)
                        for huh in range(progress_of_reactions.shape[0]):
                            for bruh in range(progress_of_reactions.shape[1]):
                                progress_of_reactions[huh][bruh] = rool.value_pusher2D(progress_of_reactions[huh][bruh],1, row_start, col_pushed)
                                progress_of_reactions[huh][bruh] = rool.value_pusher2D(progress_of_reactions[huh][bruh],1, row_start, col_pushed)

                if (curr_sill+1)<n_sills:
                    curr_sill +=1
                else:
                    break
                dV = dx*dx*dy
                if model=='silli':
                    RCO2_silli = RCO2_silli*density*dV/100
                    tot_RCO2.append(np.sum(RCO2_silli))
                elif model=='sillburp':
                    RCO2 = RCO2*density*dV/100
                    tot_RCO2.append(np.sum(RCO2))
            props_total_array[l-time_steps.index(current_time)] = props_array
        if model=='silli':
            carbon_model_params = tot_RCO2, props_array, RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli
        elif model=='sillburp':
            carbon_model_params = tot_RCO2, props_array_unused, RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions
        return props_total_array, carbon_model_params

    def example_run(self):
        #Dimensions of the 2D grid
        x = 300000 #m - Horizontal extent of the crust
        y = 35000 #m - Vertical thickness of the crust

        dx = 250 #m node spacing in x-direction
        dy = 250 #m node spacing in y-direction

        a = int(y//dy) #Number of rows
        b = int(x//dx) #Number of columns

        #Temp at the surface
        T_surf = 0 #deg C

        #Magmatic temperature
        T_mag = 1000 #deg C

        #Initializing diffusivity field
        k = np.ones((a,b))*31.536 #m2/yr

        dt = (min(dx,dy)**2)/(5*np.max(k))

        #Shape of the sills
        shape = 'elli'

        #Initializing the temp field
        T_field = np.zeros((a,b))
        T_field[-1,:] = T_mag
        T_field = cool.heat_flux(k, a, b, dx, dy, T_field, 'straight')
        rock = np.empty((a,b), dtype = object)

        rock[:] = 'granite'
        rock[0:int(5000/dy),:] = 'shale'
        rock[int((5000/dy)+1):int(15000/dy),:] = 'sandstone'
        rock[int((30000/dy)+1):,:] = 'peridotite'
        #Setting up the remaining property arrays#
        porosity = np.zeros_like(rock)
        density = np.zeros_like(rock)
        TOC = np.zeros_like(rock)

        for i in range(a):
            for j in range(b):
                porosity[i,j] = self.rock_prop_dict[rock[i,j]]['Porosity']
                density[i,j] = self.rock_prop_dict[rock[i,j]]['Density']
                TOC[i,j] = self.rock_prop_dict[rock[i,j]]['TOC'] 
        ###Building the 3d properties array###
        props_array = np.empty((len((self.prop_dict.keys())),a,b), dtype = object)

        props_array[self.Temp_index] = T_field
        props_array[self.rock_index] = rock
        props_array[self.poros_index] = porosity
        props_array[self.dense_index] = density
        props_array[self.TOC_index] = TOC

        ###Setting up sill dimensions and locations###
        min_thickness = 900 #m
        max_thickness = 3500 #m

        mar = 7
        sar = 2.5

        min_emplacement = 1000 #m
        max_emplacement = 15500 #m
        n_sills = 20000

        tot_volume = int(0.5e6*1e9)
        flux = int(30e9)

        thermal_mat_time = 3e6
        model_time = tot_volume/flux
        cooling_time = int(1e6)

        phase_times = [thermal_mat_time, model_time, cooling_time]
        time_steps = np.arange(0, np.sum(phase_times), dt)

        sillcube, n_sills, emplacement_params = self.build_sillcube(x, y, dx, dy, dt, [min_thickness, max_thickness, 500], [mar, sar], [min_emplacement, max_emplacement, 5000], [x//3, 2*x//3, x//6], phase_times, tot_volume, flux, n_sills)
        carbon_model_params = self.get_silli_initial_thermogenic_state(props_array, dx, dy, dt,k, thermal_mat_time, method='conv smooth')
        props_total_array, carbon_model_params = self.emplace_sills(props_array, k, dx, dy, dt, n_sills, x//2, 'conv smooth', 'silli', time_steps, emplacement_params[0], carbon_model_params, emplacement_params)

        tot_RCO2 = carbon_model_params[0]
        plt.plot(time_steps, tot_RCO2)
        plt.show()
