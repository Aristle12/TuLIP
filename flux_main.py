import numpy as np
import rule_functions as rool
import functions as cool
from tqdm import trange
import matplotlib.pyplot as plt
import carbon_emissions as emit
import h5py
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pickle

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

###Setting up the properties dictionary to translate properties to indices for 3D array###

prop_dict = {'Temperature':0,
             'Lithology':1,
             'Porosity':2,
             'Density':3,
             'TOC':4}

#Lithology dictionary to translate rock types into numerical codes for numpy arrays
lith_plot_dict = {'granite':0,
                  'shale':1,
                  'sandstone':2,
                  'peridotite':3,
                  'basalt':4}


Temp_index = prop_dict['Temperature']
rock_index = prop_dict['Lithology']
poros_index = prop_dict['Porosity']
dense_index = prop_dict['Porosity']
TOC_index = prop_dict['TOC']

magma_prop_dict = {'Temperature': T_mag,
             'Lithology': 'basalt',
             'Porosity': 0.2,
             'Density': 2850, #kg/m3
             'TOC':0} #wt%

rock_prop_dict = {
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


###Constructing bedrock###
rock = np.empty((a,b), dtype = object)

rock[:] = 'granite'
rock[0:int(5000/dy),:] = 'shale'
rock[int((5000/dy)+1):int(15000/dy),:] = 'sandstone'
rock[int((30000/dy)+1):,:] = 'peridotite'


plot_rock = np.zeros((a,b), dtype = int)

for i in range(a):
    for j in range(b):
        plot_rock[i,j] = lith_plot_dict[rock[i,j]]


labels = [key for key in lith_plot_dict]

# Visualize the rock array
plt.imshow(plot_rock, cmap='viridis')
cbar = plt.colorbar(ticks=list(lith_plot_dict.values()))
cbar.set_ticklabels(list(labels))
cbar.set_label('Rock Type')
plt.title('Bedrock Composition')
plt.show()

#Setting up the remaining property arrays#
porosity = np.zeros_like(rock)
density = np.zeros_like(rock)
TOC = np.zeros_like(rock)

for i in range(a):
    for j in range(b):
        porosity[i,j] = rock_prop_dict[rock[i,j]]['Porosity']
        density[i,j] = rock_prop_dict[rock[i,j]]['Density']
        TOC[i,j] = rock_prop_dict[rock[i,j]]['TOC']


###Building the 3d properties array###
props_array = np.empty((len((prop_dict.keys())),a,b), dtype = object)

props_array[Temp_index] = T_field
props_array[rock_index] = rock
props_array[poros_index] = porosity
props_array[dense_index] = density
props_array[TOC_index] = TOC


###Setting up sill dimensions and locations###
min_thickness = 900 #m
max_thickness = 3500 #m

mar = 7
sar = 2.5

min_emplacement = 1000 #m
max_emplacement = 15500 #m
n_sills = 20000

empl_heights = rool.randn_heights(n_sills, max_emplacement, min_emplacement, 5000, dy)
#Checking to see if there are any assignments outside the distribution#
n = 0
while ((empl_heights>max_emplacement/dy).any() or (empl_heights<min_emplacement/dy).any()):
    print('Heights')
    print((len(empl_heights[empl_heights>max_emplacement/dy])+len(empl_heights[empl_heights<(min_emplacement/dy)]))*100/n_sills, '%')
    n = int(n+1)
    print('Cycle', n ,'reassigning')
    if (empl_heights>max_emplacement/dy).any():
        empl_heights[empl_heights>max_emplacement/dy] = rool.randn_heights(np.sum(empl_heights>max_emplacement/dy), max_emplacement, min_emplacement, 5000, dy)
    if (empl_heights<min_emplacement/dy).any():
        empl_heights[empl_heights<min_emplacement/dy] = rool.randn_heights(np.sum(empl_heights<(min_emplacement/dy)), max_emplacement, min_emplacement, 5000, dy)
plt.hist(empl_heights*dy)
plt.show()

x_space = rool.x_spacings(n_sills, x//3, 2*x//3, x//6, dx)
    
n = 0
while ((x_space>0.66*x/dx).any() or (x_space<(x//(3*dx))).any()):
    print((len(x_space[x_space>0.66*x/dx])+len(x_space[x_space<(x//(3*dx))]))*100/n_sills, '%')
    print('Horizontal space')
    n = int(n+1)
    print('Cycle', n ,'reassigning')
    if (x_space>0.66*x/dx).any():
        x_space[x_space>0.66*x/dx] = rool.x_spacings(np.sum(x_space>0.66*x/dx), x//3, 2*x//3, x//6, dx)
    if (x_space<(x//(3*dx))).any():
        x_space[x_space<(x//(3*dx))] = rool.x_spacings(np.sum(x_space<(x//(3*dx))), x//3, 2*x//3, x//6, dx)
plt.hist(x_space*dx)
plt.show()
width, thickness = rool.randn_dims(min_thickness, max_thickness, 700, mar, sar, n_sills)

###Defining flux rate and setting time###
flux = int(30e9) #m3/yr
tot_volume = int(0.15e6*1e9) #m3
total_empl_time = tot_volume/flux
thermal_maturation_time = int(3e6) #yr

model_time = total_empl_time+thermal_maturation_time+50000
time_steps = np.arange(model_time,step=dt)
saving_time_step_index = np.min(np.where(time_steps>=thermal_maturation_time)[0])
print(saving_time_step_index, time_steps[saving_time_step_index])
empl_times = []
plot_time = []
cum_volume = []

if shape == 'elli':
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
plt.plot(plot_time, cum_volume)
lol = (np.array(empl_times)-thermal_maturation_time)*flux
plt.plot(empl_times, lol)
plt.show()

#Building the third dimension#
z_coords = rool.x_spacings(n_sills, x//3, 2*x//3, x//6, dx)

sillcube = rool.sill_3Dcube(x,y,x,dx,dy,n_sills, x_space, empl_heights, z_coords, width, thickness, empl_times,shape)
plot_cube = np.where(sillcube!='',1,0)
plot_cube = np.where(sillcube!='',1,0)

print('3D cube built')
print(f'Shape of cube:{sillcube.shape}')
'''
plot_x, plot_y, plot_z = np.meshgrid(np.arange(0,x, dx),
                            np.arange(0,y, dy),
                            np.arange(0,x, dx),
                            indexing = 'ij')

x_flat = plot_x.flatten()
y_flat = plot_y.flatten()
z_flat = plot_z.flatten()
c_flat = plot_cube.flatten()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(x_flat, y_flat, z_flat, c=c_flat, cmap=cm.viridis)

# Add a color bar which maps values to colors
fig.colorbar(scatter, ax=ax, label='Value')

# Set labels for the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()
'''
np.save('sillcube', sillcube)
z_index = int(b//2)
print(z_index)
while np.sum(sillcube[z_index]!='')==0:
    z_index = np.random.randint(int(b//3), int(2*b//3))
sillslice = sillcube[z_index]
print(f'z_index is {z_index}')
unempty = sillslice[sillslice!='']
print(f'Non-empty nodes:{unempty}')

#Initializing the hdf5 datafile to store the generated data

shape_indices = [len(time_steps[saving_time_step_index:])] + list(props_array.shape)
shape_indices = [len(time_steps[saving_time_step_index:])] + list(props_array.shape)
print(shape_indices)
props_h5 = h5py.File('PropertyEvolution.hdf5', 'w')

props_total_array = np.zeros(shape_indices, dtype = object)

#Emplacing sills and running the 2D model
method = 'conv smooth'
H = np.zeros_like(T_field)
tot_RCO2 = np.zeros(len(time_steps))
curr_sill = 0
for l in trange(len(time_steps)):
    curr_time = time_steps[l]
    T_field = cool.diff_solve(k, a, b, dx, dy, dt, T_field, np.nan, method, H)
    props_array[Temp_index] = T_field
    curr_TOC_silli = props_array[TOC_index]
    TOC = rool.prop_updater(rock, lith_plot_dict, prop_dict, 'TOC')
    if l==0:
        RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli = emit.SILLi_emissions(T_field, density, rock, porosity, TOC, dt, dy)
    else:
        RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli = emit.SILLi_emissions(T_field, density, rock, porosity, curr_TOC_silli, dt, dy, TOC, W_silli)
    props_array[TOC_index] = curr_TOC_silli
    while time_steps[l]==empl_times[curr_sill] and curr_sill<n_sills:
        #print(f'Now emplacing sill {curr_sill}')
        props_array, row_start, col_pushed = rool.sill3D_pushy_emplacement(props_array, prop_dict, sillcube, curr_sill, magma_prop_dict, z_index, empl_times[curr_sill])
        if (col_pushed!=0).all():
            RCO2_silli = rool.value_pusher2D(RCO2_silli,0, row_start, col_pushed)
            Rom_silli = rool.value_pusher2D(Rom_silli,0, row_start, col_pushed)
            percRo_silli = rool.value_pusher2D(percRo_silli, 0, row_start, col_pushed)
            curr_TOC_silli = rool.value_pusher2D(curr_TOC_silli,0, row_start, col_pushed)
            for m in range(W_silli.shape[0]):
                W_silli[m] = rool.value_pusher2D(W_silli[m],0, row_start, col_pushed)
        if (curr_sill+1)<n_sills:
            curr_sill +=1
        else:
            break
    tot_RCO2[l] = np.sum(RCO2_silli)
    if l>=saving_time_step_index:
        props_total_array[l-saving_time_step_index] = props_array
    if l>=saving_time_step_index:
        props_total_array[l-saving_time_step_index] = props_array
try:
    plt.plot(time_steps[1:], np.log10(tot_RCO2[1:]+1e-5))
except RuntimeWarning:
    print("Warning: Divide by zero error encountered. Some values in tot_RCO2 are zero.")
    plt.plot(time_steps[1:], tot_RCO2[1:])
except Exception as e:
    print(f"An unexpected error occurred: {e}")
#plt.plot(time_steps[1:], tot_RCO2[1:])
plt.show()

reverse_lith_dict = {value: key for key, value in lith_plot_dict.items()}

for l in range(shape_indices[0]):
    for i in range(a):
        for j in range(b):
            props_total_array[l][rock_index][i,j] = lith_plot_dict[rock[i,j]]

#props_h5.create_dataset('props_array', data = props_total_array)


pickled_props = pickle.dumps(props_total_array)

with h5py.File('PropertyEvolution.hdf5', 'w') as hf:
    hf.create_dataset('props_total_array', data=pickled_props, chunks = True, compression = 'gzip')
with h5py.file('PropertyEvolution.hdf5', 'a') as hdf:
    group = hdf.require('prop_dict')

    for key, value in prop_dict.items():
        if isinstance(value, dict):
            sub_group = group.require_group(key)
            for sub_key, sub_value in value.items():
                sub_group.create_dataset(sub_key, data=sub_value)
        else:
            group.create_dataset(key, data = value)

with h5py.file('PropertyEvolution.hdf5', 'a') as hdf:
    group = hdf.require('lith_dict')

    for key, value in reverse_lith_dict.items():
        if isinstance(value, dict):
            sub_group = group.require_group(key)
            for sub_key, sub_value in value.items():
                sub_group.create_dataset(sub_key, data=sub_value)
        else:
            group.create_dataset(key, data = value)

props_h5.create_dataset('RCO2', data = tot_RCO2)

