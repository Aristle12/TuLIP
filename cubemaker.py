import matplotlib.pyplot as plt
import numpy as np
from TuLIP import sill_controls
import pandas as pd
from tqdm import trange

sc = sill_controls()

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
T_field = sc.cool.heat_flux(k, a, b, dx, dy, T_field, 'straight')
rock = np.empty((a,b), dtype = object)
rock[:] = 'granite'
rock[0:int(5000/dy),:] = 'shale'
rock[int((5000/dy)+1):int(15000/dy),:] = 'sandstone'
rock[int((30000/dy)+1):,:] = 'peridotite'

plot_rock = np.zeros((a,b), dtype = int)

for i in range(a):
    for j in range(b):
        plot_rock[i,j] = sc.lith_plot_dict[rock[i,j]]


labels = [key for key in sc.lith_plot_dict]

# Visualize the rock array
plt.imshow(plot_rock, cmap='viridis', extent = [0, x/1000, y/1000, 0])
plt.ylabel('Depth (km)')
plt.xlabel('Lateral extent (km)')
cbar = plt.colorbar(ticks=list(sc.lith_plot_dict.values()), orientation = 'horizontal')
cbar.set_ticklabels(list(labels))
cbar.set_label('Rock Type')
plt.title('Bedrock Composition')
plt.savefig('plots/bedrock_distribution.png', format = 'png')
plt.close()

#Setting up the remaining property arrays#
porosity = np.zeros_like(rock)
density = np.zeros_like(rock)
TOC = np.zeros_like(rock)

for i in range(a):
    for j in range(b):
        porosity[i,j] = sc.rock_prop_dict[rock[i,j]]['Porosity']
        density[i,j] = sc.rock_prop_dict[rock[i,j]]['Density']
        TOC[i,j] = sc.rock_prop_dict[rock[i,j]]['TOC'] 
###Building the 3d properties array###
props_array = np.empty((len((sc.prop_dict.keys())),a,b), dtype = object)

props_array[sc.Temp_index] = T_field
props_array[sc.rock_index] = rock
props_array[sc.poros_index] = porosity
props_array[sc.dense_index] = density
props_array[sc.TOC_index] = TOC

###Setting up sill dimensions and locations###
min_thickness = 300 #m
max_thickness = 1500 #m

mar = 7
sar = 2.5

min_emplacement = 1000 #m
max_emplacement = 15500 #m
n_sills = 20000

tot_volume_start = int(0.05e6*1e9)
tot_volume_end = int(0.5e6*1e9)
tot_volume = np.arange(tot_volume_start, tot_volume_end, int(0.05e6*1e9))
flux = int(30e9)

thermal_mat_time = int(3e6)
model_time = tot_volume/flux
cooling_time = int(1e6)

n_sills_array = []


for l in range(len(tot_volume)):
    print(f'Building cube for {tot_volume[l]}')
    phase_times = np.array([thermal_mat_time, model_time[l], cooling_time])
    print(phase_times)
    time_steps = np.arange(0, np.sum(phase_times), dt)
    print(f'Length of time_steps:{len(time_steps)}')

    sillcube, n_sills, emplacement_params = sc.build_sillcube(x, y, dx, dy, dt, [min_thickness, max_thickness, 500], [mar, sar], [min_emplacement, max_emplacement, 5000], [x//3, 2*x//3, x//6], phase_times, tot_volume[l], flux, n_sills)
    print('sillcube built')

    n_sills_array.append(n_sills)

    np.save('sillcubes/sillcube'+str(np.round(tot_volume[l], 2)), sillcube)


pd.DataFrame(n_sills_array).to_csv('sillcubes/n_sills.csv')
