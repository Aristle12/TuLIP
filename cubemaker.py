import matplotlib.pyplot as plt
import numpy as np
from TuLIP import sill_controls
import pandas as pd
from tqdm import trange
import pdb
from numba import jit
import pyvista as pv

#@jit
def int_maker(sillcube):
    for i in trange(sillcube.shape[0]):
        for j in range(sillcube.shape[1]):
            for g in range(sillcube.shape[2]):
                if sillcube[i,j,g] != 0:
                    ele = str(sillcube[i,j,g])
                    sillcube[i,j,g] = ele[1:3] if 's' not in ele[1:3] else ele[1]
    return sillcube
sc = sill_controls()

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

dt = (min(dx,dy)**2)/(5*np.max(k))
#Shape of the sills
shape = 'elli'

'''
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
'''
###Setting up sill dimensions and locations###
min_thickness = 100 #m
max_thickness = 600 #m

mar = 19.23
sar = 9.74

min_emplacement = 500 #m
max_emplacement = 4000 #m
n_sills = 20000

volume = x*y*z

tot_volume_start = 0.05*volume
tot_volume_end = 0.21*volume
tot_volume = np.arange(tot_volume_start, tot_volume_end, 0.03*volume)
flux = int(3e9)

thermal_mat_time = int(3e6)
model_time = tot_volume/flux
cooling_time = int(1e6)

n_sills_array = []
z_range = [0, z, z//3]


for l in range(len(tot_volume)):
    print(f'Building cube for {tot_volume[l]}: Cube {l} of {len(tot_volume)}')
    phase_times = np.array([thermal_mat_time, model_time[l], cooling_time])
    print(phase_times)
    time_steps = np.arange(0, np.sum(phase_times), dt)
    print(f'Length of time_steps:{len(time_steps)}')

    sillcube, n_sills1, emplacement_params = sc.build_sillcube(x, y, z, dx, dy, dt, [min_thickness, max_thickness, 500], [mar, sar], [min_emplacement, max_emplacement, 5000], z_range, [x//3, 2*x//3, x//6], phase_times, tot_volume[l], flux, n_sills)
    print('sillcube built')
    #pdb.set_trace()
    n_sills_array.append(n_sills1)

    np.save('sillcubes/sillcube'+str(np.round(tot_volume[l], 2)), sillcube)
    print('Done Here')
    mask = sillcube!=0
    sillcube[sillcube==''] = 0
    sillcube = int_maker(sillcube)
    sillcube = sillcube.astype(int)
    #pdb.set_trace()
    grid = pv.ImageData()
    grid.dimensions = sillcube.shape
    grid.point_data["sillcube"] = sillcube.flatten(order="F")
    grid.save('sillcubes/sillcube'+str(tot_volume[l])+'.vtk')
    emplace_frame = pd.DataFrame(np.transpose(emplacement_params), columns=['empl_times', 'empl_heights', 'x_space', 'width', 'thickness'])
    emplace_frame.to_csv('sillcubes/emplacement_params'+str(tot_volume[l])+'.csv')

pd.DataFrame({'n_sills': n_sills_array, 'volumes': tot_volume}).to_csv('sillcubes/n_sills.csv')
