from TuLIP import sill_controls
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import pandas as pd


#Dimensions of the 2D grid
x = 300000/300 #m - Horizontal extent of the crust
y = 12000 #m - Vertical thickness of the crust
z = 30000 #m - Third dimension for cube

ecks = x*300

dx = 50 #m node spacing in x-direction
dy = 50 #m node spacing in y-direction

a = int(y//dy) #Number of rows
b = int(x//dx) #Number of columns
bee = int(ecks//dx)

#Temp at the surface
T_surf = 0 #deg C

#Magmatic temperature
T_mag = 1000 #deg C

#Initializing diffusivity field
k = np.ones((a,b))*31.536 #m2/yr

dt = np.round((min(dx,dy)**2)/(5*np.max(k)),3)
#Shape of the sills
shape = 'elli'

sc = sill_controls(x, y, dx, dy, k_const = True, include_external_heat = False)

#Initializing the temp field
T_field = np.zeros((a,b))
#T_field[-1,:] = T_magpre
q = k[-1,:]*(30/1000)
T_field = sc.cool.heat_flux(k, a, b, dx, dy, T_field, 'straight', q)

plt.imshow(T_field)
plt.colorbar()
plt.show()

rock = np.empty((a,b), dtype = object)
rock[:] = 'granite'
rock[0:4000//dy,:] = 'shale'


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

thermal_mat_time = (int(3e6//dt)+1)*dt
print(thermal_mat_time)

params = sc.get_silli_initial_thermogenic_state(props_array, dt, 'conv smooth', time = thermal_mat_time)

tiled_props_array = np.empty((len((sc.prop_dict.keys())),a,bee), dtype = object)

tiled_props_array[sc.Temp_index] = np.tile(props_array[sc.Temp_index],(1,300))
tiled_props_array[sc.rock_index] = np.tile(props_array[sc.rock_index],(1,300))
tiled_props_array[sc.poros_index] = np.tile(props_array[sc.poros_index],(1,300))
tiled_props_array[sc.dense_index] = np.tile(props_array[sc.dense_index],(1,300))
tiled_props_array[sc.TOC_index] = np.tile(props_array[sc.TOC_index],(1,300))

curr_time = params[0]
#csv = pd.read_csv('sillcubes/n_sills.csv')
#csv['curr_time'] = current_time
#csv.to_csv('sillcubes/n_sills.csv')
#pd.DataFrame(np.array(curr_time)).to_csv('sillcubes/n_sills.csv')
np.save('sillcubes/curr_time', np.array(curr_time))
tot_RCO2, props_array, RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli = params[1:]
ace = np.array([RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli])
tiled_RCO2 = np.tile(RCO2_silli,(1,300))
tiled_Rom = np.tile(Rom_silli,(1,300))
tiled_percRo = np.tile(percRo_silli,(1,300))
tiled_TOC = np.tile(curr_TOC_silli,(1,300))
tiled_props_array[sc.TOC_index] = np.tile(curr_TOC_silli,(1,300))
tot_RCO2 = tot_RCO2*300


tiled_W = np.empty((W_silli.shape[0],a,bee))
for i in range(W_silli.shape[0]):
    tiled_W[i] = np.tile(W_silli[i],(1,300))

data = pv.ImageData()
# Set the dimensions and spacing
data.dimensions = [tiled_RCO2.shape[0], tiled_RCO2.shape[1], 1]

data.point_data['RCO2_silli'] = np.array(tiled_RCO2, dtype=float).flatten()
data.point_data['Rom_silli'] = np.array(tiled_Rom, dtype=float).flatten()
data.point_data['percRo_silli'] = np.array(tiled_percRo, dtype=float).flatten()
data.point_data['curr_TOC_silli'] = np.array(tiled_TOC, dtype=float).flatten()
# Print the point data to verify
data.save('sillcubes/initial_silli_state_carbon.vtk')

W_data = pv.ImageData()
W_data.dimensions = tiled_W.shape
W_data.point_data['data'] = tiled_W.flatten()
W_data.save('sillcubes/W_data.vtk')

props_array_vtk = pv.ImageData()
props_array_vtk.dimensions = tiled_props_array.shape
props_array_vtk.point_data['data'] = tiled_props_array.flatten()
props_array_vtk.save('sillcubes/initial_silli_state_properties.vtk')

pd.DataFrame(tot_RCO2).to_csv('sillcubes/tot_RCO2.csv')