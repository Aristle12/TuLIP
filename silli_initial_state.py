from TuLIP import sill_controls
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import pandas as pd
sc = sill_controls()

#Dimensions of the 2D grid
x = 300000/300 #m - Horizontal extent of the crust
y = 8000 #m - Vertical thickness of the crust
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

dt = (min(dx,dy)**2)/(5*np.max(k))
#Shape of the sills
shape = 'elli'



#Initializing the temp field
T_field = np.zeros((a,b))
#T_field[-1,:] = T_mag
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
        TOC[i,j] = sc.rock_prop_dict[rock[i,j]]['TOC'] if rock[i,j]=='granite' else 2
###Building the 3d properties array###
props_array = np.empty((len((sc.prop_dict.keys())),a,b), dtype = object)

props_array[sc.Temp_index] = T_field
props_array[sc.rock_index] = rock
props_array[sc.poros_index] = porosity
props_array[sc.dense_index] = density
props_array[sc.TOC_index] = TOC

thermal_mat_time = int(3e6)

params = sc.get_silli_initial_thermogenic_state(props_array, dx, dy, dt, 'conv smooth', k, time = thermal_mat_time)

tiled_props_array = np.empty((len((sc.prop_dict.keys())),a,bee), dtype = object)
for i in range(tiled_props_array.shape[0]):
    tiled_props_array[i] = np.tile(props_array[i],(1,300))

current_time = params[0]
csv = pd.read_csv('sillcubes/n_sills.csv')
csv['curr_time'] = current_time
csv.to_csv('sillcubes/n_sills.csv')
tot_RCO2, props_array, RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli = params[1:]
ace = np.array([RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli])
tiled_params = np.empty((ace.shape[0],a,bee))
tiled_props_array[sc.TOC_index] = np.tile(curr_TOC_silli,(1,300))
for i in range(ace.shape[0]):
    tiled_params[i] = np.tile(ace[i],(1,300))
tiled_W = np.empty((W_silli.shape[0],a,bee))
for i in range(W_silli.shape[0]):
    tiled_W[i] = np.tile(W_silli[i],(1,300))

data = pv.ImageData()
# Set the dimensions and spacing
data.dimensions = tiled_params.shape
data.spacing = (1.0, 1.0, 1.0)  # Adjust spacing as needed
# Assign the tiled parameters to the point data
data.point_data['data'] = np.array(tiled_params, dtype=float).flatten(order="F")
# Print the point data to verify
print(data.point_data['data'])
data.save('sillcubes/initial_silli_state_carbon.vtk')

W_data = pv.ImageData()
W_data.dimensions = W_silli.shape
W_data.point_data['data'] = W_silli.flatten(order = 'F')
W_data.save('sillcubes/W_data.vtk')

props_array_vtk = pv.ImageData()
props_array_vtk.dimensions = tiled_props_array.shape
props_array_vtk.point_data['data'] = tiled_props_array.flatten(order = 'F')
props_array_vtk.save('sillcubes/initial_silli_state_properties.vtk')