import numpy as np
import rule_functions as rool
import functions as cool
from tqdm import trange
import matplotlib.pyplot as plt

#Dimensions of the 2D grid
x = 300000 #m - Horizontal extent of the crust
y = 35000 #m - Vertical thickness of the crust

dx = 200 #m node spacing in x-direction
dy = 200 #m node spacing in y-direction

a = int(y//dy) #Number of rows
b = int(x//dx) #Number of columns

#Temp at the surface
T_surf = 0 #deg C

#Magmatic temperature
T_mag = 1000 #deg C

#Initializing diffusivity field
k = np.ones((a,b))*31.536 #m2/yr

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
}
rock = np.empty((a,b), dtype = 'str')
#Constructing bedrock#
rock[:] = 'granite'
rock[0:int(5*dy),:] = 'shale'
rock[int((5*dy)+1):int(10*dy),:] = 'sandstone'

# Visualize the rock array
plt.imshow(rock, cmap='viridis')
plt.colorbar(ticks=['granite', 'shale', 'sandstone'], label='Rock Type')
plt.title('Bedrock Composition')
plt.show()

###Setting up sill dimensions and locations###
