import numpy as np
import pyvista as pv
import pandas as pd

l = 0
iter = 0
flux = int(3e9)

n_sills_dataframe = pd.read_csv('sillcubes/n_sills.csv')
current_time = n_sills_dataframe['curr_time'][iter]
n_sills = n_sills_dataframe['n_sills'][iter]
volumes = n_sills_dataframe['volumes'][iter]
dir = 'sillcubes/'+str(format(flux, '.3e'))+'/'+str(format(volumes, '.3e'))

data = pv.read(dir+'/Properties_'+str(l)+'.vtk')

print(data.dimensions)

array = data.point_data['Temperature'].reshape(data.dimensions)

print(array[0])

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

dt = (min(dx,dy)**2)/(5*np.max(k))/10
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