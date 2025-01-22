from TuLIP import sill_controls as sc
from TuLIP import rules as rool
import numpy as np
from tqdm import trange
import pandas as pd
from scipy.spatial import KDTree
import matplotlib.pyplot as plt



#Read sills file
sillsquare = np.load('sillcube36.npy', allow_pickle=True)
print('Data loaded')
#Assign temperatures
T_field = np.zeros_like(sillsquare, dtype = float)

print('Temperature assigned')
print(sillsquare.shape)
a,b = sillsquare.shape
'''
is_sill = (sillsquare!='')
boundary_finder = np.array(is_sill, dtype=int)
boundary_finder[1:-1, 1:-1] = (
boundary_finder[:-2, 1:-1] +  # Above
boundary_finder[2:, 1:-1] +   # Below
boundary_finder[1:-1, :-2] +  # Left
boundary_finder[1:-1, 2:])     # Right
#Run test

plt.imshow(boundary_finder)
plt.show()
'''
sillsquare[sillsquare==''] = -100

for g in trange(20):
    quer = '_'+str(g)+'s'
    sillsquare[rool.index_finder(sillsquare,quer)] = g


T_field[sillsquare!=-100] = 1000
plt.imshow(T_field)
#plt.show()

sill = np.max(sillsquare)
sillsquare[sillsquare==-100] = ''
print(sillsquare[sillsquare!=''])
'''
sills_number = sillsquare.copy()
sills_number[sills_number==''] = -1
curr_sill = 8
T_solidus = 800
condition = (T_field>T_solidus) & (sills_number!=-1) & (sills_number!=curr_sill)
print(condition.shape)
array = np.zeros_like(T_field) + condition*1
plt.imshow(array)
plt.show()
'''


#for i in trange(sill+1):
sills_data = sc.check_closest_sill_temp(T_field, sillsquare, 8)
print(sills_data)


# Example inputs
T_field = np.random.rand(100, 100) * 1000  # Random temperature field
sills_array = np.random.randint(0, 10, size=(100, 100))  # Random sills array
curr_sill = 5  # Current sill to analyze

# Call the function
result = sc.check_closest_sill_temp(T_field, sills_array, curr_sill)

# Display the result
print(result)
