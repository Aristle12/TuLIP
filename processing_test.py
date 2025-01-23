'''
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

sillsquare[sillsquare==''] = -1

for g in trange(20):
    quer = '_'+str(g)+'s'
    sillsquare[rool.index_finder(sillsquare,quer)] = g


T_field[sillsquare!=-100] = 1000
plt.imshow(np.array(sillsquare, dtype = float))
plt.colorbar(orientation = 'horizontal')
plt.show()

sill = np.max(sillsquare)
sillsquare[sillsquare==-1] = ''
print(sillsquare[sillsquare!=''])

sills_number = sillsquare.copy()
sills_number[sills_number==''] = -1
curr_sill = 8
T_solidus = 800
condition = (T_field>T_solidus) & (sills_number!=-1) & (sills_number!=curr_sill)
print(condition.shape)
array = np.zeros_like(T_field) + condition*1
plt.imshow(array)
plt.show()



#for i in trange(sill+1):
sills_data = sc.check_closest_sill_temp(T_field, sillsquare, 6)
print(sills_data)
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import KDTree
import TuLIP

a = 100
b = 500

T_field = np.zeros((a,b))

a1_start = 30
a1_end = 50
a2_start = 50
a2_end = 70

b1_start = 150
b1_end = 250
b2_start = 300
b2_end = 400

T_field[a1_start:a1_end, b1_start:b1_end] = 1100
T_field[a2_start:a2_end, b2_start:b2_end] = 1100

#plt.imshow(T_field)

silli = np.zeros_like(T_field)
silli[T_field==1100] = 1

silli[a2_start:a2_end, b2_start:b2_end] = 2


rock = np.zeros_like(T_field, dtype = 'object')
rock[:] = 'sandstone'
rock[T_field!=0] = 'basalt'
dx = 50
dy = 50
k = np.ones_like(T_field)*31.536

dt = (dx**2)/(5*k[0,0])

density = np.ones_like(T_field)*2700
density[rock=='basalt'] = 2850

iters = 1000
time_steps = np.arange(0,iters,1)*dt
cool = TuLIP.cool()
Cp = 850

H = TuLIP.rules.get_latH(T_field, rock)/(density*Cp)
#plt.imshow(H)
#plt.colorbar(orientation = 'horizontal')
#plt.show()
#print(np.max(H))
all_H = np.zeros(iters)
max_T = np.zeros(iters)
max_T_noH = np.zeros(iters)
max_T_totH = np.zeros(iters)
T_field1 = T_field.copy()
T_field2 = T_field.copy()
for l in range(iters):
    T_field1 = cool.diff_solve(k, a, b, dx, dy, dt, T_field1, q = np.nan, method = 'conv smooth',  H = np.zeros_like(T_field))
    T_field = cool.diff_solve(k, a, b, dx, dy, dt, T_field, q = np.nan, method = 'conv smooth',  H = H)
    H = TuLIP.rules.get_latH(T_field, rock)/(density*Cp)
    tot_H = H + TuLIP.rules.get_radH(T_field, density,dx)/(density*Cp)
    T_field2 = cool.diff_solve(k, a, b, dx, dy, dt, T_field2, q = np.nan, method = 'conv smooth',  H = tot_H)
    print('T', np.max(T_field), 'H',np.max(H))
    #plt.imshow(T_field)
    #plt.colorbar(orientation = 'horizontal')
    #plt.show()
    all_H[l] = np.max(H)
    max_T[l] = np.max(T_field)
    max_T_noH[l] = np.max(T_field1)
    max_T_totH[l] = np.max(T_field2)

plt.plot(time_steps, all_H, label = 'H')
plt.legend()
plt.show()

plt.plot(time_steps, max_T, label = 'T with latent heat')
plt.plot(time_steps, max_T_noH, label = 'T without latent heat')
plt.plot(time_steps, max_T_totH, label = 'T with latent and radiogenic heat')
plt.legend()
plt.show()
