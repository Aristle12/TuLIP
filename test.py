from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

a = loadmat('dat/Dolostone.mat')
#print(a.keys())

#print(a['Dolo']['CO2'])

dolo = loadmat('dat/Dolostone.mat')
evap = loadmat('dat/DolostoneEvaporite.mat')
marl = loadmat('dat/Marl.mat')
T = np.array(dolo['Dolo']['T'][0][:][0])
P = np.array(dolo['Dolo']['P'][0][0][0])
#P1 = np.array(P[0][0][0])
T = T[:,0]
print(T)
print(P)
dolo_CO2 = np.array(dolo['Dolo']['CO2'][0][0])
evap_co2 = np.array(evap['Dol_ev']['CO2'][0][0])
marl_CO2 = np.array(marl['Marl']['CO2'][0][0])

dolo_inter = RegularGridInterpolator((T,P), marl_CO2)
a = dolo_inter([752.6400000000001, 30])
print(a)
Tg, Pg = np.meshgrid(T, P, indexing = 'ij')
points = np.vstack([Tg.ravel(), Pg.ravel()]).T
co2_array = dolo_inter(points)
co2_array = co2_array.reshape(Tg.shape)
fig, (ax1, ax2) = plt.subplots(1,2)

ax1.imshow(marl_CO2)
ax1.set_xlabel('T')
ax1.set_ylabel('P')
ax1.set_title('Original function')


ax2.imshow(co2_array)
ax2.set_xlabel('T')
ax2.set_ylabel('P')
ax2.set_title('Interpolated function')


plt.show()