import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from tqdm import trange
from math import dist
import time as tl

flux_rates = np.arange(1000000/300, 1500000/300, 100000/300) #m2/yr


print('Time recorded')
for i in range(0, len(flux_rates)):
    start_time = tl.time()
    T_f = h5py.File('TempField_'+str(i)+'.hdf5', 'r')
    print(tl.time() - start_time, ' seconds to read file')
    T_field = np.array(T_f['Temp_field'])
    dike_nets = np.array(T_f['dike_net'])
    tyme = np.array(T_f['time'])
    print(tyme)
    time_target = [int(1e6), int(1.5e6) , int(2e6)]
    diff = np.repeat([int(1e9)],len(time_target))
    times = np.zeros_like(time_target)
    target_index = np.zeros_like(time_target)
    for j in range(0, len(time_target)):
        for k in range(0, len(tyme)):
            diff1 = np.abs(tyme - time_target[j])
            diff[j] = np.min(diff1)
            stime = tyme[diff1==diff[j]]
            times[j] = stime[0]
    print(tl.time() - start_time, 'seconds to calculate minimum')
    print(stime)
    print(times)

    for j in range(0, len(time_target)):
        Temp = T_field[target_index[j], :, :]
        magma_map = dike_nets[target_index[j], :, :]
        T_hist = np.array([])
        a = len(Temp[:,0])
        b = len(Temp[0,:])
       
        for eye in range(0,a):
            for k in range(0, b):
                dist = np.empty_like(T_field)
                dist[:] = np.nan
                for l in range(0,a):
                    for m in range(0,b):
                        if magma_map[l,m]==1:
                            dist[l,m] == dist([eye,k],[l,m])*200
            inds = dist>4900 & dist<5100
            T_hist = T_hist.append(Temp[inds])
        print(tl.time() - start_time, 'seconds to get Temps')
        plt.hist(T_hist)
        plt.savefig('hisT_'+str(flux_rates[i])+'.png', format = 'png')