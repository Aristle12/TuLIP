import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange

def CO2_adder(time_steps, current_time, tot_RCO2, curr_co2):
    for i in range(len(time_steps)):
        if current_time>time_steps[i] and current_time<time_steps[i+1]:
            if current_time-time_steps[i]<=time_steps[i+1]-current_time:
                tot_RCO2[i] += curr_co2
            else:
                tot_RCO2[i+1] += curr_co2
            break
    return tot_RCO2


target_volume = int(1e6)*int(1e9) #m3/year

fluxes = [int(3e7), int(3e8), int(3e9)] #m3/year

times_csv = pd.read_csv('sillcubes/'+str(format(fluxes[0], '.3e'))+'/300/times.csv')
time_steps = times_csv['time_steps']
area_per_cube = 300*30 #km

n = 10


tot_RO2 = np.zeros_like(time_steps)
mont_RCO2= np.array([])
total_area = 0
current_volume = 0 #m3/year
remaining_volume = target_volume - current_volume

for j in trange(n):
    while current_volume<target_volume:
        flux_index = np.random.randint(0,2)
        flux = fluxes[flux_index]

        print(f'Chosen flux is {flux:.3e}')

        if flux==int(3e9):
            itera = np.random.randint(0,3)
        else:
            itera = np.random.randint(0,2)

        load_dir = 'sillcubes/'+str(format(flux, '.3e'))

        volumes = pd.read_csv(load_dir+'/n_sills.csv')
        volume = volumes['tot_volume'][itera]

        print(f'Volume is {volume:.3e}')

        scales = pd.read_csv(load_dir+'scales.csv')
        scale = scales['scale'][itera]
        volume_cube = scales['tot_volume'][itera]

        times = pd.read_csv(load_dir+'/300/times.csv')

        RCO2_cube = np.array(times['tot_RCO2'])
        time_cube = np.array(times['time_steps'])

        for l in range(len(time_cube)):
            tot_RCO2 = CO2_adder(time_steps, time_cube[l], tot_RCO2, RCO2_cube[l])
        current_volume+=volume_cube
        remaining_volume-=volume
        total_area+=area_per_cube
        print(f'Currently, we have emplaced {current_volume:.3e} $m^3$ over {total_area} $m^2')
        
        if len(mont_RCO2)==0:
            mont_RCO2=tot_RCO2.reshape(1,-1)
        else:
            np.append(mont_RCO2,tot_RCO2.reshape(1, -1), axis = 0)

province_RCO2_mean = np.mean(mont_RCO2, axis = 0)
province_RCO2_std = np.std(mont_RCO2, axis = 0)

print(len(province_RCO2_mean), len(time_steps))

plt.errorbar(time_steps, province_RCO2_mean, yerr = province_RCO2_std)
plt.show()