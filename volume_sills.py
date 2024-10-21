import numpy as np
import matplotlib.pyplot as plt
import rule_functions as rool
from tqdm import trange
#Dimensions of the 2D grid
x = 300000 #m - Horizontal extent of the crust
y = 35000 #m - Vertical thickness of the crust

dx = 200 #m node spacing in x-direction
dy = 200 #m node spacing in y-direction

a = int(y//dy) #Number of rows
b = int(x//dx) #Number of columns

#Initializing diffusivity field
k = np.ones((a,b))*31.536 #m2/yr

dt = (min(dx,dy)**2)/(5*np.max(k))

#Shape of the sills
shape = 'elli'

min_thickness = 900 #m
max_thickness = 3500 #m

mar = 7
sar = 2.5

min_emplacement = 1500 #m
max_emplacement = 25500 #m

flux = int(30e9) #m3/yr
total_volume = np.linspace(0.05, 0.5, 10)
n_simulations = 1000
sills_emplaced = np.zeros((len(total_volume), n_simulations))
for eye in range(len(total_volume)):
    for jay in trange(n_simulations):
        n_sills = 20000
        empl_heights = rool.randn_heights(n_sills, max_emplacement, min_emplacement, 5000, dy)
        #Checking to see if there are any assignments outside the distribution#
        n = 0
        while ((empl_heights>max_emplacement/dy).any() or (empl_heights<min_emplacement/dy).any()):
            #print('Heights')
            #print((len(empl_heights[empl_heights>max_emplacement/dy])+len(empl_heights[empl_heights<(min_emplacement/dy)]))*100/n_sills, '%')
            n = int(n+1)
            #print('Cycle', n ,'reassigning')
            if (empl_heights>max_emplacement/dy).any():
                empl_heights[empl_heights>max_emplacement/dy] = rool.randn_heights(np.sum(empl_heights>max_emplacement/dy), max_emplacement, min_emplacement, 5000, dy)
            if (empl_heights<min_emplacement/dy).any():
                empl_heights[empl_heights<min_emplacement/dy] = rool.randn_heights(np.sum(empl_heights<(min_emplacement/dy)), max_emplacement, min_emplacement, 5000, dy)
        #plt.hist(empl_heights*dy)
        #plt.show()

        x_space = rool.x_spacings(n_sills, x//3, 2*x//3, x//6, dx)
            
        n = 0
        while ((x_space>0.66*x/dx).any() or (x_space<(x//(3*dx))).any()):
            #print((len(x_space[x_space>0.66*x/dx])+len(x_space[x_space<(x//(3*dx))]))*100/n_sills, '%')
            #print('Horizontal space')
            n = int(n+1)
            #print('Cycle', n ,'reassigning')
            if (x_space>0.66*x/dx).any():
                x_space[x_space>0.66*x/dx] = rool.x_spacings(np.sum(x_space>0.66*x/dx), x//3, 2*x//3, x//6, dx)
            if (x_space<(x//(3*dx))).any():
                x_space[x_space<(x//(3*dx))] = rool.x_spacings(np.sum(x_space<(x//(3*dx))), x//3, 2*x//3, x//6, dx)
        #plt.hist(x_space*dx)
        #plt.show()
        width, thickness = rool.randn_dims(min_thickness, max_thickness, 700, mar, sar, n_sills)


        tot_volume = int(total_volume[eye]*1e6*1e9) #m3
        total_empl_time = tot_volume/flux
        thermal_maturation_time = int(0.25e6) #yr

        model_time = total_empl_time+thermal_maturation_time+50000
        time_steps = np.arange(model_time,step=dt)
        empl_times = []
        plot_time = []
        cum_volume = []

        if shape == 'elli':
            volume = (4*np.pi/3)*width*width*thickness
        elif shape=='rect':
            volume = width*width*thickness

        unemplaced_volume = 0
        #print(f'{np.sum(volume):.5e}, {float(tot_volume):.5e}, {np.sum(volume)<tot_volume}')
        #print('Time steps:', len(time_steps))
        n = 0
        for l in range(len(time_steps)):
            if time_steps[l]<thermal_maturation_time:
                continue
            else:
                unemplaced_volume += flux*dt
                if unemplaced_volume>=volume[n]:
                    empl_times.append(time_steps[l])
                    plot_time.append(time_steps[l])
                    cum_volume.append(volume[n])
                    unemplaced_volume -= volume[n]
                    #print(f'Emplaced sill {n} at time {time_steps[l]}')
                    #print(f'Remaining volume to emplace: {tot_volume-np.sum(volume[:n]):.4e}')
                    n+=1
                    while unemplaced_volume>0:
                        empl_times.append(time_steps[l])
                        unemplaced_volume -= volume[n]
                        cum_volume[-1]+=volume[n]
                        #print(f'Emplaced sill {n} at time {time_steps[l]}')
                        #print(f'Remaining volume to emplace: {tot_volume-np.sum(volume[:n]):.4e}')
                        n+=1

                if (n>0) and (np.sum(volume[0:n-1])>tot_volume):
                    #print('Total sills emplaced:', n)
                    n_sills = n
                    empl_heights = empl_heights[0:n_sills]
                    x_space = x_space[0:n_sills]
                    width = width[0:n_sills]
                    thickness = thickness[0:n_sills]
                    break
        sills_emplaced[eye, jay] = n_sills
    
mean_no_sills = np.mean(sills_emplaced, axis = 1)
stdev_sills = np.std(sills_emplaced, axis = 1)
print(len(total_volume), len(mean_no_sills))
plt.errorbar(total_volume, mean_no_sills, yerr = stdev_sills, fmt='ro-', capsize=5, label='Number of sills')
plt.fill_between(total_volume,np.min(sills_emplaced, axis = 1), np.max(sills_emplaced, axis=1), alpha = 0.5, color = '#b3d9ff')
plt.xlabel(r'Volume of sills 10$^6 km$^3$')
plt.ylabel('Number of sills')
plt.show()
