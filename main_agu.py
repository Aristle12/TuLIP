import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions as cool
import rule_functions as rool
from tqdm import trange
import h5py

x = 300000 #m
y = 35000 #m

dx = 200
dy = 200

a = int(y//dy)
b = int(x//dx)

mu = np.ones((a,b))*31.536 #m2/yr
T_surf = 0
T_bot = 1200
T_field = np.zeros_like(mu)
T_field[-1,:] = T_bot
T_field = cool.heat_flux(mu, a, b, dx, dy, T_field, 'straight')
plt.imshow(T_field)
plt.show()
n_sills = 10000

##Initiating lithology
rock = pd.DataFrame( data = [], index = range(a), columns = range(b))
rock[:] = 'granite'

start_flux = int(30e9) 
end_flux = int(300e9)
flux_rates = np.arange(start_flux, end_flux, 30e9) #m3/yr

tot_volume = int(1e6*1e9) #m3


min_thickness = 900 #m
max_thickness = 3500 #m

mar = 3.33
sar = 0.75

min_emplacement = 1500 #m
max_emplacement = 25500 #m

dike_net = np.zeros_like(T_field)



for k in range(0, len(flux_rates)):
    times = np.array([])
    emplace_window = int(tot_volume/flux_rates[k]) #years
    empl_heights = rool.randn_heights(n_sills, max_emplacement, min_emplacement, 5000, dy)
    n = 0
    print('Heights')
    while ((empl_heights>max_emplacement/dy).any() or (empl_heights<min_emplacement/dy).any()):
        print((len(empl_heights[empl_heights>max_emplacement/dy])+len(empl_heights[empl_heights<(min_emplacement/dy)]))*100/n_sills, '%')
        n = int(n+1)
        print('Cycle', n ,'reassigning')
        if (empl_heights>max_emplacement/dy).any():
            empl_heights[empl_heights>max_emplacement/dy] = rool.randn_heights(np.sum(empl_heights>max_emplacement/dy), max_emplacement, min_emplacement, 5000, dy)
        if (empl_heights<min_emplacement/dy).any():
            empl_heights[empl_heights<min_emplacement/dy] = rool.randn_heights(np.sum(empl_heights<(min_emplacement/dy)), max_emplacement, min_emplacement, 5000, dy)
    plt.hist(empl_heights*dy)
    plt.show()


    x_space = rool.x_spacings(n_sills, x//3, 2*x//3, x//6, dx)
    
    n = 0
    while ((x_space>0.66*x/dx).any() or (x_space<(x//(3*dx))).any()):
        print((len(x_space[x_space>0.66*x/dx])+len(x_space[x_space<(x//(3*dx))]))*100/n_sills, '%')
        print()
        n = int(n+1)
        print('Cycle', n ,'reassigning')
        if (x_space>0.66*x/dx).any():
            x_space[x_space>0.66*x/dx] = rool.x_spacings(np.sum(x_space>0.66*x/dx), x//3, 2*x//3, x//6, dx)
        if (x_space<(x//(3*dx))).any():
            x_space[x_space<(x//(3*dx))] = rool.x_spacings(np.sum(x_space<(x//(3*dx))), x//3, 2*x//3, x//6, dx)
    plt.hist(x_space*dx)
    plt.show()
    width, thickness = rool.randn_dims(min_thickness, max_thickness, 700, mar, sar, n_sills)
    volume = (4*np.pi/3)*width*width*thickness
    n_reps = 0
    t_list = np.array([])
    rem_volume = tot_volume
    for l in range(0, n_sills):
        time = volume/flux_rates[k]
        t_list = np.append(t_list, time)
        rem_volume = rem_volume - volume[l]
        n_reps = n_reps+1
        if rem_volume<=0:
            break

    dt = (dx**2)/(5*mu[0,0])
    print('Time Step:', dt)
    total_time = emplace_window + int(1e6) #years
    t_steps = int(total_time//dt + n_reps)
    t_now = 0
    n_now = 0
    t_empl = t_list[n_now]
    t_model = 0
    method = 'conv smooth'
    H = np.zeros_like(T_field)
    T_f = np.empty((t_steps, a, b))
    dike_nets = np.empty((t_steps, a, b))
    for i in trange(0, t_steps):
        if np.abs(t_now-t_empl)<dt:
            dtee = np.abs(t_now-t_empl)
        else:
            dtee = dt
        t_now = t_now + dtee
        t_model = t_model+dtee
        T_field = cool.diff_solve(mu, a, b, dx, dy, dtee, T_field, np.nan, method, H)
        if n_now<n_reps:
            if rool.to_emplace(t_now, t_list[n_now]):
                T_field, dike_net, rock = rool.mult_sill(T_field, width[n_now]//dx, thickness[n_now]//dy, empl_heights[n_now], x_space[n_now], a, b, dx, dy,  dike_net, rock = rock, shape = 'elli')
                n_now = n_now+1
                t_now = 0
                t_empl = t_now+t_list[n_now]
        times = np.append(times, t_now)
        T_f[i,:,:] = T_field
        dike_nets[i,:,:] = dike_net
    T_h5 = h5py.File('TempField'+'_'+str(k)+'.hdf5', 'w')
    T_h5.create_dataset('Temp_field', data = T_f)
    T_h5.create_dataset('dike_net', data = dike_nets)
    T_h5.create_dataset('time', data=times)
    T_h5.attrs['flux'] = flux_rates[k]