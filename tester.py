import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from TuLIP import sill_controls, emit

sc = sill_controls()
x = 70000 #m
y = 12000 #m

dx = 50 #m
dy = 50 #m

a = int(y//dy)
b = int(x//dx)

k = np.ones((a,b))*31.536 #m2/yr

dt = (min(dx,dy)**2)/(5*np.max(k))

#Shape of the sills
shape = 'elli'

#Initializing the temp field
T_field = np.zeros((a,b))

q = k[-1,:]*(30/1000)

T_field = sc.cool.heat_flux(k, a, b, dx, dy, T_field, 'straight', q)

plt.imshow(T_field)
plt.colorbar(orientation = 'horizontal')
plt.show()

lith_plot_dict = {'granite':0,
                  'shale':1,
                  'sandstone':2,
                  'peridotite':3,
                  'basalt':4}

magma_prop_dict = {'Temperature': 1000,
             'Lithology': 'basalt',
             'Porosity': 0.2,
             'Density': 2850, #kg/m3
             'TOC':0} #wt%

rock_prop_dict = {
    "shale":{
        'Porosity':0.1,
        'Density':2500,
        'TOC':2
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
    "basalt":{
        'Porosity': 0.2,
        'Density': 2850, #kg/m3
        'TOC':0
    },
    "peridotite":{
        'Porosity': 0.05,
        'Density': 3100, #kg/m3
        'TOC':0
    }

}

rock = np.zeros(a,b, dtype = object)
rock[:] = 'granite'
rock[0:a//4,:] = 'shale'

density = np.zeros(a,b, dtype = object)
density[rock=='shale'] = rock_prop_dict['shale']['Density']
density[rock=='granite'] = rock_prop_dict['granite']['Density']

porosity = np.zeros(a,b, dtype = object)
porosity[rock=='shale'] = rock_prop_dict['shale']['Porosity']
porosity[rock=='granite'] = rock_prop_dict['granite']['Porosity']

TOC = np.zeros(a,b, dtype = object)
TOC[rock=='shale'] = rock_prop_dict['shale']['TOC']
TOC[rock=='granite'] = rock_prop_dict['granite']['TOC']

width, height = [10000//dx, 500//dy]

depth = 0.25*a
x_space = b//2
thermal_maturation_time = int(3e6//dt)*dt
empl_time = 10*dt
cooling_time = 30000
empl_act_time = thermal_maturation_time+empl_time
H = np.zeros(a,b)
method = 'conv smooth'


time_steps = (0, empl_time+thermal_maturation_time+cooling_time, dt)
tot_CO2_silli = np.zeros_like(time_steps)
tot_CO2_sillburp = np.zeros_like(time_steps)

for l in range(len(time_steps)):
    curr_time = time_steps[l]
    T_field = sc.cool.diff_solve(k, a, b, dx, dy, dt, T_field, np.nan, method, H)
    if l==0:
        RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli = emit.SILLi_emissions(T_field, density, rock, porosity, TOC, dt)
        reaction_energies = emit.get_sillburp_reaction_energies()
        RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions = emit.sillburp(T_field, TOC, density, rock, porosity, dt, reaction_energies)
        tot_CO2_silli[l] = np.sum(RCO2_silli)
        tot_CO2_sillburp[l] = np.sum(RCO2)
    else:
        RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli = emit.SILLi_emissions(T_field, density, rock, porosity, curr_TOC_silli, dt, TOC, W_silli)
        RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions = emit.sillburp(T_field, curr_TOC, density, rock, porosity, dt, reaction_energies, TOC, oil_production_rate, progress_of_reactions, rate_of_reactions)
        tot_CO2_silli[l] = np.sum(RCO2_silli)
        tot_CO2_sillburp[l] = np.sum(RCO2)
    if curr_time == empl_act_time:
        T_field, rock, _ = sc.rool.mult_sill(T_field, width, height, depth, x_space, dx, dy, rock, dike_empl=False)
        density = sc.rool.prop_updater(rock, lith_plot_dict, rock_prop_dict, 'Density')
        porosity = sc.rool.prop_updater(rock, lith_plot_dict, rock_prop_dict, 'Porosity')

plt.plot(time_steps, np.log10(tot_CO2_silli/int(1e9)), label = 'silli')
plt.plot(time_steps, np.log10(tot_CO2_sillburp/int(1e9)), label = 'sillburp')
plt.savefig('plots/high_res_compare.png', format = 'png')


    
 

