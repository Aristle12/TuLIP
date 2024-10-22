import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions as cool
import rule_functions as rool
import carbon_emissions as emit
from tqdm import trange

#Length of the grid in x (horizontal) and y (vertical) directions
x = 30000 #m
y = 10000 #m

#Node spacing in x and y directions
dx = 200
dy = 200

#Total number of nodes in y (vertical) and x (horizontal) directions
a = int(y//dy)
b = int(x//dx)
print(a,b)
mu = np.ones((a,b))*31.536 #m2/yr

T_surf = 0
T_bot = 1200

prop_dict = {'Temperature':0,
             'Lithology':1,
             'Porosity':2,
             'Density':3,
             'TOC':4}

emp_index = prop_dict['Temperature']
rock_index = prop_dict['Lithology']
poros_index = prop_dict['Porosity']
dense_index = prop_dict['Porosity']
TOC_index = prop_dict['TOC']

magma_prop_dict = {'Temperature': T_bot,
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
###Setting the initial temperature grid###
T_field = np.zeros_like(mu)
T_field[-1,:] = T_bot
T_field = cool.heat_flux(mu, a, b, dx, dy, T_field, 'straight')

##Initiating lithology
rock = np.zeros((a,b), dtype=object)#pd.DataFrame( data = [], index = range(a), columns = range(b))
rock[:] = 'granite' 
rock[0:int(a//3),:] = 'limestone'
rock[int(a//3)+1:int(2*a//3),:] = 'shale'
rock[int(2*a//3)+1:a,:] = 'sandstone'


##Initiating TOC
TOC = np.zeros_like(mu)
TOC[rock=='shale'] = 7
TOC[rock=='sandstone'] = 5
TOC[rock=='limestone'] = rock_prop_dict['limestone']['TOC']


##Initiating density
density = np.zeros_like(mu)
density[rock=='shale'] = 2400 #kg/m3
density[rock=='sandstone'] = 2600 #kg/m3
density[rock=='limestone'] = rock_prop_dict['limestone']['Density']

##Initiating porosity

porosity = np.zeros_like(mu)
porosity[rock=='shale'] = 0.1
porosity[rock=='sandstone'] = 0.25
porosity[rock=='limestone'] = rock_prop_dict['limestone']['Porosity']


time = 500000 #years
dt = (dx**2)/(5*mu[0,0])
t_steps = np.arange(0, time, dt)
H = np.zeros_like(T_field)

x_coords = int(b//2)
y_coords = int(a//2)

t_empl = t_steps[999]
curr_time = 0

props_array = np.empty((len((prop_dict.keys())),a,b), dtype = object)
#shape_indices = [len(t_steps)]+ list(props_array.shape)
#tot_prop_array = np.empty(shape_indices, dtype = object)
Temp_index = prop_dict['Temperature']
rock_index = prop_dict['Lithology']
poros_index = prop_dict['Porosity']
dense_index = prop_dict['Porosity']
TOC_index = prop_dict['TOC']

cm_array = np.empty((a,b), dtype = object)
cm_array[0:-10,:] = 'crust'
cm_array[-10:,:] = 'mantle'

props_array[Temp_index] = T_field
props_array[rock_index] = rock
props_array[poros_index] = porosity
props_array[dense_index] = density
props_array[TOC_index] = TOC

tot_RCO2 = np.zeros(len(t_steps))
tot_RCO2_silli = np.zeros(len(t_steps))
reaction_energies = emit.get_sillburp_reaction_energies()
push = True
for l in trange(0,len(t_steps)):
    curr_time = t_steps[l]
    #print('Current time:',curr_time, t_steps[l])
    T_field = cool.diff_solve(mu,a,b,dx,dy,dt,T_field,q=np.nan,method = 'conv smooth', H=H)
    ##carbon emissions calculation step
    if l==0:
        RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli = emit.SILLi_emissions(T_field, density, rock, porosity, TOC, dt, dy)
        RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions = emit.sillburp(T_field, TOC, density, rock, porosity, dt, reaction_energies)
        breakdown_CO2 = emit.get_init_CO2_percentages(T_field, rock, density, dy)
    else:
        RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli = emit.SILLi_emissions(T_field, density, rock, porosity, curr_TOC_silli, dt, TOC, W_silli)
        breakdown_CO2, _ = emit.get_breakdown_CO2(T_field, rock, density, breakdown_CO2, dy, dt)
        #if (progress_of_reactions>=1).all():
        #    print('Carbon is over')
        #print(progress_of_reactions[(progress_of_reactions!=0) & (progress_of_reactions!=1)])
        RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions = emit.sillburp(T_field, curr_TOC, density, rock, porosity, dt, reaction_energies, TOC, oil_production_rate, progress_of_reactions, rate_of_reactions)
    if curr_time==t_empl:
        print('Sill emplaced')
        #T_field = rool.single_sill(T_field,x_coords,y_coords, 5000//dx, 3000//dy, T_bot)
        T_field, new_dike = rool.mult_sill(T_field,5000, 3000, y_coords*dy, x_coords*dx, dx, dy, push=push)
        if push == True:
            columns_pushed = np.sum(new_dike, axis =0, dtype=int)
            row_push_start = np.zeros(b, dtype = int)
            for n in range(b):
                for m in range(a):
                    if new_dike[m,n]==1:
                        #if row_push_start[n]==0:
                        row_push_start[n] = m
                        break
            RCO2_silli = rool.value_pusher2D(RCO2_silli,0, row_push_start, columns_pushed)
            Rom_silli = rool.value_pusher2D(Rom_silli,0, row_push_start, columns_pushed)
            percRo_silli = rool.value_pusher2D(percRo_silli,0, row_push_start, columns_pushed)
            curr_TOC_silli = rool.value_pusher2D(curr_TOC_silli,0, row_push_start, columns_pushed)
            W_silli = rool.value_pusher2D(W_silli,0, row_push_start, columns_pushed)

            RCO2 = rool.value_pusher2D(RCO2,0, row_push_start, columns_pushed)
            Rom = rool.value_pusher2D(Rom,0, row_push_start, columns_pushed)
            percRo = rool.value_pusher2D(percRo,0, row_push_start, columns_pushed)
            curr_TOC = rool.value_pusher2D(curr_TOC,0, row_push_start, columns_pushed)
            W = rool.value_pusher2D(W,0, row_push_start, columns_pushed)


    tot_RCO2[l] = np.sum(RCO2)+np.sum(breakdown_CO2)
    tot_RCO2_silli[l] = np.sum(RCO2_silli)+np.sum(breakdown_CO2)
#print(tot_RCO2[tot_RCO2!=0])
#plt.imshow(np.sum(np.sum(progress_of_reactions, axis = 0), axis = 0))
#plt.colorbar()
#plt.show()
#plt.plot(t_steps[1:-1], np.log10(tot_RCO2[1:-1]), label = 'sillburp')
plt.plot(t_steps[1:-1], np.log10(tot_RCO2_silli[1:-1]), label = 'SILLi')
plt.xlabel(r'Time (yr)')
plt.ylabel(r'Carbon emissions log kg/yr')
plt.legend()
plt.savefig('cabon_emissions.png', format = 'png')

pd.DataFrame({'Time': t_steps, 'CO2_sillburp': tot_RCO2, 'CO2_silli': tot_RCO2_silli}).to_csv('carbon.csv')