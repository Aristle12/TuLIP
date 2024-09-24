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

###Setting the initial temperature grid###
T_field = np.zeros_like(mu)
T_field[-1,:] = T_bot
T_field = cool.heat_flux(mu, a, b, dx, dy, T_field, 'straight')

##Initiating lithology
rock = pd.DataFrame( data = [], index = range(a), columns = range(b))
rock[:] = 'granite' 
rock.loc[0:int(a//2),:] = 'shale'
rock.loc[int(a//2)+1:int(a),:] = 'sandstone'


##Initiating TOC
TOC = np.zeros_like(mu)
TOC[rock=='shale'] = 7
TOC[rock=='sandstone'] = 5


##Initiating density
density = np.zeros_like(mu)
density[rock=='shale'] = 2400 #kg/m3
density[rock=='sandstone'] = 2600 #kg/m3

##Initiating porosity

porosity = np.zeros_like(mu)
porosity[rock=='shale'] = 0.1
porosity[rock=='sandstone'] = 0.25


time = 500000 #years
dt = (dx**2)/(5*mu[0,0])
t_steps = np.arange(0, time, dt)
H = np.zeros_like(T_field)

x_coords = int(b//2)
y_coords = int(a//2)

t_empl = t_steps[999]
curr_time = 0

tot_RCO2 = np.zeros(len(t_steps))
tot_RCO2_silli = np.zeros(len(t_steps))

for l in trange(0,len(t_steps)):
    curr_time = t_steps[l]
    #print('Current time:',curr_time, t_steps[l])
    T_field = cool.diff_solve(mu,a,b,dx,dy,dt,T_field,q=np.nan,method = 'conv smooth', H=H)
    ##carbon emissions calculation step
    if l==0:
        RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli = emit.SILLi_emissions(T_field, density, rock, porosity, TOC, dt)
        RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions = emit.sillburp(T_field, TOC, density, rock, porosity, dt)
    else:
        RCO2_silli, Rom_silli, percRo_silli, curr_TOC_silli, W_silli = emit.SILLi_emissions(T_field, density, rock, porosity, curr_TOC_silli, dt, TOC, W_silli)
        #if (progress_of_reactions>=1).all():
        #    print('Carbon is over')
        #print(progress_of_reactions[(progress_of_reactions!=0) & (progress_of_reactions!=1)])
        RCO2, Rom, progress_of_reactions, oil_production_rate, curr_TOC, rate_of_reactions = emit.sillburp(T_field, curr_TOC, density, rock, porosity, dt, TOC, oil_production_rate, progress_of_reactions, rate_of_reactions)
    if curr_time==t_empl:
        print('Sill emplaced')
        T_field = rool.single_sill(T_field,x_coords,y_coords, 5000//dx, 3000//dy, T_bot)
    tot_RCO2[l] = sum(np.sum(RCO2))
    tot_RCO2_silli[l] = sum(np.sum(RCO2_silli))
print(tot_RCO2[tot_RCO2!=0])
#plt.imshow(np.sum(np.sum(progress_of_reactions, axis = 0), axis = 0))
#plt.colorbar()
#plt.show()
plt.plot(t_steps[1:-1], np.log10(tot_RCO2[1:-1]), label = 'sillburp')
plt.plot(t_steps[1:-1], np.log10(tot_RCO2_silli[1:-1]), label = 'silli')
plt.legend()
plt.savefig('cabon_emissions.png', format = 'png')

pd.DataFrame({'Time': t_steps, 'CO2_sillburp': tot_RCO2, 'CO2_silli': tot_RCO2_silli}).to_csv('carbon.csv')