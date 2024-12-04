import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context('poster')

Ctot = 4.1261e19

pco2i = 1000

# Constants
Mrat = 7.8
DICcorr = 1.1
Matm = 1.77e20
C_molmass = 12

# Calculate initial total carbon
Ctot_i = ((0.23 + 2233 / (pco2i + 34)) * Mrat * DICcorr + 1) * Matm * 1e-6 * pco2i * 12

# Create a DataFrame for pCO2
pCO2 = np.linspace(pco2i, 9500, 100)  # Example range for pCO2
Cadd = pd.DataFrame({'pCO2': pCO2})

# Calculate Cadd and CO2add
Cadd['Cadd'] = (Matm * 1e-6 * Cadd['pCO2'] * C_molmass * ((0.23 + 2233 / (Cadd['pCO2'] + 34)) * Mrat * DICcorr + 1) - Ctot_i) / 1e12
Cadd['CO2add'] = Cadd['Cadd'] * 44 / 12
print(np.array(Cadd['pCO2'])[-1])
print(Cadd.shape)
print(np.array(Cadd['CO2add'])[-1])
'''
# Second case with different initial pCO2
pco2i = 500

# Calculate initial total carbon for the second case
Ctot_i = ((0.23 + 2233 / (pco2i + 34)) * Mrat * DICcorr + 1) * Matm * 1e-6 * pco2i * 12

# Create a DataFrame for the second case
Cadd2 = pd.DataFrame({'pCO2': pCO2})

# Calculate Cadd and CO2add for the second case
Cadd2['Cadd'] = (Matm * 1e-6 * Cadd2['pCO2'] * C_molmass * ((0.23 + 2233 / (Cadd2['pCO2'] + 34)) * Mrat * DICcorr + 1) - Ctot_i) / 1e12
Cadd2['CO2add'] = Cadd2['Cadd'] * 44 / 12

# Add CO2add_500 to the first DataFrame
Cadd['CO2add_500'] = Cadd2['CO2add']

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(Cadd['CO2add'], Cadd['pCO2'])
plt.xlabel('Carbon added (Teragram)')
plt.ylabel('Final pCO2 (ppm)')
#plt.xlim(0, 1.5e7)
plt.title('Carbon added vs Final pCO2')
plt.show()

# Using seaborn for the second plot
plt.figure(figsize=(10, 6))
sns.lineplot(x=Cadd['CO2add'] / 1000, y=Cadd['pCO2'] - 1000)
sns.lineplot(x=Cadd['CO2add_500'] / 1000, y=Cadd['pCO2'] - 500, color='blue')
plt.xlabel('CO2 added (Petagram)')
plt.ylabel('pCO2 increase (ppm)')
plt.xlim(0, 60000)
plt.title('CO2 added vs pCO2 increase')
plt.show()
'''
from scipy.interpolate import interp1d

p = interp1d(Cadd['CO2add'], Cadd['pCO2'], kind='linear')
#p = np.poly1d(z)
fluxs = [int(3e7), int(3e8), int(3e9)]
iters = [0,1,2]




'''
plt.figure(figsize=(10, 6))
plt.plot(Cadd['CO2add'],p(Cadd['CO2add']))
plt.xlabel('Carbon added (Teragram)')
plt.ylabel('Final pCO2 (ppm)')
#plt.xlim(0, 1.5e7)
plt.title('Carbon added vs Final pCO2')
plt.show()
'''



LIP_volume = int(1e5)*int(1e9) #m3
LIP_timescale = int(2e4) #years

conf_array = np.zeros((len(fluxs), len(iters)))
for i in range(len(fluxs)):
    for j in range(len(iters)):
        flux = fluxs[i]
        itera = iters[j]
        load_dir = 'sillcubes/'+str(format(flux,'.3e'))+'/'

        n_sills = pd.read_csv(load_dir+'n_sills.csv')
        volumes = n_sills['volumes']
        volume = volumes[itera]
        print(format(volume, '.3e'))

        z_index = [191, 284, 300, 493, 506]

        times = pd.read_csv(load_dir+str(format(volume, '.3e'))+'/300/times.csv')
        time = times['time_steps']
        dts = np.array([time[i]-time[i-1] for i in range(1, len(time))])
        dts = np.append(dts, dts[-1])
        tot_RCO2 = (times['tot_RCO2']-times['tot_RCO2'][0])*dts 
        cum_RCO2 = np.cumsum(tot_RCO2)

        scales_csv = pd.read_csv(load_dir+'scales.csv')
        scale = scales_csv['scale'][itera]
        inter = scales_csv['intercept'][itera]
        volume_scale = scales_csv['tot_volume'][itera]
        print(format(volume_scale, '.3e'))

        empl_params = pd.read_csv(load_dir+'emplacement_params'+str(float(volume))+'.csv')
        empl_time = np.array(empl_params['empl_times'])[-1]-int(3e6)


        LIP_flux = 10*int(1e9)#LIP_volume/empl_time
        #print(f'LIP flux is {LIP_flux:.3e}')


        flux_real = volume_scale/empl_time #m3/yr
        scale_factor_flux = LIP_flux/flux_real
        #print(f'Real flux is {flux_real:.3e}')
        #print(f'Scale factor is {scale_factor_flux}')

        CO2 = (scale*volume_scale + inter)*scale_factor_flux*LIP_timescale/empl_time

        #tot_CO2 = np.array(cum_RCO2)[-1]/int(1e9)
        #print(format(CO2/int(1e9), '.3e'))
        pCO2_calc = p(CO2/int(1e9))
        #print(pCO2_calc)
        ECS1 = 3

        deg_change = np.log2(pCO2_calc/pco2i)*ECS1
        conf_array[i,j] = deg_change 
        print(f'Degree change for {flux:.3e} and {volume:.3e} is {deg_change}')

plt.figure(figsize =[9,6])
sns.heatmap(conf_array.T, annot=True, cmap = 'RdBu_r', fmt = '.3f', xticklabels = np.array(fluxs)/int(1e9), yticklabels = np.array(volumes[0:3])/int(1e9))
plt.xlabel(r'Flux $km^3$')
plt.ylabel(r'Volume $km^3$')
plt.title(r'Warming in ${^\circ}C$')
plt.tight_layout()
plt.savefig('sillcubes/present_plots/warming_summary.png', format = 'png')