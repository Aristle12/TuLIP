import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pCO2 = np.arange(500,2500,1)
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
pCO2 = np.linspace(pco2i, 2000, 100)  # Example range for pCO2
Cadd = pd.DataFrame({'pCO2': pCO2})

# Calculate Cadd and CO2add
Cadd['Cadd'] = (Matm * 1e-6 * Cadd['pCO2'] * C_molmass * ((0.23 + 2233 / (Cadd['pCO2'] + 34)) * Mrat * DICcorr + 1) - Ctot_i) / 1e12
Cadd['CO2add'] = Cadd['Cadd'] * 44 / 12

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
'''
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(Cadd['CO2add'], Cadd['pCO2'])
plt.xlabel('Carbon added (Teragram)')
plt.ylabel('Final pCO2 (ppm)')
#plt.xlim(0, 1.5e7)
plt.title('Carbon added vs Final pCO2')
plt.show()
'''
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
flux = int(3e9)

plt.figure(figsize=(10, 6))
plt.plot(Cadd['CO2add'],p(Cadd['CO2add']))
plt.xlabel('Carbon added (Teragram)')
plt.ylabel('Final pCO2 (ppm)')
#plt.xlim(0, 1.5e7)
plt.title('Carbon added vs Final pCO2')
plt.show()

load_dir = 'sillcubes/'+str(format(flux,'.3e'))+'/'

n_sills = pd.read_csv(load_dir+'n_sills.csv')
volumes = n_sills['volumes']
volume = volumes[2]

z_index = [191, 284, 300, 493, 506]

times = pd.read_csv(load_dir+str(format(volume, '.3e'))+'/300/times.csv')
time = times['time_steps']
dts = np.array([time[i]-time[i-1] for i in range(1, len(time))])
dts = np.append(dts, dts[-1])
tot_RCO2 = times['tot_RCO2']*dts
cum_RCO2 = np.cumsum(tot_RCO2)

tot_CO2 = np.array(cum_RCO2)[-1]/int(1e9)
print(tot_CO2)
pCO2_calc = p(tot_CO2)
print(pCO2_calc)
ECS1 = 3

deg_change = np.log2(pCO2_calc/pco2i)*ECS1

print(deg_change)

