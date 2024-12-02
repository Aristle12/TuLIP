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
pCO2 = np.linspace(0, 2000, 100)  # Example range for pCO2
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

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(Cadd['Cadd'], Cadd['pCO2'])
plt.xlabel('Carbon added (Teragram)')
plt.ylabel('Final pCO2 (ppm)')
plt.xlim(0, 1.5e7)
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

slope2, slope1, intercept = np.polyfit(Cadd['pCO2'], Cadd['Cadd'], deg = 2)

