import numpy as np
from numba import jit

def SILLi_emissions(T_field, dT, density, lithology, porosity, I_prev, TOC_prev, dt, TOCo):
    '''
    Python implementation of SILLi (Iyer et al. 2018) based on the EasyRo% method of Sweeney and Burnham (1990)
    T_field - temperature field (array)
    dT = Rate of cooloing array
    density - Rock density array
    lithology - Lithology array
    porosity - porosity array
    '''
    a = len(T_field[:,0])
    b = len(T_field[0,:])
    calc_parser = lithology[lithology=='shale' or lithology=='sandstone']
    A = 1e13
    a1 = 2.334733
    a2 = 0.250621
    b1 = 3.330657
    b2 = 1.681534
    R = 1.9872036e-3 #kcal/K/mol
    E = [34, 36, 38, 40, 72] #kcal/m
    f = [0.03, 0.03, 0.04, 0.01]
    total = 1 - np.sum(f)
    I_curr = np.empty_like(I_prev)
    del_I = np.empty_like(I_prev)
    w_ratio = np.empty_like(E)
    fl = np.empty_like(E)
    for l in range(len(I_prev[:,0,0])):
        Ert = E[l]/(R*T_field)
        I_curr[l,:,:] = T_field*A*np.exp(Ert)*(1-((Ert**2+(a1*Ert)+a2)/(Ert**2+(b1*Ert)+b2)))
        del_I[l] = (I_curr[l]-I_prev[l])/dT
        w_ratio[l] = 1 - np.exp(-del_I[l])
        fl[l] = w_ratio[l]*f[l]
    Frac = 1 - np.sum(fl)
    percRo = np.exp(-1.6+3.7*Frac) #vitrinite reflectance
    TOC = TOCo*Frac*calc_parser
    dTOC = (TOC-TOC_prev)/dt
    Rom = (1-porosity)*density*dTOC
    RCO2 = Rom*3.67
    return RCO2, Rom, percRo, I_curr, TOC

def SILLi_I(T_field):
    '''
    Initialization of I for the carbon model
    '''
    a = len(T_field[:,0])
    b = len(T_field[0,:])
    A = 1e13
    a1 = 2.334733
    a2 = 0.250621
    b1 = 3.330657
    b2 = 1.681534
    R = 1.9872036e-3 #kcal/K/mol
    E = [34, 36, 38, 40, 72] #kcal/m
    f = [0.03, 0.03, 0.04, 0.01]
    I = np.empty_like(len(E),a,b)
    for l in range(len(E)):
        Ert = E[l]/(R*T_field)
        I[l,:,:] = T_field*A*np.exp(Ert)*(1-((Ert**2+(a1*Ert)+a2)/(Ert**2+(b1*Ert)+b2)))
    return I