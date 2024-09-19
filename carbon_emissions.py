import numpy as np
from scipy.special import erf, erfinv

def SILLi_emissions(T_field, density, lithology, porosity, TOC_prev, dt, TOCo=np.nan, W=np.nan):
    '''
    Python implementation of SILLi (Iyer et al. 2018) based on the EasyRo% method of Sweeney and Burnham (1990)
    T_field - temperature field (array)
    dT = Rate of cooloing array
    density - Rock density array
    lithology - Lithology array
    porosity - porosity array
    '''
    calc_parser = (lithology=='shale') | (lithology=='sandstone')
    A = 1e13
    R = 8.314 #J/K/mol
    E = np.array([34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72])*4184 #J/mole
    f = np.array([0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06, 0.04, 0.04, 0.07, 0.06, 0.06, 0.06, 0.05, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01])
    T_field = T_field + 273.15
    if np.isnan(W).all():
        W = np.ones((len(E),len(T_field[:,0]),len(T_field[0,:])))
        TOCo = TOC_prev
    dW = np.empty_like(W)
    fl = np.empty_like(W)
    for l in range(0, len(E)):
        exp_term = np.exp(-E[l]*dt/(R*T_field))
        dW[l,:,:] = -W[l,:,:]*A*exp_term
        fl[l,:,:] = f[l]*dW[l,:,:]
        W[l,:,:] = W[l,:,:] + dW[l,:,:]
    fl_sum = np.sum(fl, axis = 0)
    Frac = 1 - fl_sum
    percRo = np.exp(-1.6+3.7*Frac) #vitrinite reflectance
    TOC = TOCo*Frac*calc_parser
    if (W!=0).all():
        dTOC = (TOC-TOC_prev)/dt
        Rom = (1-porosity)*density*dTOC
        RCO2 = Rom*3.67
    else:
        Rom = 0
        RCO2 = Rom*3.67
    return RCO2, Rom, percRo, TOC, W


def analytical_Ro(T_field, dT, density, lithology, porosity, I_prev, TOC_prev, dt, TOCo, W):
    calc_parser = (lithology=='shale') | (lithology=='sandstone')
    a1 = 2.334733
    a2 = 0.250621
    b1 = 3.330657
    b2 = 1.681534
    A = 1e13
    R = 8.314 #J/K/mol
    E = [34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72]*4184 #J/mole
    f = [0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06, 0.04, 0.04, 0.07, 0.06, 0.06, 0.06, 0.05, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01]
    I_curr = np.empty_like(I_prev)
    del_I = np.empty_like(I_prev)
    w_ratio = np.empty_like(E)
    fl = np.empty_like(E)
    for l in range(len(I_prev[:,0,0])):
        Ert = E[l]/(R*T_field)
        I_curr[l,:,:] = T_field*A*np.exp(Ert)*(1-((Ert**2+(a1*Ert)+a2)/(Ert**2+(b1*Ert)+b2)))
        del_I[l] = (I_curr[l]-I_prev[l])/dT
        w_ratio[l] = np.maximum(np.exp(-del_I[l]),0)
        fl[l] = (1 - w_ratio[l])*f[l]
    Frac = 1 - np.sum(fl, axis = 0)
    percRo = np.exp(-1.6+3.7*Frac) #vitrinite reflectance
    TOC = TOCo*Frac*calc_parser
    dTOC = (TOC-TOC_prev)/dt
    Rom = (1-porosity)*density*dTOC
    RCO2 = Rom*3.67
    return RCO2, Rom, percRo, I_curr, TOC

def analyticalRo_I(T_field):
    '''
    Initialization of I for the SILLi carbon model
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
    I = np.empty(len(E),a,b)
    for l in range(len(E)):
        Ert = E[l]/(R*T_field)
        I[l,:,:] = T_field*A*np.exp(Ert)*(1-((Ert**2+(a1*Ert)+a2)/(Ert**2+(b1*Ert)+b2)))
    return I
'''
def sillburp(T_field, TOC_prev, density, lithology, porosity, dt, TOCo = np.nan, oil_production_rate = 0, progress_of_reactions = np.nan):
    if np.isnan(TOCo).all():
        TOCo = TOC_prev
    a = len(T_field[:,0])
    b = len(T_field[0,:])
    calc_parser = (lithology=='shale') | (lithology=='sandstone')
    sqrt_2pi = np.sqrt(2*np.pi)
    n_reactions = 4
    reactants = ['LABILE', 'REFRACTORY', 'VITRINITE', 'OIL']
    OIL = reactants.index('OIL')
    As = [1.58e13, 1.83e18, 4e10, 1e13] #pre-exponential constants for the different reactions
    mean_E = [208e3, 279e3, 242e3, 230e3] #mean activation energies for the reactions
    sd_E = [5e3, 13e3, 41e3, 5e3] #Standard deviation of the  normal distributions of the activation energies
    no_reactions = [7, 21, 55, 7] #Number of reactions for each kerogen type
    reaction_energies = np.zeros((n_reactions, max(no_reactions)))
    for i in range(0, n_reactions):
        s_r2 = sd_E[i]*np.sqrt(2)
        N = no_reactions[i]
        fraction = 2/N
        E_0 = 0
        E_1 = 0
        #Number of reaction in the middle of the series
        n_middle = N//2

        #Looping over individual reactions to get the distribution
        for i_approx in range(n_middle, N):
            # Central reaction: set the activation energy to the mean activation energy (EQ 4A or original sillburp code Jones et al. 2019)
            if i_approx == n_middle:
                reaction_energies[i][i_approx] = mean_E[i]
                if N != 1:
                    E_0 = mean_E[i] - s_r2 * erfinv(-1.0 / N)
                continue
            # Reaction on the edge of the distribution
            if i_approx==N-1:
                reaction_energies[i][i_approx] = N * (sd_E[i]/ sqrt_2pi*np.exp(-(mean_E[i] - E_0)**2 / (2.0 * sd_E[i]**2))+mean_E[i] / 2.0 * (1.0 + erf((mean_E[i] - E_0) / s_r2)))
            else:
                right_side = erf((mean_E[i] - E_0) / s_r2) - fraction
                erf_inv = erfinv(right_side)
                E_1 = mean_E[i] - erf_inv * s_r2
                reaction_energies[i][i_approx] = N * (-sd_E[i] / sqrt_2pi *
                    (np.exp(-(mean_E[i] - E_1)**2 / (2.0 * sd_E[i]**2)) - np.exp(-(mean_E[i] - E_0)**2 / (2.0 * sd_E[i]**2))) -
                    mean_E[i] / 2.0 * (erf((mean_E[i] - E_1) / s_r2) - erf((mean_E[i] - E_0) / s_r2)))

            # Probability distribution is symmetrical
            reaction_energies[i][N - i_approx - 1] = 2.0 * mean_E[i] - reaction_energies[i][i_approx]
            
            # Remember activation energy for the next approximate reaction
            E_0 = E_1
    
    if np.isnan(progress_of_reactions).all():
        progress_of_reactions = np.zeros((n_reactions, max(no_reactions), a, b))
        progress_of_reactions_old = np.zeros_like(progress_of_reactions)
        rate_of_reactions = np.zeros_like(progress_of_reactions)

    progress_of_reactions_old = progress_of_reactions
    
    do_labile_reaction = progress_of_reactions[:,reactants.index('LABILE'),:,:]<1
    do_refractory_reaction = progress_of_reactions[:,reactants.index('REFRACTORY'),:,:]<1  # Example value
    do_oil_reaction = progress_of_reactions[:,reactants.index('OIL'),:,:]<1  # Example value
    do_vitrinite_reaction = progress_of_reactions[:,reactants.index('VITRINITE'),:,:]<1
    do_reaction = [do_labile_reaction, do_refractory_reaction, do_vitrinite_reaction, do_oil_reaction]
    mass_frac_labile_to_gas = 0.2 #From sillburp code
    for i in range(0,a):
        for j in range(0,b):
            for i_reaction in range(0, n_reactions):
                if do_reaction[i_reaction].all()==True:
                    continue
                for i_approx in range(0, no_reactions[i_reaction]):
                    initial_product_conc = progress_of_reactions[i_reaction][i_approx][a][b]
                    activation_energy = reaction_energies[i_reaction][i_approx]
                    reaction_rate = As[i_reaction]*np.exp(-activation_energy/8.314/(T_field[a,b]+273.15))
                    if reactants[i_reaction]!= 'OIL':
                        progress_of_reactions[i_reaction][i_approx][i][j] = min((1.0 - (1.0 - initial_product_conc) * np.exp(-reaction_rate * dt)), 1)
                        # Save reaction rate
                        rate_of_reactions[i_reaction][i_approx][i][j] = (1.0 - initial_product_conc) * (1.0 - np.exp(-reaction_rate * dt)) /dt
                    #Oil cracking reaction
                    else:
                        # The oil production rate is spread evenly over the oil cracking reactions, hence divide by n_approx_reactions[OIL]
                        S_over_k = 0.0 if reaction_rate == 0 else oil_production_rate / reaction_rate / no_reactions[OIL]
                        # Take reaction step using equation [5] above
                        progress_of_reactions[i_reaction][i_approx][i][j] = min(1.0 - S_over_k - (1.0 - initial_product_conc - S_over_k) * np.exp(-reaction_rate * dt), 1)

                        # Save average reaction rate in time step
                        rate_of_reactions[i_reaction][i_approx][i][j] = (1.0 - initial_product_conc - S_over_k) * (1.0 - np.exp(-reaction_rate * dt)) / dt

                    # Store average oil production rate over the time step for use in oil cracking calculation
                    if i_reaction==reactants.index('LABILE'):
                        if i_approx == 0:
                            oil_production_rate = 0.0

                        oil_production_rate += -reaction_rate * (1.0 - progress_of_reactions[i_reaction][i_approx][i][j]) / no_reactions[reactants.index('LABILE')]
                        if i_approx == no_reactions[reactants.index('LABILE')] - 1:
                            oil_production_rate *= (1.0 - mass_frac_labile_to_gas)
    time_step_progress = progress_of_reactions - progress_of_reactions_old
    products_progress = np.mean(progress_of_reactions, axis = 0)
    products_progress = np.mean(products_progress, axis = 0)
    time_step_summarized = np.mean(time_step_progress, axis = 0)
    time_step_summarized = np.mean(time_step_progress, axis = 0)
    TOC = TOCo*(1-products_progress)*calc_parser
    dTOC = (TOC_prev - TOC)/dt
    Rom = (1-porosity)*density*dTOC
    RCO2 = Rom*3.67
    return RCO2, Rom, progress_of_reactions, oil_production_rate, TOC
'''
def sillburp(T_field, TOC_prev, density, lithology, porosity, dt, TOCo=np.nan, oil_production_rate=0, progress_of_reactions=np.nan, rate_of_reactions = np.nan):
    if np.isnan(TOCo).all():
        TOCo = TOC_prev
    
    a, b = T_field.shape
    calc_parser = (lithology == 'shale') | (lithology == 'sandstone')
    sqrt_2pi = np.sqrt(2 * np.pi)
    n_reactions = 4
    reactants = ['LABILE', 'REFRACTORY', 'VITRINITE', 'OIL']
    OIL = reactants.index('OIL')
    As = [1.58e13, 1.83e18, 4e10, 1e13]  # pre-exponential constants for the different reactions
    mean_E = [208e3, 279e3, 242e3, 230e3]  # mean activation energies for the reactions
    sd_E = [5e3, 13e3, 41e3, 5e3]  # Standard deviation of the normal distributions of the activation energies
    no_reactions = [7, 21, 55, 7]  # Number of reactions for each kerogen type
    reaction_energies = np.zeros((n_reactions, max(no_reactions)))
    
    for i in range(n_reactions):
        s_r2 = sd_E[i] * np.sqrt(2)
        N = no_reactions[i]
        fraction = 2 / N
        E_0 = 0
        E_1 = 0
        n_middle = N // 2
        
        for i_approx in range(n_middle, N):
            if i_approx == n_middle:
                reaction_energies[i, i_approx] = mean_E[i]
                if N != 1:
                    E_0 = mean_E[i] - s_r2 * erfinv(-1.0 / N)
                continue
            if i_approx == N - 1:
                reaction_energies[i, i_approx] = N * (sd_E[i] / sqrt_2pi * np.exp(-(mean_E[i] - E_0)**2 / (2.0 * sd_E[i]**2)) + mean_E[i] / 2.0 * (1.0 + erf((mean_E[i] - E_0) / s_r2)))
            else:
                right_side = erf((mean_E[i] - E_0) / s_r2) - fraction
                erf_inv = erfinv(right_side)
                E_1 = mean_E[i] - erf_inv * s_r2
                reaction_energies[i, i_approx] = N * (-sd_E[i] / sqrt_2pi *
                    (np.exp(-(mean_E[i] - E_1)**2 / (2.0 * sd_E[i]**2)) - np.exp(-(mean_E[i] - E_0)**2 / (2.0 * sd_E[i]**2))) -
                    mean_E[i] / 2.0 * (erf((mean_E[i] - E_1) / s_r2) - erf((mean_E[i] - E_0) / s_r2)))
            
            reaction_energies[i, N - i_approx - 1] = 2.0 * mean_E[i] - reaction_energies[i, i_approx]
            E_0 = E_1
    
    if np.isnan(progress_of_reactions).all():
        progress_of_reactions = np.zeros((n_reactions, max(no_reactions), a, b))
        progress_of_reactions_old = np.zeros_like(progress_of_reactions)
        rate_of_reactions = np.zeros_like(progress_of_reactions)
    else:
        progress_of_reactions_old = progress_of_reactions.copy()
    
    do_labile_reaction = [True]#progress_of_reactions[:, reactants.index('LABILE'), :, :] < 1
    do_refractory_reaction = [True]#progress_of_reactions[:, reactants.index('REFRACTORY'), :, :] < 1
    do_oil_reaction = [True]#progress_of_reactions[:, reactants.index('OIL'), :, :] < 1
    do_vitrinite_reaction = [True]#progress_of_reactions[:, reactants.index('VITRINITE'), :, :] < 1
    do_reaction = np.array([do_labile_reaction, do_refractory_reaction, do_vitrinite_reaction, do_oil_reaction])
    mass_frac_labile_to_gas = 0.2
    
    for i in range(a):
        for j in range(b):
            for i_reaction in range(n_reactions):
                if ~do_reaction[i_reaction]:#.all():
                    continue
                for i_approx in range(no_reactions[i_reaction]):
                    initial_product_conc = progress_of_reactions[i_reaction, i_approx, i, j]
                    activation_energy = reaction_energies[i_reaction, i_approx]
                    reaction_rate = As[i_reaction] * np.exp(-activation_energy / 8.314 / (T_field[i, j] + 273.15))
                    
                    if reactants[i_reaction] != 'OIL':
                        progress_of_reactions[i_reaction, i_approx, i, j] = min((1.0 - (1.0 - initial_product_conc) * np.exp(-reaction_rate * dt)), 1)
                        rate_of_reactions[i_reaction, i_approx, i, j] = (1.0 - initial_product_conc) * (1.0 - np.exp(-reaction_rate * dt)) / dt
                    else:
                        S_over_k = 0.0 if reaction_rate == 0 else oil_production_rate / reaction_rate / no_reactions[OIL]
                        progress_of_reactions[i_reaction, i_approx, i, j] = min(1.0 - S_over_k - (1.0 - initial_product_conc - S_over_k) * np.exp(-reaction_rate * dt), 1)
                        rate_of_reactions[i_reaction, i_approx, i, j] = (1.0 - initial_product_conc - S_over_k) * (1.0 - np.exp(-reaction_rate * dt)) / dt
                    
                    if i_reaction == reactants.index('LABILE'):
                        if i_approx == 0:
                            oil_production_rate = 0.0
                        oil_production_rate += -reaction_rate * (1.0 - progress_of_reactions[i_reaction, i_approx, i, j]) / no_reactions[reactants.index('LABILE')]
                        if i_approx == no_reactions[reactants.index('LABILE')] - 1:
                            oil_production_rate *= (1.0 - mass_frac_labile_to_gas)
    
    time_step_progress = progress_of_reactions - progress_of_reactions_old
    products_progress = np.mean(progress_of_reactions, axis=0)
    products_progress = np.mean(products_progress, axis=0)
    time_step_summarized = np.mean(time_step_progress, axis=0)
    time_step_summarized = np.mean(time_step_summarized, axis=0)
    TOC = TOCo * (1 - products_progress) * calc_parser
    dTOC = (TOC_prev - TOC) / dt
    Rom = (1 - porosity) * density * dTOC
    RCO2 = Rom * 3.67
    
    return RCO2, Rom, progress_of_reactions, oil_production_rate, TOC, rate_of_reactions
