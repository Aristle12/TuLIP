import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import seaborn as sns

def units(array):
    GtCO2 = (array-array[0]+int(1e-6))/int(1e12)
    GtC = GtCO2/3.67
    return GtCO2, GtC

def truncate(number):
        # Convert the number to a string
        number_str = str(number)
        
        # Find the position of the decimal point
        decimal_index = number_str.find('.')
        
        # If there's no decimal point, return the number as is
        if decimal_index == -1:
            return float(number_str)
        
        # Truncate the string to three decimal places
        truncated_str = number_str[:decimal_index + 4]
        
        # Convert the truncated string back to a float
        return float(truncated_str)

def plot_time_history(flux, iter, save_dir):
    volumes_load_dir = 'sillcubes/'+str(format(flux, '.3e'))

    n_sills_csv = pd.read_csv(volumes_load_dir+'/n_sills.csv')

    volumes = n_sills_csv['volumes']

    volume = volumes[iter]

    load_dir = volumes_load_dir+'/'+str(format(volume, '.3e'))+'/300'

    times_csv = pd.read_csv(load_dir+'/times.csv')

    time = times_csv['time_steps']
    dt = time[1] - time[0]
    time1 = np.arange(0, len(time))*dt
    tot_RCO2 = times_csv['tot_RCO2']
    GtCO2, GtC = units(tot_RCO2)
    plt.plot(time1, np.log10(GtCO2))
    plt.xlabel('Time (years after thermal maturation time)')
    plt.ylabel(r'Total CO$_2$ emissions (log$_{10}$ Gt CO$_{2}$/year)')
    filename = 'emissions_'+str(format(flux, '.3e'))+'flux'+str(format(volume, '.3e'))
    plt.savefig(save_dir+'/'+filename+'.png', format = 'png')

def scale_emissions(flux, z_indexs, dx, dy, load_dir):
    dv = dx*dx*dy
    #load_dir = 'sillcubes/'+str(format(flux, '.3e'))+'/'
    n_sills = pd.read_csv(load_dir+'n_sills.csv')
    volumes = n_sills['volumes']
    vol_CO2s = []
    sill_volumes = []
    scale_factors = []
    scale_ints = []
    tot_slopes = []
    tot_ints = []
    for i in range(len(volumes)):
        index_co2s = []
        index_volumes = []
        volume = volumes[i]
        sillcube = np.load(load_dir+'sillcube'+str(float(volume))+'.npy', allow_pickle = True)
        array = sillcube[sillcube!='']
        array = [str(ele) for ele in array]
        counter = 0
        for l in trange(len(array)):
            counter+= array[l].count('s')
        empl_params = pd.read_csv(load_dir+'emplacement_params'+str(float(volume))+'.csv')
        empl_time = np.array(empl_params['empl_times'])[-1]
        

        #print(format(counter*dv, '.3e'))#, format(np.sum(sillcube[sillcube!=''])*dv, '.3e'))
        #index_volume = np.sum(sillcube!='')*dv
        #print(format(index_volume, '.3e'))
        sill_volume = counter*dv
        prop_CO2 = []
        tot_time = []
        empl_end_time = []
        empl_CO2s = []
        for j in range(len(z_indexs)):
            z_index = z_indexs[j]
            co2_dir = load_dir+str(format(volume,'.3e')+'/'+str(z_index))
            times = pd.read_csv(co2_dir+'/times.csv')
            time = times['time_steps']
            dts = np.array([time[i]-time[i-1] for i in range(1,len(time))])
            empl_index = np.where(time==truncate(empl_time))[0]
            if len(empl_index)==0:
                print(f'empl_time {empl_index} is not in time_steps')
                if np.min(np.abs(time - empl_time))<np.mean(dts):
                    print(f'empl_time has been readjusted for a difference of {np.min(np.abs(time - empl_time)):.2e}')
                    empl_index = np.where(np.abs(time - empl_time)== np.min(np.abs(time-empl_time)))[0]
                    print(f'empl_time is now {time[empl_index]}')
                else:
                    print(f'Difference between empl_time and time is {np.min(np.abs(time - empl_time))}')
                    raise ValueError(f'Emplacement time {empl_time} is not in time steps')
            tot_time.append(len(time)*dts[0])
            empl_end_time.append(float(time[empl_index])-int(3e6))
            
            dts = np.append(dts, [dts[-1]])
            tot_RCO2 = np.array(times['tot_RCO2']-times['tot_RCO2'][0])
            index_CO2_cum = np.cumsum(tot_RCO2)*dts
            index_CO2 = index_CO2_cum[-1]
            empl_CO2 = index_CO2_cum[empl_index]
            prop_CO2.append((empl_CO2/index_CO2)[0])
            empl_CO2s.append(empl_CO2[0])
            print(f'Proportion of CO2 is {empl_CO2/index_CO2}')
            zillcube = sillcube[z_index]
            index_array = zillcube[zillcube!='']
            index_array = np.array([str(ele) for ele in index_array])
            counter = 0
            for l in range(len(index_array)):
                counter+=index_array[l].count('s')
            index_volume = counter*dv
            print(f'Volume for {z_index} is {index_volume} and CO2 is {index_CO2}')
            index_volumes.append(index_volume)
            index_co2s.append(index_CO2)
        cum_volume = np.cumsum(index_volumes)
        cum_CO2 = np.cumsum(index_co2s)
        CO2_timeframe = pd.DataFrame({'z_index':z_indexs, 'tot_time':tot_time, 'tot_CO2':index_co2s, 'prop_CO2': prop_CO2, 'model_end_time': empl_end_time})
        CO2_timeframe.to_csv(load_dir+str(format(volume, '.3e'))+'/CO2_stats.csv')
        empl_CO2s_cum = np.cumsum(empl_CO2s)
        slope, intercept = np.polyfit(cum_volume, empl_CO2s_cum, deg = 1)
        tot_slope, tot_intercept = np.polyfit(cum_volume, cum_CO2, deg = 1)
        sill_volumes.append(sill_volume)
        vol_CO2s.append((sill_volume*slope)+intercept)
        scale_factors.append(slope)
        scale_ints.append(intercept)
        tot_slopes.append(tot_slope)
        tot_ints.append(tot_intercept)
        #plt.plot(cum_volume, cum_CO2, 'ro')
        #plt.plot(cum_volume, empl_CO2s_cum, 'go')
        #y = cum_volume*slope + intercept
        #z = cum_volume*tot_slope + tot_intercept
        #plt.plot(cum_volume, y, 'b-')
        #plt.plot(cum_volume, z, 'g-')
        #plt.title(format(volume,'.3e'))
        #plt.show()
    scales = pd.DataFrame({'tot_volume': sill_volumes, 'tot_CO2': vol_CO2s, 'empl_CO2s':empl_CO2s, 'scale': scale_factors, 'intercept': scale_ints, 'tot_slope': tot_slopes, 'tot_ints': tot_ints})
    scales.to_csv(load_dir+'scales.csv')


def plot_average_emission_rates(fluxs, itera):
    plt.figure(figsize = [18, 9])
    plot_frame = pd.DataFrame()
    flux_list = []
    vol_list = []
    z_list = []
    rate_list = []
    for flux in fluxs:
        load_dir = str(format(flux, '.3e'))+'/'
        z_indexs = [191, 284, 300, 493, 506]
        n_sills = pd.read_csv(load_dir+'n_sills.csv')
        volumes = n_sills['volumes']

        for i in itera:
            volume = volumes[i]
            slice_dir = load_dir + str(format(volume, '.3e'))
            CO2_stats = pd.read_csv(slice_dir+'/CO2_stats.csv')
            prop_CO2 = CO2_stats['prop_CO2']
            time = CO2_stats['tot_time']
            tot_CO2 = np.array(CO2_stats['tot_CO2'])
            empl_CO2 = np.array(tot_CO2*prop_CO2)
            print(f'Emplaced CO2 for {flux/int(1e9)} and {volume/int(1e9)} is {empl_CO2}')
            avg_rate = np.array(empl_CO2/CO2_stats['model_end_time']/int(1e12))
            print(f' Corresponding rate is {avg_rate}')
            flux_list = np.append(flux_list, np.repeat(flux, len(z_indexs)))
            vol_list = np.append(vol_list, np.repeat(volume, len(z_indexs)))
            z_list = np.append(z_list, z_indexs)
            rate_list = np.append(rate_list, avg_rate)

    plot_frame['flux'] = flux_list/int(1e9)
    plot_frame['volume'] = vol_list/(1e9)
    plot_frame['z_index'] = z_list
    plot_frame['rates'] = rate_list
    plot_frame['size'] = rate_list*0.0+200.0
    sns.set_context('poster')

    fig = sns.scatterplot(data = plot_frame, x = 'volume', y = 'rates', hue = 'flux', palette = 'icefire',s=1000)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'Average CO$_2$ emissions (log$_{10}$ Gt CO$_{2}$/year)')
    plt.xlabel(r'Volume $km^3$')

    # Place the legend outside the plot area with a title
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., title='Flux')
    # Remove the original legend


    # Set the title of the colorbar
    #cbar.set_label(r'Rates (GtCO$_2$/yr)', rotation=270, labelpad=15)
    plt.savefig('present_plots/CO2_emissions_per_slice.png', format = 'png', bbox_inches = 'tight')

def plot_heatmap_average_emissions(fluxs, iters, save_dir):
    load_dir = str(format(fluxs[1], '.3e'))+'/'
    n_sills = pd.read_csv(load_dir+'n_sills.csv')
    volumes = np.array(n_sills['volumes'], dtype = int)
    volumes = [volumes[i] for i in iters]
    vol_fraction = np.array([0.15, 0.225, 0.30, 0.375, 0.45])
    conf_array = np.zeros((len(fluxs), len(volumes)))
    sns.set_context('talk')
    for i in range(len(fluxs)):
        flux = fluxs[i]
        load_dir = str(format(flux, '.3e'))+'/'
        for j in range(len(volumes)):
            volume = volumes[j]
            slice_dir = load_dir + str(format(volume, '.3e'))
            print(slice_dir)
            CO2_stats = pd.read_csv(slice_dir+'/CO2_stats.csv')
            prop_CO2 = CO2_stats['prop_CO2']
            time = CO2_stats['tot_time']
            tot_CO2 = np.array(CO2_stats['tot_CO2'])
            empl_CO2 = np.array(tot_CO2*prop_CO2)
            avg_rate = np.array(empl_CO2/CO2_stats['model_end_time']/int(1e12))
            conf_array[i,j] = np.log10(avg_rate[2])
    plt.figure(figsize =[9,6])
    sns.heatmap(conf_array.T, annot=True, cmap = 'RdBu_r', fmt = '.2f', xticklabels = fluxs/int(1e9), yticklabels = vol_fraction)
    plt.xlabel(r'Flux $km^3$')
    plt.ylabel(r'Sill volume fraction')
    plt.title(r'Average CO$_2$ emissions (log$_{10}$ Gt CO$_{2}$/year)', fontsize = 14)
    filename = 'avg_rates'
    plt.savefig(save_dir+'/'+filename+'.png', format = 'png')

def plot_heatmap_warming(fluxs, iters, save_dir):
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
    fluxs = np.array([int(3e9), int(3*10**(8.5)), int(3e8), int(3*10**(7.5)), int(3e7)])
    iters = [0,3,1,4,2]





    #plt.figure(figsize=(10, 6))
    #plt.plot(Cadd['CO2add'],p(Cadd['CO2add']))
    #plt.xlabel('Carbon added (Teragram)')
    #plt.ylabel('Final pCO2 (ppm)')
    #plt.xlim(0, 1.5e7)
    #plt.title('Carbon added vs Final pCO2')
    #plt.show()




    LIP_volume = int(1e5)*int(1e9) #m3
    LIP_timescale = int(2e4) #years

    conf_array = np.zeros((len(fluxs), len(iters)))
    conf_scale = np.zeros((len(fluxs), len(iters)))
    conf_CO2 = np.zeros((len(fluxs), len(iters)))
    for i in range(len(fluxs)):
        for j in range(len(iters)):
                flux = fluxs[i]
                itera = iters[j]
                load_dir = str(format(flux,'.3e'))+'/'

                n_sills = pd.read_csv(load_dir+'n_sills.csv')
                volumes = n_sills['volumes']
                volume = volumes[itera]
                print(format(volume, '.3e'))

                z_index = [191, 284, 300, 493, 506]

                times = pd.read_csv(load_dir+str(format(volume, '.3e'))+'/300/times.csv')
                time = times['time_steps']
                empl_params = pd.read_csv(load_dir+'emplacement_params'+str(float(volume))+'.csv')
                empl_time = np.array(empl_params['empl_times'])[-1]-int(3e6)

                CO2_stats = pd.read_csv(load_dir+str(format(volume, '.3e'))+'/CO2_stats.csv')
                prop_CO2 = CO2_stats['prop_CO2'][2]
                #empl_CO2s = CO2_stats['empl_CO2s'][2]

                dts = np.array([time[i]-time[i-1] for i in range(1, len(time))])
                dts = np.append(dts[0], dts)
                tot_RCO2 = (times['tot_RCO2']-times['tot_RCO2'][0])*dts 
                cum_RCO2 = np.cumsum(tot_RCO2)

                scales_csv = pd.read_csv(load_dir+'scales.csv')
                scale = scales_csv['scale'][itera]
                inter = scales_csv['intercept'][itera]
                volume_scale = scales_csv['tot_volume'][itera]
                print(format(volume_scale, '.3e'))




                LIP_flux = 10*int(1e9)#LIP_volume/empl_time
                #print(f'LIP flux is {LIP_flux:.3e}')


                flux_real = volume_scale/empl_time #m3/yr
                scale_factor_flux = LIP_flux/flux_real
                #print(f'Real flux is {flux_real:.3e}')
                #print(f'Scale factor is {scale_factor_flux*LIP_timescale/empl_time}')



                CO2 = (scale*volume_scale + inter)*scale_factor_flux*LIP_timescale/empl_time
                conf_CO2[i,j] = CO2/(scale_factor_flux*LIP_timescale/empl_time*1e9)
                #tot_CO2 = np.array(cum_RCO2)[-1]/int(1e9)
                #print(format(CO2/int(1e9), '.3e'))
                try:
                    pCO2_calc = p(CO2/int(1e9))
                except:
                    pCO2_calc = 0
                #print(pCO2_calc)
                ECS1 = 3

                deg_change = np.log2(pCO2_calc/pco2i)*ECS1
                conf_array[i,j] = deg_change 
                conf_scale[i,j] = scale_factor_flux*LIP_timescale/empl_time
                print(f'Degree change for {flux:.3e} and {volume:.3e} is {deg_change}')

    plt.figure(figsize =[9,6])
    sns.heatmap(conf_array.T, annot=True, cmap = 'RdBu_r', fmt = '.3f', xticklabels = np.array(fluxs)/int(1e9), yticklabels = np.array([volumes[i] for i in iters])/int(1e9))
    plt.xlabel(r'Flux $km^3$')
    plt.ylabel(r'Volume $km^3$')
    plt.title(r'Warming in ${^\circ}C$')
    #plt.savefig('present_plots/warming_summary.png', format = 'png')
    plt.show()

    sns.heatmap(conf_scale.T, annot=True, cmap = 'RdBu_r', fmt = '.1f', xticklabels = np.array(fluxs)/int(1e9), yticklabels = np.array([volumes[i] for i in iters])/int(1e9))
    plt.show()

    plt.figure(figsize =[9,6])
    sns.heatmap(conf_CO2.T, annot=True, cmap = 'RdBu_r', xticklabels = np.array(fluxs)/int(1e9), yticklabels = np.array([volumes[i] for i in iters])/int(1e9))
    filename = 'warming_emplCO2'
    plt.savefig(save_dir+'/'+filename+'.png', format = 'png')

    for i in range(len(fluxs)):
        for j in range(len(iters)):
                flux = fluxs[i]
                itera = iters[j]
                load_dir = str(format(flux,'.3e'))+'/'

                n_sills = pd.read_csv(load_dir+'n_sills.csv')
                volumes = n_sills['volumes']
                volume = volumes[itera]
                print(format(volume, '.3e'))

                z_index = [191, 284, 300, 493, 506]

                times = pd.read_csv(load_dir+str(format(volume, '.3e'))+'/300/times.csv')
                time = times['time_steps']
                empl_params = pd.read_csv(load_dir+'emplacement_params'+str(float(volume))+'.csv')
                empl_time = np.array(empl_params['empl_times'])[-1]-int(3e6)

                CO2_stats = pd.read_csv(load_dir+str(format(volume, '.3e'))+'/CO2_stats.csv')
                prop_CO2 = CO2_stats['prop_CO2'][2]
                #empl_CO2s = CO2_stats['empl_CO2s'][2]

                dts = np.array([time[i]-time[i-1] for i in range(1, len(time))])
                dts = np.append(dts[0], dts)
                tot_RCO2 = (times['tot_RCO2']-times['tot_RCO2'][0])*dts 
                cum_RCO2 = np.cumsum(tot_RCO2)

                scales_csv = pd.read_csv(load_dir+'scales.csv')
                scale = scales_csv['tot_slope'][itera]
                inter = scales_csv['tot_ints'][itera]
                volume_scale = scales_csv['tot_volume'][itera]
                print(format(volume_scale, '.3e'))




                LIP_flux = 10*int(1e9)#LIP_volume/empl_time
                #print(f'LIP flux is {LIP_flux:.3e}')


                flux_real = volume_scale/empl_time #m3/yr
                scale_factor_flux = LIP_flux/flux_real
                #print(f'Real flux is {flux_real:.3e}')
                #print(f'Scale factor is {scale_factor_flux*LIP_timescale/empl_time}')



                CO2 = (scale*volume_scale + inter)*scale_factor_flux*LIP_timescale/empl_time
                conf_CO2[i,j] = CO2/(scale_factor_flux*LIP_timescale/empl_time*1e9)
                #tot_CO2 = np.array(cum_RCO2)[-1]/int(1e9)
                #print(format(CO2/int(1e9), '.3e'))
                try:
                    pCO2_calc = p(CO2/int(1e9))
                except:
                    pCO2_calc = 0
                #print(pCO2_calc)
                ECS1 = 3

                deg_change = np.log2(pCO2_calc/pco2i)*ECS1
                conf_array[i,j] = deg_change 
                conf_scale[i,j] = scale_factor_flux*LIP_timescale/empl_time
                print(f'Degree change for {flux:.3e} and {volume:.3e} is {deg_change}')

    plt.figure(figsize =[9,6])
    sns.heatmap(conf_array.T, annot=True, cmap = 'RdBu_r', fmt = '.3f', xticklabels = np.array(fluxs)/int(1e9), yticklabels = np.array([volumes[i] for i in iters])/int(1e9))
    plt.xlabel(r'Flux $km^3$')
    plt.ylabel(r'Volume $km^3$')
    plt.title(r'Warming in ${^\circ}C$')
    #plt.savefig('present_plots/warming_summary.png', format = 'png')
    plt.show()

    sns.heatmap(conf_scale.T, annot=True, cmap = 'RdBu_r', fmt = '.1f', xticklabels = np.array(fluxs)/int(1e9), yticklabels = np.array([volumes[i] for i in iters])/int(1e9))
    plt.show()

    plt.figure(figsize =[9,6])
    sns.heatmap(conf_CO2.T, annot=True, cmap = 'RdBu_r', xticklabels = np.array(fluxs)/int(1e9), yticklabels = np.array([volumes[i] for i in iters])/int(1e9))
    filename = 'warming_totCO2'
    plt.savefig(save_dir+'/'+filename+'.png', format = 'png')