from TuLIP import sill_controls as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
x = 300000 #m - Horizontal extent of the crust
y = 12000 #m - Vertical thickness of the crust
dx = dz = 50 #m node spacing in x-direction
dy = 50 #m node spacing in y-direction
sc_dist = sc(x,y, dx, dy, T_liquidus=1175,T_solidus=800,
                   include_external_heat=True,
                   calculate_closest_sill=True)
root_dir = 'sillcubes/'
fluxs = [int(3e9), int(3e8), int(3e7), int(3*(10**7.5)), int(3*(10**8.5))]

for i in range(0, len(fluxs)):
    n_sills_dat = pd.read_csv(root_dir+str(format(fluxs[i],'.3e')+'/n_sills.csv'))
    n_sills = n_sills_dat['n_sills']
    volumes = n_sills_dat['volumes']
    for j in range(0, len(volumes)):
        print(f'Currently showing flux {fluxs[i]:3e} and volume: {volumes[j]:3e}')
        load_dir = root_dir+str(format(fluxs[i],'.3e'))+'/'+str(format(volumes[j],'.3e'))+'/300/'
        conf_mat = np.zeros((n_sills[j], n_sills[j]))
        sill_distance = pd.read_csv(load_dir+'sill_distances.csv')
        choose = np.where(sill_distance['distance']<1e31)[0]
        sill_distance['criteria'] = 3*np.array(sill_distance['width of closest sill'])
        for k in range(n_sills[j]):
            if k in sill_distance['sills'].values:
                row = sill_distance.loc[sill_distance['sills'] == k].iloc[0]
                closest_sill = int(row['closest_sill'])
                dist = row['distance']
                criteria = row['criteria']
                if closest_sill != -1 and dist < criteria:
                        conf_mat[k, closest_sill] = 1#row['distance']
        conf_mat = np.array(conf_mat, dtype = bool)
        visualizer = nx.from_numpy_array(conf_mat)
        dirs = 'sillcubes/present_plots/network_plots/'+str(format(fluxs[i], '.3e')+'/')
        os.makedirs(dirs, exist_ok=True)
        dirx = dirs+str(format(volumes[j], '.3e'))
        sc_dist.plot_Full_Graph(visualizer, graph_layout='spring', save_me=True, name_file_dir=dirx)
    '''
        sills = np.array(sill_distance['sills'][choose])
        closest_sill = np.array(sill_distance['closest_sill'][choose], dtype=int)
        distance = np.array(sill_distance['distance'][choose])
        for m in range(len(sills)):
            if i==0 and j== 6:
                print(closest_sill[m])
                print(f'sill is {sills[m]}')
            conf_mat[sills[m], closest_sill[m]] = 1/(distance[m]+1e-6)
        sc_dist.plot_Full_Graph(conf_mat)
    ''' 
