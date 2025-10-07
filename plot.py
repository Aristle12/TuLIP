import bloom

fluxs = [int(3.000e9)]#, int(3.002e9), int(3.003e9), int(3.004e9), int(3.005e9)]#[int(3e9), int(3*10**8.5), int(3e8), int(3*10**7.5), int(3e7)]
z_indexs = [191, 284, 300, 493, 506] #[160, 191,  278, 284, 300, 303, 493, 506, 515]
lat_ranges = [0.45, 0.4, 0.35, 0.25, 0.2]

for flux in fluxs:
    for lat in lat_ranges:
        load_dir = str('sillcubes2/'+format(flux, '.3e'))+'/'+str(lat)+'/'

        bloom.scale_emissions(z_indexs, 50, 50, load_dir)

