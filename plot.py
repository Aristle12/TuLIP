import bloom

fluxs = [int(3e9)]#, int(3*10**8.5), int(3e8), int(3*10**7.5), int(3e7)]
z_indexs = [191, 284, 300, 493, 506]

for flux in fluxs:
    load_dir = str('sillcubes/'+format(flux, '.3e'))+'/'

    bloom.scale_emissions(z_indexs, 50, 50, load_dir)

