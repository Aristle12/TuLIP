import numpy as np
from numba import jit
from tqdm import trange

def to_emplace(t_now, t_thresh):
    if (t_now<t_thresh):
        return False
    elif t_now>=t_thresh:
        return True

def build_lith_dict(lithology):
    a,b = lithology.shape
    lith_dict = {0:str(lithology[0,0])}
    n = 1
    for i in range(a):
        for j in range(b):
            if not (lithology[i,j] in lith_dict.values()):
                lith_dict.update({n:lithology[i,j]})
                n = n+1
    return lith_dict

def build_prop_dict(prop, lithology):
    a,b = lithology.shape
    prop_dict = {lithology[0,0]: prop[0,0]}
    for i in range(a):
        for j in range(b):
            if not lithology[i,j] in prop_dict:
                prop_dict.update({lithology[i,j]:prop[i,j]})
    return prop_dict

def single_sill(T_field, x_space, height, width, thick, T_mag):
    """
    Emplacing a simple sill without a dike tail
    """
    T_field[int(height-(thick//2)):int(height+(thick//2)), int(x_space-(width//2)):int(x_space+(width//2))] = T_mag
    return T_field

def circle_sill(T_field, x_space, height, r, T_mag, a, b, dx, dy):
    """
    Emplacing a simple circular sill without the dike tail
    """
    x = np.arange(0,b*dx, dx)
    y = np.arange(0, a*dy, dy)
    for m in range(0, len(T_field[1,:])-1):
            for n in range(0, len(T_field[:,1])-1):
                x_dist = ((x[m]-x[int(x_space)])**2)/(((r)//2)**2)
                y_dist = ((y[n]-y[int(height)])**2)/(((r)//2)**2)
                if (x_dist+y_dist)<=1:
                    T_field[n,m]=T_mag
    #T_field[int(height):-1,int(x_space)] = T_mag
    return T_field

def randn_heights(n_sills, l_sill, h_sill, sd, dy):
    """
    Get random emplacement heights over a normal distribution within the specified range. Output is in nodes.
    n_sills - number of sills int
    l_sills - Depth of lowest sill emplacement range (m) int
    h_sills - Depth of shallowest depth emplacement range (m) int
    sd = Standard Deviation of the heights distribution
    dy - Grid spacing in the y direction (m) int
    """
    if h_sill<l_sill:
        pass
    else:
        print('l_sill:', l_sill)
        print('h_sill:', h_sill)
        raise ValueError('l_sill should be greater than h_sill')
    bean = np.mean([l_sill/dy, h_sill/dy])
    heights = np.round((sd/dy)*np.random.randn(n_sills) + bean)
    while ((heights>l_sill/dy).any() or (heights<h_sill/dy).any()):
        if (heights>l_sill/dy).any():
            heights[heights>l_sill/dy] = randn_heights(np.sum(heights>l_sill/dy), l_sill, h_sill, sd, dy)
        if (heights<h_sill/dy).any():
            heights[heights<h_sill/dy] = randn_heights(np.sum(heights<(h_sill/dy)), l_sill, h_sill, sd, dy)
    return heights

def x_spacings(n_sills, x_min, x_max, sd, dx):
    """
    Nodes for x-coordinate space chosen as a random normal distribution
    n_sills - number of sills int
    x_min - The lower range (left side) (m) int
    x_max - The upper range (right side) (m) int
    sd - Standard deviation of the distribution. For the entire distribution to fit within the range, a maximum of 10% of the distribution is recommended. 
    """
    space = np.round((sd/dx)*np.random.randn(n_sills)+ np.mean([x_min/dx, x_max/dx]))
    while ((space>x_max/dx).any() or (space<x_min/dx).any()):
            if (space>x_max/dx).any():
                space[space>x_max/dx] = x_spacings(np.sum(space>x_max/dx), x_min, x_max, sd, dx)
            if (space<x_min/dx).any():
                space[space<x_min/dx] = x_spacings(np.sum((space<x_min/dx)), x_min, x_max, sd, dx)
    return space

def uniform_heights(n_sills, l_sill, h_sill, dy):
    """
    Get heights spacing randomly picked from a uniform distribution
    """
    heights = np.round(np.random.uniform(l_sill, h_sill, n_sills)/dy)
    return heights


def uniform_x(n_sills, x_min, x_max, dx):
    """
    Nodes for x-coordinate space chosen as a random normal distribution
    """
    space = np.round(np.random.uniform(x_min, x_max, n_sills)/dx)
    return space

def empirical_CDF(n_sills, xarray, cdf):
    """Function to give random numbers from a specific empirical distribution
    n_sills - number of sills needed int
    xarray - array of domain for empirical CDF
    cdf - array of CDF for the x array"""
    why = np.zeros(n_sills)
    for k in range(0,n_sills):
        a = np.random.uniform(0,1)
        gee = np.argmax(cdf>=a)-1
        why[k] = xarray[gee+1]
    return why
    

def get_scaled_dims(min_min, min_max, mar, sar, heights, n_sills, dx, dy):
    """
    Linearly scaled with height (inversely) plus noise for both aspect ratio and shape
    Returns the width and height respectively in the number of nodes
    min_min = Minimum value for the thickness (m)
    min_max = Maximum value for the thickness (m)
    mar = Mean aspect ratio (Width/Thickness)
    sar = Standard deviation for the distribution of the aspect ratios
    n_sills = Number of sills
    dx = Node spacing in the x-direction
    dy = Node spacing in the y-direction
    Returns dims in length units
    """
    fact_min = ((min_max-min_min)/min_max)*((np.max(heights)-np.min(heights))/np.max(heights))
    major = np.zeros(n_sills)
    minor = np.zeros(n_sills)
    aspect_ratio = sar*np.random.randn(n_sills) + mar
    for i in range(0, n_sills):
        minor[i] = min_min + fact_min*heights[i] + np.round(2*np.random.randn())
        major[i] = minor[i]*aspect_ratio[i]
    return major, minor

def randn_dims(min_min, min_max, sd_min, mar, sar, n_sills):
    """
    Random normal distribution of dims for aspect ratio and shape
    min_min = Minimum value for the thickness (m)
    min_max = Maximum value for the thickness (m)
    mar = Mean aspect ratio (Width/Thickness)
    sar = Standard deviation for the distribution of the aspect ratios
    n_sills = Number of sills
    Returns dims in length units
    """
    aspect_ratio = sar*np.random.randn(n_sills) + mar
    minor = np.round(sd_min*np.random.randn(n_sills)+np.mean([min_min, min_max]))
    while ((minor>min_max).any() or (minor<min_min).any()):
            if (minor>min_max).any():
                minor[(minor>min_max)] = np.round(sd_min*np.random.randn(np.sum(minor>min_max))+np.mean([min_min, min_max]))
            if (minor<min_min).any():
                minor[minor<min_min] = np.round(sd_min*np.random.randn(np.sum(minor<min_min))+np.mean([min_min, min_max]))
    major = np.multiply(minor, aspect_ratio)
    return major, minor

def uniform_dims(min_min, min_max, min_ar, max_ar, n_sills):
    """
    Random uniform distribution of dims for aspect ratio and shape
    min_min = Minimum value for the thickness (m)
    min_max = Maximum value for the thickness (m)
    min_ar = Minimum aspect ratio (Width/Thickness)
    max_ar =  Maximum aspect ratio
    n_sills = Number of sills
    Returns dims in length units
    """
    aspect_ratio = np.random.uniform(min_ar, max_ar, n_sills)
    minor = np.round(np.random.randn(min_min, min_max, n_sills))
    major = np.multiply(minor, aspect_ratio)
    return major, minor

def value_pusher(array, new_value, push_index, push_value):
    x,y = push_index
    if push_value<=0:
        raise ValueError("push_value must be greater than 0")
    # Ensure the push operation does not exceed the array bounds
    if x + push_value >= len(array[0,:]):
        push_value = len(array[0,:])-x-1
    # Shift the values down
    array[x+push_value:x-1:-1, y] = array[x:x-push_value-1:-1, y]
    array[x:x+push_value,y] = new_value
    return array

def prop_updater(lithology, lith_dict, prop_dict):
    '''
    This function updates the associated rock properties once everything has shiftes. This is done to avoid thermopgenic carbon generation from popints that are now pure magma'''
    prop = np.zeros_like(lithology)
    for rock in lith_dict.values():
        prop[lithology==rock] = prop_dict[rock]
    return prop

def value_pusher2D(array, new_value, row_index, push_amount):
    a,b = array.shape
    if len(row_index) != b or len(push_amount) != b:
        raise ValueError("row_index and push_values must have the same length as the number of columns")
    for j in range(b):
        if row_index[j] + push_amount[j] >= a:
                raise ValueError(f"Push value for column {j} exceeds array bounds")
        print(row_index, push_amount)
        array[row_index[j]+push_amount[j]:,j] = array[row_index[j]:a-push_amount[j], j]
        array[row_index[j]:row_index[j]+push_amount[j],j] = new_value
    return array




#@jit
def mult_sill(T_field, majr, minr, height, x_space, dx, dy, dike_net, cm_array = [], cmb = [], rock = np.array([]), T_mag = 1000, shape = 'rect', dike_empl = True, cmb_exists = False):
    a,b = T_field.shape
    if dike_empl:
        T_field[int(height):-1,int(x_space)] = T_mag
        if rock.size>0:
            rock.loc[int(height):-1,int(x_space)] = 'basalt'

    if cmb_exists:
        new_dike = np.zeros_like(T_field)
        if shape == 'rect':
            new_dike[int(height-(minr//2)):int(height+(minr//2)), int(x_space-(majr//2)):int(x_space+(majr//2))] = 1
        elif shape=='elli':
            x = np.arange(0,b*dx, dx)
            y = np.arange(0, a*dy, dy)
            majr = majr*dx
            minr = minr*dy
            for m in range(0, a):
                for n in range(0, b):
                    if (x_dist+y_dist)<=1:
                        T_field[m,n]=T_mag
                        new_dike[m,n] = 1
        cm_mov = np.sum(new_dike, axis = 0)
        cmb = cmb + cm_mov
        for l in range(0, b):
            if cm_mov[l]!=0:
                for m in range(0,a):
                    if new_dike[m,l]==1:
                        T_field = value_pusher(T_field, T_mag,[m,l], cm_mov[l])
                        rock = value_pusher(rock,'basalt',[m,l],cm_mov[l])
                        continue
        for i in range(0, a):
            for j in range(0, b):
                if i>cmb[i]:
                    cm_array.loc[i,j] = 'mantle'
        return T_field, dike_net, rock, cm_array
    else:
        new_dike = np.zeros_like(T_field)
        if shape == 'rect':
            T_field[int(height-(minr//2)):int(height+(minr//2)), int(x_space-(majr//2)):int(x_space+(majr//2))] = T_mag
            new_dike[int(height-(minr//2)):int(height+(minr//2)), int(x_space-(majr//2)):int(x_space+(majr//2))] = 1
        elif shape == 'elli':
            x = np.arange(0,b*dx, dx)
            y = np.arange(0, a*dy, dy)
            majr = majr*dx
            minr = minr*dy
            for m in range(0, a):
                for n in range(0, b):
                    x_dist = ((x[m]-x[int(x_space)])**2)/(((majr)//2)**2)
                    y_dist = ((y[n]-y[int(height)])**2)/(((minr)//2)**2)
                    if (x_dist+y_dist)<=1:
                        T_field[m,n]=T_mag
                        new_dike[m,n] = 1
                        if rock.size>0:
                            rock.loc[n,m] = 'basalt'
        dike_net = dike_net + new_dike
        return T_field, dike_net, rock

def get_H(T_field, rho, CU, CTh, CK, T_sol, dike_net, a, b):
    """
    Function to calculate external heat sources generated through latent heat of crystallization and radiactive heat generation
    T_field = Temp field, int
    rho = Density kg/m3
    CU, CTh = U, Th concentrations in ppm, array
    CK = K conc in wt %, array
    T_sol = Solidus temperature
    """
    J = 0.25 #J/kg latent heat of crystallization
    Cp = 1450 #J/kgK specific heat capacity
    H = np.zeros_like(T_field)
    for i in range(0,a):
        for j in range(0, b):
            if T_field[i,j]>T_sol and dike_net[i,j]!=0:
                H[i,j] = J/(rho*Cp)
    A = rho*1e-5*(9.52*CU + 2.56*CTh + 3.48*CK) #Formula from Rybach and Cermack 1982 - Radioactive heat generation in rocks
    H = H+A
    return H

def get_diffusivity(T_field, lithology):
    K = 1e-6
    return K

'''
@jit(forceobj=True)
def sill_3Dcube(x, y, z, dx, dy, n_sills, x_coords, y_coords, z_coords, maj_dims, min_dims, empl_times, shape = 'elli', dike_tail = False):
    Function to gnenerate sills in 3D space to employ fluxes as a control for sill emplacement. Choose any 1 slice for a 2D coolong model, or multiple slices for multiple cooling models
    x, y, z = width, height and third dimension extension of the crustal slice (m)
    n_sills = Number of sills to be emplaced
    dx, dy = Node spacing
    x_coords = x coordinates for the center of the sills
    y_coords = y coordinates for the center of the sills
    z_coords = z coordinates for the center of the sills
    maj_dims, minor dims = dimensions of the 2D sills. Implicit assumption of circularity in the z-direction is present in the code (m)

    a = int(y//dy)
    b = int(x//dx)
    c = int(z//dx)
    sillcube = np.empty([c,a,b], dtype=object)
    sillcube[:,:,:] = ''
    x_len = np.arange(0,x, dx)
    y_len = np.arange(0,y, dy)
    z_len = np.arange(0, z, dx)
    if shape=='elli':
        for l in trange(0, n_sills):
            for i in range(0,b):
                for j in range(0, a):
                    for q in range(0,c):
                        x_dist = ((x_len[i] - x_coords[l])**2)/((0.5*maj_dims[l])**2)
                        y_dist = ((y_len[j] - y_coords[l])**2)/((0.5*min_dims[l])**2)
                        z_dist = ((z_len[q] - z_coords[l])**2)/((0.5*maj_dims[l])**2)
                        if (x_dist+y_dist+z_dist)<=1:
                            if sillcube[q,j,i]!='':
                                print('Sill intersection detected')
                            sillcube[q,j,i] = sillcube[q,j,i]+'_'+str(l)+'s'+str(empl_times[l])
            if dike_tail:
                sillcube[int(z_coords[l]), int(y_coords[l]):-1, int(x_coords[l])] += '_'+str(l)+'s'+str(empl_times[l])
    elif shape=='rect':
        for l in range(0, n_sills):
            sillcube[int(z_coords[l]-(maj_dims//2)):int(z_coords[l]+(maj_dims//2)),int(y_coords[l]-(maj_dims//2)):int(x_coords[l]+(maj_dims//2)),int(y_coords[l]-(min_dims//2)):int(y_coords[l]+(min_dims//2))]+= '_'+str(l)+'s'+str(empl_times[l])
            if dike_tail:
                sillcube[int(z_coords[l]), int(y_coords[l]):-1, int(x_coords[l])] += '_'+str(l)+'s'+str(empl_times[l])
    return sillcube
'''

def sill_3Dcube(x, y, z, dx, dy, n_sills, x_coords, y_coords, z_coords, maj_dims, min_dims, empl_times, shape='elli', dike_tail=False):
    '''
    Function to generate sills in 3D space to employ fluxes as a control for sill emplacement.
    Choose any 1 slice for a 2D cooling model, or multiple slices for multiple cooling models
    x, y, z = width, height and third dimension extension of the crustal slice (m)
    n_sills = Number of sills to be emplaced
    dx, dy = Node spacing
    x_coords = x coordinates for the center of the sills
    y_coords = y coordinates for the center of the sills
    z_coords = z coordinates for the center of the sills
    maj_dims, minor dims = dimensions of the 2D sills. Implicit assumption of circularity in the z-direction is present in the code (m)
    '''
    a = int(y // dy)
    b = int(x // dx)
    c = int(z // dx)
    sillcube = np.empty([c, a, b], dtype=object)
    sillcube[:, :, :] = ''
    z_len, y_len, x_len = np.mgrid[:c, :a, :b]



    maj_dims = maj_dims/dx
    min_dims = min_dims/dy

    if shape == 'elli':
        for l in trange(n_sills):
            mask = ((((z_len-z_coords[l])**2)/maj_dims[l]**2)+(((y_len-y_coords[l])**2)/min_dims[l]**2)
            +(((x_len-x_coords[l])**2)/maj_dims[l]**2))<=1
            sillcube[mask] += '_' + str(l) + 's' + str(empl_times[l])
            if dike_tail:
                sillcube[int(z_coords[l]), int(y_coords[l]):-1, int(x_coords[l])] += '_' + str(l) + 's' + str(empl_times[l])

    elif shape == 'rect':
        for l in trange(n_sills):
            z_start = int(z_coords[l] - (maj_dims[l] // 2))
            z_end = int(z_coords[l] + (maj_dims[l] // 2))
            y_start = int(y_coords[l] - (maj_dims[l] // 2))
            y_end = int(y_coords[l] + (maj_dims[l] // 2))
            x_start = int(x_coords[l] - (min_dims[l] // 2))
            x_end = int(x_coords[l] + (min_dims[l] // 2))

            sillcube[z_start:z_end, y_start:y_end, x_start:x_end] += '_' + str(l) + 's' + str(empl_times[l])
            if dike_tail:
                sillcube[int(z_coords[l]), int(y_coords[l]):-1, int(x_coords[l])] += '_' + str(l) + 's' + str(empl_times[l])

    return sillcube
def emplace_3Dsill(T_field, sillcube, n_rep, T_mag, z_index, curr_empl_time):
    '''
    Function to empalce a sill into the 2D slice T_field
    T_field = 2D temperature array
    sillcube = 3D sill array
    n_rep = the number of the sill being emplaced
    z_index = The 2D slice from the 3D sill array being considered
    '''
    string_finder = str(n_rep)+'s'+str(curr_empl_time)
    if len(sillcube.shape)!=3:
        raise IndexError('sillcube array must be three-dimensional')
    if T_field.size==0:
        raise IndexError("T_feild cannot be empty")
    T_field[string_finder in sillcube[z_index]] = T_mag
    return T_field

def lithology_3Dsill(rock, sillcube, nrep, z_index, rock_type = 'basalt'):
    '''
    Function to keep track of lithology changes in the 2D array
    rock = 2D lithology array
    sillcube = 3D sill array
    nrep = number of the sill being emplaced
    z_index = The 2D slice from the 3D sill array being considered
    rock_type = the type of rock formed by the magma
    '''
    sillcube[sillcube==nrep] = rock_type
    rock_2d = sillcube[z_index,:,:]
    rock[rock_2d==rock_type]==rock_type
    return rock

def cmb_3Dsill(cm_array, cmb, sillcube, nrep, z_index):
    new_sill = sillcube==nrep
    new_sill = new_sill[z_index,:,:]
    cm_mov = np.sum(new_sill, axis = 0)
    cmb = cmb+cm_mov
    for i in range(0, cm_array[:,0]):
            for j in range(0, cm_array[0,:]):
                if i>cmb[i]:
                    cm_array.loc[i,j] = 'mantle'
    return cm_array

'''Broken function
def array_shifter(array_old,array_new, sillcube_z, n_rep, curr_empl_time):
    string_finder = str(n_rep)+'s'+str(curr_empl_time)
    y_shifts = np.sum(np.array([string_finder in sillcube_z]).astype(int), axis = 0)
    for i in range(0, len(array_new[:,0])):
        if y_shifts[i]!=0:
            for j in range(0, len(array_new[0,:])):
                if array_old[i,j]!= array_new[i,j]:
                    array_new[i,j+y_shifts::-1] = array_old[i,::-y_shifts]
                    continue
    return array_new
    '''

def sill3D_pushy_emplacement(props_array, props_dict, sillcube, n_rep, mag_props_dict, z_index, curr_empl_time):
    string_finder = str(n_rep)+'s'#+str(curr_empl_time)
    print(string_finder)
    T_field_index = props_dict['Temperature']
    T_field = props_array[T_field_index]
    a,b = T_field.shape
    if len(sillcube.shape)!=3:
        raise IndexError('sillcube array must be three-dimensional')
    if T_field.size==0:
        raise ValueError("Temperature values in props_array cannot be empty")
    new_dike = np.zeros_like(T_field)
    new_dike[string_finder in sillcube[z_index]] = 1
    print(f'Nodes that have magma: {np.sum(new_dike), np.sum(string_finder in sillcube[z_index])}')
    print(f'The actual nodes{string_finder in sillcube[z_index]}')
    columns_pushed = np.sum(new_dike, axis =0)
    row_push_start = np.empty(b)
    for n in range(b):
        for m in range(a):
            if new_dike[m,n]==1:
                print('It is')
                if np.isnan(row_push_start[n]):
                    row_push_start[n] = m
                else:
                    raise LookupError('Redudancy - Should not come up')
                continue
    print(len(row_push_start), len(columns_pushed))
    if np.sum(string_finder in sillcube[z_index])==0:
        print(f'Sill {n_rep} was NOT emplaced in this slice')
    else:
        print(f'Sill {n_rep} was emplaced in this slice')
        reverse_prop_dict = {value: key for key, value in props_dict.items()}
        for i in [listy for listy in props_dict.values()]:
            props_array[i] = value_pusher2D(props_array[i],mag_props_dict[reverse_prop_dict[i]],row_push_start, columns_pushed)
    return props_array, row_push_start, columns_pushed