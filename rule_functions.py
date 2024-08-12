import numpy as np
from numba import jit

def to_emplace(t_now, t_thresh):
    if (t_now<t_thresh):
        return False
    elif t_now>=t_thresh:
        return True



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
        print('l_sill should be greater than h_sill')
        print('l_sill:', l_sill)
        print('h_sill:', h_sill)
        exit()
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
                space[space>x_max/dx] = randn_heights(np.sum(space>x_max/dx), x_min, x_max, sd, dx)
            if (space<x_min/dx).any():
                space[space<x_min/dx] = randn_heights(np.sum((space<x_min/dx)), x_min, x_max, sd, dx)
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

#@jit
def mult_sill(T_field, majr, minr, height, x_space, a, b, dx, dy, dike_net, cm_array = [], cmb = [], rock = np.array([]), T_mag = 1000, shape = 'rect', dike_empl = True, cmb_exists = False):
    if shape == 'rect':
        T_field[int(height-(minr//2)):int(height+(minr//2)), int(x_space-(majr//2)):int(x_space+(majr//2))] = T_mag
    elif shape == 'elli':
        x = np.arange(0,b*dx, dx)
        y = np.arange(0, a*dy, dy)
        new_dike = np.zeros_like(T_field)
        majr = majr*dx
        minr = minr*dy
        for m in range(0, len(T_field[1,:])-1):
            for n in range(0, len(T_field[:,1])-1):
                x_dist = ((x[m]-x[int(x_space)])**2)/(((majr)//2)**2)
                y_dist = ((y[n]-y[int(height)])**2)/(((minr)//2)**2)
                if (x_dist+y_dist)<=1:
                    T_field[n,m]=T_mag
                    new_dike[n,m] = 1
                    if rock.size>0:
                        rock.loc[n,m] = 'basalt'
    dike_net = dike_net + new_dike
    if dike_empl:
        T_field[int(height):-1,int(x_space)] = T_mag
        if rock.size>0:
            rock.loc[int(height):-1,int(x_space)] = 'basalt'

    if cmb_exists:
        cm_mov = np.sum(new_dike, axis = 0)
        cmb = cmb + cm_mov
        for i in range(0, a):
            for j in range(0, b):
                if i>cmb[i]:
                    cm_array.loc[i,j] = 'mantle'
        return T_field, dike_net, rock, cm_array
    else:
        return T_field, dike_net, rock

def get_H(T_field, rho, CU, CTh, CK):
    """
    Function to calculate external heat sources generated through latent heat of crystallization and radiactive heat generation
    T_field = Temp field, int
    rho = Density kg/m3
    CU, CTh = U, Th concentrations in ppm
    CK = K conc in wt %"""
    J = 0.25 #J/kg latent heat of crystallization
    Cp = 1450 #J/kgK specific heat capacity
    H = np.zeros_like(T_field)
    H = J/(rho*Cp)
    A = rho*1e-5*(9.52*CU + 2.56*CTh + 3.48*CK) #Formula from Rybach and Cermack 1982 - Radioactive heat generation in rocks
    H = H+A
    return H

def get_diffusivity(T_field, lithology):
    K = 1e-6
    return K


def sill_3Dcube(x, y, z, dx, dy, n_sills, x_coords, y_coords, z_coords, maj_dims, min_dims, dike_tail = False):
    '''
    Function to gnenerate sills in 3D space to employ fluxes as a control for sill emplacement. Choose any 1 slice for a 2D coolong model, or multiple slices for multiple cooling models
    x, y, z = width, height and third dimension extension of the crustal slice (m)
    n_sills = Number of sills to be emplaced
    dx, dy = Node spacing
    x_coords = x coordinates for the center of the sills
    y_coords = y coordinates for the center of the sills
    z_coords = z coordinates for the center of the sills
    maj_dims, minor dims = dimensions of the 2D sills. Implicit assumption of circularity in the z-direction is present in the code (m)
    '''
    sillcube = np.zeros([z,y, x])
    x_len = np.arange(0,x, dx)
    y_len = np.arange(0,y, dy)
    z_len = np.arange(0, z, dx)
    for l in range(0, n_sills):
        for i in range(0,x):
            for j in range(0, y):
                for q in range(0,z):
                    x_dist = ((x_len[i] - x_coords[l])**2)/((0.5*maj_dims[l])**2)
                    y_dist = ((y_len[j] - y_coords[l])**2)/((0.5*min_dims[l])**2)
                    z_dist = ((z_len[q] - z_coords[l])**2)/((0.5*maj_dims[l])**2)
                    if (x_dist+y_dist+z_dist)<=1:
                        if sillcube[q,j,i]!=0:
                            print('Sill intersection detected')
                        sillcube[q,j,i] = l
                    if dike_tail:
                        sillcube[int(z_coords), int(y_coords[l]):-1, int(x_coords)] = l
    return sillcube

def emplace_3Dsill(T_field, sillcube, n_rep, T_mag, z_index):
    '''
    Function to empalce a sill into the 2D slice
    T_field = 2D temperature array
    sillcube = 3D sill array
    n_rep = the number of the sill being emplaced
    z_index = The 2D slice from the 3D sill array being considered
    '''
    sillcube[sillcube==n_rep] = T_mag
    sillcube[sillcube!=T_mag] = 0
    T_field = T_field + sillcube[z_index,:,:]

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