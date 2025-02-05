import anaylitical_functions as anaf
from TuLIP import rules, cool
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from scipy.integrate import quad
import os
import glob

def analytical_T(T_ana, T_init, T_surf, a, b, L, h, i, dt, dx, dy, aye, bee,sd):
    iy=i+1
    L=(L+1e-9)
    h = (h+1e-9)
    l1 = bee*dx
    l2 = aye*dy
    cx = l1/2
    cy = l2/2
    for m in trange(0,aye):
        for n in range(0,bee):
            x = ((n)*dx)
            y = ((m)*dy)
            #T_ana[m,n] = ((0.25*(erf((L+x-(a*dx))/np.sqrt(4*sd*iy*dt))+erf((L-x+(a*dx))/np.sqrt(4*sd*iy*dt)))*(erf((h+y-(b*dy))/np.sqrt(4*sd*iy*dt))+erf((h-y+(b*dy))/np.sqrt(4*sd*iy*dt))))*(T_init-T_surf))+T_surf
            x_int = quad(anaf.pulse_diri, 0, l1, args=(x,cx,L,l1,sd,iy*dt), limit = 100)
            y_int = quad(anaf.pulse_diri, 0, l2, args=(y,cy,L,l1,sd,iy*dt), limit = 100)
            T_ana[m,n] = (4*x_int[0]*y_int[0]/(l1*l2))*(T_init-T_surf) + T_surf
    return T_ana


T_init = 1000 #Initial magmatic temeprature
T_surf = 0 #Surface/ background temperature in the cube

np.seterr('ignore')
x = 9000 #m
y = 9000 #m
dx = 50 #m
dy = 50 #m

a = int(y/dy)
b = int(x/dx)
print(a,b)

k = 31.536*np.ones((a,b)) #m2/year
#k[:,200:300] = 31.536*2

dt = (min(dx, dy)**2)/(10*np.mean(k)) #years
time = 40*dt #years

print('Time step:', dt, ' years')

T_field = np.zeros((a,b))
T_ana = np.zeros((a,b))

x_space = b//2
height = a//2
L = 1000 #m
h = 1000 #m

os.makedirs('analtyical', exist_ok=True)
files = glob.glob('analytical/*')
for f in files:
    os.remove(f)
print('Image Output Directory Cleared')

T_ana = analytical_T(T_ana, T_init, T_surf, x_space, height, L, h, -1, dt, dx, dy, a, b, k[0,0]) #rool.single_sill(T_field, x_space, height, L//dx, h//dy, T_init)
T_field = T_ana.copy()

plt.imshow(T_ana, cmap = 'RdBu_r')
plt.colorbar()
plt.show()

