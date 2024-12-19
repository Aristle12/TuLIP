import numpy as np
from numba import jit
@jit
def tan_signal(e,a,L,s):
    return 0.5*(np.tanh((e-a+(0.5*L))/s)-np.tanh((e-a-(0.5*L))/s))

@jit
def circle_signal(x, y, a, b, L):
    return int((((x+a)**2)+((y+b)**2))/(L**2)<=1)

@jit
def rect(e,a,L):
    return int(np.abs(e-a)<np.abs((L/2)))

@jit
def circ(x, a, L):
    return np.exp(-0.5*((x-a)**2)/L**2)

@jit
def cos_signal(e, a, L):
    return np.cos((3*(e-a))/(2*L))*int(np.abs(e-a)<(L/2))


@jit
def db_pulse(e, f, ecks, why, cx, cy, L, l1, l2, sd,t):
    delta = circle_signal(ecks, why, cx, cy, L)
    its = 1000
    tee = np.zeros(its)
    lee = np.zeros(its)
    for i in range(0,its):
        #print(i)
        tee[i] = np.sin(i*np.pi*ecks/l1)*np.sin(i*np.pi*(e)/l1)*np.exp(-((np.pi*i)**2)*sd*t/(l1**2))
        lee[i] = np.sin(i*np.pi*why/l2)*np.sin(i*np.pi*(f)/l2)*np.exp(-((np.pi*i)**2)*sd*t/(l2**2))
    return np.sum(tee)*np.sum(lee)*delta

@jit
def pulse_diri(e, ecks,a,L,l1,sd,t):
    s = 10
    pulsee = cos_signal(e, a, L)
    its = 900
    tee = np.zeros(its)
    for i in range(0,its):
        tee[i] = np.sin(i*np.pi*ecks/l1)*np.sin(i*np.pi*(e)/l1)*np.exp(-((np.pi*i)**2)*sd*t/(l1**2))
    return pulsee*np.sum(tee)

def pulse_neum(e, ecks, a, L, l1, sd, t):
    its = 500
    pulsee = rect(e, a, L)
    tee = np.zeros(its)
    for i in range(0,its):
        tee[i] = np.cos(i*np.pi*ecks/l1)*np.cos(i*np.pi*e/l1)*np.exp(-((np.pi*i)**2)*sd*t/(l1**2))
    return pulsee*(0.5+np.sum(tee))
