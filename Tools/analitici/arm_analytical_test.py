
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from matplotlib.animation import FuncAnimation


sys.path.insert(0, os.getcwd())

import DFT 
import time_evolution as __ 

try:
	PATH = str(sys.argv[1])
	sys.path.insert(0, PATH)

	from Par import *

except:
	print('There are not arguments: picking Default Par')
	sys.path.insert(0, '/home/falzo/Scrivania/Sol_eqS')

	from Parameters import  *
	PATH = os.getcwd()

print('ANALYTICAL')
print(PATH)

flag_s = 1


def einge_evolution(Ψ_0, n, h, ω, t):
    
    E_n = (0.5 + n) * h * ω

    return np.exp(-1j * E_n * t / h) * Ψ_0

def sup_evolution(x, x0, m, ω, h, c_n, vector, t):
    
    Y = np.zeros((len(c_n), len(x)))
    y = np.zeros(len(x))


    for i in range(len(vector)):
        Y[i] = DFT.arm_ES(x, x0, vector[i], m, ω, h)

        y = y + c_n[i] * einge_evolution(Y[i], vector[i], h, ω, t)
    
    return y

dx = x[1] - x[0]

#FRAME = 1000
o = np.load(PATH + '/0.npz')

x_0	 = o['x'] 
V	 = o['V'] 
Ψ	 = o['Ψ']

Ψ_0 = Ψ

if flag_s == 1:
    c = np.load(PATH + '/custom.npz')

    x_c = c['x']
    a_0	= c['Ψ'] 
    c_n = c['c_n']
    vector = c['vector']

    for i in range(len(x_c)):
        if x_c[i] != x_0[i]:
            print('Error: x_c != x_0')
            exit()

    a = a_0


else:
    a = Ψ
    a = einge_evolution(Ψ_0 , n, h, ω, 0)


t = np.empty(0)


for i in range(FRAME):

    o = np.load(PATH + '/' + str(i)+'.npz')
    Ψ = o['Ψ']

    
    if flag_s == 1: a = sup_evolution(x, x0, m, ω, h, c_n, vector, i*dt) 
    else: a = einge_evolution(Ψ_0 , n, h, ω, i*dt)
	
    t = np.append(t, max(abs(Ψ - a)))

plt.figure()

plt.plot(np.linspace(0, FRAME, FRAME), t)

plt.show()

print(max(abs(Ψ - a)), np.argmax(abs(Ψ - a)))

np.savez(PATH + '/results_t.npz', time = np.linspace(0, FRAME*dt, FRAME), dist =  t)