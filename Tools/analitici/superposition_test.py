#! /bin/python3

import numpy as np

import os 

import sys
sys.path.insert(0, os.getcwd())
from Parameters import *
import DFT 
import time_evolution as __ 

try:
	PATH = str(sys.argv[1])
	sys.path.insert(0, PATH)
	print('check: ',PATH)

	from Par import *

except:
	print('There are not arguments: picking Default Par')
	sys.path.insert(0, os.getcwd())

	from Parameters import  *
	PATH = os.getcwd()


import matplotlib.pyplot as plt


vector = [0, 1]

c_n = [np.sqrt(1./2), np.sqrt(1./2)]

#Level = 2
y = DFT.arm_ES_superposition(x, x0, m, ω, h, c_n, vector)

#Level = 1
y_1 = DFT.arm_ES_superposition(x, x0, m, ω, h, c_n, vector)

plt.figure()
#plt.plot(x, abs(y)**2)
plt.plot(x, abs(y_1)**2)


np.savez(PATH + '/custom.npz', x = x,  Ψ = y, c_n = c_n, vector = vector )

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

y_t = sup_evolution(x, x0, m, ω, h, c_n, vector, 0)

for i in range(len(x)):
	if y_t[i] != y_1[i]: 
		print('diversi')
		exit()

plt.plot(x, abs(y_t)**2)
plt.show()

print(c_n, vector)







