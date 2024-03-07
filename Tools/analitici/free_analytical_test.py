
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from matplotlib.animation import FuncAnimation


sys.path.insert(0, '/home/falzo/Scrivania/Sol_eqS')

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

#FRAME = 300

def analitical(x, x0, p0, σ, h, m, t ):
    
    COST_1 = (2*σ**2 / np.pi)**(1./4)
    COST_2 = 1. / (σ**4 + ( (4 * h**2 * t**2)/ m**2 ))**(1./4)

    esp_1 = 1j * p0 * (x-x0) / h

    esp_2 = - (x -x0 - (p0 * t / m))**2 / (σ**2 + (1j * 2 * h * t / m))

    esp_3 = 1j * (- np.arctan(2 * h * t / (m* σ**2 )) / 2 - (p0**2 / (2*m*h)) * t ) # -1j * (p0**2 / (2*m*h)) * t  

    return COST_1 * COST_2 * np.exp(esp_1) * np.exp(esp_2) * np.exp(esp_3)


dx = x[1] - x[0]
t = np.empty(0)

#FRAME = 1000
o = np.load(PATH + '/0.npz')

x_0	 = o['x'] 
V	 = o['V'] 
Ψ	 = o['Ψ'] 

a = analitical(x, x0, p0, σ0, h, m, 0)
cont, norma = DFT.check_norma(DFT.norma(a, dx), cont, norma, STD_NORMA)

t = np.append(t, max(abs(Ψ - a)))

for i in range(1, FRAME):

    o = np.load(PATH + '/' + str(i)+'.npz')
    Ψ = o['Ψ']

    a = analitical(x, x0, p0, σ0, h, m, (i-1)*dt)
    cont, norma = DFT.check_norma(DFT.norma(a, dx), cont, norma, STD_NORMA)

    t = np.append(t, max(abs(Ψ - a)))

plt.figure()

plt.plot(np.linspace(0, FRAME*dt, FRAME), t)

plt.ylabel(r'$ sup |Ψ - a|$')
plt.xlabel('t')

plt.show()

print(max(abs(Ψ - a)), np.argmax(abs(Ψ - a)))

np.savez(PATH + '/results_t.npz', time = np.linspace(0, FRAME*dt, FRAME), dist =  t)



