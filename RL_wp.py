
import os
from readline import replace_history_item
import sys
from tkinter.font import NORMAL
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from matplotlib.animation import FuncAnimation


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


def RL_wp(x, x0, p0, σ, h, m, t, α):

    st = σ*(1 + 1j * ( h * t / (2 * m * σ**2)) )

    cost = 1 / (2 * np.pi * st**2)**(1/4)

    v = p0 / m 

    esp1 = (-(x-x0-v*t)**2 / (4 * st * σ))
    esp2 = 1j * (p0/h) * (x - v*t / 2)

    y_G = cost * np.exp(esp1 + esp2)

    RL = (  (1j * (p0 / h)) -  (( x - x0 - v*t) / (2 *  σ * st) )    - α * np.tanh(α * x)  ) * y_G

    cost =  1 / np.sqrt(DFT.norma(RL, np.zeros(N, dtype = complex), N, x[1] - x[0]))

    return cost * RL, cost

def RL_wp_2(x, x0, xV, p0, σ, h, m, α, N, t=0, norm_factor = 0):

    st = σ*(1 + 1j * ( h * t / (2 * m * σ**2)) )

    cost = 1 / (2 * np.pi * st**2)**(1/4)

    v = p0 / m 

    esp1 = (-((x-xV)-x0-v*t)**2 / (4 * st * σ))
    esp2 = 1j * (p0/h) * ((x-xV) - v*t / 2)

    y_G = cost * np.exp(esp1 + esp2)

    RL = (  (1j * (p0 / h)) -  (( (x-xV) - x0 - v*t) / (2 *  σ * st) )    - α * np.tanh(α * (x-xV))  ) * y_G

    if norm_factor == 0: 
        print('bella')
        norm_factor =  1 / np.sqrt(DFT.norma(RL, np.zeros(N, dtype = complex), N, x[1] - x[0]))

    return norm_factor * RL, norm_factor




''' 
    COST_1 = (2*σ**2 / np.pi)**(1./4)
    COST_2 = 1. / (σ**4 + ( (4 * h**2 * t**2)/ m**2 ))**(1./4)

    esp_1 = 1j * p0 * (x-x0) / h

    esp_2 = - (x -x0 - (p0 * t / m))**2 / (σ**2 + (1j * 2 * h * t / m))

    esp_3 = 1j * (- np.arctan(2 * h * t / (m* σ**2 )) / 2 - (p0**2 / (2*m*h)) * t ) # -1j * (p0**2 / (2*m*h)) * t  

    rl_term = (1j * (p0/m) - (  (x-x0 - (p0/m) * t)**2 / (2* σ**2 * (1 + (1j*h*t / (2*m*σ**2))))) - α * np.tanh(α*x))

    y = COST_1 * COST_2 * np.exp(esp_1) * np.exp(esp_2) * np.exp(esp_3) * rl_term

    cost =  1 / np.sqrt(DFT.norma(y, x[1] - x[0]))

    return cost * y
'''


y, norm_1 = RL_wp(x, x0+2, p0, σ0, h, m, 0, α)
y_2, norm_2 = RL_wp_2(x, x0+2, 0, p0, σ0, h, m, α, N) #, norm_factor=norm_1)

print(norm_1 - norm_2)

#print(DFT.norma(y, np.zeros(N, dtype = complex), N, x[1] - x[0]))

plt.figure()
plt.plot(x, abs(y_2)**2, label = '2')
plt.plot(x, abs(y)**2, label = '1')
plt.legend()
plt.show()


#np.savez(PATH + '/custom.npz', x = x,  Ψ = y)
