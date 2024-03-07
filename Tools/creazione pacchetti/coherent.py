
import os
from pickletools import long4
from readline import replace_history_item
import sys
from tkinter.font import NORMAL
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from matplotlib.animation import FuncAnimation

import math


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

N_MAX = 171  # float( 171!) da errore

def coherent_anal(x, x0, ω, h, m, t, α_0):

    α = np.exp(-1j *  ω* t) *  α_0
    
    cost = (m * ω / np.pi / h)**(1/4)

    esp1 = -(m * ω / 2 / h) * ((x-x0) - np.sqrt(  2 * h / (m * ω ) ) * α.real)

    esp2 = 1j * np.sqrt( 2 * m * ω / h ) * α.imag

    esp3 = 1j * (- ω * t / 2)  +  abs(α_0)**2 * np.sin(2 * ω * t) / 2


    return cost * np.exp(esp1)* np.exp(esp2)* np.exp(esp3)


def coherent(k, ε, x, x0, m, ω, h):

    s = 0
    α = 0
    c_n = 1
    n = 0
    c = np.empty(0)
    while n < N_MAX : #abs(c_n) > ε or n < abs(20*k):
        
        c_n = np.exp( - abs(k)**2 / 2) * k**n / ((math.factorial(n)))**0.5
        
        y_n = DFT.arm_ES(x, x0, n, m, ω, h)

        α += c_n * y_n

        print(n, c_n)
        c = np.append(c, c_n)
        s += abs(c_n)**2

        n+=1
        if n == N_MAX: 
            print('n has reach the MAXIMUM value, sum has been truncated at N = 170')
            break
            

    print(s)

    return α , c

y, c = coherent(5, 10**(-30), x, x0, m, ω, h)


s = 0
for i in c : 
    s += abs(i)**2 
print(s)

print(DFT.norma(y, x[1] - x[0]))

plt.figure()
#plt.plot(x, abs(y)**2)
plt.plot(range(0, len(c)), c.real, label = 'Real')
plt.plot(range(0, len(c)), c.imag, label = 'Im')
plt.legend()
plt.show()

plt.figure()
#plt.plot(x, abs(y)**2)
plt.plot(x, abs(y)**2)
plt.show()


np.savez(PATH + '/custom.npz', x = x,  Ψ = y)
