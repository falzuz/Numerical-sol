from ast import arg
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from matplotlib.animation import FuncAnimation

#sys.path.insert(0, '/home/falzo/Scrivania/Sol_eqS')

import DFT 
import time_evolution as __ 

try:
	PATH = str(sys.argv[1])
	sys.path.insert(0, PATH)
	print('check: ',PATH)

	from Par import *

except:
	print('There are not arguments: picking Default Par')
	from Parameters import  *
	PATH = os.getcwd()
	print(PATH)

print('Immagini')
print(PATH)


dx = x[1] - x[0]

#FRAME = 1000
o = np.load(PATH + '/0.npz')

x_0	 = o['x'] 
V	 = o['V'] 
Ψ	 = o['Ψ']

Ψ_0 = Ψ

E, T, U = __.H_mean(x, p, Ψ, x[1] - x[0], p[1]- p[0], m, V, h, P_MAX, int(N), L, int(N), np.ndarray((N, N), dtype=complex), np.zeros(N, dtype = complex), np.zeros(N, dtype = complex))

print(E)

if Potential == Potential_list[8]:
    print('ciao')
    E = 1/2 * E 
    for i in range(len(V)):
        V[i] = __.step(x[i], 5.02,  100) 

ampiezza = np.sqrt(2 * h / m / ω) * abs(5)
print(ampiezza)


norm_factor = 0

fontsize = 24
lim_inf =  1.1 * min(V) #-0.1 * max(abs(Ψ)**2)
lim_sup = 1.1 * max(abs(Ψ)**2)#

fig, axis = plt.subplots(3, 2, figsize=(16,12), dpi=100, sharex = True, sharey = 'row')


delta = 70
k = 0
k0 = 30
for i in range(3):
    for j in range(2):

        NUM = k * delta + k0
        o = np.load(PATH + '/' + str( NUM)+'.npz')
        Ψ = o['Ψ']
        V = o['V']

        axis[i,  j].plot(x, V,  
                        linewidth=2, 
                        label = 'V', #'Potenziale V', 
                        color='slategray', 
                        zorder = 1
                        )

        axis[i,  j].plot(x, np.full(len(x), E), '--', 
                        linewidth=2, 
                        label = r'$E_0$', #'Livello di energia', 
                        color='firebrick', 
                        zorder = 2
                        ) #ax.hlines(E , 0.1, L-0.1, colors = 'red', label = 'E', zorder = 0)


        axis[i,  j].plot(x, abs(Ψ)**2,  
                        linewidth=2, 
                        label = r'$|ψ|^2$', #'Numerica', #
                        color='#007FFF',   
                        zorder = 3
                        )

        axis[i,  j].fill_between(x, V, np.full(len(x), -V_MAX), facecolor = '#CCCCCC', zorder = 0)

        axis[i][j].set_ylim(lim_inf, lim_sup)

        axis[i][j].text(0, 0.8*lim_sup, 
                        't = ' + str(NUM), fontsize = fontsize - 4 )

        #axis[i][j].set_xlim(-0.1, L/2+ 0.02*L)

 

        

        k += 1

axis[0][0].legend(loc = 'upper right', fontsize = fontsize -2)


axis[2][0].tick_params(axis='x',  labelsize=fontsize)
axis[2][1].tick_params(axis='x',  labelsize=fontsize)
#axis[2][2].tick_params(axis='x',  labelsize=fontsize)

axis[2][0].locator_params(axis='x',   nbins = 5)
axis[2][1].locator_params(axis='x',   nbins = 5)
#axis[1][2].locator_params(axis='x',   nbins = 5)


axis[2][0].set_xlabel("x", fontsize = fontsize)
axis[2][1].set_xlabel("x", fontsize = fontsize)
#axis[2][2].set_xlabel("x", fontsize = fontsize)


axis[0][0].set_ylabel("Energia", fontsize = fontsize, labelpad = 20)
axis[1][0].set_ylabel("Energia", fontsize = fontsize, labelpad = 20)
axis[2][0].set_ylabel("Energia", fontsize = fontsize, labelpad = 20)


axis[0][0].tick_params(axis='y',  labelsize=fontsize)
axis[1][0].tick_params(axis='y',  labelsize=fontsize)
axis[2][0].tick_params(axis='y',  labelsize=fontsize)

fig.tight_layout()
plt.savefig("hole.png", bbox_inches='tight')
plt.show()



