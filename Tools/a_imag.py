# Crea l'immagine da mettere nel tex
# I parametri sono più in basso riga 180

import os
from readline import replace_history_item
import sys
from time import time
from tkinter.font import NORMAL
from tokenize import Number
from matplotlib.lines import lineStyles
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from matplotlib.animation import FuncAnimation

import time

#from Parameters_numba import Potential

sys.path.insert(0, os.getcwd())

print(os.getcwd())

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

t0 =  np.pi / 2 / ω


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

def free_evolution(x, x0, p0,  σ0, h, m, t ):
    
    COST_1 = (2* σ0**2 / np.pi)**(1./4)
    COST_2 = 1. / ( σ0**4 + ( (4 * h**2 * t**2)/ m**2 ))**(1./4)

    esp_1 = 1j * p0 * (x-x0) / h

    esp_2 = - (x -x0 - (p0 * t / m))**2 / ( σ0**2 + (1j * 2 * h * t / m))

    esp_3 = 1j * (- np.arctan(2 * h * t / (m*  σ0**2 )) / 2 - (p0**2 / (2*m*h)) * t ) # -1j * (p0**2 / (2*m*h)) * t  

    return COST_1 * COST_2 * np.exp(esp_1) * np.exp(esp_2) * np.exp(esp_3)

def RL_evolution(x, x0, p0,  σ0, h, m, α, x_pot, norm_factor, t):

    st =  σ0*(1 + 1j * ( h * t / (2 * m *  σ0**2)) )

    cost = 1 / (2 * np.pi * st**2)**(1/4)

    v = p0 / m 

    esp1 = (-(x-x0-v*t)**2 / (4 * st *  σ0))
    esp2 = 1j * (p0/h) * (x - v*t / 2)

    y_G = cost * np.exp(esp1 + esp2)

    RL = (  (1j * (p0 / h)) -  ((x - x_pot - x0 - v*t) / (2 *   σ0 * st) )    - α * np.tanh(α * (x - x_pot))  ) * y_G

    return norm_factor * RL

#NOTA: ritorna già il modulo quadro
def coherent_evolution(x, x0, ω, h, m, amp, t):
    
    cost = (m * ω / np.pi / h)**(1/2)
    esp = - (m * ω / h) * (x -x0 - amp * np.cos( - ω * t))**2

    return cost * np.exp(esp) 

def analitical(Potential, ψ_name, x, x0, p0,  σ0, h, m, α, ω, Ψ_0, n, amp,  x_pot,  norm_factor, t):

    if Potential == 'Reflectionless':  
        y, c = DFT.RL_wp(x, x0-x_pot, x_pot, p0, σ0, h, m, α, N, t, norm_factor = norm_factor)     #RL_evolution(x, x0, p0,  σ0, h, m, α, x_pot,  norm_factor, t)
        return y

    elif Potential == 'Free': 
       return free_evolution(x, x0, p0,  σ0, h, m, t )

    elif Potential == 'Armonic Oscillator':
        if ψ_name == 'Custom':            
            c = np.load(PATH + '/custom.npz')

            x_c = c['x']
            a_0	= c['Ψ'] 
            c_n = c['c_n']
            vector = c['vector']

            print(c_n, vector, x, x0, m, ω, h)

            return sup_evolution(x, x0, m, ω, h, c_n, vector, t)
        
        elif ψ_name == 'Coherent':  return coherent_evolution(x, x0, ω, h, m, amp, t)
        
        else: return einge_evolution(Ψ_0, n, h, ω, t)

def choosed_analitical(choice, x, x0, p0,  σ0, h, m, α, ω, Ψ_0, n, amp, x_pot, norm_factor, t):
    
    if choice == 1: 
        y, norm_factor = DFT.RL_wp(x, x0, x_pot, p0, σ0, h, m, α, N, t, norm_factor = norm_factor)     #RL_evolution(x, x0, p0,  σ0, h, m, α, x_pot,  norm_factor, t)
        return y
    
    elif choice == 2: return free_evolution(x, x0, p0,  σ0, h, m, t )

    elif choice == 3:
    
            c = np.load(PATH + '/custom.npz')

            x_c = c['x']
            a_0	= c['Ψ'] 
            c_n = c['c_n']
            vector = c['vector']

            print(c_n, vector, x, x0, m, ω, h)

            return sup_evolution(x, x0, m, ω, h, c_n, vector, t)
        
    elif choice == 4 : return coherent_evolution(x, x0, ω, h, m, amp, t)
        
    elif choice == 5: return einge_evolution(Ψ_0, n, h, ω, t)

    else: 
        print('Error choice')
        exit()


dx = x[1] - x[0]

#FRAME = 1000
o = np.load(PATH + '/0.npz')

x_0	 = o['x'] 
V	 = o['V'] 
Ψ	 = o['Ψ']

Ψ_0 = Ψ

E, T, U = __.H_mean(x, p, Ψ, x[1] - x[0], p[1]- p[0], m, V, h, P_MAX, int(N), L, int(N), np.ndarray((N, N), dtype=complex), np.zeros(N, dtype = complex), np.zeros(N, dtype = complex))

print(E)

ampiezza = np.sqrt(2 * h / m / ω) * abs(5)
print(ampiezza)

cc = 0
auto_man = int(input('Choose: \n (1) auto \n (2) Choose \n\n' ))
if auto_man == 2: cc = int(input('Choose: \n (1) Free \n (2) Reflectionless \n (3) Superposition \n (4) Coherent state \n (5) Armonic eigenstate \n\n'))


norm_factor = 0

fontsize = 24
lim_inf = -0.1 #* max(abs(Ψ)**2)
lim_sup = 1.1 * max(abs(Ψ)**2)

fig, axis = plt.subplots(3, 2, figsize=(16,12), dpi=100, sharex = True, sharey = 'row')

x_corr = 0 #0.04683980635342477 # 0.029781798015256378

delta = 90
k = 0
k0 = 0
for i in range(3):
    for j in range(2):

        NUM = k * delta + k0

        o = np.load(PATH + '/' + str(NUM)+'.npz')
        Ψ = o['Ψ']

        if auto_man == 1: a = analitical(Potential, ψ_name, x, x0+x_corr, p0,  σ0, h, m, α, ω, Ψ_0, n, ampiezza, x_V_centered,  norm_factor, NUM)
        elif auto_man == 2: 
            a = choosed_analitical(cc, x, x0, p0,  σ0, h, m, α, ω, Ψ_0, n, ampiezza , x_V_centered,  norm_factor, NUM + t0)

        #a_free = analitical('Free', 'Wave Packet', x, x0, p0,  σ0, h, m, α, ω, Ψ_0, n, ampiezza, x_V_centered, norm_factor, NUM)

        axis[i,  j].plot(x, V,  
                        linewidth=2, 
                        label = 'Potenziale V', 
                        color='slategray', 
                        zorder = 1
                        )

        axis[i,  j].plot(x, np.full(len(x), E), '--', 
                        linewidth=2, 
                        label = r'$E_0$', #'Livello energia', 
                        color='firebrick', 
                        zorder = 2
                        ) #ax.hlines(E , 0.1, L-0.1, colors = 'red', label = 'E', zorder = 0)


        axis[i,  j].plot(x, abs(Ψ)**2,  
                        linewidth=2, 
                        label = 'Numerica', #r'$|ψ|^2$', 
                        color='#007FFF',   
                        zorder = 4
                        )


        if ψ_name == 'Coherent' or cc == 4: 

            axis[i,  j].plot(x, a,
                                linewidth=2, 
                                label = 'Coerente',  #  'Analitica', #r'$|a|^2$', 
                                color='green', 
                                zorder = 5
                                )

        
        else: 
            axis[i,  j].plot(x, abs(a)**2, 
                                linewidth=2, 
                                label = 'Analitica', #r'$|a|^2$', 
                                color='green', 
                                zorder = 5)

        '''
        axis[i,  j].plot(x, abs(a_free)**2, 
                        linewidth=2, 
                        label = 'Libera', #r'$|a|^2$', 
                        color='orange', 
                        zorder = 3)
        '''

        axis[i,  j].fill_between(x, V, np.full(len(x), -V_MAX), facecolor = '#CCCCCC', zorder = 1)

        axis[i][j].set_ylim(lim_inf, lim_sup)


        axis[i][j].text(0, 1.7, 't = ' + str(NUM), fontsize = fontsize -2 )
        
       
        #axis[i][j].axvline(__.mean_x(x, Ψ,      dx, N, np.zeros(N, dtype = complex), mean_x = 0), linestyle = '--', color='#007FFF', zorder = 4)
        #axis[i][j].axvline(__.mean_x(x, a,      dx, N, np.zeros(N, dtype = complex), mean_x = 0), linestyle = '--', color='green', zorder = 0) 
        #axis[i][j].axvline(__.mean_x(x, a_free, dx, N, np.zeros(N, dtype = complex), mean_x = 0), linestyle = '--', color='orange', zorder = 0) 

        '''
        mean_c  = 0
        print(mean_c)
        for figa in range(N):
            mean_c += (a[figa] * x[figa]) * dt

        axis[i][j].axvline( x[np.argmax(abs(Ψ)**2)] , linestyle = '--', color='#007FFF', zorder = 0)
        axis[i][j].axvline( x[np.argmax(a)] , 
                            linestyle = '--', color='green', zorder = 0) 
        '''

        k += 1

axis[0][0].legend(loc = 'upper right', fontsize = fontsize - 4) #, bbox_to_anchor = [1.1, 1])


axis[2][0].tick_params(axis='x',  labelsize=fontsize)
axis[2][1].tick_params(axis='x',  labelsize=fontsize)


axis[2][0].set_xlabel("x", fontsize = fontsize)
axis[2][1].set_xlabel("x", fontsize = fontsize)


axis[0][0].set_ylabel("Energia", fontsize = fontsize, labelpad = 20)
axis[1][0].set_ylabel("Energia", fontsize = fontsize, labelpad = 20)
axis[2][0].set_ylabel("Energia", fontsize = fontsize, labelpad = 20)


axis[0][0].tick_params(axis='y',  labelsize=fontsize)
axis[1][0].tick_params(axis='y',  labelsize=fontsize)
axis[2][0].tick_params(axis='y',  labelsize=fontsize)

'''
# inset axes....
axins = axis[2][1].inset_axes([0.05, 0.37, 0.5, 0.6])
# sub region of the original image
x1, x2, y1, y2 = 6, 7.4, 0.25, 0.55
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels([])
axins.set_yticklabels([])

axins.axvline(__.mean_x(x, Ψ,      dx, N, np.zeros(N, dtype = complex), mean_x = 0), linestyle = '--', color='#007FFF', zorder = 0)
axins.axvline(__.mean_x(x, a,      dx, N, np.zeros(N, dtype = complex), mean_x = 0), linestyle = '--', color='green', zorder = 0) 
axins.axvline(__.mean_x(x, a_free, dx, N, np.zeros(N, dtype = complex), mean_x = 0), linestyle = '--', color='orange', zorder = 0) 


axins.plot(x, abs(Ψ)**2,  
                linewidth=2, 
                label = 'WP onde piane', #'Numerica', #r'$|ψ|^2$', 
                color='#007FFF',   
                zorder = 4
                )


axins.plot(x, abs(a)**2, 
                        linewidth=2, 
                        label = 'WP autostati', #'Analitica', #r'$|a|^2$', 
                        color='green', 
                        zorder = 3)


axins.plot(x, abs(a_free)**2, 
                linewidth=2, 
                label = 'Libera', #r'$|a|^2$', 
                color='orange', 
                zorder = 3)

axis[2][1].indicate_inset_zoom(axins, edgecolor = "gray")
'''

fig.tight_layout()
plt.savefig("superposition.png", bbox_inches='tight')
plt.show()

#zoom
'''
plt.figure(figsize=(16,8), dpi=100)

a = analitical(Potential, ψ_name, x, x0, p0,  σ0, h, m, α, ω, Ψ_0, n, ampiezza, x_V_centered,  norm_factor, NUM)
a_free = analitical('Free', 'Wave Packet', x, x0, p0,  σ0, h, m, α, ω, Ψ_0, n, ampiezza, x_V_centered, norm_factor, NUM)

o = np.load(PATH + '/' + str(NUM)+'.npz')
Ψ = o['Ψ']


plt.plot(x, V,  
                        linewidth=2, 
                        label = 'Potenziale V', 
                        color='slategray', 
                        zorder = 1
                        )

plt.plot(x, np.full(len(x), E), '--', 
                        linewidth=2, 
                        label = r'$E_0$', #'Livello energia', 
                        color='firebrick', 
                        zorder = 2
                        ) #ax.hlines(E , 0.1, L-0.1, colors = 'red', label = 'E', zorder = 0)


plt.plot(x, abs(Ψ)**2,  
                        linewidth=2, 
                        label = 'WP onde piane', #'Numerica', #r'$|ψ|^2$', 
                        color='#007FFF',   
                        zorder = 5
                        )


plt.plot(x, abs(a_free)**2, 
                        linewidth=2, 
                        label = 'Libera', #r'$|a|^2$', 
                        color='orange', 
                        zorder = 3)






plt.legend(loc = 'upper left', fontsize = fontsize - 4) #, bbox_to_anchor = [1.1, 1])

plt.xlim(4, 10)
plt.ylim(lim_inf, 1.2*max(abs(a_free)**2))

plt.text(9, 0.55, 't = ' + str(NUM), fontsize = fontsize - 2 )

plt.savefig("WP_RL_zoom.png", bbox_inches='tight')

plt.show()
'''