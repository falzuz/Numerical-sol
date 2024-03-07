# Mette a confronto l'evoluzione analitica con quella numerica
# imposta t0 = 0.....


# Potenzilae figo - (1/2000) * (2**2) * 20*(20+1) *   ( 1 / ((numpy.cosh((x - 20) * 2))**2) + 1 / ((numpy.cosh((x - 40) * 2 ))**2) + 1 / ((numpy.cosh((x -60) * 2 ))**2))


import os
from readline import replace_history_item
import sys
from time import time
from tkinter.font import NORMAL
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

#FRAME = 410
print('ANALYTICAL')
print(PATH)

t0 = 0 #np.pi / 2 / ω

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



#NOTA: ritorna già il modulo quadro
def coherent_evolution(x, x0, ω, h, m, amp, t):
    
    cost = (m * ω / np.pi / h)**(1/2)
    esp = - (m * ω / h) * (x -x0 - amp * np.cos( - ω * t))**2

    return cost * np.exp(esp) 

def analitical(Potential, ψ_name, x, x0, p0,  σ0, h, m, α, ω, Ψ_0, n, amp,  x_pot,  norm_factor, t):

    if Potential == 'Reflectionless':  
        x0 = x0 - x_pot
        y, c = DFT.RL_wp(x, x0 + x_pot, x_pot, p0, σ0, h, m, α, N, t, norm_factor = norm_factor)     #RL_evolution(x, x0, p0,  σ0, h, m, α, x_pot,  norm_factor, t)
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

#FRAME = 360
o = np.load(PATH + '/0.npz')

x_0	 = o['x'] 
V	 = o['V'] 
Ψ	 = o['Ψ']

Ψ_0 = Ψ

E, T, U = __.H_mean(x, p, Ψ, x[1] - x[0], p[1]- p[0], m, V, h, P_MAX, int(N), L, int(N), np.ndarray((N, N), dtype=complex), np.zeros(N, dtype = complex), np.zeros(N, dtype = complex))

print(E)
print(norm_factor)

ampiezza = np.sqrt(2 * h / m / ω) * abs(5)
print(ampiezza)

x_corr = 0 # 0.04683980635342477 #0.029781798015256378
#norm_factor = 0
debug_norma = False
auto_man = int(input('Choose: \n (1) auto \n (2) Choose \n\n' ))
cc = 0
if auto_man == 1: a = analitical(Potential, ψ_name, x, x0 + x_corr, p0,  σ0, h, m, α, ω, Ψ_0, n, ampiezza, x_V_centered, norm_factor, t0)
elif auto_man == 2:
    cc = int(input('Choose: \n (2) Reflectionless \n (1) Free \n (3) Superposition \n (4) Coherent state \n (5) Armonic eigenstate \n\n'))
    a = choosed_analitical(cc, x, x0, p0,  σ0, h, m, α, ω, Ψ_0, n, ampiezza, x_V_centered, norm_factor, t0)

if debug_norma: cont, norm = DFT.check_norma(DFT.norma(a, np.zeros(N, dtype = complex), N , dx), cont, norm, STD_NORMA)

fontsize = 22
fig = plt.figure(figsize=(16,8), dpi=100)
ax = fig.add_subplot()

ax.set_xlabel("x", fontsize = fontsize, labelpad =5)
ax.set_ylabel("Energia", fontsize = fontsize)# ,labelpad =5)

ax.tick_params(axis='x',  labelsize=fontsize)
ax.tick_params(axis='y',  labelsize=fontsize)

ax.set_xticks(np.linspace(0, x[-1], 11))#, endpoint = False))
ax.set_xticklabels(['0', '', '', '', '','', '', '', '', '', 'L'])

#confronto tra piane e coerenti
'''
coe = np.load('/home/falzo/Scrivania/Dati/oscillatore/coerenti/Run/0.npz')

V_coe	 = coe['V'] 
Ψ_coe	 = coe['Ψ']

E_c, T, U = __.H_mean(x, p, Ψ_coe, x[1] - x[0], p[1]- p[0], m, V, h, P_MAX, int(N), L, int(N), np.ndarray((N, N), dtype=complex), np.zeros(N, dtype = complex), np.zeros(N, dtype = complex))


print('>>>>>' , V - V_coe, E - E_c)

ax.plot(x, np.full(len(x), E_c), '--', 
                    linewidth=2, 
                    label = 'E_c', 
                    color='blue', 
                    zorder = 2
                    )
'''

#confronto in RL
'''
a_free = analitical('Free', 'Wave Packet', x,
                x0,#  - 0.009, # + x_V_centered + 0.065, 
                p0,  σ0, h, m, α, ω, Ψ_0, n, ampiezza, x_V_centered, norm_factor, t0)

E_f, T, U = __.H_mean(x, p, a_free, x[1] - x[0], p[1]- p[0], m, V, h, P_MAX, int(N), L, int(N), np.ndarray((N, N), dtype=complex), np.zeros(N, dtype = complex), np.zeros(N, dtype = complex))
E_a , T, U =  __.H_mean(x, p, a, x[1] - x[0], p[1]- p[0], m, V, h, P_MAX, int(N), L, int(N), np.ndarray((N, N), dtype=complex), np.zeros(N, dtype = complex), np.zeros(N, dtype = complex))

print('>>>>>>>>> ', E_f - E, E_a - E)

line_free, = ax.plot(x, abs(a_free)**2, 
                        linewidth=2, 
                        label = 'Libera', #r'$|a|^2$', 
                        color='orange', 
                        zorder = 3)
'''

line1, = ax.plot(x, V,  
                    linewidth=2, 
                    label = 'Potenziale V', 
                    color='slategray', 
                    zorder = 1
                    )

line3, = ax.plot(x, np.full(len(x), E), '--', 
                    linewidth=2, 
                    label = 'Livello di energia', 
                    color='firebrick', 
                    zorder = 2
                    ) #ax.hlines(E , 0.1, L-0.1, colors = 'red', label = 'E', zorder = 0)



line2, = ax.plot(x, abs(Ψ)**2,  
                    linewidth=2, 
                    label = 'Numerica', #r'$|ψ|^2$', 
                    color='#007FFF',   
                    zorder = 3
                    )


if ψ_name == 'Coherent' or cc == 4: 
    
    linea, = ax.plot(x, a,
                        linewidth=2, 
                        label = 'Analitica', #r'$|a|^2$', 
                        color='green', 
                        zorder = 3
                        )


else: 
    linea, = ax.plot(x, abs(a)**2, 
                        linewidth=2, 
                        label = 'Analitica', #r'$|a|^2$', 
                        color='green', 
                        zorder = 3)

line4 = ax.fill_between(x, V, np.full(len(x), -V_MAX), facecolor = '#CCCCCC', zorder = 1)
ax.set_ylim(-0.1 * max(abs(Ψ)**2), 1.1*max(abs(Ψ)**2))


line5 = plt.Line2D(x, Ψ.real, label = r'$Re(ψ)$', color='palegreen', zorder = 0)
line6 = plt.Line2D(x, Ψ.imag, linestyle='--', label = r'$Im(ψ)$', color='palevioletred', zorder = 0)

if real_flag == 1:
    ax.add_line(line5)


if imag_flag == 1:
    ax.add_line(line6)

ax.legend(fontsize = fontsize - 2, loc = 'upper right', bbox_to_anchor=(1.1, 1.15), framealpha = 1 )

#ax.axvline(__.mean_x(x, Ψ,      dx, N, np.zeros(N, dtype = complex), mean_x = 0), linestyle = '--', color='#007FFF', zorder = 0)
#ax.axvline(__.mean_x(x, a,      dx, N, np.zeros(N, dtype = complex), mean_x = 0), linestyle = '--', color='green', zorder = 0) 
#ax.axvline(__.mean_x(x, a_free, dx, N, np.zeros(N, dtype = complex), mean_x = 0), linestyle = '--', color='orange', zorder = 0) 

print(__.mean_x(x, Ψ,  dx, N, np.zeros(N, dtype = complex), mean_x = 0) - __.mean_x(x, a,      dx, N, np.zeros(N, dtype = complex), mean_x = 0))

#plt.show()

# initialization function: plot the background of each frame
def init_w():
    line2.set_data([], [])
    linea.set_data([], [])
    return line2, linea

# animation function.  This is called sequentially
def animate_w(i):
    x = x_0

    global cont, norm, Ψ_0, auto_man, cc

    
    o = np.load(PATH + '/' + str(i)+'.npz')
    Ψ = o['Ψ']
    V = o['V']
    
    line2.set_data(x, abs(Ψ)**2)

    if auto_man == 1: a = analitical(Potential, ψ_name, x, x0+ x_corr, p0,  σ0, h, m, α, ω, Ψ_0, n, ampiezza, x_V_centered,  norm_factor,(i-1)*dt + t0)
    elif auto_man == 2: a = choosed_analitical(cc, x, x0, p0,  σ0, h, m, α, ω, Ψ_0, n, ampiezza , x_V_centered,  norm_factor, (i-1)*dt + t0)

    if debug_norma: cont, norm = DFT.check_norma(DFT.norma(a, np.zeros(N, dtype = complex), N , dx), cont, norm, STD_NORMA)

    if ψ_name == 'Coherent' or cc == 4: linea.set_data(x, a)
    else: linea.set_data(x, abs(a)**2)
    '''
    a_free = analitical('Free', 'Wave Packet', x, 
                            x0, #- 0.009, # + x_V_centered + 0.065, 
                            p0,  σ0, h, m, α, ω, Ψ_0, n, ampiezza, x_V_centered, norm_factor, (i-1)*dt + t0)
    line_free.set_data(x, abs(a_free)**2)
    '''

    return line2, linea #, line_free


anim_w = animation.FuncAnimation(fig, animate_w, init_func=init_w,
                               frames=FRAME, interval=20, blit=True)

#fig.tight_layout()
plt.show()

V_flag = True
if V_flag == True: anim_w.save(PATH + '/video.gif', fps=2*FPS)
#if V_flag == True: anim_r.save(PATH + '/video.gif', fps= 2* FPS)
