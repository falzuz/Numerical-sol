'''
NOTE:
    - calocoli direttamente il modulo quadro pronto da plottare
'''


import os
from readline import replace_history_item
import sys
from tkinter.font import NORMAL
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

norma = norm

def analitical(x, x0, ω, h, m, t, amp):
    
    cost = (m * ω / np.pi / h)**(1/2)
    esp = - (m * ω / h) * (x -x0 - amp * np.cos( - ω * t))**2

    return cost * np.exp(esp) 



dx = x[1] - x[0]

#FRAME = 1000
o = np.load(PATH + '/0.npz')

x_0	 = o['x'] 
V	 = o['V'] 
Ψ	 = o['Ψ'] 

E = __.H_mean(x, p, Ψ, x[1] - x[0], p[1]- p[0], m, V, h, P_MAX, N, L)

print(E)
ampiezza = np.sqrt(E.real / a)
print(ampiezza)

a = analitical(x, x0, ω, h, m, 0, ampiezza)


#cont, norma = DFT.check_norma(DFT.norma(a, dx), cont, norma, STD_NORMA)



fig = plt.figure(figsize=(16,8), dpi=100)
ax = fig.add_subplot()
line1, = ax.plot(x, V, label = 'V', color='slategray', zorder = 1)
line2, = ax.plot(x, abs(Ψ)**2, label = r'$|ψ|^2$', color='#007FFF', zorder = 3)
line3, = ax.plot(x, np.full(len(x), E), '--', label = 'E', color='firebrick', zorder = 2) #ax.hlines(E , 0.1, L-0.1, colors = 'red', label = 'E', zorder = 0)

#linea, = ax.plot(x, abs(a)**2, label = r'$|a|^2$', color='green', zorder = 3)
linea, = ax.plot(x, a, label = r'$|a|^2$', color='green', zorder = 3)

val_medio = ax.axvline(__.mean_x(x, Ψ, dx)) 


line4 = ax.fill_between(x, V, np.full(len(x), -V_MAX), facecolor = '#CCCCCC', zorder = 1)
ax.set_ylim(-0.2 * max(abs(Ψ)**2), 1.2*max(abs(Ψ)**2))

ax.set_xlabel("x")
ax.set_ylabel("E")


#plt.show()
#exit()

#if real_flag == True: line5, = ax.plot(x, Ψ.real, label = r'$Re(ψ)$', color='palegreen', zorder = 0)
#if imag_flag == True: line6, = ax.plot(x, Ψ.imag, '--', label = r'$Im(ψ)$', color='palevioletred', zorder = 0)

line5 = plt.Line2D(x, Ψ.real, label = r'$Re(ψ)$', color='palegreen', zorder = 0)
line6 = plt.Line2D(x, Ψ.imag, linestyle='--', label = r'$Im(ψ)$', color='palevioletred', zorder = 0)

if real_flag == 1:
    ax.add_line(line5)


if imag_flag == 1:
    ax.add_line(line6)

ax.legend()

# initialization function: plot the background of each frame
def init():
    line2.set_data([], [])

    line5.set_data([], [])
    line6.set_data([], [])

    return line2, line5, line6

# animation function.  This is called sequentially
def animate(i):
    x = x_0

    o = np.load(PATH + '/' + str(i)+'.npz')
    Ψ = o['Ψ']
    
    line2.set_data(x, abs(Ψ)**2)

    line5.set_data(x, Ψ.real)
    line6.set_data(x, Ψ.imag)

    return line2, line5, line6

# initialization function: plot the background of each frame
def init_r():
    line2.set_data([], [])

    line5.set_data([], [])


    return line2, line5

# animation function.  This is called sequentially
def animate_r(i):
    x = x_0

    o = np.load(PATH + '/' + str(i)+'.npz')
    Ψ = o['Ψ']
    
    line2.set_data(x, abs(Ψ)**2)

    line5.set_data(x, Ψ.real)

    return line2, line5

# initialization function: plot the background of each frame
def init_i():
    line2.set_data([], [])

    line6.set_data([], [])

    return line2, line6

# animation function.  This is called sequentially
def animate_i(i):
    x = x_0

    o = np.load(PATH + '/' + str(i)+'.npz')
    Ψ = o['Ψ']
    
    line2.set_data(x, abs(Ψ)**2)

    line6.set_data(x, Ψ.imag)

    return line2, line6

# initialization function: plot the background of each frame
def init_w():
    line2.set_data([], [])
    linea.set_data([], [])

    #val_medio.set_xdata([])

    return line2,linea, val_medio


# animation function.  This is called sequentially
def animate_w(i):
    x = x_0

    global cont, norma, ampiezza

    o = np.load(PATH + '/' + str(i)+'.npz')
    Ψ = o['Ψ']

    
    line2.set_data(x, abs(Ψ)**2)

    a = analitical(x, x0, ω, h, m, i*dt, ampiezza)
    #cont, norma = DFT.check_norma(DFT.norma(a, dx), cont, norma, STD_NORMA)

    #linea.set_data(x, abs(a)**2)
    linea.set_data(x, a)

    val_medio.set_xdata([__.mean_x(x, Ψ, dx), __.mean_x(x, Ψ, dx)])

    return line2,linea, val_medio


# call the animator.  blit=True means only re-draw the parts that have changed.
if real_flag==1 and imag_flag==1: anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=FRAME, interval=20, blit=True)

elif real_flag==1 and imag_flag==0:   anim_r = animation.FuncAnimation(fig, animate_r, init_func=init_r,
                               frames=FRAME, interval=20, blit=True)

elif real_flag==0 and imag_flag==1: anim_i = animation.FuncAnimation(fig, animate_i, init_func=init_i,
                               frames=FRAME, interval=20, blit=True)

else: anim_w = animation.FuncAnimation(fig, animate_w, init_func=init_w,
                               frames=FRAME, interval=60, blit=True)

plt.show()

if V_flag == True: anim.save(PATH + '/video.mp4', fps=FPS)