
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
sys.path.insert(0, '/home/falzo/Scrivania/numba')

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

def analitical(x, x0, p0, σ, h, m, t, α):

    st = σ*(1 + 1j * ( h * t / (2 * m * σ**2)) )

    cost = 1 / (2 * np.pi * st**2)**(1/4)

    v = p0 / m 

    esp1 = (-(x-x0-v*t)**2 / (4 * st * σ))
    esp2 = 1j * (p0/h) * (x - v*t / 2)

    y_G = cost * np.exp(esp1 + esp2)

    RL = (  (1j * (p0 / h)) -  ((x - x0 - v*t) / (2 *  σ * st) )    - α * np.tanh(α * x)  ) * y_G

    global N
    cost =  1 / np.sqrt(   DFT.norma(RL, np.zeros(N, dtype = complex), N , x[1] - x[0]) )

    return cost * RL 


dx = x[1] - x[0]

#FRAME = 1000
o = np.load(PATH + '/0.npz')

x_0	 = o['x'] 
V	 = o['V'] 
Ψ	 = o['Ψ'] 

a = analitical(x, x0, p0, σ0, h, m, 0, α)

s = time.time()
E = __.H_mean(x, p, Ψ, x[1] - x[0], p[1]- p[0], m, V, h, P_MAX, int(N), L, int(N), np.ndarray((N, N), dtype=complex), np.zeros(N, dtype = complex))
print(time.time() - s)

#cont, norma = DFT.check_norma(DFT.norma(a, dx), cont, norma, STD_NORMA)


'''
x = x_0

plt.figure()
plt.plot(x, Ψ.real, '--' , label='real', zorder = 1)
plt.plot(x, Ψ.imag, '--' , label='imag', zorder = 1)
plt.plot(x, abs(ψ)**2, label='Ψ', zorder = 2 )
plt.plot(x, V, label = 'V', zorder = 0)
#plt.hlines(E , 1, L-1, colors = 'red', label = 'E', zorder = 0)
plt.legend()
plt.ylim(-0.1 * np.max(abs(ψ)**2), 1.1*np.max(abs(ψ)**2))
#plt.savefig('/mnt/Archivio/Sol_eqS/step/plot/0_mod.png')
plt.show()
plt.close()
'''

fig = plt.figure(figsize=(16,8), dpi=100)
ax = fig.add_subplot()
line1, = ax.plot(x, V, label = 'V', color='slategray', zorder = 1)
line2, = ax.plot(x, abs(Ψ)**2, label = r'$|ψ|^2$', color='#007FFF', zorder = 3)
line3, = ax.plot(x, np.full(len(x), E), '--', label = 'E', color='firebrick', zorder = 2) #ax.hlines(E , 0.1, L-0.1, colors = 'red', label = 'E', zorder = 0)

linea, = ax.plot(x, abs(a)**2, label = r'$|a|^2$', color='green', zorder = 3)

ax.plot(x, np.full(len(x), 1.5))

line4 = ax.fill_between(x, V, np.full(len(x), -V_MAX), facecolor = '#CCCCCC', zorder = 1)
ax.set_ylim(-0.2 * max(abs(Ψ)**2), 1.2*max(abs(Ψ)**2))

ax.set_xlabel("x")
ax.set_ylabel("E")





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
    return line2, linea

# animation function.  This is called sequentially
def animate_w(i):
    x = x_0

    global cont, norma

    o = np.load(PATH + '/' + str(i)+'.npz')
    Ψ = o['Ψ']

    
    line2.set_data(x, abs(Ψ)**2)

    a = analitical(x, x0, p0, σ0, h, m, (i-1)*dt, α)
#    cont, norma = DFT.check_norma(DFT.norma(a, dx), cont, norma, STD_NORMA)

    linea.set_data(x, abs(a)**2)

    return line2,linea


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