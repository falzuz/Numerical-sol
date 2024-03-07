
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

level = 2
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

    print(c_n, vector, x, x0, m, ω, h)

    
    a = sup_evolution(x, x0, m, ω, h, c_n, vector, 0)


    #check if everything is correct
    for i in range(len(x_c)):
        if x_c[i] != x_0[i] or x[i] != x_0[i]:
            print('Error: x_c != x_0')
            exit()
        


        if a_0[i] != Ψ_0[i] : #or a[i] != a_0[i]:

            print('a_0 != Ψ_0' )

            plt.figure()
            plt.plot(x_c, abs(Ψ_0 )**2 )
            plt.plot(x_c, abs(a_0 )**2 )
            plt.plot(x_c, abs(a )**2, label = 'a')
            plt.legend()
            plt.show()


            exit()




else:
    a = Ψ
    a = einge_evolution(Ψ_0 , n, h, ω, 0)



cont, norma = DFT.check_norma(DFT.norma(a, dx), cont, norma, STD_NORMA)


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
line2, = ax.plot(x, abs(Ψ_0)**2, label = r'$|ψ|^2$', color='#007FFF', zorder = 3)
line3, = ax.plot(x, np.full(len(x), E), '--', label = 'E', color='firebrick', zorder = 2) #ax.hlines(E , 0.1, L-0.1, colors = 'red', label = 'E', zorder = 0)

linea, = ax.plot(x, abs(a)**2, label = r'$|a|^2$', color='green', zorder = 3)


line4 = ax.fill_between(x, V, np.full(len(x), -V_MAX), facecolor = '#CCCCCC', zorder = 1)
ax.set_ylim(-0.2 * max(abs(Ψ)**2), 1.2*max(abs(Ψ)**2))

ax.set_xlabel("x")
ax.set_ylabel("E")

#plt.show()


#if real_flag == True: line5, = ax.plot(x, Ψ.real, label = r'$Re(ψ)$', color='palegreen', zorder = 0)
#if imag_flag == True: line6, = ax.plot(x, Ψ.imag, '--', label = r'$Im(ψ)$', color='palevioletred', zorder = 0)

line5 = plt.Line2D(x, Ψ.real, label = r'$Re(ψ)$', color='palegreen', zorder = 2)
line6 = plt.Line2D(x, Ψ.imag, linestyle='--', label = r'$Im(ψ)$', color='palevioletred', zorder = 2)

lineaR = plt.Line2D(x, a.real, label = r'$Re(a)$', color='orange', zorder = 3)
lineaI = plt.Line2D(x, a.imag, label = r'$Im(a)$', linestyle='--', color='green', zorder = 3)


if real_flag == 1:
    ax.add_line(line5)
    ax.add_line(lineaR)


if imag_flag == 1:
    ax.add_line(line6)
    ax.add_line(lineaI)

ax.legend()

# initialization function: plot the background of each frame
def init():
    line2.set_data([], [])

    line5.set_data([], [])
    line6.set_data([], [])

    linea.set_data([], [])

    lineaR.set_data([], [])
    lineaI.set_data([], [])


    return line2, line5, line6,  linea, lineaR, lineaI

# animation function.  This is called sequentially
def animate(i):
    x = x_0

    global Ψ_0

    o = np.load(PATH + '/' + str(i)+'.npz')
    Ψ = o['Ψ']
    
    line2.set_data(x, abs(Ψ)**2)

    line5.set_data(x, Ψ.real)
    line6.set_data(x, Ψ.imag)

    if flag_s == 1: a = sup_evolution(x, x0, m, ω, h, c_n, vector, i*dt) 
    else: a = einge_evolution(Ψ_0 , n, h, ω, i*dt)

    linea.set_data(x, abs(a)**2)

    lineaR.set_data(x, a.real)
    lineaI.set_data(x, a.imag)




    return line2, line5, line6, linea, lineaR, lineaI

# initialization function: plot the background of each frame
def init_r():
    line2.set_data([], [])

    line5.set_data([], [])


    return line2, line5, linea, lineaR

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

    global cont, norma, Ψ_0 

    o = np.load(PATH + '/' + str(i)+'.npz')
    Ψ = o['Ψ']

    
    line2.set_data(x, abs(Ψ)**2)

    if flag_s == 1: a = sup_evolution(x, x0, m, ω, h, c_n, vector, i*dt) 
    else: a = einge_evolution(Ψ_0 , n, h, ω, i*dt)

    cont, norma = DFT.check_norma(DFT.norma(a, dx), cont, norma, STD_NORMA)

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