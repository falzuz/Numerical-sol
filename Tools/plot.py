
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from matplotlib.animation import FuncAnimation


try:
	PATH = str(sys.argv[1]) #os.getcwd()
	sys.path.insert(0, PATH)

	from Par import *

except:
	print('There are not arguments: picking Default Par')
	sys.path.insert(0, '/home/falzo/Scrivania/Sol_eqS')

	from Parameters import  *
	PATH = os.getcwd()

print('PLOT')
print(PATH)


#testing...
flag_a = 1

def analitical(x, x0, p0, σ, h, m, t ):
    
    COST_1 = (2*σ**2 / np.pi)**(1./4)
    COST_2 = 1. / (σ**4 + ( (4 * h**2 * t**2)/ m**2 ))**(1./4)

    esp_1 = 1j * p0 * (x-x0) / h

    esp_2 = - (x -x0 - (p0 * t / m))**2 / (σ**2 + (1j * 2 * h * t / m))

    esp_3 = 1j * (- np.arctan(2 * h * t / (m* σ**2 )) / 2 - (p0**2 / (2*m*h)) * t ) # -1j * (p0**2 / (2*m*h)) * t  

    return COST_1 * COST_2 * np.exp(esp_1) * np.exp(esp_2) * np.exp(esp_3)

def einge_evolution(Ψ_0, n, h, ω, t):
    
    E_n = (0.5 + n) * h * ω

    return np.exp(-1j * E_n * t / h) * Ψ_0





choose = input('Select range of frames:\n'+
        '1) all\n' +
        '2) list\n')

try: choose = int(choose)
except: print('input error')

if choose == 1: frames = np.arange(0, FRAME)
if choose == 2: 
    ch_2 = input('Select range of frames:\n'+
        '1) from ... to...\n' +
        '2) manual\n')

    try: ch_2 = int(ch_2)
    except: print('input error')

    if ch_2 == 1: 
        start, end = input("insert... start end\n").split(' ')
        try:
            start = int(start)
            end = int(end)
        except:
            print('input error')
        
        frames = np.arange(start, end)
    
    if ch_2 == 2: 
        try:
            frames = []
      
            while True:
                frames.append(int(input()))
          
        except:
            print(frames)

o = np.load(PATH + '/0.npz')

x_0	 = o['x'] 
V	 = o['V'] 

print(frames)
for i in frames:

    #FRAME = 1000
    o = np.load(PATH + '/' + str(i) + '.npz')
    Ψ	 = o['Ψ'] 


    fig = plt.figure(figsize=(16,8), dpi=100)
    ax = fig.add_subplot()
    line1, = ax.plot(x, V, label = 'V', color='slategray', zorder = 1)
    line2, = ax.plot(x, abs(Ψ)**2, label = r'$|ψ|^2$', color='#007FFF', zorder = 3)
    line3, = ax.plot(x, np.full(len(x), E), '--', label = 'E', color='firebrick', zorder = 2) #ax.hlines(E , 0.1, L-0.1, colors = 'red', label = 'E', zorder = 0)

    line4 = ax.fill_between(x, V, np.full(len(x), -V_MAX), facecolor = '#CCCCCC', zorder = 1)
    ax.set_ylim(-0.2 * max(abs(Ψ)**2), 1.2*max(abs(Ψ)**2))

    ax.set_xlabel("x")
    ax.set_ylabel("E")


    #if real_flag == True: line5, = ax.plot(x, Ψ.real, label = r'$Re(ψ)$', color='palegreen', zorder = 0)
    #if imag_flag == True: line6, = ax.plot(x, Ψ.imag, '--', label = r'$Im(ψ)$', color='palevioletred', zorder = 0)

    line5 = plt.Line2D(x, Ψ.real, label = r'$Re(ψ)$', color='palegreen', zorder = 2)
    line6 = plt.Line2D(x, Ψ.imag, linestyle='--', label = r'$Im(ψ)$', color='palevioletred', zorder = 2)

    
    
    if flag_a == 1:  

        a = analitical(x, x0, p0, σ0, h, m, (i-1)*dt)

        M = np.argmax(abs(Ψ) - abs(a))
        print(max(abs(Ψ - a)), M)

        ax.vlines(x[M], -0.2 * max(abs(Ψ)**2), 1.2*max(abs(Ψ)**2))


        linea, = ax.plot(x, abs(a)**2, color = 'green', label = r'$|a|^2$', zorder = 5)

        lineaR = plt.Line2D(x, a.real, color = 'green', label = r'$Re(a)$', zorder = 5)
        lineaI = plt.Line2D(x, a.imag, linestyle='--', color = 'green', label = r'$Im(a)$', zorder = 5)

    if flag_a == 2:  

        a = einge_evolution(Ψ, n, h, ω, 0)


        M = np.argmax(abs(Ψ) - abs(a))
        print(max(abs(Ψ - a)), M)

        ax.vlines(x[M], -0.2 * max(abs(Ψ)**2), 1.2*max(abs(Ψ)**2))


        linea, = ax.plot(x, abs(a)**2, color = 'green', label = r'$|a|^2$', zorder = 5)

        lineaR = plt.Line2D(x, a.real, color = 'green', label = r'$Re(a)$', zorder = 5)
        lineaI = plt.Line2D(x, a.imag, linestyle='--', color = 'green', label = r'$Im(a)$', zorder = 5)


    if real_flag == 1:
        ax.add_line(line5)
        if flag_a ==1: ax.add_line(lineaR)


    if imag_flag == 1:
        ax.add_line(line6)
        if flag_a ==1:  ax.add_line(lineaI)

    ax.legend()


    plt.show()

