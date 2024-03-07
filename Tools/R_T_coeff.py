from ast import arg
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from matplotlib.animation import FuncAnimation

import tkinter
from tkinter import *
from tkinter import ttk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

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

print('COEFFICIENTI')
print(PATH)


lim_R = 0.4        # limit on x axis
lim_T = 0.6



o = np.load(PATH + '/0.npz')

x	 = o['x'] 
V	 = o['V'] 
Ψ	 = o['Ψ'] 

def zoom_factory(ax,base_scale = 2.):
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            print(event.button)
        # set new limits
        ax.set_xlim([xdata - cur_xrange*scale_factor,
                     xdata + cur_xrange*scale_factor])
        ax.set_ylim([ydata - cur_yrange*scale_factor,
                     ydata + cur_yrange*scale_factor])
        plt.draw() # force re-draw

    fig = ax.get_figure() # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('scroll_event',zoom_fun)

    #return the function
    return zoom_fun

def update_vlines(value):
    lim_R = float(R_.get())
    lim_T = float(T_.get())

    global lineR, lineT

    lineR.remove()
    lineT.remove()

    #lineR = ax.vlines(lim_R*L, -0.2 * max(abs(Ψ)**2), 1.2*max(abs(Ψ)**2) )
    #lineT = ax.vlines(lim_T*L, -0.2 * max(abs(Ψ)**2), 1.2*max(abs(Ψ)**2) )

    lineR = ax.axvspan(0, lim_R*L, 
                            -V_MAX, +V_MAX, #-0.2 * max(abs(Ψ)**2), 1.2*max(abs(Ψ)**2), 
                            alpha = 0.3, color = 'red', zorder  = 1)
    lineT = ax.axvspan(lim_T*L, max(x), 
                            -V_MAX, +V_MAX, #-0.2 * max(abs(Ψ)**2), 1.2*max(abs(Ψ)**2), 
                            alpha = 0.3, color = 'palegreen', zorder = 1)

    canvas.draw()

def update_graph(value):

    global flag

    if flag != 0:
        ax.lines.pop(-1)
        ax.lines.pop(-1)

        ax.texts.pop(-1)
        ax.texts.pop(-1)

        flag = 0
    
    o = np.load(PATH + '/' + F_.get() +'.npz' )

    global Ψ 
 
    Ψ = o['Ψ'] 

    line2.set_data(x, abs(Ψ)**2)

    canvas.draw()


def start():

    global flag

    if flag != 0:
        ax.lines.pop(-1)
        ax.lines.pop(-1)

        ax.texts.pop(-1)
        ax.texts.pop(-1)

    flag += 1

    lim_R = float(R_.get())
    lim_T = float(T_.get())

    dx = x[1]-x[0]

    x_R = x[x < lim_R*L]
    x_T = x[x > lim_T*L]

    Ψ_R = Ψ[x < lim_R*L]
    Ψ_T = Ψ[x > lim_T*L]

    T = DFT.norma(Ψ_T, np.zeros(len(Ψ_T), dtype = complex), len(Ψ_T), dx)
    R = DFT.norma(Ψ_R, np.zeros(len(Ψ_R), dtype = complex), len(Ψ_R), dx)

    ax.plot(x_T, abs(Ψ_T)**2, color = 'green', zorder = 5)
    ax.plot(x_R, abs(Ψ_R)**2, color = 'red', zorder = 5)

    ax.text(0.8*L, 0.7*limsup, 'T = ' + '%.2f' % T.real, color = 'green', fontsize = fontsize,  fontweight='bold')
    ax.text(0.1*L, 0.7*limsup, 'R = ' + '%.2f' % R.real, color = 'red', fontsize = fontsize, fontweight='bold')

    canvas.draw()

    f = open(PATH + '/coefficient.txt', 'w')
    f.write('N = ' + str(N) + '\T = ' + str(T) + '\nR = ' + str(R))
    f.close()

def abort():
	f = open('abort.log', 'w')
	f.write('Simulation aborted')
	f.close()

	exit()



w = Tk()
w.title('')

style = ttk.Style(w)
w.tk.call('source', os.getcwd() +'/azure.tcl')
style.theme_use('azure')

flag = 0
E, T, U = __.H_mean(x, p, Ψ, x[1] - x[0], p[1]- p[0], m, V, h, P_MAX, int(N), L, int(N), np.ndarray((N, N), dtype=complex), np.zeros(N, dtype = complex), np.zeros(N, dtype = complex))

fontsize = 18


fig = plt.figure(figsize=(10, 5), dpi=100)
ax = fig.add_subplot()
line1, = ax.plot(x, V, label = 'V', color='slategray', zorder = 1)
line2, = ax.plot(x, abs(Ψ)**2, label = r'$|ψ|^2$', color='#007FFF', zorder = 3)
line3, = ax.plot(x, np.full(len(x), E), '--', label = 'E', color='firebrick', zorder = 2) #ax.hlines(E , 0.1, L-0.1, colors = 'red', label = 'E', zorder = 0)

line4 = ax.fill_between(x, V, np.full(len(x), -V_MAX), 
            facecolor = '#CCCCCC', alpha = 0.3, zorder = 1)

liminf = -0.2 * max(abs(Ψ)**2)
limsup =  1.2*max(abs(Ψ)**2)
ax.set_ylim(liminf, limsup)


lineR = ax.axvspan(0, lim_R*L, 
                            -V_MAX, +V_MAX, #-0.2 * max(abs(Ψ)**2), 1.2*max(abs(Ψ)**2), 
                            alpha = 0.3, color = 'red', zorder  = 1)
lineT = ax.axvspan(lim_T*L, max(x), 
                            -V_MAX, +V_MAX, #-0.2 * max(abs(Ψ)**2), 1.2*max(abs(Ψ)**2), 
                            alpha = 0.3, color = 'palegreen', zorder = 1)

ax.set_xlabel("x", fontsize = fontsize, labelpad = 10)
ax.set_ylabel("Energia", fontsize = fontsize, labelpad = 10 )

ax.tick_params(axis='both',  labelsize= fontsize)
#ax.set_yticks(np.linspace(-max(V), max(V)), 10)



#if real_flag == True: line5, = ax.plot(x, Ψ.real, label = r'$Re(ψ)$', color='palegreen', zorder = 0)
#if imag_flag == True: line6, = ax.plot(x, Ψ.imag, '--', label = r'$Im(ψ)$', color='palevioletred', zorder = 0)

line5 = plt.Line2D(x, Ψ.real, label = r'$Re(ψ)$', color='palegreen', zorder = 0)
line6 = plt.Line2D(x, Ψ.imag, linestyle='--', label = r'$Im(ψ)$', color='palevioletred', zorder = 0)


scale = 1.2
old_fig = zoom_factory(ax,base_scale = scale)

canvas = FigureCanvasTkAgg(fig, master=w)  # A tk.DrawingArea.
canvas.draw()

toolbar = NavigationToolbar2Tk(canvas, w, pack_toolbar=False)
toolbar.update()



canvas.get_tk_widget().grid(column = 0, row = 0)
toolbar.grid(column = 0, row = 1)




BAR = ttk.Frame(w, padding = 10)
BAR.grid(row = 2)

BUT= ttk.Frame(w, padding = 10)
BUT.grid(row = 3)





start = ttk.Button(BUT, text = 'Start', command = start)
start.grid(column = 0, row = 6, padx=1, pady = 5)

stop = ttk.Button(BUT, text = 'Abort', command = abort)
stop.grid(column = 1, row = 6, padx= 1, pady = 5)


F_Lab = ttk.Label(BAR, text="Frame")
F_ = ttk.Entry(BAR, width = 10, justify='center')

F_.bind('<Return>', update_graph)

F_Lab.grid(column = 1, row = 0, padx= 10, pady = 5)
F_.grid(column = 2, row = 0, padx= 10, pady = 5)

R_var = tkinter.DoubleVar()
T_var = tkinter.DoubleVar()

R_var.set(lim_R)
T_var.set(lim_T)

R_ = ttk.Scale(BAR,  length = 200,  from_=0, to=1, 
                            orient='horizontal' , command=update_vlines,
                            variable = R_var)
T_ = ttk.Scale(BAR,  length = 200,  from_=0, to=1, 
                            orient='horizontal', command=update_vlines,
                            variable = T_var)

R_.grid(column = 0, row = 0, padx= 1, pady = 5)
T_.grid(column = 3, row = 0, padx= 1, pady = 5)

w.protocol("WM_DELETE_WINDOW", abort)
w.mainloop()
