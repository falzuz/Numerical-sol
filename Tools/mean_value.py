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

import DFT

try:
	PATH = str(sys.argv[1])
	sys.path.insert(0, PATH)

	from Par import *

except:
	print('There are not arguments: picking Default Par')
	sys.path.insert(0, '/home/falzo/Scrivania/Sol_eqS')

	from Parameters import  *
	PATH = os.getcwd()

print('VALORI MEDI')
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


def update_graph(value):

    global flag

    if flag != 0:

        ax.texts.pop(-1)
        ax.texts.pop(-1)
        ax.texts.pop(-1)


        flag = 0
    
    o = np.load(PATH + '/' + F_.get() +'.npz' )

    global Ψ 
 
    Ψ = o['Ψ'] 

    line2.set_data(x, abs(Ψ)**2)

    canvas.draw()

def mean_x(x, y, dt):     #compute the mean_x value
    
    y_c = np.empty(0)
    for i in y:
        y_c = np.append (y_c, DFT.cong(i))
    
    mean_x = 0
    for i in range(len(x)):
        mean_x += (y_c[i] * x[i] * y[i]) * dt
    return mean_x

def mean_p(p, f, dp):     #compute the mean_x value
    
    global P_MAX

    f_c = np.empty(0)
    for i in f:
        f_c = np.append (f_c, DFT.cong(i))
    
    '''
    f = DFT.shift(f, p, P_MAX)
    f_c = DFT.shift(f_c, p, P_MAX)

    p = DFT.shift_p(p, P_MAX)
    '''

    mean_p = 0
    for i in range(len(p)):
        mean_p += (f_c[i] * p[i] * f[i]) * dp
    return mean_p

def H_mean(x, p, y, dx, dp, m, V):

    global h, P_MAX, N, L
    
    y_c = np.empty(0)
    for i in y:
        y_c = np.append (y_c, DFT.cong(i))

    ps, fy = DFT.full_Dft(y, x, p, h, P_MAX, N, L)
    #ps, fy_c = DFT.full_Dft(y_c, x, p, h, P_MAX, N, L)

    T = (1/2/m) * mean_x(ps**2, fy, dp)
    V = mean_x(V, y, dx)

    return T+V


def start():

    global flag, dx, lineM, dp, m, V

    if flag != 0:
        ax.texts.pop(-1)
        ax.texts.pop(-1)
        ax.texts.pop(-1)

    flag += 1

    MX = mean_x(x, Ψ, dx)
    MH = H_mean(x, p, Ψ, dx, dp, m, V)

    ps, f = DFT.full_Dft(Ψ, x, p, h, P_MAX, N, L)
    MP = mean_x(ps, f, dp)

    lineM.remove()

    ax.text(0.6*L, max(abs(Ψ)**2), '<X> = ' + '%.2f' % MX.real, color = 'black', fontsize = 18,  fontweight='bold')
    lineM = ax.vlines(MX, -0.2 * max(abs(Ψ)**2), 1.2*max(abs(Ψ)**2))
    
    ax.text(0.2*L, max(abs(Ψ)**2), 'H = ' + '%.2f' % MH.real, color = 'red', fontsize = 18, fontweight='bold')
    ax.text(0.2*L, 0.5*max(abs(Ψ)**2), '<P> = ' + '%.2f' % MP.real, color = 'red', fontsize = 18, fontweight='bold')


    canvas.draw()

    f = open('mean_x.txt', 'w')
    f.write('<x> = ' + str(MX) + '\n'
            '<p> = ' + str(MP) + '\n'
            '<H> = ' + str(MH))
    f.close()

def abort():
	f = open('abort.log', 'w')
	f.write('Simulation aborted')
	f.close()

	exit()



w = Tk()
w.title('')

style = ttk.Style(w)
w.tk.call('source', 'azure.tcl')
style.theme_use('azure')

flag = 0

fig = plt.figure(figsize=(10,3), dpi=100)
ax = fig.add_subplot()
line1, = ax.plot(x, V, label = 'V', color='slategray', zorder = 1)
line2, = ax.plot(x, abs(Ψ)**2, label = r'$|ψ|^2$', color='#007FFF', zorder = 3)
line3, = ax.plot(x, np.full(len(x), E), '--', label = 'E', color='firebrick', zorder = 2) #ax.hlines(E , 0.1, L-0.1, colors = 'red', label = 'E', zorder = 0)

line4 = ax.fill_between(x, V, np.full(len(x), -V_MAX), 
            facecolor = '#CCCCCC', alpha = 0.3, zorder = 1)
ax.set_ylim(-0.2 * max(abs(Ψ)**2), 1.2*max(abs(Ψ)**2))


ax.set_xlabel("x")
ax.set_ylabel("E")


#if real_flag == True: line5, = ax.plot(x, Ψ.real, label = r'$Re(ψ)$', color='palegreen', zorder = 0)
#if imag_flag == True: line6, = ax.plot(x, Ψ.imag, '--', label = r'$Im(ψ)$', color='palevioletred', zorder = 0)

line5 = plt.Line2D(x, Ψ.real, label = r'$Re(ψ)$', color='palegreen', zorder = 0)
line6 = plt.Line2D(x, Ψ.imag, linestyle='--', label = r'$Im(ψ)$', color='palevioletred', zorder = 0)


dx = x[1]-x[0]
MX = mean_x(x, Ψ, dx)

lineM = ax.vlines(MX, -0.2 * max(abs(Ψ)**2), 1.2*max(abs(Ψ)**2))

dp = p[1]-p[0]
MH = H_mean(x, p, Ψ, dx, dp, m, V)

ps, f = DFT.full_Dft(Ψ, x, p, h, P_MAX, N, L)
MP = mean_x(ps, f, dp)


scale = 1.2
old_fig = zoom_factory(ax,base_scale = scale)

canvas = FigureCanvasTkAgg(fig, master=w)  # A tk.DrawingArea.
canvas.draw()

toolbar = NavigationToolbar2Tk(canvas, w, pack_toolbar=False)
toolbar.update()



canvas.get_tk_widget().grid()
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

w.protocol("WM_DELETE_WINDOW", abort)
w.mainloop()
