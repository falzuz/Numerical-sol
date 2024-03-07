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

print('ANIMAZIONE')
print(PATH)

if STOP_BOUNDARIES: 
	debug_norma = False
	debug_E = False

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




FRAME = 150
o = np.load(PATH + '/0.npz')

x_0	 = o['x'] 
V	 = o['V'] 
Ψ	 = o['Ψ'] 




E, T, U = __.H_mean(x, p, Ψ, x[1]-x[0], p[1]-p[0], m, V, h, P_MAX, N, L, N, np.zeros((N,N) , dtype=complex), np.zeros(N, dtype=complex),  np.zeros(N, dtype=complex))


fontsize = 22


fig = plt.figure(figsize=(16,8), dpi=100)
ax = fig.add_subplot()
line1, = ax.plot(x, V, label = 'V', color='slategray', zorder = 1)
line2, = ax.plot(x, abs(Ψ)**2, label = r'$|ψ|^2$', color='#007FFF', zorder = 3)
line3, = ax.plot(x, np.full(len(x), E), '--', label = r'$E_0$', color='firebrick', zorder = 2) #ax.hlines(E , 0.1, L-0.1, colors = 'red', label = 'E', zorder = 0)

line4 = ax.fill_between(x, V, np.full(len(x), -V_MAX), facecolor = '#CCCCCC', zorder = 0)
ax.set_ylim(-0.2 * max(abs(Ψ)**2), 1.2*max(abs(Ψ)**2))

ax.set_xlabel("x", fontsize = fontsize, labelpad =5)
ax.set_ylabel("Energia", fontsize = fontsize,labelpad = 10)

ax.tick_params(axis='x',  labelsize=fontsize)
ax.tick_params(axis='y',  labelsize=fontsize)

ax.set_xticks(np.linspace(0, x[-1], 11))#, endpoint = False))
ax.set_xticklabels(['0', '', '', '', '','', '', '', '', '', 'L'])


#if real_flag == True: line5, = ax.plot(x, Ψ.real, label = r'$Re(ψ)$', color='palegreen', zorder = 0)
#if imag_flag == True: line6, = ax.plot(x, Ψ.imag, '--', label = r'$Im(ψ)$', color='palevioletred', zorder = 0)

line5 = plt.Line2D(x, Ψ.real, label = r'$Re(ψ)$', color='palegreen', zorder = 0)
line6 = plt.Line2D(x, Ψ.imag, linestyle='--', label = r'$Im(ψ)$', color='palevioletred', zorder = 0)

#real_flag = True
if real_flag == 1:
    ax.add_line(line5)


if imag_flag == 1:
    ax.add_line(line6)

if Potential == Potential_list[8]:
	ax.set_xlim(0, L/2)

legend = ax.legend(loc = 'upper left',fontsize = fontsize - 2)
#ax.grid(linewidth = 1, linestyle ='--', zorder = 0)

#plt.show()


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
    line1.set_data([], [])

    return line2, line1#, line4

# animation function.  This is called sequentially
def animate_w(i):
    x = x_0

    o = np.load(PATH + '/' + str(i)+'.npz')
    Ψ = o['Ψ']
    V = o['V']

    line2.set_data(x, abs(Ψ)**2)
    line1.set_data(x, V)

    global line4
    line4.remove()
    line4 = ax.fill_between(x, V, np.full(len(x), -V_MAX), facecolor = '#CCCCCC', zorder = 0)

    #E, T, U  = __.H_mean(x, p, Ψ, x[1]-x[0], p[1]-p[0], m, V, h, P_MAX, N, L, N, np.zeros((N,N) , dtype=complex), np.zeros(N, dtype=complex),  np.zeros(N, dtype=complex))
    line3.set_data(x, np.full(N, E))

    legend = ax.legend(loc = 'upper left',fontsize = fontsize - 2)



    return line2, line1, line4, line3, legend


# call the animator.  blit=True means only re-draw the parts that have changed.
if real_flag==1 and imag_flag==1: anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=FRAME, interval=20, blit=True)

elif real_flag==1 and imag_flag==0:   anim_r = animation.FuncAnimation(fig, animate_r, init_func=init_r,
                               frames=FRAME, interval=20, blit=True)

elif real_flag==0 and imag_flag==1: anim_i = animation.FuncAnimation(fig, animate_i, init_func=init_i,
                               frames=FRAME, interval=20, blit=True)

else: anim_w = animation.FuncAnimation(fig, animate_w, init_func=init_w,
                               frames=FRAME, interval=20, blit=True)


scale = 1.2
old_fig = zoom_factory(ax,base_scale = scale)

#fig.tight_layout()
plt.show()

V_flag = True
if V_flag == True: anim_w.save(PATH + '/video.gif', fps= FPS)
#if V_flag == True: anim_r.save(PATH + '/video.gif', fps= 2* FPS)


