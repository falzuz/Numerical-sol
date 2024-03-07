from ast import arg
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from matplotlib.animation import FuncAnimation


from tkinter import *
from tkinter import ttk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

try:
	PATH = str(sys.argv[1])
	sys.path.insert(0, PATH)

	from Par import *

except:
	print('There are not arguments: picking Default Par')
	sys.path.insert(0, '/home/falzo/Scrivania/Sol_eqS')

	from Parameters import  *
	PATH = os.getcwd()

print('ANIMAZIONE')
print(PATH)




#FRAME = 1000
o = np.load(PATH + '/0.npz')

x_0	 = o['x'] 
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
    return line2,

# animation function.  This is called sequentially
def animate_w(i):
    x = x_0

    o = np.load(PATH + '/' + str(i)+'.npz')
    Ψ = o['Ψ']
    
    line2.set_data(x, abs(Ψ)**2)

    return line2,


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

if V_flag == True: anim.save(PATH + '/video.mp4', fps=FPS)



def abort():
	f = open('abort.log', 'w')
	f.write('Simulation aborted')
	f.close()

	exit()



w = Tk()
w.title('')

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

style = ttk.Style(w)
w.tk.call('source', 'azure.tcl')
style.theme_use('azure')

bar = ttk.Progressbar(BAR, orient = 'horizontal', length = 500, mode = 'determinate')
bar.grid(row = 1)

per = ttk.Label(BAR, text = '0 %')
per.grid(row = 0)

start = ttk.Button(BUT, text = 'Start')
start.grid(column = 0, row = 2, padx=1, pady = 5)

stop = ttk.Button(BUT, text = 'Abort', command = abort)
stop.grid(column = 1, row = 2, padx= 1, pady = 5)

w.protocol("WM_DELETE_WINDOW", abort)
w.mainloop()
