from ast import arg
import os
import sys
from urllib.parse import DefragResult
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from matplotlib.animation import FuncAnimation

from numba import jit

sys.path.insert(0, os.getcwd())

import DFT 
import time_evolution as __ 

try:
    PATH = str(sys.argv[1])
    sys.path.insert(0, PATH)

    from Par import *
    import DFT

except:
    print('There are not arguments: picking Default Par')
    sys.path.insert(0, os.getcwd())

    from Parameters import  *
    import DFT
    PATH = os.getcwd()

print('ANIMAZIONE')
print(PATH)

'''
def full_Dft(y):
	global x, p, h, P_MAX, N, L

	ff = DFT.f_matrix(x, p, h)
	p_s = DFT.shift_p(p, P_MAX)

	fy = DFT.Dft(y, x, p, ff, N, L, h)
	fy_s = DFT.shift(fy, p, P_MAX)

	return p_s, fy_s
'''

#@jit
def save_F(x, p, h, P_MAX, N, L, FRAME, i):

    while i < FRAME:

        o = np.load(PATH + '/' + str(i)+'.npz')
        Ψ = o['Ψ']
        V = o['V']
        

        P, F = DFT.full_Dft(Ψ, x, p, h, P_MAX, N, L, N, np.zeros((N,N) , dtype=complex), np.zeros(N, dtype=complex))

        np.savez(PATH + '/' + str(i) , Ψ = Ψ, F = F, V = V)

        i+=1
    
    return 1

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




o = np.load(PATH + '/0.npz')

x_0	 = o['x'] 
V	 = o['V'] 
Ψ	 = o['Ψ'] 

P, F = DFT.full_Dft(Ψ, x, p, h, P_MAX, N, L, N, np.zeros((N,N) , dtype=complex), np.zeros(N, dtype=complex))

np.savez(PATH +'/0', x = x, V = V, Ψ = Ψ, P = P, F = F)

save = save_F(x, p, h, P_MAX, N, L, FRAME, 1)

E, T, U = __.H_mean(x, p, Ψ, x[1]-x[0], p[1]-p[0], m, V, h, P_MAX, N, L, N, np.zeros((N,N) , dtype=complex), np.zeros(N, dtype=complex), np.zeros(N, dtype=complex))
print(E)

fontsize = 24
fig, ax = plt.subplots(2, figsize=(16, 8))


#line1, = ax[0].plot(x, V, label = 'Potenziale V', color='slategray', zorder = 1)
line2, = ax[0].plot(x, abs(Ψ)**2, label = r'$|ψ|^2$', color='#007FFF', zorder = 3)
line3, = ax[0].plot(x, np.full(len(x), E), '--', label = r'$E_0$', color='firebrick', zorder = 2) #ax[0].hlines(E , 0.1, L-0.1, colors = 'red', label = 'E', zorder = 0)

#line4 = ax[0].fill_between(x, V, np.full(len(x), -V_MAX), facecolor = '#CCCCCC', zorder = 1)
ax[0].set_ylim(-0.1 * max(abs(Ψ)**2), 1.1*max(abs(Ψ)**2))

ax[0].set_xlabel("x", fontsize = fontsize, labelpad = 10)
ax[0].set_ylabel("Energia", fontsize = fontsize, labelpad = 17)


#if real_flag == True: line5, = ax[0].plot(x, Ψ.real, label = r'$Re(ψ)$', color='palegreen', zorder = 0)
#if imag_flag == True: line6, = ax[0].plot(x, Ψ.imag, '--', label = r'$Im(ψ)$', color='palevioletred', zorder = 0)

line5 = plt.Line2D(x, Ψ.real, label = r'$Re(ψ)$', color='palegreen', zorder = 0)
line6 = plt.Line2D(x, Ψ.imag, linestyle='--', label = r'$Im(ψ)$', color='palevioletred', zorder = 0)

if real_flag == 1:
    ax[0].add_line(line5)


if imag_flag == 1:
    ax[0].add_line(line6)

ax[0].legend(fontsize = fontsize)


line_p, = ax[1].plot(P, abs(F)**2, label = r'$|\mathcal{F} \, (ψ)|^2$', color='#007FFF', zorder = 3)
line_Ep, = ax[1].plot(P, np.full(len(P), E), '--', label = r'$E_0$', color='firebrick', zorder = 2) #ax[0].hlines(E , 0.1, L-0.1, colors = 'red', label = 'E', zorder = 0)


ax[1].set_ylim(-0.1 * max(abs(F)**2), 1.1*max(abs(F)**2))

ax[1].set_xlabel("p", fontsize = fontsize)
ax[1].set_ylabel("Energia", fontsize = fontsize, labelpad = 17)


#if real_flag == True: line_pr, = ax[1].plot(x, Ψ.real, label = r'$Re(ψ)$', color='palegreen', zorder = 0)
#if imag_flag == True: line_pi, = ax[1].plot(x, Ψ.imag, '--', label = r'$Im(ψ)$', color='palevioletred', zorder = 0)

line_pr = plt.Line2D(P, F.real, label = r'$Re(ψ)$', color='palegreen', zorder = 0)
line_pi = plt.Line2D(P, F.imag, linestyle='--', label = r'$Im(ψ)$', color='palevioletred', zorder = 0)

if real_flag == 1:
    ax[1].add_line(line_pr)


if imag_flag == 1:
    ax[1].add_line(line_pi)

ax[1].legend(fontsize = fontsize)

#ax[0].tick_params(axis = 'both', fontsize = fontsize)
#ax[1].tick_params(axis = 'both', fontsize = fontsize)

ax[0].tick_params(axis='x',  labelsize=fontsize)
ax[0].tick_params(axis='y',  labelsize=fontsize)
ax[1].tick_params(axis='x',  labelsize=fontsize)
ax[1].tick_params(axis='y',  labelsize=fontsize)

ax[0].set_xticks(np.linspace(0, 10, 11))
ax[0].set_xticklabels(['0', '', '', '', '','', '', '', '', '', 'L'])

ax[1].set_xticks(np.linspace(-P_MAX, P_MAX, 11))
ax[1].set_xticklabels([ r'$-\pi \hbar \, N / L$', '', '', '', '', '0', '', '', '', '', r'$\pi \hbar \, N / L$'])

ax[0].text(x0+ 1.2 * σ0, 0.5, r'$x_0 =$' + '%.2f' % (x0 / L) + r'$\, L$', fontsize = fontsize)
ax[1].text(p0 - 70, 0.05, r'$p_0 =$' +  '%.2f' % (p0 / P_MAX )+ r'$\, \pi \hbar \, N / L$', fontsize = fontsize)

ax[0].axvline(__.mean_x(x, Ψ, x[1]-x[0], N, np.zeros(N, dtype = complex), mean_x = 0), linestyle = '--', color='#007FFF', zorder = 3)
ax[1].axvline(__.mean_x(P, F, P[1]-P[0], N, np.zeros(N, dtype = complex), mean_x = 0), linestyle = '--', color='#007FFF', zorder = 3)

#ax[0].grid(zorder = 0)
#ax[1].grid(zorder = 0)


fig.tight_layout()
plt.savefig("free_p_view.png", bbox_inches='tight')
plt.show()
exit()


# initialization function: plot the background of each frame
def init():

    line2.set_data([], [])

    line5.set_data([], [])
    line6.set_data([], [])


    line_p.set_data([], [])

    line_pr.set_data([], [])
    line_pi.set_data([], [])

    return line_p, line_pr, line_pi, line2, line5, line6

# animation function.  This is called sequentially
def animate(i):
    

    o = np.load(PATH + '/' + str(i)+'.npz')
    Ψ = o['Ψ']
    F = o['F']

    
    line_p.set_data(P, abs(F)**2)

    line_pr.set_data(P, F.real)
    line_pi.set_data(P, F.imag)

    line2.set_data(x, abs(Ψ)**2)

    line5.set_data(x, Ψ.real)
    line6.set_data(x, Ψ.imag)


    return line_p, line_pr, line_pi, line2, line5, line6

# initialization function: plot the background of each frame
def init_r():
    line_p.set_data([], [])

    line_pr.set_data([], [])

    line2.set_data([], [])

    line5.set_data([], [])

    return line_p, line_pr, line2, line5

# animation function.  This is called sequentially
def animate_r(i):
    

    o = np.load(PATH + '/' + str(i)+'.npz')
    Ψ = o['Ψ']
    F = o['F']

    
    line_p.set_data(P, abs(F)**2)

    line_pr.set_data(P, F.real)

    line2.set_data(x, abs(Ψ)**2)

    line5.set_data(x, Ψ.real)


    return line_p, line_pr, line2, line5

# initialization function: plot the background of each frame
def init_i():
    line_p.set_data([], [])

    line_pi.set_data([], [])

    line2.set_data([], [])

    line6.set_data([], [])

    
    return line_p, line_pi, line2, line6

# animation function.  This is called sequentially
def animate_i(i):
    

    o = np.load(PATH + '/' + str(i)+'.npz')
    Ψ = o['Ψ']
    F = o['F']
    
    line_p.set_data(P, abs(F)**2)

    line_pi.set_data(P, F.imag)

    line2.set_data(x, abs(Ψ)**2)

    line6.set_data(x, Ψ.imag)

    return line_p, line_pi, line2, line6

# initialization function: plot the background of each frame
def init_w():
    line_p.set_data([], [])
    line2.set_data([], [])

    return line_p, line2,

# animation function.  This is called sequentially
def animate_w(i):
    

    o = np.load(PATH + '/' + str(i)+'.npz')
    Ψ = o['Ψ']
    F = o['F']

    
    line_p.set_data(P, abs(F)**2)

    line2.set_data(x, abs(Ψ)**2)

    return line_p, line2





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
old_fig_0= zoom_factory(ax[0],base_scale = scale)
#old_fig_1= zoom_factory(ax[1],base_scale = scale)

plt.show()

if V_flag == True: anim.save(PATH + '/video.mp4', fps=FPS)