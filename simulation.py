'''
NOTE: 
	- è molto meglio se creo A e B prima
	- il plot rallenta di brutto  

'''


from syslog import LOG_LPR
import numpy as np
import matplotlib.pyplot as plt 
import scipy.constants as cost
import sys
import os 

from tkinter import *
from tkinter import ttk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

import time
import threading
from numba import jit

#sys.path.insert(0, '/home/falzo/Scrivania/Sol_eqS')

import DFT 
import time_evolution as __ 


try:
	PATH = str(sys.argv[1])
	print('check: ',PATH)
	sys.path.insert(0, PATH)

	from Par import *

except:
	print('There are not arguments: picking Default Par')
	from Parameters import  *
	PATH = os.getcwd()
	print(PATH)


print('Level = ', Level)



ENDED = False

if STOP_BOUNDARIES: 
	debug_norma = False
	debug_E = False

dx = (x[1] - x[0])
dp = p[1] - p[0]

ll = len(x)

if len(x) != len(p): 
	print('Error: len(x) != len(p)')
	exit()

from numba import jit

#//////////////
#/ Simulation /
#//////////////


#Calcolo delle matrici 
ff = DFT.f_matrix(x, p, h, ll, ll, np.ndarray((ll, ll), dtype=complex))
iff = DFT.if_matrix(x, p, h, ll, ll , np.ndarray((ll, ll), dtype=complex))



#stato iniziale
print('ψ >>> ', ψ_name)

if ψ_name == ψ_list[0]: Ψ = DFT.wave_pack(x, x0, p0, σ0, N, L, h)

elif ψ_name == ψ_list[1]: Ψ = DFT.arm_ES(x, x0, n, m, ω, h)

elif ψ_name == ψ_list[2]: Ψ, norm_factor = DFT.RL_wp(x, x0, x_V_centered, p0, σ0, h, m, α, N, norm_factor = 0)
elif ψ_name == ψ_list[3]: Ψ, coefficients = DFT.coherent(n, 0, x, x0, m, ω, h, np.zeros(N, dtype = complex))


elif  ψ_name == ψ_list[4]:
	
	o = np.load(PATH + '/custom.npz')

	Ψ	 = o['Ψ']

	if len(Ψ) != len(x):  
		print('Error len(Ψ) != len(x)')
		
		f = open('abort.log', 'w')
		f.write('Simulation aborted')
		f.close()

		exit()


if debug_norma: cont, norm = DFT.check_norma(DFT.norma(Ψ, np.zeros(ll, dtype = complex), ll , dx), cont, norm, STD_NORMA)

V = np.zeros(len(x), dtype = complex)

for i in range(len(V)):
	if   Potential == Potential_list[1]:  V[i] = __.step(x[i], step_x, V0) 
	elif Potential == Potential_list[2]:  V[i] = __.barrier(x[i], barrier_width, V0, x_V_centered)
	elif Potential == Potential_list[3]:  V[i] = __.buca(x[i], barrier_width, V0, x_V_centered)
	elif Potential == Potential_list[4]:  V[i] = __.tunnel(x[i], x_V_centered, V0)
	elif Potential == Potential_list[5]:  V[i] = __.arm_osc(x[i], x_V_centered, a)
	elif Potential == Potential_list[6]:  V[i] = __.RL(x[i], RF_l, x_V_centered, h, m, α)

dic = {}
if Potential == Potential_list[7]:
	print("import " + lib + "\ndef F(x): return " + V_function)
	exec("import " + lib + "\ndef F(x): return " + V_function, dic)
        
	for i in range(len(V)): V[i] = dic['F'](x[i])

if Potential == Potential_list[8]:
	Ψ_a = DFT.anti_sim(Ψ)
	Ψ = Ψ + Ψ_a
	V = np.zeros(len(x))
	debug_norma = False


E, T, U = __.H_mean(x, p, Ψ, dx, dp, m, V, h, P_MAX, N, L, ll, np.ndarray((ll, ll), dtype=complex), np.zeros(ll, dtype = complex), np.zeros(ll, dtype = complex))
print('E =', E)

if STOP_BOUNDARIES: 
	V[x < 0.1*L] =  -1j * ST_BD_COEFF
	V[x > 0.9*L] = 	-1j * ST_BD_COEFF

#print(V)

OUT, V_Max = __.V_out_of_range(V, V_MAX)
print('V_Max = ', V_Max)


expantion_coefficients = LEC[Level-1]
print(expantion_coefficients)

terms = np.ndarray((Level, 2, ll), dtype = complex)
for j in range(Level):

	terms[j][0] = __.exp_A_(V, dt, h, expantion_coefficients[j][0])
	terms[j][1] = __.exp_B_(DFT.shift_p(p, P_MAX), m, dt, h, expantion_coefficients[j][1])




fig = plt.figure()

ax = fig.add_subplot()

ax.plot(x, V, label = 'V', color='slategray', zorder = 0)
line, = ax.plot(x, abs(Ψ)**2, label = r'$|ψ|^2$', color='#007FFF', zorder = 2)
lineE, = ax.plot(x, np.full(len(x), E), '--', label = 'E', color='firebrick', zorder = 1) #ax.hlines(E , 0.1, L-0.1, colors = 'red', label = 'E', zorder = 0)

ax.fill_between(x, V, np.full(len(x), -V_MAX), facecolor = '#CCCCCC', zorder = 0)
ax.set_ylim(-0.2 * max(abs(Ψ)**2), 1.2*max(abs(Ψ)**2))

ax.set_xlabel("x")
ax.set_ylabel("E")
ax.legend()



np.savez(PATH +'/0', x = x, V = V, Ψ = Ψ )

t = 1
i = 1
M = np.empty(0)
x_M = np.empty(0)


def step():

	start_time = time.time()

	global ENDED
	global FRAME, t, i, Level, Ψ, x, p, dt, V, m, ff, iff, cont, dx, dp, norm, N, L, h, P_MAX, STD_NORMA, E, ll 

	#Ψ_t = Ψ
	while i < FRAME:


		Ψ_n = __.evolution(Ψ, x, p, ff, iff, N, L, h, P_MAX, ll, np.zeros(ll, dtype = complex), terms, expantion_coefficients, Level)
		
		#test
		#Ψ_n = __.best_evolution(Ψ, x, p, ff, iff, N, L, h, P_MAX, ll, np.zeros(ll, dtype = complex), terms[0][0], terms[0][1])
		#Ψ_n_t = __.evolution_t_dependent(Level, Ψ_t, x, p, dt, V, m, ff, iff, N, L, h, P_MAX, ll, np.zeros(ll, dtype = complex))
		#print(i, '>> ', max(abs(Ψ_n - Ψ_n_t)))
		
		np.savez( str(PATH) + '/' + str(i), Ψ = Ψ, V = V )
		Ψ = Ψ_n
		
		#Ψ_t = Ψ_n_t


		
		if debug_norma: cont, norm = DFT.check_norma(DFT.norma(Ψ, np.zeros(ll, dtype = complex), ll , dx), cont, norm, STD_NORMA)
		print(i, '/'+str(FRAME) , end = '\r')

		t+=dt
		i+=1
		
		if i % 20 == 0:	
			bar['value'] = (i / FRAME) * 100
			per.config(text = '%.2f' %  ((i / FRAME) * 100) + ' %')
			BAR.update_idletasks()
			
			
			if debug_E:
				E_tmp, T, U = __.H_mean(x, p, Ψ, dx, dp, m, V, h, P_MAX, N, L, ll, np.ndarray((ll, ll), dtype=complex), np.zeros(ll, dtype = complex), np.zeros(ll, dtype = complex))

				__.check_E(T, U)

				if abs(E_tmp - E) / abs(E) > 10**(-1):						#stima per errore assoluto
					print('Error E value: E =', E, 'E_tmp =', E_tmp  )

				E = E_tmp
				lineE.set_data(x, np.full(ll, E))
			
			
			line.set_data(x, abs(Ψ)**2)
			canvas.draw()


	print('TIME = ' , time.time() - start_time)

	ENDED = True
	stop.config(text = 'Exit')



def abort():

	global ENDED
	if ENDED: 
		w.destroy()
		exit()

	else:
		f = open( PATH + '/abort.log', 'w')
		f.write('Simulation aborted')
		f.close()

		w.destroy()
		exit()

def thread():
	t = threading.Thread(target = step)
	t.daemon = True
	t.start()	


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

#start = ttk.Button(BUT, text = 'Start', command = step)
start = ttk.Button(BUT, text = 'Start', command = thread)

start.grid(column = 0, row = 2, padx=1, pady = 5)

stop = ttk.Button(BUT, text = 'Abort', command = abort)
stop.grid(column = 1, row = 2, padx= 1, pady = 5)

w.protocol("WM_DELETE_WINDOW", abort)
w.mainloop()
