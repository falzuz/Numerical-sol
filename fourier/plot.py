import numpy as np
import matplotlib.pyplot as plt 
import scipy.fftpack as ft
#import cmath
import scipy.constants as cost

from timeit import default_timer as timer

quale = 2

ls = ['wave_pack', 'quad', 'gaus', 'dec_exp', 'tr', 'arm']

def ratio(x,y,z):
	for i in range(0, len(x)):
			if y[i] == 0 and  x[i] == 0:
				print('0/0: ', i)
				z = np.append(z, 20) 

			if y[i] != 0:
				r = x[i] / y[i]
				if r < 100:
					z = np.append(z, r)
				else: 
					print('Exploding: ', r, i)
					z = np.append(z, 10)


			else:
				print('zero: ', i)
				z = np.append(z, 0)

	return z

def diff(x, y, z, t):
	#y = np.nan_to_num(y, nan=0.000001)

	t = t[np.logical_not(np.isnan(y))]
	x = x[np.logical_not(np.isnan(y))]
	y = y[np.logical_not(np.isnan(y))]

	M = np.max(y)
	m = np.min(y)

	delta = (M - m)
	print(delta)


	if delta == 0:
		print('Error delta = 0')

	for i in range(0, len(x)):
		z = np.append(z, abs(x[i] - y[i]))
		#z = np.append(z, abs(x[i] - y[i]) / y[i])

	return z, t


o = np.load(ls[quale]+ ".npz")

x	 = o['x'] 
p 	 = o['p']
p_s = o['p_s']
y   = o['y'] 
fy_anal= o['fy_anal']
fy= o['fy']
fy_s = o['fy_s']


#Stato iniziale....................................

plt.figure()
plt.plot(x, y.real,'.', label = 'real')
plt.plot(x, y.imag, label = 'imag')
plt.title('Starting state')
plt.legend()
plt.show()


plt.figure()
plt.plot(x, abs(y), label = 'abs')
plt.title('Starting state')
plt.legend()
plt.show()




#PLOT ..............................................


plt.figure()
#plt.plot(x, abs(y), label = 'abs')
plt.plot(p, fy, label = 'numerical')
plt.plot(p_s, fy_s, label = 'numerical shift')
plt.plot(p_s, fy_anal, label = 'anal')
plt.title('Absolute')
plt.legend()
plt.show()

plt.figure()
#plt.plot(x, abs(y), label = 'abs')
plt.plot(p, abs(fy), label = 'numerical')
plt.plot(p_s, abs(fy_s), label = 'numerical shift')
plt.plot(p_s, abs(fy_anal), label = 'anal')
plt.title('Absolute')
plt.legend()
plt.show()

rat = np.empty(0)
plt.figure()
plt.plot(p_s, ratio(abs(fy_s), abs(fy_anal), rat), label = 'ratio')
plt.title('Ratio')
plt.legend()
plt.show()

dd = np.empty(0)
dd, p_d = diff(abs(fy_s), abs(fy_anal), dd, p_s)
#print(len(dd))
#print(dd)

plt.figure()
plt.plot(p_d, dd , label = 'diff')
plt.title('differenza percentuale in modulo')
plt.legend()
plt.show()

#PLOT BACK_SHIFT.........................................

'''

	plt.figure()
	plt.plot(p, fy.real, label = 'numerical')
	plt.plot(p_s, fy.real, label = 'numerical shift')
	plt.plot(p_s, fy_anal.real, label = 'anal')
	plt.legend()
	plt.show()

	rat = np.empty(0)
	plt.figure()
	plt.plot(p_s, ratio(fy.real, fy_anal.real, rat), label = 'ratio')
	plt.legend()
	plt.show()



	plt.figure()
	plt.plot(p, fy.imag, label = 'numerical')
	plt.plot(p_s, fy.imag, label = 'numerical shift')
	plt.plot(p_s, fy_anal.imag, label = 'anal')
	plt.legend()
	plt.show()


	rat = np.empty(0)
	plt.figure()
	plt.plot(p_s, ratio(fy.imag, fy_anal.imag, rat), label = 'ratio')
	plt.legend()
	plt.show()

	plt.figure()
	plt.plot(p, abs(fy), label = 'numerical')
	plt.plot(p_s, abs(fy), label = 'numerical shift')
	plt.plot(p_s, abs(fy_anal), label = 'anal')
	plt.legend()
	plt.show()

	rat = np.empty(0)
	plt.figure()
	plt.plot(p_s, ratio(abs(fy), abs(fy_anal), rat), label = 'ratio')
	plt.legend()
	plt.show()

	'''

#PLOT INVERSE ...............................................

ls_i = ls[quale] + '_inverso.npz'

oi = np.load(ls_i)

x	 = oi['x'] 
p 	 = oi['p']
p_s  = oi['p_s']
y    = oi['y'] 
fy_anal= oi['fy_anal']
fy   = oi['fy']
i_y  = oi['i_y'] 
i_anal= oi['i_anal']
i_y_s = oi['i_y_s']

'''
plt.figure()
plt.plot(x, y.real, label = 'start')
plt.plot(x, i_y.real, label = 'num inverso')
plt.plot(x, i_anal.real, label = 'trasf anal inversa')
plt.title('Real Inverse')
plt.legend()
plt.show()

rat = np.empty(0)
plt.figure()
plt.plot(p_s, ratio(y.real, i_anal.real, rat), label = 'ratio')
plt.title('Ratio')
plt.legend()
plt.show()


plt.figure()
plt.plot(x, y.imag, label = 'start')
plt.plot(x, i_y.imag, label= 'num inverso')
plt.plot(x, i_anal.imag, label = 'trasf anal inversa')
plt.title('Imaginary inverse')
plt.legend()
plt.show()


rat = np.empty(0)
plt.figure()
plt.plot(p_s, ratio(y.imag, i_anal.imag, rat), label = 'ratio')
plt.title('ratio')
plt.legend()
plt.show()

'''

plt.figure()
plt.plot(x, abs(y), label = 'start')
plt.plot(x, abs(i_y), label= 'num inverso')
plt.plot(x, abs(i_anal), label = 'trasf anal inversa')
plt.plot(x, abs(i_y_s), label = 'num shift inversa')
plt.title('abs Inverse')
plt.legend()
plt.show()


rat = np.empty(0)
plt.figure()
plt.plot(x, ratio(abs(i_anal), abs(y), rat), label = 'ratio')
plt.title('Ratio i_anal / start')
plt.legend()
plt.show()

dd = np.empty(0)
dd, x_d = diff(abs(i_anal),abs(y), dd, x)
#print(len(dd))
#print(dd)

plt.figure()
plt.plot(x_d, dd , label = 'diff')
plt.title('differenza percentuale in modulo')
plt.legend()
plt.show()

rat = np.empty(0)
plt.figure()
plt.plot(x, ratio(abs(i_y),abs(y), rat), label = 'ratio')
plt.title('Ratio i_y / start')
plt.legend()
plt.show()

dd = np.empty(0)
dd, x_d = diff(abs(i_y), abs(y), dd, x)
#print(len(dd))
#print(dd)

plt.figure()
plt.plot(x_d, dd , label = 'diff')
plt.title('differenza percentuale in modulo')
plt.legend()
plt.show()
