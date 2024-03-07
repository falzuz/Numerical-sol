from re import S
import numpy as np
import matplotlib.pyplot as plt 
import scipy.fftpack as ft
#import cmath
import scipy.constants as cost

import sys
import os

from time import time
from timeit import default_timer as timer


sys.path.insert(0, os.getcwd())
import DFT
from Parameters import *

s = time()

ll = len(x)


dx = x[1] - x[0]
dp = p[1] - p[0]

#dp_sym = p_sym[1] - p_sym[0]

#if dp !=  dp_sym: print('error dp', dp-dp_sym)


##############
# Matrice
##############

ff = DFT.f_matrix(x, p, h, ll, ll, np.ndarray((ll, ll), dtype=complex))


iff = DFT.if_matrix(x, p, h, ll, ll, np.ndarray((ll, ll), dtype=complex))

#print(ff, iff)
#exit()

ls = ['gaus', 'dec_exp', 'wave_pack', 'quad', 'tr', 'arm'] 

i = ls[0]

#print(i)
p_s = DFT.shift_p(p, P_MAX)

#print(p_s)


if i == 'wave_pack':
	y = DFT.wave_pack(x, x0, p0, σ0, N, L, h)
	fy_anal = DFT.F_w(p_s, x0, p0, σ0, h)

'''
if i == 'quad':	
	y = np.empty(0)
	fy_anal = np.empty(0)
	for k in range(len(x)):
		y = np.append(y, DFT.quad(x[k], x0, p0, w, h))
		fy_anal = np.append(fy_anal, DFT.quad_a(p_s[k], x0, p0, w, h))
''' 

if i == 'quad':	
	y = np.empty(0)
	fy_anal = np.empty(0)
	for k in range(len(x)):
		y = np.append(y, DFT.quad(x[k], x0, p0, w, h))
		fy_anal = np.append(fy_anal, DFT.quad_a(p_s[k], x0, p0, w, h))


if i == 'gaus':
	y = DFT.gaus(x, x0, σ0)
	fy_anal = DFT.gaus_a(p_s, x0, σ0, h)

if i == 'dec_exp':
	y = np.empty(0)
	for k in x:
		y = np.append(y, DFT.dec_exp(k, x0, p0, w, h))
	fy_anal = DFT.dec_exp_a(p_s, x0, p0, w, h)

if i == 'tr':
	y = np.empty(0)
	fy_anal = np.empty(0)
	for k in range(len(x)):
		y = np.append(y, DFT.tr(x[k], x0, p0, w, H, h))
		fy_anal = np.append(fy_anal, DFT.tr_a(p_s[k], x0, p0, w, H, h))

if i == 'arm':
	y = DFT.arm_ES(x, x0, n, m, ω, h)
	fy_anal = DFT.arm_ES_a(p_s, x0, 0, n, m, ω, h)

print(DFT.norma(y, np.zeros(ll, dtype = complex), ll, dx), 'start')
print(DFT.norma(fy_anal,np.zeros(ll, dtype = complex), ll, dp), 'anal')


fy = DFT.Dft(y, x, p, ff, N, L, h, ll, np.zeros(ll, dtype = complex))
fy_s = DFT.shift(fy, p, P_MAX)


'''
fontsize = 20

fig = plt.figure()
fig.set_figheight(8)
fig.set_figwidth(16)
 
ax1 = plt.subplot2grid(shape=(3, 3), loc=(0, 0), colspan=1)
ax2 = plt.subplot2grid(shape=(3, 3), loc=(0, 1), colspan=1)
ax3 = plt.subplot2grid(shape=(3, 3), loc=(1, 0), colspan=2)

 
# plotting subplots
ax1.plot(x, y, color = '#007FFF',  linewidth=2, label = 'Stato iniziale')
ax1.set_title('Stato Iniziale', fontsize = fontsize)
ax1.set_xlabel('x', fontsize = fontsize -2 , labelpad=10)
# ax1.legend(fontsize = fontsize)

ax2.plot(p, abs(fy),  color = '#007FFF',  linewidth=2, label = 'Numerica')
ax2.set_title('Trasformata numerica', fontsize = fontsize)
ax2.axvline((p[-1] - p[0]) /2 , color = 'gray', zorder = 2, linewidth = 2)
ax2.text((p[-1] - p[0]) / 2 - 40, 0.03, r'$ + \infty $', fontsize = fontsize )
ax2.text((p[-1] - p[0]) / 2 + 5, 0.03, r'$ - \infty $', fontsize = fontsize )
ax2.set_xlabel('p', fontsize = fontsize -2 , labelpad=10)


#ax2.legend(fontsize = fontsize)

ax3.plot(p_s, abs(fy_s),  color = '#007FFF',  linewidth=2, label = 'Numerica traslata')
ax3.plot(p_s, abs(fy_anal), color = 'green',  linewidth=2, label = 'Analitica')
ax3.set_xlabel('p', fontsize = fontsize -2 , labelpad=10)

ax3.legend(fontsize = fontsize)

ax1.tick_params(axis='both',  labelsize=fontsize)
ax2.tick_params(axis='both',  labelsize=fontsize)
ax3.tick_params(axis='both',  labelsize=fontsize)





fig.tight_layout()
plt.savefig("fourier.png", bbox_inches='tight')
 
# display plot
plt.show()

exit()
'''

'''
plt.figure()
#plt.plot(x, abs(y), label = 'abs')
plt.plot(p, abs(fy), label = 'numerical')
plt.plot(p_s, abs(fy_s), label = 'numerical shift')
plt.plot(p_s, abs(fy_anal), label = 'anal')
plt.title('Absolute')
plt.legend()
plt.show()
'''


print(DFT.norma(fy, np.zeros(ll, dtype = complex), ll, dp), 'fourier')



########
# SAVE #
########


np.savez(i, x=x, p=p, p_s=p_s, y=y, fy_anal=fy_anal, fy=fy, fy_s = fy_s)

#print(x, p, p_s, y, fy_anal, fy, fy_s)


#........................................................

p_b = DFT.back_shift_p(p_s, P_MAX)

'''
for numb in range(len(p_b)):
	if p_b[numb] != p[numb] : 
		print('error back')
		exit() 
'''
#........................................................

fy_anal_s = DFT.back_shift(fy_anal, p_s)
i_fy_s = DFT.back_shift(fy_s, p_s)

plt.figure()
plt.plot(p_b, abs(fy_anal_s), label = 'a_back', zorder = 4)
plt.plot(p_b, abs(i_fy_s), label = 'back', zorder = 4)
plt.title('test')

plt.plot(p, abs(fy), label = 'numerical')
plt.plot(p_s, abs(fy_s), label = 'numerical shift')
plt.plot(p_s, abs(fy_anal), label = 'anal')
plt.legend()

plt.show()

#print(iff)

#print(fy, x, p_b, iff, L, h, ll)

i_y = DFT.iDft(fy, x, p_b, iff, L, h, ll, np.zeros(ll, dtype = complex))

#print(fy, x, p_b, iff, L, h, ll, i_y )


i_y_s = DFT.iDft(i_fy_s, x, p_b, iff, L, h, ll, np.zeros(ll, dtype = complex))

i_anal = DFT.iDft(fy_anal_s, x, p_b, iff, L, h, ll, np.zeros(ll, dtype = complex))

plt.figure()
plt.plot(x, abs(y), label = 'start')
plt.plot(x, abs(i_y), label= 'num inverso')
plt.plot(x, abs(i_anal), label = 'trasf anal inversa')
plt.plot(x, abs(i_y_s), label = 'num shift inversa')
plt.title('abs Inverse')
plt.legend()
plt.show()


print(DFT.norma(i_y, np.zeros(ll, dtype = complex), ll, dx), 'inverso')
print(DFT.norma(i_anal, np.zeros(ll, dtype = complex), ll, dx), 'inverso anal')


o = i + '_inverso'

np.savez(o, x=x, p=p, p_s=p_b, y=y, fy_anal=fy_anal, fy=fy, 
			i_y=i_y, i_anal=i_anal, i_y_s = i_y_s )

#break


print(time()-s)

