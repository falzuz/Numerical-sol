#time evolution

import numpy as np
import DFT

from numba import jit

def V_out_of_range(V, V_MAX):

	Max = np.max(V)
	Min = np.min(V)
	if Max >= (V_MAX) or Min <= -(V_MAX):
		print('V_out_of_range' + 'abs(V_MAX) = ' , V_MAX, 'now = ', max(Max, -Min) )
		f = open('abort.log', 'w')
		f.write('Simulation aborted')
		f.close()

		#exit()
		return True, max(Max, -Min)
	else:
		return False, max(Max, -Min)



@jit(nopython = True)
def exp_A_(V, dt, h, Coefficient):
	A = Coefficient * -1j*V * dt / h
	return np.exp(A)

@jit(nopython = True)
def exp_B_(k, m, dt, h, Coefficient):
	B = Coefficient * -1j*k**2 * dt / (2*m*h)
	return np.exp(B)

#@jit(forceobj = True)
def evolution(y, q, k, mat_f, mat_i, N, L, h, P_MAX, ll, f_zero, terms, coefficient, Level):
	
	for i in range(Level):
		#A
		if coefficient[i][0] != 0:
			y = terms[i][0]*y

		if coefficient[i][1] != 0:
			f_y = DFT.Dft(y ,q, k, mat_f, N, L, h,  ll, f_zero)

			#B
			f_y = DFT.shift(f_y, k, P_MAX)
			k = DFT.shift_p(k, P_MAX)
		
			f_y = terms[i][1]*f_y

			f_y = DFT.back_shift(f_y, k)
			k = DFT.back_shift_p(k, P_MAX)


			#anti
			y= DFT.iDft(f_y, q, k, mat_i, L, h, ll, f_zero)
	
	'''
	for i in terms:
		#A
		y = i[0]*y

		f_y = DFT.Dft(y ,q, k, mat_f, N, L, h,  ll, f_zero)

		#B
		f_y = DFT.shift(f_y, k, P_MAX)
		k = DFT.shift_p(k, P_MAX)
	
		f_y = i[1]*f_y

		f_y = DFT.back_shift(f_y, k)
		k = DFT.back_shift_p(k, P_MAX)


		#anti
		y= DFT.iDft(f_y, q, k, mat_i, L, h, ll, f_zero)
	'''

	return y

@jit(nopython = True)
def best_evolution(y, q, k, mat_f, mat_i, N, L, h, P_MAX, ll, f_zero, A, B):
	
	#A
	y = A * y

	#fourier
	f = DFT.Dft(y,q, k, mat_f, N, L, h, ll, f_zero)

	#B
	f = DFT.shift(f, k, P_MAX)
	k = DFT.shift_p(k, P_MAX)
	
	f = B * f

	f = DFT.back_shift(f, k)
	k = DFT.back_shift_p(k, P_MAX)

	#anti
	y= DFT.iDft(f, q, k, mat_i, L, h, ll, f_zero)


	#A
	y = A* y


	return y




@jit(nopython = True)
def exp_A(y, V, dt, h, Coefficient):
	A = Coefficient * -1j*V * dt / h
	return np.exp(A) * y 

@jit(nopython = True)
def exp_B(fy, k, m, dt, h, Coefficient):
	B = Coefficient * -1j*k**2 * dt / (2*m*h)

	return np.exp(B) * fy

#Generalized version
'''
@jit
def evolution_t_dependent(y, q, k, dt, V, m, mat_f, mat_i, N, L, h, P_MAX, ll, f_zero, coefficient, Level):

	for i in range(Level):

		#A
		if coefficient[i][0] != 0:
			y1 = exp_A(y, V, dt, h, coefficient[i][0])

		if coefficient[i][1] != 0:

			#fourier
			f_y1 = DFT.Dft(y1 ,q, k, mat_f, N, L, h,  ll, f_zero)

			#B
			f_y1 = DFT.shift(f_y1, k, P_MAX)
			k = DFT.shift_p(k, P_MAX)
			
			f_y2 = exp_B(f_y1, k, m, dt, h, coefficient[i][1])


			f_y2 = DFT.back_shift(f_y2, k)
			k = DFT.back_shift_p(k, P_MAX)


			#anti
			y2= DFT.iDft(f_y2, q, k, mat_i, L, h, ll, f_zero)

	return y2
	
'''

@jit
def evolution_t_dependent(Level, y, q, k, dt, V, m, mat_f, mat_i, N, L, h, P_MAX, ll, f_zero):

	if Level==1:	#  c1 = 1      c2 = 1
		
		#A
		y1 = exp_A(y, V, dt, h, 1)

		#fourier
		f_y1 = DFT.Dft(y1 ,q, k, mat_f, N, L, h,  ll, f_zero)

		#B
		f_y1 = DFT.shift(f_y1, k, P_MAX)
		k = DFT.shift_p(k, P_MAX)
		
		f_y2 = exp_B(f_y1, k, m, dt, h, 1.)

		#print('B_t = ', B)

		f_y2 = DFT.back_shift(f_y2, k)
		k = DFT.back_shift_p(k, P_MAX)


		#anti
		y2= DFT.iDft(f_y2, q, k, mat_i, L, h, ll, f_zero)

		return y2
	
	if Level==2:    # c1 = 1/2 d1 = 1 c2 = 1/2 d2 = 0
		#A
		y1 = exp_A(y, V, dt, h, 1./2)

		#fourier
		f_y1 = DFT.Dft(y1,q, k, mat_f, N, L, h, ll, f_zero)

		#B
		f_y1 = DFT.shift(f_y1, k, P_MAX)
		k = DFT.shift_p(k, P_MAX)
		
		f_y2 = exp_B(f_y1, k, m, dt, h, 1)

		f_y2 = DFT.back_shift(f_y2, k)
		k = DFT.back_shift_p(k, P_MAX)

		#anti
		y2= DFT.iDft(f_y2, q, k, mat_i, L, h, ll, f_zero)


		#A
		y3 = exp_A(y2, V, dt, h, 1./2)


		return y3

	if Level==3:   # c1 = 1/3 c2 = 1/3 c3 = 1/3 d1 = 1/2  d2 = 1/2  d3 = 0
					# a = 1  b = -1/24  c = -2/3  d = 3/4  e = 2/3  f = 7/24

		#A
		y1 = exp_A(y, V, dt, h, 1.)

		#fourier
		f_y1 = DFT.Dft(y1,q, k, mat_f, N, L, h, ll, f_zero)

		#B
		f_y1 = DFT.shift(f_y1, k, P_MAX)
		k = DFT.shift_p(k, P_MAX)
		
		f_y2 = exp_B(f_y1, k, m, dt, h, -1./24)

		f_y2 = DFT.back_shift(f_y2, k)
		k = DFT.back_shift_p(k, P_MAX)

		#anti
		y2= DFT.iDft(f_y2, q, k, mat_i, L, h, ll, f_zero)


		#A
		y3 = exp_A(y2, V, dt, h, -2./3)

		#fourier
		f_y3 = DFT.Dft(y3, q, k, mat_f, N, L, h, ll, f_zero)

		#B
		f_y3 = DFT.shift(f_y3, k, P_MAX)
		k = DFT.shift_p(k, P_MAX)
		
		f_y4 = exp_B(f_y3, k, m, dt, h, 3./4)

		f_y4 = DFT.back_shift(f_y4, k)
		k = DFT.back_shift_p(k, P_MAX)

		#anti
		y4= DFT.iDft(f_y4, q, k, mat_i, L, h, ll, f_zero)

		#A
		y5 = exp_A(y4, V, dt, h, 2./3)

		#fourier
		f_y5 = DFT.Dft(y5, q, k, mat_f, N, L, h, ll, f_zero)

		#B
		f_y5 = DFT.shift(f_y5, k, P_MAX)
		k = DFT.shift_p(k, P_MAX)
		
		f_y6 = exp_B(f_y5, k, m, dt, h, 7./24)

		f_y6 = DFT.back_shift(f_y6, k)
		k = DFT.back_shift_p(k, P_MAX)

		#anti
		y6= DFT.iDft(f_y6, q, k, mat_i, L, h, ll, f_zero)
	
		return y6

#Valori medi
#@jit # (forceobj = True)
def mean_x(x, y, dt, N, y_c, mean_x = 0):     #compute the mean_x value
    
    for i in range(N):
        y_c[i] = DFT.cong(y[i])
    
    for i in range(N):
        mean_x += (y_c[i] * x[i] * y[i]) * dt
    return mean_x

def mean_p(p, f, dp, P_MAX):     #compute the mean_x value
    
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

#@jit
def H_mean(x, p, y, dx, dp, m, V, h, P_MAX, N, L, n, mat, f, y_c):
    
    #for i in range(N):
    #    y_c[i] = DFT.cong(y[i])

    ps, fy = DFT.full_Dft(y, x, p, h, P_MAX, N, L, n, mat, f)

    T = (1/2/m) * mean_x(ps**2, fy, dp, N, y_c)
    U = mean_x(V, y, dx, N, y_c)

    return T+U, T, U

def check_E(T, V):
	if abs(T.imag) > abs(T.real)*10**(-5) or abs(V.imag) > abs(V.real)*10**(-5):

		print('Error on E value: \n')
		print('T = ', T, 'V =', V)

	


''' Potenziali '''

#step
def step (s, x0, V0):
	if s < x0:
		return 0
	else:
		return V0


#Barriera finita
def barrier(s, a, V0, x0):
	if s <= x0:
		return 0
	if s >= a + x0:
		return 0
	else:
		return V0

#Buca finita
def buca(s, a, V0, x0):
	if s <= x0:
		return 0
	if s >= a + x0:
		return 0
	else:
		return -V0

#Oscillatore armonico
def arm_osc (s, x0, a):

	return a * (s - x0)**2

#Tipico effetto tunnel
def tunnel (s, x0, V0):
	if s < x0:
		return 0
	if s == x0:
		return V0
	if s > x0:
		k = x0 - 1/ V0
		return (1 / (s - k))

#Reflectionless
def RL (s, l, x0, h, m, α): #, V0):
	return - (h**2/(2*m)) * (α**2) * l*(l+1) / ((np.cosh((s - x0) * α ))**2)
#	return - l*(l+1) / (np.cosh((s - x0))**2)

###############
# Salvataggio #
###############

#Salva in file out intestando ogni append con t = &tempo 
def save(x, y, f, out):

	file = open(out, 'a')
	file.writelines('t = ' + str(f) + '\n')
	for i in range(len(x)):
		file.writelines(str(x[i]) + '\t' + str(y[i]) + '\n' ) 
	file.close()