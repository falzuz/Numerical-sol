# DFT

import numpy as np
import scipy.constants as cost
from scipy import special  # per polinomi di Hermite
import math

from numba import jit

@jit(nopython = True)
def f_matrix(x, p, h, n, n1, mat):
	for k in range(n):
		for i in range(n):
			mat[k][i] = np.exp(-1j * p[k] * x[i] / h)
	return mat

@jit(nopython = True)
def if_matrix(x, p, h, n, n1, mat):

	for i in range(n):
		for k in range(n):
			mat[i][k] = np.exp(+1j * x[i] * p[k] / h)
	return mat

@jit(nopython = True)
def Dft(y, x, p, mat, N, L, h, len_p, f):

	c = L / (np.sqrt(2 * np.pi * h) * N)

	
	for k in range(len_p):
		s = 0
		for i in range(len(x)):
			s += mat[k][i] * y[i]

		if np.isnan(s):
			print('Error nan')

		f[k] = c*s
	return f

@jit
def iDft(f, x, p, mat, L, h, len_x, y):

	ic = np.sqrt(2 * np.pi * h) / L


	for i in range(len_x): 
		s = 0
		for k in range(len(p)):
			if np.isnan(f[k]):
				continue

			s += mat[i][k] * f[k]


		if np.isnan(s):
			print('Error nan')


		y[i] = ic*s

	return y

'''

def c (L,N,h):
	c = float(L / (np.sqrt(2 * np.pi * h) * N))


def ic(L,h):
	ic = float(np.sqrt(2 * np.pi * h) / L)

'''

#@jit
def full_Dft(y, x, p, h, P_MAX, N, L, n, mat, f):

	ff = f_matrix(x, p, h, n, n, mat)
	p_s = shift_p(p, P_MAX)

	fy = Dft(y, x, p, ff, N, L, h, len(p), f)
	fy_s = shift(fy, p, P_MAX)

	return p_s, fy_s



###################
# For Check point #
###################

@jit
def cong(z):
	return np.conjugate(z)

@jit
def mod(z, z_c):
	return z*z_c


@jit
def norma(z, z_c, len, dt, norma = 0):
	
	z_c = cong(z)
	z = mod(z, z_c)

	for i in range(len):
		norma += z[i]* dt
	
	return norma.real

def anti_sim(y):
	y_a = np.empty(0)
	for i in range(0,len(y)):
		y_a = np.append(y_a, -y[-i])
	return y_a

#@jit
def check_norma(n, c, norma, STD_NORMA):
	if c == 0:
		norma = n

	if abs(STD_NORMA - norma) > 0.1:
		print('STD_Norma check failed: = ', n, c)
		exit()


	if n < norma-0.1 or n > norma+0.1:
		print('Norma check failed: = ', n, c)
		exit()
	
	c += 1	
	return c, norma

@jit
def mean(x, y, dt):				# compute the mean value
									#for operator x that is a multiplication
    
    y_c = np.empty(0)
    for i in y:
        y_c = np.append (y_c, cong(i))
    
    mean = 0
    for i in range(len(x)):
        mean += (y_c[i] * x[i] * y[i]) * dt
    return mean

'''

def shift(V):
	v = np.zeros(len(V))
	for i in range(len(V)):
		v[i] = V[i]

	i = 1
	while v[-i] > np.pi * h* N / L:
		v[-i] = v[-i] - 2* np.pi * h* N / L
		i+=1

	return v

def back_shift(V):
	v = np.zeros(len(V))
	for i in range(len(V)):
		v[i] = V[i]

	i = 1
	while v[-i] < 0:
		v[-i] = v[-i] + 2*np.pi*h*N / L
		i +=1 

	return v

'''

@jit
def shift(V, mom, MAX):
	v1 = V[mom < MAX]
	v2 = V[mom >= MAX]

	#print(v1, v2)

	v = np.append(v2, v1)

	if len(v) != len(V): print('Error')

	return v

@jit
def shift_p(V, MAX):

	
	v1 = V[V < MAX]
	v2 = V[V >= MAX]

	v2 = v2 - 2 * MAX

	v = np.append(v2, v1)

	if len(v) != len(V): print('Error')

	return v
	
'''
return V - MAX
'''

@jit
def back_shift(V, mom):
	v1 = V[mom < 0]
	v2 = V[mom >= 0]

	#print(v1, v2)

	v = np.append(v2, v1)

	if len(v) != len(V): print('Error')

	return v

@jit
def back_shift_p(V, MAX):
	

	v1 = V[V < 0]
	v2 = V[V >= 0]

	#print(v1, v2)
	v1 = v1 + 2* MAX
	v = np.append(v2, v1)

	if len(v) != len(V): print('Error')

	return v

'''
	return V + MAX
'''


##################
# stato iniziale #
##################

def wave_pack(x, x0, p0, σ, N, L, h):
	
	try: 
		if abs(p0) > np.pi*h*N / L: #p0 <= -np.pi*h*N / L or
			print('p0 out of range')
			raise ValueError

		else:
			cost = 1 / ((np.pi / 2 )**(0.25) * np.sqrt(σ))
			return cost * np.exp(- ((x-x0)**2)/σ**2 ) * np.exp(+1j * p0 * (x -x0) / h)
	
	except ValueError: 
		return 'p is out of range'

def gaus(x, x0, σ):
	return np.exp(- ((x-x0)**2 /  (2*σ**2))) / (np.sqrt(2*np.pi) *  σ )

def quad(x, x0, p0, w, h):
	if x < x0-w or x > x0+w:
		return 0
	else:
		return (1 / np.sqrt(2*w)) * np.exp(1j * p0 * (x -x0) / h)

def dec_exp(x, x0, p0, w, h):
	if x - x0 < -w or x -x0 > w:
		return 0
	else:
		return np.exp(-x-x0)


def tr(x, x0, p0, w, H, h):
	qp = (H + x0 * H /w)
	qm = (H - x0 * H /w)

	if x < x0-w or x > x0+w:
		return 0

	if x >= x0-w and x <= x0:
		return (x * H / w) + qm

	else:
		return - (x * H / w) + qp



#autostati oscillatore
def arm_ES(x, x0, n, m, ω, h):
    
    Q = np.sqrt(m * ω / h) * (x-x0)
    K =  1. / np.sqrt( (2.**n) * (math.factorial(n)) ) * (m*ω/(np.pi*h))**(0.25)

    HR = special.hermite(n, monic=False)
    
    return K * HR( Q ) * np.exp(-(Q**2) / 2)

def arm_ES_superposition(x, x0, m, ω, h, c_n, vector):   # c_n vettore dei coefficienti, vector vettore di autostati
	
	if len(c_n) != len(vector): 
		print('Error: len(c_n) != len(vector)' )
		exit()

	sum = 0
	for i in c_n:
		sum = sum + abs(i)**2
	
	print('Sum of |c_n|^2: ' , sum)

	if sum < 1 - 0.001 or sum > 1 + 0.001: 
		print('Error: not normalized')
		exit()

	y = np.zeros(len(x))
	for i in range(len(vector)):
		y = y + c_n[i] * arm_ES(x, x0, vector[i], m, ω, h)

	#print(norma(y, x[1]-x[0]))

	return y



import numpy.polynomial.hermite as Herm

def hermite(x, x0, n, m, ω, h):
    xi = np.sqrt(m * ω / h)*(x-x0)
    herm_coeffs = np.zeros(n+1)
    herm_coeffs[n] = 1
    return Herm.hermval(xi, herm_coeffs)
  
def stationary_state(x, x0,n, m, ω, h):
    xi = np.sqrt(m*ω/h)*(x-x0)
    prefactor = 1./math.sqrt(2.**n * math.factorial(n)) * (m*ω/(np.pi*h))**(0.25)
    psi = prefactor * np.exp(- xi**2 / 2) * hermite(x, x0, n, m, ω, h)
    return psi

#Coherent state (170! è il massimo che il computer riesce a flottare)
def coherent(k, ε, x, x0, m, ω, h, y ,  n = 0, c = np.zeros(170, dtype = complex)):

	while n < 170 : 
		
		c_n = np.exp( - abs(k)**2 / 2) * k**n / ((math.factorial(n)))**0.5

		y_n = arm_ES(x, x0, n, m, ω, h)

		y += c_n * y_n

		c[n] = c_n

		n+=1
 

	return y , c

#RL wave packet

def RL_wp(x, x0, xV, p0, σ, h, m, α, N, t=0, norm_factor = 0):
    
    σ = σ/2
    #x0 = x0 - xV

    st = σ*(1 + 1j * (h * t / (2 * m * σ**2)) )

    cost = 1 / (2 * np.pi * st**2)**(1/4)   

    v = p0 / m 

    esp1 = (-((x-xV)-x0-v*t)**2 / (4 * st * σ))
    esp2 = 1j * (p0/h) * ((x-xV) - v*t / 2)

    y_G = cost * np.exp(esp1 + esp2)

    RL = (  (1j * (p0 / h)) -  (( (x-xV) - x0 - v*t) / (2 *  σ * st) )    - α * np.tanh(α * (x-xV))  ) * y_G

    if norm_factor == 0: 
        norm_factor =  1 / np.sqrt(norma(RL, np.zeros(N, dtype = complex), N, x[1] - x[0]))

    return norm_factor * RL, norm_factor





################################
# Analitical fourier trasform  #
################################

def F_w(p, x0, p0, σ, h):
	cost = np.sqrt(σ) / (2*np.pi* h**2)**(0.25)
	return cost * np.exp(-σ**2 * (p - p0)**2 / (4 *h**2)) * np.exp(- 1j*p *x0 / h)

def quad_a(p, x0, p0, w, h):
	cost = 2*h / (np.sqrt(2*np.pi*h)) / np.sqrt(2*w)
	if (p-p0) != 0:
		return cost * np.exp(1j*(p-p0)*x0) * np.sin( (p-p0) * w / h) / (p - p0)
	else: 
		return np.sqrt(2*w) / (np.sqrt(2*np.pi*h))

def gaus_a(p, x0, σ, h):
	return (1 / np.sqrt(2*np.pi*h)) * np.exp(-1 * p**2 * σ**2 / h**2 / 2) * np.exp(- 1j*p *x0 / h)

def dec_exp_a(p, x0, p0, w, h):
	#return np.sqrt(2 / np.pi / h) * (1/a) * np.exp(-(p*a)**2 / 4 / h**2)
	#return (1 / (np.sqrt(2 * np.pi * h))) * (-1 / ((1/a) + 1j*p)) * (np.exp(-d*(1/a) + 1j*p) - np.exp(-c*(1/a) + 1j*p))
	R = (1 + 1j * (p-p0))
	return -1 / (np.sqrt(2*np.pi)*R) * (np.exp(-w*R)- np.exp(w*R))** np.exp(- 1j*p *x0 / h)

def tr_a(p, x0, p0, w, H, h):
	
	
	if p == 0:
		return 0.03989 / 2  #H*np.sqrt(2) / (w*np.pi) 
	'''
	qp = (1/w + x0/w**2)
	qm = (1/w - x0/w**2)

	primo = (1j/p) * (x0 + (w-x0)*np.exp(1j*p*w))*np.exp(-1j*p*x0)
	secondo = (-1/p**2) * (np.exp(-1j*p*x0)- np.exp(-1j*p*(x0-w)))
	terzo = (1j/p) * (np.exp(-1j*p*x0)- np.exp(-1j*p*(x0-w)))
	quarto = (1j/p) * (-x0 + (w+x0)*np.exp(-1j*p*w))*np.exp(-1j*p*x0)
	quinto = (-1/p**2) * (- np.exp(-1j*p*x0)+ np.exp(-1j*p*(x0+w)))
	sesto = (1j/p) * (- np.exp(-1j*p*x0)+ np.exp(-1j*p*(x0+w)))

	return 1/np.sqrt(2*np.pi)/20 * ((1/w**2)* (primo + secondo) + qp * terzo - (1/w**2) * (quarto + quinto) + qm * sesto )
	'''
	#return (2 / np.pi) * (np.sin(np.pi*w*p / 2 / np.pi)/(np.pi*w*p / 2 / np.pi))**2

	return H* np.sqrt(2) / (w*np.sqrt(np.pi)*p**2)* (1 - np.cos( p * w / h)) *  np.exp(- 1j*p *x0 / h)

def arm_ES_a(p, x0, p0, n, m, ω, h):
	β = np.sqrt(m * ω)

	return (-1j)**n * (1 / β) * arm_ES(p / β**2, p0 / β**2, n, m, ω, h)* np.exp(- 1j*p *x0 / h)

'''


##############
# Test tempo matrice
##############

#senza matrici


def Dft(y, L, N, x, p):
	f = np.empty(0)
	for k in p[:-2]:
		
		c = L / np.sqrt(2 * np.pi * h) / N
		
		s = 0
		contatore = 0
		for i in x[:-2]:
			s += np.exp(-1j * k * i / h) * y[contatore]
			contatore +=1

		f = np.append(f, c*s)	
	return f

def iDft(f, L, N, x, p):
	y = np.empty(0)
	for k in x[:-2]: #Sto meno due non mi convince
		
		c = np.sqrt(2 * np.pi * h) / L
		s = 0

		contatore = 0
		for i in p[:-2]:
			s += np.exp(+1j * k * i / h) * f[contatore]
			contatore +=1

		y = np.append(y, c*s)	
	return y



start = timer()

ff = f_matrix(x, p)
iff = if_matrix(x, p)

end = timer()
t0 = end - start

start = timer()
for i in range(0,10):
	mfy = mDft(y, x, p, ff)
	myi = miDft(mfy, x, p, iff)
	i += 1

end = timer()
print(t0 + end - start) 

start = timer()
for i in range(0,10):
	fy = Dft(y, L, N, x, p)
	yi = iDft(fy, L, N, x, p)
	i += 1

end = timer()
print(end - start) 

plt.figure()
plt.plot(p, abs(fy))
plt.plot(p ,abs(mfy))
plt.ylim(-4,4)
plt.show()

'''

'''



###############
# test shift
###############

def shift(V):
	v = np.zeros(len(V))
	for i in range(len(V)):
		v[i] = V[i]

	i = 1
	while v[-i] > np.pi * h* N / L:
		v[-i] = v[-i] - 2* np.pi * h* N / L
		i+=1

	return v


def back_shift(V):
	v = np.zeros(len(V))
	for i in range(len(V)):
		v[i] = V[i]

	i = 1
	while v[-i] < 0:
		v[-i] = v[-i] + 2*np.pi*h*N / L
		i +=1 

	return v


fy = mDft(y, x, p, ff)


print((len(p) - 1) /2, '\t', np.pi*h*N / L )

p_s = shift(p)

plt.figure()
plt.plot(p, abs(fy))
plt.plot(p_s, abs(fy), 'r', zorder = 1)
plt.ylim(-4,4)
plt.show()

p_s = back_shift(p)


plt.figure()
plt.plot(p, abs(fy))
plt.plot(p_s ,abs(fy))
plt.ylim(-4,4)
plt.show()


yi = miDft(fy, x, p, iff)
yi_s = miDft(fy, x, p_s, iff)




plt.figure()
plt.plot(x, y, label = 'start')
plt.plot(x, yi, label = 'i')
plt.plot(x, yi_s, label = 'shift')
plt.legend()
plt.show()




'''

'''
#############
# test DFT e inv con sin
#############

y = np.sin(2*x) + np.sin(-3*x + 2)

plt.figure()
plt.plot(x, y)
plt.show()


yf = mDft(y, x, p, ff)
iyf = miDft(yf, x, p, iff)

plt.figure()
plt.plot(p, yf)
plt.show()

plt.figure()
plt.plot(x, y)
plt.plot(x, iyf, label = 'inv')
plt.legend()
plt.show()


'''
