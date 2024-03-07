import numpy as np
import scipy.constants as cost

##################
# Set parameters #
##################

# Space par.

L = 10
N = 500 #divisioni spaziali
dt = 1
Level = 2
LEC_MAX = [1, 1/2, 1] # list of expansion coefficent for V_MAX
LEC = [	[[1, 1]], 
		[[1./2, 1], [1./2, 0]], 
		[[1, -1./24], [-2./3, 3./4], [2./3, 7./24]] ]

debug_E =  False
debug_norma = False
STOP_BOUNDARIES =  False

ST_BD_COEFF = 0.025

norm_factor = 0


sim_list = ['Standard', 't dependent', 'Performance', 'Performance t-dep',  'Best']
sim = 'Standard'

#Choose potential 
Potential_list = ['Free','Step', 'Barrier', 'Hole', 'Coulomb', 'Armonic Oscillator', 'Reflectionless', 'Insert...', 'Infinite Barrier' ] 
Potential = 'Free'

V_function = ""
lib = ""

#h = cost.hbar
h = 1
m = 1000

T = 1000
cont = 0
STD_NORMA = 1
norm = STD_NORMA


FRAME = 500
FPS = 10
V_flag = False
real_flag = False
imag_flag = False

#Packet par.
ψ_list = ['Wave Packet','Armonic oscillator eingestate','Reflecionless WP', 'Coherent' ,'Custom']
ψ_name = 'Wave Packet'



x0_coefficient = 0.5
x0 = x0_coefficient * L # -L/4

P_MAX = np.pi*h*N / L

p0_coefficient = 0.3
p0 = p0_coefficient * P_MAX  	# deve essere minore di P_MAX
								#  perché se no torna indietro

σ0_coefficient = 0.05
σ0 = σ0_coefficient * L

# Potentials par.
x_V_centered_coefficient = 1./2
x_V_centered = x_V_centered_coefficient * L		# Dove è centrato il potenziale

step_x_coefficient = 0.6
step_x = step_x_coefficient * L

barrier_width_coefficient = 0.1
barrier_width = barrier_width_coefficient * L





V_MAX = 2 * np.pi * h / dt / LEC_MAX[Level-1]
OUT = False # Flag V out of range

V0_coefficient = 0.03
V0 = V0_coefficient * V_MAX


a_MAX = (V_MAX - 0.0001) * 4 / L**2

a_coefficient = 0.1
a = a_coefficient * a_MAX
ω = np.sqrt(2 * a / m)
n = 1

RF_l = 1
α_MAX = np.sqrt(V_MAX*(2*m / h**2)*(1/(RF_l*(RF_l+1))))
α_coefficient = 0.1
α = α_coefficient * α_MAX


#FOURIER

w = 0.1 * L #exp
H = 1		#tr

c = -0.03 * L 	#tr
d = 0.03 * L	#tr


#########
# Space #
#########

space_start = 0
space_end = L

x = np.linspace(space_start, space_end, N, endpoint = False)
#x = np.linspace(-L/2, L/2, N, endpoint = True )

p = np.linspace(0, 2*P_MAX, N, endpoint = False)
#p = np.linspace(-P_MAX, P_MAX, N, endpoint = False)


complete = open('complete.txt', 'r')

complete_par = complete.read()

complete.close()
