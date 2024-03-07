import numpy as np
import matplotlib.pyplot as plt

import sys, os

sys.path.insert(0, os.getcwd())
import DFT

sys.path.insert(0, '/home/falzo/Scrivania/Dati/free/2/Run')
from Par import *

o = np.load('/home/falzo/Scrivania/Dati/free/2/Run/0.npz')
i = np.load('/home/falzo/Scrivania/Dati/free/2/Run/999.npz')

#plt.plot(o['x'], abs(o['Ψ']))
#plt.plot(o['x'], abs(i['Ψ']))

x = o['x']
y =  o['Ψ']
y_1 = i['Ψ']

P, F = DFT.full_Dft( y, x, p, h, P_MAX, N, L, N, np.zeros((N,N) , dtype=complex), np.zeros(N, dtype=complex))

P, F_1 = DFT.full_Dft( y_1, x, p, h, P_MAX, N, L, N, np.zeros((N,N) , dtype=complex), np.zeros(N, dtype=complex))

plt.figure()
plt.plot(P, abs(F))
plt.plot(P, abs(F_1))

plt.show()