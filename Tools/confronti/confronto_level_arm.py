
import numpy as np
import matplotlib.pyplot as plt
import sys

import os

flag = 1  # Ricalcola risultati

list = ['1', '2', '3']

PATH =  str(sys.argv[1])

if flag == 1:
    for i in list:
        os.system('python3 /home/falzo/Scrivania/Sol_eqS/arm_analytical_test.py ' + PATH + '/' + i)


plt.figure()
for i in list:

    o = np.load(PATH + '/' + i + '/results_t.npz')

    time = o['time']
    distanza = o['dist']

    plt.plot(time, distanza, label = i)

plt.legend()
plt.show()