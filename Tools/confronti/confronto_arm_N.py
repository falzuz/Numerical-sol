
import numpy as np
import matplotlib.pyplot as plt
import sys

import os

flag = 1  # Ricalcola risultati

FRAME = 100

list = ['10', '20',  '50', '100',  '200',  '500', '1000' ]

PATH =  str(sys.argv[1])

if flag == 1:
    for i in list:
        os.system('python3 /home/falzo/Scrivania/Sol_eqS/arm_analytical_test.py ' + PATH + '/' + i)


plt.figure()
for i in list:

    try: 
        o = np.load(PATH + '/' + i + '/results_t.npz')

        time = o['time']
        distanza = o['dist']

        plt.plot(time, distanza, label = i)

    except: print('result does not exist' + i)

plt.legend()
plt.show()