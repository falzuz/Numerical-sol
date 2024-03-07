#! /bin/python3

from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import tkinter
import numpy as np

import os 
import time

import sys
#sys.path.insert(0, '/home/falzo/Scrivania/Sol_eqS')
from Parameters import *
import DFT 
import time_evolution as __ 

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
#from matplotlib.backend_bases import key_press_handler
import matplotlib.pyplot as plt
import matplotlib.lines as ll

import threading

#def test(value):
#    print('OK')

PATH = os.getcwd()
print(PATH)
dic = {}  # dictionari for insert potential

ε = 0.01
#debug_norma = 0


############
# Commands #
############

def Start_cmd():
    global PATH, sim

    #sim = sim_.get()


    print(PATH, '\nmode: ', sim)

    log = open(PATH + '/Run/abort.log', 'w') 
    log.write('')
    log.close()

    st = time.time()

    if sim == sim_list[0]: os.system('python3 ./simulation.py  '+ str(PATH) + '/Run')
    elif sim == sim_list[1]: os.system('python3 ./simulation_t_dep.py  '+ str(PATH) + '/Run')
    elif sim == sim_list[2]: os.system('python3 ./simulation_performance.py  '+ str(PATH) + '/Run')
    elif sim == sim_list[3]: os.system('python3 ./simulation_performance_t_dep.py  '+ str(PATH) + '/Run')
    elif sim == sim_list[4]: os.system('python3 ./simulation_best.py  '+ str(PATH) + '/Run')

    print('TIME_sim: ', time.time() - st)

    log = open( PATH + '/Run/abort.log', 'r')
    if log.readline() == 'Simulation aborted': 
        Com.config(text='Simulation aborted')
        return -1
    else: 
        os.system('python3 ./animazione.py  ' + str(PATH) + '/Run')


def Save_cmd():
    global PATH, sim, norm_factor

    L = L_.get()
    N = N_.get()
    dt = dt_.get()

    m = m_.get()
    FRAME = FRAME_.get()
    
    norma = norma_.get()

    x_V_centered_coefficient = x_V_centered_.get()
    step_x_coefficient = step_x_.get()
    barrier_width_coefficient =barrier_width_.get()
    RF_l = RF_l_.get()
    V0_coefficient = V0_.get()
    a_coefficient = a_.get()
    α_coefficient = α_.get()

    x0_coefficient = x0_.get()
    p0_coefficient = p0_.get()
    σ0_coefficient = σ0_.get()
    n = n_.get()

    Level = Level_.get()
    h = h_.get()
    ψ_name = ψ_.get()
    Potential = Potential_.get()

    Video_ = Video_flag.get()
    FPS = FPS_.get()

    real_ = real_flag.get()
    imag_ = imag_flag.get()

    V_function = Insert_.get()
    lib = Ins_LIB_.get()

    sim = sim_.get()
    debug_E = var_E.get()
    debug_norma = var_norma.get()
    STOP_BOUNDARIES = var_ST.get()
    ST_BD_COEFF = ST_BD_COEFF_.get()

    try:
        int(N)
        int(FRAME)
        int(RF_l)
        int(n)

        Com.config(text='')

        print('SAVE: '+ PATH + '/Run/')
        
        try: 
            Par_file = str(PATH) + '/Run/Par.py'
            file = open(Par_file, "w")

        except: 
            os.mkdir( PATH + '/Run')
            Par_file = str(PATH) + '/Run/Par.py'
            file = open(Par_file, "w")


        

        file.write( 
            'import numpy as np \nimport scipy.constants as cost \n\n'        
            'L = ' + str(L) + '\n' +
            'N = ' + str(N) + '\n' +
            'dt = ' + str(dt) + '\n' +

            'm = ' + str(m) + '\n' +
            'FRAME = ' + str(FRAME) + '\n' +
            'STD_NORMA = ' + str(norma) + '\n' +

            'V_flag = ' + str(Video_) + '\n' +
            'FPS = ' + str(FPS) + '\n' +

            'real_flag = ' + str(real_) + '\n' +
            'imag_flag = ' + str(imag_) + '\n' +


            'x_V_centered_coefficient = ' + str(x_V_centered_coefficient) + '\n' +
            'step_x_coefficient = ' + str(step_x_coefficient) + '\n' + # + str(x_V_centered_coefficient) + '\n' + #
            'barrier_width_coefficient = ' + str(barrier_width_coefficient) + '\n' +
            'RF_l = ' + str(RF_l) + '\n' +
            'V0_coefficient = ' + str(V0_coefficient) + '\n' +
            'a_coefficient = ' + str(a_coefficient) + '\n' +
            'α_coefficient = ' + str(α_coefficient) + '\n' +

            'x0_coefficient = ' + str(x0_coefficient) + '\n' +
            'p0_coefficient = ' + str(p0_coefficient) + '\n' +
            'σ0_coefficient = ' + str(σ0_coefficient) + '\n' +
            'n = ' + str(n) + '\n' +
            'Level = ' + str(Level) + '\n' +
            'h = ' + str(h) + '\n' +
            'ψ_name = \'' + str(ψ_name) + '\'\n' + 
            'Potential = \'' + str(Potential) + '\'\n' +
            'PATH_CHECK = \'' + str(PATH) + '\'\n'
            'V_function = \'' + str(V_function) + '\'\n' +
            'lib = \'' + str(lib) + '\'\n'+
            'sim = \'' + str(sim) + '\'\n' +
            'debug_E = ' + str(debug_E) + '\n' +
            'debug_norma = ' + str(debug_norma) + '\n' +
            'STOP_BOUNDARIES = ' + str(STOP_BOUNDARIES) + '\n' +
            'ST_BD_COEFF = ' + str(ST_BD_COEFF) + '\n' 
            'norm_factor = ' + str(norm_factor) + '\n'


        )

        file.close()

        file = open(Par_file, "a")    
        file.write (complete_par)
        file.close()

        Com.config(text='Saved in '+ Par_file)


    except ValueError:
        Com.config(text='Number of points, Number of frames, \n Armonic Level and l must be integers')

    
def select_file():
    global PATH

    root.directory = filedialog.askdirectory()

    Com.config(text=root.directory)

    PATH = root.directory
    print(PATH)

def Quit_cmd():
    root.destroy()
    try: p_v.destroy()
    except: print('\n')

    exit()



def update_graph(value):

    global norm_factor

    x0_Scale_Lab.config(text = '%.2f' % x0_var.get())
    p0_Scale_Lab.config(text = '%.2f' % p0_var.get())
    σ0_Scale_Lab.config(text = '%.2f' % σ0_var.get())

    x_V_centered_Scale_Lab.config(text = '%.2f' % x_V_centered_var.get())
    step_x_Scale_Lab.config(text = '%.2f' % step_x_var.get())
    barrier_width_Scale_Lab.config(text = '%.2f' % barrier_width_var.get())
    V0_Scale_Lab.config(text = '%.2f' % V0_var.get())
    a_Scale_Lab.config(text = '%.2f' % a_var.get())
    α_Scale_Lab.config(text = '%.2f' % α_var.get())

    L = float(L_.get())
    dt = float(dt_.get())
    m = float(m_.get())
    norma = float(norma_.get())

    x_V_centered_coefficient = float(x_V_centered_.get())
    step_x_coefficient = float(step_x_.get()) # float(x_V_centered_.get()) #
    barrier_width_coefficient = float(barrier_width_.get())
    V0_coefficient = float(V0_.get())
    a_coefficient = float(a_.get())
    α_coefficient = float(α_.get())

    x0_coefficient = float(x0_.get())
    p0_coefficient = float(p0_.get())
    σ0_coefficient = float(σ0_.get())
    

    ψ_name = ψ_.get()
    Potential = Potential_.get()

    
    N = int(float(N_.get()))
    RF_l = int(float(RF_l_.get()))
    n = int(float(n_.get()))
    
    h = float(h_.get())

    V_function = Insert_.get()
    lib = Ins_LIB_.get()


    cont = 0 
    norma =STD_NORMA 
    x0 = x0_coefficient * L 
    P_MAX = np.pi*h*N / L
    p0 = p0_coefficient * P_MAX  
    σ0 = σ0_coefficient * L
    x_V_centered = x_V_centered_coefficient * L
    step_x = step_x_coefficient * L
    barrier_width = barrier_width_coefficient * L
    
    V_MAX = 2 * np.pi * h / dt / LEC_MAX[Level-1]
    V0 = V0_coefficient * V_MAX
    a_MAX = (V_MAX - ε) * 4 / L**2
    a = a_coefficient * a_MAX
    ω = np.sqrt(2 * a / m)
    α_MAX = np.sqrt(V_MAX*(2*m / h**2)*(1/(RF_l*(RF_l+1))))
    α = α_coefficient * α_MAX
    x = np.linspace(0, L, N, endpoint = False)
    p = np.linspace(0, 2*P_MAX, N, endpoint = False)


    #stato iniziale
    if ψ_name == ψ_list[0]: y = DFT.wave_pack(x, x0, p0, σ0, N, L, h)
    elif ψ_name == ψ_list[1]: y = DFT.arm_ES(x, x0, n, m, ω, h)

    
    elif ψ_name == ψ_list[2]: y, norm_factor = DFT.RL_wp(x, x0, x_V_centered, p0, σ0, h, m, α, N, norm_factor = 0)
    elif ψ_name == ψ_list[3]: y, coefficients = DFT.coherent(n, 0, x, x0, m, ω, h, np.zeros(N, dtype = complex))


    
    elif ψ_name == ψ_list[4]:
        
        o = np.load(PATH + '/custom.npz')

        y	 = o['Ψ']

        if len(Ψ) != len(x):  
            print('Error len(Ψ) != len(x)')
            
            f = open('abort.log', 'w')
            f.write('Simulation aborted')
            f.close()

            exit()
   

    V = np.zeros(len(x))    
    
    for i in range(len(V)):
        if   Potential == Potential_list[1]:  V[i] = __.step(x[i], step_x, V0) 
        elif Potential == Potential_list[2]:  V[i] = __.barrier(x[i], barrier_width, V0, x_V_centered)
        elif Potential == Potential_list[3]:  V[i] = __.buca(x[i], barrier_width, V0, x_V_centered)
        elif Potential == Potential_list[4]:  V[i] = __.tunnel(x[i], x_V_centered, V0)
        elif Potential == Potential_list[5]:  V[i] = __.arm_osc(x[i], x_V_centered, a)
        elif Potential == Potential_list[6]:  V[i] = __.RL(x[i], RF_l, x_V_centered, h, m, α)
        elif Potential == 'Infinite Barrier': V[i] = __.step(x[i], L/2, V_MAX-10**(-2)) 


    if Potential == Potential_list[7]:
        global dic
        exec("import " + lib +  "\ndef F(x, t): return " + V_function, dic)
        
        for i in range(len(V)): 
            V[i] = dic['F'](x[i], 0)
            OUT, V_max = __.V_out_of_range(V, V_MAX)
            if OUT == True:
                Com.config(text = 'V out of range')
                return -1



    E, T, U = __.H_mean(x, p, y, x[1] - x[0], p[1]- p[0], m, V, h, P_MAX, int(N), L, int(N), np.ndarray((N, N), dtype=complex), np.zeros(N, dtype = complex), np.zeros(int(N), dtype = complex))

    line1.set_data(x, V)
    line2.set_data(x, abs(y)**2)
    line3.set_data(x, np.full(len(x), E))

    global line4
    line4.remove()
    line4 = ax.fill_between(x, V, np.full(len(x), -V_MAX), facecolor='#CCCCCC', zorder= 0)

    line5.set_data(x, y.real)
    ax.add_line(line5) 

    if real_flag.get() == False: 
        ax.lines[-1].remove()

    line6.set_data(x, y.imag)
    ax.add_line(line6) 

    if imag_flag.get() == False: 
        ax.lines[-1].remove()

    ax.legend()

    # required to update canvas and attached toolbar!
    canvas.draw()
    
    if real_flag.get() == True: 
        ax.lines[-1].remove()
    
    if imag_flag.get() == True: 
        ax.lines[-1].remove()


    p, f = DFT.full_Dft(y, x, p, h, P_MAX, N, L, N, np.ndarray((N, N), dtype=complex), np.zeros(N, dtype = complex))

    line_p2.set_data(p, abs(f)**2)
    line_p3.set_data(p, np.full(len(p), E))

    line_p5.set_data(p, f.real)
    ax_p.add_line(line_p5) 

    if real_flag.get() == False: 
        ax_p.lines[-1].remove()

    line_p6.set_data(p, f.imag)
    ax_p.add_line(line_p6) 

    if imag_flag.get() == False: 
        ax_p.lines[-1].remove()

    ax_p.legend()



    # required to update canvas and attached toolbar!
    canvas_p.draw()
    
    if real_flag.get() == True: 
        ax_p.lines[-1].remove()
    
    if imag_flag.get() == True: 
        ax_p.lines[-1].remove()

    if abs(f[0])**2 > 0.01 or abs(f[-1])**2 > 0.01 or abs(f[p == 0])**2 > 0.01:
        Com_p.config(text = 'Pay attention on p boundaries')
        Com.config(text = 'Pay attention on p boundaries')
        p_v.attributes('-topmost',True)
        line_p2.set_color('red')

    else:
        Com_p.config(text = '')
        Com.config(text = '')
        p_v.attributes('-topmost',False)
        line_p2.set_color('darkorchid')

def update_menu_pot(value):

    if Potential_.get() == Potential_list[0]:
        Potential_par.grid_remove()

    else: Potential_par.grid( padx = 5, pady = 3, column=0, row=11)



    x_V_centered_Lab.grid_remove()
    barrier_width_Lab.grid_remove()
    V0_Lab.grid_remove()
    step_x_Lab.grid_remove()
    a_Lab.grid_remove()
    RF_l_Lab.grid_remove()
    α_Lab.grid_remove()

    x_V_centered_.grid_remove()
    barrier_width_.grid_remove()
    V0_.grid_remove()
    step_x_.grid_remove()
    a_.grid_remove()
    RF_l_.grid_remove()
    α_.grid_remove()

    x_V_centered_Scale_Lab.grid_remove()
    step_x_Scale_Lab.grid_remove()
    barrier_width_Scale_Lab.grid_remove()
    V0_Scale_Lab.grid_remove()
    a_Scale_Lab.grid_remove()
    α_Scale_Lab.grid_remove()

    Ins_LIB_Lab.grid_remove()
    Ins_LIB_.grid_remove()
    Insert_Lab.grid_remove()
    Insert_.grid_remove()


    if Potential_.get() == Potential_list[1]:
        step_x_Lab.grid( padx = 5, pady = 3, column=0, row=13)
        V0_Lab.grid( padx = 5, pady = 3, column=0, row=16)

        step_x_.grid( padx = 5, pady = 3, column=1, row=13)
        step_x_Scale_Lab.grid(padx = 5, pady = 3, column=2, row=13)


        V0_.grid( padx = 5, pady = 3, column=1, row=16)
        V0_Scale_Lab.grid(padx = 5, pady = 3, column=2, row=16)


    if Potential_.get() == Potential_list[2]:
        x_V_centered_Lab.grid( padx = 5, pady = 3, column=0, row=12)
        barrier_width_Lab.grid( padx = 5, pady = 3, column=0, row=14)
        V0_Lab.grid( padx = 5, pady = 3, column=0, row=16)

        x_V_centered_.grid( padx = 5, pady = 3, column=1, row=12)
        barrier_width_.grid( padx = 5, pady = 3, column=1, row=14)
        V0_.grid( padx = 5, pady = 3, column=1, row=16)

        x_V_centered_Scale_Lab.grid(padx = 5, pady = 3, column=2, row=12)
        barrier_width_Scale_Lab.grid(padx = 5, pady = 3, column=2, row=14)
        V0_Scale_Lab.grid(padx = 5, pady = 3, column=2, row=16)

    if Potential_.get() == Potential_list[3]:
        x_V_centered_Lab.grid( padx = 5, pady = 3, column=0, row=12)
        barrier_width_Lab.grid( padx = 5, pady = 3, column=0, row=14)
        V0_Lab.grid( padx = 5, pady = 3, column=0, row=16)

        x_V_centered_.grid( padx = 5, pady = 3, column=1, row=12)
        barrier_width_.grid( padx = 5, pady = 3, column=1, row=14)
        V0_.grid( padx = 5, pady = 3, column=1, row=16)

        x_V_centered_Scale_Lab.grid(padx = 5, pady = 3, column=2, row=12)
        barrier_width_Scale_Lab.grid(padx = 5, pady = 3, column=2, row=14)
        V0_Scale_Lab.grid(padx = 5, pady = 3, column=2, row=16)

    if Potential_.get() == Potential_list[4]:
        x_V_centered_Lab.grid( padx = 5, pady = 3, column=0, row=12)
        V0_Lab.grid( padx = 5, pady = 3, column=0, row=16)

        x_V_centered_.grid( padx = 5, pady = 3, column=1, row=12)
        V0_.grid( padx = 5, pady = 3, column=1, row=16)

        x_V_centered_Scale_Lab.grid(padx = 5, pady = 3, column=2, row=12)
        V0_Scale_Lab.grid(padx = 5, pady = 3, column=2, row=16)

    if Potential_.get() == Potential_list[5]:
        x_V_centered_Lab.grid( padx = 5, pady = 3, column=0, row=12)
        a_Lab.grid( padx = 5, pady = 3, column=0, row=17)

        x_V_centered_.grid( padx = 5, pady = 3, column=1, row=12)
        a_.grid( padx = 5, pady = 3, column=1, row=17)

        x_V_centered_Scale_Lab.grid(padx = 5, pady = 3, column=2, row=12)
        a_Scale_Lab.grid(padx = 5, pady = 3, column=2, row=17)

    if Potential_.get() == Potential_list[6]:
        x_V_centered_Lab.grid( padx = 5, pady = 3, column=0, row=12)
        RF_l_Lab.grid( padx = 5, pady = 3, column=0, row=15)
        #V0_Lab.grid( padx = 5, pady = 3, column=0, row=16)
        α_Lab.grid( padx = 5, pady = 3, column=0, row=14)



        x_V_centered_.grid( padx = 5, pady = 3, column=1, row=12)
        RF_l_.grid( padx = 5, pady = 3, column=1, row=15)
        #V0_.grid( padx = 5, pady = 3, column=1, row=16)
        α_.grid( padx = 5, pady = 3, column=1, row=14)



        x_V_centered_Scale_Lab.grid(padx = 5, pady = 3, column=2, row=12)
        α_Scale_Lab.grid(padx = 5, pady = 3, column=2, row=14)

    if Potential_.get() == Potential_list[7]:
        Insert_Lab.grid( padx = 5, pady = 3, column=0, row=12)
        Insert_.grid(padx = 5, pady = 3, column=1, row=12)

        Ins_LIB_Lab.grid( padx = 5, pady = 3, column=0, row=13)
        Ins_LIB_.grid(padx = 5, pady = 3, column=1, row=13)

        



def update_menu_ψ(value):
    ψ_par.grid( padx = 5, pady = 3, column=0, row=18)

    x0_Lab.grid_remove()
    p0_Lab.grid_remove()
    σ0_Lab.grid_remove()
    
    x0_.grid_remove()
    p0_.grid_remove()
    σ0_.grid_remove()

    x0_Scale_Lab.grid_remove()
    p0_Scale_Lab.grid_remove()
    σ0_Scale_Lab.grid_remove()

    n_Lab.grid_remove()
    n_.grid_remove()


    
    if ψ_.get() ==  ψ_list[0]:
        x0_Lab.grid( padx = 5, pady = 3, column=0, row=20)
        p0_Lab.grid( padx = 5, pady = 3, column=0, row=21)
        σ0_Lab.grid( padx = 5, pady = 3, column=0, row=22)
        

        x0_.grid( padx = 5, pady = 3, column=1, row=20)
        p0_.grid( padx = 5, pady = 3, column=1, row=21)
        σ0_.grid( padx = 5, pady = 3, column=1, row=22)

        x0_Scale_Lab.grid( padx = 5, pady = 3, column=2, row=20)
        p0_Scale_Lab.grid( padx = 5, pady = 3, column=2, row=21)
        σ0_Scale_Lab.grid( padx = 5, pady = 3, column=2, row=22)

    if ψ_.get() ==  ψ_list[2]:
        x0_Lab.grid( padx = 5, pady = 3, column=0, row=20)
        p0_Lab.grid( padx = 5, pady = 3, column=0, row=21)
        σ0_Lab.grid( padx = 5, pady = 3, column=0, row=22)
        

        x0_RL.grid( padx = 5, pady = 3, column=1, row=20)
        p0_.grid( padx = 5, pady = 3, column=1, row=21)
        σ0_.grid( padx = 5, pady = 3, column=1, row=22)

        x0_Scale_Lab.grid( padx = 5, pady = 3, column=2, row=20)
        p0_Scale_Lab.grid( padx = 5, pady = 3, column=2, row=21)
        σ0_Scale_Lab.grid( padx = 5, pady = 3, column=2, row=22)


    elif ψ_.get() ==  ψ_list[1] or ψ_.get() ==  ψ_list[3]:
        x0_Lab.grid( padx = 5, pady = 3, column=0, row=20)
        x0_.grid( padx = 5, pady = 3, column=1, row=20)
        x0_Scale_Lab.grid( padx = 5, pady = 3, column=2, row=20)

        n_Lab.grid( padx = 5, pady = 3, column=0, row=23)
        n_.grid( padx = 5, pady = 3, column=1, row=23)


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


def a_save():
    sim = sim_.get()
    debug_E = var_E.get()
    debug_norma = var_norma.get()
    STOP_BOUNDARIES = var_ST.get()
    ST_BD_COEFF = ST_BD_COEFF_.get()
    print(sim, debug_E, debug_norma, STOP_BOUNDARIES, ST_BD_COEFF )

    '''
    adv_save = open(PATH + '/Run/advanced_option.py', 'w')
    adv_save.write(
        'sim = ' + str(sim) + '\n' +
        'debug_E = ' + str(debug_E) + '\n' +
        'debug_norma = ' + str(debug_norma) + '\n' +
        'STOP_BOUNDARIES = ' + str(STOP_BOUNDARIES) + '\n' +
        'ST_BD_COEFF = ' + str(ST_BD_COEFF) + '\n' 
        )

    adv_save.close()
    '''

    ad.withdraw()

def adv():

    ad.deiconify()

    sim_.grid(padx = 5, pady = 3, column=0, row=0)


    debug_E_.grid(padx = 5, pady = 3, column=0, row=1)
    debug_norma_.grid(padx = 5, pady = 3, column=0, row=2)
    STOP_BOUNDARIES_.grid(padx = 5, pady = 3, column=0, row=3)

    ST_BD_COEFF_lab.grid(padx = 5, pady = 3, column=0, row=4)
    ST_BD_COEFF_.grid(padx = 5, pady = 3, column=1, row=4)

    ad_save.grid(row = 5)

    ad.protocol("WM_DELETE_WINDOW", ad.withdraw)


    ad.mainloop()


######################
# Windows and Frames #
######################

root = Tk()
root.title('Parameters Setup')

style = ttk.Style(root)
root.tk.call('source', 'azure.tcl')
style.theme_use('azure')


Parameters = ttk.Frame(root, padding=10)
Parameters.grid(column = 0, row = 0)

Frame_2 = ttk.Frame(root, padding=10)
Frame_2.grid(column = 1, row = 0)

Menu = ttk.Frame(Frame_2, padding = 2)
Menu.grid(column = 0, row = 0)

Graphic = ttk.Frame(Frame_2,padding = 2)
Graphic.grid(column = 0, row = 1)

Buttons = ttk.Frame(Frame_2, padding=2)
Buttons.grid(column = 0, row = 2)

Communications = ttk.Frame(Frame_2, padding=2)
Communications.grid(column = 0, row = 3)

p_v = Tk()
p_v.title('P rapresentation')

style = ttk.Style(p_v)
p_v.tk.call('source', 'azure.tcl')
style.theme_use('azure')

Communications_p = ttk.Frame(p_v, padding=10)
Communications_p.grid(column = 0, row = 3)

ad = Tk()
ad.withdraw()
ad.title('Advanced options')

style_ad = ttk.Style(ad)
ad.tk.call('source', 'azure.tcl')
style_ad.theme_use('azure')

ad_par = ttk.Frame(ad, padding=10)
ad_par.grid(column = 0, row = 0)



###########
# Objects #
###########


# X Canvas
Ψ = DFT.wave_pack(x, x0, p0, σ0, N, L, h)

V = np.zeros(len(x))

for i in range(len(V)):
	if   Potential == Potential_list[1]:  V[i] = __.step(x[i], step_x, V0) 
	elif Potential == Potential_list[2]:  V[i] = __.barrier(x[i], barrier_width, V0, x_V_centered)
	elif Potential == Potential_list[3]:  V[i] = __.buca(x[i], barrier_width, V0, x_V_centered)
	elif Potential == Potential_list[4]:  V[i] = __.tunnel(x[i], x_V_centered, V0)
	elif Potential == Potential_list[5]:  V[i] = __.arm_osc(x[i], x_V_centered, a)
	elif Potential == Potential_list[6]:  V[i] = __.RL(x[i], RF_l, x_V_centered, h, m, α)


#fig = Figure(figsize=(5, 4), dpi=N)
fig = plt.figure()


ax = fig.add_subplot()
line1, = ax.plot(x, V, label = 'V', color='slategray', zorder = 0)
line2, = ax.plot(x, abs(Ψ)**2, label = r'$|ψ|^2$', color='#007FFF', zorder = 2)
line3, = ax.plot(x, np.full(len(x), E), '--', label = 'E', color='firebrick', zorder = 1) #ax.hlines(E , 0.1, L-0.1, colors = 'red', label = 'E', zorder = 0)

line4 = ax.fill_between(x, V, np.full(len(x), -V_MAX), facecolor = '#CCCCCC', zorder = 0)
ax.set_ylim(-0.2 * max(abs(Ψ)**2), 1.2*max(abs(Ψ)**2))


line5 = plt.Line2D(x, Ψ.real, label = r'$Re(ψ)$', color='palegreen', zorder = 0)
line6 = plt.Line2D(x, Ψ.imag, linestyle='--', label = r'$Im(ψ)$', color='palevioletred', zorder = 0)


ax.set_xlabel("x")
ax.set_ylabel("Energy")
ax.legend()
#G = ax.grid( linewidth = 1, linestyle = '--', color = 'grey', alpha = 0.5, zorder = 0)
#G.set_xdata(np.linspace(0, L, int(N / 10)), np.linspace(-V_MAX, V_MAX, int(2*V_MAX /10)))

scale = 1.2
old_fig = zoom_factory(ax,base_scale = scale)

canvas = FigureCanvasTkAgg(fig, master=Graphic)  # A tk.DrawingArea.
canvas.draw()

# pack_toolbar=False will make it easier to use a layout manager later on.
toolbar = NavigationToolbar2Tk(canvas, Graphic, pack_toolbar=False)
toolbar.update()


fig_p = plt.figure()

p, f = DFT.full_Dft(Ψ, x, p, h, P_MAX, N, L, N, np.ndarray((N, N), dtype=complex), np.zeros(N, dtype = complex))

ax_p = fig_p.add_subplot()
line_p2, = ax_p.plot(p, abs(f)**2, label = r'$|F(\psi)|^2$', color = 'darkorchid')
line_p3, = ax_p.plot(x, np.full(len(x), E), '--', label = 'E', color='firebrick', zorder = 1) #ax.hlines(E , 0.1, L-0.1, colors = 'red', label = 'E', zorder = 0)

line_p5 = plt.Line2D(p, f.real, label = r'$Re(F(\psi))$', color='palegreen', zorder = 0)
line_p6 = plt.Line2D(p, f.imag, linestyle='--', label = r'$Im(F(\psi))$', color='palevioletred', zorder = 0)

ax_p.axvline(0, zorder = 0, linestyle = '--', alpha = 0.5, color = 'grey')

ax_p.legend()
#ax_p.grid(linewidth = 1, linestyle = '--', color = 'grey', alpha = 0.5, zorder = 0)
#xdata = np.linspace(-P_MAX, P_MAX, int(N / 10)), ydata = np.linspace(0, 1, 10),

ax_p.set_xlabel("p")
ax_p.set_ylabel("E")

f_p = zoom_factory(ax_p,base_scale = scale)

canvas_p = FigureCanvasTkAgg(fig_p, master=p_v)  # A tk.DrawingArea.
canvas_p.draw()

# pack_toolbar=False will make it easier to use a layout manager later on.
toolbar_p = NavigationToolbar2Tk(canvas_p, p_v , pack_toolbar=False)
toolbar_p.update()

#Buttons
Quit = ttk.Button(Buttons, width = 10,text="Quit", command=Quit_cmd)
Start = ttk.Button(Buttons, width = 10,text="Start", command=Start_cmd) #threading.Thread(target = Start_cmd).start)
Save = ttk.Button(Buttons, width = 10,text="Save", command=Save_cmd)

Sel_path = ttk.Button(Buttons, width = 10,text="Select path", command= select_file)
Adv = ttk.Button(Parameters, width = 20,text="Advance option", command=adv)


#View = ttk.Button(Buttons, width = 10,text="View", command=Save_cmd)

Com = ttk.Label(Communications, text = "")
Com.grid(column = 0, row = 0)

Com_p = ttk.Label(Communications_p, text = "")
Com_p.grid(column = 0, row = 0)

#Paramters

Sim_par = ttk.Label(Parameters, text="Simulation Parameters", font = 'bold', foreground='#007FFF')

L_Lab = ttk.Label(Parameters, text="Lenght")
N_Lab = ttk.Label(Parameters, text="Number of points")
dt_Lab = ttk.Label(Parameters, text="dt")

m_Lab = ttk.Label(Parameters, text="Mass")
FRAME_Lab = ttk.Label(Parameters, text="Number of frames")
FPS_Lab = ttk.Label(Parameters, text="Video fps")
norma_Lab = ttk.Label(Parameters, text = 'Norm') #text="Value of |ψ|^2")

h_Lab = ttk.Label(Menu, text="h_bar")
Potential_Lab = ttk.Label(Menu, text="Potential")
Level_Lab = ttk.Label(Menu, text="Level")



L_ = ttk.Entry(Parameters, width = 10, justify='center')
N_ = ttk.Entry(Parameters, width = 10, justify='center')
dt_ = ttk.Entry(Parameters, width = 10, justify='center')
RF_l_ = ttk.Entry(Parameters, width = 10, justify='center')
n_ = ttk.Entry(Parameters, width = 10, justify='center')




Level_ = ttk.Combobox(Menu, justify='center', textvariable=Level)
Level_['values'] = ['1', '2', '3']
Level_['state'] = 'readonly'


h_ = ttk.Combobox(Menu, justify='center', textvariable=h)
h_['values'] = ['1', str(cost.hbar)]
h_['state'] = 'readonly'


m_ = ttk.Entry(Parameters, width = 10, justify='center')
FRAME_ = ttk.Entry(Parameters, width = 10, justify='center')
FPS_ = ttk.Entry(Parameters, width = 10, justify='center')
norma_ = ttk.Entry(Parameters, width = 10, justify='center')





Potential_ = ttk.Combobox(Menu, justify='center', textvariable=Potential)
Potential_['values'] = Potential_list
Potential_['state'] = 'readonly'

ψ_ = ttk.Combobox(Menu, justify='center', textvariable=ψ_name)
ψ_['values'] = ψ_list
ψ_['state'] = 'readonly'


Potential_par = ttk.Label(Parameters, text="Potential Parameters", font = 'bold', foreground='#007FFF')

x_V_centered_Lab  = ttk.Label(Parameters, text="Potential Center")
step_x_Lab  = ttk.Label(Parameters, text="x of the step")
barrier_width_Lab  = ttk.Label(Parameters, text="Width")
RF_l_Lab  = ttk.Label(Parameters, text="l")
V0_Lab  = ttk.Label(Parameters, text="V0")
a_Lab  = ttk.Label(Parameters, text="Armonic curvature")
α_Lab  = ttk.Label(Parameters, text="α")


x0_var = tkinter.DoubleVar()
p0_var = tkinter.DoubleVar()
σ0_var = tkinter.DoubleVar()

x_V_centered_var = tkinter.DoubleVar()
step_x_var = tkinter.DoubleVar()
barrier_width_var = tkinter.DoubleVar()
V0_var = tkinter.DoubleVar()
a_var = tkinter.DoubleVar()
α_var = tkinter.DoubleVar()

x0_var.set(x0)
p0_var.set(p0)
σ0_var.set(σ0)

x_V_centered_var.set(x_V_centered_coefficient)
step_x_var.set(step_x_coefficient)
barrier_width_var.set(barrier_width_coefficient)
V0_var.set(V0_coefficient)
a_var.set(a_coefficient)
α_var.set(α_coefficient)

x0_ = ttk.Scale(Parameters,  length = 200,  from_=0, to=1, 
                            orient='horizontal' , command=update_graph,
                            variable = x0_var)

x0_RL = ttk.Scale(Parameters,  length = 200,  from_=-0.5, to=0.5, 
                            orient='horizontal' , command=update_graph,
                            variable = x0_var)

p0_ = ttk.Scale(Parameters,  length = 200,  from_=-1, to=1, 
                            orient='horizontal', command=update_graph,
                            variable = p0_var)
σ0_ = ttk.Scale(Parameters,  length = 200,  from_=0, to=0.5, 
                            orient='horizontal', command=update_graph,
                            variable = σ0_var)

x_V_centered_ = ttk.Scale(Parameters,  length = 200,  from_=0, to=1, 
                            orient='horizontal', command=update_graph,
                            variable = x_V_centered_var)
step_x_ = ttk.Scale(Parameters,  length = 200,  from_=0, to=1, 
                            orient='horizontal', command=update_graph,
                            variable = step_x_var)
barrier_width_ = ttk.Scale(Parameters,  length = 200,  from_=0, to=1, 
                            orient='horizontal', command=update_graph,
                            variable = barrier_width_var)
V0_ = ttk.Scale(Parameters,  length = 200,  from_=0 + ε, to=1 - ε, 
                            orient='horizontal', command=update_graph,
                            variable = V0_var)
a_ = ttk.Scale(Parameters,  length = 200,  from_= 0, to=1, 
                            orient='horizontal', command=update_graph,
                            variable = a_var )
α_ = ttk.Scale(Parameters,  length = 200,  from_= 0, to=1 - ε, 
                            orient='horizontal', command=update_graph,
                            variable = α_var )

x0_Scale_Lab = ttk.Label(Parameters, text = '%.2f' % x0_var.get())
p0_Scale_Lab = ttk.Label(Parameters, text = '%.2f' % p0_var.get())
σ0_Scale_Lab = ttk.Label(Parameters, text = '%.2f' % σ0_var.get())

x_V_centered_Scale_Lab = ttk.Label(Parameters, text = '%.2f' % x_V_centered_var.get())
step_x_Scale_Lab = ttk.Label(Parameters, text = '%.2f' % step_x_var.get())
barrier_width_Scale_Lab = ttk.Label(Parameters, text = '%.2f' % barrier_width_var.get())
V0_Scale_Lab = ttk.Label(Parameters, text = '%.2f' % V0_var.get())
a_Scale_Lab = ttk.Label(Parameters, text = '%.2f' % a_var.get())
α_Scale_Lab = ttk.Label(Parameters, text = '%.2f' % α_var.get())


ψ_par = ttk.Label(Parameters, text="ψ Parameters", font = 'bold', foreground='#007FFF')
ψ_Lab = ttk.Label(Menu, text="ψ0")

Video_par = ttk.Label(Parameters, text="Video Parameters", font = 'bold', foreground='#007FFF')
Video_flag = tkinter.IntVar()
Video_flag_= ttk.Checkbutton(Parameters, text="Save video", variable=Video_flag)

real_flag = tkinter.IntVar()
real_flag_= ttk.Checkbutton(Parameters, text="Show Re(ψ)", variable=real_flag ) #, command=update_graph)

imag_flag = tkinter.IntVar()
imag_flag_= ttk.Checkbutton(Parameters, text="Show Im(ψ)", variable=imag_flag ) #, command=update_graph)



x0_Lab = ttk.Label(Parameters, text="x0")
p0_Lab = ttk.Label(Parameters, text="p0")
σ0_Lab = ttk.Label(Parameters, text="σ0")
n_Lab  = ttk.Label(Parameters, text="Armonic level")

#Bind method
ψ_.bind('<<ComboboxSelected>>', lambda value: [update_graph(value), update_menu_ψ(value)])
h_.bind('<<ComboboxSelected>>', update_graph)
Potential_.bind('<<ComboboxSelected>>', lambda value: [update_graph(value), update_menu_pot(value)])


L_.bind('<Return>', update_graph)
N_.bind('<Return>', update_graph)
dt_.bind('<Return>', update_graph)
RF_l_.bind('<Return>', update_graph)
n_.bind('<Return>', update_graph)

m_.bind('<Return>', update_graph)
FRAME_.bind('<Return>', update_graph)
FPS_.bind('<Return>', update_graph)
norma_.bind('<Return>', update_graph)

#real_flag_.bind('<Activate>', update_graph)
#imag_flag_.bind('<Button-1>', update_graph)

#real_flag.trace('w', update_graph)
#imag_flag.trace('w', update_graph)

Insert_Lab = ttk.Label(Parameters, text='Insert V(x)')
Insert_ = ttk.Entry(Parameters, width = 25, justify='center')
Insert_.bind('<Return>', update_graph)

Ins_LIB_Lab = ttk.Label(Parameters, text='Insert library')
Ins_LIB_ = ttk.Entry(Parameters, width = 25, justify='center')



#Advance Parameters

var_E = tkinter.IntVar(ad)
var_norma = tkinter.IntVar(ad)
var_ST = tkinter.IntVar(ad)

sim_ = ttk.Combobox(ad_par, justify='center', textvariable=sim)
sim_['values'] = sim_list
sim_['state'] = 'read_only'
sim_.grid(padx = 5, pady = 3, column=0, row=0)


debug_E_ = ttk.Checkbutton(ad_par, text = 'Energy check', variable = var_E)
debug_norma_ = ttk.Checkbutton(ad_par, text = 'Norm check', variable = var_norma)
STOP_BOUNDARIES_ = ttk.Checkbutton(ad_par, text = 'Boundaries suppressor', variable = var_ST)

ST_BD_COEFF_lab = ttk.Label(ad_par, text = 'Boundaries suppressor coefficient')
ST_BD_COEFF_ = ttk.Entry(ad_par, width = 10, justify='center')

sim_.current(0)
ST_BD_COEFF_.insert(0, str(ST_BD_COEFF))  

ad_save = ttk.Button(ad_par, text = 'Save and exit', command = a_save)


###########
# Default #
###########

L_.insert(0, str(L))
N_.insert(0, str(N))
dt_.insert(0, str(dt))
RF_l_.insert(0, str(RF_l))
n_.insert(0, str(n))

x0_.set(x0_coefficient)
p0_.set(p0_coefficient)
σ0_.set(σ0_coefficient)

Level_.current(1)
h_.current(0)
ψ_.current(0)
Potential_.current(0)


m_.insert(0, str(m))
FRAME_.insert(0, str(FRAME))
FPS_.insert(0, str(FPS))
norma_.insert(0, str(STD_NORMA))

x_V_centered_.set(x_V_centered_coefficient)
step_x_.set(step_x_coefficient)
barrier_width_.set(barrier_width_coefficient)
V0_.set(V0_coefficient)
a_.set(a_coefficient)
α_.set(α_coefficient)

Ins_LIB_.insert(0, 'numpy')


########
# Page #
########


Sim_par.grid( padx = 5, pady = 3, column=0, row=0)

L_Lab.grid( padx = 5, pady = 3, column=0, row=2)
N_Lab.grid( padx = 5, pady = 3, column=0, row=3)
dt_Lab.grid( padx = 5, pady = 3, column=0, row=4)

m_Lab.grid( padx = 5, pady = 3, column=0, row=7)
FRAME_Lab.grid( padx = 5, pady = 3, column=0, row=8)
norma_Lab.grid( padx = 5, pady = 3, column=0, row=9)


L_.grid( padx = 5, pady = 3, column=1, row=2)
N_.grid( padx = 5, pady = 3, column=1, row=3)
dt_.grid( padx = 5, pady = 3, column=1, row=4)


m_.grid( padx = 5, pady = 3, column=1, row=7)
FRAME_.grid( padx = 5, pady = 3, column=1, row=8)
norma_.grid( padx = 5, pady = 3, column=1, row=9)

ψ_par.grid( padx = 5, pady = 3, column=0, row=18)

if ψ_.get() ==  ψ_list[0]:
    x0_Lab.grid( padx = 5, pady = 3, column=0, row=20)
    p0_Lab.grid( padx = 5, pady = 3, column=0, row=21)
    σ0_Lab.grid( padx = 5, pady = 3, column=0, row=22)
    

    x0_.grid( padx = 5, pady = 3, column=1, row=20)
    p0_.grid( padx = 5, pady = 3, column=1, row=21)
    σ0_.grid( padx = 5, pady = 3, column=1, row=22)

    x0_Scale_Lab.grid( padx = 5, pady = 3, column=2, row=20)
    p0_Scale_Lab.grid( padx = 5, pady = 3, column=2, row=21)
    σ0_Scale_Lab.grid( padx = 5, pady = 3, column=2, row=22)

elif ψ_.get() ==  ψ_list[1]:
    x0_Lab.grid( padx = 5, pady = 3, column=0, row=20)
    x0_.grid( padx = 5, pady = 3, column=1, row=20)
    x0_Scale_Lab.grid( padx = 5, pady = 3, column=2, row=20)

    n_Lab.grid( padx = 5, pady = 3, column=0, row=23)
    n_.grid( padx = 5, pady = 3, column=1, row=23)

FPS_Lab.grid( padx = 5, pady = 3, column=0, row=25)
FPS_.grid( padx = 5, pady = 3, column=1, row=25)

Video_par.grid( padx = 5, pady = 3, column=0, row=24)
Video_flag_.grid( padx = 5, pady = 3, column=0, row=26)

real_flag_.grid( padx = 5, pady = 3, column=0, row=27)

imag_flag_.grid( padx = 5, pady = 3, column=1, row=27)

Adv.grid( padx = 5, pady = 3, column=0, row=28)

Sel_path.grid( padx = 5, pady = 3, column=0, row=0)
Save.grid( padx = 5, pady = 3, column=1, row=0)
Start.grid( padx = 5, pady = 3, column=2, row=0)
Quit.grid( padx = 5, pady = 3, column=3, row=0)

Level_Lab.grid( padx = 5, pady = 3, column=2, row=1)
Level_.grid( padx = 5, pady = 3, column=3, row=1)

h_.grid( padx = 5, pady = 3, column=3, row=0)
h_Lab.grid( padx = 5, pady = 3, column=2, row=0)

ψ_Lab.grid( padx = 5, pady = 3, column=0, row=1)
ψ_.grid( padx = 5, pady = 3, column=1, row=1)

Potential_.grid( padx = 5, pady = 3, column=1, row=0)
Potential_Lab.grid( padx = 5, pady = 3, column=0, row=0)

canvas.get_tk_widget().grid()
toolbar.grid(column = 0, row = 1)

canvas_p.get_tk_widget().grid(column = 0, row = 0)
toolbar_p.grid(column = 0, row = 1)


root.protocol("WM_DELETE_WINDOW", quit)
root.mainloop()
