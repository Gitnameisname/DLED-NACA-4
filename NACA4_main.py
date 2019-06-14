# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 12:02:50 2017

@author: K_LAB
"""

"""
Info
=====
This Code for running ANN_NACA4
Use the version over than 08
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import copy

direct_code=os.path.dirname(os.path.realpath('__file__'))
direct_DB=os.path.join(direct_code,"DB")
sys.path.append(direct_code)

import NACA4_DB
import NACA4_Training
import NACA4_Rinput as Rin
import NACA4_Log as Log
import NACA4_message as msg

Log.log_clear()
Log.log(None)


direct_input = os.path.join(direct_code,'Input')
var, val = Rin.Rinput(direct_input,'input.txt',1)

# Read Training Input Parameters #
loop_start = copy.copy(int(val[0]))
loop = copy.copy(loop_start)
Max_loop = copy.copy(int(val[1]))
Max_epoch = copy.copy(int(val[2]))
mini_batch_size = copy.copy(int(val[3]))
Init_AFDB = copy.copy(int(val[4]))  # 0: Do not need to make Initial Airfoil Database
                                    # 1: Need to make Initial Airfoil Database
re_build = copy.copy(int(val[5]))   # 0: Do not re-analyze after the training will be finished
                                    # 1: Re-analyze after the training will be finished

DBname_Init_AF="Airfoil DB"+str(loop)+".txt"
DBname_Total="Airfoil DB"+str(loop + 1)+".txt"
plt.close('all')

var2, val2 = Rin.Rinput(direct_input,'input.txt',2)

time_start = time.time()
if loop == 1 and Init_AFDB ==1:
    msg.debuginfo('Building Initial DB')
    NACA4_DB.Init_configDB()
    NACA4_DB.performanceDB("Initial config DB.txt",DBname_Init_AF)

while loop < Max_loop+1:
    ##----------------------------- Check Statement -----------------------------##
    if plt.fignum_exists(1):
        plt.close()
    
    ##-------------------------- Init AF DB generation --------------------------##
    if not os.path.isfile(os.path.join(direct_DB,DBname_Init_AF)):
        message = 'Cannot found the initial configuration DB'
        msg.debuginfo(message)
        NACA4_DB.Init_configDB()
        NACA4_DB.performanceDB("Initial config DB.txt",DBname_Init_AF)
    
    ##--------------------------------- Training --------------------------------##
    # Setting the DB name
    print("## Training Loop: "+str(loop)+" Start\n\n")
    NACA4_Training.Training(loop,Max_epoch,mini_batch_size, re_build)
    
    message = '>> Training Loop: '+str(loop)+' was Finished\n\n'
    msg.debuginfo(message)
    
    loop += 1
    
time_run = time.time() - time_start