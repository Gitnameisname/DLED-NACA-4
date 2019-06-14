# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 17:04:18 2018

@author: K_LAB
"""


import subprocess as sp
import numpy as np
import time
import os
import sys
import multiprocessing
from multiprocessing.pool import ThreadPool

direct_work=os.getcwd()
# Find Code Directory
direct_code=os.path.dirname(os.path.realpath(__file__))
# If the code directory is not in PATH, add directory to import function
if direct_code not in sys.path:
    sys.path.append(direct_code)
direct_DB=os.path.join(direct_code,"DB")
    
import NACA4_Rinput as Rin
import NACA4_CalADC
import NACA4_RunXfoil
import NACA4_writeDB
import ReadDB
import NACA4_Config
import NACA4_Log as Log
import NACA4_message as msg

direct_file=os.path.join(direct_DB,"Initial config DB.txt")

def Init_configDB():
    
    if os.path.isfile(direct_file):
        os.remove(direct_file)
        
    on_off = 1
    config = np.zeros([0,3])
    
    # at C = 0 , LC = 0
    # at C != 0, LC != 0
    C = 0
    LC = 0
    T = 1
    
    C_max=9
    LC_max=9
    T_max=20
    C_min = 0
    LC_min = 1
    T_min = 1
    
    while on_off ==1:
        
        config = np.append(config,np.array([[C, LC, T]]),axis=0)
        
        T += 1
        
        if T > T_max:
            LC += 1
            T = T_min
        
        if LC > LC_max:
            C += 1
            LC = LC_min
        
        if C > C_max:
            on_off = 0
            
    NACA4_writeDB.InitConfigDB(direct_file,config)
    message = 'Initial configuration database was constructed'
    msg.debuginfo(message)
    
def performanceDB(DBname_config,DBname_airfoil):
    
    no_thread = multiprocessing.cpu_count()
    message = str("Maximum number of thread: {}".format(no_thread))
    msg.debuginfo(message)
    no_proc = no_thread-2
    message = str("Number of using thread: {}".format(int(no_proc)))
    msg.debuginfo(message)
    message = str("XFOIL Analyze Start")
    msg.debuginfo(message)
    
    time_start = time.time()

    direct_data=os.path.join(direct_DB,"Temp")
    os.chdir(direct_data)
    
    config = ReadDB.Read(DBname_config)
    split_config_DB=np.array_split(config,no_proc)
    no_data = 0
    
    i = 0
    for i in range(no_proc):
        name_partial_DB = 'partial_DB'+str(i)
        vars()[name_partial_DB] = np.zeros([0,8])
        name_analized_index = 'analized_index' + str(i)
        vars()[name_analized_index] = np.zeros([0])
        
        name_split_config_DB = 'split_DB'+str(i)
        vars()[name_split_config_DB]=np.zeros([0,3])
        vars()[name_split_config_DB]=np.append(vars()[name_split_config_DB],split_config_DB[i], axis=0)
    
    i = 0
    size_split_DB=[]
    for i in range(no_proc):
        name_split_config_DB = 'split_DB'+str(i)
        size_split_DB.append(np.shape(vars()[name_split_config_DB])[0])
        if size_split_DB[i] != size_split_DB[i-1]:
            last_max_DB = i
            
        
    max_iter = max(size_split_DB)
    min_iter = min(size_split_DB)
    
    var, val = Rin.Rinput(os.path.join(direct_code,'Input'),'Xfoil setting.txt',1)
    
    Point = int(val[0])
    print('##############################')
    while no_data < max_iter:
        
        if no_data > min_iter - 1:
            no_proc = last_max_DB
        
        
        progress = ('Process: '+str(no_data+1)+'/'+str(max_iter))
        sys.stdout.write('\r'+progress)
        
        i=0
        for i in range(no_proc):
            name_split_config_DB = 'split_DB'+str(i)
            
            C = vars()[name_split_config_DB][no_data][0]
            LC = vars()[name_split_config_DB][no_data][1]
            T = vars()[name_split_config_DB][no_data][2]
            
            NACA4_Config.NACA4(Point, C, LC, T,direct_data,i)
        
        # Xfoil analysis with multiprocessing
        pool = ThreadPool(multiprocessing.cpu_count())
        i=0
        for i in range(no_proc):
#            print('Thread'+str(i)+': run Xfoil')
            pool.apply_async(NACA4_RunXfoil.runXFOIL,(str(i)))
#            print('Thread'+str(i)+': finished')
        
        pool.close()
        pool.join()
#        print('Xfoil analysis was finished')
        
                
#        pool = ThreadPool(multiprocessing.cpu_count())
        i=0    
        for i in range(no_proc):
#            print('Thread'+str(i)+': Cal Aerodynamic')
            
            name_split_config_DB = 'split_DB'+str(i)
            name_partial_DB = 'partial_DB'+str(i)
            
            para_aero = NACA4_CalADC.run(i)
            
            if para_aero.size > 0:
                temp_data = np.append(np.expand_dims(vars()[name_split_config_DB][no_data],axis=0),para_aero,axis=1)
                vars()[name_partial_DB] = np.append(vars()[name_partial_DB],temp_data,axis=0)
                vars()[name_analized_index] = np.append(vars()[name_analized_index], 1)
                
            else:
                vars()[name_analized_index] = np.append(vars()[name_analized_index], 0)
#                print('Analyzed\n')
#            else:
#                print('Not analyzed\n')
#        print('Performance calculation was finished\n')
    
        no_data += 1
  
    time_run = time.time() - time_start
    message = str('runtime - re-Analysis: {:.2f} s, {:.2f} min, {:.2f} hr'.format(time_run,time_run/60, time_run/3600))
    msg.debuginfo(message)
    Full_DB = np.zeros([0,8])
    
    i=0
    no_proc = no_thread-2
    print(str(no_proc) +':186')
    for i in range(no_proc):
        name_partial_DB = 'partial_DB'+str(i)
        Full_DB = np.append(Full_DB,vars()[name_partial_DB],axis=0)
        
    
        
    NACA4_writeDB.NACADB(DBname_airfoil,Full_DB[:,0:3],Full_DB[:,3:])
   

if __name__ == "__main__":
#    Init_configDB()
    performanceDB("Predict Config"+str(2)+".txt","Additional DB"+str(2)+".txt" )
    