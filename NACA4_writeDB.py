# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 17:04:03 2017

@author: K_LAB
"""

"""
Info
=====
Generate Data Base
Input value is Parameter and Aerodynamic Coefficient
Parameter has 3 value and Aerodynmic has 5 value
"""
import numpy as np
import os
import sys
direct_work=os.getcwd()
# Find Code Directory
direct_code=os.path.dirname(os.path.realpath(__file__))
# If the code directory is not in PATH, add directory to import function
if direct_code not in sys.path:
    sys.path.append(direct_code)
direct_DB=os.path.join(direct_code,"DB")

import NACA4_Log as Log
import NACA4_message as msg


def InitConfigDB(DBname,config):
    npsize=np.shape(config)
    i=0
    testfiledirect=os.path.join(direct_DB,DBname)
    f=open(testfiledirect,'w')
    f.write('Initial configuration DB\n')
    f.write('         c        Lc         t\n')
    f.write(' ========= ========= =========\n')    
    f.close()
    
    f=open(testfiledirect,'a')
    while i < npsize[0]:
        f.write("{:10d}{:10d}{:10d}\n"\
                .format(int(config[i,0]),int(config[i,1]),int(config[i,2])))
        i=i+1
    f.close()

    
def ConfigDB(DBname,testpara):
    npsize=np.shape(testpara)
    i=0
    testfiledirect=os.path.join(direct_DB,DBname)
    f=open(testfiledirect,'w')
    f.write('Test Result DB\n')
    f.write('         c        Lc         t\n')
    f.write(' ========= ========= =========\n')    
    f.close()
    
    f=open(testfiledirect,'a')
    while i < npsize[0]:
        f.write("{:10.5f}{:10.5f}{:10.5f}\n"\
                .format(testpara[i,0],testpara[i,1],testpara[i,2]))
        i=i+1
    f.close()
    
def NACADB(DBname,para,Aero):
    direct_file=os.path.join(direct_DB,DBname)
    if os.path.isfile(direct_file):
        os.remove(direct_file)
    
    iter_max = np.shape(para)[0]    
    
    # Initialize the DB file format
    if not os.path.isfile(direct_file):
        message = 'Create DB as "{}"'.format(DBname)
        msg.debuginfo(message)
        try:
            f=open(direct_file,'w')    
            f.write('NACA 4 digit {}\n'.format(DBname))
            f.write('         c        Lc         t       Cl0      Cla0       Cd0       Cm0      Cma0\n')
            f.write(' ========= ========= ========= ========= ========= ========= ========= =========\n')
            f.close()
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            message = msg.errorinfo(exc_type, exc_obj, exc_tb, e)
            Log.log(message)
            f.close()
            return -1

    try:
        i=0
        f=open(direct_file,'a')
        for i in range(iter_max):
            f.write("{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}\n"\
                    .format(para[i][0],para[i][1],para[i][2],\
                            Aero[i][0],Aero[i][1],Aero[i][2],Aero[i][3],Aero[i][4]))
        f.close()
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        message = msg.errorinfo(exc_type, exc_obj, exc_tb, e)
        Log.log(message)
        f.close()
        return -1