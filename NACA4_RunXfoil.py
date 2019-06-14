# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 13:31:26 2017

@author: K_LAB
"""

"""
Info
====
Run the Xfoil Automatically
"""
 
import subprocess as sp
import numpy as np
import time
import os
import sys
from multiprocessing.pool import ThreadPool

direct_work=os.getcwd()
# Find Code Directory
direct_code=os.path.dirname(os.path.realpath(__file__))
# If the code directory is not in PATH, add directory to import function
if direct_code not in sys.path:
    sys.path.append(direct_code)
direct_DB=os.path.join(direct_code,"DB")
direct_RR=os.path.join(direct_code,"Result") ## Directory of Restore Result

import NACA4_Rinput as Rin

AoA = [-1, 0, 1]

def runXFOIL(no_proc):

    var, val = Rin.Rinput(os.path.join(direct_code,'Input'),'Xfoil setting.txt',1)
    
    Iter = val[1]
    Re = val[2]
    M = val[3]
    timelim = float(val[4])
    
    direct_data=os.path.join(direct_DB,"Temp")
    os.chdir(direct_data)
    Xfoildirect=os.path.join(direct_DB,"Temp","xfoil.exe")
    
 
    file="Temp data_"+str(no_proc)+".txt"
    direct_file=os.path.join(direct_data,file)
    # If there are same savefile, remove it before save
    if os.path.isfile(direct_file):
        os.remove(direct_file)
    name_file = "Temp Config"+str(no_proc)+".txt"
    # Popen: Excute Subprocess
    # cwd = Current Working Directory
    p=sp.Popen(Xfoildirect,shell=False,stdin=sp.PIPE,stdout=sp.PIPE,stderr=None, encoding='utf-8')
    # Command to Xfoil    
    command=str("plop\n g\n \n"+\
                "load\n" + name_file+"\n" +\
                "oper\n" +\
                "iter\n" + Iter + "\n" + \
                "visc\n" + Re + "\n" + \
                "m\n" + M + "\n" +\
                "pacc\n" + file + "\n \n" +\
                "a\n" + str(AoA[0]) + "\n" +\
                "a\n" + str(AoA[1]) + "\n" +\
                "a\n" + str(AoA[2]) + "\n" +\
                "\n \n \n quit\n")
    try:
        com=p.communicate(command,timeout=timelim)
        p.kill()
    
    # If xfoil run over the [timelim] skip to next step
    except sp.TimeoutExpired:
        p.kill()
#        print("오래걸림") 
    # For the case that program does not killed, call p.kill() one more time
    p.kill()
    
    return com

def runXFOIL_Restore():

    var, val = Rin.Rinput(os.path.join(direct_code,'Input'),'Xfoil setting.txt',1)
    
    Iter = val[1]
    Re = val[2]
    M = val[3]
    timelim = float(val[4])

    os.chdir(direct_RR)
    Xfoildirect=os.path.join(direct_RR,"xfoil.exe")
    
 
    file="Predicted NACA Aero.txt"
    direct_file=os.path.join(direct_RR,file)
    # If there are same savefile, remove it before save
    if os.path.isfile(direct_file):
        os.remove(direct_file)
    name_file = "Predicted NACA.txt"
    # Popen: Excute Subprocess
    # cwd = Current Working Directory
    p=sp.Popen(Xfoildirect,shell=False,stdin=sp.PIPE,stdout=sp.PIPE,stderr=None, encoding='utf-8')
    # Command to Xfoil    
    command=str("plop\n g\n \n"+\
                "load\n" + name_file+"\n" +\
                "oper\n" +\
                "iter\n" + Iter + "\n" + \
                "visc\n" + Re + "\n" + \
                "m\n" + M + "\n" +\
                "pacc\n" + file + "\n \n" +\
                "a\n" + str(AoA[0]) + "\n" +\
                "a\n" + str(AoA[1]) + "\n" +\
                "a\n" + str(AoA[2]) + "\n" +\
                "\n \n \n quit\n")
    try:
        com=p.communicate(command,timeout=timelim)
        p.kill()
    
    # If xfoil run over the [timelim] skip to next step
    except sp.TimeoutExpired:
        p.kill()
#        print("오래걸림") 
    # For the case that program does not killed, call p.kill() one more time
    p.kill()
    
#    return com

if __name__=="__main__":

    result = runXFOIL_Restore()
#    