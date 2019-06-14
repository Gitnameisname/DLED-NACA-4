# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 23:39:02 2018

@author: K-AI_LAB
"""

"""
Info
=====
DLED_CST log file
"""

import numpy as np
import os
import sys
import datetime
import NACA4_message as msg
   
direct_work=os.getcwd()
# Find Code Directory
direct_code=os.path.dirname(os.path.realpath(__file__))
# If the code directory is not in PATH, add directory to import function
if direct_code not in sys.path:
    sys.path.append(direct_code)
direct_log=os.path.join(direct_code,"log")
direct_logfile = os.path.join(direct_log,'log.txt')

if os.path.isfile(direct_logfile):
    os.remove(direct_logfile)

def log(get_log):
    if not os.path.isfile(direct_logfile):
        f = open(direct_logfile,'w')
        dt = datetime.datetime.now()
            
        
        try:
            f.write('Created on {}-{}-{} {}:{}:{}\n'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second))
            f.write('Deep Learning Engineering Design: CST Airfoil\n')
            f.write('Cheol-Kyun Choi\n')
            f.write('====================================================================================================\n')
            f.write('# input.txt\n#\n')
                    
            inputfile = open(os.path.join(direct_code,'Input','input.txt'))
            lines = inputfile.readlines()
            
            for i in range(len(lines)):
                f.write(lines[i])
            
            f.write('\n#input.txt end\n')
            f.write('====================================================================================================\n')
            f.close()
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            msg_get = msg.errorinfo(exc_type, exc_obj, exc_tb, e)
            f.write('{}\n'.format(exc_type))
            f.write(msg_get)
            f.close()
    
    if get_log != None:
        f = open(direct_logfile,'a')
        f.write(get_log)
        f.write('\n')
        f.close()
        
    elif get_log == None:
        f = open(direct_logfile,'a')
        f.write('\n')
        f.close()

def log_clear():
    if os.path.isfile(direct_logfile):
        os.remove(direct_logfile)