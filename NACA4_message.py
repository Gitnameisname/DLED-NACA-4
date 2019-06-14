# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 00:07:53 2018

@author: K-AI_LAB
"""

"""
Info
=====
Send the message
"""

import os, sys
from inspect import getframeinfo, stack

direct_work=os.getcwd()
# Find Code Directory
direct_code=os.path.dirname(os.path.realpath(__file__))
# If the code directory is not in PATH, add directory to import function
if direct_code not in sys.path:
    sys.path.append(direct_code)

import NACA4_Log as log

def errorinfo(exc_type, exc_obj, exc_tb, e):

    name_code = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1].split('.')[0]
    print()
    print(exc_type)
    log.log('\n! Error !')
    reply = str('{}:{} - {} '.format(name_code, exc_tb.tb_lineno, e))
    print(reply)
    
    return reply

def debuginfo(message):
    
    if message != None:
        caller = getframeinfo(stack()[1][0])
        if str('\\') in caller.filename:
            name_code = caller.filename.split('\\')[-1].split('.')[0]
        elif str('/') in caller.filename:
            name_code = caller.filename.split('/')[-1].split('.')[0]
        
        reply = str('{}:{} - {}'.format(name_code, caller.lineno, message))
        print(reply)
        log.log(reply)
        
        return caller
    
    elif message == None:
        log.log(message)
        
