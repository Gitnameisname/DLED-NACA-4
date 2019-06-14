# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 14:32:55 2018

@author: CK_DL2
"""

"""
Info
=====
Read Input values for
Result_Analysis
"""
import os
import sys
import numpy as np

def Rinput(name_inputfile,no):
    
    direct_code=os.path.dirname(os.path.realpath('__file__'))
    sys.path.append(direct_code)
    direct_Analyze=os.path.join(direct_code,'Analyzed DB')
    direct_inputfile = os.path.join(direct_Analyze,name_inputfile)
    
    f=open(direct_inputfile,'r')
    txt=f.readlines()
    
    max_line=np.shape(txt)[0]
    
    ## Get training info ##
    i=0
    printswitch=0
    case=[]
    Value_info_folder=[]
    while  i < max_line:
        l=txt[i]
        
        if l[0] == str('$'):
            printswitch=0
        if printswitch==1:
            temp=l.split('\t')
            case.append(temp[0].split(':')[0])
            Value_info_folder.append(temp[1].split('\n')[0])
#            print(l)
        if l[0:2] == str('#'+str(no)):
#            print(l)
            printswitch=1
    
        i += 1
        
    ## Make variable about training information ##

    return Value_info_folder