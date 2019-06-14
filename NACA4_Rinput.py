# -*- coding: utf-8 -*-
"""
Created on Fri May  4 13:56:01 2018

@author: K-AI_LAB
"""

"""
Info
=====
Read hyperparameter of the training model
"""

import os
import numpy as np
direct_code = os.path.dirname(os.path.realpath('__file__'))
def Rinput(name_folder,name_file,no):

    direct_code = os.path.dirname(os.path.realpath('__file__'))
    direct_file = os.path.join(direct_code,name_folder,name_file)
    
    f = open(direct_file)
    
    text = f.readlines()
    
    max_line = np.shape(text)[0]
    
    ## Get the information ##
    
    i = 0
    printswitch = 0
    variable = []
    value = []
    
    while i < max_line:
        line = text[i]
        
        if line[0] == str('$'):
            printswitch = 0
            
        if printswitch == 1:
            temp = line.split('\n')[0]
            temp = temp.split('\t')
            temp = list(filter(None,temp))
            if len(temp) == 2:
                variable.append(temp[0].split(':')[0])
                value.append(temp[-1])
        if line[0:2] == str('#'+str(no)):
            printswitch = 1
            
        i += 1
        
        
        
    return variable, value

def Conv_vals(variable, value):
    
    return_val=[]
        
    i=0
    for i in range(len(variable)):
        vars()[variable[i]] = value[i].split(',')
        j=0
        for j in range(len(vars()[variable[i]])):
            vars()[variable[i]][j] = vars()[variable[i]][j].split(' ')[-1]
        j=0
        for j in range(len(vars()[variable[i]])):
            vars()[variable[i]][j]=float(vars()[variable[i]][j])
        return_val.append(vars()[variable[i]])
        
    return return_val
    
               
if __name__ == "__main__":
    direct_saved=os.path.join(direct_code,"saved")
    var, val = Rinput(direct_saved,'Normalization20.txt',1)
    
    
    return_val=[]
        
    i=0
    for i in range(len(var)):
        vars()[var[i]] = val[i].split(',')
        j=0
        for j in range(len(vars()[var[i]])):
            vars()[var[i]][j] = vars()[var[i]][j].split(' ')[-1]
        j=0
        for j in range(len(vars()[var[i]])):
            vars()[var[i]][j]=float(vars()[var[i]][j])
        return_val.append(vars()[var[i]])
    
#    retun_val = Conv_vals(var,val)
    
#    return_val = Conv_vals(var, val)
#    
#    i=0
#    for i in range(len(var)):
#        vars()[var[i]] = val[i].split(',')        
#        j=0
#        for j in range(len(vars()[var[i]])):
#            vars()[var[i]][j]=float(vars()[var[i]][j])
#    
#    wu_max =  vars()[var[0]]
#    wu_min =  vars()[var[1]]
#    wl_max =  vars()[var[2]]
#    wl_min =  vars()[var[3]]
#    
#    wu_step = vars()[var[4]] 
#    wl_step = vars()[var[5]]

