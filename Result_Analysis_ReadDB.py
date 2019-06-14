# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 12:48:40 2017

@author: K_LAB
"""

"""
Info
=====
Read DB file and return as Array
"""

import os
import numpy as np

Name_mainfolder='ANN NACA4_V02'
Path_initial=os.path.dirname(os.path.realpath('__file__'))
Path_base=os.path.basename(os.path.normpath(Path_initial))

if Path_base == Name_mainfolder:
    codedirect = Path_initial  
else:     
    Path_fix=os.path.join(Path_initial[:Path_initial.find(Name_mainfolder)],Name_mainfolder)
    codedirect = Path_fix


def Read(direct_file):
    ## Initialize function ##
    # filename is the name of reading file with extension
    
    
    """def Read(self):"""
    f=open(direct_file,'r')
    i=0
    while True:
        # Read line by line
        line = f.readline()
        # Second line is the label of Data
        if i == 1:
            C_List=str(line)
            temp=C_List.split(' ')
            label=[]
            j=0
            # Cut the label by label
            while j < len(temp):
                if temp[j]=='':
                    j=j+1
                # Cut \n of last label
                else:
                    if j == len(temp)-1:
                        temp[j]=temp[j].split('\n')[0]
                        label.append(temp[j])
                        # Create variables using the name of label
                        vars()[temp[j]]=[]
                    else:
                        # Create variables using the name of label
                        label.append(temp[j])
                        vars()[temp[j]]=[]
                    j=j+1
        
        # Skip the 3rd line >> this is the border line between label and values
        if i > 2:
            C=str(line)
            temp=C.split(' ')
            j=0
            k=0
            while j < len(temp):
                if temp[j]=='':
                    j=j+1
                # Cut the each value
                else:
                    if j == len(temp)-1:
                        k=len(label)-1
                        temp[j]=temp[j].split('\n')[0]
                        # Create variables using the name of label
                        vars()[label[k]].append(temp[j])
                    else:
                        # Create variables using the name of label
                        vars()[label[k]].append(temp[j])
                        k=k+1
                    j=j+1
        # if the line end, break the roop
        if not line:
            break
        i=i+1
    f.close()
    
    # Initialize the return matrix
    mat=np.zeros([len(vars()[label[0]]),len(label)])
    
    # Fill the matrix
    for i in range(len(label)):
        mat[:,i]=vars()[label[i]]
    
    return mat