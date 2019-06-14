# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:54:45 2017

@author: K_LAB
"""

"""
Info
=====
Calculate Aerodynamic Characteristics
Cl0, Cla_0, Cd0, Cm0, Cma_0
"""

import numpy as np
import os
import sys
import time

direct_work=os.getcwd()
# Find Code Directory
direct_code=os.path.dirname(os.path.realpath(__file__))
# If the code directory is not in PATH, add directory to import function
if direct_code not in sys.path:
    sys.path.append(direct_code)
direct_DB=os.path.join(direct_code,"DB")
direct_data=os.path.join(direct_DB,"Temp")
direct_RR=os.path.join(direct_code,"Result") # Directory of Restore Result

def run(no_proc):
    
    ## Read File ##
    # =========== #
    # Read file and collect the aerodynamic data
    # Open file with 'r' - Read only
    name_aero = 'Temp data_' + str(no_proc) + ".txt"
    direct_file = os.path.join(direct_data, name_aero)
#    wait = 0
#    
#    # Wait until Xfoil run over
#    while wait == 0:
#        if not os.path.isfile(direct_file):
#            time.sleep(1)
#        else:
#            print('process'+str(no_proc)+': data file found')
##            time.sleep(0.1)
#            wait = 1
        
    f=open(direct_file,'r')
    i=0
    # Coefficient is recorded from the 13th line
    start_line=13
    while True:
        line = f.readline()
        i=i+1
        # 11th line is label
        if i == 11:
            C_List=str(line)
            temp=C_List.split(' ')
            label=[]
            j=0
            # Cut the label
            while j < len(temp):
                if temp[j]=='':
                    j=j+1
                else:
                    # last label has \n, so cut that value
                    if j == len(temp)-1:
                        temp[j]=temp[j].split('\n')[0]
                        label.append(temp[j])
                        vars()[temp[j]]=[]
                    else:
                        label.append(temp[j])
                        vars()[temp[j]]=[]
                    j=j+1
        # Read Coefficients
        if i > start_line - 1:
            C=str(line) # Coefficient
            temp=C.split(' ')
            j=0
            k=0
            while j < len(temp):
                if temp[j]=='':
                    j=j+1
                else:
                    if j == len(temp)-1:
                        k=len(label)-1
                        temp[j]=temp[j].split('\n')[0]
                        vars()[label[k]].append(temp[j])
                    else:
                        vars()[label[k]].append(temp[j])
                        k=k+1
                    j=j+1
        if not line:
            break    
    f.close()      
    
    ## Calculate Aerodynamic Characteristics ##
    # ======================================= #
    # Save the infomations at matrix
    # Initialize the matrix        
    mat=np.zeros([len(vars()[label[0]]),len(label)])
    
    # Record labels
    for i in range(0,len(label),1):
        mat[:,i]=vars()[label[i]]
    
    # There are two possible case
    # 1. File has the value at AoA 0 degree
    # 2. File does not have the value at AoA 0 degree
    
    # If AoA 0 is exist (Case1) 
    if 0 in mat[:,0]:
        # Find the index of AoA 0
        zero_P=np.where(mat[:,0] == 0)[0][0]
        size=np.shape(mat)
        
        # If there are AoA 0 + step, Calculate by zero, zero + step values
        if size[0]-1 > zero_P:
            Chara=np.zeros([1,5])
            Chara[0][0]=mat[zero_P,1]
            Chara[0][1]=(mat[zero_P+1,1]-mat[zero_P,1])/(mat[zero_P+1,0]-mat[zero_P,0])
            Chara[0][2]=mat[zero_P,2]
            Chara[0][3]=mat[zero_P,4]
            Chara[0][4]=(mat[zero_P+1,4]-mat[zero_P,4])/(mat[zero_P+1,0]-mat[zero_P,0])
        # If AoA 0 - step does not exist, Calculate by zero - step, zero values
        elif size[0] > 1:
            Chara=np.zeros([1,5])
            Chara[0][0]=mat[zero_P,1]
            Chara[0][1]=(mat[zero_P,1]-mat[zero_P-1,1])/(mat[zero_P,0]-mat[zero_P-1,0])
            Chara[0][2]=mat[zero_P,2]
            Chara[0][3]=mat[zero_P,4]
            Chara[0][4]=(mat[zero_P,4]-mat[zero_P-1,4])/(mat[zero_P,0]-mat[zero_P-1,0])
        else:
            Chara=np.array([])
        
        
    # If AoA 0 does not exist (Case2)
    # Use the values near the AoA 0
    elif np.shape(mat)[0] > 2:
        AoA=np.abs(mat[:,0])
        ref_P=[]
        # Find the index of nearest value from AoA 0
        ref_P.append(np.where(AoA==min(AoA))[0][0])
        # If the 0 deg AoA does not exist, Calculate by side values
        # If the min value is positive, use the before step value
        # that value is must the nearest negative value from AoA zero
        if mat[ref_P[0],0] > 0:
            ref_P.append(ref_P[0]-1)
            ref_P.sort()
        # If the min value is negative, use the next step value
        # that value is must the nearest positive value from AoA zero
        elif mat[ref_P[0],0] < 0:
            ref_P.append(ref_P[0]+1)
            ref_P.sort()
        
        Chara=np.zeros([1,5])
        Chara[0][0]=-1*mat[ref_P[0],0]*((mat[ref_P[1],1]-mat[ref_P[0],1]))/(mat[ref_P[1],0]-mat[ref_P[0],0])+mat[ref_P[0],1]
        Chara[0][1]=(mat[ref_P[1],1]-mat[ref_P[0],1])/(mat[ref_P[1],0]-mat[ref_P[0],0])
        Chara[0][2]=-1*mat[ref_P[0],0]*((mat[ref_P[1],2]-mat[ref_P[0],2]))/(mat[ref_P[1],0]-mat[ref_P[0],0])+mat[ref_P[0],2]
        Chara[0][3]=-1*mat[ref_P[0],0]*((mat[ref_P[1],4]-mat[ref_P[0],4]))/(mat[ref_P[1],0]-mat[ref_P[0],0])+mat[ref_P[0],4]
        Chara[0][4]=(mat[ref_P[1],4]-mat[ref_P[0],4])/(mat[ref_P[1],0]-mat[ref_P[0],0])
    
    else:
        Chara=np.array([])
  
    return Chara

def run_Restore():
    
    ## Read File ##
    # =========== #
    # Read file and collect the aerodynamic data
    # Open file with 'r' - Read only
    name_aero = 'Predicted NACA Aero.txt'
    direct_file = os.path.join(direct_RR, name_aero)
#    wait = 0
#    
#    # Wait until Xfoil run over
#    while wait == 0:
#        if not os.path.isfile(direct_file):
#            time.sleep(1)
#        else:
#            print('process'+str(no_proc)+': data file found')
##            time.sleep(0.1)
#            wait = 1
        
    f=open(direct_file,'r')
    i=0
    # Coefficient is recorded from the 13th line
    start_line=13
    while True:
        line = f.readline()
        i=i+1
        # 11th line is label
        if i == 11:
            C_List=str(line)
            temp=C_List.split(' ')
            label=[]
            j=0
            # Cut the label
            while j < len(temp):
                if temp[j]=='':
                    j=j+1
                else:
                    # last label has \n, so cut that value
                    if j == len(temp)-1:
                        temp[j]=temp[j].split('\n')[0]
                        label.append(temp[j])
                        vars()[temp[j]]=[]
                    else:
                        label.append(temp[j])
                        vars()[temp[j]]=[]
                    j=j+1
        # Read Coefficients
        if i > start_line - 1:
            C=str(line) # Coefficient
            temp=C.split(' ')
            j=0
            k=0
            while j < len(temp):
                if temp[j]=='':
                    j=j+1
                else:
                    if j == len(temp)-1:
                        k=len(label)-1
                        temp[j]=temp[j].split('\n')[0]
                        vars()[label[k]].append(temp[j])
                    else:
                        vars()[label[k]].append(temp[j])
                        k=k+1
                    j=j+1
        if not line:
            break    
    f.close()      
    
    ## Calculate Aerodynamic Characteristics ##
    # ======================================= #
    # Save the infomations at matrix
    # Initialize the matrix        
    mat=np.zeros([len(vars()[label[0]]),len(label)])
    
    # Record labels
    for i in range(0,len(label),1):
        mat[:,i]=vars()[label[i]]
    
    # There are two possible case
    # 1. File has the value at AoA 0 degree
    # 2. File does not have the value at AoA 0 degree
    
    # If AoA 0 is exist (Case1) 
    if 0 in mat[:,0]:
        # Find the index of AoA 0
        zero_P=np.where(mat[:,0] == 0)[0][0]
        size=np.shape(mat)
        
        # If there are AoA 0 + step, Calculate by zero, zero + step values
        if size[0]-1 > zero_P:
            Chara=np.zeros([1,5])
            Chara[0][0]=mat[zero_P,1]
            Chara[0][1]=(mat[zero_P+1,1]-mat[zero_P,1])/(mat[zero_P+1,0]-mat[zero_P,0])
            Chara[0][2]=mat[zero_P,2]
            Chara[0][3]=mat[zero_P,4]
            Chara[0][4]=(mat[zero_P+1,4]-mat[zero_P,4])/(mat[zero_P+1,0]-mat[zero_P,0])
        # If AoA 0 - step does not exist, Calculate by zero - step, zero values
        elif size[0] > 1:
            Chara=np.zeros([1,5])
            Chara[0][0]=mat[zero_P,1]
            Chara[0][1]=(mat[zero_P,1]-mat[zero_P-1,1])/(mat[zero_P,0]-mat[zero_P-1,0])
            Chara[0][2]=mat[zero_P,2]
            Chara[0][3]=mat[zero_P,4]
            Chara[0][4]=(mat[zero_P,4]-mat[zero_P-1,4])/(mat[zero_P,0]-mat[zero_P-1,0])
        else:
            Chara=np.array([])
        
        
    # If AoA 0 does not exist (Case2)
    # Use the values near the AoA 0
    elif np.shape(mat)[0] > 2:
        AoA=np.abs(mat[:,0])
        ref_P=[]
        # Find the index of nearest value from AoA 0
        ref_P.append(np.where(AoA==min(AoA))[0][0])
        # If the 0 deg AoA does not exist, Calculate by side values
        # If the min value is positive, use the before step value
        # that value is must the nearest negative value from AoA zero
        if mat[ref_P[0],0] > 0:
            ref_P.append(ref_P[0]-1)
            ref_P.sort()
        # If the min value is negative, use the next step value
        # that value is must the nearest positive value from AoA zero
        elif mat[ref_P[0],0] < 0:
            ref_P.append(ref_P[0]+1)
            ref_P.sort()
        
        Chara=np.zeros([1,5])
        Chara[0][0]=-1*mat[ref_P[0],0]*((mat[ref_P[1],1]-mat[ref_P[0],1]))/(mat[ref_P[1],0]-mat[ref_P[0],0])+mat[ref_P[0],1]
        Chara[0][1]=(mat[ref_P[1],1]-mat[ref_P[0],1])/(mat[ref_P[1],0]-mat[ref_P[0],0])
        Chara[0][2]=-1*mat[ref_P[0],0]*((mat[ref_P[1],2]-mat[ref_P[0],2]))/(mat[ref_P[1],0]-mat[ref_P[0],0])+mat[ref_P[0],2]
        Chara[0][3]=-1*mat[ref_P[0],0]*((mat[ref_P[1],4]-mat[ref_P[0],4]))/(mat[ref_P[1],0]-mat[ref_P[0],0])+mat[ref_P[0],4]
        Chara[0][4]=(mat[ref_P[1],4]-mat[ref_P[0],4])/(mat[ref_P[1],0]-mat[ref_P[0],0])
    
    else:
        Chara=np.array([])
  
    return Chara
