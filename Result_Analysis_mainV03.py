# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 14:05:17 2018

@author: CK_DL2
"""

"""
Info
=====
Collecting the Result
Analyze the Result and Error
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt


direct_code=os.path.dirname(os.path.realpath('__file__'))

direct_DB=os.path.join(direct_code,"DB")
sys.path.append(direct_code)
direct_Analyze=os.path.join(direct_code,'Analyzed DB')

import Result_Analysis_ReadDB as RDB
import Result_Analysis_Rinput as Rin


#######################################
name_inputfile='Input_Analyze.txt'
#######################################

comp_folder = Rin.Rinput(name_inputfile,1)

for i in range(len(comp_folder)):
    vars()['path'+str(i)]=os.path.join(direct_Analyze,comp_folder[i])

loop = Rin.Rinput(name_inputfile,2)

name_AddDB=[]
name_AnsDB=[]
name_ErrDB=[]

for i in range(len(comp_folder)):
    name_AddDB.append('Additional DB'+loop[i]+'.txt')
    name_AnsDB.append('Answer DB'+loop[i]+'.txt')
    name_ErrDB.append('Error DB'+loop[i]+'.txt')

name_coeff = ['C','LC','T','Cl0', 'Cla0', 'Cd0', 'Cm0', 'Cma0']

i=0
for i in range(len(comp_folder)):
    vars()['path'+str(i)+'_AddDB_'+str(loop[0])]=RDB.Read(os.path.join(vars()['path'+str(i)],name_AddDB[i]))
    vars()['path'+str(i)+'_AnsDB_'+str(loop[0])]=RDB.Read(os.path.join(vars()['path'+str(i)],name_AnsDB[i]))
    
i=0
for i in range(len(comp_folder)):
    vars()['path'+str(i)+'_ErrDB_'+str(loop[0])]=RDB.Read(os.path.join(vars()['path'+str(i)],name_ErrDB[i]))
    vars()['MSE'+str(i)+'_ErrDB_'+str(loop[0])]=np.sum(vars()['path'+str(i)+'_ErrDB_'+str(loop[0])]**2,axis=0)/np.shape(vars()['path'+str(i)+'_ErrDB_'+str(loop[0])])[0]

i=0
for i in range(len(comp_folder)):
    j=0
    for j in range(len(name_coeff)):
        vars()['path'+str(i)+'_'+name_coeff[j]]=np.append(vars()['path'+str(i)+'_AnsDB_'+str(loop[0])][:,0:3],vars()['path'+str(i)+'_ErrDB_'+str(loop[0])][:,j:j+1],axis=1)
        vars()['path'+str(i)+'_'+name_coeff[j]+'_sorted']=vars()['path'+str(i)+'_'+name_coeff[j]][vars()['path'+str(i)+'_'+name_coeff[j]][:,3].argsort()[::-1]]
        

percentile=['99%','95%','90%','85%','80%']
    
# vars()[name_coeff[j]+'_PR_Error']: Epoch vs error at each percentile rank
j=0
for j in range(len(name_coeff)):

    vars()[name_coeff[j]+'_PR_Error']=np.zeros([len(comp_folder),len(percentile)])
    i=0
    for i in range(len(comp_folder)):
        no_data=np.shape(vars()['path'+str(i)+'_ErrDB_'+str(loop[0])])[0]
        # p_col is the number of column that boundary of percentile rank at 99%, 95%, 90%, 85%, 80%
        p_col=[int(np.round(no_data*0.01)), int(np.round(no_data*0.05)),int(np.round(no_data*0.1)),int(np.round(no_data*0.15)),int(np.round(no_data*0.2))]
        
        k=0
        for k in range(len(percentile)):
            vars()[name_coeff[j]+'_PR_Error'][i,k]=vars()['path'+str(i)+'_'+name_coeff[j]+'_sorted'][p_col[k],3]
        
plt.close('all')
# Plotting 1 #

x=np.arange(len(percentile))
j=0
plt.close('all')
plt.ioff()
fig_size=[16,9]
plt.rcParams["figure.figsize"]=fig_size
bar_width=1/(len(comp_folder)+1)
for j in range(len(name_coeff)):
    fig=plt.figure(j)
    i=0
    for i in range(len(comp_folder)):
        vars()['bar_Epoch'+str(i)]=plt.bar(x+(i-int(len(comp_folder)/2))*bar_width,vars()[name_coeff[j]+'_PR_Error'][i,:],bar_width,label=comp_folder[i])
        for a, b in zip(x+(i-int(len(comp_folder)/2))*bar_width,vars()[name_coeff[j]+'_PR_Error'][i,:]):
            plt.text(a, b*1.02, '{:3.2E}'.format(b),fontsize=11,horizontalalignment='center')

            
            
        
    plt.legend(fontsize=18)
    plt.xticks(x, percentile)
    plt.tight_layout()
    plt.title('Percentile Rank: '+name_coeff[j],fontsize=28)
    plt.xlabel('Percentile Rank(from lower error)',fontsize=20)
    plt.ylabel('ABS Error',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
#    plt.grid(True)
    plt.tight_layout()
#    plt.show()

    
    direct_plot=os.path.join(direct_Analyze,'plot')
    plt.savefig(os.path.join(direct_plot,'Percentile Rank_'+name_coeff[j]))

i=0
for i in range(len(comp_folder)):
    no_data=np.shape(vars()['path'+str(i)+'_ErrDB_'+str(loop[0])])[0]
    # p_col is the number of column that boundary of percentile rank at 95%, 90%, 85%, 80%
    p95_col=int(np.round(no_data*0.05))
    
    ##########################################################################################
    name_config =['C','LC','T']
    name_config_full = ['Max.Camber','Max.Camber Location','Max.Thickness']
    range_config=[11,11,21]
    j=0
    for j in range(len(name_coeff)):
        
        for c in range(len(name_config)):
            vars()['path'+str(i)+'_'+name_coeff[j]+'_'+name_config[c]]=np.zeros([range_config[c],1])
            temp = list(np.round(vars()['path'+str(i)+'_'+name_coeff[j]+'_sorted'][:p95_col,c]))
            for k in range(len(vars()['path'+str(i)+'_'+name_coeff[j]+'_'+name_config[c]])):
                vars()['path'+str(i)+'_'+name_coeff[j]+'_'+name_config[c]][k,0]=int(temp.count(k))
             
            plt.ioff()
    
            fig=plt.figure()
            plt.bar(np.arange(range_config[c]),vars()['path'+str(i)+'_'+name_coeff[j]+'_'+name_config[c]][:,0])
            plt.title('Histogram: '+comp_folder[i]+', '+name_coeff[j]+', '+name_config_full[c], fontsize = 28)
            plt.xlabel(name_config_full[c], fontsize = 20)
            plt.ylabel('Frequency', fontsize = 20)
            
            
            plt.savefig(os.path.join(direct_plot,'Histogram_'+name_coeff[j]+'_'+name_config[c]+'_'+comp_folder[i]))
            
            plt.close()
            
        plt.ioff()
        fig=plt.figure()
        plt.plot(vars()['path'+str(i)+'_'+name_coeff[j]+'_sorted'][:p95_col,0],vars()['path'+str(i)+'_'+name_coeff[j]+'_sorted'][:p95_col,1],marker='.',linestyle='None')
        plt.title(comp_folder[i]+': C vs LC, '+name_coeff[j], fontsize = 28)
        plt.xlabel(name_config_full[0], fontsize = 20)
        plt.ylabel(name_config_full[1], fontsize = 20)
        plt.grid(True)
        plt.savefig(os.path.join(direct_plot,'Distribution_C vs LC '+name_coeff[j]+' '+comp_folder[i]))
        plt.close()
        
        plt.ioff()
        fig=plt.figure()
        plt.plot(vars()['path'+str(i)+'_'+name_coeff[j]+'_sorted'][:p95_col,1],vars()['path'+str(i)+'_'+name_coeff[j]+'_sorted'][:p95_col,2],marker='.',linestyle='None')
        plt.title(comp_folder[i]+': LC vs T, '+name_coeff[j], fontsize = 28)
        plt.xlabel(name_config_full[1], fontsize = 20)
        plt.ylabel(name_config_full[2], fontsize = 20)
        plt.grid(True)
        plt.savefig(os.path.join(direct_plot,'Distribution_LC vs T '+name_coeff[j]+' '+comp_folder[i]))
        plt.close()
        
        plt.ioff()
        fig=plt.figure()
        plt.plot(vars()['path'+str(i)+'_'+name_coeff[j]+'_sorted'][:p95_col,2],vars()['path'+str(i)+'_'+name_coeff[j]+'_sorted'][:p95_col,0],marker='.',linestyle='None')
        plt.title(comp_folder[i]+': T vs C, '+name_coeff[j], fontsize = 28)
        plt.xlabel(name_config_full[2], fontsize = 20)
        plt.ylabel(name_config_full[0], fontsize = 20)
        plt.grid(True)
        plt.savefig(os.path.join(direct_plot,'Distribution_T vs C '+name_coeff[j]+' '+comp_folder[i]))
        plt.close()
        
        
        
        
        
        
        

    