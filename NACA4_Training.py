# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 21:15:42 2018

@author: cck18
"""

"""
Info
=====
Training and Make Datafile
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time


direct_work=os.getcwd()
direct_code=os.path.dirname(os.path.realpath(__file__))

direct_DB=os.path.join(direct_code,"DB")
direct_input = os.path.join(direct_code,'Input')
Plotdirect=os.path.join(direct_code,"Plot")
sys.path.append(direct_code)

import NACA4_writeDB as writeDB
import NACA4_DB
import ReadDB
import NACA4_Rinput as Rin
import NACA4_ANN98 as AN
import NACA4_AssortData as AS
import NACA4_Log as Log
import NACA4_message as msg

# Setting is Xfoil setting

def Training(caseNO, Epoch, mini_batch_size, re_build):
    ##----------------------- Basic Setting -----------------------##

    DBname_AF="Airfoil DB"+str(caseNO)+".txt"
    DBname_Total="Airfoil DB"+str(caseNO + 1)+".txt"
        
    DBname_Predict="Predict Config"+str(caseNO)+".txt"
    DBname_Answer="Answer DB"+str(caseNO)+".txt"
    DBname_Additional="Additional DB"+str(caseNO)+".txt"
    DBname_Error="Error DB"+str(caseNO)+".txt"
    Plotname = "Training Log_"+str(caseNO)+".png"
    
    
    ##----------------------- Training ANN -----------------------##
    Log.log(None)
    message = 'Training Loop: ' + str(caseNO)
    msg.debuginfo(message)
    message = 'Start initialization for training ANN'
    msg.debuginfo(message)
    
    var, val = Rin.Rinput(direct_input,'input.txt',2)
    no_output = int(val[3])
    
    start_time=time.time()
    
    training_switch = 1
    while training_switch == 1:
        
        ANN=AN.Build_networks(DBname_AF, caseNO)
        ANN.initialize_Training()
        ANN.initialize_ANN_Structure()
        cost_train, cost_val, training_iter, result_test, ans, ans_aero = ANN.Training_ANN(Epoch,mini_batch_size)
        
        judgement = cost_val[-1]
#        judgement = np.abs((cost_val[0]-cost_val[-1]))/(cost_val[0])
        
        if judgement < 5:
            training_switch = 0
            message = 'Training was finished well'
            msg.debuginfo(message)
            if np.shape(result_test)[0] == np.shape(ans)[0]:
                training_switch = 0
                message = 'The number of predicted dataset is equal to answer dataset'
                msg.debuginfo(message)
            else:
                training_switch = 1
                message = 'Bad Training Result'
                msg.debuginfo(message)
                message = 'Predicted data less than answer'
                msg.debuginfo(message)
                message = 'Training Again!'
                msg.debuginfo(message)
        else:
            training_switch = 1
            message = 'Bad Training Result'
            msg.debuginfo(message)
            message = 'Final MSE is too high'
            msg.debuginfo(message)
            message = 'Training Again!'
            msg.debuginfo(message)
            
        runtime = time.time() - start_time
        message = str("Runtime: {:d} sec = {:d} min".format(round(runtime),round(runtime/60)))    
        msg.debuginfo(message)
        
        ##---------------------------- Save Plot ----------------------------##
        
        if os.path.isfile(os.path.join(Plotdirect,Plotname)):
            os.remove(os.path.join(Plotdirect,Plotname))
        
        plt.figure(1)
        plt.savefig(os.path.join(Plotdirect,Plotname))
        
        message = 'Test Plot was Saved'        
        msg.debuginfo(message)
        
    if re_build == 1:   
        ##------------- Predict Database Construction -------------##
        # If Initial DB not exist in DB directory, It will make
        # If Initial DB exist, It will not run
        message = '>> Create Prediction Configure DB'
        msg.debuginfo(message)
        
        if os.path.isfile(os.path.join(direct_DB,DBname_Predict)):
            os.remove(os.path.join(direct_DB,DBname_Predict))
        
        # Make Prediced Configuration DataBase
        writeDB.ConfigDB(DBname_Predict,result_test)
        
        message = str('File "{}" Created\n'.format(DBname_Predict))
        msg.debuginfo(message)
    
        ##---------------- Create Answer Database ----------------##
        # If Additional DB not exist in DB directory, It will make
        # If Additional DB exist, It will not run
        message = '>> Create Answer DB'
        
        if os.path.isfile(os.path.join(direct_DB,DBname_Answer)):
            os.remove(os.path.join(direct_DB,DBname_Answer))
        
        # Make Answer DataBase
        writeDB.NACADB(DBname_Answer,ans,ans_aero)
        message = str('File "{}" Created\n'.format(DBname_Answer))
        msg.debuginfo(message)
    
        ##---------------- Create Addtional Database ----------------##
        # If Additional DB not exist in DB directory, It will make
        # If Additional DB exist, It will not run
        message = '>> Create Additional DB'
        if os.path.isfile(os.path.join(direct_DB,DBname_Additional)):
            os.remove(os.path.join(direct_DB,DBname_Additional))
        
        # Make Addtional DB
        NACA4_DB.performanceDB(DBname_Predict,DBname_Additional)
        message = str('File "{}" Created\n'.format(DBname_Additional))
        msg.debuginfo(message)
        
        # Make Total DB
        ##--------------- Assort Addtional Database ---------------##
        
        
        AF_DB=ReadDB.Read(DBname_AF)
        
        AddDB, PrdDB, AnsDB = AS.run(caseNO)
        TotalDB=np.append(AF_DB,AddDB,axis=0)
        
        if os.path.isfile(os.path.join(direct_DB,DBname_Additional)):
            os.remove(os.path.join(direct_DB,DBname_Additional))
        
        para=AddDB[:,0:no_output]
        aero=AddDB[:,no_output:]
        writeDB.NACADB(DBname_Additional,para,aero)
        
        message = str('File "{}" Assorted\n'.format(DBname_Additional))
        msg.debuginfo(message)
        
        ##---------------- Create Answer DB Again ----------------##
        # If Additional DB not exist in DB directory, It will make
        # If Additional DB exist, It will not run
        message = '>> Create Answer DB'
        msg.debuginfo(message)
        
        if os.path.isfile(os.path.join(direct_DB,DBname_Answer)):
            os.remove(os.path.join(direct_DB,DBname_Answer))
        
        # Make Answer DataBase
        para=AnsDB[:,0:no_output]
        aero=AnsDB[:,no_output:8]
        writeDB.NACADB(DBname_Answer,para,aero)
        
        message = str('File "{}" Created\n'.format(DBname_Answer))
        msg.debuginfo(message)
        
        ##---------------------- Error Database ---------------------##
        ErrDB = np.abs(np.subtract(AddDB, AnsDB))
        
        para=ErrDB[:,0:no_output]
        aero=ErrDB[:,no_output:]
        writeDB.NACADB(DBname_Error,para,aero)
        
        message = str('File "{}" Created\n'.format(DBname_Error))
        msg.debuginfo(message)
        
        ##------------------- Intergrate Database -------------------##
        message = '>> Create Total DB'
        
        if os.path.isfile(os.path.join(direct_DB,DBname_Total)):
            os.remove(os.path.join(direct_DB,DBname_Total))
            
        para=TotalDB[:,0:no_output]
        aero=TotalDB[:,no_output:]
        writeDB.NACADB(DBname_Total,para,aero)
        
        message = str('File "{}" Created\n'.format(DBname_Total))
        msg.debuginfo(message)
        
        runtime = time.time() - start_time
        message = str("Runtime: {:d} sec = {:d} min".format(round(runtime),round(runtime/60)))
        msg.debuginfo(message)
        message = '>> Training Process Finished <<\n'
        msg.debuginfo(message)
        
        
    ##----------------- Training Information ------------------##
    DBname_Info="Training Info"+ str(caseNO)+".txt"
    filedirect=os.path.join(direct_DB,DBname_Info)    
    if os.path.isfile(filedirect):
        os.remove(os.path.join(direct_DB,DBname_Info))
    message = ('Create Training Info DB as "{}"'.format(DBname_Info))
    msg.debuginfo(message)
    
    f=open(filedirect,'w')
    f.write('Training Information\n')
    f.write('     Epoch  Cost Train Cost Valid\n')
    f.write(' =========  ========== ==========\n')
                
    i=0
    
    while i < np.shape(training_iter)[0]:
        f.write("{:10d}{:11.5f}{:11.5f}\n"\
                .format(training_iter[i],cost_train[i],cost_val[i]))
        i=i+1
        
    f.close()
        
if __name__ == "__main__":
    Training(1,50000,5000,0)
    