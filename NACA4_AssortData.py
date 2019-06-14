# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 12:43:06 2018

@author: cck18
"""

"""
Info
====
Assort Data
Analyzed
Not Analyzed
"""

import os
import sys
import numpy as np

workdirect=os.getcwd()
codedirect=os.path.dirname(os.path.realpath('__file__'))
DBdirect=os.path.join(codedirect,"DB")
sys.path.append(codedirect)
import ReadDB

def run(caseNO):
    DBname_Answer="Answer DB"+str(caseNO)+".txt"
    DBname_Additional="Additional DB"+str(caseNO)+".txt"
    DBname_Predict="Predict Config"+str(caseNO)+".txt"
    DBname_Fail="Failure DB" + str(caseNO) + ".txt"
    DBname_Out="BoundaryOut DB" + str(caseNO) + ".txt"
    boundary=np.array([[0, 10],[0, 10]])        
    
    
    
    AddDB=ReadDB.Read(DBname_Additional)
    AnsDB=ReadDB.Read(DBname_Answer)
    PrdDB=ReadDB.Read(DBname_Predict)
    
    if np.shape(AddDB)[0] == np.shape(AnsDB)[0]:
        print('All data was Analyzed')
        return AddDB, PrdDB, AnsDB
    else:
        Analyze_X=0
        Analyze_Over=0
        
        print('>> Assort Useless Data')
        
        i=0
        Fail_Prd = np.empty([1,3])
        Fail_Prd = np.delete(Fail_Prd,0,0)
        Fail_Ans = np.empty([1,8])
        Fail_Ans = np.delete(Fail_Ans,0,0)
        ## DB identical ##
        tester = 0
        while i < np.shape(AddDB)[0]:
            try:
                if AddDB[i][0] == PrdDB[i][0] and AddDB[i][1] == PrdDB[i][1] and AddDB[i][2] == PrdDB[i][2]:
                    tester += 1
                    i=i+1
                else:
                    Fail_Prd = np.append(Fail_Prd,[PrdDB[i]],0)
                    Fail_Ans = np.append(Fail_Ans,[AnsDB[i]],0)
                    PrdDB=np.delete(PrdDB,i,0)   
                    AnsDB=np.delete(AnsDB,i,0)
                    Analyze_X=Analyze_X+1
                    
            except:
                print(tester)
                print(np.shape(AddDB))
                print(np.shape(AnsDB))
                print(np.shape(PrdDB))
                        
        if np.shape(PrdDB)[0] > np.shape(AddDB)[0]:
            Fail_Prd = np.append(Fail_Prd,[PrdDB[i]],0)
            Fail_Ans = np.append(Fail_Ans,[AnsDB[i]],0)
            PrdDB=np.delete(PrdDB,i,0)   
            AnsDB=np.delete(AnsDB,i,0)
            Analyze_X=Analyze_X+1
        
        print(Analyze_X)
        print(tester)
        print(np.shape(AddDB))
        print(np.shape(AnsDB))
        print(np.shape(PrdDB))
                
        filedirect=os.path.join(DBdirect,DBname_Fail)
        if os.path.isfile(filedirect):
            print('File "{}" already exist'.format(DBname_Fail))
            os.remove(os.path.join(DBdirect,DBname_Fail))
            print("{} was removed".format(DBname_Fail))
        print('Create Initial DB as "{}"'.format(DBname_Fail))
        f=open(filedirect,'w')
        f.write('Predicted but Cannot Analyze by Xfoil\n')
        f.write(' #----Predicted Configure----# #------Answer Configure-----#  #--------------Answer Performance--------------#\n')
        f.write('         c        Lc         t         c        Lc         t       Cl0      Cla0       Cd0       Cm0      Cma0\n')
        f.write(' ========= ========= ========= ========= ========= ========= ========= ========= ========= ========= =========\n')    
        
        if np.shape(Fail_Prd)[0]==0:
            f.close()
        
        else:
            i=0
            while i < np.shape(Fail_Prd)[0]:  
                f.write("{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}\n"\
                        .format(Fail_Prd[i][0],Fail_Prd[i][1],Fail_Prd[i][2],\
                                Fail_Ans[i][0],Fail_Ans[i][1],Fail_Ans[i][2],\
                                Fail_Ans[i][3],Fail_Ans[i][4],Fail_Ans[i][5],Fail_Ans[i][6],Fail_Ans[i][7]))
                i=i+1
            
            print('File "{}" Created\n'.format(DBname_Fail))            
            print("Finished to make Addtional DB and Answer DB identical\n")
            f.close()

        ## Boundary Check ##
        # If there are the configuration over the boundary, delete #
        i=0
        
        Out_Add = np.empty([1,8])
        Out_Add = np.delete(Out_Add,0,0)
        Out_Ans = np.empty([1,8])
        Out_Ans = np.delete(Out_Ans,0,0)

        while i < np.shape(AddDB)[0]:

            if PrdDB[i][0] < boundary[0][1] and PrdDB[i][1] < boundary[1][1] and PrdDB[i][1] >= boundary[1][0] and PrdDB[i][2] > 0:
                i=i+1
            else:
                Out_Add = np.append(Out_Add,[AddDB[i]],0)
                Out_Ans = np.append(Out_Ans,[AnsDB[i]],0)               
                Analyze_Over = Analyze_Over + 1
                i=i+1
                
        if np.shape(Out_Add)[0] > 0:   
            filedirect=os.path.join(DBdirect,DBname_Out)
            if os.path.isfile(filedirect):
                print('File "{}" already exist'.format(DBname_Out))
                os.remove(os.path.join(DBdirect,DBname_Out))
                print("{} was removed".format(DBname_Out))
            print('Create Initial DB as "{}"'.format(DBname_Out))
            f=open(filedirect,'w')
            f.write('Analyzed, but Config Parameter was out of boundary\n')
            f.write(' #----Predicted Configure----# #---------------Answer Performance--------------# #------Answer Configure-----# #---------------Answer Performance--------------#\n')
            f.write('         c        Lc         t       Cl0      Cla0       Cd0       Cm0      Cma0         c        Lc         t       Cl0      Cla0       Cd0       Cm0      Cma0\n')
            f.write(' ========= ========= ========= ========= ========= ========= ========= ========= ========= ========= ========= ========= ========= ========= ========= =========\n')    
            
            i=0
            while i < np.shape(Out_Ans)[0]:  
                f.write("{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}{:10.5f}\n"\
                        .format(Out_Add[i][0],Out_Add[i][1],Out_Add[i][2],\
                                Out_Add[i][3],Out_Add[i][4],Out_Add[i][5],Out_Add[i][6],Out_Add[i][7],\
                                Out_Ans[i][0],Out_Ans[i][1],Out_Ans[i][2],\
                                Out_Ans[i][3],Out_Ans[i][4],Out_Ans[i][5],Out_Ans[i][6],Out_Ans[i][7]))
                i=i+1
            
            print('File "{}" Created\n'.format(DBname_Out))
        
        else:
            print('All Config Data in the boundary')
                    
        print(">> Finished to check boundary <<")
        f.close()
        
        return AddDB, PrdDB, AnsDB
    
if __name__=="__main__":
    AddDB, PrdDB, AnsDB = run(20)