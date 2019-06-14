# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 12:24:51 2017

@author: K_LAB
"""

"""
Info
=====
Generate Fully Connected Neural Network(FCNN)
Training FCNN about NACA 4 digit airfoils
Used Layer Normalization
Used Mini-Batch
"""

import tensorflow as tf
import numpy as np
import time
from sklearn.model_selection import train_test_split
import os
import sys
import matplotlib.pyplot as plt
from math import log10

direct_code=os.path.dirname(os.path.realpath(__file__))
direct_DB=os.path.join(direct_code,"DB")
direct_input = os.path.join(direct_code,'Input')
sys.path.append(direct_code)

import NACA4_Rinput as Rin
import ReadDB
import NACA4_Log as Log
import NACA4_message as msg

class Build_networks():
    def __init__(self,DBname,caseNO):
        # Name of DB
        self.DBname=DBname
        
        ## Data Call ##
        # read Data Base
        self.Data = ReadDB.Read(self.DBname)
        self.caseNO = caseNO
        
        ## Parameters ##
        var, val = Rin.Rinput(direct_input,'input.txt',2)
        self.L_rate = float(val[4])
        self.n_input = int(val[2])
        self.n_output = int(val[3])
        self.No_HL = int(val[0])
        self.No_Neuron = int(val[1])
        self.Active_function = val[5]
        self.dtype = val[6]
        
    def initialize_Training(self):
        
        # Reset Variable Before running
        tf.reset_default_graph()

        # For Batch Normalization
        self.epsilon = 1e-5

        # step size for save and display
        self.SnD_step = 100

        ## Prepare Data ##
        # [Geo] is Output Data
        # [Aero] is Input Data
        
        self.Geo=self.Data[:,0:self.n_output]
        self.Aero=self.Data[:,self.n_output:]

        ## Normalize ##
        # Make [info_data] for Normalize about total Input data
        self.Xdata_avg=np.average(self.Aero, axis = 0)
        self.Xdata_min=np.min(self.Aero, axis = 0)
        self.Xdata_max=np.max(self.Aero, axis = 0)
        self.Xdata_s=np.shape(self.Aero)[1]
        self.Xinfo_data=[self.Xdata_avg, self.Xdata_min, self.Xdata_max, self.Xdata_s]

        direct_save=os.path.join(direct_code,"saved")
        Name_Normfile="Normalization"+str(self.caseNO)+".txt"
        
        if os.path.isfile(os.path.join(direct_save,Name_Normfile)):
            os.remove(os.path.join(direct_save,Name_Normfile))
        f=open(os.path.join(direct_save,Name_Normfile),'w')
        f.write('# Normalziation data #\n')
        f.write('#1 Data for normalize input data #\n')
        f.write('Xdata_avg:\t{}\n'.format(self.Xdata_avg))
        f.write('Xdata_min:\t{}\n'.format(self.Xdata_min))
        f.write('Xdata_max:\t{}\n'.format(self.Xdata_max))
        f.write('Xdata_s:\t{}\n'.format(self.Xdata_s))
        f.write('$\n')
        f.close()
        
    def initialize_ANN_Structure(self):

        ## Prepare Data ##
        # Divide Dataset as a Training Data (75%) and Test Data (25%)
        # random_stata : Fix the Divide rule 
        #       (None - Divide method is always change when we run the code)
        self.X_train, self.X_test, self.y_train, self.y_test=train_test_split(self.Aero, self.Geo, random_state=0 ) # Change training/test size
        
        ## ANN Structure ##
        # Set the number of neuron of each layer
        
        message = str('Data type: {}'.format(self.dtype))
        msg.debuginfo(message)
        # Define the number of each neuron
        i=1
        while i < self.No_HL + 1:
            globals()['n_hidden'+str(i)] = self.No_Neuron # nth hiddenlayer's number of neuron
            i += 1

        # Input and Output Layer
        if self.dtype == 'float32' or 'FP32':
            self.X = tf.placeholder(tf.float32, [None, self.n_input],name='x_input')
            self.Y = tf.placeholder(tf.float32, [None, self.n_output],name='y_input')
            
            # Initialize Weight
            # Initialize: Xavier Initializer        
            i=1
            while i < self.No_HL + 1:
                if i==1:
                    globals()['w'+str(i)] = tf.get_variable("Weight_"+str(i),shape=[self.n_input, globals()['n_hidden'+str(i)]],initializer=tf.contrib.layers.xavier_initializer())  # 1st hidden layer
                else:
                    globals()['w'+str(i)] = tf.get_variable("Weight_"+str(i),shape=[globals()['n_hidden'+str(i-1)], globals()['n_hidden'+str(i)]],initializer=tf.contrib.layers.xavier_initializer())  # nth hidden layer
                i += 1
            
            # This is for a output layer
            globals()['wo'] = tf.get_variable("Weight_"+str(i),shape=[globals()['n_hidden'+str(i-1)], self.n_output],initializer=tf.contrib.layers.xavier_initializer())  # nth hidden layer
    
            # Initialize Biasis
            # Initialize: Xavier Initializer
            i=1
            while i < self.No_HL + 1:
                if i==1:
                    globals()['b'+str(i)] = tf.get_variable("Bias_"+str(i),shape=[globals()['n_hidden'+str(i)]],initializer=tf.contrib.layers.xavier_initializer())  # 1st hidden layer
                else:
                    globals()['b'+str(i)] = tf.get_variable("Bias_"+str(i),shape=[ globals()['n_hidden'+str(i)]],initializer=tf.contrib.layers.xavier_initializer())  # nth hidden layer
                i += 1
            
            # This is for a output layer
            globals()['bo'] = tf.get_variable("Bias_"+str(i),shape=[ self.n_output],initializer=tf.contrib.layers.xavier_initializer())  # nth hidden layer
            
        elif self.dtype == 'float16' or 'FP16':
            self.X = tf.placeholder(tf.float16, [None, self.n_input],name='x_input')
            self.Y = tf.placeholder(tf.float16, [None, self.n_output],name='y_input')
            # Initialize: Xavier Initializer        
            i=1
            while i < self.No_HL + 1:
                if i==1:
                    globals()['w'+str(i)] = tf.get_variable("Weight_"+str(i),shape=[self.n_input, globals()['n_hidden'+str(i)]],dtype=tf.float16,initializer=tf.contrib.layers.xavier_initializer())  # 1st hidden layer
                else:
                    globals()['w'+str(i)] = tf.get_variable("Weight_"+str(i),shape=[globals()['n_hidden'+str(i-1)], globals()['n_hidden'+str(i)]],dtype=tf.float16,initializer=tf.contrib.layers.xavier_initializer())  # nth hidden layer
                i += 1
            
            # This is for a output layer
            globals()['wo'] = tf.get_variable("Weight_"+str(i),shape=[globals()['n_hidden'+str(i-1)], self.n_output],dtype=tf.float16,initializer=tf.contrib.layers.xavier_initializer())  # nth hidden layer
    
            # Initialize Biasis
            # Initialize: Xavier Initializer
            i=1
            while i < self.No_HL + 1:
                if i==1:
                    globals()['b'+str(i)] = tf.get_variable("Bias_"+str(i),shape=[globals()['n_hidden'+str(i)]],dtype=tf.float16,initializer=tf.contrib.layers.xavier_initializer())  # 1st hidden layer
                else:
                    globals()['b'+str(i)] = tf.get_variable("Bias_"+str(i),shape=[ globals()['n_hidden'+str(i)]],dtype=tf.float16,initializer=tf.contrib.layers.xavier_initializer())  # nth hidden layer
                i += 1
            
            # This is for a output layer
            globals()['bo'] = tf.get_variable("Bias_"+str(i),shape=[ self.n_output],dtype=tf.float16,initializer=tf.contrib.layers.xavier_initializer())  # nth hidden layer        
            
            
        elif self.dtype == 'float64'  or 'FP64':
            self.X = tf.placeholder(tf.float64, [None, self.n_input],name='x_input')
            self.Y = tf.placeholder(tf.float64, [None, self.n_output],name='y_input')
            # Initialize: Xavier Initializer        
            i=1
            while i < self.No_HL + 1:
                if i==1:
                    globals()['w'+str(i)] = tf.get_variable("Weight_"+str(i),shape=[self.n_input, globals()['n_hidden'+str(i)]],dtype=tf.float64,initializer=tf.contrib.layers.xavier_initializer())  # 1st hidden layer
                else:
                    globals()['w'+str(i)] = tf.get_variable("Weight_"+str(i),shape=[globals()['n_hidden'+str(i-1)], globals()['n_hidden'+str(i)]],dtype=tf.float64,initializer=tf.contrib.layers.xavier_initializer())  # nth hidden layer
                i += 1
            
            # This is for a output layer
            globals()['wo'] = tf.get_variable("Weight_"+str(i),shape=[globals()['n_hidden'+str(i-1)], self.n_output],dtype=tf.float64,initializer=tf.contrib.layers.xavier_initializer())  # nth hidden layer
    
            # Initialize Biasis
            # Initialize: Xavier Initializer
            i=1
            while i < self.No_HL + 1:
                if i==1:
                    globals()['b'+str(i)] = tf.get_variable("Bias_"+str(i),shape=[globals()['n_hidden'+str(i)]],dtype=tf.float64,initializer=tf.contrib.layers.xavier_initializer())  # 1st hidden layer
                else:
                    globals()['b'+str(i)] = tf.get_variable("Bias_"+str(i),shape=[ globals()['n_hidden'+str(i)]],dtype=tf.float64,initializer=tf.contrib.layers.xavier_initializer())  # nth hidden layer
                i += 1
            
            # This is for a output layer
            globals()['bo'] = tf.get_variable("Bias_"+str(i),shape=[ self.n_output],dtype=tf.float64,initializer=tf.contrib.layers.xavier_initializer())  # nth hidden layer        
            
        else:
            message = 'Check the Float_Precision in input.txt file'
            msg.debuginfo(message)
        
    ## ANN Function ##
    # Multilayer Perceptron
    def MP(self):
        
        i = 1
        while i < self.No_HL + 1:
            if i == 1:
                vars()['HL'+str(i)] = tf.add(tf.matmul(self.X, globals()['w'+str(i)]),globals()['b'+str(i)])
                # Layer Normalization
                vars()['NL'+str(i)] = tf.contrib.layers.layer_norm(vars()['HL'+str(i)],center=True,scale=True)
                
                # Activation
                if self.Active_function == 'ReLU' or 'relu':
                    vars()['AL'+str(i)]=tf.nn.relu(vars()['NL'+str(i)])
                elif self.Active_function == 'ELU' or 'elu':
                    vars()['AL'+str(i)]=tf.nn.elu(vars()['NL'+str(i)])
                elif self.Active_function == 'Identiyu' or 'I':
                    vars()['AL'+str(i)]=vars()['NL'+str(i)]
                elif self.Active_function == 'Softplus' or 'softplus':
                    vars()['AL'+str(i)]=tf.nn.softplus(vars()['NL'+str(i)])
                    
                    
            elif i > 1:
                vars()['HL'+str(i)] = tf.add(tf.matmul(vars()['AL'+str(i-1)], globals()['w'+str(i)]),globals()['b'+str(i)])
                # Layer Normalization
                vars()['NL'+str(i)] = tf.contrib.layers.layer_norm(vars()['HL'+str(i)],center=True,scale=True)
                
                # Activation
                if self.Active_function == 'ReLU' or 'relu':
                    vars()['AL'+str(i)]=tf.nn.relu(vars()['NL'+str(i)])
                elif self.Active_function == 'ELU' or 'elu':
                    vars()['AL'+str(i)]=tf.nn.elu(vars()['NL'+str(i)])
                elif self.Active_function == 'Identiyu' or 'I':
                    vars()['AL'+str(i)]=vars()['NL'+str(i)]
                elif self.Active_function == 'Softplus' or 'softplus':
                    vars()['AL'+str(i)]=tf.nn.softplus(vars()['NL'+str(i)])
                    
            i += 1
        # Output layer
        
        if self.Active_function == 'ReLU' or 'relu':
            Out=tf.nn.relu(tf.add(tf.matmul(vars()['AL'+str(i-1)], globals()['wo']),globals()['bo']))
        elif self.Active_function == 'ELU' or 'elu':
            Out=tf.nn.elu(tf.add(tf.matmul(vars()['AL'+str(i-1)], globals()['wo']),globals()['bo']))
        elif self.Active_function == 'Identiyu' or 'I':
            Out=tf.add(tf.matmul(vars()['AL'+str(i-1)], globals()['wo']),globals()['bo'])
        elif self.Active_function == 'Softplus' or 'softplus':
            Out=tf.nn.softplus(tf.add(tf.matmul(vars()['AL'+str(i-1)], globals()['wo']),globals()['bo']))
        
        return Out

    # Normalize data
    # Generally, use it for Input data
    # [info_data] should have
    #   Average of Data, Minimum of Data, Maximum of Data, Number of Data label
    #   [Array, Array, Array, Array], and AVG, MIN, MAX, must have the value of each Label
    # [scale] is the scale of the Normalize
    def Normalize(self,data,info_data,scale):
        data_avg=info_data[0]
        data_min=info_data[1]
        data_max=info_data[2]
        data_s=info_data[3]
        Ndata=data.copy()
        # Normalize Calculation
        #   Average :Zero
        #   Min, Max: - scale, scale
        for ds in range(data_s):
            Ndata[:,ds]=(data[:,ds]-data_avg[ds])/(data_max[ds]-data_min[ds])
        
        return scale*Ndata

        
    def Training_ANN(self,epoch,mini_batch_size):
        # Generate ANN
        logits = self.MP()
    
        # Calculate Cost
        
        # Optimizer     : Adam Optimizer
        # Train         : Minimize Cost Function
        # Cost Function : RMSE - Root Mean Square Error
#        cost = tf.sqrt(tf.reduce_mean(tf.square(logits-self.Y)))
        # Cost Function : MSE - Mean Square Error
        cost = tf.reduce_mean(tf.square(logits-self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.L_rate,epsilon=0.01)
        train = optimizer.minimize(cost)
        
        # If want to use learning rate step by step, add train method
        # -------------------------------------------------------------
        # cost2 = tf.reduce_mean(tf.square(logits-Y))
        # optimizer2 = tf.train.AdamOptimizer(learning_rate=L_rate*0.1)
        # train2 = optimizer2.minimize(cost)
        
        # cost3 = tf.reduce_mean(tf.square(logits-Y))
        # optimizer3 = tf.train.AdamOptimizer(learning_rate=L_rate)
        # train3 = optimizer2.minimize(cost)
        # -------------------------------------------------------------
        
        # Initialize ANN' all variable
        init = tf.global_variables_initializer()
        
        ## Variables for record ##
        time_training_start=time.time()
        Training_iter=[]
        self.cost_save=[]
        self.Val_cost=[]
       
        # Initialize cost ans i
        c=100
        
        
        ## Save trained Model ##
        Saver = tf.train.Saver() 
        
        training_switch = 1

        while training_switch ==1:
            i=0
            ## Open Session ##
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
                ## Training Section ##
                # initialize ANN
                sess.run(init)
                # Normalize Train dataset
                NXtrain=self.Normalize(self.X_train,self.Xinfo_data,1)
                NXtest=self.Normalize(self.X_test,self.Xinfo_data,1)
                
                # Plot line for validation cost
                valcostline=np.array([0.,0.])
                
                # Training section based on epoch
                while  i < epoch:
                    # mini-batch Training section
                    j=0
                    minibatch_iter=0
                    cost_total = 0
                    while j < np.shape(NXtrain)[0]:
                        if j+mini_batch_size <= np.shape(NXtrain)[0]:
                            _, c = sess.run([train, cost],feed_dict={self.X: NXtrain[j:j+mini_batch_size,:], self.Y: self.y_train[j:j+mini_batch_size,:]})
                            cost_total += c * mini_batch_size
                        elif j+mini_batch_size > np.shape(NXtrain)[0]:
                            _, c = sess.run([train, cost],feed_dict={self.X: NXtrain[j:,:], self.Y: self.y_train[j:,:]})
                            cost_total += c * np.shape(NXtrain[j:,:])[0]
                        
#                        print('Iteration', '%9d' % (minibatch_iter),'/{}'.format(int(np.shape(NXtrain)[0]/mini_batch_size)))    
                        minibatch_iter += 1
                        j += mini_batch_size
                    
                    cost_total = cost_total / np.shape(NXtrain)[0]
                    Val_c = cost.eval(feed_dict={self.X: NXtest, self.Y: self.y_test})
                    
                    # record first running
                    if i==0:
                        # self.cost_err=c.copy()
                        # cost_before=c.copy()
                        Training_iter.append(i+1)
                        self.cost_save.append(cost_total)
                        self.Val_cost.append(Val_c)
                        
                        ##----------------- Plot Validation Cost ----------------##
                        plt.figure(1)
                        plt.ion()
                        plt.plot(Training_iter[0],log10(Val_c),color="dodgerblue")
                        plt.grid(True)
                        plt.xlabel('Update Epoch')
                        plt.ylabel('Cost(Log10(MSE))')
                        plt.title('Loop'+str(self.caseNO))   
                        #f1.show()
                        plt.pause(0.00001)
                            
                        valcostline[0]=log10(Val_c)
                        
                        progress = ('Epoch {:9d}, training cost={:.9f}, Validation cost={:.9f}'.format((i + 1), c, Val_c))
                        sys.stdout.write('\r'+progress)
                    # else:
                        # self.cost_err=np.abs(cost_before-c)
                        # cost_before=c.copy()
    
                    # record cost and No.training  every [save_step]
                    if (i+1) % self.SnD_step == 0:
                        Training_iter.append(i+1)
                        self.cost_save.append(c)
                        self.Val_cost.append(Val_c)
                        
                        valcostline[1]=log10(Val_c)
                        
                        ##----------------- Plot Validation Cost ----------------##
                        if (valcostline[0]*valcostline[1]) != 0:
                            plt.figure(1)
                            plt.ion()
                            plt.plot(Training_iter[-2:],valcostline,color="dodgerblue")
                            plt.grid(True)
                            #f1.show()
                            plt.pause(0.00001)
                            
                        valcostline[0]=valcostline[1]
                        
                        progress = ('Epoch {:9d}, training cost={:.9f}, Validation cost={:.9f}'.format((i + 1), c, Val_c))      
                        sys.stdout.write('\r'+progress)
                    
                    # Interim check/중간점검 #
                    if i == 9999:
                        
                        progress = ('Epoch {:9d}, training cost={:.9f}, Validation cost={:.9f}'.format((i + 1), c, Val_c))      
                        sys.stdout.write('\r'+progress)
                        
                        judgement = np.abs((self.Val_cost[0]-self.Val_cost[-1]))/(self.Val_cost[0])
                        
                        if judgement > 0.8:
                            training_switch = 0
                        else:
                            training_switch = 1
                            message = 'Bad Optimize at 10,000th epoch'
                            msg.debuginfo(message)
                            message = 'Training again'
                            msg.debuginfo(message)
                            
                            # To make i to end of epoch, reset the training sequence
                            i = epoch
                            
                    i=i+1
                
                # print and record last training info
                if i % self.SnD_step !=0:
                    progress = ('Epoch {:9d}, training cost={:.9f}, Validation cost={:.9f}'.format((i + 1), c, Val_c))
                    sys.stdout.write('\r'+progress)
                    
                    Training_iter.append(i)
                    self.cost_save.append(c)
                    self.Val_cost.append(Val_c)
    
                message = 'Optimization Finished!\n'
                msg.debuginfo(message)
                
                direct_save = os.path.join(direct_code,"saved")
                Name_model = "model"+str(self.caseNO)
                Save_path = Saver.save(sess, os.path.join(direct_save,Name_model))
                
                message = str("model was saved in " + str(Save_path))
                msg.debuginfo(message)
                
                time_test_start=time.time()
                
                Result_test = sess.run(logits, feed_dict={self.X: NXtest, self.Y: self.y_test})
                time_test = time.time() - time_test_start
                
                
        time_training = time.time() - time_training_start
        message = str('runtime - training\t: {:.2f} s, {:.2f} min, {:.2f} hr'.format(time_training,time_training/60,time_training/3600))
        msg.debuginfo(message)
        message = str('runtime - test\t: {:.2f} s, {:.2f} min, {:.2f} hr'.format(time_test,time_test/60,time_test/3600))
        msg.debuginfo(message)
        
        #print("Model saved in>> ",Save_direct)
        #print("\n")
        
        return np.array(self.cost_save), np.array(self.Val_cost), np.array(Training_iter), Result_test, self.y_test, self.X_test #Training_iter, self.NRMSE_train, self.NRMSE_test, self.target_Config, Result_test
    
