# DLED-NACA-4
Deep Learning based Engineering Design: NACA 4-digit Airfoil

- Description of this process is in my grad thesis.(But written in korean)
- To run these codes, you need python3 and tensorflow-gpu.

1. Setup training conditions at Input\input.txt and Input\Xfoil setting. txt

  input.txt
  
    #1 Training Info
      Start_loop: Start loop of Database Enhancement Loop(DEL)
      Max_loop: End loop of DEL
      Max_epoach: Training epoch
      mini_batch_size: Setup the size of mini-batch
      Initial_DB
        0: Do not make Initial DB(If you already have an Initial DB)
        1: make Initial DB for training
      re_Analysis: After training, analyze the airfoils with XFOIL that ANN predicted
        0: If you don't need to re-analyze, use this option
        1: If you need to re-analyze, or want to use Database Enhancement Loop, use this option
      
    #2 Hyperparameter of ANN
      Do not change No_input and No_output
      Only 4 Activation Functions are possible to use: Identity, ReLU, ELU, Softplus

  Xfoil setting.txt
  
    No_point: Over 400, XFOIL cannot make good results. 100~200 recommended
    Iter: Iteration number of Xfoil calculation
    Re: Reynolds number
    Mach: Mach number
    timelim: Sometimes, XFOIL stops and program cannot be shutdown by code. If the time over this value, kill the process
    
2. Run NACA_main.py

3. ANN model will be saved in "saved" folder

4. Airfoil data files are saved in "DB" folder

5. ANN test result plots during training are saved in "Plot" folder
