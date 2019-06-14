# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 12:28:14 2017

@author: Muuky
@author: K_LAB
"""

"""
Info
=====
Generate NACA 4 digit
"""

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
workdirect=os.getcwd()
codedirect=os.path.dirname(os.path.realpath(__file__))
sys.path.append(codedirect)


# C  : Max. Camber
# LC : Loc. Max. Camber
# T  : Max. Thickness
def NACA4(No_Point, C, LC, T,Savedirect,no_proc):
    
    m = C/100
    p= LC/10
    t = T/100
    name_file = "Temp Config"+str(no_proc)+".txt"
    filedirect=os.path.join(Savedirect,name_file)
    # If there are remain Temp Config file, remove #
    if os.path.isfile(filedirect):
        os.remove(filedirect)

    f=open(filedirect,'w')
    f.write("TempAirfoil\n")
    f.close()         
    point = 1-(np.logspace(0.0,1.0,No_Point/2)-1)/18
    point = np.append(point,(np.logspace(1.0,0.0,No_Point/2)-1)/18)
    point = np.flip(np.unique(point),0)

    # Write Upper side point #
    for x in point:
        upper = naca4upper(x,m,p,t)
        f=open(filedirect,"a")
        f.write("{:10.5f}{:10.5f}\n".format(upper[0], upper[1]))
        f.close()
    
    # Write Bottom side point #
    point=np.flip(point,0)
    for x in point:
        lower = naca4lower(x,m,p,t)
        f=open(filedirect,"a")
        f.write("{:10.5f}{:10.5f}\n".format(lower[0], lower[1]))
        f.close()
        
def draw_NACA4(No_Point, C, LC, T,Savedirect):
    
    m = C/100
    p= LC/10
    t = T/100
    
    filedirect=os.path.join(Savedirect,"Predicted NACA.txt")
    # If there are remain Temp Config file, remove #
    if os.path.isfile(filedirect):
        os.remove(filedirect)

    f=open(filedirect,'w')
    f.write("Predicted NACA\n")
    f.close()         
    point = 1-(np.logspace(0.0,1.0,No_Point/2)-1)/18
    point = np.append(point,(np.logspace(1.0,0.0,No_Point/2)-1)/18)
    point = np.flip(np.unique(point),0)
    
    Up_point = np.zeros([0,2])
    Lo_point = np.zeros([0,2])
    # Write Upper side point #
    for x in point:
        upper = naca4upper(x,m,p,t)
        f=open(filedirect,"a")
        f.write("{:10.5f}{:10.5f}\n".format(upper[0], upper[1]))
        f.close()
        Up_point=np.append(Up_point,np.expand_dims(np.array(upper),axis=0),axis=0)
       
    
    # Write Bottom side point #
    point=np.flip(point,0)
    for x in point:
        lower = naca4lower(x,m,p,t)
        f=open(filedirect,"a")
        f.write("{:10.5f}{:10.5f}\n".format(lower[0], lower[1]))
        f.close()
        Lo_point=np.append(Lo_point,np.expand_dims(np.array(lower),axis=0),axis=0)
    
    plt.close('all')
    plt.plot(Up_point[:,0],Up_point[:,1],label='Upper')
    plt.plot(Lo_point[:,0],Lo_point[:,1],label='Lower')
    
    plt.grid(True)
    plt.xlabel('x',fontsize=16)
    plt.ylabel('y',fontsize=16)
    plt.legend()
    plt.title('Predicted Airfoil',loc='left',fontsize=20)
    
    range_plot = 1.2
    plt.xlim([-0.1,-0.1+range_plot])
    plt.ylim([0.0-range_plot/2,0.0+range_plot/2])
    plt.savefig(os.path.join(Savedirect,'Predicted Airfoil'))
        
    return Lo_point

def cosine_spacing(num):
    beta0 = np.linspace(0.0,1.0,num+1)
    x = []
    
    for beta in beta0:
        x.append((0.5*(1.0-np.cos(beta))))
        
    return x
            
def camber_line( x, m, p):
    
    if (x>=0) & (x < p):
        return (m/(p**2.))*(2.*p*x - x**2.0)
    
    elif (x>=p) & (x<=1):
        return (m/(1-p)**2)*(1 - 2.0*p + 2.0*p*x- x**2.0)

def dyc_over_dx(x, m, p):
    
    if (x >= 0) & (x < p):
        return (2.0*m/(p**2.))*(p - x)
    
    elif (x >= p) & (x <= 1):
        return (2.0*m/((1-p)**2))*(p - x)
          
def thickness(x, t):
    term1 =  0.2969 * (np.sqrt(x))
    term2 = -0.1260 * x
    term3 = -0.3516 * x**2.0
    term4 =  0.2843 * x**3.0
    term5 = -0.1015 * x**4.0
    return 5 * t * (term1 + term2 + term3 + term4 + term5)

def naca4upper(x, m, p, t):
    dyc_dx = dyc_over_dx(x, m, p)
    th = np.arctan(dyc_dx)
    yt = thickness(x, t)
    yc = camber_line(x, m, p)
    xx = x - yt*np.sin(th)
    yy = yc + yt*np.cos(th)
    
    return (xx,yy)

def naca4lower(x,m,p,t,c=1): 
    dyc_dx = dyc_over_dx(x, m, p)
    th = np.arctan(dyc_dx)
    yt = thickness(x, t)
    yc = camber_line(x, m, p)  
    xx = x + yt*np.sin(th)
    yy = yc - yt*np.cos(th)
    return (xx,yy)