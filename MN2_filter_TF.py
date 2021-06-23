#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:05:24 2021

@author: bcourtne

MN2 filter transfer function

"""

import control



##########
# directly copied from VLT-MAN-ESO-15400-4234 : VLTI - UT Vibration Monitoring System, Software User and Maintenance Manual
##########
pi = np.pi
ts = 1/4000;     # sampling time    
w0 = 4.33*2*pi;  # pole
w1 = 5.77*2*pi;  # zero
w3 = 2.5*2*pi;   # highpass
w4 = 1000*2*pi;  # pole
w5 = 150*2*pi;   # zero

g_1 = (w1*ts/2+1)/(w0*ts/2+1)*10**(-0.3/20);
a1_1 = (w1*ts/2-1)/(w1*ts/2+1);
b1_1 = (w0*ts/2-1)/(w0*ts/2+1);
tf_1 = control.tf(g_1*np.array([1, a1_1]),[1, b1_1],ts);

g_2 = ts**2/4/(1+w3*ts/2+(w3*ts/2)**2);
a1_2 = 2;
a2_2 = 1;
b1_2 = 2*((w3*ts/2)**2-1)/(1+w3*ts/2+(w3*ts/2)**2);
b2_2 = (1-w3*ts/2+(w3*ts/2)**2)/(1+w3*ts/2+(w3*ts/2)**2);
tf_2 = control.tf(g_2*np.array([1, a1_2, a2_2]),[1, b1_2, b2_2],ts);

g_3 = 1/(1+w3*ts/2);
a1_3 = -1;
b1_3 = (w3*ts/2-1)/(w3*ts/2+1);
tf_3 = control.tf(g_3*np.array([1, a1_3]),[1, b1_3],ts);

a1_4 = (w5*ts/2-1)/(w5*ts/2+1);
b1_4 = (w4*ts/2-1)/(w4*ts/2+1);
g_4 = (1+b1_4)/(1+a1_4);
tf_4 = control.tf(g_4*np.array([1, a1_4]),[1, b1_4],ts);

tf = tf_1*tf_2*tf_3*tf_4
#inverse transfer function
itf = control.tf(tf.den,tf.num,ts)

##Bode plot  of transfer function 
mag, phase, omega = control.bode(tf, Hz=True, dB=True, omega_limits=(0.1, 1000), Plot=True)