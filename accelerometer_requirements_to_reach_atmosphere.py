#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 14:15:50 2021

@author: bcourtne

deriving g/sqrt(Hz) for the atmopsheric piston (double derivative of PSD) 

"""
import os
os.chdir('/Users/bcourtne/Documents/Hi5/vibration_analysis/utilites/')
import matplotlib.pyplot as plt
import numpy as np
import atmosphere_piston

(fs, atm_pist_psd), atm_params = atmosphere_piston.atm_piston()

plt.figure(figsize=(10,7))
g = 9.81 #ms^-2
#note  we need  a factor of 4pi^2 since we are considering PSD in freq and not angular frequency                                            
plt.loglog(fs,1e6 * 1/g* (2*np.pi)**2 * np.sqrt(fs**4 * (np.array(atm_pist_psd) * (atm_params['wvl']/(2*np.pi))**2)),label='atmospheric piston')
""" #to check slopes match theory uncomment this
plt.loglog(fs, np.sqrt(fs**(-17/3)*fs**4),label = r'$f^{2-17/6}$')#label='{}'.format(round(1e6 * 1/9.81 * 10**(-4.5),3))+r'${}\mu g/ \sqrt{Hz}$') 
plt.loglog(fs, np.sqrt(fs**(-8/3)*fs**4),label = r'$f^{2-8/6}$')#label='{}'.format(round(1e6 * 1/9.81 * 10**(-4.5),3))+r'${}\mu g/ \sqrt{Hz}$') 
"""
plt.axhline(0.3,color='k',linestyle='--',label='B&K accelerometer type 4370')
plt.ylabel(r'$\mu g\ /\ \sqrt{Hz}}$',fontsize=20)
plt.xlabel('frequency (Hz)',fontsize=20)
plt.gca().tick_params(labelsize=20)
plt.text(1e-3,1e-2,r'$D={}m, r_0={}m, \tau_0={}ms, L_0={}\ m$'.format(atm_params['diam'],round(atm_params['r_0']*(0.5/2.2),1),round(1e3*atm_params['tau_0']*(0.5/2.2)**(6/5),1), atm_params['L_0'] ),fontsize=20)
plt.legend(fontsize=15)
plt.grid()
plt.tight_layout()
#plt.savefig('accelerometer_requirements_to_reach_atmopshere.png')
