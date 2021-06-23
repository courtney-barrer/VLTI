#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:06:28 2021

@author: bcourtne

functions to calculate atmopsheric piston (and fringe motion) PSD taken from Conan 1995
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp



def atm_piston(wvl=2.2e-6, seeing=0.86, tau_0=4e-3, L_0=np.inf, diam=8):
    #calculates the atmospheric piston taken from Conan 1995.
    #returns tuple of piston (f,PSD) and dictionary of the atmospheric inputs
    
    #-----Atmospheric Parameters
    
    r_0 = 0.98*wvl/np.radians(0.86/3600)  #fried parameter at wavelength in m.
    tau_0 = (2.2/0.5)**(6/5) * 4e-3        #atm coherence time at wavelength
    v_mn = 0.314 * r_0 / tau_0               #AO definition for vbar   
    
    atm_params = {'wvl':wvl,'seeing':seeing,'r_0':r_0,'tau_0':tau_0,'L_0':L_0,'diam':diam}
    #-----configuration parameters
    
    #ky needs to have good sampling at small values where VK_model * tel_filter is maximum for the integration (course sampling outside of this since contribution is negligble)
    ky = np.concatenate([-np.logspace(-10,3,2**11)[::-1], np.logspace(-10,3,2**11) ] )  # Spatial frequency 
    nf = int(2**13)    #number of frequency samples
    minf = 1e-3  #min freq
    maxf = 5e2   #max freq
    fs = np.linspace(minf,maxf,nf)
    
    
    #-----piston PSD (rad^2/Hz)
    
    atm_pist_psd = [] #atmosphere piston PSD in rad^2/Hz 
    for i, f in enumerate(fs):
        kx = f/v_mn
        k_abs = np.sqrt(kx**2 + ky**2)
        
        #for kolmogorove PSD set L_0 to inf
        VK_model = 0.0229 * r_0**(-5/3.) * (L_0**(-2) + k_abs**2)**(-11/6.) #von karman model (Conan 2000)
        
        tel_filter = (2*sp.jv(1,np.pi*diam*k_abs)/(np.pi*diam*k_abs))**2
        Phi_opd = VK_model * tel_filter 
        atm_pist_psd.append( np.trapz(Phi_opd, ky)/v_mn ) 

    return((fs, atm_pist_psd), atm_params)
    




def atm_diff_piston(wvl=2.2e-6, seeing=0.86, tau_0=4e-3, L_0=np.inf, B=[0,60], diam=8):
    #calculates the differential atmospheric piston (fringe motion) taken from Conan 1995.
    #B is 2D telescope positions (defines baseline)
    
    #-----Atmospheric Parameters 
    
    r_0 = 0.98*wvl/np.radians(0.86/3600)  #fried parameter at wavelength in m.
    tau_0 = (2.2/0.5)**(6/5) * 4e-3        #atm coherence time at wavelength
    v_mn = 0.314 * r_0 / tau_0               #AO definition for vbar   
    
    atm_params = {'wvl':wvl,'seeing':seeing,'r_0':r_0,'tau_0':tau_0,'L_0':L_0,'B':B,'diam':diam}
    #-----configuration parameters
    
    #ky needs to have good sampling at small values where VK_model * tel_filter is maximum for the integration (course sampling outside of this since contribution is negligble)
    ky = np.concatenate([-np.logspace(-10,3,2**11)[::-1], np.logspace(-10,3,2**11) ] ) # Spatial frequency  
    nf = int(2**13)    #number of frequency samples
    minf = 1e-3  #min freq
    maxf = 5e2   #max freq
    fs = np.linspace(minf,maxf,nf)
    
    
    #-----piston PSD (rad^2/Hz)
    
    atm_dpist_psd = [] #to hold diff piston PSD (rad^2/Hz)
    for i, f in enumerate(fs):
        kx = f/v_mn
        k_abs = np.sqrt(kx**2 + ky**2)
        VK_model = 0.0229 * r_0**(-5/3.) * (L_0**(-2) + k_abs**2)**(-11/6.) #von karman model (Conan 2000)
        tel_filter = (2*sp.jv(1,np.pi*diam*k_abs)/(np.pi*diam*k_abs))**2
        bdotkappa = B[0]*kx + B[1]*ky
        Phi_opd = VK_model * tel_filter * 4 * np.sin(np.pi*bdotkappa)**2
        
        atm_dpist_psd.append( np.trapz(Phi_opd, ky)/v_mn )

    return((fs, atm_dpist_psd), atm_params)
    

