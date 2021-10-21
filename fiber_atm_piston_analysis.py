#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 10:40:20 2021

@author: bcourtne


SCRIPT OUTLINE 
=====


#Section 0 - imports and function (to hold final atm_apodized_piston function)

#Section 1 - plots to undersand power laws of Von-Karman / Kolmogorov piston filter PSDs

#Section 2 - qualatative visualizations of apodization influence on piston filtering (plot for paper)
                 # test convolution / sampling regimes 

#Section 3 - qualatative visualizations of central obscuration influence on piston (plot for paper)

#Section 4 - Quantitative analysis (verbose) of full apodized piston with realistic parameters 
                # this will be used to write/test the atm_apodized_piston

#Section 5 - derivation of theoretical acceleration PSD for VLTI (comparisoon to accelerometersw noise floor)


#Section 6 - comparison of theoretical VLTI UT/Gravity apodized piston to measured psuedo open loop on Gravity Fringe Tracker 



To DO
=====
sections 4-5

    
    M_G *= 1/np.sqrt( np.trapz(np.trapz(abs(M_G)**2, x1) ,x1) ) / len(x1)
    why len(x1) here to reproduce other results? 
    
    seems to be unsensitive to m2 radius.. doesn't make sense?'
    
    [DONE] understand units of Roddiers equation/gravity parameters comparing a-wvl/D
        - in Roddier a is in meters 
        - in Conan k is spatial frequency 1/m => piDk product is unitless => angle on sky.. 

    len(M_G) off by 1 compared to len(kx) sometimes 
    
    play with : to get good freq grid /smapling..

    
"""

import os 
os.chdir( '/Users/bcourtne/Documents/Hi5/vibration_analysis/utilities')

import fiber_fields as ff
import atmosphere_piston as ap 

import scipy
import numpy as np
import matplotlib.pyplot as plt 
from scipy import special as sp
from scipy import signal
import aotools


def overlap_int(E1, E2): 
    #fiber overlap integral to calculate coupling efficienccy 
    eta = abs( np.nansum( E1 * np.conjugate(E2) ) )**2 / ( np.nansum( abs(E1)**2 ) * np.nansum( abs(E2)**2 ) )
    
    return(eta)

def atm_w_fiber_piston(wvl=2.2e-6, NA=0.12, n_core=1, a=3e-6, seeing=0.86, \
                   tau_0 = 4e-3, L_0 = np.inf, diam=8,  diam_m2 = None, N_grid = 2**12):
    
    #-----fiber parameters
    #permitivity free space
    epsilon_o , mu_o = 8.854188e-12, 1.256637e-6
    #cladding index refraction
    n_clad = np.sqrt(n_core**2 - NA**2)    
    #V number (relates to number of modes fiber can hold)
    V = 2*np.pi*NA*a/wvl
    #simplifying coefficient
    cH = np.sqrt(2*n_clad)*(epsilon_o/mu_o)**(0.25)/np.sqrt(np.pi)    
    #beam waist (1/e)
    w = a * (0.65 + 1.619/V**(3/2) + 2.879/V**6) # same unit as a (m)
    
    
    #----- atmospheric parameters
    r_0 = 0.98*wvl/np.radians(seeing/3600)  #fried parameter at wavelength in m.
    tau_0 = (wvl*1e6 /0.5)**(6/5) * 4e-3        #atm coherence time at wavelength
    v_mn = 0.314 * r_0 / tau_0  #AO definition for vbar   

    # our spatial frequencies for convolution 
    kx, ky = np.linspace(-1e2,1e2,N_grid), np.linspace(-1e2,1e2,N_grid) #np.concatenate([-np.logspace(-4,3,2**12)[::-1], np.logspace(-4,3,2**12) ] ) , np.concatenate([-np.logspace(-10,3,2**12)[::-1], np.logspace(-10,3,2**12) ] )  #np.linspace(-5e2,5e2,2**13), np.linspace(-5e2,5e2,2**13)
    #mesh it
    kxx, kyy = np.meshgrid( kx, ky )

    # fourier transform of fiber mode in pupil plane (fiber mode in focal plane under Roddier approximation )
    kx_o, ky_o = 0, 0 
    fiber_mode = cH/w * np.exp( -(wvl*((kxx-kx_o)**2 +  (kyy-ky_o)**2) ) / w**2 ) 
    # normalize as in Woillez ( int int |field|^2 dA = 1 )
    fiber_mode *= 1/ np.sqrt( np.trapz( [ np.trapz(fiber_mode[:,i]**2,kx) for i in range(fiber_mode.shape[1]) ], ky) ) 
    
    # fourier transform of telescope pupil 
    if diam_m2:
        tel_pupil_fft = sp.jv(1,np.pi*diam*np.sqrt(kxx**2 + kyy**2))/(np.pi*diam*np.sqrt(kxx**2 + kyy**2) ) - sp.jv(1,np.pi*diam_m2*np.sqrt(kxx**2 + kyy**2))/(np.pi*diam_m2*np.sqrt(kxx**2 + kyy**2) )
    else:
        tel_pupil_fft = sp.jv(1,np.pi*diam*np.sqrt(kxx**2 + kyy**2))/(np.pi*diam*np.sqrt(kxx**2 + kyy**2) )
    #F[AB] = F[A] * F[B] (where * is convolution and F[A] is Fourier transform)
    # therefore we create our filter
    M_G = scipy.signal.fftconvolve(fiber_mode, tel_pupil_fft, mode='same') / np.sqrt( len(tel_pupil_fft) ) 
    # note: sqrt(len(x)) is because we take |M_G|^2 in the integral
    
    # ===========
    # now we define our grid that we are going to integrate over to get the PSD as a subgrid of our convolution grid
    # kx = fs / v_mn => fs = kx * v_mn !!!
    
    # make our integration grid 
    kx2 = kx[kx > 0] # we only consider positive frequecies for PSD (kx = fs/v_mn)
    fs = kx2 * v_mn  #frequencies for our final psd 
    kxx2, kyy2 = np.meshgrid(kx2,ky)
    M_G2 = M_G[:,len(M_G)//2:] # we extract the integration subgrid from M_G
    # Von-Karman model (Conan 2000)
    VK_model = 0.0229 * r_0**(-5/3.) * (L_0**(-2) + (kxx2**2 + kyy2**2) )**(-11/6) 
    
    # now integrate filtered VK_model over ky
    apod_piston = np.array( [ np.trapz( abs(M_G2[:,i])**2 * VK_model[:,i], ky ) for i in range(VK_model.shape[1])] )

    return( ( fs, apod_piston ) )



#%%  #Section 1 - plots to undersand power laws of Von-Karman / Kolmogorov piston filter PSDs
#========= UNDERSTANDING SLOPES AND REGIMES OF CONTRIBUTING FUNCTIONS 


#VON KARMAN & KOLMOGOROV
x=np.logspace(-5,3,1000)
r_0 = 0.1 #m
plt.figure()
L_0 = 25
plt.loglog(x,  0.0229 * r_0**(-5/3.) * (L_0**(-2) + x**2 )**(-11/6)  ,color='g',label='Von Karman (r0=0.1m, L0=25m)')
plt.axvline(1/L_0,color='g',linestyle=':',label='1/L0')

L_0 = 100
plt.loglog(x,  0.0229 * r_0**(-5/3.) * (L_0**(-2) + x**2 )**(-11/6)  ,color='b',label='Von Karman (r0=0.1m, L0=100m)')
plt.axvline(1/L_0,color='b',linestyle=':',label='1/L0')

L_0 = np.inf
plt.loglog(x,  0.0229 * r_0**(-5/3.) * (L_0**(-2) + x**2 )**(-11/6)  ,label='Kolmogorov (r0=0.1m)')

plt.loglog(x, 100* x**(-11/3),linestyle=':',color='k',label=r'$f^{-11/3}$')

plt.legend(bbox_to_anchor=(1,1),fontsize=15)
plt.xlabel('x',fontsize=15)
plt.ylabel('y',fontsize=15)


#BESSEL FUNCTION 
plt.figure()
aaa = 3
plt.loglog(x, abs( sp.jv(1,aaa*x)/(aaa*x) )**2,label=r'$|J_v(ax)/ax|^2$')
plt.loglog(x, x**0,label=r'$f^{0}$')
plt.loglog(x, x**-3,label=r'$f^{-3}$')
plt.axvline(1/aaa,color='b',linestyle=':',label='1/a')

plt.legend(bbox_to_anchor=(1,1), fontsize=15)
plt.xlabel('x',fontsize=15)
plt.ylabel('y',fontsize=15)


#PRODUCT 
plt.figure()
L_0 = 100
plt.loglog(x, abs(sp.jv(1,aaa*x)/(aaa * x))**2 *  0.0229 * r_0**(-5/3.) * (L_0**(-2) + x**2 )**(-11/6), label=r'$|J_v(x)|^2 * VK(x)$')
plt.loglog(x, 100* x**(-20/3),linestyle=':',color='k',label=r'$f^{-20/3}$')
plt.loglog(x, 100* x**(-11/3),linestyle=':',color='k',label=r'$f^{-11/3}$')
plt.loglog(x, 1e10* x**(0),linestyle=':',color='k',label=r'$f^{0}$')
plt.axvline(1/aaa,color='b',linestyle=':',label='1/a')
plt.axvline(1/L_0,color='g',linestyle=':',label='1/L0')
plt.legend(bbox_to_anchor=(1,1), fontsize=15)
plt.xlabel('x',fontsize=15)
plt.ylabel('y',fontsize=15)


#GUASSIAN 
plt.figure()
plt.loglog(x, np.exp(-x**2),label = r'$exp(-x^2)$')
plt.legend()



#%% #Section 2 - qualatative visualizations of apodization influence on piston filtering (plot for paper) 

#figuring out how to normalize... 
# FILTER HAS TO BE NORMALIZED SUCH THAT int |M|^2 = 1 
D = 1 #m 
sigma = 1e-2 #m 
 
# set limits based on fiber waist / diffraction limit and have at least 10 samples across min waist
#xmax = 1e2*np.max([sigma,D])
#xmin = -xmax
xmax,xmin = 2000,-2000
#x = np.linspace(xmin, xmax, int( 12 * xmax/np.min([sigma,D])) )
x = np.linspace(xmin, xmax, 12 * xmax)


#(b)smaller moving winow
x2 = x[abs(x)<10*sigma]
s2 = 1/(sigma * np.sqrt(np.pi*2)) * np.exp( -(x2/sigma)**2 )
#s2 *= 1/np.sqrt( np.trapz(s2**2, x2) ) # 1/np.trapz(s2, x2)  #

#(a)larger stationary window 
s1 = 2 * sp.jv(1,np.pi*D*x)/(np.pi * D * x)  #normalized so that integral is 1!
#s1 *= 1/np.sqrt(np.trapz(s1**2, x)) #1/np.trapz(s1, x) #
#normalize over the smaller grid 
#s1 *= 1/np.sqrt( np.trapz( ( sp.jv(1,D*x2)/(D * x2) )**2 ) )#1/np.trapz(sp.jv(1,D*x2)/(D * x2) )

####
# LOOK AT SAMPLING OF SIGNALS BEFORE CONVOLVING
####
plt.figure()
plt.plot(x,s1 , label='a')

plt.plot(x2,s2 , label='b')
plt.legend()



####
# (for figure) LOOK AT THE TWO SIGNALS (COMPARE THEIR RELATIVE WIDTHS)
####

xx = np.linspace(-20*np.max([D,sigma]), 20*np.max([D,sigma]), 1000)
plt.figure()
plt.plot(xx,sp.jv(1,D*xx)/(D * xx) , label='a',lw=3)
plt.xticks([])
plt.yticks([])
plt.axis('off')

plt.figure()
plt.plot(xx,np.exp( -(xx/sigma)**2 ), label='b',lw=3)
plt.xticks([])
plt.yticks([])
plt.axis('off')
#convolve (a*b)
s3 = np.convolve( s1, s2, 'valid') #/ len(s2)  #*  len(x)
x3 = x[len(x2)//2 : len(x)-len(x2)//2+1]

s3*= 1/np.sqrt( np.trapz(abs(s3)**2, x3) )# 1/np.sqrt( np.trapz(s3**2, x3) )

s_ref = 1/np.sqrt( np.trapz(abs(s1)**2, x) ) * s1
####
# NOW LOOK HOW CONVOLUTION EFFECTS POWER LAWS / PSD KNEES
####
plt.figure(figsize=(10,7))
plt.loglog(x3[x3>0], abs(s3[x3>0])**2, label = 'fiber apodized filter',lw=3)

plt.loglog(x[x>0],abs(s_ref[x>0])**2, label='non apodized filter',linestyle=':',color='k',alpha=0.8,lw=2)


#import matplotlib.ticker

#y_major = matplotlib.ticker.LogLocator(base = 10.0, numticks = 5)
#plt.gca().yaxis.set_major_locator(y_major)
#y_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
#plt.gca().yaxis.set_minor_locator(y_minor)
#plt.gca().yaxis.set_minor_formatter(plt.ticker.NullFormatter())
plt.tight_layout()
#plt.loglog(x[x>0],abs(s_ref[x>0])**2, label='non apodized filter')
#plt.legend(fontsize=30,bbox_to_anchor=(1,1))#loc='bottom left')
#[i.set_linewidth(0.1) for i in plt.gca().spines]
#plt.gca().spines['bottom'].set_linewidth(3)
plt.grid(True, which="both", ls=":",alpha=0.3,color='k')

plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])

plt.ylim([1e-19,10])
plt.xlim([1e-1,1e2])


#----- WHAT IVE LEARNED
# - need even grid for conv
# - need good sampling over each curve to be convolved
# - need valid convolution region (convolve option) - see above how to filter x coord too



#%% Section 3 - qualatative visualizations of central obscuration influence on piston (plot for paper)

"""
open pupil piston
central obs piston 
apodized piston 
central ob apodized piston

Should reproduce following limiting behaviour
============================================
d<<D => open pupil piston = central obs piston 
^^ This isnt working??? 


sigma << 1 (delta func relative to d, D) => central obs piston = central obs apodized piston 

for the paper just do plot of apodized (weak apodization) with / without central obs showing res peak in filter
note resonance knee cut off (to -17/3) occurs at roughly 1/d
also initial plot showing M1, M2, VLTI pupil
"""


D = 8 #m 
d = 1
sigma = 0.1 #m 


# looking at M2 filter vs M1 
# ==============
x_tmp = np.linspace(-5,5,9900)
s1_obsc =  (sp.jv(1,np.pi*D*x_tmp)/(np.pi * D * x_tmp) -  (d/D)**2 * sp.jv(1, np.pi*d*x_tmp)/(np.pi * d * x_tmp) )  #normalized so that integral is 1!

s1_m2 = (d/D)**2 * sp.jv(1, np.pi*d*x_tmp)/(np.pi * d * x_tmp)
s1_open =  sp.jv(1,np.pi*D*x_tmp)/(np.pi * D * x_tmp)  #normalized so that integral is 1!

plt.figure()
plt.semilogy(x_tmp,s1_open,label='M1 (D=8m)')
plt.semilogy(x_tmp,s1_m2,label='M2 (D=1m)')
plt.semilogy(x_tmp,s1_obsc,label='VLTI pupil')
plt.legend()


# NOW look at convolution 
# ============== 
# set limits based on fiber waist / diffraction limit and have at least 10 samples across min waist
#xmax = 1e2*np.max([sigma,D])
#xmin = -xmax
xmax,xmin = 2000,-2000
#x = np.linspace(xmin, xmax, int( 12 * xmax/np.min([sigma,D])) )
x = np.linspace(xmin, xmax, 100 * xmax)


#(b)smaller moving winow
x2 = x[abs(x)<1000*sigma]
s2 = 1/(sigma * np.sqrt(np.pi*2)) * np.exp( -(x2/sigma)**2 )
#s2 *= 1/np.sqrt( np.trapz(s2**2, x2) ) # 1/np.trapz(s2, x2)  #

#(a)larger stationary window 
s1_obsc = 2 * (sp.jv(1,np.pi*D*x)/(np.pi * D * x) - (d/D)**2 * sp.jv(1, np.pi*d*x)/(np.pi*d*x) )  #normalized so that integral is 1!
s1_open = 2 * sp.jv(1,np.pi*D*x)/(np.pi * D * x)  #normalized so that integral is 1!




####
# LOOK AT SAMPLING OF SIGNALS BEFORE CONVOLVING
####
"""plt.figure()
plt.plot(x,s1_obsc , label='a')

plt.plot(x2,s2 , label='b')
plt.legend()"""

############
#convolve (a*b)
x3 = x[len(x2)//2 : len(x)-len(x2)//2+1]

s3 = np.convolve( s1_obsc, s2, 'valid') #/ len(s2)  #*  len(x)
s3 *= 1/np.sqrt( np.trapz(abs(s3)**2, x3) )# 1/np.sqrt( np.trapz(s3**2, x3) )

#make non-obscuerd apodized pupil  reference
s_apo = np.convolve( s1_open, s2, 'valid') #/ len(s2)  #*  len(x)
s_apo *= 1/np.sqrt( np.trapz(abs(s_apo)**2, x3) )

#make obscuerd pupil  reference (have to use x3 not x)
s_obs =  (sp.jv(1,np.pi*D*x3)/(np.pi * D * x3) - (d/D)**2 * sp.jv(1, np.pi*d*x3)/(np.pi*d*x3) )  #normalized so that integral is 1!
s_obs *= 1/np.sqrt( np.trapz(abs(s_obs)**2, x3) )

#make open pupil reference
s_ope = 2 * sp.jv(1,np.pi*D*x3)/(np.pi * D * x3)  #normalized so that integral is 1!
s_ope *= 1/np.sqrt( np.trapz(abs(s_ope)**2, x3) )


####
# NOW LOOK HOW CONVOLUTION EFFECTS POWER LAWS / PSD KNEES
####
plt.figure(figsize=(10,7))
plt.loglog(x3[x3>0], abs(s3[x3>0])**2, label = 'fiber apodized, obscured filter', color='k', lw=3)

plt.loglog(x3[x3>0],abs(s_apo[x3>0])**2, label='fiber apodized, open filter',linestyle=':',color='g',alpha=0.8,lw=2)

plt.loglog(x3[x3>0],abs(s_ope[x3>0])**2, label='non apodized, open filter',linestyle=':',color='r',alpha=0.8,lw=2)

plt.loglog(x3[x3>0],abs(s_obs[x3>0])**2, label='non apodized, obscured filter',linestyle=':',color='b',alpha=0.8,lw=2)

plt.legend(fontsize=15)
plt.title(f'D={D}, d={d}, sigma={sigma}',fontsize=15)




#%% Section 4 - Quantitative analysis (verbose) of full apodized piston with realistic parameters 

"""
"The Gravity waveguides have a mode-field radius of 3.83 μm at λ = 2150 nm, 
and are therefore single-mode across the full K-band, down to 1.85 μm, 
allowing to transmit the metrology laser (λ = 1908 nm) in a single-mode regime
in the backward direction [Blind, 2015]  

Since the measurement of the mode field radius is very difficult, 
we measure the numerical aperture (NA) that is directly related to it. 
This characterization has been performed on several chips before gluing with a 
fiber array. We estimated NA by recording the far field using a linear infrared 
InSb detector. From these measurements, we computed mode simulations to deduce 
the numerical aperture of our waveguides. Our measurements gives a 
NA of 0.21 ± 0.015 which corresponds to a waist of ω0= 3.9 ± 0.1 μm which complies with our requirements.

The index difference between the core and the cladding has to be increased to 
reach Δn=0.16 to match the mode field radius requirements  ""

- (L. Jocou, "The beam combiners of Gravity VLTI instrument: concept, development, and performance in laboratory," 2014). Floride glass n=1.5 (roughly take this as core)

"""


# ==============================================================================
#                           PARAMETERS 
#========== Config parameters 
wvl = 2.2e-6
N_vis = 2**8 #grid samples for visualizing pupil 
# note we define grid samples for convution automatically from fiber to diffraction waist ratio 

#========== Telescope parameters 
D = 8 #m
d = 1#1e-3 #1e-3 #m


#========== fiber parameters 
NA = 0.21 # #numerical apperature 
#n_core = 1 #refractive index of core
delta_n = 0.16 
a = 3.9e-6 #fiber waist (m?)
foc_len = 20 #focal lenght (m) 


#Vacuum permittivity and  vacuum permeability
epsilon_o , mu_o = 8.854188e-12, 1.256637e-6 #
#cladding index refraction
"""
[1]   n_core**2 - n_clad**2 = NA**2
[2]   n_core - n_clad = delta_n    
=> 
[3]  (delta_n + n_clad)**2 - n_clad**2 = NA**2
[4]   delta_n**2 + 2*delta_n * n_clad = NA**2
=> 
[5]   n_clad = (NA**2 - delta_n**2) / (2 * delta_n)
 n_clad seems too small with the gravity parameters taken from Jocou 2014
"""
n_clad = (NA**2 - delta_n**2) / (2 * delta_n)
#V number (relates to number of modes fiber can hold)
V = 2*np.pi*NA*a/wvl
#simplifying coefficient
cH = np.sqrt(2*n_clad)*(epsilon_o/mu_o)**(0.25)/np.sqrt(np.pi)    

#beam waist (1/e)
w = a * (0.65 + 1.619/V**(3/2) + 2.879/V**6) # here waist units are in meters


#========== Atmospheric Parameters
seeing = 0.86
r_0 = 0.98*wvl/np.radians(seeing/3600)  #fried parameter at wavelength in m.
tau_0 = (2.2/0.5)**(6/5) * 4e-3        #atm coherence time at wavelength
v_mn = 0.314 * r_0 / tau_0               #AO definition for vbar   
L_0 = np.inf

#atm_params = {'wvl':wvl,'seeing':seeing,'r_0':r_0,'tau_0':tau_0,'L_0':L_0,'diam':diam}

# ==============================================================================


#for a in np.logspace(-7,-4,12):
#print(a)
# ==============================================================================
#                          pupil visulization 

#focal plane coordinates
f_array = np.linspace(-1e-5,1e-5, N_vis ) 
# relate pupil plane coordinates to focal through wvl
p_array = f_array / wvl

#create coordinate grids 
x_f,y_f = np.meshgrid(f_array, f_array)
x_p,y_p = np.meshgrid(p_array,p_array)

#differential elements 
dx_f = np.diff(x_f)[0][0]
dx_p = np.diff(x_p)[0][0]

#search where pupil coord = 4 (radius for VLT) then divide by 2 since x_p is symmetric around 0
pupil_m1 = aotools.functions.pupil.circle(radius= np.argmin(abs(x_p[x_p>0]-D/2)) , size=N_vis , circle_centre=(0, 0), origin='middle')
#m2 diameter = 1 m
pupil_m2 =  aotools.functions.pupil.circle(radius= np.argmin(abs(x_p[x_p>0]-d/2)) , size=N_vis , circle_centre=(0, 0), origin='middle')

#consider with and without central obscuration
pupil = pupil_m1
pupil_obs = pupil_m1 - pupil_m2 

#create masks for calculating statistics in the pupil (0->nan)
pupil_mask = pupil.copy() #need to mask outside pupil by nan for calculating averages 
pupil_mask[pupil==0] = np.nan

pupil_obs_mask = pupil_obs.copy()
pupil_obs_mask[pupil_obs_mask==0] = np.nan

field_pupil = cH/np.sqrt(np.pi) * np.exp(-w**2 * (x_p**2 + y_p**2) )
field_pupil *= 1/np.max(field_pupil) #normalized for easy visualization

field_focal = cH/w * np.exp(-(np.sqrt(x_f**2 + y_f**2)/w)**2)
field_focal *= 1/np.max(field_focal) 

plt.figure(figsize=(5,5))
plt.title('fiber apodized pupil')
plt.pcolormesh(p_array, p_array, pupil_obs_mask * np.log10(field_pupil),vmin=0,vmax=1)
plt.gca().set_aspect('equal')
plt.xlabel('x [meters]' , fontsize=15)
plt.ylabel('y [meters]' , fontsize=15)
plt.gca().tick_params(labelsize=15)


#overlap efficiency
eta = overlap_int(pupil_obs_mask, np.exp(1j*field_pupil) )
eta_max = overlap_int(pupil_obs_mask, pupil_obs_mask )
print(f'\nfiber coupling efficiency eta = {eta}\n')


# ==============================================================================

# ==============================================================================
#               calculating the corresponding apodized piston 

# get fiber field and PSF widths to creat reasonable grids for convolution
#      - create two (equally sampled) grids for convolution 
#      - for each have at least 12 samples across lambda/D or fiber waist 
#      - extend grids to sample at least "rel_size_factoor" greater then largest waist (or should it be where function drops 10 magnitudes below peak?)

#      - to get reasonable frequencies it will be necessary to do log sampling for larger grid outside region of interest
#           - therefore linear sampling where convolution matters (functions are non-negligible), 
#           - log sampling outside of this region     


# setting our sample spacing beased on beam widths
dx = np.min([w, (wvl/D) * foc_len] ) / 20 # at least 10 samples across 

x1 = np.arange(-100 * w , 100 * w , dx)  #fiber linear grid 
x2 = np.arange(-100 * (wvl/D) * foc_len , 100 * (wvl/D) * foc_len, dx) #diffraction linear grid

#frequencies for our final psd (definied by windspeed)
fs = v_mn / x1 

#MESH THE GRIDS 
if len(x1)< 1e5: # check the biggest grid isn't tooo big before meshing
    
    xx1, yy1 = np.meshgrid(x1,x1)
    xx2, yy2 = np.meshgrid(x2,x2)
    kxx, kyy = np.meshgrid(x1,x1)
    
else:
    print('kx1 grid very big.. consider resizing')



# get the meshed modes 
k_abs = np.sqrt(xx2**2 + yy2**2)/ ( wvl * foc_len ) # pupil spatial freq (1/m - as in Conon 1995)

tel_mode = ( np.pi * D**2/( wvl * foc_len ) ) * ( 2 * (sp.jv(1,np.pi * D * k_abs )/(np.pi * D * k_abs ) \
    - (d/D)**2 * sp.jv(1,np.pi * d * k_abs)/(np.pi * d * k_abs ) ) )
#tel_mode *=  1/np.sqrt( np.trapz(np.trapz(abs(tel_mode)**2, x2) , x2) )
      
x_o, y_o = 0,0
fib_mode = cH/(w/foc_len) * np.exp( -( (xx1-x_o)**2 +  (yy1-y_o)**2 )  / w**2 ) 
#fib_mode *= 1/np.sqrt( np.trapz(np.trapz(abs(fib_mode)**2, x1) , x1) )

# plot functions before convoling 
verbose =1 
if verbose:
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    filt1 = abs(x1) < 6* wvl/D * foc_len
    filt2 = abs(x2) < 6* wvl/D * foc_len
    ax[0].pcolormesh(x2[filt2], x2[filt2], tel_mode[filt2,:][:,filt2])
    ax[0].set_title('tel mode')
    ax[1].pcolormesh(x1[filt1],x1[filt1], fib_mode[filt1,:][:,filt1])
    ax[1].set_title('fib mode')
    ax[0].set_xlabel('x [meters]')
    ax[0].set_ylabel('y [meters]')
    ax[1].set_xlabel('x [meters]')
    ax[1].set_ylabel('y [meters]')
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')

##### OUR FILTER ######    
M_G = scipy.signal.fftconvolve(fib_mode, tel_mode, mode='same')  
#normalze such that int |M_G|^2 dA = 1
#M_G *= 1/np.sqrt( np.trapz(abs(M_G)**2, x1,axis=1) )
M_G *= 1/np.sqrt( np.trapz(np.trapz(abs(M_G)**2, x1) ,x1) ) / len(x1)

# atmospheric phase PSD - Von-Karman model (Conan 2000)
k_abs = np.sqrt( (xx1**2 + yy1**2) ) / (foc_len * wvl) #spatial frequency (1/m)
VK_model = 0.0229 * r_0**(-5/3.) * (L_0**(-2) + k_abs**2 )**(-11/6) 
#VK_model = 0.0229 * r_0**(-5/3.) * (L_0**(-2) + (xx1**2 + yy1**2) )**(-11/6) 

verbose = 0 
if verbose:
    fig,ax = plt.subplots(2,2,figsize=(10,5))
    filt1 = abs(x1) < 6* wvl/D * foc_len
    filt2 = abs(x2) < 6* wvl/D * foc_len
    ax[0,0].pcolormesh(x2[filt2], x2[filt2], tel_mode[filt2,:][:,filt2])
    ax[0,0].set_title('tel mode')
    ax[0,1].pcolormesh(x1[filt1],x1[filt1], fib_mode[filt1,:][:,filt1])
    ax[0,1].set_title('tel mode')    
    ax[1,0].pcolormesh(x1[filt1],x1[filt1], M_G[filt1,:][:,filt1])
    ax[1,0].set_title('M_G *')
    ax[1,1].pcolormesh(x1[filt1],x1[filt1], np.log10(abs(M_G[filt1,:][:,filt1])**2 * VK_model[filt1,:][:,filt1]))
    ax[1,1].set_title('|M_G|^2 * VK')    
    
    ax[1].set_title('fib mode')
    ax[0].set_xlabel('x [meters]')
    ax[0].set_ylabel('y [meters]')
    ax[1].set_xlabel('x [meters]')
    ax[1].set_ylabel('y [meters]')
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    
f_indx = np.where(x1>0)[0] 
apo_piston = 1/v_mn * np.array( [ np.trapz( abs(M_G[i,:])**2 * VK_model[i,:], x1 ) for i in f_indx ] )


# compare it to non-apodized atmospheric piston 
atm_piston , _ = ap.atm_piston(wvl= wvl, seeing=seeing ,tau_0 = tau_0, L_0 = np.inf, diam = D)

plt.figure()
plt.loglog(v_mn * x1[x1>0]/(foc_len * wvl), apo_piston, label='apodized');
plt.loglog(*atm_piston, label='non-apodized')
plt.legend(fontsize=15)    
plt.xlim(1e-2,500)


#%%











##### K array is POSITION !!!!!
# LONGER ARRAY (in linspaced grid)
k1_lin = np.linspace(-100*w, 100*w, 1000) 
# logspacing outside of this (so we can get a reasonable frequency range)
k1_log = np.logspace(np.log10(xmax), 5, 2**10) #play here to get reasonable freq range 
# now append them 
kx1 = k1_lin #np.array(list(-k1_log[::-1]) + list(k1_lin) + list(k1_log))
kx2 = np.linspace(-100*w, 100*w, 1000)
kx = kx1[len(kx2)//2  : len(kx1)-len(kx2)//2+1] 
ky = kx

#frequencies for our final psd (definied by windspeed)
fs = v_mn / kx  

verbose=1
if verbose:
    #1D plot to see lif linear region makes sense 
    #Vacuum permittivity and  vacuum permeability
    epsilon_o , mu_o = 8.854188e-12, 1.256637e-6 #
    #cladding index refraction
    n_clad = np.sqrt(n_core**2 - NA**2)    
    #V number (relates to number of modes fiber can hold)
    V = 2*np.pi*NA*a/wvl
    #simplifying coefficient
    cH = np.sqrt(2*n_clad)*(epsilon_o/mu_o)**(0.25)/np.sqrt(np.pi)    
    
    #beam waist (1/e)
    w = a/foc_len * (0.65 + 1.619/V**(3/2) + 2.879/V**6) # here waist units are in radians (on sky)
    
    
    fib_mode = cH/w * np.exp( -( kx1**2 / w**2 ) ) 
    
    tel_mode = ( np.pi * D**2/( wvl * foc_len ) ) * 2 * (sp.jv(1,np.pi *D * kx2/( wvl * foc_len ) )/(np.pi *D* kx2/( wvl * foc_len ) )  \
            - (d/D)**2 * sp.jv(1,np.pi * d * kx2/( wvl * foc_len ) )/(np.pi * d * kx2/( wvl * foc_len ) ) )
    plt.figure()
    plt.plot(kx1, 100*fib_mode)
    plt.plot(kx2, tel_mode)
    plt.xlabel('position (m)')



#MESH THE GRIDS 
if len(kx1)< 1e5: # check the biggest grid isn't tooo big before meshing
    
    k1xx, k1yy = np.meshgrid(kx1,kx1)
    k2xx, k2yy = np.meshgrid(kx2,kx2)
    kxx, kyy = np.meshgrid(kx,kx)
    
else:
    print('kx1 grid very big.. consider resizing')



#note these are both in units of radians (on sky) 
# alpha = D*k => d*k = alpha*d/D (where we have bad labels alpha = kxx etc)
def build_tel_mode(D,d,kx,ky):
    
    tel_mode = ( np.pi * D**2/( wvl * foc_len ) ) * 2 * (sp.jv(1,np.pi * D * np.sqrt(kx**2 + ky**2)/ ( wvl * foc_len ) )/(np.pi * D * np.sqrt(kx**2 + ky**2) ) \
        - (d/D)**2 * sp.jv(1,np.pi * d * np.sqrt(kx**2 + ky**2)/( wvl * foc_len ))/(np.pi * d * np.sqrt(kx**2 + ky**2)/( wvl * foc_len ) ) )
    
    return(tel_mode)


def build_fib_mode(wvl,n_core,NA,a,foc_len, kx,ky,kx_o,ky_o):
    
    
    #Vacuum permittivity and  vacuum permeability
    epsilon_o , mu_o = 8.854188e-12, 1.256637e-6 #
    #cladding index refraction
    n_clad = np.sqrt(n_core**2 - NA**2)    
    #V number (relates to number of modes fiber can hold)
    V = 2*np.pi*NA*a/wvl
    #simplifying coefficient
    cH = np.sqrt(2*n_clad)*(epsilon_o/mu_o)**(0.25)/np.sqrt(np.pi)    
    
    #beam waist (1/e)
    w = a/foc_len * (0.65 + 1.619/V**(3/2) + 2.879/V**6) # here waist units are in radians (on sky)

    #WHY DO I HAVE WVL HERE
    fib_mode = cH/(w/foc_len) * np.exp( -( (kx-kx_o)**2 +  (ky-ky_o)**2 )  / w**2 ) 
    
    return(fib_mode)
    
    
#create modes to be convolved
kx_o,ky_o = 0, 0 #fiber offsets (set to zero for now)
if 0: #if wvl/D >= a/foc_len: #then tel_mode needs to be on the big mesh (k1)

    #== Telescope mode (diffraction pattern from pupil)
    tel_mode = build_tel_mode(D,d,k1xx,k1yy)#sp.jv(1,np.pi * D * np.sqrt(k1xx**2 + k1yy**2))/(np.pi * D * np.sqrt(k1xx**2 + k1yy**2) )  - sp.jv(1,np.pi * d * np.sqrt(k1xx**2 + k1yy**2))/(np.pi * d * np.sqrt(k1xx**2 + k1yy**2) )
    
    #== Fiber mode 
    #fiber offsets   
    fib_mode = build_fib_mode(wvl,n_core,NA,a,foc_len,k2xx,k2yy,kx_o,ky_o) #cH/w * np.exp( -(wvl*((k2xx-kx_o)**2 +  (k2yy-ky_o)**2) ) / w**2 ) 
    
if 1: #elif wvl/D < a/foc_len: #then fib_mode needs to be on the big mesh (k1)

    #== Telescope mode (diffraction pattern from pupil)
    tel_mode = build_tel_mode(D,d,k2xx,k2yy)
    
    #== Fiber mode 
    #fiber offsets   
    fib_mode = build_fib_mode(wvl,n_core,NA,a,foc_len,k1xx,k1yy,kx_o,ky_o)
    

##### OUR FILTER ######    
M_G = scipy.signal.fftconvolve(fib_mode, tel_mode, mode='same')  
#normalze such that int |M_G|^2 dA = 1
M_G *= 1/np.sqrt( np.trapz(abs(M_G)**2, kx1) )
######### MAKE SURE FILTER CUT-OFF DOESN'T ALWAYS OCCUR WHERE WE CHANGE FROM LIN-LOG SPACING...
#plt.loglog(kx[:-1][kx[:-1]>0], M_G[:,len(M_G)//2][kx[:-1]>0])

# atmospheric phase PSD - Von-Karman model (Conan 2000)
VK_model = 0.0229 * r_0**(-5/3.) * (L_0**(-2) + (k1xx**2 + k1yy**2) )**(-11/6) 
# fix bad pixel near 0 freq

if np.sum(~np.isfinite(VK_model)):
    bad_i = np.where(~np.isfinite(VK_model))  #bad index
    VK_model[bad_i] = VK_model[bad_i[0][0]+1, bad_i[1][0]+1] #just nearest neighbour interp 
    
        

# now integrate filtered VK_model over ky
freq_indx = np.where(kx1>0)[0]
apo_piston = np.array( [ np.trapz( abs(M_G[i,:])**2 * VK_model[i,:], kx1 ) for i in freq_indx ] )

# compare it to non-apodized atmospheric piston 
atm_piston , _ = ap.atm_piston(wvl= wvl, seeing=seeing ,tau_0 = tau_0, L_0 = np.inf, diam = D)

plt.figure()
plt.loglog(v_mn/kx1[freq_indx], apo_piston);plt.loglog(*atm_piston)
    
#%%

"""

    understand units of Roddiers equation/gravity parameters comparing diameter to wvl/D
        - in Roddier a is in meters 
        - in Conan k is spatial frequency 1/m => piDk product is unitless => angle on sky.. 
        - can represent fiber function also as angle on sky (focal plane spatial coordinate a=focal lenght * alpha_w)
        - therefore create grid as function of angular cooordinates alpha.
            - remember under frozen flow tempooral freq f = k * V = (alpha/D) * V (remember DK = alpha)
            
"""

rel_size_factoor = 42 # rel_size_factoor * samples across waist = no samples in each array
xmax = rel_size_factoor * np.max([wvl/D, a/foc_len]) #extend grids to sample at least rel_size_factoor greater then largest waist
xmin = -xmax

#number of samples along the linspaced part of grid:
N = int( 12 * xmax/np.min([wvl/D, a/foc_len]) )  #for each have at least 12 samples across lambda/D or fiber waist 

# coordinates before convolution (remember these are anular coordinates (Dk = radians))

# LONGER ARRAY (in linspaced grid)
k1_lin = np.linspace(xmin, xmax, N) 
# logspacing outside of this (so we can get a reasonable frequency range)
k1_log = np.logspace(np.log10(xmax), 5, 2**10) #play here to get reasonable freq range 
# now append them 
kx1 = np.array(list(-k1_log[::-1]) + list(k1_lin) + list(k1_log))
#if np.mod(len(kx1),2) != 0:
#    kx1 = kx1[1:]
ky1 = kx1 
# SHORTER ARRAY
kx2, ky2 = kx1[np.where(abs(kx1)<rel_size_factoor * np.min([wvl/D, a/foc_len ]))], kx1[np.where(abs(kx1)<rel_size_factoor * np.min([wvl/D, a/foc_len]))]

# coordinates after valid convolution 
kx, ky =  kx1[len(kx2)//2  : len(kx1)-len(kx2)//2+1] , kx1[len(kx2)//2: len(kx1)-len(kx2)//2+1 ]

#frequencies for our final psd (definied by windspeed)
fs = kx/D * v_mn  


#MESH THE GRIDS 
if len(kx1)< 1e5: # check the biggest grid isn't tooo big before meshing
    
    k1xx, k1yy = np.meshgrid(kx1,ky1)
    k2xx, k2yy = np.meshgrid(kx2,ky2)
    kxx, kyy = np.meshgrid(kx,ky)
    
else:
    print('kx1 grid very big.. consider resizing')


#note these are both in units of radians (on sky) 
# alpha = D*k => d*k = alpha*d/D (where we have bad labels alpha = kxx etc)
def build_tel_mode(D,d,kx,ky):
    
    tel_mode = ( np.pi * D**2/( wvl * f ) ) * 2 * (sp.jv(1,np.pi * D * np.sqrt(kx**2 + ky**2)/ ( wvl * f ) )/(np.pi * D * np.sqrt(kx**2 + ky**2) ) \
        - (d/D)**2 * sp.jv(1,np.pi * d * np.sqrt(kx**2 + ky**2))/(np.pi * d * np.sqrt(kx**2 + ky**2) ) )
    
    return(tel_mode)


def build_fib_mode(wvl,n_core,NA,a,foc_len, kx,ky,kx_o,ky_o):
    
    
    #Vacuum permittivity and  vacuum permeability
    epsilon_o , mu_o = 8.854188e-12, 1.256637e-6 #
    #cladding index refraction
    n_clad = np.sqrt(n_core**2 - NA**2)    
    #V number (relates to number of modes fiber can hold)
    V = 2*np.pi*NA*a/wvl
    #simplifying coefficient
    cH = np.sqrt(2*n_clad)*(epsilon_o/mu_o)**(0.25)/np.sqrt(np.pi)    
    
    #beam waist (1/e)
    w = a/foc_len * (0.65 + 1.619/V**(3/2) + 2.879/V**6) # here waist units are in radians (on sky)

    #WHY DO I HAVE WVL HERE
    fib_mode = cH/(w/foc_len) * np.exp( -( (kx-kx_o)**2 +  (ky-ky_o)**2 )  / w**2 ) 
    
    return(fib_mode)


#create modes to be convolved
kx_o,ky_o = 0, 0 #fiber offsets (set to zero for now)
if 0: #if wvl/D >= a/foc_len: #then tel_mode needs to be on the big mesh (k1)

    #== Telescope mode (diffraction pattern from pupil)
    tel_mode = build_tel_mode(D,d,k1xx,k1yy)#sp.jv(1,np.pi * D * np.sqrt(k1xx**2 + k1yy**2))/(np.pi * D * np.sqrt(k1xx**2 + k1yy**2) )  - sp.jv(1,np.pi * d * np.sqrt(k1xx**2 + k1yy**2))/(np.pi * d * np.sqrt(k1xx**2 + k1yy**2) )
    
    #== Fiber mode 
    #fiber offsets   
    fib_mode = build_fib_mode(wvl,n_core,NA,a,foc_len,k2xx,k2yy,kx_o,ky_o) #cH/w * np.exp( -(wvl*((k2xx-kx_o)**2 +  (k2yy-ky_o)**2) ) / w**2 ) 
    
if 1: #elif wvl/D < a/foc_len: #then fib_mode needs to be on the big mesh (k1)

    #== Telescope mode (diffraction pattern from pupil)
    tel_mode = build_tel_mode(D,d,k2xx,k2yy)
    
    #== Fiber mode 
    #fiber offsets   
    fib_mode = build_fib_mode(wvl,n_core,NA,a,foc_len,k1xx,k1yy,kx_o,ky_o)
    
"""
#make sure to fix non-finite values 
np.isfinite(tel_mode).sum() == tel_mode.shape[0]**2
np.isfinite(fib_mode).sum() == fib_mode.shape[0]**2
then fix it:
"""
if np.sum(~np.isfinite(tel_mode)) : # == tel_mode.shape[0]**2:
    bad_i = np.where(~np.isfinite(tel_mode))  #bad index
    tel_mode[bad_i] = tel_mode[bad_i[0][0]+1, bad_i[1][0]+1] #just nearest neighbour interp 
""" """

##### OUR FILTER ######    
M_G = scipy.signal.fftconvolve(fib_mode, tel_mode, mode='valid')  
#normalze such that int |M_G|^2 dA = 1
M_G *= 1/np.sqrt( np.trapz(abs(M_G)**2, kx) )

######### MAKE SURE FILTER CUT-OFF DOESN'T ALWAYS OCCUR WHERE WE CHANGE FROM LIN-LOG SPACING...
#plt.loglog(kx[:-1][kx[:-1]>0], M_G[:,len(M_G)//2][kx[:-1]>0])

# atmospheric phase PSD - Von-Karman model (Conan 2000)
VK_model = 0.0229 * r_0**(-5/3.) * (L_0**(-2) + (kxx**2 + kyy**2) )**(-11/6) 
# fix bad pixel near 0 freq

if np.sum(~np.isfinite(VK_model)):
    bad_i = np.where(~np.isfinite(VK_model))  #bad index
    VK_model[bad_i] = VK_model[bad_i[0][0]+1, bad_i[1][0]+1] #just nearest neighbour interp 
    
        

# now integrate filtered VK_model over ky
freq_indx = np.where(kx>0)[0]
apo_piston = np.array( [ np.trapz( abs(M_G[i,:])**2 * VK_model[i,:], ky ) for i in freq_indx ] )

# compare it to non-apodized atmospheric piston 
atm_piston , _ = ap.atm_piston(wvl= wvl, seeing=seeing ,tau_0 = tau_0, L_0 = np.inf, diam = D)

plt.figure()
plt.loglog(fs[freq_indx], apo_piston);plt.loglog(*atm_piston)











#%%
for a in np.logspace(-6,-2,10):
    
    #permitivity free space
    epsilon_o , mu_o = 8.854188e-12, 1.256637e-6
    #cladding index refraction
    n_clad = np.sqrt(n_core**2 - NA**2)    
    #V number (relates to number of modes fiber can hold)
    V = 2*np.pi*NA*a/wvl
    #simplifying coefficient
    cH = np.sqrt(2*n_clad)*(epsilon_o/mu_o)**(0.25)/np.sqrt(np.pi)    

    
    #beam waist (1/e)
    w = a * (0.65 + 1.619/V**(3/2) + 2.879/V**6) # radians on sky? 

    field_pupil = cH/np.sqrt(np.pi) * np.exp(-w**2 * (x_p**2 + y_p**2) )
    
    field_focal = cH/w * np.exp(-(np.sqrt(x_f**2 + y_f**2)/w)**2)
    
    print(1/w, np.nanstd(pupil_mask * field_pupil ))
    
    #fs, pist1 = atm_w_fiber_piston(wvl=2.2e-6, NA=NA, n_core=n_core , a=a, seeing=seeing, \
    #                   tau_0 = tau_0, L_0 = np.inf, diam=8,  diam_m2 = None, N_grid = 2**11)




    # ---------[[[]]]----------------------
    N_grid = 2**13
    # our spatial frequencies for convolution 
    kx, ky = np.concatenate([-np.logspace(-4,3,2**12)[::-1], np.logspace(-4,3,2**12) ] ) , np.concatenate([-np.logspace(-4,3,2**12)[::-1], np.logspace(-4,3,2**12) ] )
    #np.linspace(-5e2,5e2,N_grid), np.linspace(-5e2,5e2,N_grid) #np.concatenate([-np.logspace(-4,3,2**12)[::-1], np.logspace(-4,3,2**12) ] ) , np.concatenate([-np.logspace(-10,3,2**12)[::-1], np.logspace(-10,3,2**12) ] )  #np.linspace(-5e2,5e2,2**13), np.linspace(-5e2,5e2,2**13)
    # mesh it
    kxx, kyy = np.meshgrid( kx, ky )

    # fourier transform of fiber mode in pupil plane (fiber mode in focal plane under Roddier approximation )
    kx_o, ky_o = 0, 0 
    fiber_mode = cH/w * np.exp( -(wvl*((kxx-kx_o)**2 +  (kyy-ky_o)**2) ) / w**2 ) 
    # normalize as in Woillez ( int int |field|^2 dA = 1 )
    fiber_mode *= 1/ np.sqrt( np.trapz( [ np.trapz(fiber_mode[:,i]**2,kx) for i in range(fiber_mode.shape[1]) ], ky) ) 
    
    # fourier transform of telescope pupil 
    diam_m2 = None
    if diam_m2:
        tel_pupil_fft = sp.jv(1,np.pi*diam*np.sqrt(kxx**2 + kyy**2))/(np.pi*diam*np.sqrt(kxx**2 + kyy**2) ) - sp.jv(1,np.pi*diam_m2*np.sqrt(kxx**2 + kyy**2))/(np.pi*diam_m2*np.sqrt(kxx**2 + kyy**2) )
    else:
        tel_pupil_fft = sp.jv(1,np.pi*diam*np.sqrt(kxx**2 + kyy**2))/(np.pi*diam*np.sqrt(kxx**2 + kyy**2) )
    #F[AB] = F[A] * F[B] (where * is convolution and F[A] is Fourier transform)
    # therefore we create our filter
    M_G = scipy.signal.fftconvolve(fiber_mode, tel_pupil_fft, mode='same') / np.sqrt( len(tel_pupil_fft) ) 
    # note: sqrt(len(x)) is because we take |M_G|^2 in the integral
    
    # ===========
    # now we define our grid that we are going to integrate over to get the PSD as a subgrid of our convolution grid
    # kx = fs / v_mn => fs = kx * v_mn !!!
    
    # make our integration grid 
    kx2 = kx[kx > 0] # we only consider positive frequecies for PSD (kx = fs/v_mn)
    fs = kx2 * v_mn  #frequencies for our final psd 
    kxx2, kyy2 = np.meshgrid(kx2,ky)
    M_G2 = M_G[:,len(M_G)//2:] # we extract the integration subgrid from M_G
    # Von-Karman model (Conan 2000)
    VK_model = 0.0229 * r_0**(-5/3.) * (L_0**(-2) + (kxx2**2 + kyy2**2) )**(-11/6) 
    
    # now integrate filtered VK_model over ky
    pist1 = np.array( [ np.trapz( abs(M_G2[:,i])**2 * VK_model[:,i], ky ) for i in range(VK_model.shape[1])] )
    





#%% Section 5 - derivation of theoretical acceleration PSD for VLTI (comparisoon to accelerometersw noise floor)



#%% Section 6 - comparison of theoretical VLTI UT/Gravity apodized piston to measured psuedo open loop on Gravity Fringe Tracker 






















#%% SOME USEFUL STUFF BELOW



#%%

x1 = np.concatenate([-np.logspace(-10,4,int(1e3)), np.logspace(-10,4,int(1e3))])
x1 = np.linspace(-1e3,1e3, int(1e4))
#x2 = x1[(x1>1e-7) & (x1<1e4)]
x2 = x1[x1 > 0 ]
x3 = x1[abs(x1)<500]

xn = x1[len(x2)//2 : (len(x1) - len(x2)//2 )]

sigma = 0.5
aaa = 0.1
plt.semilogx(x2,np.exp(-(x2/sigma)**2)) 
plt.semilogx(x2,sp.jv(1,aaa*x2)/(aaa * x2) ) 

plt.figure()
plt.loglog(x1, np.convolve( sp.jv(1,aaa*x1)/(aaa * x1)  , np.exp(-(x1/sigma)**2),  'same') )
#plt.semilogy(np.convolve( sp.jv(1,aaa*x1)/(aaa * x2)  , np.exp(-(x1/sigma)**2), 'full') )

plt.figure()
plt.loglog(np.convolve( sp.jv(1,aaa*x1)/(aaa * x1)  , np.exp(-(x3/sigma)**2),  'full') )

plt.figure()
plt.loglog(np.convolve( sp.jv(1,aaa*x1)/(aaa * x1)  , np.exp(-(x2/sigma)**2),  'full') )
plt.loglog(np.convolve( sp.jv(1,aaa*x2)/(aaa * x2)  , np.exp(-(x1/sigma)**2), 'full') )


#%%  ------- # WHAT DOES CONVOLUTION DO TO POWER LAWS 

a, b = 1, 2
x1 = np.logspace(-10,4,int(1e4))
x2 = x1[(x1>1e-4) & (x1<1e3)]

xn = x1[len(x2)//2 : (len(x1) - len(x2)//2 )]

plt.loglog(xn, np.convolve(x1**b, x2**a,'valid'),label=x1)
plt.loglog(xn,xn**b,label=r'$f^{b}$')

# now swap inidices 
plt.loglog(xn, np.convolve(x1**a, x2**b,'valid'))
plt.loglog(xn,xn**a,label=r'$f^{a}$')

plt.legend()

# checking commutivity 
plt.figure()
plt.semilogy(np.convolve(x1**a, x2**b,'full'),label='x1**a * x2**b',color='b'); 
plt.semilogy(np.convolve(x1**b, x2**a,'full'),label='x1**b * x2**a',color='g');
plt.axvline(len(x1),label='len(x1)',color='b');
plt.axvline(len(x2),label='len(x2)',color='g')
plt.legend()

plt.figure()
plt.semilogy(np.convolve(x1**a, x2**b,'full'),label='x1**a * x2**b',color='b'); 
plt.semilogy(np.convolve(x2**a, x1**b,'full'),label='x2**a * x1**b',color='g');
plt.axvline(len(x1),label='len(x1)',color='g');
plt.axvline(len(x2),label='len(x2)',color='b')
plt.legend()

#plt.loglog(x,x**b,label=r'$f^{b}$')
#plt.loglog(x,x[::-1]**a * x**b,label=r'$f^{ab}$')
#plt.loglog(x, x**(-a+b + 1),label='f^(a+b)')


#plt.loglog(x,1e20* x**-1)
#plt.loglog(x,1e18* x**-5)




#%%

wvl = 2.2e-6
a = 3e-6 #3e-6 # 1e-6 #fiber core diameter
NA = 0.12 #.45 #numerical apperature 
n_core = 1 #refractive index of core
N = 2**10



#-----fiber parameters
#permitivity free space
epsilon_o , mu_o = 8.854188e-12, 1.256637e-6
#cladding index refraction
n_clad = np.sqrt(n_core**2 - NA**2)    
#V number (relates to number of modes fiber can hold)
V = 2*np.pi*NA*a/wvl
#simplifying coefficient
cH = np.sqrt(2*n_clad)*(epsilon_o/mu_o)**(0.25)/np.sqrt(np.pi)    
#beam waist (1/e)
w = a * (0.65 + 1.619/V**(3/2) + 2.879/V**6) # radians on sky? 

#want to sample at least 3x the field waist in focal plane :
    
f_array = np.linspace(-8*w,8*w,N) 
# relate pupil plane coordinates to focal through wvl
p_array = f_array / wvl

#create coordinate grids 
x_f,y_f = np.meshgrid(f_array, f_array)
x_p,y_p = np.meshgrid(p_array, p_array)

#differential elements 
dx_f = np.diff(x_f)[0][0]
dx_p = np.diff(x_p)[0][0]


field_pupil = cH/np.sqrt(np.pi) * np.exp(-w**2 * (x_p**2 + y_p**2) )

field_focal = cH/w * np.exp(-(np.sqrt(x_f**2 + y_f**2)/w)**2)


pupil_m1 = aotools.functions.pupil.circle(radius=N//2, size=N, circle_centre=(0, 0), origin='middle')
#m2 diameter = 1 m
pupil_m2 =  aotools.functions.pupil.circle(radius=N//16, size=N, circle_centre=(0, 0), origin='middle')

#consider with and without central obscuration
pupil = pupil_m1
pupil_obs = pupil_m1 - pupil_m2 

#create masks for calculating statistics in the pupil (0->nan)
pupil_mask = pupil.copy() #need to mask outside pupil by nan for calculating averages 
pupil_mask[pupil==0] = np.nan

pupil_obs_mask = pupil_obs.copy()
pupil_obs_mask[pupil_obs_mask==0] = np.nan



# get our fiber appodized piston 
plt.figure(figsize=(8,5))
for a in np.logspace(-8,-5,4):
    pist1 = atm_w_fiber_piston(wvl=2.2e-6, NA=0.12, n_core=1, a=a, seeing=0.86, \
                       tau_0 = 4e-3, L_0 = np.inf, diam=8,  diam_m2 = None, N_grid = 2**12)
        
    plt.loglog(*pist1,label='{:.2e}'.format(a))
    plt.xlabel('frequency',fontsize=15)
    plt.ylabel('PSD',fontsize=15)
    plt.legend()
    
    
    
    

"""
fig8 = plt.figure(constrained_layout=False, figsize=(15,5))


gs1 = fig8.add_gridspec(nrows=1, ncols=3, left=0.05, right=0.48, wspace=0.05)
f8_ax1 = fig8.add_subplot(gs1[:, :2])
f8_ax2 = fig8.add_subplot(gs1[:, -1])

f8_ax1.loglog(*pist1 )
f8_ax2.pcolormesh( pupil_mask * field_pupil ) 
f8_ax2.set_aspect('equal')

"""




#%% Visualization 


#========== fiber parameters 
#-----Atmospheric Parameters
wvl = 2.2e-6
seeing = 0.86
r_0 = 0.98*wvl/np.radians(seeing/3600)  #fried parameter at wavelength in m.
tau_0 = (2.2/0.5)**(6/5) * 4e-3        #atm coherence time at wavelength
v_mn = 0.314 * r_0 / tau_0               #AO definition for vbar   
L_0 = np.inf
diam = 8 #m
#atm_params = {'wvl':wvl,'seeing':seeing,'r_0':r_0,'tau_0':tau_0,'L_0':L_0,'diam':diam}
#-----configuration parameters


#  we will loop over a = 3e-6 #3e-6 # 1e-6 #fiber core diameter
NA = 0.12 #.45 #numerical apperature 
n_core = 1 #refractive index of core
N = 2**10



#========== pupil visulization 
f_array = np.linspace(-1e-4,1e-4,N) 
# relate pupil plane coordinates to focal through wvl
p_array = f_array / wvl

#create coordinate grids 
x_f,y_f = np.meshgrid(f_array, f_array)
x_p,y_p = np.meshgrid(p_array,p_array)

#differential elements 
dx_f = np.diff(x_f)[0][0]
dx_p = np.diff(x_p)[0][0]

#search where pupil coord = 4 (radius for VLT) then divide by 2 since x_p is symmetric around 0
pupil_m1 = aotools.functions.pupil.circle(radius=np.argmin(abs(x_p-4))//2, size=N, circle_centre=(0, 0), origin='middle')
#m2 diameter = 1 m
pupil_m2 =  aotools.functions.pupil.circle(radius=np.argmin(abs(x_p-1))//2, size=N, circle_centre=(0, 0), origin='middle')

#consider with and without central obscuration
pupil = pupil_m1
pupil_obs = pupil_m1 - pupil_m2 

#create masks for calculating statistics in the pupil (0->nan)
pupil_mask = pupil.copy() #need to mask outside pupil by nan for calculating averages 
pupil_mask[pupil==0] = np.nan

pupil_obs_mask = pupil_obs.copy()
pupil_obs_mask[pupil_obs_mask==0] = np.nan


# compare it to atmospheric piston 
atm_pis , atm_dict = ap.atm_piston(wvl=2.2e-6, seeing=0.86,tau_0 = 4e-3, L_0 = np.inf, diam=8)


for a in np.logspace(-6,-2,10):
    
    #permitivity free space
    epsilon_o , mu_o = 8.854188e-12, 1.256637e-6
    #cladding index refraction
    n_clad = np.sqrt(n_core**2 - NA**2)    
    #V number (relates to number of modes fiber can hold)
    V = 2*np.pi*NA*a/wvl
    #simplifying coefficient
    cH = np.sqrt(2*n_clad)*(epsilon_o/mu_o)**(0.25)/np.sqrt(np.pi)    

    
    #beam waist (1/e)
    w = a * (0.65 + 1.619/V**(3/2) + 2.879/V**6) # radians on sky? 

    field_pupil = cH/np.sqrt(np.pi) * np.exp(-w**2 * (x_p**2 + y_p**2) )
    
    field_focal = cH/w * np.exp(-(np.sqrt(x_f**2 + y_f**2)/w)**2)
    
    print(1/w, np.nanstd(pupil_mask * field_pupil ))
    
    #fs, pist1 = atm_w_fiber_piston(wvl=2.2e-6, NA=NA, n_core=n_core , a=a, seeing=seeing, \
    #                   tau_0 = tau_0, L_0 = np.inf, diam=8,  diam_m2 = None, N_grid = 2**11)




    # ---------[[[]]]----------------------
    N_grid = 2**13
    # our spatial frequencies for convolution 
    kx, ky = np.concatenate([-np.logspace(-4,3,2**12)[::-1], np.logspace(-4,3,2**12) ] ) , np.concatenate([-np.logspace(-4,3,2**12)[::-1], np.logspace(-4,3,2**12) ] )
    #np.linspace(-5e2,5e2,N_grid), np.linspace(-5e2,5e2,N_grid) #np.concatenate([-np.logspace(-4,3,2**12)[::-1], np.logspace(-4,3,2**12) ] ) , np.concatenate([-np.logspace(-10,3,2**12)[::-1], np.logspace(-10,3,2**12) ] )  #np.linspace(-5e2,5e2,2**13), np.linspace(-5e2,5e2,2**13)
    # mesh it
    kxx, kyy = np.meshgrid( kx, ky )

    # fourier transform of fiber mode in pupil plane (fiber mode in focal plane under Roddier approximation )
    kx_o, ky_o = 0, 0 
    fiber_mode = cH/w * np.exp( -(wvl*((kxx-kx_o)**2 +  (kyy-ky_o)**2) ) / w**2 ) 
    # normalize as in Woillez ( int int |field|^2 dA = 1 )
    fiber_mode *= 1/ np.sqrt( np.trapz( [ np.trapz(fiber_mode[:,i]**2,kx) for i in range(fiber_mode.shape[1]) ], ky) ) 
    
    # fourier transform of telescope pupil 
    diam_m2 = None
    if diam_m2:
        tel_pupil_fft = sp.jv(1,np.pi*diam*np.sqrt(kxx**2 + kyy**2))/(np.pi*diam*np.sqrt(kxx**2 + kyy**2) ) - sp.jv(1,np.pi*diam_m2*np.sqrt(kxx**2 + kyy**2))/(np.pi*diam_m2*np.sqrt(kxx**2 + kyy**2) )
    else:
        tel_pupil_fft = sp.jv(1,np.pi*diam*np.sqrt(kxx**2 + kyy**2))/(np.pi*diam*np.sqrt(kxx**2 + kyy**2) )
    #F[AB] = F[A] * F[B] (where * is convolution and F[A] is Fourier transform)
    # therefore we create our filter
    M_G = scipy.signal.fftconvolve(fiber_mode, tel_pupil_fft, mode='same') / np.sqrt( len(tel_pupil_fft) ) 
    # note: sqrt(len(x)) is because we take |M_G|^2 in the integral
    
    # ===========
    # now we define our grid that we are going to integrate over to get the PSD as a subgrid of our convolution grid
    # kx = fs / v_mn => fs = kx * v_mn !!!
    
    # make our integration grid 
    kx2 = kx[kx > 0] # we only consider positive frequecies for PSD (kx = fs/v_mn)
    fs = kx2 * v_mn  #frequencies for our final psd 
    kxx2, kyy2 = np.meshgrid(kx2,ky)
    M_G2 = M_G[:,len(M_G)//2:] # we extract the integration subgrid from M_G
    # Von-Karman model (Conan 2000)
    VK_model = 0.0229 * r_0**(-5/3.) * (L_0**(-2) + (kxx2**2 + kyy2**2) )**(-11/6) 
    
    # now integrate filtered VK_model over ky
    pist1 = np.array( [ np.trapz( abs(M_G2[:,i])**2 * VK_model[:,i], ky ) for i in range(VK_model.shape[1])] )
    
    # -----------[[[]]]--------------------


    fig8 = plt.figure(constrained_layout=False, figsize=(15,5))
    
    
    gs1 = fig8.add_gridspec(nrows=1, ncols=3, left=0.05, right=0.48, wspace=0.05)
    f8_ax1 = fig8.add_subplot(gs1[:, :2])
    f8_ax2 = fig8.add_subplot(gs1[:, -1])
    
    f8_ax1.loglog(fs, pist1 , label = 'apodized (a={:.2e})'.format(a))
    f8_ax1.loglog(*atm_pis , label = 'non-apodized')
    f8_ax1.legend(fontsize=14)
    
    
    f8_ax2.pcolormesh( p_array , p_array , pupil_mask * field_pupil , vmin=0, vmax = 2e-2) 
    f8_ax2.set_aspect('equal')
    f8_ax2.axis('off')
    
    plt.show()
    
    # looking at the filters 
    fff = 20
    fig,ax = plt.subplots(6,1,sharex=True,figsize=(10,20)) 
    
    ax[0].semilogy(kx* v_mn, fiber_mode[N_grid//2,:], label='fiber mode (a)')
    ax[0].legend(fontsize=fff,loc='lower right')
        
    ax[1].semilogy(kx* v_mn, tel_pupil_fft[N_grid//2,:],label='tel diffraction (b)')
    ax[1].legend(fontsize=fff,loc='lower right')
       
    ax[2].semilogy(kx* v_mn, abs(M_G[N_grid//2,:])**2, label = r'$|M_G|^2=|a * b|^2$')
    ax[2].legend(fontsize=fff,loc='lower right')
    #ax[2].set_ylim([1e-50,1e-2])
    
    ax[3].semilogy(kx* v_mn, VK_model[:,-1],label=r'$W_\phi$' + '(Von Karman)')
    ax[3].legend(fontsize=fff,loc='lower right')
    
    ax[4].semilogy(kx * v_mn, abs(M_G[N_grid//2,:])**2 * VK_model[:,-1], label=r'$|M_G|^2 W_\phi$')
    ax[4].legend(fontsize=fff,loc='lower right')
    
    ax[5].plot(kx * v_mn, np.cumsum(abs(M_G[N_grid//2,:])**2 * VK_model[:,-1]), label=r'$\int_{-\infty}^{k_y} |M_G|^2 W_\phi$')
    ax[5].legend(fontsize=fff,loc='lower right')
    #ax[5].set_xlim([-500, 500])
    
    
    fig, ax = plt.subplots(2,1,sharex=True , figsize = (10,5))
    ax[0].loglog(fs, pist1,label = 'apodized')
    ax[0].loglog(*atm_pis , label = 'non-apodized')
    ax[1].pcolormesh(fs, ky, np.log10(abs(M_G2)**2 * VK_model) )

    
#%% ORIGINAL DO NOT TOUCH 


## don't normalize analytically because requires proper sampling , do it by making sure field in current grid is normalized! 

"""
# understanding convolution normalization 
delta = np.zeros(field_focal.shape)
delta[len(delta)//2, len(delta)//2] = 2
conv = scipy.signal.fftconvolve(delta, field_focal, 'same')
print( np.sum(field_focal), np.sum(conv) )"""
#-----Atmospheric Parameters

r_0 = 0.98*wvl/np.radians(0.86/3600)  #fried parameter at wavelength in m.
tau_0 = (2.2/0.5)**(6/5) * 4e-3        #atm coherence time at wavelength
v_mn = 0.314 * r_0 / tau_0               #AO definition for vbar   
L_0 = np.inf
diam = 8 #m
#atm_params = {'wvl':wvl,'seeing':seeing,'r_0':r_0,'tau_0':tau_0,'L_0':L_0,'diam':diam}
#-----configuration parameters

#ky needs to have good sampling at small values where VK_model * tel_filter is maximum for the integration (course sampling outside of this since contribution is negligble)
#ky = np.concatenate([-np.logspace(-10,3,2**11)[::-1], np.logspace(-10,3,2**11) ] )  # Spatial frequency 

wvl = 2.2e-6
a = 3e-6 #3e-6 # 1e-6 #fiber core diameter
NA = 0.12 #.45 #numerical apperature 
n_core = 1 #refractive index of core
N = 2**10


# ===========
# now to create filter 
# NOTE the convolution has to be integral over infinity wiith both fields centered!!
#   therefore we need two grids, one for the convolutioon, and the other for integration of the filter with phase PSD
#    we begin with creating the convolution grid 

# our spatial frequencies for convolution 
kx, ky = np.linspace(-1e2,1e2,2**12), np.linspace(-1e2,1e2,2**12) #np.concatenate([-np.logspace(-4,3,2**12)[::-1], np.logspace(-4,3,2**12) ] ) , np.concatenate([-np.logspace(-10,3,2**12)[::-1], np.logspace(-10,3,2**12) ] )  #np.linspace(-5e2,5e2,2**13), np.linspace(-5e2,5e2,2**13)

kxx, kyy = np.meshgrid( kx, ky )

# fourier transform of fiber mode in pupil plane (fiber mode in focal plane)
kx_o, ky_o = 0, 0 
fiber_mode = cH/w * np.exp( -(wvl*((kxx-kx_o)**2 +  (kyy-ky_o)**2) ) / w**2 ) 
# normalize as in Woillez ( int int |field|^2 dA )
fiber_mode *= 1/ np.sqrt( np.trapz( [ np.trapz(fiber_mode[:,i]**2,kx) for i in range(fiber_mode.shape[1]) ], ky) ) 


# plt.plot(tel_pupil_fft[len(tel_pupil_fft)//2,4000:4500]/npp.max(tel_pupil_fft))
# plt.plot(fiber_mode[len(fiber_mode)//2,4000:4500]/np.max(fiber_mode))

# fourier transform of telescope pupil 
diam_m2 = 1
tel_pupil_fft = sp.jv(1,np.pi*diam*np.sqrt(kxx**2 + kyy**2))/(np.pi*diam*np.sqrt(kxx**2 + kyy**2) ) #- sp.jv(1,np.pi*diam_m2*np.sqrt(kxx**2 + kyy**2))/(np.pi*diam_m2*np.sqrt(kxx**2 + kyy**2) )

#NOTE: we can add offsets to fiber mode our ppupil center in the above ( kx -> (kx-ko) ) !!!

# F[AB] = F[A] * F[B] (where * is convolution and F[A] is Fourier transform)
# therefore we create our filter
M_G = scipy.signal.fftconvolve(fiber_mode, tel_pupil_fft, mode='same') / np.sqrt( len(tel_pupil_fft) ) 
# note: can set this ^^ to  tel_pupil_fft and check (it does) reproduce standard atmospheric piston with ap.atm_piston()
#plt.imshow(M_G).. Also sqrt(len(x)) is because we take |M_G|^2 in the integral

# ===========
# now we define our grid that we are going to integrate over to get the PSD as a subgrid of our convolution grid
# kx = fs / v_mn => fs = kx * v_mn !!!

#fs = np.linspace(1e-3,5e2,2**10) #frequencies for our psd 
#kx = fs / v_mn # we divide by wind speed (frozen flow hypoth) to convert temporal freq to spatial
#ky = np.linspace(-5e2,5e2,2**13)  # oour y frequencies that we integrate over infinity (in appprox)

# make our integration grid 
kx2 = kx[kx > 0] # we only consider positive frequecies for PSD (kx = fs/v_mn)
fs = kx2 * v_mn  #frequencies for our final psd 
kxx2, kyy2 = np.meshgrid(kx2,ky)
M_G2 = M_G[:,len(M_G)//2:] # we extract the integration subgrid from M_G
# Von-Karman model (Conan 2000)
VK_model = 0.0229 * r_0**(-5/3.) * (L_0**(-2) + (kxx2**2 + kyy2**2) )**(-11/6) 

# now integrate filtered VK_model over ky
apod_piston = np.array( [ np.trapz( abs(M_G2[:,i])**2 * VK_model[:,i], ky ) for i in range(VK_model.shape[1])] )


# compare it to atmospheric piston 
atm_pis , atm_dict = ap.atm_piston()


plt.figure()
plt.loglog(fs,apod_piston ,label='apodized')
plt.loglog(*atm_pis,label='open')
plt.legend()
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD')
plt.axvline(0.3 * v_mn / diam,color='k')

