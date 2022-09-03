#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 08:53:03 2022

@author: bcourtne
"""


import pyzelda.zelda as zelda
import pyzelda.ztools as ztools
import pyzelda.utils.aperture as aperture
import pyzelda.utils.zernike as zernike
import pyzelda.utils.mft as mft

import argparse
from astropy.io import fits
import numpy as np
import scipy.signal as sig
from scipy.interpolate import interp1d 
from scipy.optimize import curve_fit
import pylab as plt
import pandas as pd
import os 
import glob
from astroquery.simbad import Simbad
from matplotlib import gridspec
import aotools

os.chdir('/Users/bcourtne/Documents/mac1_bk/Hi5/vibration_analysis/utilities')
import atmosphere_piston as atm


def opd_rms( f, opd_psd , bandwidth = [0,1e4]):
    # calculate the opd rms from OPD (or OPL) PSD , f is frequency and opd_psd is the psd
    mask = (f <= bandwidth [1]) & (f >= bandwidth [0])
    opd_rms = np.sqrt( np.sum( opd_psd[mask] * np.diff(f[mask])[1] ) )
    return(opd_rms) 


def overlap_integral(E1, E2): 
    #fiber overlap integral to calculate coupled efficienccy and phase. note injected energy = abs(eta)**2 !!!
    eta = ( np.nansum( E1 * np.conjugate(E2) ) ) / ( np.nansum(E1 * np.conjugate(E1) ) * np.nansum( E2 * np.conjugate(E2)) )**0.5
    
    return(eta)



def AT_pupil(dim, diameter, spiders_thickness=0.008, strict=False, cpix=False):
    '''Auxillary Telescope theoretical pupil with central obscuration and spiders
    
    function adapted from pyzelda..
    
    
    Parameters
    ----------
    dim : int
        Size of the output array
    
    diameter : int
        Diameter the disk
    spiders_thickness : float
        Thickness of the spiders, in fraction of the pupil
        diameter. Default is 0.008
    spiders_orientation : float
        Orientation of the spiders. The zero-orientation corresponds
        to the orientation of the spiders when observing in ELEV
        mode. Default is 0
    dead_actuators : array
        Position of dead actuators in the pupil, given in fraction of
        the pupil size. The default values are for SPHERE dead
        actuators but any other values can be provided as a Nx2 array.
    dead_actuator_diameter : float
        Size of the dead actuators mask, in fraction of the pupil
        diameter. This is the dead actuators of SPHERE. Default is
        0.025
    strict : bool optional
        If set to Trye, size must be strictly less than (<), instead of less
        or equal (<=). Default is 'False'
    
    cpix : bool optional
        If set to True, the disc is centered on pixel at position (dim//2, dim//2).
        Default is 'False', i.e. the disc is centered between 4 pixels
    
    Returns
    -------
    pup : array
        An array containing a disc with the specified parameters
    '''

    # central obscuration (in fraction of the pupil)
    obs  = 0.13/1.8
    spiders_orientation = 0

    pp1 = 2.5
    # spiders
    if spiders_thickness > 0:
        # adds some padding on the borders
        tdim = dim+50

        # dimensions
        cc = tdim // 2
        spdr = int(max(1, spiders_thickness*dim))
            
        ref = np.zeros((tdim, tdim))
        ref[cc:, cc:cc+spdr] = 1
        spider1 = aperture._rotate_interp(ref, -pp1 , (cc, cc+diameter/2))

        ref = np.zeros((tdim, tdim))
        ref[:cc, cc-spdr+1:cc+1] = 1
        spider2 = aperture._rotate_interp(ref, -pp1 , (cc, cc-diameter/2))
        
        ref = np.zeros((tdim, tdim))
        ref[cc:cc+spdr, cc:] = 1
        spider3 = aperture._rotate_interp(ref, pp1 , (cc+diameter/2, cc))
        
        ref = np.zeros((tdim, tdim))
        ref[cc-spdr+1:cc+1, :cc] = 1
        spider4 = aperture._rotate_interp(ref, pp1 , (cc-diameter/2, cc))

        spider0 = spider1 + spider2 + spider3 + spider4

        spider0 = aperture._rotate_interp(spider1+spider2+spider3+spider4, 45+spiders_orientation, (cc, cc))
        
        spider0 = 1 - spider0
        spider0 = spider0[25:-25, 25:-25]
    else:
        spider0 = np.ones(dim)

    # main pupil
    pup = aperture.disc_obstructed(dim, diameter, obs, diameter=True, strict=strict, cpix=cpix)

    # add spiders
    pup *= spider0

    return (pup >= 0.5).astype(int)

#def calc_omega(X,Y,W):
#    omega = np.nansum( X*np.conjugate(Y)*W ) / ( np.nansum( X*np.conjugate(X) * W ) * np.nansum( Y*np.conjugate(Y) * W ) )**0.5
#    return(omega)

"""
1. temporal PSD of Zernike modes plot their coefficients, compare to c_ij(D/r0)^2 result for first few.
2. generate phase screens with aotools 
"""


#%% !. temporal PSD of Zernike modes plot their coefficients, compare to c_ij(D/r0)^2 result for first few.
basis = zernike.zernike_basis(20)
#noll coefficients for first 3 radial orders 
c_ij = np.array( [0.449,0.449,0.0232,0.0232,0.0232, 0.00619, 0.00619, 0.00619,0.00619] )


D = 8
wvl = 2.2e-6

seeing = 0.86 
r0 = 0.98*wvl/np.radians(seeing/3600) 


aaa=[atm.atm_zernike(j, theta=0, wvl=wvl, seeing=seeing, L_0=np.inf, diam=8) for j in range(1,15)] 

coes = [np.sum(aaa[j][0][1])*np.diff(aaa[0][0][0])[0] for j in range(len(aaa))]

coes = [coes[i]/coes[1] for i in range(1,len(coes))]

plt.semilogy(coes ,label='calculated from temporal PSD')
plt.plot( c_ij * (D/r0)**(5/3) / ( c_ij[0] * (D/r0)**(5/3) ) ,label = 'theoretical coefficient ' ) 
plt.legend()
plt.ylabel(r'rad$^2$')
plt.xlabel('noll index')


#%% 2. generate phase screens with aotools, confirm units and confirm temporal PSD of rolling screens matches Kolmogorov


"""D=8 #m
wvl = 1.2e-6
nx_size = 2**8
pixel_scale = D/nx_size # m/pixel
r0 = 0.1 #m 
L0 = 24 #m

vlt_pup = aperture.vlt_pupil(dim = nx_size, diameter = nx_size, dead_actuator_diameter=0)

scrn = aotools.turbulence.infinitephasescreen.PhaseScreenVonKarman(nx_size, pixel_scale,\
      r0 , L0, n_columns=2,random_seed = 1)

#to work out units, the std over a mask of diamter = r0 should be 1rad rms..
r0_disk = aperture.disc(dim=nx_size, size = int(1*r0 / pixel_scale) )   # number of pixels across r0 is r0/pixel_scale 

r0_meas = []
for i in range(100):
    
    scrn.add_row()
    r0_meas.append( np.std(scrn.scrn[r0_disk!=0]) )
    
print( 'mean std across mask of diam = r0 (total of {} pixels),  is {} +- {} \ntheoretcially it should be 1 rad '.format(np.sum(r0_disk), np.mean(r0_meas), np.std(r0_meas)))

# I looks consistent that units are radians 
"""

tel_assoc_param_dict = {'ut':(8.2, 20,2**8 ),'at':(1.8, 4.5, 2**7)}

D, foc_len, nx_size = tel_assoc_param_dict['at']

n_terms = 10 ## of zernike basis terms to consider 
#D= 1.8 #8 #m
wvl = 1.65e-6
#nx_size = 2**7 #2**9
pixel_scale = D/nx_size # m/pixel
r0 = 0.1 #m  @ 500nm
L0 = 24 #m
V_phase = 50 # 20 # velocity of phase screen (m/s)
dt = pixel_scale / V_phase # how many seconds pass between rolling phase screen by 1 row (1 iteration)  

basis = zernike.zernike_basis(n_terms, npix=nx_size)


print(f'sim dt = {dt*1e3}ms')


phase_screen = aotools.turbulence.infinitephasescreen.PhaseScreenVonKarman(nx_size, pixel_scale,\
      r0 , L0, n_columns=2,random_seed = 1)
    
z_coes = [] # to hold array of zernike coefficients 
iterations = int(4e3) 

print(f'max freq = {V_phase/pixel_scale}, min freq = {V_phase/( iterations * pixel_scale )}, knee transition = {0.2 * V_phase/D} ')
for i in range(iterations):
    if np.mod(i,100)==0:
        print(f'  {round(i/iterations,2)}% complete')
    z_coes.append( zernike.opd_expand(wvl/(2*np.pi) * basis[0] * phase_screen.scrn, nterms=n_terms, basis=zernike.zernike_basis) ) 
    # next phase screen
    phase_screen.add_row()

cf = 3e-1 #correction factor 
psd = sig.welch( cf * 2*np.pi / wvl * np.array(z_coes), fs=1/dt, window='hann',nperseg=2**11,axis=0)


fig, ax = plt.subplots(1,3,figsize=(15,5),sharey=True)
ax[0].set_ylabel(r'PSD [rad$^2$/Hz]',fontsize=15)

for i,axx in zip([2,4,8], ax):
    
    z_psd = atm.atm_zernike(i+1, theta=0, wvl=wvl, seeing=3600 * 180/np.pi *(0.98*500e-9/r0),  v_mn=V_phase, L_0=np.inf, diam=D) 
    
    f_min = np.max( [ z_psd[0][0][0], psd[0][1] ] )
    f_max = np.min( [ z_psd[0][0][-1], psd[0][-1] ] )
    
    axx.set_title(f'{zernike.zern_name(1+i)}',fontsize=14)
    axx.loglog( *z_psd[0] , color='k',linestyle='-', label='theory (Kolmogorov)')
    axx.loglog(psd[0], psd[1][:,i+1],color='k',linestyle='--', label='simulation (AOtools)')
    axx.set_xlabel('frequency [Hz]',fontsize=15)
    axx.legend(fontsize=14)
    axx.set_xlim([f_min,f_max])
    axx.tick_params(labelsize=15)
    axx.grid()
    
    opd_theory  = opd_rms( np.array(z_psd[0][0]) , np.array(z_psd[0][1]), bandwidth = [f_min,f_max])
    opd_sim = opd_rms( psd[0],psd[1][:,i+1], bandwidth = [f_min,f_max] )
    
    aaa.append(opd_theory /opd_sim )
    print(opd_theory /opd_sim )
plt.tight_layout()
#plt.savefig('/Users/bcourtne/Documents/ANU_PHD2/heimdallr/temporal_psd_comparison_aotools_v_kolmogorov.png') 

"""
plt.figure()
plt.title(f'Noll index={zernike.zern_name(1+i)}')
plt.loglog( *z_psd[0] , label='Kolmogorov (Conan 1995)')
plt.loglog(psd[0], psd[1][:,i+1], label='AO tools result')
plt.ylabel(r'PSD [rad$^2$/Hz]')
plt.xlabel('frequency [Hz]')
plt.legend()

f_min = np.max( [ z_psd[0][0][0], psd[0][1] ] )
f_max = np.min( [ z_psd[0][0][-1], psd[0][-1] ] )

opd_theory  = opd_rms( np.array(z_psd[0][0]) , np.array(z_psd[0][1]), bandwidth = [f_min,f_max])
opd_sim = opd_rms( psd[0],psd[1][:,i+1], bandwidth = [f_min,f_max] )

plt.xlim([f_min,f_max])"""
#plt.text(1,1e-7, f'opd_theory  rms = {opd_theory} rad between {f_min}-{f_max}Hz' )
#plt.text(1,1e-8, f'opd_sim  rms = {opd_sim} rad between {f_min}-{f_max}Hz' )

#%% optimizing J band coupling 

def ruilier_coupling_efficiency(wvl, f, D, w0 ,alpha):
    """
    wvl - wavelength (m) 
    f = focal length (m)
    D = telescope diameter (m) 
    w0 = fiber mode waist in focal plane (m)
    alpha = central obscuration ratio 
    """
    b = np.pi/2 * D/f * w0/wvl  #beta in Rullier 
    
    
    #coupling efficiency from rullier 1998 (SPIE) assuming perfect wavefront 
    rho = 2 * ( np.exp(-b**2) * (1-np.exp(b**2*(1-alpha**2)) ) / (b * np.sqrt(1-alpha**2)) )**2  
    
    return(rho)

# for alpha = 0 optimal coupling beta = 1.12 


D_ut = 8.2 #m
D_at = 1.8 #m
alpha_ut = 1/D_ut
alpha_at = 0.138/D_at #m https://www.eso.org/sci/facilities/paranal/telescopes/vlti/at/technic.html

f_ut = 20
f_at = 4.5
wvl = 1.25e-6
w0 = np.logspace(-8,-2,200)

beta_ut = np.pi/2 * D_ut/f_ut * w0/wvl
beta_at = np.pi/2 * D_at/f_at * w0/wvl

eta_ut = [ruilier_coupling_efficiency(wvl, f_ut, D_ut, w ,alpha_ut) for w in w0]
eta_at = [ruilier_coupling_efficiency(wvl, f_at, D_at, w ,alpha_at) for w in w0]

print(f'optimal waist = {w0[np.nanargmax(eta_ut)]}')

plt.figure()
plt.semilogx( w0, eta_ut ,label=f'UT (D={D_ut}m, f={f_ut}m, '+r'$\alpha$='+f'{round(alpha_ut,2)})',color='k',linestyle=':')
plt.semilogx( w0, eta_at ,label=f'AT (D={D_at}m, f={f_at}m, '+r'$\alpha$='+f'{round(alpha_at,2)})',color='k',linestyle='-.')
plt.ylabel('coupling efficiency',fontsize=12)
plt.xlabel('waveguide field waist [m]',fontsize=12)
plt.legend(loc='lower left',fontsize=12)
plt.title(r'$\lambda$='+f'{1e6*wvl}'+r'$\mu m$, ' +'Strehl ratio = 1')
plt.tight_layout()
#plt.savefig('/Users/bcourtne/Documents/ANU_PHD2/heimdallr/coupling_efficiency_v_waist.png') 

#%% 

iris_data = pd.read_csv('/Users/bcourtne/Downloads/irisCorr.2022-09-02T04.51.56.txt',skiprows=2,delimiter='\s+')
"""
1st row time (s), second column is time usec, 3rd is error evector x in pieels (1 pixel = 140mas on sky ), 4th is error vector in y.
"""
tiptilt_iris = (iris_data.T.iloc[2]**2 + iris_data.T.iloc[3]**2)**0.5 * 140  #milli arcsec
plt.figure()
plt.hist( tiptilt_iris ,histtype='step',color='k');plt.axvline(np.mean( tiptilt_iris ) ,label=f'$\mu$' )
plt.xlabel('iris tip-tilt error vector rms (mas on sky) ')
plt.ylabel('counts')
plt.legend()
plt.title('irisCorr.2022-09-02T04.51.56.txt')


cf = 3e-1#1.9e-1
# tel_assoc_param_dict holds tuple : D, foc_len, nx_size, pup
tel_assoc_param_dict = {'ut':(8.2, 20,2**8 , aperture.vlt_pupil(dim = 2**8, diameter = 2**8, dead_actuator_diameter=0) ),'at':(1.8, 4.5, 2**7,AT_pupil(dim = 2**7, diameter = 2**7))}

naomi_sim_dict = {'bright': (1e-3 , 14), 'faint': (8.5 * 3e-3 , 7)} #AO_lag, n_terms

naomi_expected_perform =  {'bright': (0.58 , 30), 'faint': (0.21 , 85)} # H strehl , tip-tilt rms (m")
"""
AO_lag = full loop delay of AO system between measurement and correction (s) - Woillez AA 629, A41 (2019) 
n_terms = number corrected zernike modes from naomi 
"""

#========== atm parameters 

wvl=1.25e-6
r0 = 0.092 #m at 500nm 
seeing_500 = 180/np.pi*3600*(0.98 * 500e-9/r0) #seeing (") at 500nm
L0 = 24 #m
V_phase = 85 # velocity of phase screen (m/s)
#Greenwood frequency.
print( f'\ntau0 = inverse greenwood frequency = {1/(0.43*V_phase/r0) }s\nseeing 500nm = {seeing_500 }\n\n')

# ========= AO error terms 
Dtmp, r0tmp, tautmp, tau0tmp, Nact_tmp  = 1.8, 0.09, 41e-3, 2.5e-3, 241
sigma2_fit = 0.257 * (Dtmp/r0tmp)**(5/3) * Nact_tmp**(-5/6)
sigma2_servo = (tautmp/tau0tmp)**(5/3)

print(f'sigma2_fit={sigma2_fit},\nsigma2_servo={sigma2_servo}')
#========== fiber parameters 

NA = 0.25 #0.21 # #numerical apperature 
#n_core = 1 #refractive index of core
delta_n = 0.16 #0.16 
a = 1.9e-6 #3.9e-6 #fiber waist (m?)

#fratio = 5.3 #optimal at V=2.4 according to Roddier (we're at V=2.339)
#foc_len = 20 # 4.5  #fratio * D #focal lenght (m)    (used 30 from w/F=0.71 wvl/D (Rulier))

#Vacuum permittivity and  vacuum permeability
epsilon_o , mu_o = 8.854188e-12, 1.256637e-6 #
#cladding index refraction
"""
[1]   n_core**2 - n_clad**2 = NA**2
[2]   n_core - n_clad = delta_n    
=> 
[3]  (delta_n + n_clad)**2 - n_clad**2 = NA**21
[4]   delta_n**2 + 2*delta_n * n_clad = NA**2
=> 
[5]   n_clad = (NA**2 - delta_n**2) / (2 * delta_n)
 n_clad seems too small with the gravity parameters taken from Jocou 2014
"""
n_clad = (NA**2 - delta_n**2) / (2 * delta_n)
n_core = delta_n + n_clad
#V number (relates to number of modes fiber can hold)
V = 2*np.pi*NA*a/wvl
#simplifying coefficient
cH = np.sqrt(2*n_clad)*(epsilon_o/mu_o)**(0.25)/np.sqrt(np.pi)    

#beam waist (1/e)
w = a * (0.65 + 1.619/V**(3/2) + 2.879/V**6) 
#==========  

#figures 
fig1, ax1 = plt.subplots(1,2,figsize=(15,5),sharey=False)
fig2, ax2 = plt.subplots(1,2,figsize=(15,5),sharey=False)
fig3, ax3 = plt.subplots(1,2,figsize=(15,5),sharey=False)
sim_dict = {}  #main dict to hold simulations results for different AO regimes 
for iii, target_type in enumerate(naomi_sim_dict):
    
    print( f'\n===== simulating {target_type} targets =====\n ')
    D, foc_len, nx_size, pup = tel_assoc_param_dict['at']
    
    AO_lag, n_terms = naomi_sim_dict[target_type]
    
    expected_h_strehl, expected_tip_tilt = naomi_expected_perform[target_type]
    
    #D=8 #1.8 #m
    
    #nx_size = 2**8
    pixel_scale = D/nx_size # m/pixel
    
    # number of Zernike terms corrected by naomi 
    #n_terms = 7 # 14
    #zernike basis
    basis = zernike.zernike_basis(n_terms, npix=nx_size)
    
    #pup = AT_pupil(dim = nx_size, diameter = nx_size) 
    # other options 
    # aperture.vlt_pupil(dim = nx_size, diameter = nx_size, dead_actuator_diameter=0)
    # basis[0] 

    
    V_pix = V_phase / pixel_scale # how many pixels to jump per second 
    
    #AO_lag = 9.5 * 4.6e-3 # full loop delay of AO system between measurement and correction (s) - Woillez AA 629, A41 (2019) 
    
    # how many pixels to jump between AO measurement and correction 
    jump = int( AO_lag * V_pix )
    
    # sim sampling time dt = jump * pixel_scale / V_phase 
    
    #apodization (Analytic Fourier transform of single mode fiber Gaussian approx in image plane)
    X = np.arange(-D/2,D/2,pixel_scale)
    XX,YY = np.meshgrid(X,X) # pupil plane coordinate grid (m)
    field_pupil = cH/np.sqrt(np.pi) * np.exp(-w**2 * ((XX/foc_len/wvl)**2 + (YY/foc_len/wvl)**2) ) # here w is in m => XX,YY must be in 1/m

    print(f'w={w}, V={V}, abs(overlap_integral( field_pupil, np.exp(1j*pup)))**2 = {abs(overlap_integral( field_pupil, np.exp(1j*pup)))**2}')

    #########################
    # now simulate performance (strehl , coupling efficiency etc) for either slow (faint) or fast (bright)  naomi correction 
    
    
    #initiate 20 new phase screens.. roll each one for 2s at ~200Hz sampling (200 * jump * pixel_scale)
    instance_dict = {} 
    for phase_screen_instance in range(25): #25
        if np.mod(phase_screen_instance,10)==0:
            print( phase_screen_instance )
        #init screen (radians .. from above analysis)
        phase_screen = aotools.turbulence.infinitephasescreen.PhaseScreenVonKarman(nx_size, pixel_scale,\
              r0 , L0, n_columns=2,random_seed = 1)
        
            
        strehl_h = []
        strehl_j = []
        Ec = []
        coupling_eff_j = []
        zernike_residuals = []
        
    
        for roll in range(80):  #400
        
            #opd at WFS measurements
            then_opd = cf * wvl / (2*np.pi) * phase_screen.scrn * pup # opd in m
            #forcefully remove piston 
            then_opd[pup>0] -= np.nanmean(then_opd[pup>0])
            #the measured coeeficients on Zernike basis
            then_coe = zernike.opd_expand(then_opd, nterms=n_terms, basis=zernike.zernike_basis)
            
            #reconstruct DM shape from measured coeeficients 
            DM_opd = np.sum( basis*np.array(then_coe)[:,np.newaxis, np.newaxis] , axis=0) 
            #forcefully remove piston 
            DM_opd[pup>0] -= np.nanmean(DM_opd[pup>0])
             
            # propagate phase screen determined by wind velocity and AO latency 
            for i in range( jump ): 
                 phase_screen.add_row()
                 
            now_opd = cf * wvl / (2*np.pi) * phase_screen.scrn * pup
            now_opd[pup>0] -= np.nanmean(now_opd[pup>0])
            
            corrected_opd = pup*(now_opd - DM_opd) # m
            #corrected_phase = 2* np.pi/wvl * corrected_opd

            res_coe_tiptilt = zernike.opd_expand(corrected_opd, nterms=3, basis=zernike.zernike_basis) 
            
            ######
            # add another 30nm RMS to both tip and tilt (42nm RMS totoal) in same sign as residual coefficient 
            # to simulate imperfect optics / lab turbulence before reaching NAOMI 
            ######
            corrected_opd += pup * 150e-9 * np.random.normal(loc=0.0, scale=1.0, size=None) * (np.sign(res_coe_tiptilt[1]) * basis[1] + np.sign(res_coe_tiptilt[2]) * basis[2])
            
            # for strehl ratio calc (need to get rid of nan's)            
            corrected_opd_copy = corrected_opd.copy() 
            corrected_opd_copy[np.isnan(corrected_opd_copy)]=0
            
            # now calculate the residuals 
            res_coe = zernike.opd_expand(corrected_opd, nterms=20, basis=zernike.zernike_basis) 
            
            Ec.append( overlap_integral( field_pupil, np.exp(1j* (2*np.pi) / wvl * corrected_opd)) )
            coupling_eff_j.append( abs( Ec[-1] )**2 )
            #strehl_h.append( np.exp(-np.nanvar(2*np.pi / 1.65e-6 * corrected_opd[pup>0])) ) 
            #strehl_j.append( np.exp(-np.nanvar(2*np.pi / 1.25e-6 * corrected_opd[pup>0])) )
            

            strehl_j.append( (np.max( abs( np.fft.fftshift( np.fft.fft2(pup * np.exp(1j*np.pi*2/wvl * corrected_opd_copy) ) ) )**2) / abs( np.fft.fftshift( np.fft.fft2( pup ) ) )[len(pup)//2, len(pup)//2 ]**2) )
            strehl_h.append( (np.max(abs( np.fft.fftshift( np.fft.fft2(pup * np.exp(1j*np.pi*2/1.65e-6 * corrected_opd_copy) ) ) )**2) / abs( np.fft.fftshift( np.fft.fft2( pup ) ) )**2)[len(pup)//2, len(pup)//2 ] )
            
            coupling_eff_j.append( abs( Ec[-1] )**2 )
            zernike_residuals.append( res_coe )
            
            for jjj in range(jump):
                phase_screen.add_row()
                
        ac_unwrap = np.angle(Ec[0]) + np.cumsum( np.angle( Ec[1:] * np.conjugate(Ec[:-1])) )
        
        instance_dict[phase_screen_instance] = { 'corrected_opd':corrected_opd, 'strehl_h':strehl_h,'strehl_j':strehl_j,\
                                                'coupling_eff_j':coupling_eff_j,'zernike_residuals':zernike_residuals,\
                                                   'coupled_pist':ac_unwrap }
        
        #ac_unwrap = np.angle(Ec[0]) + np.cumsum( np.angle( Ec[1:] * np.conjugate(Ec[:-1])) )
        #ac_unwrap_psd = sig.welch(ac_unwrap, fs = 1/dt, nperseg=2**10)
    
    
    abs( np.fft.fftshift( np.fft.fft2(np.exp(1j*np.pi*2/wvl * corrected_opd) ) ) )**2      
    #=====
    sim_dict[target_type] = instance_dict
    #=====
    
    
    
    print( 'np.mean( strehl_h )',np.mean( strehl_h ), '\nnp.mean( coupling_j )',np.mean( coupling_eff_j ))
    #plt.loglog( *sig.welch( instance_dict[5]['coupled_pist'], fs = 1/(jump * pixel_scale / V_phase)))
    
    #-------- measuring coherence time 
    i=0
    go=1
    aaa= np.std( phase_screen.scrn[pup>0.5] )
    while go: 
        test = aaa-np.std( phase_screen.add_row()[pup>0] )
        if abs(test) < 1:
            i+=1
        elif abs(test) > 1:
            go = False
            print(f'{i} iterations for rms phase change by 1rad, therefore simulated tau0 = {i*pixel_scale / V_phase} ' ) 
        else:
            print('we have an issue')
    
    
    
    
    ###### PLOTTING & CALCULATING 
    ########################
    
    # ----- Simulated H Strehl, J Strehl, residuals ,  tip-tilt mas -rms 
    h_strehls = np.array( [instance_dict[i]['strehl_h'] for i in range(len(instance_dict)) ] ).ravel()
    j_strehls = np.array( [instance_dict[i]['strehl_j'] for i in range(len(instance_dict)) ] ).ravel()
    res = np.array([ np.array( instance_dict[i]['zernike_residuals'])[:]  for i in range(len(instance_dict)) ] )#.ravel()
    tip_res = np.array([ np.array( instance_dict[i]['zernike_residuals'])[:,1]  for i in range(len(instance_dict)) ] ).ravel() # m at wvl
    tilt_res = np.array([ np.array( instance_dict[i]['zernike_residuals'])[:,2] for i in range(len(instance_dict)) ] ).ravel() # m at wvl
    
    res_rms = 1e9 * np.std( np.array( [instance_dict[i]['zernike_residuals'] for i in range(len(instance_dict))])  , axis=(0,1))
    tip_tilt_mas = ( (tip_res*(1.65/1.25) )**2 + (tilt_res*(1.65/1.25) )**2 )**0.5  / 1.8  * ( 1e3* 3600 * 180/np.pi )
    #j_coupling = np.array( [instance_dict[i]['coupling_eff_j'] for i in range(len(instance_dict)) ] ).ravel()
    
    
    #1e-9 * (res_rms[2]**2 + res_rms[1]**2)**0.5 *(1.65/1.25) /1.8  * (  1e3* 3600 * 180/np.pi )
    

    #fig, ax = plt.subplots(1,2,figsize=(15,5),sharey=True)
    ax1[iii].set_ylabel('residual zernike coefficient '+r'$a_i$'+'\n[nm]')
    mode_names = [zernike.zern_name(j+1) for j in range(20)]
    for i in range(len(instance_dict)):
        ax1[iii].plot(mode_names[1:], 1e9 * np.array( instance_dict[i]['zernike_residuals'] ).T[1:,:],'.',color='k',alpha=0.01)
    ax1[iii].plot(mode_names[1:], 1e9 * np.std( np.array( [instance_dict[i]['zernike_residuals'] for i in range(len(instance_dict))]) , axis=(0,1))[1:] , '-', color='r',label=r'$<a_i^2>_t$')
    ax1[iii].plot(mode_names[1:], 1e9 * np.mean( np.array( [instance_dict[i]['zernike_residuals'] for i in range(len(instance_dict))]) , axis=(0,1))[1:] , '-', color='b',label=r'$<a_i>$')
    ax1[iii].legend() 
    ax1[iii].set_title(f'{target_type} target\n'+r'seeing = 1.1", $\tau_0$=2.5ms')
    
    ax1[iii].tick_params('x', labelrotation=90)
    ax1[iii].set_ylim([-600,600])
    fig1.tight_layout()
    fig1.savefig('/Users/bcourtne/Documents/ANU_PHD2/heimdallr/naomi_zernike_residuals_simulation.png') 
    #plt.xtick(zernike.zern_name(i+1))
    
    
    #fig, ax = plt.subplots(1,2,figsize=(15,5),sharey=True)
    ax2[iii].set_ylabel('counts',fontsize=15)
    ax2[iii].hist(h_strehls, label='simulation', color='k',histtype='step' ,density=True ,stacked=True) 
    ax2[iii].axvline(np.mean(h_strehls),color='blue',label='simulation mean')
    ax2[iii].axvline(expected_h_strehl, color='cyan',label='measured - Woillez AA 629, A41 (2019)')
    ax2[iii].set_xlabel('H-band Strehl Ratio',fontsize=15)
    ax2[iii].set_title(f'{target_type} target\n'+r'seeing = 1.1", $\tau_0$=2.5ms')
    ax2[iii].legend(loc='upper left')
    fig2.tight_layout()
    fig2.savefig('/Users/bcourtne/Documents/ANU_PHD2/heimdallr/naomi_H-strehl_simulation.png') 
    
    #fig, ax = plt.subplots(1,2,figsize=(15,5),sharey=True)
    ax3[iii].set_ylabel('counts',fontsize=15)
    ax3[iii].hist(tip_tilt_mas, label='simulation', color='k',histtype='step',density=True ,stacked=True) 
    ax3[iii].axvline(np.mean(tip_tilt_mas),color='blue',label='simulation mean')
    if target_type=='bright':
        ax3[iii].hist( tiptilt_iris , histtype='step',density=True,stacked=True,color='brown',linestyle=':',label='VLTI/IRIS measurements')#; plt.axvline(np.mean( tiptilt_iris ) ,label=f'$\mu$' )
        ax3[iii].axvline(np.mean(tiptilt_iris),color='lightblue',label='VLTI/IRIS mean')
    ax3[iii].axvline(expected_tip_tilt, color='cyan',label='measured - Woillez AA 629, A41 (2019)')
    ax3[iii].set_xlabel('on sky tip-tilt RMS [m"]',fontsize=15)
    
    ax3[iii].set_title(f'{target_type} target\n'+r'seeing = 1.1", $\tau_0$=2.5ms')
    ax3[iii].legend(loc='upper right')
    fig3.tight_layout()
    fig3.savefig('/Users/bcourtne/Documents/ANU_PHD2/heimdallr/naomi_tip_tilt_res_simulation.png') 
    
    
    # ----- Phase screens before /after naomi correction 
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    fig = plt.figure(figsize=(15, 5))
    
    ax11 = fig.add_subplot(131)
    ax11.set_title('uncorrected pupil',fontsize=20)
    ax11.axis('off')
    im11 = ax11.imshow( 1e6 * then_opd )
    ax11.text(nx_size//4,nx_size//4, r'$\lambda$ = '+f'{round(1e6*wvl,2)}'+r'$\mu m$',color='r', fontsize=29)
    
    divider = make_axes_locatable(ax11)
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    cbar = fig.colorbar( im11, cax=cax, orientation='horizontal')
    cbar.set_label( r'OPD $(\mu m)$', rotation=0, fontsize=20)
    cax.tick_params(labelsize=15)
    
    ax22 = fig.add_subplot(132)
    ax22.set_title('NAOMI DM shape',fontsize=20)
    ax22.axis('off')
    im22 = ax22.imshow( 1e6 * DM_opd )
    
    divider = make_axes_locatable(ax22)
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    cbar = fig.colorbar(im22, cax=cax, orientation='horizontal')
    cbar.set_label( r'OPD $(\mu m)$', rotation=0, fontsize=20)
    cax.tick_params(labelsize=15)
    
    ax33 = fig.add_subplot(133)
    ax33.set_title('NAOMI residual',fontsize=20)
    ax33.axis('off')
    im33 = ax33.imshow( 1e6 * corrected_opd )
    
    divider = make_axes_locatable(ax33)
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    cbar = fig.colorbar(im33, cax=cax, orientation='horizontal')
    cbar.set_label( r'OPD $(\mu m)$', rotation=0, fontsize=20)
    cax.tick_params(labelsize=15)

    fig.tight_layout()
    fig.savefig(f'/Users/bcourtne/Documents/ANU_PHD2/heimdallr/{target_type}_naomi_phasescreens_sim.png') 
#plt.imshow(corrected_opd )
#print( np.nanvar(corrected_opd[pup>0]) )


print(f'strehl at {round(1e6 * wvl,2)}um = {np.exp(-np.nanvar(2*np.pi / wvl * corrected_opd[pup>0]))}' )
    
    




#plt.plot( 1e9 * np.std( np.array( instance_dict[0]['zernike_residuals'] )  , axis=0) ,'.')

#tip-tilt_mas_rms = (80**2 + 50**2)**0.5*1e-9/1.8 * 1e3 * 3600 * 180/np.pi

########################
