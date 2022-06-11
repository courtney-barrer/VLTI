#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 03:31:29 2022

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
#import pyzelda
#from pyzelda import utils

import astropy.units as u
from astropy import coordinates as coord 
from astropy.time import Time
import datetime

os.chdir('/Users/bcourtne/Documents/mac1_bk/Hi5/vibration_analysis/utilities')
import fiber_fields as fields

def overlap_integral(E1, E2): 
    #fiber overlap integral to calculate coupled efficienccy and phase
    eta = ( np.nansum( E1 * np.conjugate(E2) ) ) / ( np.nansum(E1 * np.conjugate(E1) ) * np.nansum( E2 * np.conjugate(E2)) )**0.5
    
    return(eta)

def calc_omega(X,Y,W):
    omega = np.nansum( X*np.conjugate(Y)*W ) / ( np.nansum( X*np.conjugate(X) * W ) * np.nansum( Y*np.conjugate(Y) * W ) )**0.5
    return(omega)

"""
#%% Section 0: hard coded stuff
#%% Section 1: theoretical cross coupling simulation
#%% Section 2: gravity/macao case study 
"""

#%%  Section 0: hard coded stuff 

base2telname = [[4, 3], [4, 2], [4, 1], [3, 2], [3, 1], [2, 1]]
tel2telname = [4, 3, 2, 1]
base2tel = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
tel2base = [[0, 1, 2], [0, 3, 4], [1, 3, 5], [2, 4, 5]]
color = ['b', 'g', 'r', 'c', 'm', 'y']
M_matrix = np.array([[1.0, -1.0, 0.0, 0.0],
                     [1.0, 0.0, -1.0, 0.0],
                     [1.0, 0.0, 0.0, -1.0],
                     [0.0, 1.0, -1.0, 0.0],
                     [0.0, 1.0, 0.0, -1.0],
                     [0.0, 0.0, 1.0, -1.0]])

#========== pupil coordinates
D = 8 # telescope diameter (m)
D_pixel = 2**9+1 #number of pixels in pupil
dX = D/D_pixel  # differential element in pupil plane (m)
X = np.arange(-D/2,D/2,dX) # pupil plane coordinates (m)
XX,YY = np.meshgrid(X,X) # pupil plane coordinate grid (m)

#========== fiber parameters 
wvl=2.2e-6
NA = 0.21 # #numerical apperature 
#n_core = 1 #refractive index of core
delta_n = 0.16 
a = 3.9e-6 #fiber waist (m?)

fratio = 5.3 #optimal at V=2.4 according to Roddier (we're at V=2.339)
foc_len = 30  #fratio * D #focal lenght (m)    (used 30 from w/F=0.71 wvl/D (Rulier))


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


#telescope pupil 
pup = aperture.disc(size=D_pixel//2,dim=D_pixel)
pup_obs = aperture.disc_obstructed(size=D_pixel//2,dim=D_pixel,obs=1/8)
pup_vlt = aperture.vlt_pupil(D_pixel,diameter=D_pixel,dead_actuator_diameter=0.0)

#Zernike basis 
basis = np.nan_to_num(zernike.zernike_basis(nterms=60, npix=pup_vlt.shape[0])) 
# make sure it really is zero mean over pupil (no piston!)
for i in range(1,len(basis)):
    filt = pup!=np.min(pup)
    basis[i][filt] = basis[i][filt] -np.mean(basis[i][filt])
    
#apodization (Analytic Fourier transform of single mode fiber Gaussian approx in image plane)
field_pupil = cH/np.sqrt(np.pi) * np.exp(-w**2 * ((XX/foc_len/wvl)**2 + (YY/foc_len/wvl)**2) ) # here w is in m => XX,YY must be in 1/m

#check normalization <Z|Z>_P/<1,1>_P = 1 <1|1>_p = sum(P) = area (number of pixels)
#print( 1/((len(basis[3])/2)**2*np.pi ) * np.sum((basis[3][pup!=0])**2) )

#%% Section 1: theoretical cross coupling simulations 


#field_pupil = cH/np.sqrt(np.pi) * np.exp(-w**2 * ((XX)**2 + (YY)**2) ) # here w is in m => XX,YY must be in 1/m

#Check overlap integral (should be closed to maximum...)
print(f'calc coupling efficiency={abs(overlap_integral(pup , field_pupil))**2}, theoretical max = 0.78 (Shaklan & Roddier) ')

"""
aaa = []
for i in np.logspace(-10, 10, 100):
    
    field_pupil = cH/np.sqrt(np.pi) * np.exp(-w**2 * ((XX/i)**2 + (YY/i)**2) )
    
    aaa.append(abs(overlap_integral(pup , field_pupil))**2)
    
plt.loglog(np.logspace(-10, 10, 100), aaa)
np.logspace(-10, 10, 100)[np.argmax(aaa)]



plt.figure()
from matplotlib import colors
plt.pcolormesh( pup_vlt * field_pupil / np.max(field_pupil)  ,norm=colors.LogNorm() )
plt.gca().set_aspect(1)
plt.axis('off')
plt.colorbar()
"""

"""
To study modal cross coupling to piston we consider the Zernike modal basis with each mode normalized to have spatial mean = 0 (besides $Z_0$) over a circular (non-obscured) pupil which we denote $P_0$. For each Zernike mode $Z_i$ we draw 1000 coefficients $a_i$ from a normal distribution with mean = 1, variance = 1 (simulating a modal temporal variance=1 radian$^2$). For each sample we calculate the piston for the waveguide coupled (apodized pupil) system considering the VLTI pupil which we denote $P_\mathrm{VLTI}$: 
\begin{equation}
    p^{apodized}_i =  \mathrm{arg} \frac{ \braket{E_i|M}_{P_\mathrm{VLTI}} }{ (\braket{E_i|E_i}_{P_\mathrm{VLTI}} \braket{M|M}_{P_\mathrm{VLTI}})^{1/2} } 
\end{equation}
Where $E_j=e^{i a_j Z_j)}$. Similarly for the system without waveguide coupling (non-apodized pupil):
\begin{equation}
    p^{non-apodized}_i = a_i \braket{Z_i}_{P_\mathrm{VLTI}} 
\end{equation}
Across all the sampled coefficents for a given input mode $Z_i$ we then calculate the pistons standard deviation denoted for the waveguide and non-waveguide coupled systems as $\sigma^{apodized}_i$ and $\sigma^{non-apodized}_i$ respectively. These are plotted in figure XXXX. 

"""




#####
# ----- mapping spatial variance of zernike mode to fiber coupled piston 

ais = np.logspace(-3,1,100)
phase_tilt = [ np.angle( calc_omega( field_pupil, np.exp(1j*ai*basis[1]), pup_vlt) ) for ai in ais ] 
phase_focu = [ np.angle( calc_omega( field_pupil, np.exp(1j*ai*basis[3]), pup_vlt) ) for ai in ais ] 
phase_asti = [ np.angle( calc_omega( field_pupil, np.exp(1j*ai*basis[4]), pup_vlt) ) for ai in ais ] 
phase_sphe = [ np.angle( calc_omega( field_pupil, np.exp(1j*ai*basis[10]),pup_vlt) ) for ai in ais ] 

#fig,ax = plt.subplots(1,2, figsize=(10,5))
fig = plt.figure( figsize=(15,5) ) 
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
plt.subplots_adjust(hspace=0)

ax1.plot(ais, phase_tilt , label = zernike.zern_name(2))
ax1.plot(ais, phase_focu , label = zernike.zern_name(4))
ax1.plot(ais, phase_asti , label = zernike.zern_name(5))
ax1.plot(ais, phase_sphe , label = zernike.zern_name(11))
ax1.set_ylabel( r'coupled phase = arg $\Omega$',fontsize=15)
ax1.set_xlabel( r'spatial std ($a_i$) of input Zernike mode (radians)',fontsize=15)
ax1.legend(fontsize=15)
ax1.tick_params(labelsize=15)

ax2.pcolormesh( pup_vlt * field_pupil / np.max(field_pupil) , norm=colors.LogNorm() )
ax2.set_aspect(1)
ax2.set_title('apodized VLTI pupil \n(log norm color scale)',y=-0.18,fontsize=15)
ax2.axis('off')

plt.tight_layout()
plt.savefig('/Users/bcourtne/Documents/ANU_PHD2/SPIE_2022/var_zern_2_couled_phase.png')

#####
# ----- cross coupling coefficients with different a_i var and pupil and apodization  

cross_coupling_dict = {}
N_samp = 1000# number of samples drawn for each zernike 
N_zern = 15 #no zernike modes to consider cross coupling 

#get our coefficients 
"""coes = {}
ais = np.random.normal(loc=0, scale=1.0, size=N_samp )
for i in range(N_zern ):
    z_mode = zernike.zern_name(i+1)
    coes[z_mode] = ais"""
    
ai_stds = [1, 0.5] #std to use in a_i distribution to simulate case of strong / weak aberrations 

strong_ais = np.random.normal(loc=0, scale=ai_stds[0] , size=N_samp )
weak_ais = np.random.normal(loc=0, scale=ai_stds[1], size=N_samp )

for coe_lab, coes in zip(['strong', 'weak'],[strong_ais, weak_ais]):
    
    print(f'calculating for {coe_lab} turbulence residuals')
    
    cross_coupling_dict[coe_lab] = {}
    
    for p_lab, p in zip(['circ','obs','vlti'],[pup, pup_obs, pup_vlt]):
        
        print('calculating for ', p_lab,' pupil')
        p_apod = {}
        p_noapod = {}
        #coes ={}
        for i in range(N_zern ):
            #name of zernike mode 
            z_mode = zernike.zern_name(i+1)
            # sampling our zernike coeficients 
            #coes[z_mode] = np.random.normal(loc=0, scale=1.0, size=N_samp )
           
            # fiber coupled piston 
            p_apod[z_mode] = [ np.angle( calc_omega( field_pupil, np.exp(1j*ai*basis[i]), p) ) for ai in coes ]
            # non-fiber coupled piston 
            p_noapod[z_mode] = [ np.sum( ai * basis[i] * p) / np.sum(pup) for ai in coes ]
        
        
        #calculate the piston rms (std) over the simulation 
        sigma_p_apod = np.array( [np.std(p_apod[z_mode]) for z_mode in p_apod.keys()] ) 
        sigma_p_noapod = np.array( [np.std(p_noapod[z_mode]) for z_mode in p_noapod.keys()] )  
        sigma_coes = np.std(coes) 
        
        cross_coupling_dict[coe_lab][p_lab] = { 'apod':p_apod, 'no_apod':p_noapod ,'coes':coes, \
                                      'sigma_apod': sigma_p_apod, 'sigma_no_apod':sigma_p_noapod, 'sigma_coes':sigma_coes,\
                                        'sigma_apod_err': sigma_p_apod/np.sqrt(2*(N_samp-1)), 'sigma_no_apod_err':sigma_p_noapod/np.sqrt(2*(N_samp-1)), 'sigma_coes_err':sigma_coes/np.sqrt(2*(N_samp-1))  }
        

#Plotting 
fig,ax = plt.subplots(2,2,sharey='row',sharex='col',figsize=(15,14))
plt.subplots_adjust(hspace=0)

ax[0,1].set_title(r'$\bf{small\ temporal\ mode\ variance}$',fontsize=21,fontweight="bold")#, fontname="Times New Roman")
ax[0,0].set_title(r'$\bf{large\ temporal\ mode\ variance}$',fontsize=21, fontweight="bold")#,fontname="Times New Roman")

for i,coe_lab in enumerate(['strong', 'weak']):
    
    
    
    ax[0,i].errorbar(cross_coupling_dict[coe_lab]['circ']['no_apod'].keys(), cross_coupling_dict[coe_lab]['circ']['sigma_no_apod'] / ai_stds[i], yerr = 2*cross_coupling_dict[coe_lab]['circ']['sigma_no_apod_err'] / ai_stds[i] , label = r'$P_0$')  
    #ax[0,i].errorbar(cross_coupling_dict[coe_lab]['circ']['no_apod'].keys(), cross_coupling_dict[coe_lab]['obs']['sigma_no_apod'] / ai_stds[i] , yerr = 2*cross_coupling_dict[coe_lab]['obs']['sigma_no_apod_err'] /ai_stds[i], label = r'$P_1$')  
    ax[0,i].errorbar(cross_coupling_dict[coe_lab]['circ']['no_apod'].keys(), cross_coupling_dict[coe_lab]['vlti']['sigma_no_apod'] /ai_stds[i], yerr = 2*cross_coupling_dict[coe_lab]['vlti']['sigma_no_apod_err'] /ai_stds[i]  , label = r'$P_{VLTI}$')     
    ax[0,i].tick_params(axis ='y',labelsize=20)
    ax[0,i].legend(fontsize=20)
    ax[0,i].grid()
    ax[0,i].text(5,0.8,r'$Var(a_i) = {} rad^2$'.format(round(ai_stds[i]**2,2)),fontsize=18)
    
    
    ax[1,i].errorbar(cross_coupling_dict[coe_lab]['circ']['apod'].keys(), cross_coupling_dict[coe_lab]['circ']['sigma_apod']/ai_stds[i]  , yerr = 2*cross_coupling_dict[coe_lab]['circ']['sigma_apod_err'] /ai_stds[i], label = r'$P_0$')  
    #ax[1,i].errorbar(cross_coupling_dict[coe_lab]['circ']['apod'].keys(), cross_coupling_dict[coe_lab]['obs']['sigma_apod']/ai_stds[i], yerr = 2*cross_coupling_dict[coe_lab]['obs']['sigma_apod_err'] /ai_stds[i] , label = r'$P_1$' ) 
    ax[1,i].errorbar(cross_coupling_dict[coe_lab]['circ']['apod'].keys(), cross_coupling_dict[coe_lab]['vlti']['sigma_apod']/ai_stds[i], yerr = 2*cross_coupling_dict[coe_lab]['vlti']['sigma_apod_err'] /ai_stds[i] , label = r'$P_{VLTI}$' )  
    ax[1,i].legend(fontsize=20)
    ax[1,i].grid()
    ax[1,i].text(5,0.8,r'$Var(a_i) = {} rad^2$'.format(round(ai_stds[i]**2,2)),fontsize=18)
for i in [0,1]:
    ax[1,i].set_xticklabels(p_apod.keys(),rotation=90,fontsize=16)
    ax[1,i].set_xlabel('input Zernike mode',fontsize=21)
    ax[i,0].tick_params(axis ='y',labelsize=20)
    
ax[0,0].set_ylabel(r'$\bf{(non-apodized)}$'+'\npiston cross coupling coefficient',fontsize=21)
ax[1,0].set_ylabel(r'$\bf{(apodized)}$'+'\npiston cross coupling coefficient',fontsize=21)#, fontname="Times New Roman")

plt.tight_layout()
#plt.savefig('/Users/bcourtne/Documents/ANU_PHD2/SPIE_2022/cross_coupling_coef_2_piston.png')


#%% Section 2: gravity/macao case study 

#########
# Check gravity data for observed around 6:06 with MACAO 
#########


os.chdir('/Volumes/vol1BCB/UT-gravity_july_run/25-07-21/gravity')

gravi = fits.open( '/Volumes/vol1BCB/UT-gravity_july_run/25-07-21/gravity/GRAVI.2021-07-26T06-05-44.677.fits' ) 

grav_dict = {}
macao_dict = {} 
macao_delay = {} #how much later was MACAO startec
for i in [1,2,3,4]: #UT telescopes 
    
    macao_file = glob.glob(f'/Volumes/vol1BCB/UT-gravity_july_run/25-07-21/UT{i}/AO_data/RTCData_2021-07-26_06-06-*.fits')
    macao_dict[i] = fits.open( macao_file[0] )
    
    macao_delay[i] = ( macao_dict[1][0].header['MJD-OBS'] - gravi[0].header['MJD-OBS'] ) * 24 * 60 * 60 # seconds 

    

print('AO system = {}'.format(gravi[0].header['HIERARCH ESO COU AO SYSTEM']))   
print('start time template = {}'.format(gravi[0].header['HIERARCH ESO TPL START']))
print('exptime for template = {}'.format(gravi[0].header['EXPTIME']))
print('gravity object = {} '.format(gravi[0].header['*OBJECT*'] ))#13.1 Vmag, 10.0 Kmag 
print('FT DIT = {}'.format(gravi[0].header['HIERARCH ESO DET3 SEQ1 DIT'])) # = 3ms


##########################
####### processing gravity data 
##########################

opdc = gravi['OPDC'].data
opdc_time = opdc['TIME']
opdc_opd = opdc['OPD']
opdc_kopd = opdc['KALMAN_OPD']
opdc_kpiezo = opdc['KALMAN_PIEZO']
opdc_steps = opdc['STEPS']
opdc_state = opdc['STATE']

dt = np.median(np.diff(opdc_time))*1e-6

## --- timestamps!!!
grav_time = datetime.datetime.strptime(gravi[0].header['HIERARCH ESO TPL START'] , '%Y-%m-%dT%H:%M:%S') \
        + pd.to_timedelta(opdc_time, 'us')


# Recover missing samples in OPDC
dit = np.median(np.diff(opdc_time))
missing = np.where(np.diff(opdc_time) > (1.5*dit))[0]
print("Fixed {0} missing steps in OPDC".format(len(missing)))
n = 0
for i in missing:
    n += 1
    opdc_time   = np.insert(opdc_time,   i+n, opdc_time[i+n-1]+dit, axis=0)
    opdc_opd    = np.insert(opdc_opd,    i+n, opdc_opd[i+n-1],      axis=0)
    opdc_kopd   = np.insert(opdc_kopd,   i+n, opdc_kopd[i+n-1],     axis=0)
    opdc_kpiezo = np.insert(opdc_kpiezo, i+n, opdc_kpiezo[i+n-1],   axis=0)
    opdc_steps  = np.insert(opdc_steps,  i+n, opdc_steps[i+n-1],    axis=0)

# Reconstruct phase modulation
opdc_mods = np.zeros([opdc_steps.shape[0], 4])
for iTel in range(4):
    opdc_mods[:, iTel] = (((opdc_steps >> (4*iTel))&15)*np.pi/8.0)
    
# Convert opdc_kpiezo, and opdc_mods from op to opd (i.e. telescope to baseline)
opdc_kpiezo = (M_matrix @ opdc_kpiezo.T).T
opdc_mods = (M_matrix @ opdc_mods.T).T


# Reconstruct residuals without phase modulations
phase_residuals = (opdc_opd - opdc_mods + np.pi) % (2.0*np.pi) - np.pi

# Reconstruct disturbances
phase_disturbances = opdc_kpiezo + (opdc_opd - (opdc_kopd-np.pi)) % (2.0*np.pi) + (opdc_kopd-np.pi)

# Convert phase in [rad] to length in [µm]
disturbances = phase_disturbances * 2.25 / (2.*np.pi)
residuals    = phase_residuals    * 2.25 / (2.*np.pi)

for idx,b in enumerate([f'{i}{j}' for i,j in base2telname]):
    grav_dict[b]={}
    grav_dict[b]['res'] = residuals[:,idx]
    grav_dict[b]['disturb'] = disturbances[:,idx]

##########################
####### processing macao data 
##########################

# --- timestamps!!!
macao_time = {}


for i in [1,2,3,4]:      
    
    t_tmp = np.arange(0,1e-6 * np.median(np.diff(macao_dict[i][1].data['TimeStamp'][:,1])) * len(macao_dict[i][1].data['TimeStamp']), np.median(np.diff(macao_dict[i][1].data['TimeStamp'][:,1]))*1e-6 )  # seconds from start 
    
    macao_time[i] =  grav_time[0] + pd.to_timedelta(macao_delay[i] ,unit='s') + pd.to_timedelta(t_tmp , unit='s')





##########################
####### plotting PSDs of Gravity and AO residual coupled piston 
##########################
b = '32'

T1 = int(b[0])
T2 = int(b[1])

basis = np.nan_to_num(zernike.zernike_basis(nterms=70, npix=pup_vlt.shape[0])) 
samples = macao_dict[4][1].data['CWSZernike'].shape[0] // 4
#phi_res = 0*basis[0]

phi_res={}
p_res={}
for T in [T1,T2]:
    print('calculating for UT',T)
    phi_res[T]={}
    
    for j in range(samples):
        if np.mod(j,1000)==0:
            print(round(j/samples,3) , ' complete for UT ',T)
        # tip/tilt
        phi_res[T][j] = macao_dict[T][1].data['CWSTip'][j] * basis[1] + macao_dict[T][1].data['CWSTilt'] * basis[2] 
        # HO modes
        for i  in range(macao_dict[T][1].data['CWSZernike'].shape[1]):
            
            phi_res[T][j] += macao_dict[T][1].data['CWSZernike'][j,i] * basis[i+3] 
    
        #phi_res[T][j] = phi_res[j]

    #macao_dict[T][0].header['HIERARCH ESO AOS RTC LOOP STREHL']
    #np.nanmean( [np.exp(-np.nanvar(phi_res[j][pup!=0])) for j in range(samples)] )
    
    p_res[T] = np.array( [ np.angle( calc_omega( field_pupil, np.exp( 1j* phi_res[T][j] ), pup_vlt ) ) for j in range(samples) ] )
    
    
    

grav_psd = sig.welch(grav_dict[b]['disturb'],fs=1/gravi[0].header['HIERARCH ESO DET3 SEQ1 DIT'], nperseg=2**10)

#MACAO central wvl = 0.7um . convert to um 
p1_res_psd = sig.welch( 0.7/(np.pi*2) * p_res[T1] ,fs=macao_dict[T1][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**10,axis=0)
p2_res_psd = sig.welch( 0.7/(np.pi*2) * p_res[T2] ,fs=macao_dict[T1][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**10,axis=0)
macao_opd_res_psd = sig.welch( 0.7/(np.pi*2) * (p_res[T2]-p_res[T1]) ,fs=macao_dict[T1][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**9,axis=0)

macao1_HO_psd = sig.welch( 0.7/(np.pi*2) * macao_dict[T1][1].data['CWSZernike'],fs=macao_dict[T1][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**10,axis=0)
macao2_HO_psd = sig.welch( 0.7/(np.pi*2) * macao_dict[T2][1].data['CWSZernike'],fs=macao_dict[T2][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**10,axis=0)

macao1_pist_psd = sig.welch( 0.7/(np.pi*2) * macao_dict[T1][1].data['EstPiston'],fs=macao_dict[T2][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**10,axis=0)
macao1_tip_psd = sig.welch( 0.7/(np.pi*2) * macao_dict[T1][1].data['CWSTip'],fs=macao_dict[T2][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**10,axis=0)
macao1_tilt_psd = sig.welch( 0.7/(np.pi*2) * macao_dict[T1][1].data['CWSTilt'],fs=macao_dict[T2][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**10,axis=0)

macao1_DM_psd = sig.welch( 0.7/(np.pi*2) * macao_dict[T1][1].data['DMZernike'],fs=macao_dict[T1][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**10,axis=0)

macao2_pist_psd = sig.welch( 0.7/(np.pi*2) * macao_dict[T2][1].data['EstPiston'],fs=macao_dict[T2][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**10,axis=0)
macao2_tip_psd = sig.welch( 0.7/(np.pi*2) * macao_dict[T2][1].data['CWSTip'],fs=macao_dict[T2][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**10,axis=0)
macao2_tilt_psd = sig.welch( 0.7/(np.pi*2) * macao_dict[T2][1].data['CWSTilt'],fs=macao_dict[T2][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**10,axis=0)

plt.figure(figsize=(10,8))
plt.loglog(*grav_psd, color='g',lw=2, label='Gravity fringe tracker pseudo open loop OPD ')
plt.loglog(*macao_opd_res_psd ,lw=2, color='r', label='Gravity coupled OPD from MACAO residuals')
plt.loglog(*macao1_HO_psd, alpha=0.3, color='k')
plt.loglog([-1],[1], alpha=0.3, color='k',label='MACAO Zernike mode residuals')
plt.xlim([np.min(grav_psd[0]),np.max(grav_psd[0])])
plt.gca().tick_params(labelsize=20)
plt.xlabel('frequency (Hz) ',fontsize=20)
plt.ylabel(r'PSD $(\mu m^2/Hz)$',fontsize=20)
plt.legend(fontsize=15)
#plt.tight_layout()

plt.savefig()


##########################
####### plotting ts
##########################


b = '32'

T1 = int(b[0])
T2 = int(b[1])



t0 = np.min( [np.min(macao_time[T1]), np.min(macao_time[T2])] )
t1 = np.max( [np.max(macao_time[T1]), np.max(macao_time[T2])] )
www=5 #rollingwindow
time_filt = (grav_time <= t1) & (grav_time >= t0)

grav_time_wind = grav_time[time_filt]
grav_dict_wind = grav_dict[b]['res'][time_filt]

for aaa in range(32):
    plt.figure()
    i1,i2=(aaa*1000),(aaa+1)*1000
    
    time_filt2 = (grav_time_wind <= macao_time[T1][i2]) & (grav_time_wind >= macao_time[T1][i1])
    
    plt.plot(grav_time_wind[time_filt2] , 10*grav_dict_wind[time_filt2], '-',color='r')
    #plt.plot(macao_time[T1][i1:i2], 60+10*pd.Series( macao_dict[T1][1].data['CWSTilt'][i1:i2] ).rolling(1000).median(), color='b')
    plt.plot(macao_time[T1][i1:i2], 25+ pd.Series(macao_dict[T1][1].data['CWSTip'][i1:i2]).rolling(www).mean(), color='b')
    plt.plot(macao_time[T1][i1:i2], 50+ pd.Series(macao_dict[T1][1].data['CWSTip'][i1:i2]).rolling(www).mean(), color='b')
    
    plt.plot(macao_time[T2][i1:i2], -25- pd.Series(macao_dict[T2][1].data['CWSTip'][i1:i2]).rolling(www).mean(), color='b')
    plt.plot(macao_time[T2][i1:i2], -50- pd.Series(macao_dict[T2][1].data['CWSTip'][i1:i2]).rolling(www).mean(), color='b')
    for i in range(8):
        plt.plot(macao_time[T1][i1:i2], 75 + 25*i + pd.Series(macao_dict[T1][1].data['CWSZernike'][i1:i2,i]).rolling(www).mean(), color='k')
        plt.plot(macao_time[T2][i1:i2], -75 - 25*i + pd.Series(macao_dict[T2][1].data['CWSZernike'][i1:i2,i]).rolling(www).mean(), color='k')
        #plt.plot(macao_time[T1][i1:i2], 60+macao_dict[T1][1].data['CWSTilt'][i1:i2], color='b')
        #plt.plot(macao_time[T2][i1:i2], 30+macao_dict[T2][1].data['CWSZernike'][i1:i2,i], color='k')
    

##########################
####### plotting PSD
##########################


b = '21'

T1 = int(b[0])
T2 = int(b[1])

grav_dict_wind = grav_dict[b]['disturb'][time_filt]

grav_psd = sig.welch(grav_dict[b]['disturb'],fs=1/gravi[0].header['HIERARCH ESO DET3 SEQ1 DIT'], nperseg=2**10)

#MACAO central wvl = 0.7um . convert to um 
macao1_HO_psd = sig.welch( 0.7/(np.pi*2) * macao_dict[T1][1].data['CWSZernike'],fs=macao_dict[T1][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**10,axis=0)
macao2_HO_psd = sig.welch( 0.7/(np.pi*2) * macao_dict[T2][1].data['CWSZernike'],fs=macao_dict[T2][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**10,axis=0)

macao1_pist_psd = sig.welch( 0.7/(np.pi*2) * macao_dict[T1][1].data['EstPiston'],fs=macao_dict[T2][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**10,axis=0)
macao1_tip_psd = sig.welch( 0.7/(np.pi*2) * macao_dict[T1][1].data['CWSTip'],fs=macao_dict[T2][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**10,axis=0)
macao1_tilt_psd = sig.welch( 0.7/(np.pi*2) * macao_dict[T1][1].data['CWSTilt'],fs=macao_dict[T2][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**10,axis=0)

macao1_DM_psd = sig.welch( 0.7/(np.pi*2) * macao_dict[T1][1].data['DMZernike'],fs=macao_dict[T1][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**10,axis=0)

macao2_pist_psd = sig.welch( 0.7/(np.pi*2) * macao_dict[T2][1].data['EstPiston'],fs=macao_dict[T2][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**10,axis=0)
macao2_tip_psd = sig.welch( 0.7/(np.pi*2) * macao_dict[T2][1].data['CWSTip'],fs=macao_dict[T2][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**10,axis=0)
macao2_tilt_psd = sig.welch( 0.7/(np.pi*2) * macao_dict[T2][1].data['CWSTilt'],fs=macao_dict[T2][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**10,axis=0)


fig,ax = plt.subplots(2,1,figsize=(10,8))

#plt.loglog(*macao1_HO_psd,alpha=0.3, color='k')
ax[0].loglog(*macao1_HO_psd,alpha=0.3, color='k')
ax[0].loglog(*macao1_pist_psd,alpha=0.8, color='r')
ax[0].loglog(*macao1_tip_psd,alpha=0.8, color='y')
ax[0].loglog(*macao1_tilt_psd,alpha=0.8, color='orange')
ax[0].loglog(*grav_psd)#;plt.loglog(grav_psd[0]**(-8/3))
ax[0].set_xlim([grav_psd[0][0],grav_psd[0][-1]])

ax[1].loglog(*macao2_HO_psd,alpha=0.3, color='k')
ax[1].loglog(*macao2_pist_psd,alpha=0.8, color='r')
ax[1].loglog(*macao2_tip_psd,alpha=0.8, color='y')
ax[1].loglog(*macao2_tilt_psd,alpha=0.8, color='orange')
ax[1].loglog(*grav_psd)#;plt.loglog(grav_psd[0]**(-8/3))
ax[1].set_xlim([grav_psd[0][0],grav_psd[0][-1]])





#%% old other stuff 

#Could also plot timeseries of a_i and p !!! interesting to see whats happening 
plt.plot(ais, alpha = 0.5)
plt.plot(cross_coupling_dict['vlti']['apod']['Focus'], alpha = 0.5)

#########
#----- with circular pupil 
p_apod = {}
p_noapod = {}
coes ={}
for i in range(15):
    #name of zernike mode 
    z_mode = zernike.zern_name(i+1)
    # sampling our zernike coeficients 
    coes[z_mode] = np.random.normal(loc=0, scale=1.0, size=1000)
   
    # fiber coupled piston 
    p_apod[z_mode] = [ np.angle( calc_omega( field_pupil, np.exp(1j*ai*basis[i]), pup) ) for ai in coes[z_mode] ]
    # non-fiber coupled piston 
    p_noapod[z_mode] = [ np.sum( ai * basis[i] * pup) / np.sum(pup) for ai in coes[z_mode] ]
    
sigma_coes = np.array( [np.std(coes[z_mode]) for z_mode in coes.keys()] )
sigma_p_apod = np.array( [np.std(p_apod[z_mode]) for z_mode in p_apod.keys()] ) 
sigma_p_noapod = np.array( [np.std(p_noapod[z_mode]) for z_mode in p_noapod.keys()] )  
    
plt.plot(sigma_p_apod/sigma_coes,'.'); plt.plot(sigma_p_noapod/sigma_coes,'.')
plt.gca().set_xticklabels(p_apod.keys())
#########
#----- with VLTI pupil 
p_apod = {}
p_noapod = {}
coes ={}
for i in range(15):
    #name of zernike mode 
    z_mode = zernike.zern_name(i+1)
    # sampling our zernike coeficients 
    coes[z_mode] = np.random.normal(loc=0, scale=1.0, size=1000)
    # fiber coupled piston 
    p_apod[z_mode] = [ np.angle( calc_omega( field_pupil, np.exp(1j*ai*basis[i]), pup_vlt ) ) for ai in coes[z_mode] ]
    # non-fiber coupled piston 
    p_noapod[z_mode] = [ np.sum( ai * basis[i] * pup_vlt ) / np.sum(pup_vlt) for ai in coes[z_mode] ]
    
sigma_coes = np.array( [np.std(coes[z_mode]) for z_mode in coes.keys()] )
sigma_p_apod = np.array([np.std(p_apod[z_mode]) for z_mode in p_apod.keys()] ) 
sigma_p_noapod = np.array( [np.std(p_noapod[z_mode]) for z_mode in p_noapod.keys()] )  
    
plt.plot(sigma_p_apod/sigma_coes,'.'); plt.plot(sigma_p_noapod/sigma_coes,'.')


plt.plot( [np.std(p_apod['Focus']) ), np.std(p_noapod['Focus'])

plt.plot([calc_omega(basis[0],basis[i], pup) for i in range(15)],'.')
plt.plot([calc_omega(basis[0],basis[i], pup_vlt) for i in range(15)],'.')
plt.plot([calc_omega(basis[0],basis[i], pup_vlt*field_pupil) for i in range(15)],'.')


plt.plot([np.angle( calc_omega(pup,np.exp(1j* basis[i]), pup) /\
                   calc_omega(pup,np.exp(1j* 0*basis[0]), pup ) )for i in range(15)],'.',label='disk,non-apodized')
    
plt.plot([np.angle( calc_omega(pup_vlt,np.exp(1j* basis[i]), pup_vlt) /\
                   calc_omega(pup_vlt,np.exp(1j* 0*basis[0]), pup_vlt) ) for i in range(15)],'.',label='VLTI,non-apodized')
    
plt.plot([np.angle( calc_omega(pup_vlt,np.exp(1j* 0*basis[i]), field_pupil)/\
                   calc_omega(pup_vlt,np.exp(1j* 0*basis[0]), field_pupil)) for i in range(15)],'.',label='VLTI,apodized')

plt.legend()



plt.plot([ np.mean( basis[i][pup!=0] ) for i in range(15)],'.',label='disk,non-apodized')
    
plt.plot([ np.mean( basis[i][pup_vlt!=0] )  for i in range(15)],'.',label='VLTI,non-apodized')
    
plt.plot([ np.sum( basis[i][pup!=0]*field_pupil[pup!=0] ) / np.sum( field_pupil[pup!=0] ) for i in range(15)],'.',label='disk,apodized')
plt.plot([ np.sum( basis[i][pup_vlt!=0]*field_pupil[pup_vlt!=0] ) / np.sum( field_pupil[pup_vlt!=0] ) for i in range(15)],'.',label='VLTI,apodized')

plt.plot([np.angle( calc_omega(pup_vlt,np.exp(1j* 1e-2* basis[i]), field_pupil)/\
                   calc_omega(pup_vlt,np.exp(1j* 0*basis[0]), field_pupil)) for i in range(15)],'.',label='VLTI,apodized 2')



plt.legend()


# Check < Z_j | 1 >_P0 = 0 for j>0
omega_master_dict = {}
aa = 1e-3 #zernike scaling factor 
for weight, w_lab in zip([pup, field_pupil],['non-apo','apo']):
    omega_dict = {} 
    print('\n\n')
    for noll_idx in range(0,13):
        
        z_name = zernike.zern_name(noll_idx+1)
        p = pup #pup_vlt 
        filt = p!=1.3e4# np.min(p)
    
        X_phi = p[filt] * np.exp(1j * aa * basis[noll_idx][filt] )
        X_0 = p[filt] 
        Y =  p[filt] 
        W = weight[filt]
        
        omega_phi = calc_omega(X_phi,Y,W)
        
        omega_0 = calc_omega(X_0,Y,W)
        
        omega_dict[z_name] = np.angle(omega_phi/omega_0)
        print(noll_idx, np.angle(omega_phi))
        
    omega_master_dict[w_lab] = omega_dict

fig,ax =plt.subplots(1,1,figsize=(8,5))
omega_non_apo = abs(np.array(list(omega_master_dict['non-apo'].values())))
omega_apo = abs(np.array(list(omega_master_dict['apo'].values())))
#ax.plot( (omega_apo - omega_non_apo) / aa )
ax.plot(omega_non_apo ,'.',label='non-apodized'+r'($\Omega_{j,g}$)')
ax.plot(omega_apo ,'.',label='apodized'+r'($\Omega_{j,0}$)')
ax.set_ylabel(r'$\frac{\Omega_{\phi,j} }{ \Omega_{0,j} }$',fontsize=20)
ax.legend()



omega_master_dict = {}
for weight, w_lab in zip([pup_vlt, field_pupil],['non-apo','apo']):
    omega_dict = {} 
    for noll_idx in range(0,30):
        
        z_name = zernike.zern_name(noll_idx+1)
        p = pup_vlt 
        filt = p!=np.min(p)
    
        X_phi = p[filt] * np.exp(1j * aa * basis[noll_idx][filt] )
        X_0 = p[filt] 
        Y =  p[filt] 
        W = weight[filt]
        
        def calc_omega(X,Y,W):
            omega = np.nansum( X*np.conjugate(Y)*W ) / ( np.nansum( X*np.conjugate(X) * W ) * np.nansum( Y*np.conjugate(Y) * W ) )**0.5
            return(omega)
        
        omega_phi = calc_omega(X_phi,Y,W)
        
        omega_0 = calc_omega(X_0,Y,W)
        
        omega_dict[z_name] = np.angle(omega_phi/omega_0)
        
    omega_master_dict[w_lab] = omega_dict

fig,ax =plt.subplots(1,1,figsize=(8,5))
ax.plot(abs(np.array(list(omega_master_dict['non-apo'].values()))),'.',label='non-apodized'+r'($\Omega_{j,g}$)')
ax.plot(abs(np.array(list(omega_master_dict['apo'].values()))),'.',label='apodized'+r'($\Omega_{j,0}$)')
ax.set_ylabel(r'$\frac{\Omega_{\phi,j} }{ \Omega_{0,j} }$',fontsize=20)
ax.legend()


#check result Ruilier
np.angle(omega)
np.mean(basis[3][filt])


phi_i = []
for i in range(3,15):
    phi = basis[i]
    omega_phi = np.nansum( phi[pup!=0] * field_pupil[pup!=0]  ) 
    omega_0 = np.nansum( pup[pup!=0] * field_pupil[pup!=0]  ) 
    
    omega_phi = np.nansum( phi[pup!=0] * pup[pup!=0]  ) 
    omega_0 = np.nansum( pup[pup!=0] * pup[pup!=0]  ) 
    ratio = omega_phi/omega_0
    phi_i.append( np.angle(ratio ) )
    
    
np.nansum( phi[pup!=0] * field_pupil[pup!=0]  )  / np.nansum( field_pupil[pup!=0]  ) 
np.nansum( phi[pup!=0] * pup[pup!=0]  )  / np.nansum( pup[pup!=0]  )

"""fiber_field_ip = np.zeros(pup.shape)
for i,x_row in enumerate(X/wvl):
    #if np.mod(i,100)==0:
    #    print('{}% complete'.format(i/len(x)))
    fiber_field_ip[i,:] = [fields.gaussian_field(NA=NA,n_core=n_core, a=a, L=wvl,r=np.sqrt(x_row**2 + x_col**2)) for x_col in X]

import scipy.fftpack as sfft
fiber_field_pp = abs(sfft.fftshift(sfft.ifft2(fiber_field_ip)) )

#plt.imshow( abs(sfft.fftshift(sfft.ifft2(aperture.disc(size=D_pixel//8,dim=D_pixel))) )[100:150,100:150] )"""



#%%

"""
read this 
https://wiki.lbto.org/pub/AdaptiveOptics/RoundTwoTraining/AoTrnngDay01slides04may2020.pdf


also from Julien M. tutorial 
slope = Gradients[time_stamp]
mode  = S2M @ slope
volt  = M2V @ mode
res_turbulence = (IMF @ volt).reshape((240, 240))

# compute theoretical Strehl
wave   = 1600
err    = res_turbulence[res_turbulence != 0].std()
sigma  = 2*np.pi*err/wave
strehl = np.exp(-sigma**2)
print('Residual OPD:     {0:6.1f} nm'.format(err))
print('Estimated Strehl: {0:6.1f}%'.format(strehl*100))

# plot
plt.figure()
im = plt.imshow(res_turbulence, vmin=-300, vmax=300,origin='lower')
plt.title('Residual phase')
plt.colorbar(im, orientation='vertical', label='OPD [nm]')



CIAO has 60 actuators with 9x9 SH-WFS
"""


#%%
macao[0].header['HIERARCH ESO AOS RTC LOOP STREHL ']

print('np.mean(np.exp(-macao[1].data[WFE]))={}\nmacao[0].header[HIERARCH ESO AOS RTC LOOP STREHL]={}'.format(np.mean(np.exp(-macao[1].data['WFE'])),macao[0].header['HIERARCH ESO AOS RTC LOOP STREHL ']))

#ok so mean exp(-wfe) is telemetry published  strehl => wfe = variance (radians) of residual mode 
#now can we recontrcut wfe from CWSZernike, tip , tilt PSDs? 

HO = macao[1].data['CWSZernike']
tip = macao[1].data['CWSTip']
tilt = macao[1].data['CWSTilt']
pist = macao[1].data['EstPiston']

HO_psd = sig.welch(HO,fs=1050,axis=0)
tip_psd = sig.welch(tip,fs=1050,axis=0)
tilt_psd = sig.welch(tilt,fs=1050,axis=0)
#can we see correlation in WFE and CWSzernike 
n = 50000
plt.plot(abs(np.mean(HO,axis=1)[:n]**2) + abs(tip)[:n]**2 + abs(tilt)[:n]**2 , macao[1].data['WFE'][:n],'.',alpha=0.01)
plt.plot(macao[1].data['WFE'][:n],macao[1].data['WFE'][:n])
plt.xlabel('sum CWS^2')
plt.ylabel('telemetry wfe')

#Conclusion : psd are in units rad^2/Hz
"""
plt.plot((np.sum(HO[:,:1]**2,axis=1)[:n]**2 + tip[:n]**2 + tilt[:n]**2)**0.5 , macao[1].data['WFE'][:n],'.',alpha=0.01)
plt.plot(macao[1].data['WFE'][:n],macao[1].data['WFE'][:n])
plt.xlabel('sum CWS^2')
plt.ylabel('telemetry wfe')
"""
np.sum(HO_psd[1]*np.diff(HO_psd[0]))

print( np.mean(macao[1].data['WFE']) , ( np.sum(tilt_psd[1][:]*np.diff(tilt_psd[0])[1]) + \
 np.sum(HO_psd[1][10:]*np.diff(HO_psd[0])[1]) + \
     np.sum(tip_psd[1][:]*np.diff(tip_psd[0])[1]) )**0.5 )

    
#sigma_phi^2 (if these are the zernike residual coefficients ) 
#sigma_phi2 = np.sum(np.sum(aaa[1],axis=0)*np.diff(aaa[0])[0])
#print('strehl in header = {}, strehl calc from residual "coefficients"={}'.format(macao[0].header['HIERARCH ESO AOS RTC LOOP STREHL '], np.exp(-sigma_phi2)))

#what is in headers
print('HIERARCH ESO AOS RTC LOOP FREQUENCY = {}Hz'.format(macao[0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY']))
# what is median sample from data 
print('median sample rate from data = ',np.median(np.diff(macao[1].data['TimeStamp'][:,1])))
#matches expectatioons 

#########
# Check gravity data for observed around 6:06 with MACAO 
#########


os.chdir('/Volumes/vol1BCB/UT-gravity_july_run/25-07-21/gravity')

gravi = fits.open( '/Volumes/vol1BCB/UT-gravity_july_run/25-07-21/gravity/GRAVI.2021-07-26T06-05-44.677.fits' ) 

grav_dict = {}
macao_dict = {} 
macao_delay = {} #how much later was MACAO startec
for i in [1,2,3,4]: #UT telescopes 
    
    macao_file = glob.glob(f'/Volumes/vol1BCB/UT-gravity_july_run/25-07-21/UT{i}/AO_data/RTCData_2021-07-26_06-06-*.fits')
    macao_dict[i] = fits.open( macao_file[0] )
    
    macao_delay[i] = ( macao_dict[1][0].header['MJD-OBS'] - gravi[0].header['MJD-OBS'] ) * 24 * 60 * 60 # seconds 

    

print('AO system = {}'.format(gravi[0].header['HIERARCH ESO COU AO SYSTEM']))   
print('start time template = {}'.format(gravi[0].header['HIERARCH ESO TPL START']))
print('exptime for template = {}'.format(gravi[0].header['EXPTIME']))
print('gravity object = {} '.format(gravi[0].header['*OBJECT*'] ))#13.1 Vmag, 10.0 Kmag 
print('FT DIT = {}'.format(gravi[0].header['HIERARCH ESO DET3 SEQ1 DIT'])) # = 3ms

##########################
####### processing gravity data 
##########################

opdc = gravi['OPDC'].data
opdc_time = opdc['TIME']
opdc_opd = opdc['OPD']
opdc_kopd = opdc['KALMAN_OPD']
opdc_kpiezo = opdc['KALMAN_PIEZO']
opdc_steps = opdc['STEPS']
opdc_state = opdc['STATE']

dt = np.median(np.diff(opdc_time))*1e-6

## --- timestamps!!!
grav_time = datetime.datetime.strptime(gravi[0].header['HIERARCH ESO TPL START'] , '%Y-%m-%dT%H:%M:%S') \
        + pd.to_timedelta(opdc_time, 'us')


# Recover missing samples in OPDC
dit = np.median(np.diff(opdc_time))
missing = np.where(np.diff(opdc_time) > (1.5*dit))[0]
print("Fixed {0} missing steps in OPDC".format(len(missing)))
n = 0
for i in missing:
    n += 1
    opdc_time   = np.insert(opdc_time,   i+n, opdc_time[i+n-1]+dit, axis=0)
    opdc_opd    = np.insert(opdc_opd,    i+n, opdc_opd[i+n-1],      axis=0)
    opdc_kopd   = np.insert(opdc_kopd,   i+n, opdc_kopd[i+n-1],     axis=0)
    opdc_kpiezo = np.insert(opdc_kpiezo, i+n, opdc_kpiezo[i+n-1],   axis=0)
    opdc_steps  = np.insert(opdc_steps,  i+n, opdc_steps[i+n-1],    axis=0)

# Reconstruct phase modulation
opdc_mods = np.zeros([opdc_steps.shape[0], 4])
for iTel in range(4):
    opdc_mods[:, iTel] = (((opdc_steps >> (4*iTel))&15)*np.pi/8.0)
    
# Convert opdc_kpiezo, and opdc_mods from op to opd (i.e. telescope to baseline)
opdc_kpiezo = (M_matrix @ opdc_kpiezo.T).T
opdc_mods = (M_matrix @ opdc_mods.T).T


# Reconstruct residuals without phase modulations
phase_residuals = (opdc_opd - opdc_mods + np.pi) % (2.0*np.pi) - np.pi

# Reconstruct disturbances
phase_disturbances = opdc_kpiezo + (opdc_opd - (opdc_kopd-np.pi)) % (2.0*np.pi) + (opdc_kopd-np.pi)

# Convert phase in [rad] to length in [µm]
disturbances = phase_disturbances * 2.25 / (2.*np.pi)
residuals    = phase_residuals    * 2.25 / (2.*np.pi)

for idx,b in enumerate([f'{i}{j}' for i,j in base2telname]):
    grav_dict[b]={}
    grav_dict[b]['res'] = residuals[:,idx]
    grav_dict[b]['disturb'] = disturbances[:,idx]

##########################
####### processing macao data 
##########################

# --- timestamps!!!
macao_time = {}


for i in [1,2,3,4]:      
    
    t_tmp = np.arange(0,1e-6 * np.median(np.diff(macao_dict[i][1].data['TimeStamp'][:,1])) * len(macao_dict[i][1].data['TimeStamp']), np.median(np.diff(macao_dict[i][1].data['TimeStamp'][:,1]))*1e-6 )  # seconds from start 
    
    macao_time[i] =  grav_time[0] + pd.to_timedelta(macao_delay[i] ,unit='s') + pd.to_timedelta(t_tmp , unit='s')



##########################
####### plotting ts
##########################


b = '32'

T1 = int(b[0])
T2 = int(b[1])



t0 = np.min( [np.min(macao_time[T1]), np.min(macao_time[T2])] )
t1 = np.max( [np.max(macao_time[T1]), np.max(macao_time[T2])] )
www=5 #rollingwindow
time_filt = (grav_time <= t1) & (grav_time >= t0)

grav_time_wind = grav_time[time_filt]
grav_dict_wind = grav_dict[b]['res'][time_filt]

for aaa in range(32):
    plt.figure()
    i1,i2=(aaa*1000),(aaa+1)*1000
    
    time_filt2 = (grav_time_wind <= macao_time[T1][i2]) & (grav_time_wind >= macao_time[T1][i1])
    
    plt.plot(grav_time_wind[time_filt2] , 10*grav_dict_wind[time_filt2], '-',color='r')
    #plt.plot(macao_time[T1][i1:i2], 60+10*pd.Series( macao_dict[T1][1].data['CWSTilt'][i1:i2] ).rolling(1000).median(), color='b')
    plt.plot(macao_time[T1][i1:i2], 25+ pd.Series(macao_dict[T1][1].data['CWSTip'][i1:i2]).rolling(www).mean(), color='b')
    plt.plot(macao_time[T1][i1:i2], 50+ pd.Series(macao_dict[T1][1].data['CWSTip'][i1:i2]).rolling(www).mean(), color='b')
    
    plt.plot(macao_time[T2][i1:i2], -25- pd.Series(macao_dict[T2][1].data['CWSTip'][i1:i2]).rolling(www).mean(), color='b')
    plt.plot(macao_time[T2][i1:i2], -50- pd.Series(macao_dict[T2][1].data['CWSTip'][i1:i2]).rolling(www).mean(), color='b')
    for i in range(8):
        plt.plot(macao_time[T1][i1:i2], 75 + 25*i + pd.Series(macao_dict[T1][1].data['CWSZernike'][i1:i2,i]).rolling(www).mean(), color='k')
        plt.plot(macao_time[T2][i1:i2], -75 - 25*i + pd.Series(macao_dict[T2][1].data['CWSZernike'][i1:i2,i]).rolling(www).mean(), color='k')
        #plt.plot(macao_time[T1][i1:i2], 60+macao_dict[T1][1].data['CWSTilt'][i1:i2], color='b')
        #plt.plot(macao_time[T2][i1:i2], 30+macao_dict[T2][1].data['CWSZernike'][i1:i2,i], color='k')
    

##########################
####### plotting PSD
##########################


b = '21'

T1 = int(b[0])
T2 = int(b[1])

grav_dict_wind = grav_dict[b]['disturb'][time_filt]

grav_psd = sig.welch(grav_dict[b]['disturb'],fs=1/gravi[0].header['HIERARCH ESO DET3 SEQ1 DIT'], nperseg=2**10)

#MACAO central wvl = 0.7um . convert to um 
macao1_HO_psd = sig.welch( 0.7/(np.pi*2) * macao_dict[T1][1].data['CWSZernike'],fs=macao_dict[T1][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**10,axis=0)
macao2_HO_psd = sig.welch( 0.7/(np.pi*2) * macao_dict[T2][1].data['CWSZernike'],fs=macao_dict[T2][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**10,axis=0)

macao1_pist_psd = sig.welch( 0.7/(np.pi*2) * macao_dict[T1][1].data['EstPiston'],fs=macao_dict[T2][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**10,axis=0)
macao1_tip_psd = sig.welch( 0.7/(np.pi*2) * macao_dict[T1][1].data['CWSTip'],fs=macao_dict[T2][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**10,axis=0)
macao1_tilt_psd = sig.welch( 0.7/(np.pi*2) * macao_dict[T1][1].data['CWSTilt'],fs=macao_dict[T2][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**10,axis=0)

macao1_DM_psd = sig.welch( 0.7/(np.pi*2) * macao_dict[T1][1].data['DMZernike'],fs=macao_dict[T1][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**10,axis=0)

macao2_pist_psd = sig.welch( 0.7/(np.pi*2) * macao_dict[T2][1].data['EstPiston'],fs=macao_dict[T2][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**10,axis=0)
macao2_tip_psd = sig.welch( 0.7/(np.pi*2) * macao_dict[T2][1].data['CWSTip'],fs=macao_dict[T2][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**10,axis=0)
macao2_tilt_psd = sig.welch( 0.7/(np.pi*2) * macao_dict[T2][1].data['CWSTilt'],fs=macao_dict[T2][0].header['HIERARCH ESO AOS RTC LOOP FREQUENCY'], nperseg=2**10,axis=0)


fig,ax = plt.subplots(2,1,figsize=(10,8))

#plt.loglog(*macao1_HO_psd,alpha=0.3, color='k')
ax[0].loglog(*macao1_HO_psd,alpha=0.3, color='k')
ax[0].loglog(*macao1_pist_psd,alpha=0.8, color='r')
ax[0].loglog(*macao1_tip_psd,alpha=0.8, color='y')
ax[0].loglog(*macao1_tilt_psd,alpha=0.8, color='orange')
ax[0].loglog(*grav_psd)#;plt.loglog(grav_psd[0]**(-8/3))
ax[0].set_xlim([grav_psd[0][0],grav_psd[0][-1]])

ax[1].loglog(*macao2_HO_psd,alpha=0.3, color='k')
ax[1].loglog(*macao2_pist_psd,alpha=0.8, color='r')
ax[1].loglog(*macao2_tip_psd,alpha=0.8, color='y')
ax[1].loglog(*macao2_tilt_psd,alpha=0.8, color='orange')
ax[1].loglog(*grav_psd)#;plt.loglog(grav_psd[0]**(-8/3))
ax[1].set_xlim([grav_psd[0][0],grav_psd[0][-1]])



#%%

os.chdir('/Volumes/vol1BCB/UT-gravity_july_run/25-07-21/UT1/AO_data/DATA_EXPO-000004')

#logger_files = glob.glob('*DATA_LOGGER*')

#LOOP[1].data['Gradients'].shape = (60000, 136)
HO_IM = fits.open('RecnOptimiser.HO_IM_0001.fits') #HO_IM[0].data.shape = (136, 60)
M2V = fits.open('RecnOptimiser.M2V_0001.fits')  #M2V[0].data.shape = (62, 45)
S2M =  fits.open('RecnOptimiser.M2V_0001.fits') #S2M[0].data.shape = (62, 45)
HOCtr = fits.open('HOCtr.ACT_POS_REF_MAP_0001.fits') #HOCtr[0].data.shape = (1, 60)
REFSLP = fits.open('Acq.DET1.REFSLP_0001.fits') #REFSLP[0].data.shape = (1, 136)
AVC =  fits.open('CIAO_AVC_0001.fits')
LOOP = fits.open('CIAO_LOOP_0001.fits')
GAINS = fits.open('RecnOptimiser.APPLIED_GAINS_0001.fits')
CM = fits.open('Recn.REC1.CM_0001.fits')  #CM[0].data.shape=(62, 136)
#PIXELS = fits.open('CIAO_LOOP_0001.fits')

HO_IM[0].data @ LOOP[1].data['Gradients']

#frequency 
f_ao = 1/np.median(np.diff(LOOP[1].data['useconds']))*1e6 #Hz

sig.welch(LOOP[1].data['HODM_Positions'] , fs=f_ao, nperseg=2**10, axis=0)
plt.plot(np.std(LOOP[1].data['HODM_Positions'][:,:],axis=0),'.')

#LOOP[1].data['Gradients'].shape
plt.plot(np.median(LOOP[1].data['HODM_Positions'][:,:],axis=0),'.')

LOOP[1].data['Gradients'][:-1,1], 
LOOP[1].data['HODM_Positions'][1:,i]

S2M[0].data * LOOP[1].data['Gradients'][0,:]