#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 17:04:37 2021

@author: bcourtne
"""


import os
import numpy as np
import pandas as pd
import opticstools
import aotools
import scipy
import scipy.fftpack as sfft
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.signal as sig 
import matplotlib.gridspec as gridspec
os.chdir('/Users/bcourtne/Documents/Hi5/vibration_analysis/utilites')

import fiber_fields as fields
import NAOMI_corrected_wavefront_simulation as AO_sim 
import atmosphere_piston


#hardcoded parameters 
wvl, f,D = 2.2e-6, 20, 8 #wavelength, focal length, telescope radius (m)
rad2arcsec = 180/np.pi*60*60 #radians to arcsec scaling factor
N = 2**11 #number of pixels across square pupil grid
D_pixel = int(N/(2**3)) #number of pixels across telescope diameter
a = 0.2e-6  #fiber core diameter
NA = 0.11 #numerical apperature 
N_phot = 1e6 #number of photons at wavelength / m^2

#setting up image and pupil plane grids 
dfX = D/D_pixel # differential element in pupil plane coordinates (telescope diameter / number of pixels within diameter)
fX = dfX*np.arange( -N/2.,N/2.) #pupil plane coordinates
fXX,fYY = np.meshgrid(fX,fX) #grid for pupil plane

x = wvl * f * np.arange( -N/2.,N/2.)/(N*dfX) #image plane coordinates adjusted for focal length and wavelength
dx = np.diff(x)[1] # differential element in image plane


pupil_m1 = aotools.functions.pupil.circle(radius=D_pixel/2, size=D_pixel, circle_centre=(0, 0), origin='middle')
#m2 diameter = 1 m
pupil_m2 =  aotools.functions.pupil.circle(radius=D_pixel/16, size=D_pixel, circle_centre=(0, 0), origin='middle')


#consider with and without central obscuration
pupil = pupil_m1
pupil_obs = pupil_m1 - pupil_m2 

#create masks for calculating statistics in the pupil (0->nan)
pupil_mask = pupil.copy() #need to mask outside pupil by nan for calculating averages 
pupil_mask[pupil==0] = np.nan

pupil_obs_mask = pupil_obs.copy()
pupil_obs_mask[pupil_obs_mask==0] = np.nan

#%% Functions 
"""
Function 1. grid & atm parameters -> list of phase screens
function 2. list of phase screens and pupil definitions -> AO corrected phase screens for each scenario
function 3. fiber properties & AO corrected phase screens -> fiber coupled phase 

"""

# --- function 1 
def init_phase_screens(r0,L0,wvl,V, dt, D_pixels,pixel_scale,iterations):
    """
    

    Parameters
    ----------
    r0 : TYPE
        DESCRIPTION.
    L0 : TYPE
        DESCRIPTION.
    wvl : TYPE
        DESCRIPTION.
    V : TYPE
        DESCRIPTION.
    dt : TYPE
        DESCRIPTION.
    D_pixels : TYPE
        DESCRIPTION.
    pixel_scale : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    col2move = int(V * dt / dfX) #how many columns screen moves by per iteration (defined by windspeed, pixel scale, and sampling rate)
    #infinte screen to be iterated through 
    master_phscreens = aotools.turbulence.infinitephasescreen.PhaseScreenVonKarman(nx_size=D_pixel,pixel_scale=dfX,r0=r0,L0=L0,n_columns=col2move,random_seed = 1)
    #doing our iteration 
    screens = []
    for i in range(int(iterations)):
        #each iteration moves scrren by col2move * dfX
        screens.append(master_phscreens.add_row())
               
    return(screens)


#test 
#aaa=init_phase_screens(r0,L0,wvl,V_w, dt, D_pixels=D_pixel,pixel_scale=dfX,iterations=2e3)

# --- function 2
def AO_correction(screens, dt, V, dfx, no_modes_corrected, D_pixel, pupil_definitions_dict):
    """
    

    Parameters
    ----------
    screens : TYPE
        DESCRIPTION.
    dt : TYPE
        DESCRIPTION.
        
    no_modes_corrected : TYPE
        DESCRIPTION.
    D_pixel : TYPE
        DESCRIPTION.
        
    frame_lag : TYPE
        DESCRIPTION.
        
    pupil_definitions_dict : TYPE
        DESCRIPTION.
        dictionary with 
            keys: pupil combo label
            values: tuple pupil combo masks
            
            {'donut-circle':(donut_mask, circle_mask)}
            
            1st entry is the real pupil seen by the WFS 
            2nd entry is the pupil used for the piston removal algorithm

    Returns
    -------
    None.

    """
    #how lagged is the sensed screen from the current screen
    framelag = int(V * dt / dfX)
    N_modes_considered = no_modes_corrected + 1
    #no_modes_corrected = N_modes_considered + 1
    
    #get a list of the zernike modes across our pupil so dont have to recalculate every iteration
    zernikes = [aotools.functions.zernike.zernike_noll(m+2,int(D_pixel)) for m in range(N_modes_considered)]
    #the zernike correction coefficient (1=full correction, 0 = no correction)
    f_j = list(np.ones(no_modes_corrected)) + list(np.zeros(N_modes_considered-no_modes_corrected))


    #temporal_lag = framelag * dt
    
    """
    there is a real pupil that the wave passes 
    there is a percieved pupil that the DM applies a 'piston free' correction
     
    """
    
    #the list is tuple of the masks [(real pupil, piston free pupil) ... ]
    pupil_combination_list = [pp_tuple for pp_tuple in pupil_definitions_dict.values()]
    pupil_combination_labels = [pp_label for pp_label in pupil_definitions_dict.keys()] #['pp','po','op','oo']
    
    
    AO_correction_dict = dict({pupil_def : dict({'pupil_masks':pupil_combination_list, 'current_screen':[], 'lagged_screen':[], 'DM_screen':[], 'residual_screen':[]}) for pupil_def in pupil_combination_labels})
    for i in range(framelag,len(screens),1):
        #check progress
        if np.mod(i,500)==0:
            print('{}% complete'.format(i/len(screens)))
            
        for pm,pm_key in zip(pupil_combination_list, AO_correction_dict.keys()): #for each pupil mask we're considering 
            
            #########
            # use the "real" telescope pupil (index 0) for the phase screens          
            #current phase across telescope pupil
            current_screen = pm[0] * screens[i]
            #the WFS measured phase accounting for AO lag 
            sensed_screen = pm[0] * screens[i-framelag]
            
            #the list to hold current phase screen zernike coeficients in the pupil
            a_j = []
            #init the superposition of zernike modes to subtract from phase screen
            #the DM shape is applied across the pupil so just use standard definition here 
            DM_shape = pupil_mask * np.zeros([int(D_pixel),int(D_pixel)])
            
            for m in range(N_modes_considered):
                
                if f_j[m] != 0 : #if this mode is one that must be corrected by AO system
                    
                    #create zernike mode ### WHAT MASK TO APPLY HERE!??!
                    z_i = pupil_mask * zernikes[m] #need i+2 since we dont consider piston (index = 1)
                    #normalizing (they should be normalized.. but just incase)
                    norm = dfX**2*np.nansum(z_i)
                    if norm > 1e-10: #if not well enough normalized.. ill do it here
                        z_i *= 1/(dfX**2*np.nansum(z_i))
                        
                        
                    #print(dfX**2 * np.nansum(z_i))
                    #find the phase screen coefficient for this mode
                    a_i = dfX**2*np.nansum(sensed_screen * z_i)
                    #append the coefficient to the a_j list
                    a_j.append(a_i)
                    #subtract the amount of AO correction
                    DM_shape = DM_shape + f_j[m]*a_j[m]*z_i
                    
                    """if np.mod(m,10)==0:
                        plt.figure()
                        plt.imshow(DM_shape)"""
        
                    
            ###### -------- PISTON FREE DEFINITIONS ---------- #########
            #use the 'piston free pupil' for the mask on the DM (index 1) 
            AO_piston_basis = pm[1] * 1/np.sqrt(np.nansum(pm[1]**2 * dfX**2 )) # <P|P>=1
            
            DM_shape = DM_shape - np.nansum(AO_piston_basis * DM_shape *dfX**2) * AO_piston_basis
            
            #DM_shape = DM_shape - np.nansum(pm[1] * DM_shape * dfX**2) / np.nansum(pm[1] * dfX**2)
            
            ###### ------------------------------------------ #########    
            #print(np.nanmean(DM_pf0))
            #print(dfX**2*np.nansum(DM_pf0))
        
            #Applying AO correction
            AO_correction_dict[pm_key]['current_screen'].append( current_screen )
            AO_correction_dict[pm_key]['lagged_screen'].append( sensed_screen ) #what the AO sees
            AO_correction_dict[pm_key]['DM_screen'].append(DM_shape)
            AO_correction_dict[pm_key]['residual_screen'].append( current_screen - DM_shape)
            #DM_screen.append( DM_pf0 )
            #residual_screen.append( current_screen - DM_pf0 )
            
    return(AO_correction_dict)



def crop_big_array(big_array, small_array):
    #crop big array to be centered in small array  (assuming both are square!!!)

    xxx = small_array.shape[0]
    yyy = big_array.shape[0]
    return(big_array[(yyy-xxx)//2:(yyy+xxx)//2 , (yyy-xxx)//2:(yyy+xxx)//2 ] )

#%%

#test 

#go to pupil plane (its real in the focal plane so reasonable to enforce that its real in pupil also (take abs))
fiber_field_pp = abs(sfft.fftshift(sfft.ifft2(fiber_field_ip)) )
#normalization from (Perrin, Woillez 2019) np.nansum(abs(fiber_field_pp)**2 * dfX**2) = 1 
fiber_field_pp *= 1/np.sqrt(np.nansum(abs(fiber_field_pp)**2 * dfX**2 ))

# remember its fiber field, and then phase screens!!! 
test_pupil_mask = np.ones(fiber_field_pp.shape)
#test_pupil_mask *= 1/(np.nansum(test_pupil_mask))# * dfX**2)
test_object_field = test_pupil_mask * np.exp(-1j*np.angle(fiber_field_pp))

#test that we get 1 for perfect overlap without telescope pupil
eta = fiber_coupling_eta(test_pupil_mask, fiber_field_pp, [np.angle(fiber_field_pp)] , dfX)
print('for perfect overlap eta = 1.. calculated here: eta = {}'.format(eta))
#for fiber parameters and clear telescope pupil what is maximum efficiency 
eta = fiber_coupling_eta(pupil_mask, fiber_field_pp, [np.angle(fiber_field_pp)] , dfX)
print('for fiber parameters and clear telescope pupil maximum efficiency eta = {}'.format(eta))
#for fiber parameters and clear telescope pupil with central obscuration (VLT) what is maximum efficiency 
eta = fiber_coupling_eta(pupil_obs_mask, fiber_field_pp, [np.angle(fiber_field_pp)] , dfX)
print('for fiber parameters and clear telescope pupil with central obscuration (VLT) maximum efficiency eta = {}'.format(eta))


#np.exp(1j * np.angle(fiber_field_pp))

#with perfect coupling coupled field power = before coupling power. i.e.
#np.nansum(fiber_field_pp**2 * dfX**2) == np.nansum((eta * fiber_field_pp)**2 * dfX**2) => eta = 1
#print(eta, np.nansum(test_pupil_mask * fiber_field_pp * dfX**2))
#print( abs( np.nansum(fiber_field_pp**2 * dfX**2)), abs( np.nansum((eta * fiber_field_pp)**2 * dfX**2) ) )

#print( abs(fiber_coupling(pupil_mask,fiber_field_pp, [fiber_field_pp] , dfX)  ) )

#%% ========== atmopsheric piston vs zernike PSDs ================

#basic parameter setup
col2move = 2
print('each iteration moves screen by col2move * dfX = {}m'.format(col2move * dfX))
#master screen
r0 = 0.1 #m (at 500nm)
L0 = 25 #m
wvl = 2.2e-6 #m
seeing = (0.5/2.2)**(6/5) * 0.98*wvl/r0 * 3600 * 180/np.pi #arcsec (at 500nm)
dt = 1e-3 
V_w = col2move * dfX / dt 
screens = init_phase_screens(r0,L0,wvl,V_w, dt, D_pixel,pixel_scale=dfX,iterations=1e3)

screen_piston = [np.nanmean(screens[i]*pupil_mask) for i in range(len(screens))]
screen_obs_piston = [np.nanmean(screens[i]*pupil_obs_mask) for i in range(len(screens))]

#consider 1st 50 zernikes (include piston)
zernikes = [aotools.functions.zernike.zernike_noll(m+1,int(D_pixel)) for m in range(50)]

screen_zernikes = [[dfX**2 * np.nansum(screens[i]*pupil_obs_mask * zernikes[j]) for i in range(len(screens))] for j in range(len(zernikes))]
zernike_psds = [sig.welch(p, fs=1/dt , nperseg=2**9, window='hann',detrend='linear') for p in screen_zernikes]


for i, psd in enumerate(zernike_psds):
    if i==0:        
        plt.loglog(*psd,color='r',alpha=1,label='piston')
    if i==1:        
        plt.loglog(*psd,color='k',alpha=0.1,label=r'$Z_i$')
    else:
        plt.loglog(*psd,color='k',alpha=0.1)
plt.loglog(psd[0],1e8*psd[0]**(-17/3),linestyle='--',label=r'$f^{-17/3}$')
plt.gca().tick_params(labelsize=15)
plt.xlabel('Frequency (Hz)',fontsize=20)
plt.ylabel(r'PSD $(rad^2/Hz)$',fontsize=20)
plt.legend(fontsize=13)
plt.text(3,1e-8,r'$\lambda$={}$\mu m$, D={}m, $r_0$={}m, $L_0$=25m, V={}m/s'.format(1e6*wvl, D,r0,L0,V_w))
#define the time stamps, note that sampling rate effectively determines wind speed parameter 


#get mean zernike coefficients with / without obscured pupil mask 
#zernikes = [aotools.functions.zernike.zernike_noll(m+1,int(D_pixel)) for m in range(50)]
#screen_zernikes_1 = [[dfX**2 * np.nansum(screens[i]*pupil_mask * zernikes[j]) for i in range(len(screens))] for j in range(50)]
#screen_zernikes_2 = [[dfX**2 * np.nansum(screens[i]*pupil_obs_mask * zernikes[j]) for i in range(len(screens))] for j in range(50)]

#JUst want to project zernikes onto normalized centrally obscured pupil
zernikes = [aotools.functions.zernike.zernike_noll(m+2,int(D_pixel)) for m in range(200)]
#zernikes = [zzz*1/np.nansum(dfX**2*zzz**2) for zzz in zernikes]
pupil_obs_mask_norm = 1/np.nansum(pupil_obs_mask**2*dfX**2) * pupil_obs_mask #<P|P>1
zerike2pup_obs_projection = [dfX**2 * np.nansum(pupil_obs_mask_norm * zzz) for zzz in zernikes]

from mpl_toolkits.axes_grid.inset_locator import inset_axes
plt.figure()
#plt.plot(np.mean(screen_zernikes_1,axis=1),'.',label='without obscuration')
plt.plot(zerike2pup_obs_projection,'.',label='coefficient')
plt.plot(np.cumsum(zerike2pup_obs_projection),label='cummulative')
plt.legend(bbox_to_anchor=(1,1),fontsize=12)
plt.gca().tick_params(labelsize=15)
plt.xlabel('Zernike Noll index (i)',fontsize=20)
plt.ylabel(r'$\langle Z_i|P \rangle$',fontsize=20)
# this is an inset axes over the main axes
ins = plt.gca().inset_axes([1.05,0.2,0.2,0.5],aspect=1)
ins.imshow(pupil_obs_mask_norm)
ins.text(1,1,r'$|P\rangle$',fontsize=15)
ins.axis('off')
#inset_axes = inset_axes(plt.gca()), 
#                    width=1, #"50%", # width = 30% of parent_bbox
#                    height=1.0, # height : 1 inch
#                    loc=2)



#How pure zernikes with central obscuration cross couple to piston 

#initialize parameters and fiber field array in image plane
a = 1e-6 #fiber core diameter
NA = .45 #numerical apperature 
n_core=2 #refractive index of core
fiber_field_ip = np.nan*np.ones([len(x),len(x)])
#calculate it
for i,x_row in enumerate(x):
    #if np.mod(i,100)==0:
    #    print('{}% complete'.format(i/len(x)))
    fiber_field_ip[i,:] = [fields.gaussian_field(NA=NA,n_core=2.7,a=a,L=wvl,r=np.sqrt(x_row**2 + x_col**2)) for x_col in x]

#go to pupil plane (its real in the focal plane so reasonable to enforce that its real in pupil also (take abs))
fiber_field_pp = abs(sfft.fftshift(sfft.ifft2(fiber_field_ip)) )
#normalization from (Perrin, Woillez 2019) np.nansum(abs(fiber_field_pp)**2 * dfX**2) = 1 
fiber_field_pp *= 1/np.sqrt(np.nansum(abs(fiber_field_pp)**2 * dfX**2 ))

#normalize pupil mask <P|P> = 1 
pupil_mask_norm = pupil_mask/np.nansum(pupil_mask * pupil_mask)
#normalize pupil mask <P|P> = 1 
pupil_obs_mask_norm = pupil_obs_mask/np.nansum(pupil_obs_mask * pupil_obs_mask )
#includes piston, and apply nan mask to zernikes  
zernikes = [pupil_mask * aotools.functions.zernike.zernike_noll(m+2,int(D_pixel)) for m in range(200)]

apodized_zernikes = [pupil_mask * crop_big_array(fiber_field_pp,pupil_mask) * aotools.functions.zernike.zernike_noll(m+2,int(D_pixel)) for m in range(200)]

plt.plot(range(2,len(zernikes)+2,1),[np.nansum(pupil_mask_norm * zzz) for zzz in zernikes],label='circular pupil')
plt.plot(range(2,len(zernikes)+2,1),[np.nansum(pupil_obs_mask_norm * zzz) for zzz in zernikes],label='donut pupil')
#plt.plot(range(2,len(zernikes)+2,1),[np.nansum(pupil_mask_norm * zzz) for zzz in apodized_zernikes],label='apodized circular pupil')
#plt.plot(range(2,len(zernikes)+2,1),[np.nansum(pupil_obs_mask_norm * zzz) for zzz in apodized_zernikes],label='apodized donut pupil')

plt.legend()
plt.xlabel('Noll index')
plt.ylabel('piston (normalized)')

#show in overlap integral that piston term comes out as constant...d

#then take some non-zero coefficient of zernike coupling to piston and plot scaled zernike PSD vs piston 

    
#%% Simple circle donut cases - what does piston PSD look like for incorrect AO def


#basic parameter setup
col2move = 2
print('each iteration moves screen by col2move * dfX = {}m'.format(col2move * dfX))
#master screen
r0 = 0.1 #m (at 500nm)
L0 = 25 #m
wvl = 2.2e-6 #m
seeing = (0.5/2.2)**(6/5) * 0.98*wvl/r0 * 3600 * 180/np.pi #arcsec (at 500nm)
dt = 1e-3 
V_w = col2move * dfX / dt 

## setting up different obscurations to test piston with incorrect pupil def in removal algorithm

#[(real pupil, pupil def)] - keep pupil def as open pupil and the real pupil with different sizes
"""
experiment:
    have a real (1m radius) obscuration, and then play with the
    AO piston free pupil def with different central obscurations (including open pupil) 
    #1 of the the piston free pupil definitions should match the real - 
        we expect this to have the lowest OPD after AO correction
"""
real_pupil_mask = pupil_obs_mask.copy()
real_piston_basis = real_pupil_mask * 1/np.sqrt(np.nansum( real_pupil_mask**2 * dfX**2 ) )
#---- for simple circle - donut test 
pupil_tuples = [(pupil_obs_mask.copy(),  pupil_obs_mask.copy()), (pupil_obs_mask.copy(), pupil_mask.copy())]
labels = ['donut','circle'] 

for pp in pupil_tuples:
    pp[1][pp[1]==0]=np.nan #need to use nan mask
pupil_definition_dict = {x:y for x,y in zip(labels, pupil_tuples)}

#pupil_definition_dict = dict({'dd':(pupil_obs_mask, pupil_obs_mask),'dc':(pupil_obs_mask,pupil_mask)})
screens = init_phase_screens(r0,L0,wvl,V_w, dt, D_pixel,pixel_scale=dfX,iterations=1e3)
# time stamps for the screens
t = np.arange(0, dt * len(screens), dt)
no_modes_corrected = 50
AO_correction_dict = AO_correction(screens, dt, V_w, dfX, no_modes_corrected, D_pixel, pupil_definition_dict )


#get piston timeseries after DM under different pupil definitions  (takes 1 minute)
#pistons_ts_after_A0 = [[ np.nanmean(p) for p in AO_correction_dict[x]['residual_screen'] ] for x in labels]
pistons_ts_after_A0 = [[ np.nansum(real_piston_basis * p * dfX**2) for p in AO_correction_dict[x]['residual_screen'] ] for x in labels]

#get piston timeseries before DM
#piston_ts_before_A0 = [ np.nanmean(p) for p in AO_correction_dict['donut']['lagged_screen'] ]
piston_ts_before_A0 = [ np.nansum(real_piston_basis * p * dfX**2) for p in AO_correction_dict['donut']['lagged_screen'] ]

#get piston psd's from each timeseries
pistons_psd_after_AO = [sig.welch(p, fs=1/dt , nperseg=2**9, window='hann',detrend='linear') for p in pistons_ts_after_A0]
piston_psd_before_A0 = sig.welch(piston_ts_before_A0 , fs=1/dt , nperseg=2**9, window='hann',detrend='linear')
#plotting



import matplotlib.gridspec as gridspec
fig = plt.figure(tight_layout=True,figsize=(16,6))
gs = gridspec.GridSpec(6, 16)

ax0 = fig.add_subplot(gs[:, :8])
ax1 = fig.add_subplot(gs[0:3, 9:12])
#ax1.axis('off')
ax2 = fig.add_subplot(gs[0:3, 12:15])
#ax2.axis('off')
ax3 = fig.add_subplot(gs[3:6, 9:12])
#ax3.axis('off')
ax4 = fig.add_subplot(gs[3:6, 12:15])
#ax4.axis('off')

labels = ['After AO correction, correct piston free basis', 'After AO correction, in-correct piston free basis'  ]
#plt.figure(figsize=(15,10))
ax0.loglog(*piston_psd_before_A0,label='before AO correction',color='k',lw=3,linestyle='--')
for i, psd in enumerate(pistons_psd_after_AO):
    ax0.loglog(*psd,label='{}'.format( labels[i] ))
    print(np.sum(psd[1])*np.diff(psd[0])[1])
    #plt.loglog(psd[0],np.cumsum(psd[1][::-1]*np.diff(psd[0])[1])[::-1])
ax0.legend(loc='lower left',fontsize=15)
ax0.tick_params(labelsize=22)
ax0.set_xlabel('Frequency (Hz)',fontsize=22)
ax0.set_ylabel(r'Piston PSD $(rad^2/Hz)$',fontsize=22)
ax0.set_title(r'D={}m, $r_0$={}m, $L_0$=25m, V={}m/s'.format(D,r0,L0,V_w),fontsize=20)

ax1.imshow(pupil_tuples[0][0])
ax2.imshow(pupil_tuples[0][1])
ax3.imshow(pupil_tuples[1][0])
ax4.imshow(pupil_tuples[1][1])

ax1.set_title('True Piston\nBasis',fontsize=15,fontweight="bold")
ax2.set_title('AO Piston\nBasis',fontsize=15,fontweight="bold")
ax1.set_ylabel('correct piston\nfree basis',fontsize=15,fontweight="bold")
ax3.set_ylabel('in-correct piston\nfree basis',fontsize=15,fontweight="bold")
for axx in [ax1, ax2,ax3,ax4]:
    axx.set_xticks([])
    axx.set_yticks([])
    
#plt.savefig('/Users/bcourtne/Documents/ANU-PhD/1st_phd_pres_figures/psds_piston_free_basis_def_central_obs.png')



#%% piston PSD vs correct / incorrect pupil definitions with varying obscuration size
# -- more complicated case of variying central obscurations 

#basic parameter setup
col2move = 2
print('each iteration moves screen by col2move * dfX = {}m'.format(col2move * dfX))
#master screen
r0 = 0.1 #m (at 500nm)
L0 = 25 #m
wvl = 2.2e-6 #m
seeing = (0.5/2.2)**(6/5) * 0.98*wvl/r0 * 3600 * 180/np.pi #arcsec (at 500nm)
dt = 1e-3 
V_w = col2move * dfX / dt 

## setting up different obscurations to test piston with incorrect pupil def in removal algorithm

#[(real pupil, pupil def)] - keep pupil def as open pupil and the real pupil with different sizes
"""
experiment:
    have a real (1m radius) obscuration, and then play with the
    AO piston free pupil def with different central obscurations (including open pupil) 
    #1 of the the piston free pupil definitions should match the real - 
        we expect this to have the lowest OPD after AO correction
"""
real_pupil_mask = pupil_obs_mask.copy()
#--- for when testing different central obscurations in AO piston definition:
M2_ratios = [4,8,16,32,64]
labels = [f'D/{x}' for x in M2_ratios]
pupil_tuples = [(real_pupil_mask, pupil_m1 - aotools.functions.pupil.circle(radius=int(D_pixel/i), size=D_pixel, circle_centre=(0, 0), origin='middle')) for i in M2_ratios] 
#---- for simple circle - donut test 
#pupil_tuples = [(pupil_obs_mask.copy(),  pupil_obs_mask.copy()), (pupil_obs_mask.copy(), pupil_mask.copy())]
#labels = ['donut','circle'] 

for pp in pupil_tuples:
    pp[1][pp[1]==0]=np.nan #need to use nan mask
pupil_definition_dict = {x:y for x,y in zip(labels, pupil_tuples)}

#pupil_definition_dict = dict({'dd':(pupil_obs_mask, pupil_obs_mask),'dc':(pupil_obs_mask,pupil_mask)})
screens = init_phase_screens(r0,L0,wvl,V_w, dt, D_pixel,pixel_scale=dfX,iterations=1e3)
# time stamps for the screens
t = np.arange(0, dt * len(screens), dt)
no_modes_corrected = 50
AO_correction_dict = AO_correction(screens, dt, V_w, dfX, no_modes_corrected, D_pixel, pupil_definition_dict )



#get piston timeseries after DM under different pupil definitions  (takes 1 minute)
pistons_ts_after_A0 = [[ np.nanmean(p) for p in AO_correction_dict[x]['residual_screen'] ] for x in labels]
#get piston timeseries before DM
piston_ts_before_A0 = [ np.nanmean(p) for p in AO_correction_dict['D/16']['lagged_screen'] ]
#get piston psd's from each timeseries
pistons_psd_after_AO = [sig.welch(p, fs=1/dt , nperseg=2**9, window='hann',detrend='linear') for p in pistons_ts_after_A0]
piston_psd_before_A0 = sig.welch(piston_ts_before_A0 , fs=1/dt , nperseg=2**9, window='hann',detrend='linear')
#plotting


plt.figure(figsize=(15,10))
plt.loglog(*piston_psd_before_A0,label='before DM',color='k',lw=3,linestyle='--')
for i, psd in enumerate(pistons_psd_after_AO):
    if i != 2:
        plt.loglog(*psd,label='after DM, DM piston definition has {}m central obscuration'.format( labels[i] ))
    else:
        plt.loglog(*psd,label='after DM, DM piston definition has {}m (true) central obscuration'.format( labels[i] ))
        #plt.loglog(psd[0],np.cumsum(psd[1][::-1]*np.diff(psd[0])[1])[::-1])
plt.legend(loc='lower left',fontsize=15)
plt.gca().tick_params(labelsize=22)
plt.xlabel('Frequency (Hz)',fontsize=22)
plt.ylabel(r'Piston PSD $(rad^2/Hz)$',fontsize=22)
plt.title(r'D={}m, $r_0$={}m, $L_0$=25m, V={}m/s'.format(D,r0,L0,V_w),fontsize=20)


# GET OPD (rad) VS piston definition 
opds_after_AO = [np.trapz(psd[1],psd[0])**0.5 for psd in pistons_psd_after_AO]
opd_before_AO = np.trapz(piston_psd_before_A0[1],psd[0])**0.5

#get central obscuration error
ror = 16 #real obscuration radius
central_obs_error = [ror/cor for cor in M2_ratios]
plt.figure(figsize=(8,5))
plt.loglog(central_obs_error,np.array(opds_after_AO)/opd_before_AO,'-o')
plt.gca().tick_params(labelsize=15)
#plt.xlabel('central obscuration error (%) in pupil definition',fontsize=15)
plt.xlabel(r'$M2_{piston definition}/M2_{real}$',fontsize=15)
plt.ylabel('increase in OPD (%)',fontsize=15)


fig,ax = plt.subplots(len(pupil_tuples),2,figsize=(4,10))
for i in range(len(pupil_tuples)):
    ax[i,0].imshow(pupil_tuples[i][0])
    ax[i,1].imshow(pupil_tuples[i][1])
    ax[i,0].axis('off')
    ax[i,1].axis('off')
    
ax[0,0].set_title('True Piston\nBasis',fontsize=15)
ax[0,1].set_title('AO Piston\nBasis',fontsize=15)

#%% Creating / optimizing fiber mode and apodized pupil

#play with fiber parameters to find reasonable one that is close to diffraction limit 
for NA in np.logspace(-2,1,20):
#for n_core in np.linspace(1,5,10):
    n_core=4 # NA has to be less then n_core (N-clad = sqrt(N_core**2 - NA**2)  )
    a = np.logspace(-9,-5,100)
    waist =  a * (0.65 + 1.619/(2*np.pi*NA*a/wvl)**(3/2) + 2.879/(2*np.pi*NA*a/wvl)**6) # wvl/D
    plt.figure()
    
    plt.loglog(a, waist,color='k')
    plt.axhline(wvl/D,color='k',linestyle='--',label=r'$\lambda$/D')
    
    plt.legend()
    plt.xlabel('fiber core diameter (m)')
    plt.ylabel('gaussian aprox waist (m)')
    plt.title(f'NA={NA}')
    
    axx = plt.twinx()    
    #V = 2*np.pi*NA*a/wvl , single mode cut-off at 2.4
    axx.semilogx(a, 2*np.pi*NA*a/wvl, color='green')
    axx.axhline(2.4, linestyle='--', color='green')
    axx.set_ylabel('V')

#initialize parameters and fiber field array in image plane
a = 1.8e-7 # 1e-6 #fiber core diameter
NA = 3.36 #.45 #numerical apperature 
n_core = 4 #refractive index of core
fiber_field_ip = np.nan*np.ones([len(x),len(x)])
#calculate it
for i,x_row in enumerate(x):
    #if np.mod(i,100)==0:
    #    print('{}% complete'.format(i/len(x)))
    fiber_field_ip[i,:] = [fields.gaussian_field(NA=NA,n_core=n_core, a=a, L=wvl,r=np.sqrt(x_row**2 + x_col**2)) for x_col in x]

#go to pupil plane (its real in the focal plane so reasonable to enforce that its real in pupil also (take abs))
fiber_field_pp = abs(sfft.fftshift(sfft.ifft2(fiber_field_ip)) )
#normalization from (Perrin, Woillez 2019) np.nansum(abs(fiber_field_pp)**2 * dfX**2) = 1 
fiber_field_pp *= 1/abs(np.sqrt(np.nansum(fiber_field_pp)**2 * dfX**2 ))

"""
Here we do a quick check that fiber coupling gives expected results:
    eta = 1 for perfect overlap
    
input field (fiber_field_pp) must be normalized properly (Perrin, Woillez 2019)
np.nansum(abs(fiber_field_pp)**2) = 1 
"""


#get the weighted pupil appodized mask with (note its reasonable to assume real/ignore imaginary part)
pupil_apodized_mask = abs(pupil_obs_mask * crop_big_array(fiber_field_pp , pupil_obs_mask))
#normalize
pupil_apodized_mask *= 1/abs(np.sqrt(np.nansum(pupil_apodized_mask**2*dfX**2)))

#lets take a look
plt.figure()
plt.imshow(pupil_apodized_mask)

### --------- Plot showing fiber parameters and fiber field in pupil 

fig = plt.figure(tight_layout=True,figsize=(16,6))
gs = gridspec.GridSpec(10, 15)

ax0 = fig.add_subplot(gs[:, :10])
ax1 = fig.add_subplot(gs[2:7, 11:14])
#ax1.axis('off')

aaa = np.logspace(-9,-5,100)
a_real = a #selected fiber core diameter
waist =  aaa * (0.65 + 1.619/(2*np.pi*NA*aaa/wvl)**(3/2) + 2.879/(2*np.pi*NA*aaa/wvl)**6) # wvl/D
plt.figure()

ax0.loglog(aaa, waist,color='k')
ax0.axhline(wvl/D,color='k',linestyle='--',lw=3,label=r'$\lambda$/D')
ax0.axvline(a,color='k',linestyle='-.',lw=3,label='selected fiber diameter')
ax0.legend(fontsize=20)
ax0.set_xlabel('Fiber core diameter (m)',fontsize=20)
ax0.set_ylabel('Gaussian waist (m)',fontsize=20)
ax0.set_title(f'NA={NA}')
ax0.tick_params(labelsize=20)
#ax0.set_ylim([1e-9,1e1])

axx = ax0.twinx()    
#V = 2*np.pi*NA*a/wvl , single mode cut-off at 2.4
axx.semilogx(aaa, 2*np.pi*NA*aaa/wvl, color='green')
axx.axhline(2.4, linestyle='--', color='green')
axx.axhspan(0, 2.4, facecolor='green', alpha=0.2,label='single mode regime')
axx.legend(loc='lower left',fontsize=20)
axx.set_ylabel('V number',fontsize=20)
axx.yaxis.label.set_color('green')
axx.set_ylim([0,15])
axx.tick_params(labelsize=20)
axx.tick_params(axis='y', colors='green')


ax1.imshow(pupil_apodized_mask)
ax1.axis('off')
ax1.set_title('Fiber Apodised\nPupil',fontsize=20 )

#fig.savefig('/Users/bcourtne/Documents/ANU-PhD/1st_phd_pres_figures/optimzing_fiber_coupling.png')    

#%% Experiment with fiber coupling and different AO piston free definitions 

#remember tuple order (real telescope pupil, pupil definition applied in piston removal)

    
"""    Original - ithink what we want one  """
pupil_definition_dict4fiber = dict({'correct':(pupil_apodized_mask, pupil_apodized_mask),\
                     'incorrect':(pupil_apodized_mask, pupil_obs_mask)})
    
#labels4fiber = list(pupil_definition_dict.keys())

#A0 correction
AO_correction_dict4fiber = AO_correction(screens,dt, V_w, dfX, no_modes_corrected, D_pixel, pupil_definition_dict4fiber  )

#fiber coupling 
coupled_fields = [[ np.nansum(np.exp(-1j*p) * pupil_apodized_mask * dfX**2 ) for p in AO_correction_dict4fiber[x]['residual_screen'] ] for x in AO_correction_dict4fiber.keys()]

#piston
coupled_phase = [np.unwrap(np.angle(f)) for f in coupled_fields]

#efficiency 
coupled_efficiency = [[ abs(np.nansum(np.exp(-1j*p) * pupil_apodized_mask * dfX**2 ))**2 /\
                       ( np.nansum(abs(np.exp(1j*p))**2 * dfX**2) * \
                        np.nansum(abs(pupil_apodized_mask)**2 * dfX**2)) \
                           for p in AO_correction_dict4fiber[x]['residual_screen'] ] for x in AO_correction_dict4fiber.keys()]

    
#CHECK
print('check this {}=1'.format(abs(np.nansum(pupil_apodized_mask**2 * dfX**2 ))**2 /\
                       ( np.nansum(abs(pupil_apodized_mask)**2 * dfX**2) * \
                        np.nansum(abs(pupil_apodized_mask)**2 * dfX**2))   ) )
    
#PSD
coupled_psds = [sig.welch(p, fs=1/dt , nperseg=2**9, window='hann',detrend='linear') for p in coupled_phase]


fig = plt.figure(tight_layout=True,figsize=(16,6))
gs = gridspec.GridSpec(6, 16)

ax0 = fig.add_subplot(gs[:, :8])
ax1 = fig.add_subplot(gs[0:3, 9:12])
#ax1.axis('off')
ax2 = fig.add_subplot(gs[0:3, 12:15])
#ax2.axis('off')
ax3 = fig.add_subplot(gs[3:6, 9:12])
#ax3.axis('off')
ax4 = fig.add_subplot(gs[3:6, 12:15])
#ax4.axis('off')

ax0.loglog(*piston_psd_before_A0,label='orginal atmospheric piston')
ax0.loglog(*coupled_psds[0],label='fiber coupled piston w apodised piston free basis')
ax0.loglog(*coupled_psds[1],label='fiber coupled piston w/o apodised piston free basis')
ax0.loglog(coupled_psds[0][0],1e3*coupled_psds[0][0]**(-8/3),label=r'$f^{-8/3}$',color='k',linestyle=':')
ax0.loglog(coupled_psds[0][0],1e3*coupled_psds[0][0]**(-17/3),label=r'$f^{-17/3}$',color='k',linestyle='--')
ax0.legend(loc='lower left',fontsize=15)
ax0.tick_params(labelsize=22)
ax0.set_xlabel('Frequency (Hz)',fontsize=22)
ax0.set_ylabel(r'Piston PSD $(rad^2/Hz)$',fontsize=22)
ax0.set_title(r'D={}m, $r_0$={}m, $L_0$=25m, V={}m/s'.format(D,r0,L0,V_w),fontsize=20)


ax1.imshow(pupil_definition_dict4fiber['correct'][0])
ax2.imshow(pupil_definition_dict4fiber['correct'][1])
ax3.imshow(pupil_definition_dict4fiber['incorrect'][0])
ax4.imshow(pupil_definition_dict4fiber['incorrect'][1])

ax1.set_title('Apodised Pupil',fontsize=15,fontweight="bold")
ax2.set_title('AO Piston\nBasis',fontsize=15,fontweight="bold")
#ax1.set_ylabel('correct piston\nfree basis',fontsize=15,fontweight="bold")
#ax3.set_ylabel('in-correct piston\nfree basis',fontsize=15,fontweight="bold")
for axx in [ax1, ax2,ax3,ax4]:
    axx.set_xticks([])
    axx.set_yticks([])

#plt.savefig('/Users/bcourtne/Documents/ANU-PhD/1st_phd_pres_figures/psds_piston_free_basis_def_fiber_coupling.png')    


#%%
"""



                IGNORE BELOW HERE 


"""






























#%% Just doing simple average of each zernike mode across different pupils

#normalize pupil mask <P|P> = 1 
pupil_mask_norm = pupil_mask/np.nansum(pupil_mask * pupil_mask)
#normalize pupil mask <P|P> = 1 
pupil_obs_mask_norm = pupil_obs_mask/np.nansum(pupil_obs_mask * pupil_obs_mask )
#includes piston, and apply nan mask to zernikes  
zernikes = [pupil_mask * aotools.functions.zernike.zernike_noll(m+2,int(D_pixel)) for m in range(200)]

apodized_zernikes = [pupil_mask * crop_big_array(fiber_field_pp,pupil_mask) * aotools.functions.zernike.zernike_noll(m+2,int(D_pixel)) for m in range(200)]

plt.plot(range(2,len(zernikes)+2,1),[np.nansum(pupil_mask_norm * zzz) for zzz in zernikes],label='circular pupil')
plt.plot(range(2,len(zernikes)+2,1),[np.nansum(pupil_obs_mask_norm * zzz) for zzz in zernikes],label='donut pupil')
plt.plot(range(2,len(zernikes)+2,1),[np.nansum(pupil_mask_norm * zzz) for zzz in apodized_zernikes],label='apodized circular pupil')
plt.plot(range(2,len(zernikes)+2,1),[np.nansum(pupil_obs_mask_norm * zzz) for zzz in apodized_zernikes],label='apodized donut pupil')

plt.legend()
plt.xlabel('Noll index')
plt.ylabel('piston (normalized)')

#show in overlap integral that piston term comes out as constant...d

#then take some non-zero coefficient of zernike coupling to piston and plot scaled zernike PSD vs piston 



#motivation for piston free definition in AO... Can this be corrected


















#%%  


#pupil = aotools.functions.pupil.circle(radius=D_pixel/2, size=D_pixel, circle_centre=(0, 0), origin='middle')

col2move = 2
print('each iteration moves screen by col2move * dfX = {}m'.format(col2move * dfX))
#master screen
r0 = 0.1 #m (at 500nm)
L0 = 25 #m
wvl = 2.2e-6 #m
seeing = (0.5/2.2)**(6/5) * 0.98*wvl/r0 * 3600 * 180/np.pi #arcsec (at 500nm)
master_phscreens = aotools.turbulence.infinitephasescreen.PhaseScreenVonKarman(nx_size=D_pixel,pixel_scale=dfX,r0=r0,L0=L0,n_columns=col2move,random_seed = 1)
#master_phscreens = aotools.turbulence.infinitephasescreen.PhaseScreenKolmogorov(nx_size=D_pixel,pixel_scale=dfX,r0=r0,L0=25,random_seed = 1)
#generate 1000 iterations of phase screen 
screens = []
for i in range(int(2e3)):
    #each iteration moves scrren by col2move * dfX
    screens.append(master_phscreens.add_row())
    
# calculate piston (with standard )
screen_piston = [np.nanmean(screens[i]*pupil_mask) for i in range(len(screens))]
screen_obs_piston = [np.nanmean(screens[i]*pupil_obs_mask) for i in range(len(screens))]


#define the time stamps, note that sampling rate effectively determines wind speed parameter 
dt = 1e-3 
t = np.arange(0, dt * len(screens), dt)
V_w = col2move * dfX / dt 

#timeseries plot of screen_piston
plt.figure()
plt.plot(screen_piston)
plt.ylabel('piston (rad)',fontsize=15)
plt.xlabel('frames',fontsize=15)

#calculate PSD of screen piston (note that sample frequency here defines effective windspeed in sim!)
piston_sim_PSD = sig.welch(screen_piston, fs=1/dt , nperseg=2**9, window='hann',detrend='linear')
#calculate PSD of screen piston with central M2 obscuration
piston_obs_sim_PSD = sig.welch(screen_obs_piston, fs=1/dt , nperseg=2**9, window='hann',detrend='linear')

#theoretical piston from Conan 1995 (using AO definition of tau0)
piston_theory_PSD , atm_params = atmosphere_piston.atm_piston(wvl=wvl, seeing=seeing, tau_0=0.314 * r0/V_w, L_0=L0, diam=8)

#plotting - theory vs simulation  
plt.figure()
plt.loglog(*piston_sim_PSD, label='simulated')
plt.loglog(*piston_theory_PSD, label='theory')
plt.loglog(piston_sim_PSD[0],1e2*piston_sim_PSD[0]**(-8/3),label=r'$f^{-8/3}$',color='k',linestyle=':')
plt.loglog(piston_sim_PSD[0],1e6*piston_sim_PSD[0]**(-17/3),label=r'$f^{-17/3}$',color='k',linestyle='-.')
plt.xlim([1e-2,3e2])
plt.axvline(0.3*V_w/D,color='red',lw=0.8,linestyle=':',label=r'0.3V/D')
plt.legend(fontsize=12)
plt.gca().tick_params(labelsize=15)
plt.xlabel('Frequency (Hz)',fontsize=15)
plt.ylabel(r'Piston PSD $(rad^2/Hz)$',fontsize=15)
plt.title(r'D={}m, $r_0$={}m, $L_0$=25m, V={}m/s'.format(D,r0,L0,V_w))


#plotting - open pupil vs central obscuration piston PSD 
plt.figure()
plt.loglog(*piston_sim_PSD, label='simulation without M2 obscuration')
plt.loglog(*piston_obs_sim_PSD, label='simulation with M2 obscuration')
#plt.loglog(piston_sim_PSD[0],1e2*piston_sim_PSD[0]**(-8/3),label=r'$f^{-8/3}$',color='k',linestyle=':')
#plt.loglog(piston_sim_PSD[0],1e6*piston_sim_PSD[0]**(-17/3),label=r'$f^{-17/3}$',color='k',linestyle='-.')
plt.xlim([1e-2,3e2])
#plt.axvline(0.3*V_w/D,color='red',lw=0.8,linestyle=':',label=r'0.3V/D')
plt.legend(fontsize=12)
plt.gca().tick_params(labelsize=15)
plt.xlabel('Frequency (Hz)',fontsize=15)
plt.ylabel(r'Piston PSD $(rad^2/Hz)$',fontsize=15)
plt.title(r'D={}m, $r_0$={}m, $L_0$=25m, V={}m/s'.format(D,r0,L0,V_w))


#%% AFTER AO 

N_modes_considered = 55
N_modes_corrected = 50

#get a list of the zernike modes across our pupil so dont have to recalculate every iteration
zernikes = [aotools.functions.zernike.zernike_noll(m+2,int(D_pixel)) for m in range(N_modes_considered)]
#the zernike correction coefficient (1=full correction, 0 = no correction)
f_j = list(np.ones(N_modes_corrected)) + list(np.zeros(N_modes_considered-N_modes_corrected))

framelag = 2 
temporal_lag = framelag * dt

"""
there is a real pupil that the wave passes 
there is a percieved pupil that the DM applies a 'piston free' correction
 
"""

#the list is tuple of the masks [(real pupil, piston free pupil) ... ]
pupil_combination_list = [(pupil_mask, pupil_mask), (pupil_mask, pupil_obs_mask), (pupil_obs_mask, pupil_mask), (pupil_obs_mask,  pupil_obs_mask ) ]
pupil_combination_labels = ['pp','po','op','oo']

#DM_screen = [] #list to hold DM shape applied to do AO correction
#residual_screen = [] #list to hold timeseries of AO corrected pahse screen in pupil
AO_correction_dict = dict({pupil_def : dict({'pupil_masks':pupil_combination_list, 'current_screen':[], 'lagged_screen':[], 'DM_screen':[], 'residual_screen':[]}) for pupil_def in pupil_combination_labels})
for i in range(framelag,len(screens),1):
    #check progress
    if np.mod(i,500)==0:
        print('{}% complete'.format(i/len(screens)))
        
    for pm,pm_key in zip(pupil_combination_list, AO_correction_dict.keys()): #for each pupil mask we're considering 
        
        #########
        # use the "real" telescope pupil (index 0) for the phase screens          
        #current phase across telescope pupil
        current_screen = pm[0] * screens[i]
        #the WFS measured phase accounting for AO lag 
        sensed_screen = pm[0] * screens[i-framelag]
        
        #the list to hold current phase screen zernike coeficients in the pupil
        a_j = []
        #init the superposition of zernike modes to subtract from phase screen
        #the DM shape is applied across the pupil so just use standard definition here 
        DM_shape = pupil_mask * np.zeros([int(D_pixel),int(D_pixel)])
        
        for m in range(N_modes_considered):
            
            if f_j[m] != 0 : #if this mode is one that must be corrected by AO system
                
                #create zernike mode ### WHAT MASK TO APPLY HERE!??!
                z_i = pupil_mask * zernikes[m] #need i+2 since we dont consider piston (index = 1)
                #normalizing (they should be normalized.. but just incase)
                norm = dfX**2*np.nansum(z_i)
                if norm > 1e-10: #if not well enough normalized.. ill do it here
                    z_i *= 1/(dfX**2*np.nansum(z_i))
                    
                    
                #print(dfX**2 * np.nansum(z_i))
                #find the phase screen coefficient for this mode
                a_i = dfX**2*np.nansum(sensed_screen * z_i)
                #append the coefficient to the a_j list
                a_j.append(a_i)
                #subtract the amount of AO correction
                DM_shape = DM_shape + f_j[m]*a_j[m]*z_i
                
                """if np.mod(m,10)==0:
                    plt.figure()
                    plt.imshow(DM_shape)"""
    
                
        ###### -------- PISTON FREE DEFINITIONS ---------- #########
        #use the 'piston free pupil' for the mask on the DM (index 1) 
        #DM_shape = DM_shape - np.nanmean(pm[1] * DM_shape) 
        DM_shape = DM_shape - np.nansum(pm[1] * DM_shape * dfX**2) / np.nansum(pm[1] * dfX**2)
        
        ###### ------------------------------------------ #########    
        #print(np.nanmean(DM_pf0))
        #print(dfX**2*np.nansum(DM_pf0))
    
        #Applying AO correction
        AO_correction_dict[pm_key]['current_screen'].append( current_screen )
        AO_correction_dict[pm_key]['lagged_screen'].append( sensed_screen ) #what the AO sees
        AO_correction_dict[pm_key]['DM_screen'].append(DM_shape)
        AO_correction_dict[pm_key]['residual_screen'].append( current_screen - DM_shape)
        #DM_screen.append( DM_pf0 )
        #residual_screen.append( current_screen - DM_pf0 )


#check pistons match with agreeing piston free pupil defintions  

#check piston mis-match with dis-agreeing piston free pupil definition 
op_piston = [ np.nanmean(p) for p in AO_correction_dict['op']['residual_screen'] ]
oo_piston = [ np.nanmean(p) for p in AO_correction_dict['oo']['residual_screen'] ]

#op_piston = [ 1/np.nansum(pupil_obs_mask) * np.nansum(p * pupil_obs_mask) for p in AO_correction_dict['op']['residual_screen'] ]
#oo_piston = [ 1/np.nansum(pupil_obs_mask) * np.nansum(p * pupil_obs_mask) for p in AO_correction_dict['oo']['residual_screen'] ]

plt.plot(op_piston,label='op')
plt.plot(screen_obs_piston[:100],label='actual')
plt.legend()

obs_piston_psd = sig.welch(screen_obs_piston,fs=1/dt , nperseg=2**9, window='hann',detrend='linear')
op_piston_psd = sig.welch(op_piston ,fs=1/dt , nperseg=2**9, window='hann',detrend='linear')

plt.figure(figsize=(12,9))
plt.loglog(*piston_sim_PSD , label='before AO piston')
plt.loglog(*obs_piston_psd , label='after AO with correct "piston free" definition')
plt.loglog(*op_piston_psd  , label='after AO with in-correct "piston free" definition')
#plt.loglog(op_piston_psd[0], op_piston_psd[0]**(-17/3) ,label=r'$f^{-17/3}$')
plt.legend(loc='lower left',fontsize=15)
plt.gca().tick_params(labelsize=22)
plt.xlabel('Frequency (Hz)',fontsize=22)
plt.ylabel(r'Piston PSD $(rad^2/Hz)$',fontsize=22)
plt.title(r'D={}m, $r_0$={}m, $L_0$=25m, V={}m/s'.format(D,r0,L0,V_w),fontsize=20)

#OPD differences
np.sqrt(np.trapz(op_piston_psd[1],op_piston_psd[0]))
np.sqrt(np.trapz(obs_piston_psd[1],obs_piston_psd[0]))


#plotting pupils
xxx = np.linspace(0,dfX*pupil.shape[0],pupil.shape[0]) 
plt.pcolormesh(xxx,xxx,AO_correction_dict['circle']['current_screen'][0])
plt.xlabel('x (m)',fontsize=20)
plt.ylabel('y (m)',fontsize=20)
plt.gca().tick_params(labelsize=15)
plt.gca().axis('equal')
plt.xlim([0,8])
plt.ylim([0,8])
del xxx

xxx = np.linspace(0,dfX*pupil.shape[0],pupil.shape[0]) 
plt.pcolormesh(xxx,xxx,AO_correction_dict['donut']['current_screen'][0])
plt.xlabel('x (m)',fontsize=20)
plt.ylabel('y (m)',fontsize=20)
plt.gca().tick_params(labelsize=15)
plt.gca().axis('equal')
plt.xlim([0,8])
plt.ylim([0,8])
del xxx

plt.imshow(AO_correction_dict['op']['current_screen'][0])
plt.axis('off')
plt.imshow(AO_correction_dict['pp']['current_screen'][0])
plt.axis('off')


#%% Fiber coupling 

#######Exploring parameter space for fiber coupling 
n_core=2
for NA in np.logspace(-2,1,10):
#for n_core in np.linspace(1,5,10):

    a = np.logspace(-9,-5,100)
    waist =  a * (0.65 + 1.619/(2*np.pi*NA*a/wvl)**(3/2) + 2.879/(2*np.pi*NA*a/wvl)**6) # wvl/D
    plt.figure()
    
    plt.loglog(a, waist,color='k')
    plt.axhline(wvl/D,color='k',linestyle='--',label=r'$\lambda$/D')
    
    plt.legend()
    plt.xlabel('fiber core diameter (m)')
    plt.ylabel('gaussian aprox waist (m)')
    plt.title(f'NA={NA}')
    
    axx = plt.twinx()    
    #V = 2*np.pi*NA*a/wvl , single mode cut-off at 2.4
    axx.semilogx(a, 2*np.pi*NA*a/wvl, color='green')
    axx.axhline(2.4, linestyle='--', color='green')
    axx.set_ylabel('V')
    
    
#hardcoded parameters Exploring parameter space 
wvl, f,D = 2.2e-6, 20, 8 #wavelength, focal length, telescope radius (m)
a = 1e-6 #fiber core diameter
NA = .45 #numerical apperature 
n_core=2

# init fiber fields
#initialize fiber field array in image plane
fiber_field_ip = np.nan*np.ones([len(x),len(x)])
#calculate it
for i,x_row in enumerate(x):
    #if np.mod(i,100)==0:
    #    print('{}% complete'.format(i/len(x)))
    fiber_field_ip[i,:] = [fields.fiber_field(NA=NA,n_core=2.7,a=a,L=wvl,r=np.sqrt(x_row**2 + x_col**2)) for x_col in x]
#normalization
fiber_field_ip *= 1/np.sum(fiber_field_ip[np.isfinite(fiber_field_ip)]*dx**2)
#for some reason inf value at r=0... need to investigate, for now interpolate 
fiber_field_ip[abs(fiber_field_ip) == np.inf] = np.nan
fiber_field_ip[len(x)//2,:] = pd.Series(fiber_field_ip[len(x)//2,:]).interpolate().values

fiber_field_ip2 = np.nan*np.ones([len(x),len(x)])
#calculate it
for i,x_row in enumerate(x):
    #if np.mod(i,100)==0:
    #    print('{}% complete'.format(i/len(x)))
    fiber_field_ip2[i,:] = [fields.gaussian_field(NA=NA,n_core=2.7,a=a,L=wvl,r=np.sqrt(x_row**2 + x_col**2)) for x_col in x]
#normalization
fiber_field_ip2 *= 1/np.sum(fiber_field_ip2*dx**2) 

#diffraction limit 
difr_limit_ip = sfft.fftshift(sfft.fft2(pupil))
difr_limit_ip *= 1/np.sum(difr_limit_ip * dx**2) 


#put fiber field in pupil plane (approximation)
fiber_field_pp2 = sfft.fftshift(sfft.ifft2(fiber_field_ip2))
fiber_field_pp2 *= 1/np.nansum(fiber_field_pp2 * dfX**2 )

#comparison of theory vs approx fiber field 
plt.figure()
plt.plot(1e6*x[N//4:3*N//4] , fiber_field_ip[N//2,N//4:3*N//4],label='fiber field')
plt.plot(1e6*x[N//4:3*N//4] , fiber_field_ip2[N//2,N//4:3*N//4],label='fiber field approx')
plt.legend()
axx=plt.twinx()
axx.plot(1e6*x[N//4:3*N//4] , difr_limit_ip[N//2,N//4:3*N//4],label='diffraction limit',color='red')


plt.figure()
plt.imshow(abs(sfft.fftshift(sfft.fft2(pupil))))
plt.xlabel('pos (um)')
plt.ylabel('field amplitude')
plt.legend()


#put fiber field in pupil plane (approximation)

plt.figure()
plt.imshow(abs(fiber_field_pp2[500:1500,500:1500]))


#%%
#### FIBER COUPLING 
def crop_big_array(big_array, small_array):
    #crop big array to be centered in small array  (assuming both are square!!!)

    xxx = small_array.shape[0]
    yyy = big_array.shape[0]
    return(big_array[(yyy-xxx)//2:(yyy+xxx)//2 , (yyy-xxx)//2:(yyy+xxx)//2 ] )


apodized_field , nonapodized_field = [], []
for i in range(len( AO_correction_dict['op']['residual_screen'] )):
    z_i = AO_correction_dict['op']['residual_screen'][i]
    z_jjj = AO_correction_dict['pp']['residual_screen'][i]
    #field after fiber coupling with single mode (Woillez, Perrin 2019) - I think I have to use obs_mask here because that is indeed the physical system..
    apodized_field.append( np.nansum(np.exp(-1j*z_i) * pupil_obs_mask * crop_big_array(fiber_field_pp2,z_i) * dfX**2) )
    
    #field after fiber coupling that matches diffraction limit (i.e. pupil) (Woillez, Perrin 2019)
    #pup_norm = pupil.copy() / np.sum(pupil * dfX**2)
    nonapodized_field.append( np.nansum(np.exp(-1j*z_jjj) * pupil_obs_mask * crop_big_array(fiber_field_pp2,z_i) * dfX**2) )

unwrapped_phase_apodized = np.unwrap(np.angle(apodized_field))
unwrapped_phase_nonapodized = np.unwrap(np.angle(nonapodized_field))


psd_phase_apodized = sig.welch(unwrapped_phase_apodized, fs=1/dt , nperseg=2**9, window='hann',detrend='linear')
psd_phase_nonapodized = sig.welch(unwrapped_phase_nonapodized, fs=1/dt , nperseg=2**9, window='hann',detrend='linear')

plt.loglog(*psd_phase_apodized,label='correct pupil def 4 piston free')
plt.loglog(*psd_phase_nonapodized,label='correct pupil def 4 piston free')
plt.legend(loc='lower left',fontsize=15)
plt.gca().tick_params(labelsize=22)
plt.xlabel('Frequency (Hz)',fontsize=22)
plt.ylabel(r'Piston PSD $(rad^2/Hz)$',fontsize=22)
plt.title(r'D={}m, $r_0$={}m, $L_0$=25m, V={}m/s'.format(D,r0,L0,V_w),fontsize=20)

#theoretical piston from Conan 1995 (using AO
#theoretical piston from Conan 1995 (using AO

#plt.plot(np.unwrap(np.angle(apodized_field)))



#plt.imshow(abs(fiber_field_pp2[(yyy-xxx)//2:(yyy+xxx)//2 , (yyy-xxx)//2:(yyy+xxx)//2 ]))





















"""i=5
plt.figure()    
plt.imshow(pupil_mask*screens[i])
plt.figure() 
plt.imshow(DM_screen[i])
plt.figure() 
plt.imshow(residual_screen[i])"""

#%% random pupil obscuration analysis (need to update)
#define our pupils 
pupil_m1 = aotools.functions.pupil.circle(radius=D_pixel/2, size=N, circle_centre=(0, 0), origin='middle')
pupil_m2 =  aotools.functions.pupil.circle(radius=D_pixel/8, size=N, circle_centre=(0, 0), origin='middle')

#consider with and without central obscuration
pupil = pupil_m1
pupil_obs = pupil_m1 - pupil_m2 

#define a simple piston free phase mode for rad R, if r>r1 phi=pi, r<r1 phi=-pi for conserved areas r1 = 1/sqrt(2) * R
phi1 = np.pi * (aotools.functions.pupil.circle(radius=D_pixel/2, size=N, circle_centre=(0, 0), origin='middle') - aotools.functions.pupil.circle(radius=round(1/np.sqrt(2)*D_pixel/2), size=N, circle_centre=(0, 0), origin='middle'))
phi2 = -np.pi * aotools.functions.pupil.circle(radius=int(1/np.sqrt(2)*D_pixel/2), size=N, circle_centre=(0, 0), origin='middle')

#or a given Zernike Mode

z_i_crop = aotools.functions.zernike.zernike_noll(5,int(D_pixel))
z_i = pupil.copy()
nb = pupil.shape[0]
na = z_i_crop.shape[0]
lower = (nb) // 2 - (na // 2)
upper = (nb // 2) + (na // 2)
z_i[lower:upper, lower:upper] = z_i_crop

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(fXX,fYY, phi1+phi2, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

#piston with / without central obscuration 
np.mean(pupil*(phi1+phi2))
np.mean(pupil_obs*(phi1+phi2))

##########
# Could also look at this theoretically with Connon 1995 subtracting out central obscuration 
#########


#%% Exploring parameter space for fiber coupling 
wvl, f,D = 2.2e-6, 20, 8 #wavelength, focal length, telescope radius (m)
n_core=2



for NA in np.logspace(-2,1,10):
#for n_core in np.linspace(1,5,10):

    a = np.logspace(-9,-5,100)
    waist =  a * (0.65 + 1.619/(2*np.pi*NA*a/wvl)**(3/2) + 2.879/(2*np.pi*NA*a/wvl)**6) # wvl/D
    plt.figure()
    
    plt.loglog(a, waist,color='k')
    plt.axhline(wvl/D,color='k',linestyle='--',label=r'$\lambda$/D')
    
    plt.legend()
    plt.xlabel('fiber core diameter (m)')
    plt.ylabel('gaussian aprox waist (m)')
    plt.title(f'NA={NA}')
    
    axx = plt.twinx()    
    #V = 2*np.pi*NA*a/wvl , single mode cut-off at 2.4
    axx.semilogx(a, 2*np.pi*NA*a/wvl, color='green')
    axx.axhline(2.4, linestyle='--', color='green')
    axx.set_ylabel('V')
    
    
#hardcoded parameters Exploring parameter space 
wvl, f,D = 2.2e-6, 20, 8 #wavelength, focal length, telescope radius (m)
a = 1e-6 #fiber core diameter
NA = .45 #numerical apperature 
n_core=2


#diffraction limit 
difr_limit_ip = sfft.fftshift(sfft.fft2(pupil))
difr_limit_ip *= 1/np.sum(difr_limit_ip * dx**2) 

#check field approx cross section to diffraction limit 
aaa = [fields.gaussian_field(NA=NA,n_core=n_core,a=a,L=wvl,r=rr) for rr in x]
bbb = difr_limit_ip[N//2,:]

fig,ax = plt.subplots(2,1,sharex=True)
ax[0].plot(x, aaa)
ax[1].plot(x, bbb)



#%% init fiber fields
#initialize fiber field array in image plane
fiber_field_ip = np.nan*np.ones([len(x),len(x)])
#calculate it
for i,x_row in enumerate(x):
    #if np.mod(i,100)==0:
    #    print('{}% complete'.format(i/len(x)))
    fiber_field_ip[i,:] = [fields.fiber_field(NA=NA,n_core=2.7,a=a,L=wvl,r=np.sqrt(x_row**2 + x_col**2)) for x_col in x]
#normalization
fiber_field_ip *= 1/np.sum(fiber_field_ip[np.isfinite(fiber_field_ip)]*dx**2)
#for some reason inf value at r=0... need to investigate, for now interpolate 
fiber_field_ip[abs(fiber_field_ip) == np.inf] = np.nan
fiber_field_ip[len(x)//2,:] = pd.Series(fiber_field_ip[len(x)//2,:]).interpolate().values

fiber_field_ip2 = np.nan*np.ones([len(x),len(x)])
#calculate it
for i,x_row in enumerate(x):
    #if np.mod(i,100)==0:
    #    print('{}% complete'.format(i/len(x)))
    fiber_field_ip2[i,:] = [fields.gaussian_field(NA=NA,n_core=2.7,a=a,L=wvl,r=np.sqrt(x_row**2 + x_col**2)) for x_col in x]
#normalization
fiber_field_ip2 *= 1/np.sum(fiber_field_ip2*dx**2) 

#diffraction limit 
difr_limit_ip = sfft.fftshift(sfft.fft2(pupil))
difr_limit_ip *= 1/np.sum(difr_limit_ip * dx**2) 


#comparison of theory vs approx fiber field 
plt.figure()
plt.plot(1e6*x[N//4:3*N//4] , fiber_field_ip[N//2,N//4:3*N//4],label='fiber field')
plt.plot(1e6*x[N//4:3*N//4] , fiber_field_ip2[N//2,N//4:3*N//4],label='fiber field approx')
plt.legend()
axx=plt.twinx()
axx.plot(1e6*x[N//4:3*N//4] , difr_limit_ip[N//2,N//4:3*N//4],label='diffraction limit',color='red')


plt.figure()
plt.imshow(abs(sfft.fftshift(sfft.fft2(pupil))))
plt.xlabel('pos (um)')
plt.ylabel('field amplitude')
plt.legend()


#put fiber field in pupil plane (approximation)
fiber_field_pp2 = sfft.fftshift(sfft.ifft2(fiber_field_ip2))
fiber_field_pp2 *= 1/np.nansum(fiber_field_pp2 * dfX**2 )
plt.figure()
plt.imshow(abs(fiber_field_pp2[500:1500,500:1500]))










#%% simulation 


#get some (500) corrected phase screens
seeing=0.8
tau0=4e-3
L0=25
A0_lag = 2e-3

phase_screens = AO_sim.AO_sim(50, wvl, f, D, N , D_pixel, seeing, tau0, L0, A0_lag,iters = 500)

dt = 2e-3 #A0_lag
apodized_field = []
nonapodized_field = []
z_ii = pupil.copy()
z_ii[pupil==0] = np.nan # its important to put to nan outside of pupil (effects std error etc)
for i in range(len(phase_screens)):
    #insert them into the bigger array 
    z_i = z_ii.copy() #pupil with outside set to nan
    z_i[N//2-int(D_pixel)//2:N//2+int(D_pixel)//2,N//2-int(D_pixel)//2:N//2+int(D_pixel)//2] = phase_screens[i].data
    
    #remove piston (average WF shape)
    
    z_i = z_i - np.nanmean(z_i)   #this is right, right?
    
    #field after fiber coupling with single mode (Woillez, Perrin 2019)
    apodized_field.append( np.nansum(np.exp(-1j*z_i) * fiber_field_pp2 * dfX**2) )
    
    #field after fiber coupling that matches diffraction limit (i.e. pupil) (Woillez, Perrin 2019)
    pup_norm = pupil.copy() / np.sum(pupil * dfX**2)
    nonapodized_field.append( np.nansum(np.exp(-1j*z_i) * pup_norm * dfX**2) )


plt.figure()
plt.plot(np.arange(0, dt* len( apodized_field ),dt), np.unwrap(np.angle( apodized_field)),label='apodized')
plt.plot(np.arange(0, dt* len( apodized_field ),dt), np.unwrap(np.angle( nonapodized_field)) ,label='non-apodized')
plt.legend()
plt.ylabel('phase (rad)',fontsize=15)
plt.xlabel('time (s)',fontsize=15)
plt.title('fiber coupled phase (piston)',fontsize=15)
plt.gca().tick_params(labelsize=15)
apPSD = sig.welch(np.unwrap(np.angle( apodized_field)) ,fs=1/dt , nperseg=2e9, window='hann',detrend='linear')
noapPSD = sig.welch(np.unwrap(np.angle( nonapodized_field)) ,fs=1/dt , nperseg=2e9, window='hann',detrend='linear')

plt.figure(figsize=(8,5))
plt.loglog(*apPSD,label='apodized pupil',color='red')
plt.loglog(*noapPSD,label='non-apodized pupil',color='orange')

plt.loglog(apPSD[0],np.cumsum(apPSD[1][::-1]*1/dt)[::-1],linestyle='--' ,color='red')
plt.loglog(noapPSD[0],np.cumsum(noapPSD[1][::-1]*1/dt)[::-1],linestyle='--' ,color='orange')

plt.loglog(apPSD[0],1e2*apPSD[0]**(-8/3),label=r'$f^{-8/3}$',color='k',linestyle=':')
plt.loglog(apPSD[0],1e6*apPSD[0]**(-17/3),label=r'$f^{-17/3}$',color='k',linestyle='-.')

plt.ylim([1e-12,1e3])
plt.xlim([1e0,3e2])
plt.ylabel(r'PSD $(rad^2/Hz)$'+'\n'+r'reverse cummulative $(rad^2)$',fontsize=15)
plt.xlabel('Frequency (Hz)',fontsize=15)
plt.title('single telescope fiber coupled phase (piston)',fontsize=15)
plt.gca().tick_params(labelsize=15)
plt.legend(fontsize=15)
plt.text(4e1,1e-8,r'D={}m, seeing = {}"'.format(D, seeing))
plt.text(4e1,1e-9,r'$\tau_0$={}ms, $L_0$={}m'.format(1e3*tau0, L0))
#print('piston apodized = {}, piston non-apodized = {}'.format(np.angle(apodized_filed),np.angle(nonapodized_filed )))


#%%


def fiber_coupling_eta(pupil_mask, fiber_field_pupil, screens , dfX):
    """
    calculate eta from Perrin & Woillez 2019 
    ASSUMES fiber field and screens do not have pupil mask applied. the mask is specified here!!!!
    
    Parameters
    ----------
    fiber_field_in_pupil : array
        DESCRIPTION.  fiber field in pupil, does not assume the pupil mask has been applied
    pupil_mask : TYPE
        DESCRIPTION. - APODIZED WEIGHTED PUPIL MASK
    screens : List 
        DESCRIPTION. list of phase screens (should be multiplied by pupil)
    dfX : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    #Do I normalize pupil mask? 
    #pupil_mask =  pupil_mask * 1/np.nansum(pupil_mask * dfX**2) # np.zeros(pupil_mask.shape) #*= 10000 #1/np.nansum(pupil_mask * dfX**2)
    
    eta = []
    
    #applying the pupil mask to fiber field in the pupil
    if fiber_field_pupil.shape[0] > pupil_mask.shape[0]: #if need to crop fiber field to fit pupil mask
        print( 'WARNING: cropping fiber field to fit pupil mask' )
        fiber_field_pupil = crop_big_array(fiber_field_pupil,pupil_mask)
    
    elif fiber_field_pupil.shape[0] < pupil_mask.shape[0]: 
        print( 'WARNING: cropping pupil_mask to fit fiber field' )
        fiber_field_pupil = crop_big_array(pupil_mask,fiber_field_pupil)
        
        
    #should also check if screens match the mask before applying the mask
    #if screens[0].shape[0] == pupil_mask.shape[0]:
    #    #screens = [  screens[i] for i in range(len(screens))]  
        
    if screens[0].shape[0] > pupil_mask.shape[0]:
        print( 'WARNING: cropping phase screens to fit pupil mask' )
        screens = [ crop_big_array(screens[i],pupil_mask) for i in range(len(screens))]
        
    elif screens[0].shape[0] < pupil_mask.shape[0]:
        raise TypeError('the shape of the phase screen arrays are smaller than the pupil mask')
        
    # do overlap integral for fiber coupling each screen
    for screen in screens: 
        
        
        #tel_field = np.exp(1j * np.angle(fiber_field_pp) )
        #tel_field *= 1/(np.nansum(tel_field)**2 * dfX**2)

        # NOTE FROM PERRIN WOILLEZ 2019 they dont take square in overlap integral ...  I cant get eta = 1 without this 
        eta.append( np.nansum((pupil_mask * fiber_field_pupil * np.exp(-1j*screen))**2 * dfX**2) )
    
    """    
    if screens[0].shape[0] > pupil_mask.shape[0]: #if need to crop fiber field to fit pupil mask
        print( 'WARNING: cropping fiber field to fit pupil mask' )
        for screen in screens: 
            coupled_field.append( np.nansum(np.exp(-1j*fiber_field_pupil) * np.exp(1j*crop_big_array(screen,pupil_mask)) * dfX**2) )
    
    elif screens[0].shape[0] == pupil_mask.shape[0]: #no cropping needed
        for screen in screens:
            coupled_field.append( np.nansum(np.exp(-1j*fiber_field_pupil) * np.exp(1j*screen) * dfX**2) )
   
    else:
        raise TypeError('pupil mask array is bigger than fiber field array in pupil')
    """
    
    return(np.array(eta))
    