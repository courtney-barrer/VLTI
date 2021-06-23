
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 14:26:27 2021

@author: bcourtne

send this a bunch of gravity files (or one) and it will return 2 dictionaries 
my_stat_dict - this has general stats about OPD 
PSD_dict - this has processed OPDs including PSDs etc
"""
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

import astropy.units as u
from astropy import coordinates as coord 


#os.chdir('/Users/bcourtne/Documents/Hi5/vibration_analysis/')


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

atm_para_list = [
'HIERARCH ESO ISS AMBI TAU0 END',
'HIERARCH ESO ISS AMBI TAU0 START',
'HIERARCH ESO ISS AMBI WINDSP',
'HIERARCH ESO ISS AMBI FWHM END' ,
'HIERARCH ESO ISS AMBI FWHM START',
'HIERARCH ESO ISS AMBI IRSKY TEMP', 
'HIERARCH ESO ISS AMBI WINDDIR',
'HIERARCH ESO ISS AMBI TEMP',
'HIERARCH ESO ISS AMBI IWV END',
'HIERARCH ESO ISS AMBI IWV START',
'HIERARCH ESO ISS AIRM START'
]





def do_OPD_analysis(file_list):
    
    psd_dict = dict({'objects':[],'disturb_ts':[],'residual_ts':[],'disturb_psd':[],'residual_psd':[],'coherence':[],\
                     'tf':[], 'residual_filt_psd':[],'disturb_rcum':[],'residual_rcum':[],'residual_filt_rcum':[],\
                         'no_samples':[],'help':[]}) #to hold tuples of (f,PSD) - i.e. sig.welch() output
    
    my_stat_dict = dict({'objects':[],'opd_std':[],'opd_LPF_std':[],'ao_type':[],'ft_pol_mode':[]}) 
    
    #ALL OPDs need to be arrays for each baseline!!! therefore cannot be in my_stat_dict if going to turn to np.array
    #unless unique columns for each baseline OPD.. maybe makes more sense... also need to include baseline length
    
     
    #init atm parameter keys
    for tmp in atm_para_list:
        my_stat_dict[tmp] = []  
    
    #note my_stat_dict values will be turned into numpy array at end of script so values must be compatible 
    
    
    
    #query stellar rad and look at OPD as function of radius vs baseline ratio !!!
    customSimbad = Simbad()
    customSimbad.add_votable_fields('sptype')
    customSimbad.add_votable_fields('flux(V)')
    customSimbad.add_votable_fields('flux(J)')
    customSimbad.add_votable_fields('flux(K)')
    
    
    
    
        
    target_dict = dict()  
    
    #file_list = os.listdir(data_path)
    
    for iii,file in enumerate(file_list): 
        
        
        with fits.open( file ) as hdulist:
            
            # ================
            # query object magnitudes, spectral type etc
            # ================
            
            #get object name
            targ_tmp = hdulist[0].header['ESO OBS TARG NAME']
            if '2MASS' in targ_tmp: #the 2mass object name usually contains special characters that fail the query so we try to fix it 
                targ_tmp = '2MASS '+targ_tmp.split('2MASS')[1][1:]
            
            coord_tmp = ( hdulist[0].header['RA'], hdulist[0].header['DEC'] )
    
            #append object name to our analysis dictionaries 
            psd_dict['objects'].append(targ_tmp)
            my_stat_dict['objects'].append(targ_tmp)
            
            # if we have not already queried this object, query it!
            if targ_tmp not in target_dict:
                
                #init dictionary key with 
                target_dict[targ_tmp] = dict({ 'sptype':None,'Vmag':None,'Jmag':None,'Kmag':None })  
                    
                #e.g. to get list of kmags [target_dict[object]['Kmag'] for object in target_dict]
                
                #print("query new target")
                try:
                    
                    simbadResult_tmp=customSimbad.query_object(targ_tmp)
                    try:
                        target_dict[targ_tmp]['sptype'] = float(simbadResult_tmp['SP_TYPE'][0])
                    except:
                        target_dict[targ_tmp]['sptype'] = np.nan
    
        
                    try:             
                        target_dict[targ_tmp]['Vmag'] = float(simbadResult_tmp["FLUX_V"][0])
                    except:
                        target_dict[targ_tmp]['Vmag'] = np.nan
    
                    try:             
                        target_dict[targ_tmp]['Kmag'] = float(simbadResult_tmp["FLUX_K"][0])
                    except:
                        target_dict[targ_tmp]['Kmag'] = np.nan
                        
                    try:             
                        target_dict[targ_tmp]['Jmag'] = float(simbadResult_tmp["FLUX_J"][0])
                    except:
                        target_dict[targ_tmp]['Jmag'] = np.nan
                except:
                    
                    for key_tmp in target_dict[targ_tmp]:
                        target_dict[targ_tmp][key_tmp] = np.nan
                    
                
            # ================
            # Store atmospheric parameters from headers 
            # ================        
            
            
            for tmp in atm_para_list:
                my_stat_dict[tmp].append(hdulist[0].header[tmp])
            
            
            # ================
            # Determine the extver for FT, based on polarization mode
            # ================
    
            tel_name = []
            for iTel in [4, 3, 2, 1]:
                tel_name.append(hdulist[0].header['ESO ISS CONF T{0}NAME'.format(iTel)])
            if hdulist[0].header['HIERARCH ESO FT POLA MODE'] == 'COMBINED':
                extver = 20
            else:
                extver = 21
        
            my_stat_dict['ft_pol_mode'].append( hdulist[0].header['HIERARCH ESO FT POLA MODE'] ) 
            # ================
            # Process header
            # ================
    
            dit = hdulist[0].header['HIERARCH ESO DET3 SEQ1 DIT']
            if hdulist[0].header['HIERARCH ESO COU GUID MODE'] in ['FIELD_STAB', 'FIELD_STAB_TCCD']:
                ao_type = 'STRAP'
            elif hdulist[0].header['HIERARCH ESO COU AO SYSTEM'] == 'NAOMI':
                ao_type = 'NAOMI'
            else:
                ao_type = 'CIAO' if hdulist[0].header['HIERARCH ESO COU GUID WAVELEN'] > 1000 else 'MACAO'
             
            my_stat_dict['ao_type'].append( ao_type )
            
            
            # ================
            # Process OPDC
            # ================
    
            # Extract the OPDC data starting after the first FT frame
            opdc = hdulist['OPDC'].data
            opdc_time = opdc['TIME']
            opdc_opd = opdc['OPD']
            opdc_kopd = opdc['KALMAN_OPD']
            opdc_kpiezo = opdc['KALMAN_PIEZO']
            opdc_steps = opdc['STEPS']
            opdc_state = opdc['STATE']
    
            dt = np.median(np.diff(opdc_time))*1e-6
            
            
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
            
            # Extract longuest PD Tracking interval
            intervals = []
            lengths = []
            current_state = 0
            for i, state in enumerate(opdc_state):
                if (current_state == 3) and (state != 3):
                    i1 = i-1
                    intervals.append([i0, i1])
                elif (current_state != 3) and (state == 3):
                    i0 = i
                current_state = state
            if current_state == 3:
                intervals.append([i0, i])
            
            print(intervals)
            
            #now do analysis on valid intervals 
            if len(intervals) > 0 : #if we get a valid interval 
                #print('l242',intervals)
                [iStart, iStop] = intervals[np.argmax(np.diff(intervals, axis=1))]
                print("Longuest tracking interval: {0:.1f} seconds".format((opdc_time[iStop]-opdc_time[iStart])*1e-6))
                
                for iStart, iStop in intervals:
                    if (iStop - iStart) > 4e4: # refer to WHY_I_NEED_MIN_SAMPLING_THRESH.png for why threshold set here 
                        #5e4 was originally set, cahnged to 4e4 since many UT split data failed this threshold
                        
                        psd_dict['no_samples'].append(iStop - iStart )
                        
                        opdc_time_window   = opdc_time  [iStart:iStop]
                        opdc_opd_window    = opdc_opd   [iStart:iStop]
                        opdc_kopd_window   = opdc_kopd  [iStart:iStop]
                        opdc_kpiezo_window = opdc_kpiezo[iStart:iStop]
                        opdc_steps_window  = opdc_steps [iStart:iStop]
                        opdc_mods_window   = opdc_mods  [iStart:iStop]
                    
                        # Reconstruct residuals without phase modulations
                        phase_residuals = (opdc_opd_window - opdc_mods_window + np.pi) % (2.0*np.pi) - np.pi
                    
                        # Reconstruct disturbances
                        phase_disturbances = opdc_kpiezo_window + (opdc_opd_window - (opdc_kopd_window-np.pi)) % (2.0*np.pi) + (opdc_kopd_window-np.pi)
                
                        # Convert phase in [rad] to length in [µm]
                        disturbances = phase_disturbances * 2.25 / (2.*np.pi)
                        residuals    = phase_residuals    * 2.25 / (2.*np.pi)
                        
                        psd_dict['help'].append((targ_tmp, hdulist[0].header['HIERARCH ESO TPL START'], my_stat_dict['ao_type'][-1], my_stat_dict['HIERARCH ESO ISS AMBI TAU0 START'][-1], my_stat_dict['HIERARCH ESO ISS AMBI TAU0 END'][-1],my_stat_dict['HIERARCH ESO ISS AMBI FWHM START'][-1],my_stat_dict['HIERARCH ESO ISS AMBI FWHM END'][-1],target_dict[targ_tmp]['Kmag'],target_dict[targ_tmp]['Vmag']))
                        
                        psd_dict['disturb_ts'].append((opdc_time_window,disturbances))
                        psd_dict['residual_ts'].append((opdc_time_window,residuals ))
                        
                        
                        nperseg = 2**12
                        #opd 
                        f_res, psd_res = sig.welch(residuals, fs=1./dt, nperseg=nperseg, axis=0)
                        
                        #opd (to dampen high freq vib) 
                        f_res_filt, psd_res_filt = sig.welch(residuals, fs=1./dt, nperseg=nperseg, axis=0)
                        #disturbance (puesdo open loop)
                        f_dist, psd_dist = sig.welch(disturbances, fs=1./dt, nperseg=nperseg, axis=0)
                        
                        #see Julien's gravity_vibrations.py
                        freq_xy, Pxy = sig.csd(disturbances, residuals, fs=1./dt, nperseg=nperseg, axis=0)
                        freq_coh, coh = sig.coherence(disturbances, residuals, fs=1./dt, nperseg=nperseg, axis=0)
                        tf = Pxy/psd_dist
                        
                        psd_dict['tf'].append((freq_xy, tf))
                        psd_dict['coherence'].append((freq_coh, coh))
                        
                        
                        
                        # Estimate and remove the photon noise floor (do we need this?)
                        
                        psd_res -= np.median(psd_res, axis=0)
                        psd_res = np.clip(psd_res, 0.0, 1e10)
                        
                        psd_res_filt -= np.median(psd_res_filt, axis=0)
                        psd_res_filt = np.clip(psd_res_filt, 0.0, 1e10)
                        
                        psd_dist -= np.median(psd_dist, axis=0)
                        psd_dist = np.clip(psd_dist, 0.0, 1e10)
                                        
                        
                        #vib filter parameters 
                        rolling_window = 100 #60
                        filter_cut = 8
                        
                        #apply rolling median fiter 
                        #psd_res_filt[f_res_filt > filter_cut] =  pd.DataFrame( psd_res_filt[f_res_filt > filter_cut]).rolling(rolling_window,center=True,min_periods=1,axis=0).median().values
                        #try:
                        #    psd_res_filt[1700:1780] = np.linspace(psd_res_filt[1700], psd_res_filt[1780],1780-1700)
                        #except:
                        #print('EXCEPTION \n')
                        psd_res_filt[f_res_filt > filter_cut] =  pd.DataFrame( psd_res_filt[f_res_filt > filter_cut]).rolling(rolling_window,center=True,min_periods=1,axis=0).median().values
                        
                        #appending results
                        psd_dict['disturb_psd'].append((f_dist,psd_dist))
                        psd_dict['residual_psd'].append((f_res,psd_res))
                        psd_dict['residual_filt_psd'].append((f_res_filt,psd_res_filt))
                        
                        #reverse cumulative 
                        freq_filt = (f_res<500) & (f_res>1) #assume f_res is same as f_dist etc
                        df =  np.nanmean(np.diff(f_res))
                        
                        psd_dist_rcum = df * np.cumsum(psd_dist[freq_filt][::-1,:],axis=0)[::-1,:]
                        psd_res_rcum = df * np.cumsum(psd_res[freq_filt][::-1,:],axis=0)[::-1,:]
                        psd_resf_rcum = df * np.cumsum(psd_res_filt[freq_filt][::-1,:],axis=0)[::-1,:]
                        
                        psd_dict['disturb_rcum'].append((f_dist[freq_filt], psd_dist_rcum))
                        psd_dict['residual_rcum'].append((f_res[freq_filt], psd_res_rcum ))
                        psd_dict['residual_filt_rcum'].append((f_res_filt[freq_filt], psd_resf_rcum))
                        
        
                        #opd std calculated from psd at 1s (1st index with freq_filt = (f_res<500) & (f_res>1))
                        my_stat_dict['opd_std'].append(psd_res_rcum[0]) 
                        #opd std from psd when high freq vibs are dampended
                        my_stat_dict['opd_LPF_std'].append(psd_resf_rcum[0]) 
                #could also do cum psd at some cut off, would be interesting to explore plot stats science int vs OPD


    return(psd_dict, my_stat_dict)
