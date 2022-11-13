
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 14:26:27 2021

@author: bcourtne

edit processing_gravity_OPD script to put in offline to process new fits files and upload to datalabs 
input gravity file 

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

def get_gravity_residuals(grav_file_path):
    """
    calculate the gravity fringe tracker closed and pseudo open loop residuals (unwrapped - both timeseries and psds) 
    does this on longest cont. period of fringe tracking (without phase slips) 
    
    input:
        gravity file path 
    output:
        gravity_dict - with closed and pseudo open loop residual timeseries (timestamped) as well as psds for all baselines 
    
    """
    
    nperseg = 2**11 
    
    gravity_dict = {} 
    
    with fits.open( grav_file_path ) as hdulist:


        # Extract the OPDC data starting after the first FT frame
        opdc = hdulist['OPDC'].data
        opdc_time = opdc['TIME']
        opdc_opd = opdc['OPD']
        opdc_kopd = opdc['KALMAN_OPD']
        opdc_kpiezo = opdc['KALMAN_PIEZO']
        opdc_steps = opdc['STEPS']
        opdc_state = opdc['STATE']

        tpl_start = datetime.datetime.strptime(hdulist[0].header['HIERARCH ESO TPL START'], '%Y-%m-%dT%H:%M:%S')

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
        i0=0
        for i, state in enumerate(opdc_state):
            if (current_state == 3) and (state != 3):
                i1 = i-1
                intervals.append([i0, i1])
            elif (current_state != 3) and (state == 3):
                i0 = i
            current_state = state
        if current_state == 3:
            intervals.append([i0, i])
        
        print(f'intevals = {intervals}')
        iStart, iStop = intervals[ np.where(np.max(np.diff(intervals)))[0][0] ]
        """###  IF I WANT FULL TIMESERIES!!!!
        iStart, iStop = 0, len(opdc_time) #intervals[ np.where(np.max(np.diff(intervals)))[0][0] ]
        ##"""
        print( f'-----\nlongest fringe tracking inteval without phase slips = {1e-6*dit*(iStop - iStart)}s\n-----')

        # get timestamps for this interval 
        timestamps = [tpl_start + datetime.timedelta(microseconds = float(opdc_time  [iStart:iStop][i]) ) for i in range(iStop-iStart)]

        opdc_time_window   = opdc_time  [iStart:iStop]
        opdc_opd_window    = opdc_opd   [iStart:iStop]
        opdc_kopd_window   = opdc_kopd  [iStart:iStop]
        opdc_kpiezo_window = opdc_kpiezo[iStart:iStop]
        opdc_steps_window  = opdc_steps [iStart:iStop]
        opdc_mods_window   = opdc_mods  [iStart:iStop]

        # timestamps (calculated from template start time)
        gravity_dict['timestamp'] = timestamps
        # closed loop residual (phase)
        gravity_dict['phase_residual'] = (opdc_opd_window - opdc_mods_window + np.pi) % (2.0*np.pi) - np.pi

        # pseudo open loop residual (phase) 
        gravity_dict['phase_disturbance'] = opdc_kpiezo_window + (opdc_opd_window - (opdc_kopd_window-np.pi)) % (2.0*np.pi) + (opdc_kopd_window-np.pi)

        # Convert phase in [rad] to length in [µm]
        gravity_dict['disturbance'] = gravity_dict['phase_disturbance'] * 2.25 / (2.*np.pi)
        gravity_dict['residual']    = gravity_dict['phase_residual']    * 2.25 / (2.*np.pi)

        gravity_dict['psd_disturbance'] = sig.welch(gravity_dict['disturbance'] , fs=1./dt, nperseg=nperseg, axis=0)
        gravity_dict['psd_residual'] = sig.welch(gravity_dict['residual'] , fs=1./dt, nperseg=nperseg, axis=0)
        
        
        # create fits file 
        fits_out = fits.HDUList([fits.PrimaryHDU(  ), fits.PrimaryHDU( )])
        fits_out[0].header = hdulist[0].header 
        fits_out[1].header['COL 1'] = 'timestamp' 
        fits_out[1].header['COL 2'] = 'gravity OPD closed loop residual (um)' 
        fits_out[1].header['COL 3'] = 'gravity OPD pseudo open loop residual (um)' 
        fits_out[1].data = pd.DataFrame(gravity_dict)[['timestamp', 'psd_residual', 'psd_disturbance']]

        
        
    return(gravity_dict) 






