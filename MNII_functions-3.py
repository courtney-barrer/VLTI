# -*- coding: utf-8 -*-
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

"""
Benjamin Courtney-Barrer
Functions for processing and analysing Manhattan II data (MNII) 
"""

import numpy as np
import h5py
import os
import matplotlib.pyplot as plt 
import scipy.signal as sig
#import emcee
from scipy.stats import norm
import pandas as pd 
import glob
from statsmodels.tsa.stattools import adfuller
import datetime 
from scipy.interpolate import interp1d
import dlt 
from astropy.io import fits
from scipy.stats import moment


#mapping from sensor indices to mirror position 
i2m_dict = {1:'M3a',2:'M3b',3:'M2',4:'empty',5:'M1+y',6:'M1-x',7:'M1-y',8:'M1+x',9:'M4',10:'M5',11:'M6',12:'M7'} 
m2i_dict = {v: k for k, v in i2m_dict.items()}

# gravity baseline to telescope mapping 
baselabels = ['43','42','41','32','31','21']
base2telname = [[4, 3], [4, 2], [4, 1], [3, 2], [3, 1], [2, 1]]
tel2telname = [4, 3, 2, 1]
base2tel = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
tel2base = [[0, 1, 2], [0, 3, 4], [1, 3, 5], [2, 4, 5]]
M_matrix = np.array([[1.0, -1.0, 0.0, 0.0],
                     [1.0, 0.0, -1.0, 0.0],
                     [1.0, 0.0, 0.0, -1.0],
                     [0.0, 1.0, -1.0, 0.0],
                     [0.0, 1.0, 0.0, -1.0],
                     [0.0, 0.0, 1.0, -1.0]])

V2acc_gain = 0.01  # from MNII amplifiers
nperseg = 2**11 #for PSDs

# path to where manhattan files are stored on ESO datalake
vib_path = '/home/jovyan/datalake/rawdata/vlt-vibration/{year}/{month:02}/ldlvib{ut}_raw_{year}-{month:02}-{day:02}.hdf5'




def process_daily_MN2_file(file, verbose=True, post_upgrade=True):
    """
    file name - string (e.g. 'ldlvib{UT}_raw_{yyyy}-{mm}-{dd}.hdf5') 
    ... note needs to be raw file and not psd 
    
    returns dictionary of acceleration from each system (sensor/mirrors etc) in matrix , coloumns is time, row is sample
    """
    mn2_data = h5py.File(file, 'r')
    master_acc = {}
    
    if not post_upgrade: # only consider sensors installed up to m3
        
        for i, time_key in enumerate( mn2_data.keys() ):
            acc = {}
            for _,s in enumerate([1,2,3,5,6,7,8]):
                acc[f'sensor_{s}'] = mn2_data[time_key][f'sensor{s}'][:]

            acc['m3'] = -V2acc_gain *(mn2_data[time_key]['sensor1'][:] + mn2_data[time_key]['sensor2'][:] )/2.0
            acc['m2'] = -V2acc_gain *mn2_data[time_key]['sensor3'][:] 
            acc['m1'] = +V2acc_gain *(mn2_data[time_key]['sensor5'][:] + mn2_data[time_key]['sensor6'][:] \
                            + mn2_data[time_key]['sensor7'][:] + mn2_data[time_key]['sensor8'][:] )/4.0

            # Convert to piston, based on geometry
            acc['m1'] *= 2.0
            acc['m2'] *= 2.0
            acc['m3'] *= np.sqrt(2.0)

            # combined geometry up to m3
            acc['m1-3'] = acc['m1'] + acc['m2'] + acc['m3']

            for key in acc:
                acc[key]= list(acc[key])

                if len(acc[key]) == 10000: # 10s at 1kHz sampling = 10000 samples.. ensure this! 
                    if i==0:
                        master_acc[key] = [acc[key]]
                    else:
                        master_acc[key].append( acc[key] )

                else: #missing samples .. for now we ignore these samples and set to np.nan array (we don't know which samples are missing, beginning or end? 

                    if i==0:
                        master_acc[key] = [list(np.nan * np.empty(10000))]
                    else:
                        master_acc[key].append( list(np.nan * np.empty(10000)) )
                        if verbose:
                            print(f'went wrong for {key}, {time_key} ,where master_acc[key].shape = { len(master_acc[key]) } and acc[key].shape = { len( acc[key] )}')

        return(master_acc) 

    elif post_upgrade: # consider sensors installed up to m7
        
        for i, time_key in enumerate( mn2_data.keys() ):
            acc = {}

            for _,s in enumerate([1,2,3,4,5,6,7,8,9,10,11,12]):
                acc[f'sensor_{s}'] = mn2_data[time_key][f'sensor{s}'][:]

            acc['m3'] = -V2acc_gain *(mn2_data[time_key]['sensor1'][:] + mn2_data[time_key]['sensor2'][:] )/2.0
            acc['m2'] = -V2acc_gain *mn2_data[time_key]['sensor3'][:] 
            acc['m1'] = +V2acc_gain *(mn2_data[time_key]['sensor5'][:]  + mn2_data[time_key]['sensor6'][:] \
                            + mn2_data[time_key]['sensor7'][:]  + mn2_data[time_key]['sensor8'][:] )/4.0
            acc['m4'] = -V2acc_gain *mn2_data[time_key]['sensor9'][:] 
            acc['m5'] = -V2acc_gain *mn2_data[time_key]['sensor10'][:] 
            acc['m6'] = -V2acc_gain *mn2_data[time_key]['sensor11'][:] 
            acc['m7'] = -V2acc_gain *mn2_data[time_key]['sensor12'][:] 

            # Convert to piston, based on geometry
            acc['m1'] *= 2.0
            acc['m2'] *= 2.0
            acc['m3'] *= np.sqrt(2.0)
            acc['m4'] *= np.sqrt(2.0)
            acc['m5'] *= 1.9941 # valued checked with Salman on Aug 13 2022
            acc['m6'] *= 1.8083 # valued checked with Salman on Aug 13 2022
            acc['m7'] *= 1.9822 # valued checked with Salman on Aug 13 2022

            # combined geometry 
            acc['m1-3'] = acc['m1'] + acc['m2'] + acc['m3']
            acc['m4-7'] = acc['m4'] + acc['m5'] + acc['m6'] + acc['m7']   
            acc['m1-7'] = acc['m1'] + acc['m2'] + acc['m3'] + acc['m4'] + acc['m5'] + acc['m6'] + acc['m7']



            for key in acc:
                acc[key]= list(acc[key])

                if len(acc[key]) == 10000: # 10s at 1kHz sampling = 10000 samples.. ensure this! 
                    if i==0:
                        master_acc[key] = [acc[key]]
                    else:
                        master_acc[key].append( acc[key] )

                else: #missing samples .. for now we ignore these samples and set to np.nan array (we don't know which samples are missing, beginning or end? 

                    if i==0:
                        master_acc[key] = [list(np.nan * np.empty(10000))]
                    else:
                        master_acc[key].append( list(np.nan * np.empty(10000)) )
                        if verbose:
                            print(f'went wrong for {key}, {time_key} ,where master_acc[key].shape = { len(master_acc[key]) } and acc[key].shape = { len( acc[key] )}')

    return(master_acc) 


def basic_filter(data, outlier_thresh, replace_value  ) :
    # basic filter to replace any absolute values in data > outlier_thresh with replace_value 
    
    if not (outlier_thresh is None):
                
        #sensor_tmp = mn2_data[time_key][f'sensor{s}'][:].copy()
        # get indicies where abs sensor values are above user 'outlier threshold'
        outlier_indx = abs( data ) > outlier_thresh 

        # replace outliers with users 'replace_value' 
        data[outlier_indx] = replace_value

        # how many outliers are replaced? 
        no_replaced = [np.sum(outlier_indx)] 
                
    else:
        no_replaced  = [0]     
        
    return(data, no_replaced) 

        

def process_single_mn2_sample(file, time_key, post_upgrade=True, user_defined_geometry=None, outlier_thresh = None, replace_value=0, ensure_1ms_sampling=False):
    """
    file name - string (e.g. 'ldlvib{UT}_raw_{yyyy}-{mm}-{dd}.hdf5') 
    ... note needs to be raw file and not psd 
    time_key - string (e.g 'hh:mm:ss' ) 
    post_upgrade - boolean: do we consider new accelerometers 
    user_defined_geometry - list with strings for user defined geometry for mirror piston combination
        e.g. to combine m1, m2, m3 and m5 the user input user_defined_geometry = ['m1', 'm2', 'm3' and 'm5']
    
    outlier_thresh = float or None , if not None then any absolute raw sensor value above outlier_thresh will be 
    replaced with replace_value (default is np.nan in case that replace_value = None
    
    replace_value = float, int etc 
    
    ensure_1ms_sampling = boolean
        if True we manually check that each sample contains exactly 10k samples (1ms sampling), if not then we set the data to np.nan. This is importanat for merging multiple samples
    """
    
    mn2_data = h5py.File(file, 'r')
    acc = {}
    
    if not post_upgrade: #only consider sensors installed up to m3 
        
        for _,s in enumerate([1,2,3,5,6,7,8]):
            
            # start to fill acc dictionary , 
            # these sensor values should NOT be scaled or manipulated!!! since later functions depend on raw value
            acc[f'sensor_{s}'] = mn2_data[time_key][f'sensor{s}'][:]
            
            #acc[f'sensor_{s}'], acc[f'sensor_{s}_no_outliers_replaced'] = basic_filter(data, outlier_thresh, replace_value  ) 
            
            """
            if not (outlier_thresh is None):
                
                #sensor_tmp = mn2_data[time_key][f'sensor{s}'][:].copy()
                # get indicies where abs sensor values are above user 'outlier threshold'
                outlier_indx = abs( acc[f'sensor_{s}'] ) > outlier_thresh 
                
                # replace outliers with users 'replace_value' 
                acc[f'sensor_{s}'][outlier_indx] = replace_value
                    
                # how many outliers are replaced? 
                acc[f'sensor_{s}_no_outliers_replaced'] = [np.sum(outlier_indx)] 
                
            else:
                acc[f'sensor_{s}_no_outliers_replaced'] = [0]     

            """
                
        acc['m3'] = -V2acc_gain *(mn2_data[time_key]['sensor1'][:] + mn2_data[time_key]['sensor2'][:] )/2.0
        acc['m2'] = -V2acc_gain *mn2_data[time_key]['sensor3'][:]
        
        acc['m1'] = +V2acc_gain *(mn2_data[time_key]['sensor5'][:]  + mn2_data[time_key]['sensor6'][:] \
                        + mn2_data[time_key]['sensor7'][:]  + mn2_data[time_key]['sensor8'][:] )/4.0


        # Convert to piston, based on geometry
        acc['m1'] *= 2.0
        acc['m2'] *= 2.0
        acc['m3'] *= np.sqrt(2.0)

        #combined geometry up to m3
        acc['m1-3'] = acc['m1'] + acc['m2'] + acc['m3']

        tmp_keys = list(acc.keys()) # so we don't iterate on dict keys while changing its keys
        for key in tmp_keys:
            acc[key], acc[f'{key}_no_outliers_replaced'] = basic_filter(acc[key], outlier_thresh, replace_value  ) 
            acc[key]=list(acc[key])

        if ensure_1ms_sampling:
            for key in tmp_keys:
                if len( acc[key] ) != 10000:
                    acc[key] = list(np.nan * np.empty(10000))
                    
                    
        return(acc) 
    
    elif post_upgrade: # consider sensors installed up to m7
        
        for _,s in enumerate([1,2,3,5,6,7,8,9,10,11,12]):
            
            # start to fill acc dictionary , 
            # these sensor values should NOT be scaled or manipulated!!! since later functions depend on raw value
            acc[f'sensor_{s}'] = mn2_data[time_key][f'sensor{s}'][:]
            
            """
            if not (outlier_thresh is None):
                
                #sensor_tmp = mn2_data[time_key][f'sensor{s}'][:].copy()
                # get indicies where abs sensor values are above user 'outlier threshold'
                outlier_indx = abs( acc[f'sensor_{s}'] ) > outlier_thresh 
                
                # replace outliers with users 'replace_value' 
                acc[f'sensor_{s}'][outlier_indx] = replace_value
                    
                # how many outliers are replaced? 
                acc[f'sensor_{s}_no_outliers_replaced'] = [np.sum(outlier_indx)]
                
            else:
                acc[f'sensor_{s}_no_outliers_replaced'] = [0]     
            """

        acc['m3'] = -0.01*(mn2_data[time_key]['sensor1'][:] + mn2_data[time_key]['sensor2'][:] )/2.0
        acc['m2'] = -0.01*mn2_data[time_key]['sensor3'][:]
        
        acc['m1'] = +0.01*(mn2_data[time_key]['sensor5'][:]  + mn2_data[time_key]['sensor6'][:] \
                        + mn2_data[time_key]['sensor7'][:]  + mn2_data[time_key]['sensor8'][:] )/4.0




        # STILL NEED to include geometric factor for upgrade! 

        acc['m4'] = -V2acc_gain *mn2_data[time_key]['sensor9'][:] 
        acc['m5'] = -V2acc_gain *mn2_data[time_key]['sensor10'][:]
        acc['m6'] = -V2acc_gain *mn2_data[time_key]['sensor11'][:] 
        acc['m7'] = -V2acc_gain *mn2_data[time_key]['sensor12'][:]

        # Convert to piston, based on geometry
        acc['m1'] *= 2.0
        acc['m2'] *= 2.0
        acc['m3'] *= np.sqrt(2.0)
        acc['m4'] *= np.sqrt(2.0)
        acc['m5'] *= 1.9941 # valued checked with Salman on Aug 13 2022
        acc['m6'] *= 1.8083 # valued checked with Salman on Aug 13 2022
        acc['m7'] *= 1.9822 # valued checked with Salman on Aug 13 2022

        #combined geometry 
        acc['m1-3'] = acc['m1'] + acc['m2'] + acc['m3']
        acc['m4-7'] = acc['m4'] + acc['m5'] + acc['m6'] + acc['m7']   
        acc['m1-7'] = acc['m1'] + acc['m2'] + acc['m3'] + acc['m4'] + acc['m5'] + acc['m6'] + acc['m7']
        
        #user defined geometry 
        if not (user_defined_geometry is None):
            acc['custom_geom'] = sum([acc[mm] for mm in user_defined_geometry])
        
        tmp_keys = list(acc.keys()) # so we don't iterate on dict keys while changing its keys
        for key in tmp_keys:
            acc[key], acc[f'{key}_no_outliers_replaced'] = basic_filter(acc[key], outlier_thresh, replace_value  ) 
            acc[key]=list(acc[key])
            
            
        if ensure_1ms_sampling:
            for key in tmp_keys:
                if len( acc[key] ) != 10000:
                    acc[key] = list(np.nan * np.empty(10000))
                
            
        return(acc) 

def opd_rms( f, opd_psd , bandwidth = [0,1e4]):
    # calculate the opd rms from OPD (or OPL) PSD , f is frequency and opd_psd is the psd
    mask = (f <= bandwidth [1]) & (f >= bandwidth [0])
    opd_rms = np.sqrt( np.sum( opd_psd[mask] * np.diff(f[mask])[1] ) )
    return(opd_rms) 




def format_number(x):
    # a function to add zeros infront of numbers that are less then 10 
    # this is used to format numbers in dates 
    if x < 10:

        return('0'+str(x))

    elif x >= 10: 

        return(str(x))

    else:
        print('something went wrong')


def get_focus(time, focus_periods):
    # get the focus (Nas A, Nas B etc) at a given time. focus_periods should be the df output 
    #dlt.query_focus_times(env, initial_time, final_time, names=False) and time a datetime object
    
    mask = (time <= focus_periods['final_time']) & (time >= focus_periods['initial_time'] )
    if sum(mask):
        f = focus_periods['focus_name'][mask].values[0]
    else: #telescope was not in any defined focus
        f = np.nan
    
    return(f)



def acc_psd(acc_ts):
    #standard psd from 1khz MN2 sample 

    f,psd = sig.welch(acc_ts, fs=1e3, nperseg=nperseg, axis=0)
    return(f,psd )

def double_integrate(f, acc_psd):
    #double integrate PSD in freq domain 
    
    psd_pos = 1/(2*np.pi)**4 * acc_psd[1:] / f[1:]**4
    
    return( f[1:], psd_pos )

def double_integrate_cutoff(f,acc_psd,hc):
    #double integrate PSD in freq domain 
    idx = np.where(f < hc)
    acc_psd[idx] = 0
    psd_pos = 1/(2*np.pi)**4 * acc_psd[1:] / f[1:]**4
    
    return( f[1:], psd_pos )


def is_between(time, df):
    if sum( (time <= df['final_time']) & (time >= df['initial_time'] ) ): # if time is between at least 1 initial and final time columns
        return(1)
    else:
        return(0)


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
        for i, state in enumerate(opdc_state):
            if (current_state == 3) and (state != 3):
                i1 = i-1
                intervals.append([i0, i1])
            elif (current_state != 3) and (state == 3):
                i0 = i
            current_state = state
        if current_state == 3:
            intervals.append([i0, i])

        iStart, iStop = intervals[ np.where(np.max(np.diff(intervals)))[0][0] ]

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

    return(gravity_dict) 




def gravity_vs_mn2(grav_file_path , baseline , sensor_1 = 'm1-7', sensor_2 = 'm1-7', plot = True , save=False ,verbose=True,outlier_thresh = None, replace_value=np.nan, scale_raw_sensors=True):  

    """
    compares gravity FT residuals on a given (UT) baseline to nearest MNII data on each respective telescope (defined by baseline) 
    for the user defined sensors (sensor_1 and sensor_2). Standard sensor options are are mirror piston recombinations 'm1', 'm2'...'m7'
    or piston along various mirrors 'm1-3', 'm4-7', 'm1-7' or individual sensors(e.g. 'sensor_1', 'sensor_2', ..'sensor_12' )
    !!!Note: raw sensor values (e.g 'sensor_X' are not scaled correctly - so better to use mirrors (e.g. 'mX') when comparing to
    gravity!!!
    Users can also define custom geometry by input list of mirror strings. e.g. to combine m1, m2, m3 and m5 on telescope 1 the user
    should input sensor_1 = ['m1', 'm2', 'm3' and 'm5']
    
    input:
        grav_file_path - string with the FULL path to gravity fits file 
        baseline - string containing UT telescopes that make up baseline (e.g. input '21' for UT2-UT1 baseline)
        sensor_1 = string or list of strings - which sensor or piston reconstruction to consider for telescope 1 (defined by 1st number
        in baseline string)?
        sensor_2 = string or list of strings - which sensor or piston reconstruction to consider for telescope 2 (defined by 2nd number
        in baseline string)?
        
        plot = boolean - do we want to plot the FT pseudo open loop to nearest MNII samples for given sensors?
        save = boolean - do we want to save this plot? 
        scale_raw_sensors = boolean - for indivual sensors, when plotting do we want to scale (calibrate) to m/s^2 units? 
        
    output: 
        master_dict = { 'gravity':gravity_dict, f'UT{T1}':T1_dict, f'UT{T2}':T2_dict }
        
        where:
            gravity_dict - dictionary holding FT closed and pseudo open loop time series, PSDs and timestamps for given input file
            T1_dict - dictionary holding *sensor_1* accelerometer time series, PSD, OPL PSD and timedifference to middle of gravity obs
            T2_dict - dictionary holding *sensor_2* accelerometer time series, PSD, OPL PSD and timedifference to middle of gravity obs

        and of course the plot if desired (plot = True). 
    
    """
    
    
    
    gravity_dict = get_gravity_residuals(grav_file_path)

    # get year, month, day of file timestamp (to readin corresponding MNII daily file) 
    grav_file_date = grav_file_path.split('/')[-1].split('.')[1].split('T')[0]

    file_year = grav_file_path.split('/')[-1].split('.')[1].split('-')[0]
    file_month = grav_file_path.split('/')[-1].split('.')[1].split('-')[1] 
    file_day = grav_file_path.split('/')[-1].split('.')[1].split('-')[2].split('T')[0]

    # gravity timestamp to match closet MNII sample
    grav_timestamp_2_match = gravity_dict['timestamp'][len(gravity_dict['timestamp'])//2]
    

    # get corresponding UT telescopes from user input baseline
    T1 = int(baseline[0])
    T2 = int(baseline[1])

    mn2_file_1 = f'/datalake/rawdata/vlt-vibration/{file_year}/{file_month}/ldlvib{T1}_raw_{file_year}-{file_month}-{file_day}.hdf5'
    mn2_file_2 = f'/datalake/rawdata/vlt-vibration/{file_year}/{file_month}/ldlvib{T2}_raw_{file_year}-{file_month}-{file_day}.hdf5'

    #read in daily MNII file matching gravity file timestamp 
    mn2_data_1 = h5py.File(mn2_file_1, 'r')
    mn2_data_2 = h5py.File(mn2_file_2, 'r')

    mn2_timestamps_1 = np.array([datetime.datetime.strptime(grav_file_date +'T'+t ,'%Y-%m-%dT%H:%M:%S') for t in mn2_data_1.keys()])
    mn2_timestamps_2 = np.array([datetime.datetime.strptime(grav_file_date +'T'+t ,'%Y-%m-%dT%H:%M:%S') for t in mn2_data_2.keys()])

    time_key_1 = list(mn2_data_1.keys())[np.argmin( abs( grav_timestamp_2_match - mn2_timestamps_1) )]
    time_key_2 = list(mn2_data_2.keys())[np.argmin( abs( grav_timestamp_2_match - mn2_timestamps_2) )]

    deltatime_1 = np.min( abs( grav_timestamp_2_match - mn2_timestamps_1) )
    deltatime_2 = np.min( abs( grav_timestamp_2_match - mn2_timestamps_2) )
    

    print(f'\n-----\nUT{T1} closest MNII data is {mn2_file_1}\n at timestamp {time_key_1} which has time difference to center of gravity FT observations  = {deltatime_1}'  ) 
    print(f'\n-----\nUT{T2} closest MNII data is {mn2_file_2}\n at timestamp {time_key_2} which has time difference to center of gravity FT observations  = {deltatime_2}'  ) 


    #process the nearest MNII files to the respective gravity FT observation.  
    
    if isinstance(sensor_1, str): # then standard geometry automatically processed in 'process_single_mn2_sample' function
        acc_dict_1 = process_single_mn2_sample(mn2_file_1, time_key_1, post_upgrade=True, user_defined_geometry=None,\
                                              outlier_thresh=outlier_thresh, replace_value=replace_value)
        sensor_1_key = sensor_1
    elif isinstance(sensor_1, list): # then custom geometry needs to be considered 
        acc_dict_1 = process_single_mn2_sample(mn2_file_1, time_key_1, post_upgrade=True, user_defined_geometry=sensor_1,\
                                              outlier_thresh=outlier_thresh, replace_value=replace_value)
        sensor_1_key = 'custom_geom'
        
    if isinstance(sensor_2, str):  
        acc_dict_2 = process_single_mn2_sample(mn2_file_2, time_key_2, post_upgrade=True, user_defined_geometry=None,\
                                              outlier_thresh=outlier_thresh, replace_value=replace_value)    
        sensor_2_key = sensor_2
    elif isinstance(sensor_2, list): # then custom geometry needs to be considered 
        acc_dict_2 = process_single_mn2_sample(mn2_file_2, time_key_2, post_upgrade=True, user_defined_geometry=sensor_2,\
                                              outlier_thresh=outlier_thresh, replace_value=replace_value)
        sensor_2_key = 'custom_geom'

    # calculate OPL PSD from each MNII file for the user defined sensor
    nperseg = 2**11
    
    acc_psd_1 = sig.welch(acc_dict_1[sensor_1_key], fs=1e3, nperseg=nperseg, axis=0)
    pos_psd_1 = double_integrate(*acc_psd_1)

    acc_psd_2 = sig.welch(acc_dict_2[sensor_2_key], fs=1e3, nperseg=nperseg, axis=0)
    pos_psd_2 = double_integrate(*acc_psd_2)

    T1_dict = {'acc':acc_dict_1[sensor_1_key], 'acc_psd':acc_psd_1, 'pos_psd':pos_psd_1, 'deltatime_1':deltatime_1} 
    T2_dict = {'acc':acc_dict_2[sensor_2_key], 'acc_psd':acc_psd_2, 'pos_psd':pos_psd_2, 'deltatime_1':deltatime_2} 

    master_dict = { 'gravity':gravity_dict, f'UT{T1}':T1_dict, f'UT{T2}':T2_dict }

    if plot:
        fig,axx= plt.subplots(1,1,sharey=True, sharex=True,figsize=(15,6))
        axx.set_title(grav_file_path.split('/')[-1])

        # find baseline index in gravity PSD array 
        B_indx = np.where( f'{max([T1,T2])}{min([T1,T2])}' == np.array(baselabels) )[0][0]

        #gravity FT OPD pseudo open loop psd 
        axx.loglog(gravity_dict['psd_disturbance'][0], gravity_dict['psd_disturbance'][1][:,B_indx], color='grey' ,linestyle='--', label=f'Gravity OPD pseudo open loop (UT-{baselabels[B_indx]})',alpha=0.8)
        #gravity FT OPD closed loop
        axx.loglog(gravity_dict['psd_residual'][0], gravity_dict['psd_residual'][1][:,B_indx], color='k' ,linestyle='-', label=f'Gravity OPD closed loop (UT-{baselabels[B_indx]})',alpha=0.8)
        #reverse cumulative
        axx.loglog(gravity_dict['psd_residual'][0], np.diff(gravity_dict['psd_residual'][0])[1] * (np.cumsum(gravity_dict['psd_residual'][1][:,B_indx][::-1])[::-1]), color='k' ,linestyle='--', label=f'Gravity OPD closed loop reverse cumulative',alpha=0.8)
        #100nm, 200nm 
        axx.axhline(0.1**2,linestyle=':',color='lime',label='100nm')
        axx.axhline(0.2**2,linestyle=':',color='green',label='200nm')
        
        
        #MNII sensor OPL PSDs 
        
        
        # Sensor 1

        if 'sensor' in sensor_1:
            
            #sensor number index
            s1_no = int( sensor_2.split('_')[-1] )
            # convert to meaningful position label
            lab_tmp = i2m_dict[ s1_no ]
            # warn that they are plotting an individual accelerometer
            print('\n---\nNote that you are plotting data on {} for an individual accelerometer ({}),\nthere is no geometric correction factor applied here even if scale_raw_sensors=True\n'.format(T1,lab_tmp ))
            if scale_raw_sensors: # we convert V^2/Hz to m^/s^4/Hz 
                axx.loglog(pos_psd_1[0], 1e12 * V2acc_gain * pos_psd_1[1], color='red' , label=f'UT{T1} OPL - {lab_tmp}',alpha = 0.6) 
            else: # leave in units V^2/Hz 
                axx.loglog(pos_psd_1[0], 1e12 * pos_psd_1[1], color='red' , label=f'UT{T1} OPL - {lab_tmp}',alpha = 0.6)
        else: # sensor_1 is some combined geometry
            axx.loglog(pos_psd_1[0], 1e12 * pos_psd_1[1], color='red' , label=f'UT{T1} OPL - {sensor_1}',alpha = 0.6)
 
        # Sensor 2
    
        if 'sensor' in sensor_2:
            
            #sensor number index
            s2_no = int( sensor_2.split('_')[-1] )
            # convert to meaningful position label
            lab_tmp = i2m_dict[ s2_no ] 
            # warn that they are plotting an individual accelerometer
            print('\n--- Note that you are plotting data on {} for an individual accelerometer ({}),\nthere is no geometric correction factor applied here even if scale_raw_sensors=True\n'.format(T2,lab_tmp ))
            
            if scale_raw_sensors: # we convert V^2/Hz to m^/s^4/Hz 
                axx.loglog(pos_psd_2[0], 1e12 * V2acc_gain * pos_psd_2[1], color='blue' , label=f'UT{T2} OPL - {lab_tmp}',alpha = 0.6) 
            else: # leave in units V^2/Hz 
                axx.loglog(pos_psd_2[0], 1e12 * pos_psd_2[1], color='blue' , label=f'UT{T2} OPL - {lab_tmp}',alpha = 0.6)
        else: # sensor_1 is some combined geometry
            axx.loglog(pos_psd_2[0], 1e12 * pos_psd_2[1], color='blue' , label=f'UT{T2} OPL - {sensor_2}',alpha = 0.6)

        #closed loop opd rms of gravity ft in given baseline (um)
        closed_loop_opd = opd_rms(gravity_dict['psd_residual'][0],gravity_dict['psd_residual'][1][:,B_indx]) 
        
        axx.text(1e1,1e2, 'gravity closed loop OPD RMS = {}nm'.format(1e3*round(closed_loop_opd,3)) ,fontsize=12)
        axx.grid()
        axx.legend(bbox_to_anchor=(1,1),fontsize=15)
        axx.set_xlabel('frequency [Hz]',fontsize=15)
        axx.set_ylabel(r'PSD [$\mu m^2$/Hz]'+'\nreverse cumulative '+r'[$\mu m^2$]',fontsize=15)
        axx.tick_params(labelsize=15) 
        axx.set_xlim([0.5, 500])
        axx.set_ylim([1e-8, 5e2])
        #for i, (B_index,axx) in enumerate(zip(mn2.baselabels, ax.reshape(-1))):
        plt.tight_layout()

        if save: 
            raw_file = grav_file_path.split('/')[-1]
            plt.savefig(f'/home/jovyan/usershared/BCB/mn2_upgrade/gravity_v_MN2_UT{baseline}_{raw_file}-MN2_{sensor_1}-MN2_{sensor_2}.png')

        plt.show()

    return( master_dict ) 




def classify_freq_bin( f,psd, quantiles, opl_thresh, freq_lims = [5, np.inf] ):
    """
    Classify new opl psd (um^2/Hz) based on historic quantiles 
    - f is float (freq value of psd. e.g. psd(f)) 
    - psd is float, 
    - quantile is pandas series with index = ['q10','q20','q30','q40','q50','q60','q70','q80','q90'] quantiles indeally from a
    f'/home/jovyan/usershared/BCB/UT{UT}/operations_{sensor}_distribution_features.csv' file
    - opl_thresh is a user defined opl threshold that we activate an alarm (pink classification) 
    """
    if (psd >= quantiles['q10']) & (psd <= quantiles['q90']) :
        c = 'silver'
    elif (psd < quantiles['q10']):
        c='lime'
    elif (psd > quantiles['q90'])& (not psd > opl_thresh):
        c='orange' 
    elif (psd > quantiles['q90']) & (psd > opl_thresh): 
        if (f>freq_lims[0]) & (f<freq_lims[-1]): # if within valid freq bins
            c='red'
        else:
            c='orange'
    else:
        print('... we"re missing a case here') 
        
    return(c)

def get_flag_frequencies_indicies( freq_classification, flag_color='red' ): 
    """
    get frequencies that correspond to the input flag color. 
    input
        freq_classification - panda series with classifications, indexed by frequency 
        flag_color - what flag we're looking for 
    output 
        np.array of frequencies correpsonding to flag color 
    """
        
    f = np.array(list(freq_classification.index))
    
    return( f[np.where(freq_classification.values == flag_color)] )

    
                        
def _get_psd_reference_file(reference_dir, UT_name,state_name, focus_name, sensor) :
        
    # Get the reference file that corresponds to the input file state / sensor to compare PSD statistics 
    psd_reference_file = glob.glob(reference_dir + f'{UT_name}/*{UT_name}*_{state_name}_*{focus_name}*{sensor}_psd_distribution_features.csv') 

    if len(psd_reference_file)==0:
        raise TypeError(f'\n--------\nNo psd reference file found in {reference_dir} for {UT_name} for {state_name}, {focus_name}, {sensor}.\n\
        You should put one there or make sure naming convention is correct!\n--------\n\n')

    elif len(psd_reference_file)>1:

        # pick the most recent one
        psd_reference_file = [ np.sort( psd_reference_file )[-1] ]

        print(f'\n-------\nThere were multiple reference files in {reference_dir} for {UT_name} for {state_name} state, focus={focus_name}, sensor={sensor}.\
        \n We will pick the most recent which is {psd_reference_file}') 

    print(f'\ncomparing to input MNII data to:\n   {psd_reference_file}')

    return( psd_reference_file )                       
                        

def MNII_classification(file, time_key, opl_thresh, sensors='all', reference_dir = '/home/jovyan/usershared/BCB/MNII_PROCESSED_DATA/REFERENCE_PSD_STATISTICS/' , post_upgrade=False, plot=True , verbose = False):    
    """
    
    !!!!!!!!!!
    NOTE : at the moment this only classifies based on historic values while in operations.. later we may consider other states
    therefore do not pay too much attention to results if input file was not in operational state (e.g. guiding with encolure open) 
    !!!!!!!!!!
    
    
    input:
        file - string (e.g.  f'/datalake/rawdata/vlt-vibration/2022/08/ldlvib{UT}_raw_2022-08-19.hdf5') 
        time_key - what time do we want to look at in the file (e.g. '03:29:00'.. note always rounded to nearest minute)
        sensors - string or list , what sensors do we want to classify? 'all' for all of them (note m4-m7 sensors only available if 
        opl_thresh - float, what threshold do we want to put on the PSD OPL (units um^2/Hz) to trigger red flag (alarm classification?)
        reference_dir - string, directory where reference PSD distribution features are 
        post_upgrade = True), or list of sensors strings e.g. ['sensor_1', 'm1', 'm1-3']
        post_upgrade = boolean - do we want to consider the new (m4-m7) MNII sensors? 
        plot = boolean - do we want to plot results (PSDs) 
        verbose = boolean - do you want to print some details of whats happening? 
        
    output: 
        freq_classifications - pandas series indexed by frequency with the PSD frequency binned classification 
            - 'silver' = nomial (between 10-90th percentiles) 
            - 'green' = better then usual (<10th percentile) 
            - 'orange' = worse then usual (>90th percentile) 
            - 'red' = alarm, redflag! (>90th percentile & PSD(f) > opl_thresh)
            
        
            
    """
    
    # extract UT# from file name
    UT = file.split('ldlvib')[-1].split('_')[0]
    UT_name = f'UT{UT}'
    #file data
    if ('raw_' in file) & ('.hdf5' in file):
        file_date_str = file.split('raw_')[-1].split('.hdf5')[0]
        file_date = datetime.datetime.strptime(file_date_str,'%Y-%m-%d')
    else:
        raise TypeError('input file is either "raw" or not a hdf5 file')
    # working directory to get distributions from    (do we look at current month or previous?? 
    #wdir = basedir + f'{file_date.year}/{file_date.month:02}/{UT_name}/'
    
    #wdir_tmp = wdir # this one we might change if we need to look back to previous months 
    
    # now get dictionary with basic state information of the input MNII file
    input_file_state = get_state( UT, file_date_str+'T'+time_key) 
    
    # now get focus names consistent with naming convention f'{UT}_{state_name}_{focus_name}_{sensor}_psd_distribution_features.csv'
    if input_file_state['Coude']:
        focus_name = 'coude'
    elif input_file_state['Nasmyth A']:
        focus_name = 'nasA'
    elif input_file_state['Nasmyth B']:
        focus_name = 'nasB'
    elif input_file_state['Cassegrain']:
        focus_name = 'cas'
        
    # now get state names consistent with feature_distribution csv file naming convention
    if input_file_state['open_enclosure'] & input_file_state['guiding']:
        state_name = 'operations'
    else:
        state_name = 'non-operations'
        
    

    # INIT FIGURES 
    if (sensors == 'all'): #& (not post_upgrade) 
        fig = plt.figure(constrained_layout=True,figsize=(30,20))
        gs = fig.add_gridspec(7,3)

        ax11 = fig.add_subplot(gs[0, 0]) #acc 
        ax12 = fig.add_subplot(gs[1, 0]) #acc 
        ax13 = fig.add_subplot(gs[2, 0]) #acc 
        ax14 = fig.add_subplot(gs[3, 0]) #acc 


        ax21 = fig.add_subplot(gs[0, 1]) #acc 

        ax31 = fig.add_subplot(gs[0, 2]) #acc 
        ax32 = fig.add_subplot(gs[1, 2]) #acc 

        ax1 = fig.add_subplot(gs[5, 0]) # m1 combined 
        ax2 = fig.add_subplot(gs[5, 1]) # m2 combined  
        ax3 = fig.add_subplot(gs[5, 2]) # m3 combined 
        ax4 = fig.add_subplot(gs[6, 0]) # m123

    elif type(sensors)==type([]): #& (not post_upgrade):

        # initiate axx == len list

        no_plots = len(sensors) 
        fig, ax = plt.subplots( no_plots , 1 , figsize=(8, 5*no_plots) )


    if post_upgrade:
        print('post upgrade data on new sensors not ready for classification yet... coming soon\n\
        for now we just consider sensors up to m3') 
        
   
    # process data 
    # ====================
    acc_dict = process_single_mn2_sample(file, time_key, post_upgrade = post_upgrade) 
    # ====================

    #print(acc_dict.keys() )
    #init dict 
    freq_classification_dict = {}
    red_flag_freqs = {}

    if (sensors == 'all') :
        
        for j, (sensor, ax) in enumerate( zip( ['sensor_5','sensor_6','sensor_7','sensor_8','sensor_3','sensor_1','sensor_2','m1','m2','m3','m1-3'], [ax11,ax12,ax13,ax14,ax21,ax31,ax32, ax1,ax2,ax3,ax4] ) ):
            
            # Get the reference file that corresponds to the input file state / sensor to compare PSD statistics 
            psd_reference_file = _get_psd_reference_file(reference_dir, UT_name,state_name,focus_name, sensor)
            
            
            f_acc, accel_psd  = acc_psd(acc_dict[sensor])

            f, opl_psd = double_integrate(f_acc, accel_psd )

            if 'sensor' in sensor: #for raw sensors we apply additional gain factor (V2acc_gain) to convert V^2/Hz to m^2/Hz 

                opl_psd *= V2acc_gain * 1e12 
                
                dist_features = pd.read_csv( psd_reference_file[0] ,index_col=0) 
                if dist_features.empty:
                    raise TypeError(f'\n!!!!!!\n\nPSD reference file {psd_reference_file[0]} is empty.\n\n  This occurs when there was insufficient PSD data to derive statistics (e.g. a M1 recoating or instrument change for the focus etc).\n\n   Consider finding & uploading a new reference file for the given UT, focus, state, and sensor!')
                    
                #if empty then we use wdir_tmp to go back
                
                
                #imterpolate onto same grid (to be safe) 
                fn = interp1d( f, opl_psd , kind = 'linear' , bounds_error=False, fill_value=np.nan) 
                opl_psd_interp = fn( list(dist_features.index) )

                quantile_df = V2acc_gain * dist_features[[f'q{int(x)}' for x in np.linspace(10,90,9)]]
                freq_classifications = [classify_freq_bin(dist_features.index[i], opl_psd_interp[i], quantile_df.iloc[i], opl_thresh=opl_thresh) for i in range(len(quantile_df))]
                # make as panda series indexed by frequency 
                freq_classification_dict[sensor] = pd.Series( freq_classifications , index = f)
                # publish frequencies with red flags 
                red_flag_freqs[sensor] = get_flag_frequencies_indicies( freq_classification_dict[sensor], flag_color='red' )  

            else: 
                opl_psd *= 1e12 # make it um^2/Hz

                dist_features = pd.read_csv( psd_reference_file[0] , index_col=0) 
                if dist_features.empty:
                    raise TypeError(f'\n!!!!!!\n\nPSD reference file {psd_reference_file[0]} is empty.\n\n  This occurs when there was insufficient PSD data to derive statistics (e.g. a M1 recoating or instrument change for the focus etc).\n\n   Consider finding & uploading a new reference file for the given UT, focus, state, and sensor!')
                #imterpolate onto same grid (to be safe) 
                fn = interp1d( f, opl_psd , kind = 'linear' , bounds_error=False, fill_value=np.nan) 
                opl_psd_interp = fn( list(dist_features.index) )

                quantile_df = dist_features[[f'q{int(x)}' for x in np.linspace(10,90,9)]]
                freq_classifications = [classify_freq_bin(dist_features.index[i], opl_psd_interp[i], quantile_df.iloc[i], opl_thresh=opl_thresh) for i in range(len(quantile_df))]
                # make as panda series indexed by frequency 
                freq_classification_dict[sensor] = pd.Series( freq_classifications , index = f)
                # publish frequencies with red flags 
                red_flag_freqs[sensor] = get_flag_frequencies_indicies( freq_classification_dict[sensor], flag_color='red' ) 
                
            #############
            # PLOTTING 
            #############

            # make background pink if there is a redflag
            if 'red' in freq_classification_dict[sensor].values:
                ax.set_facecolor(color='pink')
            else:
                ax.set_facecolor(color='white')

            #plotting PSD 
            

            ax.scatter(f, opl_psd, s=30, color=freq_classification_dict[sensor].values, label=sensor)

            ax.loglog(f, quantile_df['q10'],color='k',linestyle='--',alpha=0.5)# ,label='q10-90')
            ax.loglog(f, quantile_df['q90'],color='k',linestyle='--',alpha=0.5)
            
            ax.fill_between(f, quantile_df['q10'], quantile_df['q90'], color='green',alpha=0.4) #label=f'historic 10-90 percentile range \n{min(psds.index)[:10]} - {max(psds.index)[:10]}')
                        

            
            #reverse cumulative 
            ax.plot(f, np.cumsum(opl_psd[::-1])[::-1] * np.diff(f)[1], color='k', linestyle = '-') 
            
            #ax.loglog(f, np.cumsum(quantile_df['q90'][::-1])[::-1] * np.diff(f)[1],color='grey',linestyle=':')
            #ax.loglog(f, np.cumsum(quantile_df['q90'][::-1])[::-1] * np.diff(f)[1],color='grey',linestyle=':')
            
            ax.legend(fontsize=24)
            ax.grid()
            ax.tick_params(labelsize=18)
            ax.set_ylabel('PSD '+r'[$\mu m^2$/Hz]'+'\nreverse cum. '+r'[$\mu m^2$]',fontsize=20)
            ax.set_xlabel('frequency [Hz]',fontsize=20)
            #############
            #############

            if verbose:
                if len(red_flag_freqs[sensor]) > 0:
                    print('\n{} has red flags (alarm!) at frequencies = {} Hz\n'.format( sensor, red_flag_freqs ) ) 
        
        plt.show() 
        
        return( red_flag_freqs, freq_classification_dict) 
    
    
    else: # user defined sensors
        
        if len(sensors) == 1: # if only 1 sensor to consider
            
            sensor = sensors[0]
            
            # Get the reference file that corresponds to the input file state / sensor to compare PSD statistics 
            psd_reference_file = _get_psd_reference_file(reference_dir, UT_name, state_name, focus_name, sensor)
            
            
            axx = ax 
            
            f_acc, accel_psd  = acc_psd(acc_dict[sensor])

            f, opl_psd = double_integrate(f_acc, accel_psd )

            if 'sensor' in sensor: #for raw sensors we apply additional gain factor (V2acc_gain) to convert V^2/Hz to m^2/Hz 

                opl_psd *= V2acc_gain * 1e12 

                dist_features = pd.read_csv( psd_reference_file[0], index_col=0) 
                if dist_features.empty:
                    raise TypeError(f'\n!!!!!!\n\nPSD reference file {psd_reference_file[0]} is empty.\n\n  This occurs when there was insufficient PSD data to derive statistics (e.g. a M1 recoating or instrument change for the focus etc).\n\n   Consider finding & uploading a new reference file for the given UT, focus, state, and sensor!')
                    
                #imterpolate onto same grid (to be safe) 
                fn = interp1d( f, opl_psd , kind = 'linear' , bounds_error=False, fill_value=np.nan) 
                opl_psd_interp = fn( list(dist_features.index) )

                quantile_df = V2acc_gain * dist_features[[f'q{int(x)}' for x in np.linspace(10,90,9)]]
                freq_classifications = [classify_freq_bin(dist_features.index[i], opl_psd_interp[i], quantile_df.iloc[i], opl_thresh=opl_thresh) for i in range(len(quantile_df))]
                # make as panda series indexed by frequency 
                freq_classification_dict[sensor] = pd.Series( freq_classifications , index = f)
                # publish frequencies with red flags 
                red_flag_freqs[sensor] = get_flag_frequencies_indicies( freq_classification_dict[sensor], flag_color='red' ) 

            else: 
                opl_psd *= 1e12 # make it um^2/Hz

                dist_features = pd.read_csv( psd_reference_file[0], index_col=0) 
                if dist_features.empty:
                    raise TypeError(f'\n!!!!!!\n\nPSD reference file {psd_reference_file[0]} is empty.\n\n  This occurs when there was insufficient PSD data to derive statistics (e.g. a M1 recoating or instrument change for the focus etc).\n\n   Consider finding & uploading a new reference file for the given UT, focus, state, and sensor!')
                    
                #imterpolate onto same grid (to be safe) 
                fn = interp1d( f, opl_psd , kind = 'linear' , bounds_error=False, fill_value=np.nan) 
                opl_psd_interp = fn( list(dist_features.index) )

                quantile_df = dist_features[[f'q{int(x)}' for x in np.linspace(10,90,9)]]
                freq_classifications = [classify_freq_bin(dist_features.index[i], opl_psd_interp[i], quantile_df.iloc[i], opl_thresh=opl_thresh) for i in range(len(quantile_df))]
                # make as panda series indexed by frequency 
                freq_classification_dict[sensor] = pd.Series( freq_classifications , index = f)
                # publish frequencies with red flags 
                red_flag_freqs[sensor] = get_flag_frequencies_indicies( freq_classification_dict[sensor], flag_color='red' ) 


            #############
            # PLOTTING 
            #############
            # make background pink if there is a redflag
            if 'red' in freq_classification_dict[sensor].values:
                axx.set_facecolor(color='pink')
            else:
                axx.set_facecolor(color='white')

            #plotting 
            
            axx.scatter(f, opl_psd, s=30, color=freq_classification_dict[sensor].values, label=sensor)

            axx.loglog(f, quantile_df['q10'],color='k',linestyle='--',alpha=0.5)# ,label='q10-90')
            axx.loglog(f, quantile_df['q90'],color='k',linestyle='--',alpha=0.5)
            
            axx.fill_between(f, quantile_df['q10'], quantile_df['q90'], color='green',alpha=0.4) 
            
            
            #reverse cumulative 
            axx.plot(f, np.cumsum(opl_psd[::-1])[::-1] * np.diff(f)[1], color='k', linestyle = '-') 

            axx.legend(fontsize=24)
            axx.grid()
            axx.tick_params(labelsize=18)
            axx.set_ylabel('PSD '+r'[$\mu m^2$/Hz]'+'\nreverse cum. '+r'[$\mu m^2$]',fontsize=20)
            axx.set_xlabel('frequency [Hz]',fontsize=20)
            #############
            #############
            
        else: 
            
            for j, (sensor, axx) in enumerate( zip( sensors, ax ) ):
                
                # Get the reference file that corresponds to the input file state / sensor to compare PSD statistics 
                psd_reference_file = _get_psd_reference_file(reference_dir, UT_name, state_name, focus_name, sensor)
                
                f_acc, accel_psd = acc_psd(acc_dict[sensor])

                f, opl_psd = double_integrate(f_acc, accel_psd )

                if 'sensor' in sensor: #for raw sensors we apply additional gain factor (V2acc_gain) to convert V^2/Hz to m^2/Hz 

                    opl_psd *= V2acc_gain * 1e12 

                    dist_features = pd.read_csv( psd_reference_file[0], index_col=0) 
                    if dist_features.empty:
                        raise TypeError(f'\n!!!!!!\n\nPSD reference file {psd_reference_file[0]} is empty.\n\n  This occurs when there was insufficient PSD data to derive statistics (e.g. a M1 recoating or instrument change for the focus etc).\n\n   Consider finding & uploading a new reference file for the given UT, focus, state, and sensor!')
                    #imterpolate onto same grid (to be safe) 
                    fn = interp1d( f, opl_psd , kind = 'linear' , bounds_error=False, fill_value=np.nan) 
                    opl_psd_interp = fn( list(dist_features.index) )

                    quantile_df = V2acc_gain * dist_features[[f'q{int(x)}' for x in np.linspace(10,90,9)]]
                    freq_classifications = [classify_freq_bin(dist_features.index[i], opl_psd_interp[i], quantile_df.iloc[i], opl_thresh=opl_thresh) for i in range(len(quantile_df))]
                    # make as panda series indexed by frequency 
                    freq_classification_dict[sensor] = pd.Series( freq_classifications , index = f)
                    # publish frequencies with red flags 
                    red_flag_freqs[sensor] = get_flag_frequencies_indicies( freq_classification_dict[sensor], flag_color='red' ) 

                else: 
                    opl_psd *= 1e12 # make it um^2/Hz

                    dist_features = pd.read_csv( psd_reference_file[0], index_col=0) 
                    if dist_features.empty:
                        raise TypeError(f'\n!!!!!!\n\nPSD reference file {psd_reference_file[0]} is empty.\n\n  This occurs when there was insufficient PSD data to derive statistics (e.g. a M1 recoating or instrument change for the focus etc).\n\n   Consider finding & uploading a new reference file for the given UT, focus, state, and sensor!')
                    #imterpolate onto same grid (to be safe) 
                    fn = interp1d( f, opl_psd , kind = 'linear' , bounds_error=False, fill_value=np.nan) 
                    opl_psd_interp = fn( list(dist_features.index) )

                    quantile_df = dist_features[[f'q{int(x)}' for x in np.linspace(10,90,9)]]
                    freq_classifications = [classify_freq_bin(dist_features.index[i], opl_psd_interp[i], quantile_df.iloc[i], opl_thresh=opl_thresh) for i in range(len(quantile_df))]
                    # make as panda series indexed by frequency 
                    freq_classification_dict[sensor] = pd.Series( freq_classifications , index = f)
                    # publish frequencies with red flags 
                    red_flag_freqs[sensor] = get_flag_frequencies_indicies( freq_classification_dict[sensor], flag_color='red' ) 

                #############
                # PLOTTING 
                #############

                # make background pink if there is a redflag
                if 'red' in freq_classification_dict[sensor].values:
                    axx.set_facecolor(color='pink')
                else:
                    axx.set_facecolor(color='white')

                #plotting 
                axx.scatter(f, opl_psd, s=30, color=freq_classification_dict[sensor].values, label=sensor)

                axx.loglog(f, quantile_df['q10'],color='k',linestyle='--',alpha=0.5)# ,label='q10-90')
                axx.loglog(f, quantile_df['q90'],color='k',linestyle='--',alpha=0.5)

                axx.fill_between(f, quantile_df['q10'], quantile_df['q90'], color='green',alpha=0.4) 
            

                #reverse cumulative 
                axx.plot(f, np.cumsum(opl_psd[::-1])[::-1] * np.diff(f)[1], color='k', linestyle = '-') 

                axx.legend(fontsize=24)
                axx.grid()
                axx.tick_params(labelsize=18)
                axx.set_ylabel('PSD '+r'[$\mu m^2$/Hz]'+'\nreverse cum. '+r'[$\mu m^2$]',fontsize=20)
                axx.set_xlabel('frequency [Hz]',fontsize=20)
                #############
                #############



        if verbose:
            if len(red_flag_freqs[sensor]) > 0:
                print('\n{} has red flags (alarm!) at frequencies = {} Hz\n'.format( sensor, red_flag_freqs ) ) 

        plt.show() 

        return( red_flag_freqs, freq_classification_dict) 

        


        

def bimodal_coef(data):
    """
    from Kang 2019 - Development of Hartigan’s Dip Statistic with Bimodality Coefficient to Assess Multimodality of Distributions 
    https://www.hindawi.com/journals/mpe/2019/4819475/ 
    
    original reference 
    “Miscellaneous Formulas” of the SAS User’s Guide (SAS Institute Inc, 1990, p. 561)
    
    data is 1d numpy array 
    
    returns BC - bimodality coefficient.
    if BC <= 0.555 (BC for uniform distribution) then it is considered bimodal 
    """
    data = data[ np.isfinite(data) ]
    
    n = len(data) 

    mu = np.nanmean( data )
    
    
    if n==0:
        print( 'input data length = 0, returning BC = np.nan' )
        BC = np.nan
        return(BC)
    
    elif not np.isfinite(mu):
        print( f'np.nanmean(data) = {mu} is not finite, returning BC = np.nan' )
        BC = np.nan
        return(BC)
        
    else:
        if (np.sqrt(n * (n-1)) / (n-2) != 0) & ( (np.sqrt( (1/n) * np.sum( (data-mu)**2 )) )**3 != 0 ): # check no division by zero

            m3 = np.sqrt(n * (n-1)) / (n-2)  *  (\
                    (1/n) * np.sum( (data-mu)**3 ) / (np.sqrt( (1/n) * np.sum( (data-mu)**2 )) )**3 \
                                          )
        else:
            m3 = np.inf 


        if ((n-1)/((n-2)*(n-3)) != 0) & ( ((1/n) * np.sum( (data-mu)**2 ) )**2 != 0): # check no division by zero

            m4 = (n-1)/((n-2)*(n-3)) * ( (n+1) * (\
                    (1/n) * np.sum( (data-mu)**4 ) / ((1/n) * np.sum( (data-mu)**2 ) )**2 \
                                                 ) - 3*(n-1) )
        else:
            m4 = np.inf

        BC = ( m3**2 + 1 ) / ( m4 + 3*( (n-1)**2 / ((n-2)*(n-3)) ) )

        if not np.isfinite( BC ): # set infinite values to nan
            BC = np.nan

        return(BC) 



def amplifier_filter():
    print('to do')


def apply_MN2_filter():
    print('to do')

    

    

def vibration_analysis(psd, detection_method = 'median', window=50, thresh_factor = 3, plot_psd = True, plot_peaks = False):
    
    '''
    
    from an input power spectral density (psd) this function detects 
    peaks above some threshold (determined by detection_method) and
    returns a pandas dataframe with the detected peak frequencies, 
    spectral width, prominence (height above the local psd floor) and the 
    absolute and relative contributions of the peak (the integrated psd and 
    and integrated psd - local floor over spectral width respectively)

    Parameters:
        psd (tuple): PSD tuple (freq, PSD) where freq and PSD are numpy arrays
        
        detection_method (string) : string to indicate the detection method 
        ('fit' or 'median' - fit assumes noise floor follows single power law over frequency domain)
        
        window (int) : the window size to apply rolling aggregate for calculating 
            the peak detection threshold (Only used for detection_method = 'median')
        
        thresh_factor (float) : factor to multiply a reference line (std or median) to set 
            vibration detection threshold 
            
        plot_psd (boolean) : Do you want to plot the PSD with the marked detected peaks?
        
        plot_peaks (boolean) : Do you want to plot the individual (cropped)
            detected peaks ?

    Returns:
        vibration_df (pandas dataframe): output dataframe containing the detected 
        peak frequencies, spectral width, prominence (height above the local 
        psd floor) and the absolute and relative contributions of the peak 
        (the integrated psd and and integrated psd - local floor over spectral 
        width respectively)  
    
    '''
    f,psd = psd #psd_dict['disturb_psd'][0][0],psd_dict['disturb_psd'][0][1][:,0]
    

    if detection_method == 'fit':

        df = np.diff(f)[0] #assumes this is linearly spaced! 
        param, cov = np.polyfit(np.log10(f), np.log10(psd), 1, cov=True) 

        grad,inter = param[0], param[1]
        dg, di = np.sqrt(cov[0,0]), np.sqrt(cov[1,1])

        psd_fit_1 = 10**(grad*np.log10(f) + inter) 
        psd_fit_1_uncert = psd_fit_1 * np.log(10) * np.sqrt( (dg*np.log10(f))**2  + (di)**2 ) #standard error propagration
        #psd_fit_1 = 10**(param[0]*np.log10(f) + param[1]) 


        #======== remove outliers
        outlier_thresh  = psd_fit_1 + 2 * psd_fit_1_uncert  #2 * psd_fit_1
        indx = np.where( abs(psd_fit_1 - psd) < outlier_thresh )
        
        #re-fit after removing outliers 
        param, cov = np.polyfit(np.log10(f[indx]), np.log10(psd)[indx], 1, cov=True)

        grad,inter = param[0], param[1]
        dg, di = np.sqrt(cov[0,0]), np.sqrt(cov[1,1])

        psd_fit_2 = 10**(grad*np.log10(f) + inter) 
        psd_fit_2_uncert = psd_fit_1 * np.log(10) * np.sqrt( (dg*np.log10(f))**2  + (di)**2 )  ##standard error propagration

        if plot_psd:
            fig,ax = plt.subplots(1,1) 
            ax.semilogy(f,psd_fit_1,label='fit without outlier removal',linestyle=':')
            ax.semilogy(f,psd_fit_2,label='fit with outlier removal')
            ax.semilogy(f,psd)
            ax.legend(fontsize=12)
            ax.set_xlabel('frequency [Hz]',fontsize=12)
            ax.set_ylabel('PSD ',fontsize=12)



        #======== get vibration regions (frequency blocks where PSD above threshold )
        vib_thresh = psd_fit_2 + thresh_factor * psd_fit_2_uncert # x2 std error of fit (with outlier rejection) 

        previous_point_also_above_threshold = False #init boolean that the previous point was not above vib_thresh
        block_indxs = [] # list to hold [block start, block end] indicies 
        block_indx = [np.nan, np.nan] # init current [block start, block end] indicies

 
        for i,_ in enumerate(f):
            
            if (i != len(f)) & (i!=0):
                if ( psd[i] > vib_thresh[i] ) & (not previous_point_also_above_threshold):
                    # then we are at the beggining of a block
                    block_indx[0] = i-1 # subtract 1 index from block start (since this is point above threshold)
                    previous_point_also_above_threshold = True
                    #print(block_indx )
        
        
        
                elif ( psd[i] <= vib_thresh[i] ) & previous_point_also_above_threshold:
                    #Then we are at an end of a block 
                    block_indx[1] = i 
        
                    # append block
                    block_indxs.append(block_indx)
                    # re-init block index tuple 
                    previous_point_also_above_threshold = False
                    block_indx = [np.nan, np.nan]
                    
            #deal with special case i=0 - we ignore this case, it is not necessary
            #elif  i==0:
            #    if ( psd[i] > vib_thresh[i] ):
            #        block_indx[0] = i 
            #        previous_point_also_above_threshold = True
                
            #deal with special case that last index is still within a block
            elif (i == len(f)) & (not np.isnan(block_indx[0])):
                
                block_indx[1] = i
                # append block
                block_indxs.append(block_indx)
        
        

        if plot_psd:

    
            fig, ax = plt.subplots(figsize=(8,5)) 
            ax.loglog(f,psd,color='k')
            ax.loglog(f,psd_fit_2, color='green',label=r'$af^{-b}$')
            ax.loglog(f,vib_thresh,color='red',label='detection threshold',linestyle='--')
            
            plt.fill_between(f, vib_thresh, psd, where=vib_thresh < psd,color='r',alpha=0.3,label='detected vibrations')
        
            ax.legend(fontsize=15)
            ax.set_xlabel('frequency [Hz]',fontsize=15)
            ax.set_ylabel('PSD '+r'$[m^2/Hz]$',fontsize=15)
            #lets visualize these 
            """for b in block_indxs: #period ends
                
                ax.axvspan(f[b[0]],f[b[1]],color='r',alpha=0.3)
                #plt.axvline(f[b[0]],color='g',linestyle=':') #green is start of block 
                #plt.axvline(f[b[1]],color='r',linestyle=':')"""


        # for each block detrend the PSD region with power law fit and then do peak detections (if wider then a few Hz)
        # get freq, width, prominence, relative OPL, absolute OPL 


        if plot_peaks: # set up axes to display the relative contributions from each peak 
            fig, ax = plt.subplots(round(np.sqrt( len(block_indxs) ) ), round(np.sqrt( len(block_indxs) ))+1 ,figsize=(15,15))
            axx = ax.reshape(1,-1)

        #init our dictionary to hold features 
        vib_feature_dict = {'vib_freqs':[], 'fwhm':[], 'prominence':[], 'abs_contr':[], 'rel_contr':[]}    
        for jj, (i1,i2) in enumerate(block_indxs):

            #detrend the psd with the power law fit
            psd_detrend = psd[i1:i2]/psd_fit_2[i1:i2] 

            if f[i2]-f[i1] < 50: # if block width < 50Hz we don't worry about fitting and just do basic calculations 



                #calculate frequency of vibration peak (detrend for this )
                i_max = np.argmax(psd_detrend)
                f_peak = f[i1:i2][i_max]

                #calculate prominence 
                prominence = psd[i1:i2][i_max] - psd_fit_2[i1:i2][i_max] # m^2 / Hz

                # interpolate this onto finner grid 
                fn_interp = interp1d(f[i1:i2] , psd_detrend , fill_value='extrapolate')
                f_interp = np.linspace(f[i1],f[i2] , 10*(i2 - i1) ) #grid 10x finer then current
                df_interp = np.diff(f_interp)[0]
                psd_detrend_interp = fn_interp(f_interp)

                # calculate FWHM as df x the how many frequency bins are above the half max within the block
                fwhm = df_interp * np.sum( psd_detrend_interp > np.max(psd_detrend_interp)/2 ) 

                # calculate absolute and relative OPL contributions of the vibration peak 
                abs_contr = np.trapz(psd[i1:i2+1], f[i1:i2+1]) # m^2
                rel_contr = np.trapz(psd[i1:i2+1], f[i1:i2+1]) - np.trapz(psd_fit_2[i1:i2+1], f[i1:i2+1]) # m^2

                if plot_peaks:
                    # for plotting we extend the range just a bit to make peaks clearer 
                    #plt.figure()

                    axx[0,jj].semilogy(f_interp, psd_detrend_interp)
                    axx[0,jj].semilogy(f_interp, np.ones(len(psd_detrend_interp)))

                    #axx[0,jj].semilogy(f[i1:i2+1], psd[i1:i2+1])
                    #axx[0,jj].semilogy(f[i1:i2+1], psd_fit_2[i1:i2+1])
                    #axx[0,jj].fill_between(f[i1:i2+1], psd_fit_2[i1:i2+1], psd[i1:i2+1] , where = psd_fit_2[i1:i2+1] <= psd[i1:i2+1],label='rel contr')
                    """
                    if i2-i1 > 2:
                        axx[0,jj].fill_between(f[i1:i2], psd_fit_2[i1:i2], psd[i1:i2] , where = psd_fit_2[i1:i2] <= psd[i1:i2],label='rel contr')
                    else:
                        axx[0,jj].fill_between(f[i1-2:i2+2], psd_fit_2[i1-2:i2+2], psd[i1-2:i2+2] , where = psd_fit_2[i1-2:i2+2] <= psd[i1-2:i2+2],label='rel contr')
                    #axx[0,jj].legend()"""


                vib_feature_dict['vib_freqs'].append(f_peak) #Hz
                vib_feature_dict['fwhm'].append(fwhm) #Hz 
                vib_feature_dict['abs_contr'].append(abs_contr) #m^2 
                vib_feature_dict['rel_contr'].append(rel_contr) #m^2 
                vib_feature_dict['prominence'].append(prominence) #m^2/Hz

            else: # to do.. implement multiple curve fitting within block
                raise TypeError('THIS CASE IS NOT CODED ',f[i1]) 
                # estimate how many peaks are in this block 

                # for each peak fit a lorenzian profile on the detrended PSD with initial guess centered on peak freq

                # extract features from fit 


                
    elif detection_method == 'median':

        a1, a0 = np.polyfit(np.log10(f), np.log10(psd), 1) 
        linfit = 10**(a0 + a1 * np.log10(f[1:]))


        med_floor = pd.Series(psd).rolling(window,center=True).median()
        h_thresh = thresh_factor * med_floor.values  #freqs**0.2 * pd.Series(psd).rolling(100,center=True).mean().values #2*med_floor.values #peak needs to be x2 neighborhood median to be counted 

        peaks = sig.find_peaks(psd, height = h_thresh) #height = height.values
        peak_proms = sig.peak_prominences(psd, peaks[0], wlen=50)
        peak_widths = sig.peak_widths(psd,peaks[0],prominence_data=peak_proms,rel_height=0.9)
        #calc widths at base where peak prominence is measured 




        #contour height to indicate prominence 
        contour_heights = psd[peaks[0]] - peak_proms[0]

        # look at widths 
        li = (peaks[0]-np.round(peak_widths[0]/2)).astype(int)
        ui = (peaks[0]+np.round(peak_widths[0]/2)).astype(int)

        #plot peaks inidcating their calculated prominence 
        if plot_psd:
            plt.figure()

                # plot psd, median floor and peak detection threshold
            plt.semilogy(f,psd)
            plt.semilogy(f,med_floor,linestyle=':', color='k' )
            plt.semilogy(f, h_thresh, linestyle=':', color='r' )

            plt.semilogy(f[peaks[0]],psd[peaks[0]],'.',color='r')
            plt.vlines(x=f[peaks[0]], ymin=contour_heights, ymax=psd[peaks[0]],color='k')
            #plt.xlim([0,30])

            #plt.semilogy(f[li],psd[li],'x',color='g')
            #plt.semilogy(f[ui],psd[ui],'x',color='b')

            plt.xlabel('Frequency [Hz]',fontsize=14)
            plt.ylabel('PSD',fontsize=14)        
            plt.gca().tick_params(labelsize=14)

        cummulative = np.cumsum(psd) 


        abs_contr, rel_contr = [], [] #lists to hold absolute and relative PSD peak contributions 

        for i, (lo,up) in enumerate(zip(li,ui)):
            abs_contr.append( cummulative[up] - cummulative[lo] )

            if up - lo == 2: #if only 2 samples in upper/lower limits we add one sampe (so symmetric around peak point)
                up+=1

            interp_base =  linfit[lo:up]
            rel_contr.append( np.trapz( psd[lo:up] - interp_base, f[lo:up] ) )

            #need to visulize relative level for each peak
            if plot_peaks :
                plt.figure()
                plt.plot(f[lo-5:up+5], psd[lo-5:up+5] ,color='k')
                plt.plot(f[lo:up],interp_base,color='orange')
                plt.plot(f[peaks[0][i]],psd[peaks[0][i]],'.',color='b',label='detected peak')
                plt.fill_between(f[lo:up],y1=interp_base,y2=psd[lo:up],color='red',alpha=0.5,label='relative contr.') 

                plt.xlabel('Frequency [Hz]',fontsize=14)
                plt.ylabel('PSD',fontsize=14)        
                plt.gca().tick_params(labelsize=14)

                plt.text(f[peaks[0][i]], 1.1*psd[peaks[0][i]], 'peak frequency = {:.1f}Hz,\nrel contribution = {:.3e}'.format(f[peaks[0][i]],rel_contr[i]))
                plt.ylim([0, 1.3*psd[peaks[0][i]] ])
                plt.legend(loc='lower left')



        vib_feature_dict = {'vib_freqs':f[peaks[0]], 'fwhm':peak_widths[0], 'prominence':peak_proms[0], \
         'abs_contr':abs_contr, 'rel_contr':rel_contr}


    return(vib_feature_dict)




def plot_all_timeseries(file, time_key, post_upgrade=True, save=False):
    """   
    plots time series of 10s sample from  MNII accelerometer sensors (raw) given a:
    
    input 
        file name - string (e.g. 'ldlvib{UT}_raw_{yyyy}-{mm}-{dd}.hdf5')
        ... note needs to be raw file and not psd 
        time_key - string (e.g 'hh:mm:ss' ) 
        post_upgrade - do you want to include the m4-m7 sensors? 
        save - do you want to save the file?
        
    """
    plt.figure(figsize=(10,8))
    plt.title(f'{file}\nT{time_key}',fontsize=15)
    
    mn2_data = h5py.File(file, 'r')
    
    if post_upgrade:
        
        for i,acc in enumerate([1,2,3,4,5,6,7,8,9,10,11,12]):
            acc_data = 10*i + mn2_data[time_key][f'sensor{acc}'][:]
            plt.plot(acc_data,'k',alpha=0.9)
            plt.text(0, 10*i+2, i2m_dict[i+1], fontsize=12 ,color='red')

        plt.xlabel('samples',fontsize=20)
        plt.ylabel('acceleration '+r'$(m/s^2)$'+'\n',fontsize=20)
        plt.yticks([])
        plt.gca().tick_params(labelsize=15)
        plt.tight_layout()

        if save:

            plt.savefig(f'/home/jovyan/usershared/BCB/mn2_upgrade/timeseries_{file}-{time_key}.png')

        plt.show()
        
        
    elif not post_upgrade:
        
        for i,acc in enumerate([1,2,3,4,5,6,7,8]):
            acc_data = 10*i + mn2_data[time_key][f'sensor{acc}'][:]
            plt.plot(acc_data,'k',alpha=0.9)
            plt.text(0, 10*i+2, i2m_dict[i+1], fontsize=12 ,color='red')

        plt.xlabel('samples',fontsize=20)
        plt.ylabel('acceleration '+r'$(m/s^2)$'+'\n',fontsize=20)
        plt.yticks([])
        plt.gca().tick_params(labelsize=15)
        plt.tight_layout()

        if save:

            plt.savefig(f'/home/jovyan/usershared/BCB/mn2_upgrade/timeseries_all_sensors_{file}-{time_key}.png')

        plt.show()


def plot_timeseries(file, time_key, sensor, save=False):
    """
    plots time series of 10s sample from  MNII accelerometer sensor (raw) given a:
    
    input 
        file name - string (e.g. 'ldlvib{UT}_raw_{yyyy}-{mm}-{dd}.hdf5')
        ... note needs to be raw file and not psd 
        time_key - string (e.g 'hh:mm:ss' ) 
        sensor - str (sensor_{X}, 'm1', 'm1-3' etc ... the keys from process_single_mn2_sample output)
        
    """
    plt.figure(figsize=(8,5))
    plt.title(f'{sensor} - {file}\nT{time_key}',fontsize=15)
    
    acc = process_single_mn2_sample(file, time_key, post_upgrade=True)
    
    plt.plot(acc[sensor],color=k)
    plt.xlabel('samples',fontsize=20)
    plt.ylabel('acceleration (undefined units)',fontsize=20)
    plt.gca().tick_params(labelsize=15)
    plt.tight_layout()    
    
    if save:

        plt.savefig(f'/home/jovyan/usershared/BCB/mn2_upgrade/timeseries_{sensor}_{file}-{time_key}.png')
        
    plt.show()




def plot_all_psds(file, time_key, sensor, double_integrate=True, save=False):
    print('to do') 
    """
    plots PSD of 10s sample from MNII accelerometer sensors:
    
    input 
        file name - string (e.g. 'ldlvib{UT}_raw_{yyyy}-{mm}-{dd}.hdf5')
        ... note needs to be raw file and not psd 
        time_key - string (e.g 'hh:mm:ss' ) 
        sensor - str (sensor_{X}, 'm1', 'm1-3' etc ... the keys from process_single_mn2_sample output)
        double_integrate - do we want to double integrate the accelerometer signal? 
    """


    """
    # plot an entire night 

    mirrors = [f'm{x}' for x in range(1,8)]+['m1-3','m4-7','m1-7']
    acc = process_single_mn2_sample(file_upgraded , time_key) #upgraded_process_daily_MN2_file(file_upgraded , verbose=True)

    #acc = process_from_tac_sample('/home/jovyan/usershared/BCB/UT1/2022-08-06T165754_UT1.dat')
    file_upgraded =  f'/datalake/rawdata/vlt-vibration/2022/08/ldlvib{UT}_raw_2022-08-14.hdf5'

    mn2_data = h5py.File(file_upgraded , 'r')

    date_tmp = datetime.datetime.strptime( file_upgraded.split('_')[-1].split('.')[0] +'T'+ '12:00:00','%Y-%m-%dT%H:%M:%S') 


    #periods of telescope guiding
    guiding_periods = dlt.query_guide_times(env, file_date - datetime.timedelta(days=1), date_tmp + datetime.timedelta(days=1))

    #periods of enclosure open  
    open_enc_periods = dlt.query_enclosure_times(env, file_date - datetime.timedelta(days=1), date_tmp + datetime.timedelta(days=1))



    fig,ax = plt.subplots(10,1, figsize=(8,50))
    ax2 = []
    for axx, m_label in zip(ax, mirrors  ) :
        #axx.set_title( m_label , fontsize=12)
        axx.set_xlabel('frequency (Hz)',fontsize=15)
        axx.set_ylabel(f'{m_label} PSD '+ r'$(\mu m^2/Hz)$'+'\nreverse cumulative '+r'($\mu m^2$)',fontsize=15)

        ax2.append(axx.twinx())

    for time_key in list(mn2_data.keys())[::]:

        #print(time_key)

        file_date = datetime.datetime.strptime( file_upgraded.split('_')[-1].split('.')[0] +'T'+ time_key,'%Y-%m-%dT%H:%M:%S') 

        guiding, open_enc = is_between(file_date, guiding_periods) , is_between(file_date, open_enc_periods)


        if (guiding == 1) & (open_enc == 1):
            # process the current file at the given time 
            acc = process_single_mn2_sample(file_upgraded , time_key, post_upgrade=True) #upgraded_process_daily_MN2_file(file_upgraded , verbose=True)

            for axx, m in zip(ax, mirrors  ):

                psd_acc = sig.welch(acc[m], fs=1e3, nperseg=2**11, axis=0) 

                f, psd_pos = double_integrate(psd_acc[0],psd_acc[1])

                if m == 'm1-7':
                    psd_pos_m1_7 =  psd_pos

                axx.loglog(f, 1e12 * psd_pos, color='k', lw=1, alpha=0.01)

                axx.loglog(f, 1e12 * np.cumsum(psd_pos[::-1])[::-1] * np.diff(f)[1], color='grey',linestyle='--', lw=1, alpha=0.01)

                axx.tick_params(labelsize=13)
                axx.grid()


            for axx2, m in zip(ax2, mirrors  ):

                psd_acc = sig.welch(acc[m], fs=1e3, nperseg=2**11, axis=0) 

                f, psd_pos = double_integrate(psd_acc[0],psd_acc[1])

                if m != 'm1-7':
                    axx2.loglog(f,  psd_pos / psd_pos_m1_7, color='g', lw=1, alpha=0.01)

                    color = 'tab:green'
                    axx2.set_ylabel(f'PSD ratio ({m}/ m1-7)', fontsize=15, color = 'g')
                    axx2.tick_params(axis ='y', labelcolor = 'g')
                    axx2.tick_params(labelsize=13)

    """

    

def plot_psd(file, time_key, sensor, double_integrate=True, save=False):
    """
    plots PSD of 10s sample from MNII accelerometer sensors:
    
    input 
        file name - string (e.g. 'ldlvib{UT}_raw_{yyyy}-{mm}-{dd}.hdf5')
        ... note needs to be raw file and not psd 
        time_key - string (e.g 'hh:mm:ss' ) 
        sensor - str (sensor_{X}, 'm1', 'm1-3' etc ... the keys from process_single_mn2_sample output)
        double_integrate - do we want to double integrate the accelerometer signal? 
    """
    
    plt.figure(figsize=(8,5))
    plt.title(f'{sensor} - {file}\nT{time_key}',fontsize=15)
    
    acc = process_single_mn2_sample(file, time_key, post_upgrade=True)
    
    f, psd = sig.welch(acc[sensor], fs=1e3, nperseg=2**10, axis=0)
    
    if double_integrate:
        f,psd = double_integrate(f,psd)
        if 'sensor' in sensor:
            plt.loglog(f,psd)
            plt.ylabel(r'{s PSD [$V^2/Hz$]',fontsize=15)
            
        else:
            plt.loglog(f,1e12 * psd)
            plt.ylabel(r'{s PSD [$\mu m^2/Hz$]',fontsize=15)
            
    else:
        if 'sensor' in sensor:
            plt.loglog(f,psd)
            plt.ylabel(r'{s PSD [$V^2/s^4/Hz$]',fontsize=15)
            
        else:
            plt.loglog(f,1e12 * psd)
            plt.ylabel(r'{s PSD [$\mu m^2/s^4/Hz$]',fontsize=15)
            

    plt.xlabel(r'frequency [Hz]',fontsize=15)
    plt.legend( fontsize=15)
    plt.gca().tick_params(labelsize=15)
    
    plt.tight_layout()    
    
    if save:

        plt.savefig(f'/home/jovyan/usershared/BCB/mn2_upgrade/psd_{sensor}_{file}-{time_key}.png')
        
    plt.show()

  

        

def compare_file_to_historic_psds_plot(file, time_key, sensor, save=False):
    """
    compares the raw accelerometer PSD from a given sensor to frequency binned percentiles 
    from historic data (typically ~1month). The historic data filtered for samples that match telescope state (eg. guiing ,  that the was realized during the input MNII file w
    
    input:
        file: string (eg. ldlvib{UT}_raw_{YYYY-MM-DD}.hdf5') 
        time_key: string (e.g 'hh:mm:00' hint '23:28:00')
        sensor: int (sensor index, see i2m_dict for label to index mapping)
        
    output:
        loglog plot of historic frequency binned PSD percentiles vs PSD of input MNII file @ time_key. 
        PSDs are calculated from either raw accelerometer signals or piston combined geometry
        
    """
    
    ### ---- prelims
    
    # get UT from file name
    UT = file.split('vib')[-1][0] 
    
    # define telescope environment
    env = 'wt{}tcs'.format(UT)
    
    # define timestamp from file
    file_date = datetime.datetime.strptime( file.split('_')[-1].split('.')[0] +'T'+ time_key,'%Y-%m-%dT%H:%M:%S') 

    #label for the sensor/mirror
    mirror_label = i2m_dict[sensor]


    ### ---- load historic telescope states and psds
    
    # dataframe with the given UT (historic) states (these are read in from MNII_bigdata_processing.ipynb)
    states =  pd.read_csv( f'/home/jovyan/usershared/BCB/2022/08/UT{UT}/states.csv' ,index_col=0)
    
    # dataframe with the given UT (historic)  psds (these are read in from MNII_bigdata_processing.ipynb)
    psds = pd.read_csv( f'/home/jovyan/usershared/BCB/2022/08/UT{UT}/sensor_{sensor}_psds.csv',index_col=0 )
    
    #read frequencies as float
    f = np.array( [float(x) for x in psds.columns] )
    
    
    ### ---- Define telescope states of input file
    
    #periods of telescope guiding
    guiding_periods = dlt.query_guide_times(env, file_date - datetime.timedelta(days=1), file_date + datetime.timedelta(days=1))

    #periods of enclosure open  
    open_enc_periods = dlt.query_enclosure_times(env, file_date - datetime.timedelta(days=1), file_date + datetime.timedelta(days=1))

    # we dont consider presetting at the moment because I don't trust it sometimes it reports telescopes presetting for > 30min...
    
    # put these as boolean values in dictionary that match keys of the historic 'states' dataframe
    file_state = {}
    file_state['open_enclosure'] = bool( is_between(file_date, open_enc_periods) )
    file_state['guiding'] =  bool( is_between(file_date, guiding_periods) )

    print('\n--------------\nthe telescope state for the input file is:\n', file_state,'\n--------------')
    
    ### ---- Process the current file at the given time_key to get the 
    
    # !!! always confirm 'process_single_mn2_sample()' matches the processing done in MNII_bigdata_processing.ipynb to generate historic PSDs  !!!
    acc = process_single_mn2_sample(file , time_key, post_upgrade=False) 

    #take PSD of input files processed accelerometer signal
    f_n, psd_n = sig.welch(acc[f'sensor_{sensor}'], fs=1e3, nperseg=2**10, axis=0)
    
    #filter historic PSDs for those that match the same state as input file
    state_filter = (states['open_enclosure']==file_state['open_enclosure']) \
                        & (states['guiding']==file_state['guiding']) #(states['presetting']==file_state['presetting'])
    
    # get percentiles of historic psds
    y2 = np.nanquantile( psds[state_filter] ,0.9,axis=0)
    y1 = np.nanquantile( psds[state_filter] ,0.1,axis=0)

    plt.figure(figsize=(10,6))
    
    plt.loglog(f_n, psd_n, color='k',label='input file ({} @ {})'.format(file.split('/')[-1],time_key))
    
    plt.loglog(f, y1 ,color='k',lw=0.3)
    plt.loglog(f, y2 ,color='k',lw=0.3)
    plt.fill_between(f, y1, y2, color='green',alpha=0.5,label=f'historic 10-90 percentile range \n{min(psds.index)[:10]} - {max(psds.index)[:10]}')
    plt.title(f'UT{UT}, sensor {sensor} ({mirror_label})',fontsize=15)
    plt.gca().tick_params(labelsize=15)
    plt.ylabel('raw accelerometer\n'+r'signal PSD [$V^2/s^4/Hz$]',fontsize=15)
    plt.xlabel(r'frequency [Hz]',fontsize=15)
    plt.legend( fontsize=15)
    
    plt.xlim([1e-1,500])
    plt.ylim([1e-6,1e1]) 
    plt.grid()
    plt.tight_layout()
    
    plt.show()
    
    if save:
        plt.savefig(f'/home/jovyan/usershared/BCB/mn2_upgrade/psd_comparison_{file_date}-UT{UT}-sensor{sensor}.png')





def get_telescope_states_and_flags(env, initial_date, final_date, state_switches, flag_switches):    
    """
    NOTE: dlt.query_logtext_elastic has been updated so probably wont work here anymore 
    
    get telescope states and flags
    
    state - 
    values can either be on or off (1-0) depending on if the telescope is or is not 
    in the given state. descibed by *ing words (e.g. presetting, updating, offsetting, etc)
    
    flag - 
    A timestamp where some event has occured. (e.g. bad guiding, lost guide star, etc). 


    input
    =====
    env - 
        string corresponding to telescope environment (e.g. env = 'wt3tcs')
    
    
    initial_date - 
        initial date to obtain states and flags (datetime)
    
    
    final_date  - 
        final date to obtain states and flags (datetime) 
    
    
    state_switches - 
        A dictionary that has tuples with the strings (that we search the in the logFile)
        which indicates on or off of a particular state - given by the respective dictionary key 
        e.g. state_switches = {state_name_1 : ('string that turns state_1 on', 'string that turns state_1 off'), state_2: .. }


    flag_switches - 
        A dictionary with the flag names (keys) and string that is searched in the logtext
        to activate the flag (get timestamp)
        e.g. flag_switches = {flag_name_1 : 'string the activates flag_1'}
    
    output
    =====
    state_dict -  
        A dictionary keyed by state name (defined by state_switches keys) with panda dataframe
        values. The dataframe has columns = ['time','state_value'] and numeric indicies.
        
        
    flag_dict - 
        A dictionary keyed by a given flag (defined by flag_switches keys) with numpy array 
        values. The array's contain timestamp where the flag was activated.
          
    """
    
    state_dict = {} #
    for state in state_switches: 
    
        # time stamps where the state is turned on 
        ones = dlt.query_logtext_elastic(env, state_switches[state][0], initial_date, final_date)['time'].values
        # time stamps where the state is turned off
        zeros = dlt.query_logtext_elastic(env, state_switches[state][1], initial_date, final_date)['time'].values

        #=======
        #merge timestamps where we have flag changes and sort by timestamps, then append to our state dictionary (e.g. [[time, state], [YYYY-MM-DDTHH:mm:ss1, 0] , [YYYY-MM-DDTHH:mm:ss2, 1],... ]   etc)
        state_dict[state]=pd.DataFrame( sorted( {**{x:0 for x in zeros} , **{x:1 for x in ones}}.items() ) ,\
                                       columns = ['time','state_value'])
        #=======
    
    flag_dict = {} 
    for flag in flag_switches: 
    
        # time stamps where the flag is turned on 
        flag_dict[flag] = dlt.query_logtext_elastic(env, flag_switches[flag], initial_date, final_date)['time'].values
        
    return(state_dict, flag_dict)


def get_telescope_flags(  env, flag_identifiers, initial_date, final_date ) :
    """
    same as get_telescope_states_and_flags but updated to avoid dlt.query_logtext_elastic and only deals with flags (could also do states if you want) 
    """
    flag_dict = {} 
    for flag in flag_identifiers: 

        # time stamps where the flag is turned on 
        t_tmp = dlt.query_elastic_logs( f'{env} AND '+flag_identifiers[flag] , initial_date, final_date)
        
        
        if not t_tmp.empty:
            t_tmp = t_tmp['@timestamp'].unique() 
            # convert timestamp string to datetime and store in dictionary
            flag_dict[flag] = [dlt.get_datetime(x) for x in t_tmp]
            
        else:
            flag_dict[flag] = []
        
    return(flag_dict)



def get_active_state_periods(state_dict, state):
    """
    get initial and final timestamps between where tbe state was turned on AND sucessfully terminated 
    (without repeating the ON process again - i.e. preset started then subsequently finished)
    NOTE: this is different from dlt functions that don't check if repeated the on action'
    from here we can use directly dlt.filter_dataframe_by_time_ranges() function to furhter filter our dataframes


    more specifically we look for periods where the flag was turned on (flag = 1) and then the subsequently 
    turned off (flag=0), thereby filtering out where a flag was turned on and then turned on again (i.e. a bad preset that doesnt complete)
    in this case the difference in subsequent flags is: state_value[i] - state_value[i+1] = 1-0 = 1. so we filter like this.
    
    we keep periods where 
        flag[i] - flag[i+1] = 1-0 = 1
    we filter out periods where 
        state_value[i] - state_value[i+1] = 1-1 = 0, 
        state_value[i] - state_value[i+1] = 0-1 = -1, 
        state_value[i] - state_value[i+1] = 0-0 = 0, 
    
    """
        
    indx = np.where(state_dict[state]['state_value'][:-1].values - state_dict[state]['state_value'][1:].values == 1)
    
    initial_time = state_dict[state].iloc[indx[0]]['time'].values
    final_time = state_dict[state].iloc[indx[0]+1]['time'].values
    
    return(pd.DataFrame({'initial_time':initial_time,'final_time':final_time}) )

    



def get_features(init_df, feature_identifiers, env, initial_time, final_time, buffer = datetime.timedelta(minutes=1)):
    """
    Parameters:
        init_df: pandas dataframe 
            initial pandas dataframe to merge the features with, this MUST have a column called 'time' with datetime entries to do the merging / interpolation to  
        
        env: str 
            telescope environment (e.g. 'wt1tcs' for UT1 environment) 
        initial_time: datetime 
            extract features for merging to init_df after this initial_time , 
        final_time: datetime 
            extract features for merging to init_df before this final_time 
        buffer: datetime.timedelta
            timebuffer for searching beyond final_time and intial_time 
    Returns:
        dataframe with features interpolated to the input 'time' timestamps from init_df
    """
    
     # --- FEATURES (this could be a user input..) 

    #feature_identifiers = { 'alt[deg]':['TEL.ALT', 'TEL.ALT.POS', 'TEL.ACTO.ALTPOS'], 'az[deg]':['TEL.AZ', 'TEL.AZ.POS', 'TEL.ACTO.AZPOS'] ,\
    #                 'adapter_pos[deg]':['TEL.AD.POS'], 'cas_rot_pos[deg]':['TEL.ROT4.POS'],'nasA_rot_pos[deg]':['TEL.ROT2.POS'],'nasB_rot_pos[deg]':['TEL.ROT3.POS'],\
    #                 'hbs_oil_pressure[bar]':['TEL.HBS.PVM2'], 'windsp[m/s]':['TEL.AMBI.WINDSP'],'wind_direction[0=N]':['TEL.AMBI.WINDDIR'] }    

    # extract features from logtext
    feature_dict = {} # to hold feature dataframes returned for each query from feature_identifiers dictionary
    for k,v in feature_identifiers.items():

        feature_df_tmp = dlt.query_tss(env, v, initial_time-buffer, final_time+buffer)

        if isinstance(v, list) &  (len(v)==1): 
            feature_dict[k] = feature_df_tmp[0]
            feature_dict[k] = feature_dict[k].rename(columns={'value':k})

        elif isinstance(v, list) & (len(v)>1): # if we have more then 1 keyword identifier for the feature then we merge all instances 
            feature_dict[k] = dlt.merge_timeseries(feature_df_tmp)
            feature_dict[k] = feature_dict[k].rename(columns={'value':k})

        else:
            print('something went wrong when searching for {k} with keyword {v}')
            feature_dict[k] = np.nan

    # now merge features to common dataframe and interpolate onto center timestamp of MNII sample with a 5 minute interpolation tolerance.
    feature_df =  init_df.copy()
    for k in feature_dict:
        if len(feature_dict[k])>0:
            feature_df = pd.merge_asof(feature_df, feature_dict[k], on = 'time', direction = 'nearest', tolerance=pd.Timedelta("5min"))
        else:
            feature_df[k] = pd.DataFrame( np.nan * np.zeros(len(feature_df)))

    return( feature_df )



def get_states( sampled_df, flag_identifiers, env, initial_time, final_time , buffer = datetime.timedelta(days=1) ):
    """
    Parameters:
        sampled_df: pandas dataframe 
            holds start and finish periods of relavent sample (e.g. 10s MNII sample) that we want to calculate the state for. this MUST have
            'initial_time' and 'final_time' columns with datetime formats. Output state_df will hold a 'time' column which will be equal to 
            sampled_df['initial_time'] where state values will be calculated
        env: str 
            telescope environment (e.g. 'wt1tcs' for UT1 environment) 
        initial_time: datetime 
            extract states for merging to init_df after this initial_time , 
        final_time: datetime 
            extract states for merging to init_df before this final_time 
        buffer: datetime.timedelta
            timebuffer for searching beyond final_time and intial_time (this sometimes helps with boundary problems)
    Returns:
        state_df: pandas dataframe
            dataframe with states calculated within sampled_df initial and final times, timestamped with initial_times 
    """
    
    #initiate pandas dataframe with timestamps to calculate the state
    init_df = pd.DataFrame(  sampled_df['initial_time'].values, columns=['time'] )
    
    # dictionary with flag label and query text to filter if that state flag occured 
    #flag_identifiers = { 'bad_guiding' :'agERR_CCD AND error AND vectors AND intesity AND zero',\
    #               'AS_update' : 'm1asSetGlbRel AND AS AND Forces',\
    #                'PS_update' : 'm1psMovMiAPos AND PS AND Received AND cmd' }

    # ------- basic states established in dlt 
    state_dict = {} 
    #periods of telescope guiding
    state_dict['guiding'] = dlt.query_guide_times(env, initial_time-buffer, final_time+buffer)

    #periods of telescope presets 
    not_presetting_df = dlt.query_preset_times(env, initial_time-buffer, final_time+buffer, gap=1) 
    # note this initial time and final time for non-presetting periods.. presetting period are between these so we have to re-arrange (this is clearer for me!)
    state_dict['presetting'] = pd.DataFrame( {'initial_time': not_presetting_df['final_time'].values[:-1], 'final_time': not_presetting_df['initial_time'].values[1:] } )

    #periods of enclosure open  
    state_dict['enc_open'] = dlt.query_enclosure_times(env, initial_time-buffer, final_time+buffer)

    #per
    focus_periods = dlt.query_focus_times(env, initial_time-buffer, final_time+buffer, names=True)
    #focus_periods['focus_name'] = [dlt.get_focus_name(focus_periods['focus'][i]) for i in range(len(focus_periods))]

    state_dict['coude_focus'] = focus_periods[focus_periods['focus'] == 'Coude']
    state_dict['nasA_focus'] = focus_periods[focus_periods['focus'] == 'Nasmyth A']
    state_dict['nasB_focus'] = focus_periods[focus_periods['focus'] == 'Nasmyth B']
    state_dict['cas_focus'] = focus_periods[focus_periods['focus'] == 'Cassegrain']

    state_df = init_df.copy()

    for k in state_dict:
        
        state_df[k] = [ is_between(time, state_dict[k]) for time in init_df['time'] ] 
    
    # ------- more custom states using flags 
    
    # getting timestamps where a flag (defined in flag_identifiers dictionary) is raised in logs. This is stored in dictionary index by flag_identifiers.keys()
    flag_dict = get_telescope_flags( env, flag_identifiers, initial_time, final_time )

    for k in flag_dict:

        # MNII data only taken during first 10s every minute, so we can filter from this knowledge which makes alot faster
        which_less_ten_s = np.array( [x.second for x in flag_dict[k]] ) < 10
        
        #filter timestamps
        times_tmp = np.array( [pd.Timestamp(x) for x in flag_dict[k]] )[which_less_ten_s]

        # initialize all values to zero for new key
        state_df[k] = np.zeros(len(state_df)).astype(int) 

        for t in times_tmp:
            # get index of nearest MN2 sample to current flag timestamp
            loc_name = sampled_df.index[ np.argmin( abs( t - sampled_df['initial_time'] + datetime.timedelta(seconds=5) ) ) ]
            # if flag stamp lays within the MN2 sampling period then change state (of flag) to 1 in the dataframe
            if (sampled_df['initial_time'].loc[loc_name] <= t) & (sampled_df['final_time'].loc[loc_name] >= t):

                state_df.loc[loc_name, k] = 1  # raise the flag 

    return( state_df )


def get_state(UT, date): # I use this one for more simple purposes.. there is a more detailed function called get_states if you what more detail! ^^ above.. 
    """
    input:
        UT = string - e.g. input '1' for UT1 
        date = string - 'YYYY-MM-DDTHH:MM:SS'
    output:
        dictionary with states
    """

    input_datetime =  datetime.datetime.strptime( date, '%Y-%m-%dT%H:%M:%S') 

    file = '/datalake/rawdata/vlt-vibration/2022/08/ldlvib2_raw_{}.hdf5'.format(date.split('T')[0])
    time_key = date.split('T')[-1]

    env = f'wt{UT}tcs'
    #format datetime 

    initial_time , final_time = input_datetime - datetime.timedelta(days=1), input_datetime + datetime.timedelta(days=1)

    #periods of telescope guiding
    guiding_periods = dlt.query_guide_times(env, initial_time, final_time)

    #periods of enclosure open  
    open_enc_periods = dlt.query_enclosure_times(env, initial_time, final_time)

    # we dont consider presetting at the moment because I don't trust it sometimes it reports telescopes presetting for > 30min...
    # focus periods 
    focus_periods = dlt.query_focus_times(env, initial_time, final_time, names=False)
    focus_periods['focus_name'] = [dlt.get_focus_name(focus_periods['focus'][i]) for i in range(len(focus_periods))]

    focus = get_focus(input_datetime , focus_periods)

    # put these as boolean values in dictionary that match keys of the historic 'states' dataframe
    file_state = {}
    file_state['open_enclosure'] = bool( is_between(input_datetime, open_enc_periods) )
    file_state['guiding'] =  bool( is_between(input_datetime, guiding_periods) )

    for foc_tmp in ['Coude', 'Nasmyth A', 'Nasmyth B', 'Cassegrain']:
        file_state[foc_tmp] = focus==foc_tmp
        
    return(file_state)


def big_feature_processing(UT_list, start_date, end_date, feature_identifiers ,flag_identifiers,  base_dir, jumps = 10, post_upgrade=False, write_files= False, overwrite = True):
    
    return_dict = {}
    
    for UT in UT_list:
        
        return_dict[UT] = {}
        print(f'---{UT}---\n')

        #defiine our telescope environment
        env = 'wt{}tcs'.format(UT[2])

        # generate a list of dates to process between the start and end date
        dates_2_look = [start_date + datetime.timedelta(days=x) for x in range((end_date - start_date).days+1)]  

        #initialize the current month and indx (used for checking if we add to exisiting csv monthly file or start new one, for write_files=True
        current_month = np.nan
        for ii, d in enumerate( dates_2_look ):

            # ---- CHECK FILE EXISTS for current date
            try:
                file = vib_path.format(ut=UT[2], year=d.year, month=d.month, day=d.day)
                mn2_data = h5py.File(file , 'r')
            except: #just skip to next day
                print(f'no file found for {UT} on {d.year}-{d.month:02}-{d.day:02}')
                continue # go to next day 

            # updating current_month and month_indx which is used to check if we add df to existing csv monthly file or if we start a new one when overwrite=True
            # need to be careful of case where there is no data on the first few days of month  (make sure this section is below data check condition) 
            if d.month == current_month:
                month_indx += 1
            elif d.month != current_month:
                current_month = d.month
                month_indx = 0 

            # directory where we will write files to if write_files==True
            write_dir = base_dir + f'{d.year}/{d.month:02}/{UT}/'

            # now make the directory if it doesn't exist
            if not os.path.exists( write_dir ):
                os.makedirs( write_dir )

            # current date we are processing as string
            date_tmp = f'{d.year}-{d.month:02}-{d.day:02}'
            print(f'\n.  ---looking at {date_tmp}')

            # filter keys based on how many we want to skip (jump)
            mn2_keys = list( mn2_data.keys() )[::jumps] 

            # MN2 data timestamps from daily file 
            mn2_t0_s = np.array( [ datetime.datetime.strptime( date_tmp +'T'+ t ,'%Y-%m-%dT%H:%M:%S') for t in mn2_keys ] )
            mn2_t1_s = mn2_t0_s + datetime.timedelta(seconds = 10) #10s samples every minute 
            # create our data frame with MNII sample start and end times (this will be used for extracting our states and features)
            mn2_timestamps_df = pd.DataFrame({'initial_time':mn2_t0_s, 'final_time':mn2_t1_s}) # to be compatiple with is_between(time, df) 

            # now need function to get state and feature df with these timestamps 
            initial_time = min( mn2_t0_s )
            final_time = max( mn2_t1_s )

            init_df = pd.DataFrame(mn2_t0_s,  columns=['time']) # at beggining of sample.. should match mn2_timestamps_df initial timestamps

            # ------- Get FEATURES 
            feature_df = get_features( init_df.copy(), feature_identifiers, env, initial_time, final_time , buffer = datetime.timedelta(minutes=10))
            feature_df.index = feature_df['time']

            if write_files: 
                if overwrite:# we begin by writing a new file 
                    if month_indx == 0:
                        feature_df.to_csv(write_dir + f'features_{UT}_{d.year}-{d.month:02}.csv')
                    else: # after 1st write we just keep adding 
                        feature_df.to_csv(write_dir + f'features_{UT}_{d.year}-{d.month:02}.csv',mode = 'a',index=True, header=False)

                if not overwrite: # we just keep adding to existing file
                    if os.path.exists( write_dir + f'features_{UT}_{d.year}-{d.month:02}.csv'  ):  #first check if exists (should we write the header?)
                        features_df.to_csv(write_dir + f'features_{UT}_{d.year}-{d.month:02}.csv', index=True, header=True)
                    else:
                        feature_df.to_csv(write_dir + f'features_{UT}_{d.year}-{d.month:02}.csv',mode = 'a',index=True, header=False)
        
            return_dict[UT][d] = feature_df
        
    return(return_dict) 



def big_state_processing(UT_list, start_date, end_date, feature_identifiers ,flag_identifiers,  base_dir, jumps = 10, post_upgrade=False, write_files= False, overwrite = True):
    
    return_dict = {} 
    for UT in UT_list:

        return_dict[UT] = {}
        print(f'---{UT}---\n')

        #defiine our telescope environment
        env = 'wt{}tcs'.format(UT[2])

        # generate a list of dates to process between the start and end date
        dates_2_look = [start_date + datetime.timedelta(days=x) for x in range((end_date - start_date).days+1)]  

        #initialize the current month and indx (used for checking if we add to exisiting csv monthly file or start new one, for write_files=True
        current_month = np.nan
        for ii, d in enumerate( dates_2_look ):

            # ---- CHECK FILE EXISTS for current date
            try:
                file = vib_path.format(ut=UT[2], year=d.year, month=d.month, day=d.day)
                mn2_data = h5py.File(file , 'r')
            except: #just skip to next day
                print(f'no file found for {UT} on {d.year}-{d.month:02}-{d.day:02}')
                continue # go to next day 

            # updating current_month and month_indx which is used to check if we add df to existing csv monthly file or if we start a new one when overwrite=True
            # need to be careful of case where there is no data on the first few days of month  (make sure this section is below data check condition) 
            if d.month == current_month:
                month_indx += 1
            elif d.month != current_month:
                current_month = d.month
                month_indx = 0 

            # directory where we will write files to if write_files==True
            write_dir = base_dir + f'{d.year}/{d.month:02}/{UT}/'

            # now make the directory if it doesn't exist
            if not os.path.exists( write_dir ):
                os.makedirs( write_dir )

            # current date we are processing as string
            date_tmp = f'{d.year}-{d.month:02}-{d.day:02}'
            print(f'\n.  ---looking at {date_tmp}')

            # filter keys based on how many we want to skip (jump)
            mn2_keys = list( mn2_data.keys() )[::jumps] 

            # MN2 data timestamps from daily file 
            mn2_t0_s = np.array( [ datetime.datetime.strptime( date_tmp +'T'+ t ,'%Y-%m-%dT%H:%M:%S') for t in mn2_keys ] )
            mn2_t1_s = mn2_t0_s + datetime.timedelta(seconds = 10) #10s samples every minute 
            # create our data frame with MNII sample start and end times (this will be used for extracting our states and features)
            mn2_timestamps_df = pd.DataFrame({'initial_time':mn2_t0_s, 'final_time':mn2_t1_s}) # to be compatiple with is_between(time, df) 

            # now need function to get state and feature df with these timestamps 
            initial_time = min( mn2_t0_s )
            final_time = max( mn2_t1_s )

            # ------- Get STATES 
            state_df = get_states( mn2_timestamps_df, flag_identifiers, env, initial_time, final_time , buffer =  datetime.timedelta(days=1))
            state_df.index = state_df['time']
            
            if write_files: 
                if overwrite:# we begin by writing a new file 
                    if month_indx == 0:
                        state_df.to_csv(write_dir + f'states_{UT}_{d.year}-{d.month:02}.csv')
                        
                    else: # after 1st write we just keep adding 
                        state_df.to_csv(write_dir + f'states_{UT}_{d.year}-{d.month:02}.csv', mode = 'a',index=True, header=False)

                if not overwrite: # we just keep adding to existing file 
                    if os.path.exists( write_dir + f'states_{UT}_{d.year}-{d.month:02}.csv'  ): #first check if exists (should we write the header?)
                        state_df.to_csv(write_dir + f'states_{UT}_{d.year}-{d.month:02}.csv', index=True, header=True)
                    else: 
                        state_df.to_csv(write_dir + f'states_{UT}_{d.year}-{d.month:02}.csv', mode = 'a',index=True, header=False)
        
            return_dict[UT][d] = state_df
        
    return(return_dict)


def big_psd_processing(UT_list, start_date, end_date, feature_identifiers ,flag_identifiers,  base_dir, jumps = 10, post_upgrade=False, write_files=False, overwrite = True):
    
    return_dict = {}
    
    for UT in UT_list:
        
        return_dict[UT] = {}
        
        print(f'---{UT}---\n')

        #defiine our telescope environment
        env = 'wt{}tcs'.format(UT[2])

        # generate a list of dates to process between the start and end date
        dates_2_look = [start_date + datetime.timedelta(days=x) for x in range((end_date - start_date).days+1)]  

        #initialize the current month and indx (used for checking if we add to exisiting csv monthly file or start new one, for write_files=True
        current_month = np.nan
        for ii, d in enumerate( dates_2_look ):

            # ---- CHECK FILE EXISTS for current date
            try:
                file = vib_path.format(ut=UT[2], year=d.year, month=d.month, day=d.day)
                mn2_data = h5py.File(file , 'r')
            except: #just skip to next day
                print(f'no file found for {UT} on {d.year}-{d.month:02}-{d.day:02}')
                continue # go to next day 

            # updating current_month and month_indx which is used to check if we add df to existing csv monthly file or if we start a new one when overwrite=True
            # need to be careful of case where there is no data on the first few days of month  (make sure this section is below data check condition) 
            if d.month == current_month:
                month_indx += 1
            elif d.month != current_month:
                current_month = d.month
                month_indx = 0 

            # directory where we will write files to if write_files==True
            write_dir = base_dir + f'{d.year}/{d.month:02}/{UT}/'

            # now make the directory if it doesn't exist
            if not os.path.exists( write_dir ):
                os.makedirs( write_dir )

            # current date we are processing as string
            date_tmp = f'{d.year}-{d.month:02}-{d.day:02}'
            print(f'\n.  ---looking at {date_tmp}')

            # filter keys based on how many we want to skip (jump)
            mn2_keys = list( mn2_data.keys() )[::jumps] 

            # MN2 data timestamps from daily file 
            mn2_t0_s = np.array( [ datetime.datetime.strptime( date_tmp +'T'+ t ,'%Y-%m-%dT%H:%M:%S') for t in mn2_keys ] )
            mn2_t1_s = mn2_t0_s + datetime.timedelta(seconds = 10) #10s samples every minute 
            # create our data frame with MNII sample start and end times (this will be used for extracting our states and features)
            mn2_timestamps_df = pd.DataFrame({'initial_time':mn2_t0_s, 'final_time':mn2_t1_s}) # to be compatiple with is_between(time, df) 


            # ------- Get ACCELEROMETER data 

            # note that units for raw sensors (sensor_X) are in volts, piston combined geometries (m1,m2,m3.. m1-3 etc) are calibrated in m/s^2
            acc_dict = {ts : process_single_mn2_sample(file, k, post_upgrade=False, user_defined_geometry=None, outlier_thresh = 100, replace_value=0,ensure_1ms_sampling=True) for ts, k in zip(mn2_t0_s, mn2_keys)}                          #mn2.process_daily_MN2_file(file, verbose=False, post_upgrade=post_upgrade)

            # ------- Get ACC HEALTH
            # for each sensor at each sample calculate power law slope for low / high freq , also number of amp positions #  
            #update state_df if accelerometers saturate

            acc_health_df = {} #init df as dictionary to be built

            outlier_keys = [s for s in acc_dict[mn2_t0_s[0]].keys() if 'no_outliers_replaced' in s]

            for sensor in outlier_keys:
                tmp_list=[]
                for t in acc_dict:   
                    tmp_list = tmp_list + acc_dict[t][sensor]                    
                acc_health_df[sensor] = tmp_list # how many outliers were replaced.. indicates amplifier saturation (see tutorial_MNII_functions.ipynb)


            # ------- Get PSDs (also PSD power law fits to put in accelerometer health df)
            sensor_keys = [s for s in acc_dict[mn2_t0_s[0]].keys() if 'no_outliers_replaced' not in s]

            for sensor in sensor_keys:
                f,psd = sig.welch(np.array( [acc_dict[t][sensor] for t in acc_dict] ), fs=1e3, nperseg=nperseg, axis=1) 
                psd_df = pd.DataFrame(psd, columns= f, index= acc_dict.keys() )
                
                return_dict[UT][sensor] = psd_df 
                
                # here we also fit power laws to each PSD ( beta f^(alpha) ) note we do log fit (log(PSD) = alpha * log(f) + beta, cov[0,0] corresponds to alpha covariance, cov[1,1] to beta covariance
                lin_fit_dict = {'alpha' : [] , 'beta' : [], 'cov_00' : [], 'cov_01' : [], 'cov_10' : [], 'cov_11' : []}
                for row in psd_df.index:

                    param, cov = np.polyfit(np.log10(f[1:]), np.log10(psd_df.loc[row].values[1:]), 1, cov=True) 

                    lin_fit_dict['alpha'].append( param[0] )
                    lin_fit_dict['beta'].append( param[1] )
                    lin_fit_dict['cov_00'].append( cov[0,0] )
                    lin_fit_dict['cov_01'].append( cov[0,1] )
                    lin_fit_dict['cov_10'].append( cov[1,0] )
                    lin_fit_dict['cov_11'].append( cov[1,1] )

                for pp in lin_fit_dict:
                    acc_health_df[f'{sensor}_powerlaw_fit_{pp}'] = lin_fit_dict[pp]


                # Writing PSDs
                if write_files: # only do if we want to write results to csv files
                    if overwrite:# we begin by writing a new file 
                        if month_indx == 0: # we overwrite on 1st iteration 
                            psd_df.to_csv(write_dir + f'{sensor}_psds_{UT}_{d.year}-{d.month:02}.csv')

                        else: # then keep adding to this same file (without re-writing headers) 
                            psd_df.to_csv(write_dir + f'{sensor}_psds_{UT}_{d.year}-{d.month:02}.csv', mode = 'a',index=True, header=False)

                    if not overwrite: # we just keep adding to existing file 
                        if os.path.exists( write_dir + f'{sensor}_psds_{UT}_{d.year}-{d.month:02}.csv'  ): #first check if exists (should we write the header?)
                            psd_df.to_csv(write_dir + f'{sensor}_psds_{UT}_{d.year}-{d.month:02}.csv', index=True, header=True)
                        else: 
                            psd_df.to_csv(write_dir + f'{sensor}_psds_{UT}_{d.year}-{d.month:02}.csv', mode = 'a',index=True, header=False)


            # finally put acc_health_df in dataframe 
            acc_health_df = pd.DataFrame( acc_health_df, index= acc_dict.keys() )

            # Writing accelerometer health 
            if write_files: # only do if we want to write results to csv files
                if overwrite:# we begin by writing a new file 
                    if month_indx == 0: # we overwrite on 1st iteration 
                        acc_health_df.to_csv(write_dir + f'accelerometer_HC_{UT}_{d.year}-{d.month:02}.csv')

                    else: # then keep adding to this same file (without re-writing headers) 
                        acc_health_df.to_csv(write_dir + f'accelerometer_HC_{UT}_{d.year}-{d.month:02}.csv', mode = 'a',index=True, header=False)

                if not overwrite: # we just keep adding to existing file 
                    if os.path.exists( write_dir + f'accelerometer_HC_{UT}_{d.year}-{d.month:02}.csv'  ): #first check if exists (should we write the header?)
                        acc_health_df.to_csv(write_dir + f'accelerometer_HC_{UT}_{d.year}-{d.month:02}.csv', index=True, header=True)
                    else: 
                        acc_health_df.to_csv(write_dir + f'accelerometer_HC_{UT}_{d.year}-{d.month:02}.csv', mode = 'a',index=True, header=False)


    return(return_dict)
                        
                        
                        
                        
def process_n_write_mnII_features_states_psds(UT_list, start_date, end_date, feature_identifiers ,flag_identifiers,  base_dir, jumps = 10, post_upgrade=False, overwrite = True):
    
    # this does not return anything - just used for saving files in a way that occupies minimum RAM while processing (only holds data for 1 day maximum) 
    
    print( '\n---------------------------^^^---------------------------\n   *** processing features, states, HC, PSDs ***\n---------------------------^^^---------------------------\n')
    
    # we always write to file if this function is called 
    write_files = True
    
    for UT in UT_list:

        print(f'\n---{UT}---\n')

        #defiine our telescope environment
        env = 'wt{}tcs'.format(UT[2])

        # generate a list of dates to process between the start and end date
        dates_2_look = [start_date + datetime.timedelta(days=x) for x in range((end_date - start_date).days+1)]  

        #initialize the current month and indx (used for checking if we add to exisiting csv monthly file or start new one, for write_files=True
        current_month = np.nan
        for ii, d in enumerate( dates_2_look ):

            # ---- CHECK FILE EXISTS for current date
            try:
                file = vib_path.format(ut=UT[2], year=d.year, month=d.month, day=d.day)
                mn2_data = h5py.File(file , 'r')
            except: #just skip to next day
                print(f'no file found for {UT} on {d.year}-{d.month:02}-{d.day:02}')
                continue # go to next day 

            # updating current_month and month_indx which is used to check if we add df to existing csv monthly file or if we start a new one when overwrite=True
            # need to be careful of case where there is no data on the first few days of month  (make sure this section is below data check condition) 
            if d.month == current_month:
                month_indx += 1
            elif d.month != current_month:
                current_month = d.month
                month_indx = 0 

            # directory where we will write files to if write_files==True
            write_dir = base_dir + f'{d.year}/{d.month:02}/{UT}/'

            # now make the directory if it doesn't exist
            if not os.path.exists( write_dir ):
                os.makedirs( write_dir )

            # current date we are processing as string
            date_tmp = f'{d.year}-{d.month:02}-{d.day:02}'
            print(f'\n.  ---looking at {date_tmp}')

            # filter keys based on how many we want to skip (jump)
            mn2_keys = list( mn2_data.keys() )[::jumps] 

            # MN2 data timestamps from daily file 
            mn2_t0_s = np.array( [ datetime.datetime.strptime( date_tmp +'T'+ t ,'%Y-%m-%dT%H:%M:%S') for t in mn2_keys ] )
            mn2_t1_s = mn2_t0_s + datetime.timedelta(seconds = 10) #10s samples every minute 
            # create our data frame with MNII sample start and end times (this will be used for extracting our states and features)
            mn2_timestamps_df = pd.DataFrame({'initial_time':mn2_t0_s, 'final_time':mn2_t1_s}) # to be compatiple with is_between(time, df) 

            # now need function to get state and feature df with these timestamps 
            initial_time = min( mn2_t0_s )
            final_time = max( mn2_t1_s )

            init_df = pd.DataFrame(mn2_t0_s,  columns=['time']) # at beggining of sample.. should match mn2_timestamps_df initial timestamps

            # ------- Get FEATURES 
            feature_df = get_features( init_df.copy(), feature_identifiers, env, initial_time, final_time , buffer = datetime.timedelta(minutes=10))
            feature_df.index = feature_df['time']
            feature_df = feature_df.drop(columns='time')
            
            # ------- Get STATES 
            state_df = get_states( mn2_timestamps_df, flag_identifiers, env, initial_time, final_time , buffer =  datetime.timedelta(days=1))
            state_df.index = state_df['time']
            state_df = state_df.drop(columns='time')
            
            if write_files: 
                if overwrite:# we begin by writing a new file 
                    if month_indx == 0:
                        state_df.to_csv(write_dir + f'states_{UT}_{d.year}-{d.month:02}.csv')
                        feature_df.to_csv(write_dir + f'features_{UT}_{d.year}-{d.month:02}.csv')
                    else: # after 1st write we just keep adding 
                        state_df.to_csv(write_dir + f'states_{UT}_{d.year}-{d.month:02}.csv', mode = 'a',index=True, header=False)
                        feature_df.to_csv(write_dir + f'features_{UT}_{d.year}-{d.month:02}.csv',mode = 'a',index=True, header=False)

                if not overwrite: # we just keep adding to existing file 
                    if os.path.exists( write_dir + f'states_{UT}_{d.year}-{d.month:02}.csv'  ): #first check if exists (should we write the header?)
                        state_df.to_csv(write_dir + f'states_{UT}_{d.year}-{d.month:02}.csv', index=True, header=True)
                    else: 
                        state_df.to_csv(write_dir + f'states_{UT}_{d.year}-{d.month:02}.csv', mode = 'a',index=True, header=False)

                    if os.path.exists( write_dir + f'features_{UT}_{d.year}-{d.month:02}.csv'  ):  #first check if exists (should we write the header?)
                        features_df.to_csv(write_dir + f'features_{UT}_{d.year}-{d.month:02}.csv', index=True, header=True)
                    else:
                        feature_df.to_csv(write_dir + f'features_{UT}_{d.year}-{d.month:02}.csv',mode = 'a',index=True, header=False)


            # ------- Get ACCELEROMETER data 

            # note that units for raw sensors (sensor_X) are in volts, piston combined geometries (m1,m2,m3.. m1-3 etc) are calibrated in m/s^2
            acc_dict = {ts : process_single_mn2_sample(file, k, post_upgrade=False, user_defined_geometry=None, outlier_thresh = 100, replace_value=0,ensure_1ms_sampling=True) for ts, k in zip(mn2_t0_s, mn2_keys)}                          #mn2.process_daily_MN2_file(file, verbose=False, post_upgrade=post_upgrade)

            # ------- Get ACC HEALTH
            # for each sensor at each sample calculate power law slope for low / high freq , also number of amp positions #  
            #update state_df if accelerometers saturate

            acc_health_df = {} #init df as dictionary to be built

            outlier_keys = [s for s in acc_dict[mn2_t0_s[0]].keys() if 'no_outliers_replaced' in s]

            for sensor in outlier_keys:
                tmp_list=[]
                for t in acc_dict:   
                    tmp_list = tmp_list + acc_dict[t][sensor]                    
                acc_health_df[sensor] = tmp_list # how many outliers were replaced.. indicates amplifier saturation (see tutorial_MNII_functions.ipynb)


            # ------- Get PSDs (also PSD power law fits to put in accelerometer health df)
            sensor_keys = [s for s in acc_dict[mn2_t0_s[0]].keys() if 'no_outliers_replaced' not in s]

            for sensor in sensor_keys:
                f,psd = sig.welch(np.array( [acc_dict[t][sensor] for t in acc_dict] ), fs=1e3, nperseg=nperseg, axis=1) 
                psd_df = pd.DataFrame(psd, columns= f, index= acc_dict.keys() )

                # here we also fit power laws to each PSD ( beta f^(alpha) ) note we do log fit (log(PSD) = alpha * log(f) + beta, cov[0,0] corresponds to alpha covariance, cov[1,1] to beta covariance
                lin_fit_dict = {'alpha' : [] , 'beta' : [], 'cov_00' : [], 'cov_01' : [], 'cov_10' : [], 'cov_11' : []}
                for row in psd_df.index:

                    param, cov = np.polyfit(np.log10(f[1:]), np.log10(psd_df.loc[row].values[1:]), 1, cov=True) 

                    lin_fit_dict['alpha'].append( param[0] )
                    lin_fit_dict['beta'].append( param[1] )
                    lin_fit_dict['cov_00'].append( cov[0,0] )
                    lin_fit_dict['cov_01'].append( cov[0,1] )
                    lin_fit_dict['cov_10'].append( cov[1,0] )
                    lin_fit_dict['cov_11'].append( cov[1,1] )

                for pp in lin_fit_dict:
                    acc_health_df[f'{sensor}_powerlaw_fit_{pp}'] = lin_fit_dict[pp]


                # Writing PSDs
                if write_files: # only do if we want to write results to csv files
                    if overwrite:# we begin by writing a new file 
                        if month_indx == 0: # we overwrite on 1st iteration 
                            psd_df.to_csv(write_dir + f'{sensor}_psds_{UT}_{d.year}-{d.month:02}.csv')

                        else: # then keep adding to this same file (without re-writing headers) 
                            psd_df.to_csv(write_dir + f'{sensor}_psds_{UT}_{d.year}-{d.month:02}.csv', mode = 'a',index=True, header=False)

                    if not overwrite: # we just keep adding to existing file 
                        if os.path.exists( write_dir + f'{sensor}_psds_{UT}_{d.year}-{d.month:02}.csv'  ): #first check if exists (should we write the header?)
                            psd_df.to_csv(write_dir + f'{sensor}_psds_{UT}_{d.year}-{d.month:02}.csv', index=True, header=True)
                        else: 
                            psd_df.to_csv(write_dir + f'{sensor}_psds_{UT}_{d.year}-{d.month:02}.csv', mode = 'a',index=True, header=False)


            # finally put acc_health_df in dataframe 
            acc_health_df = pd.DataFrame( acc_health_df, index= acc_dict.keys() )

            # Writing accelerometer health 
            if write_files: # only do if we want to write results to csv files
                if overwrite:# we begin by writing a new file 
                    if month_indx == 0: # we overwrite on 1st iteration 
                        acc_health_df.to_csv(write_dir + f'accelerometer_HC_{UT}_{d.year}-{d.month:02}.csv')

                    else: # then keep adding to this same file (without re-writing headers) 
                        acc_health_df.to_csv(write_dir + f'accelerometer_HC_{UT}_{d.year}-{d.month:02}.csv', mode = 'a',index=True, header=False)

                if not overwrite: # we just keep adding to existing file 
                    if os.path.exists( write_dir + f'accelerometer_HC_{UT}_{d.year}-{d.month:02}.csv'  ): #first check if exists (should we write the header?)
                        acc_health_df.to_csv(write_dir + f'accelerometer_HC_{UT}_{d.year}-{d.month:02}.csv', index=True, header=True)
                    else: 
                        acc_health_df.to_csv(write_dir + f'accelerometer_HC_{UT}_{d.year}-{d.month:02}.csv', mode = 'a',index=True, header=False)

    
 
    
def write_psd_distribution_features(UT_list, start_date, end_date , base_dir) :
    
    def _get_distribution_feature_df( opl_psds, filt ):

        state_psds = opl_psds[filt]

        if len( state_psds ) > 10 : # we probably want more then 10 samples to get meaningful statistics..
            
            # calculate bimodal coefficient for each frequency bin distribution 
            #---note we take log10 since distributions are generally log spaced 
            bimodal_coefficients = np.array( [bimodal_coef( np.log10( state_psds.T.iloc[i] ) ) for i in range(len(state_psds.T))] )

            thresh = 0.555 #
            bimodal_bins = np.where(bimodal_coefficients>thresh)[0]
            unimodal_bins = np.array(list( set(np.arange(len(state_psds.T))) - set(bimodal_bins )  ))

            ##########
            # Now get our features 
            ########### 

            #init feature lists 
            unimodal = []
            moments = []
            quantiles = []

            # get em
            for i in range(len(state_psds.T)):

                if (i in unimodal_bins) & (i not in bimodal_bins):
                    unimodal.append( 1 ) 
                elif (i not in unimodal_bins) & (i in bimodal_bins):
                    unimodal.append( 0 ) 

                # some basic checks that we assigned this correctly
                elif (i not in unimodal_bins) & (i not in bimodal_bins):
                    print(f'error index {i} not in either unimodal_bins or bimodal_bins')
                elif (i in unimodal_bins) & (i in bimodal_bins):
                    print(f'error index {i} in both unimodal_bins or bimodal_bins')
                else:
                    print('shit')

                bin_tmp = 1e12 * state_psds.T.iloc[i] 

                moments.append( [np.nanmean(bin_tmp) ] + [moment(bin_tmp, moment=m, axis=0, nan_policy='omit') for m in moments2calc[1:]] )

                quantiles.append( list( np.nanquantile( 1e12 * state_psds.T.iloc[i] , quantiles2calc ) ) )


            #print(len(unimodal), len(moments), len(quantiles))

            distribution_features_array = np.vstack( (np.array(unimodal).T , np.vstack(( np.array(moments).T , np.array(quantiles).T  ) ) ) ).T
            distribution_features_df = pd.DataFrame( distribution_features_array , columns = columns , index = state_psds.columns)

        else: 

            print( f'no (or very few!) PSD data for {UT} in {focus_name} during {state_name} state for {month:02}-{start_date.year}' )

            distribution_features_df = pd.DataFrame([], columns = columns )

        return( distribution_features_df ) 
        
    
    print( '\n----------------------^^^----------------------\n   *** deriving PSD distribution features ***\n----------------------^^^----------------------\n')
    
    moments2calc = [1,2,3,4] #moment 1 = mean, moment 2 = var etc
    quantiles2calc = np.linspace(0.1,0.9,9) # PSD binned quantiles to calculate

    # labels 
    columns = ['unimodal'] + [f'moment_{i}' for i in moments2calc] + [f'q{round(100*i)}' for i in quantiles2calc]
    
    if start_date.year != end_date.year:
        
        raise TypeError('\n\nNEED TO HAVE start_date AND end_date WITHIN THE SAME YEAR!!\n(Try process in seperate batches if crossing over to different years)\n\n')
        
    for month in range(start_date.month, end_date.month + 1):

        for UT in UT_list:

            print(f'\n -- {UT} --\n')

            wdir = base_dir + f'{start_date.year}/{month:02}/{UT}/'

            files = glob.glob(wdir+'*.csv')
            
            if len(files)>0: # if recoating or some other activity sensors can be disconnected so we won't have data 
                state_files = [f for f in files if 'states' in f]
                if len(state_files) > 1:    
                    raise TypeError(f'more then one states file for {UT} in {wdir}, only keep one of these..')  

                # read in states 
                if len(state_files)==0:
                    print('--------\n\nWARNING: state_files len = 0\n\n') 
                states = pd.read_csv( f'{state_files[0]}',index_col=0  )

                # generate our dictionary for filtering data based on states 
                state_filter_dict = { 'operations':(states['guiding']==1) & (states['enc_open']==1), 'non-operations':(states['guiding']!=1) | (states['enc_open']!=1) }
                focus_filter_dict = { 'nasA':(states['nasA_focus']==1), 'nasB':(states['nasA_focus']==1), 'cas':(states['cas_focus']==1), 'coude':(states['coude_focus']==1) }

                psd_files = [f for f in files if 'psds' in f]
                sensors =  [f.split('_psds')[0].split('/')[-1] for f in psd_files]             

                for file ,s in zip(psd_files, sensors): 

                    print(f'.    looking at {s}')

                    psds = pd.read_csv( file , index_col=0 )

                    # frequencies (Hz) 
                    f_acc = np.array( [float(x) for x in psds.columns] )

                    # double integrate acceleration to get position (OPL) PSDs
                    opl_psds = pd.DataFrame([double_integrate(f_acc, psds.iloc[i].values)[1] for i in range(len(psds))], columns = f_acc[1:],index = psds.index)
                    f = f_acc[1:] # double integration drops 1st index in freq (to avoid division by zero in integral) 

                    for state_name, state_filt in state_filter_dict.items():


                        focus_name = 'allFoci'
                        distribution_features_df = _get_distribution_feature_df( opl_psds, state_filt )


                        if not os.path.exists( wdir + 'PSD_distribution_features/'  ): 
                            os.makedirs( wdir + 'PSD_distribution_features/' )

                        distribution_features_df.to_csv(wdir+ 'PSD_distribution_features/' + f'{start_date.year}-{month:02}_{UT}_{state_name}_{focus_name}_{s}_psd_distribution_features.csv') 


                        for focus_name, focus_filt in focus_filter_dict.items():

                            distribution_features_df = _get_distribution_feature_df( opl_psds, state_filt & focus_filt )


                            

                            # write to file (NOTE PHYSICAL UNITS ARE um^2/Hz (apart from raw sensors which are uV^2/Hz.. this gets calibrated when performing MNII_classification() function )

                            if not os.path.exists( wdir + 'PSD_distribution_features/'  ): 
                                os.makedirs( wdir + 'PSD_distribution_features/' )

                            distribution_features_df.to_csv(wdir+ 'PSD_distribution_features/' + f'{start_date.year}-{month:02}_{UT}_{state_name}_{focus_name}_{s}_psd_distribution_features.csv') 

            elif len(files)==0:
                
                print( F'\n\n------------\n NO ACCELEROMETER DATA FOR {UT} DURING MONTH {month} \n...we will not derive PSD distribution features here\n------------\n\n' )




                
                
                
                
                
"""

========= EVERYTHING BELOW IS OLD AND SHOULDNT BE USED 


"""
    
    
    
    
    

def MNII_classification_old_redudant (file, time_key, opl_thresh, sensors='all', post_upgrade=False, plot=True , verbose = False):    
    """
    
    !!!!!!!!!!
    NOTE : at the moment this only classifies based on historic values while in operations.. later we may consider other states
    therefore do not pay too much attention to results if input file was not in operational state (e.g. guiding with encolure open) 
    !!!!!!!!!!
    
    
    input:
        file - string (e.g.  f'/datalake/rawdata/vlt-vibration/2022/08/ldlvib{UT}_raw_2022-08-19.hdf5') 
        time_key - what time do we want to look at in the file (e.g. '03:29:00'.. note always rounded to nearest minute)
        sensors - string or list , what sensors do we want to classify? 'all' for all of them (note m4-m7 sensors only available if 
        opl_thresh - float, what threshold do we want to put on the PSD OPL (units um^2/Hz) to trigger red flag (alarm classification?)
        
        post_upgrade = True), or list of sensors strings e.g. ['sensor_1', 'm1', 'm1-3']
        post_upgrade = boolean - do we want to consider the new (m4-m7) MNII sensors? 
        plot = boolean - do we want to plot results (PSDs) 
        verbose = boolean - do you want to print some details of whats happening? 
        
    output: 
        freq_classifications - pandas series indexed by frequency with the PSD frequency binned classification 
            - 'silver' = nomial (between 10-90th percentiles) 
            - 'green' = better then usual (<10th percentile) 
            - 'orange' = worse then usual (>90th percentile) 
            - 'red' = alarm, redflag! (>90th percentile & PSD(f) > opl_thresh)
            
        
            
    """

    # extract UT# from file name
    UT = file.split('ldlvib')[-1].split('_')[0]

    # INIT FIGURES 
    if (sensors == 'all'): #& (not post_upgrade) 
        fig = plt.figure(constrained_layout=True,figsize=(30,20))
        gs = fig.add_gridspec(7,3)

        ax11 = fig.add_subplot(gs[0, 0]) #acc 
        ax12 = fig.add_subplot(gs[1, 0]) #acc 
        ax13 = fig.add_subplot(gs[2, 0]) #acc 
        ax14 = fig.add_subplot(gs[3, 0]) #acc 


        ax21 = fig.add_subplot(gs[0, 1]) #acc 

        ax31 = fig.add_subplot(gs[0, 2]) #acc 
        ax32 = fig.add_subplot(gs[1, 2]) #acc 

        ax1 = fig.add_subplot(gs[5, 0]) # m1 combined 
        ax2 = fig.add_subplot(gs[5, 1]) # m2 combined  
        ax3 = fig.add_subplot(gs[5, 2]) # m3 combined 
        ax4 = fig.add_subplot(gs[6, 0]) # m123

    elif type(sensors)==type([]): #& (not post_upgrade):

        # initiate axx == len list

        no_plots = len(sensors) 
        fig, ax = plt.subplots( no_plots , 1 , figsize=(8, 5*no_plots) )


    if post_upgrade:
        print('post upgrade data on new sensors not ready for classification yet... coming soon\n\
        for now we just consider sensors up to m3') 
        
   
    # process data 
    # ====================
    acc_dict = process_single_mn2_sample(file, time_key, post_upgrade = post_upgrade) 
    # ====================

    #print(acc_dict.keys() )
    #init dict 
    freq_classification_dict = {}
    red_flag_freqs = {}

    if (sensors == 'all') :
        
        for j, (sensor, ax) in enumerate( zip( ['sensor_5','sensor_6','sensor_7','sensor_8','sensor_3','sensor_1','sensor_2','m1','m2','m3','m1-3'], [ax11,ax12,ax13,ax14,ax21,ax31,ax32, ax1,ax2,ax3,ax4] ) ):

            f_acc, accel_psd  = acc_psd(acc_dict[sensor])

            f, opl_psd = double_integrate(f_acc, accel_psd )

            if 'sensor' in sensor: #for raw sensors we apply additional gain factor (V2acc_gain) to convert V^2/Hz to m^2/Hz 

                opl_psd *= V2acc_gain * 1e12 

                dist_features = pd.read_csv(f'/home/jovyan/usershared/BCB/UT{UT}/operations_{sensor}_distribution_features.csv',index_col=0)
                #imterpolate onto same grid (to be safe) 
                fn = interp1d( f, opl_psd , kind = 'linear' , bounds_error=False, fill_value=np.nan) 
                opl_psd_interp = fn( list(dist_features.index) )

                quantile_df = V2acc_gain * dist_features[[f'q{int(x)}' for x in np.linspace(10,90,9)]]
                freq_classifications = [classify_freq_bin(dist_features.index[i], opl_psd_interp[i], quantile_df.iloc[i], opl_thresh=opl_thresh) for i in range(len(quantile_df))]
                # make as panda series indexed by frequency 
                freq_classification_dict[sensor] = pd.Series( freq_classifications , index = f)
                # publish frequencies with red flags 
                red_flag_freqs[sensor] = get_flag_frequencies_indicies( freq_classification_dict[sensor], flag_color='red' )  

            else: 
                opl_psd *= 1e12 # make it um^2/Hz

                dist_features = pd.read_csv(f'/home/jovyan/usershared/BCB/UT{UT}/operations_{sensor}_distribution_features.csv',index_col=0)
                #imterpolate onto same grid (to be safe) 
                fn = interp1d( f, opl_psd , kind = 'linear' , bounds_error=False, fill_value=np.nan) 
                opl_psd_interp = fn( list(dist_features.index) )

                quantile_df = dist_features[[f'q{int(x)}' for x in np.linspace(10,90,9)]]
                freq_classifications = [classify_freq_bin(dist_features.index[i], opl_psd_interp[i], quantile_df.iloc[i], opl_thresh=opl_thresh) for i in range(len(quantile_df))]
                # make as panda series indexed by frequency 
                freq_classification_dict[sensor] = pd.Series( freq_classifications , index = f)
                # publish frequencies with red flags 
                red_flag_freqs[sensor] = get_flag_frequencies_indicies( freq_classification_dict[sensor], flag_color='red' ) 
                
            #############
            # PLOTTING 
            #############

            # make background pink if there is a redflag
            if 'red' in freq_classification_dict[sensor].values:
                ax.set_facecolor(color='pink')
            else:
                ax.set_facecolor(color='white')

            #plotting PSD 
            

            ax.scatter(f, opl_psd, s=30, color=freq_classification_dict[sensor].values, label=sensor)

            ax.loglog(f, quantile_df['q10'],color='k',linestyle='--',alpha=0.5)# ,label='q10-90')
            ax.loglog(f, quantile_df['q90'],color='k',linestyle='--',alpha=0.5)
            
            ax.fill_between(f, quantile_df['q10'], quantile_df['q90'], color='green',alpha=0.4) #label=f'historic 10-90 percentile range \n{min(psds.index)[:10]} - {max(psds.index)[:10]}')
                        

            
            #reverse cumulative 
            ax.plot(f, np.cumsum(opl_psd[::-1])[::-1] * np.diff(f)[1], color='k', linestyle = '-') 
            
            #ax.loglog(f, np.cumsum(quantile_df['q90'][::-1])[::-1] * np.diff(f)[1],color='grey',linestyle=':')
            #ax.loglog(f, np.cumsum(quantile_df['q90'][::-1])[::-1] * np.diff(f)[1],color='grey',linestyle=':')
            
            ax.legend(fontsize=24)
            ax.grid()
            ax.tick_params(labelsize=18)
            ax.set_ylabel('PSD '+r'[$\mu m^2$/Hz]'+'\nreverse cum. '+r'[$\mu m^2$]',fontsize=20)
            ax.set_xlabel('frequency [Hz]',fontsize=20)
            #############
            #############

            if verbose:
                if len(red_flag_freqs[sensor]) > 0:
                    print('\n{} has red flags (alarm!) at frequencies = {} Hz\n'.format( sensor, red_flag_freqs ) ) 
        
        plt.show() 
        
        return( red_flag_freqs, freq_classification_dict) 
    
    
    else: # user defined sensors
        
        if len(sensors) == 1: # if only 1 sensor to consider
            
            sensor = sensors[0]
            
            axx = ax 
            
            f_acc, accel_psd  = acc_psd(acc_dict[sensor])

            f, opl_psd = double_integrate(f_acc, accel_psd )

            if 'sensor' in sensor: #for raw sensors we apply additional gain factor (V2acc_gain) to convert V^2/Hz to m^2/Hz 

                opl_psd *= V2acc_gain * 1e12 

                dist_features = pd.read_csv(f'/home/jovyan/usershared/BCB/UT{UT}/operations_{sensor}_distribution_features.csv',index_col=0)
                #imterpolate onto same grid (to be safe) 
                fn = interp1d( f, opl_psd , kind = 'linear' , bounds_error=False, fill_value=np.nan) 
                opl_psd_interp = fn( list(dist_features.index) )

                quantile_df = V2acc_gain * dist_features[[f'q{int(x)}' for x in np.linspace(10,90,9)]]
                freq_classifications = [classify_freq_bin(dist_features.index[i], opl_psd_interp[i], quantile_df.iloc[i], opl_thresh=opl_thresh) for i in range(len(quantile_df))]
                # make as panda series indexed by frequency 
                freq_classification_dict[sensor] = pd.Series( freq_classifications , index = f)
                # publish frequencies with red flags 
                red_flag_freqs[sensor] = get_flag_frequencies_indicies( freq_classification_dict[sensor], flag_color='red' ) 

            else: 
                opl_psd *= 1e12 # make it um^2/Hz

                dist_features = pd.read_csv(f'/home/jovyan/usershared/BCB/UT{UT}/operations_{sensor}_distribution_features.csv',index_col=0)
                #imterpolate onto same grid (to be safe) 
                fn = interp1d( f, opl_psd , kind = 'linear' , bounds_error=False, fill_value=np.nan) 
                opl_psd_interp = fn( list(dist_features.index) )

                quantile_df = dist_features[[f'q{int(x)}' for x in np.linspace(10,90,9)]]
                freq_classifications = [classify_freq_bin(dist_features.index[i], opl_psd_interp[i], quantile_df.iloc[i], opl_thresh=opl_thresh) for i in range(len(quantile_df))]
                # make as panda series indexed by frequency 
                freq_classification_dict[sensor] = pd.Series( freq_classifications , index = f)
                # publish frequencies with red flags 
                red_flag_freqs[sensor] = get_flag_frequencies_indicies( freq_classification_dict[sensor], flag_color='red' ) 


            #############
            # PLOTTING 
            #############
            # make background pink if there is a redflag
            if 'red' in freq_classification_dict[sensor].values:
                axx.set_facecolor(color='pink')
            else:
                axx.set_facecolor(color='white')

            #plotting 
            
            axx.scatter(f, opl_psd, s=30, color=freq_classification_dict[sensor].values, label=sensor)

            axx.loglog(f, quantile_df['q10'],color='k',linestyle='--',alpha=0.5)# ,label='q10-90')
            axx.loglog(f, quantile_df['q90'],color='k',linestyle='--',alpha=0.5)
            
            axx.fill_between(f, quantile_df['q10'], quantile_df['q90'], color='green',alpha=0.4) 
            
            
            #reverse cumulative 
            axx.plot(f, np.cumsum(opl_psd[::-1])[::-1] * np.diff(f)[1], color='k', linestyle = '-') 

            axx.legend(fontsize=24)
            axx.grid()
            axx.tick_params(labelsize=18)
            axx.set_ylabel('PSD '+r'[$\mu m^2$/Hz]'+'\nreverse cum. '+r'[$\mu m^2$]',fontsize=20)
            axx.set_xlabel('frequency [Hz]',fontsize=20)
            #############
            #############
            
        else: 
            
            for j, (sensor, axx) in enumerate( zip( sensors, ax ) ):

                f_acc, accel_psd = acc_psd(acc_dict[sensor])

                f, opl_psd = double_integrate(f_acc, accel_psd )

                if 'sensor' in sensor: #for raw sensors we apply additional gain factor (V2acc_gain) to convert V^2/Hz to m^2/Hz 

                    opl_psd *= V2acc_gain * 1e12 

                    dist_features=pd.read_csv(f'/home/jovyan/usershared/BCB/UT{UT}/operations_{sensor}_distribution_features.csv',index_col=0)
                    #imterpolate onto same grid (to be safe) 
                    fn = interp1d( f, opl_psd , kind = 'linear' , bounds_error=False, fill_value=np.nan) 
                    opl_psd_interp = fn( list(dist_features.index) )

                    quantile_df = V2acc_gain * dist_features[[f'q{int(x)}' for x in np.linspace(10,90,9)]]
                    freq_classifications = [classify_freq_bin(dist_features.index[i], opl_psd_interp[i], quantile_df.iloc[i], opl_thresh=opl_thresh) for i in range(len(quantile_df))]
                    # make as panda series indexed by frequency 
                    freq_classification_dict[sensor] = pd.Series( freq_classifications , index = f)
                    # publish frequencies with red flags 
                    red_flag_freqs[sensor] = get_flag_frequencies_indicies( freq_classification_dict[sensor], flag_color='red' ) 

                else: 
                    opl_psd *= 1e12 # make it um^2/Hz

                    dist_features = pd.read_csv(f'/home/jovyan/usershared/BCB/UT{UT}/operations_{sensor}_distribution_features.csv',index_col=0)
                    #imterpolate onto same grid (to be safe) 
                    fn = interp1d( f, opl_psd , kind = 'linear' , bounds_error=False, fill_value=np.nan) 
                    opl_psd_interp = fn( list(dist_features.index) )

                    quantile_df = dist_features[[f'q{int(x)}' for x in np.linspace(10,90,9)]]
                    freq_classifications = [classify_freq_bin(dist_features.index[i], opl_psd_interp[i], quantile_df.iloc[i], opl_thresh=opl_thresh) for i in range(len(quantile_df))]
                    # make as panda series indexed by frequency 
                    freq_classification_dict[sensor] = pd.Series( freq_classifications , index = f)
                    # publish frequencies with red flags 
                    red_flag_freqs[sensor] = get_flag_frequencies_indicies( freq_classification_dict[sensor], flag_color='red' ) 

                #############
                # PLOTTING 
                #############

                # make background pink if there is a redflag
                if 'red' in freq_classification_dict[sensor].values:
                    axx.set_facecolor(color='pink')
                else:
                    axx.set_facecolor(color='white')

                #plotting 
                axx.scatter(f, opl_psd, s=30, color=freq_classification_dict[sensor].values, label=sensor)

                axx.loglog(f, quantile_df['q10'],color='k',linestyle='--',alpha=0.5)# ,label='q10-90')
                axx.loglog(f, quantile_df['q90'],color='k',linestyle='--',alpha=0.5)

                axx.fill_between(f, quantile_df['q10'], quantile_df['q90'], color='green',alpha=0.4) 
            

                #reverse cumulative 
                axx.plot(f, np.cumsum(opl_psd[::-1])[::-1] * np.diff(f)[1], color='k', linestyle = '-') 

                axx.legend(fontsize=24)
                axx.grid()
                axx.tick_params(labelsize=18)
                axx.set_ylabel('PSD '+r'[$\mu m^2$/Hz]'+'\nreverse cum. '+r'[$\mu m^2$]',fontsize=20)
                axx.set_xlabel('frequency [Hz]',fontsize=20)
                #############
                #############



        if verbose:
            if len(red_flag_freqs[sensor]) > 0:
                print('\n{} has red flags (alarm!) at frequencies = {} Hz\n'.format( sensor, red_flag_freqs ) ) 

        plt.show() 

        return( red_flag_freqs, freq_classification_dict) 

            
        
"""
                            ## -----------------------------------------------
                            state_psds = opl_psds[state_filt & focus_filt].copy()

                            if len( state_psds ) > 10 : # we probably want more then 10 samples to get meaningful statistics..

                                # calculate bimodal coefficient for each frequency bin distribution 
                                #---note we take log10 since distributions are generally log spaced 
                                bimodal_coefficients = np.array( [bimodal_coef( np.log10( state_psds.T.iloc[i] ) ) for i in range(len(state_psds.T))] )

                                thresh = 0.555 #
                                bimodal_bins = np.where(bimodal_coefficients>thresh)[0]
                                unimodal_bins = np.array(list( set(np.arange(len(state_psds.T))) - set(bimodal_bins )  ))

                                ##########
                                # Now get our features 
                                ########### 

                                #init feature lists 
                                unimodal = []
                                moments = []
                                quantiles = []

                                # get em
                                for i in range(len(state_psds.T)):

                                    if (i in unimodal_bins) & (i not in bimodal_bins):
                                        unimodal.append( 1 ) 
                                    elif (i not in unimodal_bins) & (i in bimodal_bins):
                                        unimodal.append( 0 ) 

                                    # some basic checks that we assigned this correctly
                                    elif (i not in unimodal_bins) & (i not in bimodal_bins):
                                        print(f'error index {i} not in either unimodal_bins or bimodal_bins')
                                    elif (i in unimodal_bins) & (i in bimodal_bins):
                                        print(f'error index {i} in both unimodal_bins or bimodal_bins')
                                    else:
                                        print('shit')

                                    bin_tmp = 1e12 * state_psds.T.iloc[i] 

                                    moments.append( [np.nanmean(bin_tmp) ] + [moment(bin_tmp, moment=m, axis=0, nan_policy='omit') for m in moments2calc[1:]] )

                                    quantiles.append( list( np.nanquantile( 1e12 * state_psds.T.iloc[i] , quantiles2calc ) ) )


                                #print(len(unimodal), len(moments), len(quantiles))

                                distribution_features_array = np.vstack( (np.array(unimodal).T , np.vstack(( np.array(moments).T , np.array(quantiles).T  ) ) ) ).T
                                distribution_features_df = pd.DataFrame( distribution_features_array , columns = columns , index = state_psds.columns)

                            else: 

                                print( f'no PSD data for {UT} in {focus_name} during {state_name} state for {month:02}-{start_date.year}' )

                                distribution_features_df = pd.DataFrame([], columns = columns )


                            ## -----------------------------------------------
"""