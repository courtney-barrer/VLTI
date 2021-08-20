#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 05:38:11 2021

@author: bcourtne
"""


import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.special import sph_harm
import datetime
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy import special 
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%% 1 ----------- #constants ------------------

c = 3e8 #speed of light m/s
h = 6.63e-34 # plank's constant J.s
kB = 1.38e-23 #boltzman constant m2 kg /s^2/K
rad2mas = 180/np.pi * 3600 * 1e3
cm2au = 6.68459e-14

paranal_coordinate = (-24.62794830, -70.40479659) #(lat, long) degrees 

#VLTI station coordinates (https://www.eso.org/observing/etc/doc/viscalc/vltistations.html)

#_ID_______P__________Q__________E__________N____
vlti_stations = 'A0    -32.001    -48.013    -14.642    -55.812\
 A1    -32.001    -64.021     -9.434    -70.949 nl\
 B0    -23.991    -48.019     -7.065    -53.212 nl\
 B1    -23.991    -64.011     -1.863    -68.334 nl\
 B2    -23.991    -72.011      0.739    -75.899 nl\
 B3    -23.991    -80.029      3.348    -83.481 nl\
 B4    -23.991    -88.013      5.945    -91.030 nl\
 B5    -23.991    -96.012      8.547    -98.594 nl\
 C0    -16.002    -48.013      0.487    -50.607 nl\
 C1    -16.002    -64.011      5.691    -65.735 nl\
 C2    -16.002    -72.019      8.296    -73.307 nl\
 C3    -16.002    -80.010     10.896    -80.864 nl\
 D0      0.010    -48.012     15.628    -45.397 nl\
 D1      0.010    -80.015     26.039    -75.660 nl\
 D2      0.010    -96.012     31.243    -90.787 nl\
 E0     16.011    -48.016     30.760    -40.196 nl\
 G0     32.017    -48.0172    45.896    -34.990 nl\
 G1     32.020   -112.010     66.716    -95.501 nl\
 G2     31.995    -24.003     38.063    -12.289 nl\
 H0     64.015    -48.007     76.150    -24.572 nl\
 I1     72.001    -87.997     96.711    -59.789 nl\
 J1     88.016    -71.992    106.648    -39.444 nl\
 J2     88.016    -96.005    114.460    -62.151 nl\
 J3     88.016      7.996     80.628     36.193 nl\
 J4     88.016     23.993     75.424     51.320 nl\
 J5     88.016     47.987     67.618     74.009 nl\
 J6     88.016     71.990     59.810     96.706 nl\
 K0     96.002    -48.006    106.397    -14.165 nl\
 L0    104.021    -47.998    113.977    -11.549 nl\
 M0    112.013    -48.000    121.535     -8.951 nl\
 U1    -16.000    -16.000     -9.925    -20.335 nl\
 U2     24.000     24.000     14.887     30.502 nl\
 U3     64.0013    47.9725    44.915     66.183 nl\
 U4    112.000      8.000    103.306     43.999'
 
station_dict=dict()
for s in vlti_stations.split('nl'):
    #keys = station , values = (N,E) (units = meters)
    ss=s.split()
    station_dict[ss[0]]=(float(ss[-1]), float(ss[-2]))
 


#%% 2 ----------- #functions ------------------


##### ------- VLTI FUNCTIONS ----------

def get_LST(coordinates, datetime):
    #get the local sidereal time at your favourite coordinates
    #coordinates = (latitude, longitude) in degrees
    
    lat, lon = coordinates #degrees
    observing_location = EarthLocation(lat=lat*u.deg, lon=lon*u.deg)
    observing_time = datetime #Time(datetime, scale='utc', location=observing_location)
    #datetime.datetime.utcnow()
    LST = observing_time.sidereal_time('mean')
    return(LST)
    

# wish list 
# 1 - function that takes telescope positions and converts to baselines
#       tel2baseline()
#       T2B
# 2 - function that takes wvl, target coordinates, time to return baseline to UV matrix ()
#       base2uv_matrix()   

def tel2baseline(stations_list):
    
    T2B = np.array(   [[1,-1,0,0],\
                   [1,0,-1,0],\
                   [1,0,0,-1],\
                   [0,1,-1,0],\
                   [0,1,0,-1],\
                   [0,0,1,-1] ] )
    tel_pos = []        
    for s in stations_list : 
        
        tel_pos.append( list(station_dict[s]) )
        
    #baseline vectors e.g. B_vector[i] = (Bx, By, Bz) = (x2−x1, y2− y1, z2− z1).. note z2-z1=0
    B_vector = T2B @ tel_pos
    #baseline lengths
    B_scalar = np.sum((T2B @ tel_pos) * (T2B @ tel_pos),axis=1)**0.5    

    return(B_vector)

    

def baseline2uv_matrix(wvl, datetime, location , telescope_coordinates):
    #from https://web.njit.edu/~gary/728/Lecture6.html
    
    #wvl in m
    #location should be tuple of (latitude, longitude) coordinates degrees 
    #telescope coordinates should be astropy.coordinates  SkyCoord class
    
    #hour angle (radian)
    h = ( get_LST( location,  datetime ) - telescope_coordinates.ra ).radian #radian
    
    # convert dec to radians also
    d = telescope_coordinates.dec.radian #radian
    
    mat = 1/wvl * np.array([[np.sin(h), np.cos(h), 0],\
           [-np.sin(d)*np.cos(h), np.sin(d)*np.sin(h), np.cos(d)],\
           [np.cos(d)*np.cos(h), -np.cos(d)*np.sin(h), np.sin(d) ]] )
        
    return(mat)   


##### ------- STAR FUNCTIONS ----------

def B_L(wvl,T): #Spectral density as function of temperature
    #wvl is wavelength vector to calculate spectral density at
    #T is temperature of the black body 
    Bb = 2*h*c**2/wvl**5 * 1 / (np.exp(h*c/(wvl*kB*T)) - 1)
    return(Bb)
 
"""    
def create_black_body_sphere(T, wvl,Rp, r_array, center):
    # creates black body sphere at temperature T and wvl with radius Rp at center coordinates 
    # and grid defined by r_array
    
        
    x,y = r_array, r_array
    x0, y0 = center
    I_grid = np.zeros([len(x),len(y)]) 
    for i, xx in enumerate(x):
        for j,yy in enumerate(y):
            if (xx-x0)**2 + (yy-y0)**2 <= Rp**2: 
                
                I_grid[i,j] = B_L(wvl, T)
                
    return(I_grid)"""
    


def create_black_body_sphere(T, wvl,Rp, pixScale):
    # creates black body sphere at temperature T and wvl with radius Rp at center coordinates 
    # Rp in au 
    # pixscale pix/au
    
        
    x = np.linspace(-Rp, Rp,  int( 2 * np.ceil( 2 * pixScale * Rp  / 2.) ) )  #aways round to even number (easier to insert to other arrays)
    I_grid = np.zeros([len(x),len(x)]) 
    for i, xx in enumerate(x):
        for j,yy in enumerate(x):
            if xx**2 + yy**2 <= Rp**2: 
                
                I_grid[i,j] = B_L(wvl, T)
                
    return(I_grid)



def add_in_star(I1, I2, c1, c2, big_grid, pixScale, I2_eclipse_I1 = True):
    # pixScale in I1, I2 must be equal  (pixel/au)
    # c1, c2, c_grid are in au 
    #I1_eclipse_I2  = True if I1 is infront of I2
    
    # ---- FUNCTION 2 
    # - take 2 stars, respective center positions (au), and big grid to place into
    # - putting stars together 
    #     if r1+r2 < |c1 - c2| 
    #       just place into grid
    #     else:
    #       make temporary grid of r1 + r2 + |c1 - c2|, keeping physical 0 point at c1
    #       place each star into seperate grid.. keeping track of centers
    #       function to eclipse in overlapping pixels 
    # 
    
    r1, r2 =  0.5 * I1.shape[0] / pixScale ,  0.5 * I2.shape[0] / pixScale 
    # 
    c_dif = np.array(list(c1)) - np.array(list(c2))
    c_dist = np.sqrt( c_dif @  c_dif )
    
    #make number of pixels just big enough
    NN = big_grid.shape[0]
    
    phys_size = NN / pixScale #pixScale = pix/au
        
    x = np.linspace(-phys_size/2, phys_size/2, NN) 
        
    i1, j1 = np.argmin(abs(x-c1[0])) , np.argmin(abs(x-c1[1]))
    
    i2, j2 = np.argmin(abs(x-c2[0])) , np.argmin(abs(x-c2[1]))
    
    N1 = I1.shape[0]
    
    N2 = I2.shape[0]
    
        
    if r1 + r2 < c_dist: 
      
        G = big_grid.copy()
        # put stars in grid
        G[i1-N1//2 : i1+N1//2, j1-N1//2 : j1+N1//2] = I1
        
        G[i2-N2//2 : i2+N2//2, j2-N2//2 : j2+N2//2] = I2
    
    else: #overlapping stars.. need to deal with eclipse
        
        #make 2 copies of the grid
        G1, G2 = big_grid.copy(), big_grid.copy()
        
        # put stars in seperate grids
        G1[i1-N1//2 : i1+N1//2, j1-N1//2 : j1+N1//2] = I1
        
        G2[i2-N2//2 : i2+N2//2, j2-N2//2 : j2+N2//2] = I2
        
        if I2_eclipse_I1:
            
            G = eclipse_adding(G2, G1)
            
        else:
            
            G = eclipse_adding(G1, G2)
    
    return(G)


def eclipse_adding(I1, I2):
    # adding stellar intensity arrays when one star eclipses the other
    # background pixels (points) should be zero (assumes no sky signal)
    # convention is I1 passing infront of I2. It assumes I1 completely blocks light from I2
    # both I1, I2 should be a 2D array (numpy) with same shape
     
    if I1.shape == I2.shape: 
        i1 = I1.reshape(-1)
        i2 = I2.reshape(-1)        
        #indices where I1 intersects I2 (these points we assign I1 )
        eclipsing_indx = set(np.where( i1 > 0 )[0]).intersection(set(np.where( i2 > 0 )[0]))
        #non_eclipsing_indx = set(range(i1.shape[0])) - eclipsing_indx
        
        if eclipsing_indx: #if the star is actually eclipsing
        
            i3 = np.array( [i1[i] if (i in eclipsing_indx) else i1[i] + i2[i] for i in range(i1.shape[0])] )
                          
            I3 = i3.reshape(I1.shape)
        
        else:# not eclipsing so we can just add them normally
        
            I3 = I1 + I2
    
    else: #arrays don't match in shape
    
        raise TypeError( 'input array shapes dont match')
        
    
    return(I3)
    


#%% 4  ----------- #Example ------------------   

# function that accepts obs_dict with telescope and observational parameters and target dictionary  
"""

# target dictionary has keys
    stellar grid , 
    list of (radii,centers),
    list of wavelengths 

# obs dictionary has keys 
    time of obs
    tel location
    tel array 

# returns
    UV sampling 
    returns visibilities map (2D array)
    visibilities at telescope array sampling points 
    
"""

targ_dict = dict( { 'grid' :[] , 'pixScale':[], 'star_geometry':[(),(),(),()], star_coord : (), 'wvl':[1.1e-6, 2.2e-6] } )
obs_dict = dict( { 'obs_time':[] , 'tel_loc' : paranal_coordinate, 'tel_array': []  }) 


def observe(targ_dict, obs_dict):
    
    for wvl in targ_dict['wvl']: 
        
        for center, in targ_dict['star_geometry']:
            
            print('to do')            
    
    
#%%
my_vis_dict = dict() # to store (baseline, vis) for each wvl, companion position 
my_ima_dict = dict()
wvls = [1.1e-6, 2.2e-6]
phis = [-1.2, 0.5, 0, 1.2]
for wvl, colour in zip(wvls, ['black','red']):
    
    my_vis_dict[wvl] = dict()
    
    for xxx in phis: # transient across parent star:
        # == CREATE TARGET GEOMETRY == 
        
        #fundemental stellar parameters
        #primary_dist = 4.125e+7 #200pc in au
        #secondary_dist = 4.125e+7 #200pc in au
        
        
        stellar_dist  =  4.125e+7 #200pc in au
        
        Rc = 1  #0.000477895 #[au] ( 1 R_j in au (note brown dwarf radius ~ 1 R_juptier (Hatzes & Rauer, 2015)  )
        Rp = 4 * Rc #[au] ( x10 the suns radius (note that 10 R_J ~ 1 R_sun )
        
        
        cp, cc = (0,0),  (0, xxx*Rp) #stellar centers (au) 
        
        c_dif = np.array(list(cp)) - np.array(list(cc))
        c_dist = np.sqrt( c_dif @  c_dif )
        
        # force a reasonable limit on the grid
        grid_limit = 3 * max( c_dist, Rp ) #au (x2 the maximum of either distance between stars or parent radius)
        No_sample_across_Rc = 20
        # radius grid should have sufficient sampling across companion  (so resembles not too pixelated circle)
        #   dr = grid_limit / N .. No_sample_across_Rc samples across companion radius => Rc/dr = No_sample_across_Rc 
        # => Rc/grid_limit * N = No_sample_across_Rc => N = No_sample_across_Rc/Rc * grid_limit
        
        N = int( No_sample_across_Rc/Rc * grid_limit )  #number of samples along row in grid
        r_au = np.linspace( -grid_limit , grid_limit , N )   # au  - grid size
        
        pixScale = 1/(np.diff( r_au )[0])  #pixel/au
        
        #center_p = (0, 0)  # au
        #center_c = (1, 0)  # au
                 
        Tp = 3400 #K
        Tc = 1000 #K 
        
        Ip = create_black_body_sphere(Tp, wvl, Rp, pixScale)
        Ic = create_black_body_sphere(Tc, wvl, Rc, pixScale)
        
        big_grid = np.zeros([N,N])
        
        #I3 = add_in_star(I1, I2, c1, c2, big_grid, pixScale, I1_eclipse_I2 = True)
        
        I3 = add_in_star(Ip, Ic, cp, cc, big_grid, pixScale, I2_eclipse_I1 = True)

        my_ima_dict[xxx] = I3
        
        # == SET UP OBSERVATION PARAMETERS ==
        #  -- where are we sampling in UV plane?\
            
        #Target
        targ_coord = SkyCoord(ra=225*u.degree, dec=-10*u.degree, frame='icrs')
        
        #Time 
        t = Time('2020-08-01T05:02:00', scale='utc', location=paranal_coordinate)
        
        #UTs
        stations_list = ['U1','U2','U3','U4']  
        
        #x,y baseline vectors (m)
        B = tel2baseline(stations_list) #(m)
        
        #physical baseline distances (not projected on target location)
        B_scalar = np.sum( B * B, axis=1)**0.5   
        
        #matrix for baseline vectors to UV conversion
        B2UV = baseline2uv_matrix( wvl, t, paranal_coordinate, targ_coord )
          
        #our UV coordinates
        Usamp, Vsamp = B2UV[:2,:2] @ B.T
        
        
        
        """#lets take a look
        plt.figure()
        plt.plot(*(UV*wvl),'o',color='k')"""
        
        
        # == PUT OBJECT IN UV PLANE (FOURIER TRANSFORM) == 
        
        #ucoord,vcoord = 1/r_grid, 1/r_grid
        vis = np.fft.fftshift( np.fft.fft2( I3 )  )
        ucoord = np.fft.fftshift( np.fft.fftfreq(I3.shape[1],  np.diff(r_au/stellar_dist)[0]) ) # sampling in radians r_au/stellar_dist
        
        
        #lets take a look at what |V| we're sampling in the UV plane with VLTI! 
        """
        fig = plt.figure()
        im = plt.pcolormesh(wvl*ucoord, wvl*ucoord, abs(vis))
        plt.plot(Usamp*wvl, Vsamp*wvl,'x',color='k',label='VLTI sampling')
        plt.xlabel('U (m)',fontsize=15)
        plt.ylabel('V (m)',fontsize=15)   
        plt.xlim([-130,130])
        plt.ylim([-130,130])
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes('top', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='horizontal')
        cax.set_xlabel('|V|',fontsize=15)
        
        cax.xaxis.set_label_position('top') 
        cax.xaxis.set_ticks_position('top')
        """
        
        # == get V at VLTI u,v points ==
        
        #just do nearest neighbour interp (should be fine with good image sampling)
        U_indx = [np.argmin(abs(ucoord - sss)) for sss in Usamp]  
        V_indx = [np.argmin(abs(ucoord - sss)) for sss in Vsamp]  
        
        B_samp = wvl * np.array( [(ucoord[i]**2 + ucoord[j]**2)**0.5 for i,j in zip(U_indx, V_indx) ] )
        vis_samp = np.array([vis[i,j] for i,j in zip(U_indx, V_indx)], dtype='complex')
        
        #B_samp 
        plt.plot( B_scalar , abs(vis_samp)**2,'.',label='wvl = {}, companion pos = {}'.format(wvl, xxx),color=colour)
        
        my_vis_dict[wvl][xxx] = (B_samp, vis_samp)        
        
plt.legend()


fig, ax = plt.subplots(3,len(phis),sharex=False,figsize=(16,16))
wvl = wvls[0]
for i, phi in enumerate(phis): 
    ax[0,i].plot( B_scalar , abs(my_vis_dict[wvl][phi][1] )**2,'o',label=phi)
    ax[1,i].plot( B_scalar , np.angle(my_vis_dict[wvl][phi][1] ),'o',label=phi)
    ax[2,i].imshow(np.log10(my_ima_dict[phi]))
plt.legend()


plt.figure()

#%% 3 ----------- #Example ------------------     


# == CREATE TARGET GEOMETRY == 

#fundemental stellar parameters
#primary_dist = 4.125e+7 #200pc in au
#secondary_dist = 4.125e+7 #200pc in au


stellar_dist  =  4.125e+7 #200pc in au

Rc = 1  #0.000477895 #[au] ( 1 R_j in au (note brown dwarf radius ~ 1 R_juptier (Hatzes & Rauer, 2015)  )
Rp = 4 * Rc #[au] ( x10 the suns radius (note that 10 R_J ~ 1 R_sun )


cp, cc = (0,0),  (0, 0.5*Rp) #stellar centers (au) 

c_dif = np.array(list(cp)) - np.array(list(cc))
c_dist = np.sqrt( c_dif @  c_dif )

# force a reasonable limit on the grid
grid_limit = 3 * max( c_dist, Rp ) #au (x2 the maximum of either distance between stars or parent radius)
No_sample_across_Rc = 20
# radius grid should have sufficient sampling across companion  (so resembles not too pixelated circle)
#   dr = grid_limit / N .. No_sample_across_Rc samples across companion radius => Rc/dr = No_sample_across_Rc 
# => Rc/grid_limit * N = No_sample_across_Rc => N = No_sample_across_Rc/Rc * grid_limit

N = int( No_sample_across_Rc/Rc * grid_limit )  # number of samples along row in grid
r_au = np.linspace( -grid_limit , grid_limit , N )   # au  - grid size

pixScale = 1/(np.diff( r_au )[0])  #pixel/au

#center_p = (0, 0)  # au
#center_c = (1, 0)  # au

wvl = 2e-6 #um
         
Tp = 3400 #K
Tc = 1000 #K 

Ip = create_black_body_sphere(Tp, wvl, Rp, pixScale)
Ic = create_black_body_sphere(Tc, wvl, Rc, pixScale)

big_grid = np.zeros([N,N])

#I3 = add_in_star(I1, I2, c1, c2, big_grid, pixScale, I1_eclipse_I2 = True)

I3 = add_in_star(Ip, Ic, cp, cc, big_grid, pixScale, I2_eclipse_I1 = True)

#lets take a look
plt.figure()
plt.pcolormesh(r_au, r_au, I3)
plt.xlabel('radius (au)',fontsize=15)
plt.ylabel('radius (au)',fontsize=15)
plt.gca().tick_params(labelsize=15)
plt.tight_layout()
plt.gca().set_aspect('equal')





# == SET UP OBSERVATION PARAMETERS ==
#  -- where are we sampling in UV plane?\
    
#Target
targ_coord = SkyCoord(ra=225*u.degree, dec=-10*u.degree, frame='icrs')

#Time 
t = Time('2020-08-01T05:02:00', scale='utc', location=paranal_coordinate)

#UTs
stations_list = ['U1','U2','U3','U4']  

#x,y baseline vectors (m)
B = tel2baseline(stations_list) #(m)

#physical baseline distances (not projected on target location)
B_scalar = np.sum( B * B, axis=1)**0.5   

#matrix for baseline vectors to UV conversion
B2UV = baseline2uv_matrix( wvl, t, paranal_coordinate, targ_coord )
  
#our UV coordinates
Usamp, Vsamp = B2UV[:2,:2] @ B.T



"""#lets take a look
plt.figure()
plt.plot(*(UV*wvl),'o',color='k')"""


# == PUT OBJECT IN UV PLANE (FOURIER TRANSFORM) == 

#ucoord,vcoord = 1/r_grid, 1/r_grid
vis = np.fft.fftshift( np.fft.fft2( I3 )  )
ucoord = np.fft.fftshift( np.fft.fftfreq(I3.shape[1],  np.diff(r_au/stellar_dist)[0]) ) # sampling in radians r_au/stellar_dist


#lets take a look at what |V| we're sampling in the UV plane with VLTI! 
fig = plt.figure()
im = plt.pcolormesh(wvl*ucoord, wvl*ucoord, abs(vis))
plt.plot(Usamp*wvl, Vsamp*wvl,'x',color='k',label='VLTI sampling')
plt.xlabel('U (m)',fontsize=15)
plt.ylabel('V (m)',fontsize=15)   
plt.xlim([-130,130])
plt.ylim([-130,130])
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes('top', size='5%', pad=0.05)
plt.colorbar(im, cax=cax, orientation='horizontal')
cax.set_xlabel('|V|',fontsize=15)

cax.xaxis.set_label_position('top') 
cax.xaxis.set_ticks_position('top')


# == get V at VLTI u,v points ==

#just do nearest neighbour interp (should be fine with good image sampling)
U_indx = [np.argmin(abs(ucoord - sss)) for sss in Usamp]  
V_indx = [np.argmin(abs(ucoord - sss)) for sss in Vsamp]  

B_samp = wvl * np.array( [(ucoord[i]**2 + ucoord[j]**2)**0.5 for i,j in zip(U_indx, V_indx) ] )
vis_samp = np.array([vis[i,j] for i,j in zip(U_indx, V_indx)], dtype='complex')
plt.figure()
plt.plot( B_samp , abs(vis_samp),'.')







#%% checking theory disk matches algorithm disk (make sure binary companion isnt in the vis_samp)



stellar_dist  =  4.125e+7 #200pc in au

Rc = 1  #0.000477895 #[au] ( 1 R_j in au (note brown dwarf radius ~ 1 R_juptier (Hatzes & Rauer, 2015)  )
Rp = 4 * Rc #[au] ( x10 the suns radius (note that 10 R_J ~ 1 R_sun )


cp, cc = (0,0),  (0, 0.5*Rp) #stellar centers (au) 

c_dif = np.array(list(cp)) - np.array(list(cc))
c_dist = np.sqrt( c_dif @  c_dif )

# force a reasonable limit on the grid
grid_limit = 10 * max( c_dist, Rp ) #au (x2 the maximum of either distance between stars or parent radius)
No_sample_across_Rc = 40
# radius grid should have sufficient sampling across companion  (so resembles not too pixelated circle)
#   dr = grid_limit / N .. No_sample_across_Rc samples across companion radius => Rc/dr = No_sample_across_Rc 
# => Rc/grid_limit * N = No_sample_across_Rc => N = No_sample_across_Rc/Rc * grid_limit

N = int( No_sample_across_Rc/Rc * grid_limit )  #number of samples along row in grid
r_au = np.linspace( -grid_limit , grid_limit , N )   # au  - grid size

pixScale = 1 / ( np.diff( r_au )[0] )  #pixel/au

#center_p = (0, 0)  # au
#center_c = (1, 0)  # au

wvl = 2e-6 #um
         
Tp = 3400 #K
Tc = 1000 #K 

Ip = create_black_body_sphere(Tp, wvl, Rp, pixScale)
#Ic = create_black_body_sphere(Tc, wvl, Rc, pixScale)

big_grid = np.zeros([N,N])

I3 = add_in_star(Ip, Ip, cp, cp, big_grid, pixScale, I2_eclipse_I1 = True)

# == SET UP OBSERVATION PARAMETERS ==
#  -- where are we sampling in UV plane?\
    
#Target
targ_coord = SkyCoord(ra=225*u.degree, dec=-10*u.degree, frame='icrs')

#Time 
t = Time('2020-08-01T05:02:00', scale='utc', location=paranal_coordinate)

#UTs
stations_list = ['U1','U2','U3','U4']  

#x,y baseline vectors (m)
B = tel2baseline(stations_list) #(m)

#physical baseline distances (not projected on target location)
B_scalar = np.sum( B * B, axis=1)**0.5   

#matrix for baseline vectors to UV conversion
B2UV = baseline2uv_matrix( wvl, t, paranal_coordinate, targ_coord )
  
#our UV coordinates
Usamp, Vsamp = B2UV[:2,:2] @ B.T

#look at samples in UV plane
plt.pcolormesh(ucoord[700:900],ucoord[700:900], abs(vis)[700:900,700:900])
plt.plot(Usamp,Vsamp,'x',color='w')


"""#lets take a look
plt.figure()
plt.plot(*(UV*wvl),'o',color='k')"""


# == PUT OBJECT IN UV PLANE (FOURIER TRANSFORM) == 

#ucoord,vcoord = 1/r_grid, 1/r_grid
vis = np.fft.fftshift( np.fft.fft2( I3 )  )
ucoord = np.fft.fftshift( np.fft.fftfreq(I3.shape[1],  np.diff( r_au/stellar_dist )[0]) ) # sampling in radians r_au/stellar_dist


#check cut of visibility map matches disk visibility
DDD =  2 * Rp/stellar_dist

plt.figure()
#theory
V_disk =  np.array( [(2*special.j1(np.pi * DDD * b / wvl) / (np.pi * DDD * b /wvl) ) for b in (wvl * ucoord[vis.shape[0]//2:vis.shape[0]//2+100]) ] )
plt.plot(wvl * np.array(ucoord[vis.shape[0]//2:vis.shape[0]//2+100] ),  abs(V_disk),'o',color='k' )  
#predicted
ax2=plt.twinx()
#ax2.plot( wvl * ucoord[vis.shape[0]//2:vis.shape[0]//2+100]  ,  abs(vis)[vis.shape[0]//2:vis.shape[0]//2+100,vis.shape[0]//2] , lw = 2 )
#can even do diagonal 
ax2.plot(wvl * np.sqrt(2) * ucoord[vis.shape[0]//2:vis.shape[0]//2+100] , np.diag(abs(vis)[vis.shape[0]//2:vis.shape[0]//2+100,vis.shape[0]//2:vis.shape[0]//2+100]))

# the |V| cut in UV plane matches theory for a disk structure

# == get V at VLTI u,v points == Check this also matches

#just do nearest neighbour interp (should be fine with good image sampling)
U_indx = [np.argmin(abs(ucoord - sss)) for sss in Usamp]  
V_indx = [np.argmin(abs(ucoord - sss)) for sss in Vsamp]  
#can check differences:  wvl*ucoord[U_indx[0]] , Usamp[0] * wvl


# look where these interp coordinate points are in UV 
plt.figure()
plt.title('|V| simulation UD')
plt.pcolormesh(ucoord[700:900],ucoord[700:900], abs(vis)[700:900,700:900])
plt.plot(ucoord[U_indx],ucoord[V_indx],'o',color='r')
plt.plot(Usamp,Vsamp,'x',color='w')

## now I could even do a mesh grid with theoretical to make sure we're sampling same place 
uuu,vvv = np.meshgrid(ucoord[ucoord!=0],ucoord[ucoord!=0])
DDD =  2 * Rp/stellar_dist
plt.figure()
V_2Ddisk = special.j1(np.pi * DDD * np.sqrt(uuu**2 + vvv**2)) / (np.pi * DDD * np.sqrt(uuu**2 +vvv**2))
plt.title('|V| theory UD')
plt.pcolormesh(ucoord[700:900],ucoord[700:900], abs(V_2Ddisk)[700:900,700:900] )
plt.plot(ucoord[U_indx],ucoord[V_indx],'o',color='r')
plt.plot(Usamp,Vsamp,'x',color='w')

##--- pretty much checks out.. so the issue must be below:
# ----------  if we use V_2Ddisk instead of vis do we still get an error??

B_samp = wvl * np.array( [np.sqrt(ucoord[i]**2 + ucoord[j]**2) for (i,j) in zip(U_indx, V_indx) ] )
vis_samp = np.array([V_2Ddisk[i,j] for (i,j) in zip(U_indx,V_indx)])

#### LETS SEE HOW WELL ALGORITHM MATCHES THEORY 

plt.figure()
plt.plot( B_samp , abs(vis_samp),'o',label='|V| simulation UD',color='r')
V_disk =  np.array([(special.j1(np.pi * DDD * b / wvl) / (np.pi * DDD * b /wvl) ) for b in B_samp])
ax2=plt.twinx()
ax2.plot( B_samp, abs(V_disk) ,'x', label='|V| theory UD',  color='k')
plt.legend()

#%%




plt.plot( B_samp , abs(vis_samp)/max(abs(vis_samp)),'o',label='pred')
V_disk =  [(special.j1(np.pi * DDD * b / wvl) / (np.pi * DDD * b /wvl) ) for b in B_samp]

plt.plot( B_samp,V_disk ,'x', label='theory' )

V_disk =  [(special.j1(np.pi *  DDD * b/ wvl) / (np.pi *  DDD * b/wvl) ) for b in np.linspace(0,100,200)]
plt.plot( np.linspace(0,100,200), V_disk/max(V_disk ) ) 
plt.legend()





DDD =  Rp/stellar_dist
plt.plot( B_scalar , abs(vis_samp)/max(abs(vis_samp)),'o',label='pred')
V_disk =  [(special.j1(np.pi * DDD * b / wvl) / (np.pi * DDD * b /wvl) ) for b in B_scalar]
plt.plot( B_scalar, V_disk/max(V_disk ) ,'x', label='theory' )


V_disk =  [abs(special.j1(np.pi *  DDD * b/ wvl) / (np.pi *  DDD * b/wvl) ) for b in np.linspace(0.1,100,200)]
plt.plot( np.linspace(0,100,200), V_disk/max(V_disk ) ) 
ax2 = plt.twinx()
ax2.plot(B_samp , abs(vis_samp),'x')
plt.legend()


#%%





#%%

#spherical harmonic indices
m, l = 0, 1
#oscillation frequency 
omega = 2*np.pi/5
#hydrostatic equilibrium temp 
T0=3500
#temp oscilation amplitude
dT = 2000

#number of points in our grid
M = 2**7
#radius (angular coordinates on sky)
R = 10e-3 * (1/3600 * np.pi/180) #10mas"  
#our x-y cartesian coordinates 
x,z = np.linspace(-4*R,4*R,M),np.linspace(-4*R,4*R,M)
y = 10
#angle from observer 
alpha = np.pi #np.pi/4
time = np.linspace(0,5,20)
#wavelengths to evaluate black body intensity
lambdas = [1e-6, 2.2e-6, 3.5e-6]
temp_dict=dict()
intensity_dict = dict()

for t in time:
    
    print('{}% complete'.format(t/max(time)))
    
    temp_grid = np.zeros([len(x),len(z)]) 
    intensity_dict[t] = dict({wvl:np.zeros([len(x),len(z)]) for wvl in lambdas})
    
    for i, xx in enumerate(x):
        for j,zz in enumerate(z):
            if xx**2 + zz**2 <= R**2: 
                phi = np.arctan2(y,xx)
                theta = np.arctan2((xx**2 + y**2)**0.5 , zz)
                
                #temp_grid[i,j] = T0 + dT*(sph_harm(m, l, theta, phi)*np.exp(1j*omega*t)).real 
                temp_grid[i,j] = np.real(T0 + dT*(np.cos(alpha)*np.cos(theta)+np.sin(alpha)*np.sin(theta)*np.cos(phi))*np.exp(1j*omega*t)) #
                
                for wvl in lambdas:
                    intensity_dict[t][wvl][i,j] =  B_L(wvl, temp_grid[i,j])
                    
                    
    temp_dict[t] = temp_grid

fig,ax = plt.subplots(int(len(time)**0.5), int(len(time)**0.5))
for t,axx in zip(time,ax.reshape(-1)):
    axx.imshow(np.log10(intensity_dict[t][wvl])); axx.axis('off')
    
    