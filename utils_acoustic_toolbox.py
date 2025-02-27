"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
Created on Mon Mar 25 15:38:01 2024
@author: xdemoulin
"""

from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt


def write_env_file(root, env_name, f_cen, z_source, z_reception, sound_speed, zw, rmax, nb_ray, zmax, param_seabed, opening_angle, calc):
    """
    Generates an environment file (.env) for Bellhop acoustic propagation modeling.

    Parameters
    ----------
    root : Path
        Directory where the environment file will be saved.
    env_name : str
        Name of the environment file (without extension).
    f_cen : int or float
        Central frequency of the simulation (Hz).
    z_source : float
        Source depth (meters).
    z_reception : float
        Receiver depth (meters).
    sound_speed : np.array
        Sound speed profile (m/s) at corresponding depths.
    zw : np.array
        Depth values corresponding to the sound speed profile.
    rmax : float
        Maximum range for simulation (km).
    nb_ray : int
        Number of rays used in the simulation.
    zmax : float
        Maximum depth for the simulation (meters).
    param_seabed : pd.Series
        Seabed parameters: [sound speed, density ratio, attenuation].
    opening_angle : float
        Beam opening angle (degrees).
    calc : str
        Calculation type ('I' for incoherent, 'E' for eigenray, 'A' for arrivals).

    Returns
    -------
    None
        Writes the .env file to the specified directory.

    Notes
    -----
    - The function generates a Bellhop-compatible environment file.
    - Includes the sound speed profile and seabed characteristics.
    - Supports different calculation modes (incoherent, eigenray, arrivals).
    """

    # Create/Open the environment file
    with open(root / Path(f"{env_name}.env"), 'w') as fid:

        # Write file header
        fid.write(f"'{env_name}'\n")  # Environment file name
        fid.write(f"{f_cen}\n")  # Central frequency
        fid.write("1\n")  # Number of layers (single layer for water column)
        fid.write("'CVWF'\n")  # Model type (C = constant, V = variable, W = water, F = fluid bottom)
        fid.write("19.3 35. 8. 50.\n")  # Temperature, Salinity, pH, Depth of bar (fixed values)
        fid.write(f"0 0.0 {zmax}\n")  # Depth range (0 to zmax)

        # Write sound speed profile
        for z, c in zip(zw, sound_speed):
            fid.write(f"{z} {c} /\n")

        # Ensure the profile extends to zmax
        if zmax > zw[-1]:
            fid.write(f"{zmax} {sound_speed[-1]} /\n")

        # Seabed properties
        fid.write("A* 0.0\n")  # Bottom condition
        fid.write(f"{zmax} {param_seabed['bulk_soundspeed']} 0.0 {param_seabed['bulk_density']} {param_seabed['attenuation']} 0.0 /\n")  # Bottom properties

        # Source depth
        fid.write("1\n")
        fid.write(f"{z_source} /\n")

        # Receiver or Eigenray settings
        if calc == 'I':
            n_rz = round(zmax / 1) + 1  # Depth step = 1m
            fid.write(f"{n_rz}\n0 {zmax} /\n")
        elif calc in ('E', 'A'):
            fid.write("1\n")
            fid.write(f"{z_reception} /\n")

        # Range settings
        if calc == 'I':
            fid.write(f"{nb_ray}\n0 {rmax} /\nI\n1001\n")
        elif calc in ('E', 'A'):
            fid.write("1\n")
            fid.write(f"{rmax} /\n{calc}\n50001\n")

        # Beam angles
        fid.write(f"{-opening_angle} {opening_angle} /\n")

        # Grid resolution settings
        fid.write(f"0. {10 * round(1.05 * zmax / 10)} {round(1.05 * 100 * rmax) / 100}\n")

    return


def read_shd(filename=None, xs=None, ys=None):
    """
    Lecture du .shd pour les champs de pertes (opt. I)

    #****************************************************
    # Arraial do Cabo, Qui Out 20 21:48:55 WEST 2016
    # Written by Tordar
    # Based on read_shd_bin.m by Michael Porter
    #****************************************************
    """
    fid = open(filename,'rb')
    recl  = int(np.fromfile(fid,np.int32,1))
    title = fid.read(80)
    fid.seek( 4*recl )
    PlotType = fid.read(10)
    fid.seek( 2*4*recl ); # reposition to end of second record
    freq   = float( np.fromfile( fid, np.float32, 1 ) )
    Ntheta = int(   np.fromfile( fid, np.int32  , 1 ) )
    Nsx    = int(   np.fromfile( fid, np.int32  , 1 ) )
    Nsy    = int(   np.fromfile( fid, np.int32  , 1 ) )
    Nsd    = int(   np.fromfile( fid, np.int32  , 1 ) )
    Nrd    = int(   np.fromfile( fid, np.int32  , 1 ) )
    Nrr    = int(   np.fromfile( fid, np.int32  , 1 ) )
    atten  = float( np.fromfile( fid, np.float32, 1 ) )
    fid.seek( 3 * 4 * recl ); # reposition to end of record 3
    thetas = np.fromfile( fid, np.float32, Ntheta )
    if  ( PlotType[ 0 : 1 ] != 'TL' ):
       fid.seek( 4 * 4 * recl ); # reposition to end of record 4
       Xs     = np.fromfile( fid, np.float32, Nsx )
       fid.seek( 5 * 4 * recl );  # reposition to end of record 5
       Ys     = np.fromfile( fid, np.float32, Nsy )
    else:   # compressed format for TL from FIELD3D
       fid.seek( 4 * 4 * recl ); # reposition to end of record 4
       Pos_S_x     = np.fromfile( fid, np.float32, 2 )
       Xs          = np.linspace( Pos_S_x[0], Pos_S_x[1], Nsx )
       fid.seek( 5 * 4 * recl ); # reposition to end of record 5
       Pos_S_y     = np.fromfile( fid, np.float32, 2 )
       Ys          = np.linspace( Pos_S_y[0], Pos_S_y[1], Nsy )
    fid.seek( 6 * 4 * recl ) # reposition to end of record 6
    zs = np.fromfile( fid, np.float32, Nsd )
    fid.seek( 7 * 4 * recl ) # reposition to end of record 7
    zarray =  np.fromfile( fid, np.float32, Nrd )
    fid.seek( 8 * 4 * recl ) # reposition to end of record 8
    rarray =  np.fromfile( fid, np.float32, Nrr )
    if PlotType == 'rectilin  ':
       pressure = np.zeros( (Ntheta, Nsd, Nrd, Nrr) ) + 1j*np.zeros( (Ntheta, Nsd, Nrd, Nrr) )
       Nrcvrs_per_range = Nrd
    elif PlotType == 'irregular ':
       pressure = np.zeros( (Ntheta, Nsd,   1, Nrr) ) + 1j*np.zeros( (Ntheta, Nsd, Nrd, Nrr) )
       Nrcvrs_per_range = 1
    else:
       pressure = np.zeros( (Ntheta, Nsd, Nrd, Nrr) )
       Nrcvrs_per_range = Nrd
    pressure = np.zeros( (Ntheta,Nsd,Nrcvrs_per_range,Nrr) ) + 1j*np.zeros( (Ntheta,Nsd,Nrcvrs_per_range,Nrr) )
    if np.isnan( xs ):
      for itheta in range(Ntheta):
          for isd in range( Nsd ):
              for ird in range( Nrcvrs_per_range ):
                  recnum = 9 + itheta * Nsd * Nrcvrs_per_range + isd * Nrcvrs_per_range + ird
                  status = fid.seek( recnum * 4 * recl ) # Move to end of previous record
                  if ( status == -1 ):
                     print('Seek to specified record failed in readshd...')
                  temp = np.fromfile( fid, np.float32, 2 * Nrr ) # Read complex data
                  for k in range(Nrr):
                      pressure[ itheta, isd, ird, k ] = temp[ 2 * k ] + 1j * temp[ 2*k + 1 ]

    else:
       xdiff = abs( Xs - xs * 1000.0 )
       idxX  = xdiff.argmin(0)
       ydiff = abs( Ys - ys * 1000.0 )
       idxY  = ydiff.argmin(0)
       for itheta in range(Ntheta):
           for isd in range(Nsd):
               for ird in range( Nrcvrs_per_range ):
                   recnum = 9 + idxX * Nsy * Ntheta * Nsd * Nrcvrs_per_range + idxY * Ntheta * Nsd * Nrcvrs_per_range + itheta * Nsd * Nrcvrs_per_range + isd * Nrcvrs_per_range + ird
                   status = fid.seek( recnum * 4 * recl ) # Move to end of previous record
                   if ( status == -1 ):
                      print('Seek to specified record failed in read_shd_bin')
                   temp = np.fromfile( fid, np.float32, 2 * Nrr ) # Read complex data
                   for k in range(Nrr):
                       pressure[ itheta, isd, ird, k ] = temp[ 2 * k ] + 1j * temp[ 2*k + 1 ]

       fid.close()
    geometry = {"zs":zs, "f":freq,"thetas":thetas,"rarray":rarray,"zarray":zarray}

    return pressure,geometry


def plotray(filename=None):
    """
    plot des rayons propres (opt. E)
    #*******************************************************************************
    # Faro, Qua Dez  7 21:09:53 WET 2016
    # Written by Tordar
    # Based on plotray.m by Michael Porter
    #*******************************************************************************
    """
    Nsxyz       = np.zeros(3)
    NBeamAngles = np.zeros(2)
    
    fid = open(filename,'r')
    title =        fid.readline()
    freq  = float( fid.readline() )
    theline = str( fid.readline() )
    datai = theline.split()
    Nsxyz[0] = int( datai[0] )
    Nsxyz[1] = int( datai[1] )
    Nsxyz[2] = int( datai[2] )
    theline = str( fid.readline() )
    datai = theline.split()
    NBeamAngles[0] = int( datai[0] )
    NBeamAngles[1] = int( datai[1] )
    DEPTHT = float( fid.readline() )
    DEPTHB = float( fid.readline() )
    Type   = fid.readline()
    Nsx = int( Nsxyz[0] )
    Nsy = int( Nsxyz[1] )
    Nsz = int( Nsxyz[2] )
    Nalpha = int( NBeamAngles[0] )
    Nbeta  = int( NBeamAngles[1] )
    # axis limits
    rmin =  1.0e9
    rmax = -1.0e9
    zmin =  1.0e9
    zmax = -1.0e9

    for isz in range(Nsz):
        for ibeam in range(Nalpha):
	   #alpha0    = float( fid.readline() )
            theline = str( fid.readline() )
            l = len( theline )
            if l > 0:
               alpha0 = float( theline )
               theline = str( fid.readline() )
               datai = theline.split()
               nsteps    = int( datai[0] )
               NumTopBnc = int( datai[1] )
               NumBotBnc = int( datai[2] )
               r = np.zeros(nsteps)
               z = np.zeros(nsteps)
               for nj in range(nsteps):
                   theline = str(fid.readline())
                   rz = theline.split()
                   r[nj] = float( rz[0] )
                   z[nj] = float( rz[1] )        
               rmin = min( [ min(r), rmin ] )
               rmax = max( [ max(r), rmax ] )
               zmin = min( [ min(z), zmin ] )
               zmax = max( [ max(z), zmax ] )
               plt.plot( r, -z )
               plt.axis([rmin,rmax,-zmax,-zmin])
    fid.close()


def read_arrivals_asc(filename=None):
    """
    Lecture de sorties des .arr (opt. A)
    #*******************************************************************************
    # Faro, Seg 11 Abr 2022 12:50:20 WEST
    # Written by Orlando Camargo Rodriguez
    # Based on read_arrivals_asc.m by Michael Porter
    #*******************************************************************************
    """

    Arr = []
    Pos = []
    Narrmx = 100
    maxnarr = 0
    fid = open(filename,'r')
    flag = str( fid.readline() )
    if flag[2] == '2':
       freq  = float( fid.readline() )
       theline = str( fid.readline() )
       datai = theline.split()
       Nsz   = int( datai[0] )
       source_depths = np.zeros(Nsz)
       for i in range(Nsz):
           source_depths[i] = float( datai[i+1] )

       theline = str( fid.readline() )
       datai = theline.split()
       Nrz   = int( datai[0] )
       receiver_depths = np.zeros(Nrz)
       for i in range(Nrz):
           receiver_depths[i] = float( datai[i+1] )

       theline = str( fid.readline() )
       datai = theline.split()
       Nrr   = int( datai[0] )
       receiver_ranges = np.zeros(Nrr)
       for i in range(Nrr):
           receiver_ranges[i] = float( datai[i+1] )

       Narr      = np.zeros( (Nrr,         Nrz, Nsz) )	
       A         = np.zeros( (Nrr, Narrmx, Nrz, Nsz) ) + 1j*np.zeros( (Nrr, Narrmx, Nrz, Nsz) )
       delay     = np.zeros( (Nrr, Narrmx, Nrz, Nsz) ) + 1j*np.zeros( (Nrr, Narrmx, Nrz, Nsz) )
       SrcAngle  = np.zeros( (Nrr, Narrmx, Nrz, Nsz) )
       RcvrAngle = np.zeros( (Nrr, Narrmx, Nrz, Nsz) )
       NumTopBnc = np.zeros( (Nrr, Narrmx, Nrz, Nsz) )
       NumBotBnc = np.zeros( (Nrr, Narrmx, Nrz, Nsz) )

       for isd in range(Nsz):
           Narrmx2 = int( fid.readline() )
           for ird in range(Nrz):
               for irr in range(Nrr):
                   narr = int( fid.readline() )
                   Narr[ irr, ird, isd ] = narr
                   maxnarr = max( narr, maxnarr )
                   if narr > 0:
                      narr = min( narr, Narrmx )
                      for k in range(narr):
                          theline = str( fid.readline() )
                          datai = theline.split()
                          amp   = float( datai[0] )
                          phase = float( datai[1] )
                          A[ irr, k, ird, isd ] = amp*np.exp( 1j*phase*np.pi/180.0 )
                          rtau = float( datai[2] )
                          itau = float( datai[3] )
                          delay[ irr, k, ird, isd ] = rtau + 1j*itau
                          source_angle = float( datai[4] ) 
                          SrcAngle[ irr, k, ird, isd ] = source_angle
                          receiver_angle = float( datai[5] ) 
                          RcvrAngle[ irr, k, ird, isd ] = receiver_angle
                          bounces = int( datai[6] )
                          NumTopBnc[ irr, k, ird, isd ] = bounces
                          bounces = int( datai[7] )		       
                          NumBotBnc[ irr, k, ird, isd ] = bounces
       A         = A[        :,0:maxnarr,:,:]
       delay     = delay[    :,0:maxnarr,:,:]
       SrcAngle  = SrcAngle[ :,0:maxnarr,:,:]
       RcvrAngle = RcvrAngle[:,0:maxnarr,:,:]
       NumTopBnc = NumTopBnc[:,0:maxnarr,:,:]
       NumBotBnc = NumBotBnc[:,0:maxnarr,:,:]       
       
       Pos = {'freq':freq,'source_depths':source_depths,'receiver_depths':receiver_depths,'receiver_ranges':receiver_ranges}
       Arr = {'Narr':Narr,'A':A,'delay':delay,'SrcAngle':SrcAngle,'RcvrAngle':RcvrAngle,'NumTopBnc':NumTopBnc,'NumBotBnc':NumBotBnc}
    else:

       freq  = float( fid.readline() ) 

       theline = str( fid.readline() )
       datai = theline.split()
       Nsx   = int( datai[0] )
       sourcex = np.zeros(Nsx)
       for i in range(Nsx):
           sourcex[i] = float( datai[i+1] )

       theline = str( fid.readline() )
       datai = theline.split()
       Nsy   = int( datai[0] )
       sourcey = np.zeros(Nsy)
       for i in range(Nsy):
           sourcey[i] = float( datai[i+1] )

       theline = str( fid.readline() )
       datai = theline.split()
       Nsz   = int( datai[0] )
       sourcez = np.zeros(Nsz)
       for i in range(Nsz):
           sourcez[i] = float( datai[i+1] )

       theline = str( fid.readline() )
       datai = theline.split()
       Nrz   = int( datai[0] )
       receiver_depths = np.zeros(Nrz)
       for i in range(Nrr):
           receiver_depths[i] = float( datai[i+1] )

       theline = str( fid.readline() )
       datai = theline.split()
       Nrr = int( datai[0] )
       receiver_ranges = np.zeros(Nrr)
       for i in range(Nrr):
           receiver_ranges[i] = float( datai[i+1] )

       theline = str( fid.readline() )
       datai = theline.split()
       Nrtheta = int( datai[0] )
       receiver_thetas = np.zeros(Nrtheta)
       for i in range(Nrtheta):
           receiver_thetas[i] = float( datai[i+1] )

       Narr      = np.zeros( (Nrr,         Nrz, Nrtheta, Nsz) )	
       A         = np.zeros( (Nrr, Narrmx, Nrz, Nrtheta, Nsz) ) + 1j*np.zeros( (Nrr, Narrmx, Nrz, Nrtheta, Nsz) )
       delay     = np.zeros( (Nrr, Narrmx, Nrz, Nrtheta, Nsz) ) + 1j*np.zeros( (Nrr, Narrmx, Nrz, Nrtheta, Nsz) )
       SrcDeclAngle = np.zeros( (Nrr, Narrmx, Nrz, Nrtheta, Nsz) )
       SrcAzimAngle = np.zeros( (Nrr, Narrmx, Nrz, Nrtheta, Nsz) )
       RcvrDeclAngle= np.zeros( (Nrr, Narrmx, Nrz, Nrtheta, Nsz) )
       RcvrAzimAngle= np.zeros( (Nrr, Narrmx, Nrz, Nrtheta, Nsz) )
       NumTopBnc    = np.zeros( (Nrr, Narrmx, Nrz, Nrtheta, Nsz) )
       NumBotBnc    = np.zeros( (Nrr, Narrmx, Nrz, Nrtheta, Nsz) )

       for isd in range(Nsz):
           Narrmx2 = int( fid.readline() )
           for irtheta in range(Nrtheta):
               for irz in range(Nrz):
                   for irr in range(Nrr):
                       Narr[ irr, ird, irtheta, isd ] = narr
                       maxnarr = max( narr, maxnarr )
                       if narr > 0:
                          narr = min( narr, Narrmx )
                          for k in range(narr):
                              theline = str( fid.readline() )
                              datai = theline.split()
                              amp   = float( datai[0] )
                              phase = float( datai[1] )
                              A[ irr, k, ird, irtheta, isd ] = amp*np.exp( 1j*phase*np.pi/180.0 )
                              rtau = float( datai[2] )
                              itau = float( datai[3] )
                              delay[ irr, k, ird, irtheta, isd ] = rtau + 1j*itau
                              theangle = float( datai[4] ) 
                              SrcDeclAngle[ irr, k, ird, irtheta, isd ] = theangle
                              theangle = float( datai[5] ) 
                              SrcAzimAngle[ irr, k, ird, irtheta, isd ] = theangle
                              theangle = float( datai[6] ) 
                              RcvrDeclAngle[ irr, k, ird, irtheta, isd ] = theangle
                              theangle = float( datai[7] )
                              RcvrAzimAngle[ irr, k, ird, irtheta, isd ] = theangle
                              bounces = int( datai[8] )
                              NumTopBnc[ irr, k, ird, irtheta, isd ] = bounces
                              bounces = int( datai[9] )		       
                              NumBotBnc[ irr, k, ird, irtheta, isd ] = bounces
       A            = A[            :,0:maxnarr,:,:,:]
       delay        = delay[        :,0:maxnarr,:,:,:]
       SrcDeclAngle = SrcDeclAngle[ :,0:maxnarr,:,:,:]
       SrcAzimAngle = SrcAzimAngle[ :,0:maxnarr,:,:,:]
       RcvrDeclAngle= RcvrDeclAngle[:,0:maxnarr,:,:,:]
       RcvrAzimAngle= RcvrAzimAngle[:,0:maxnarr,:,:,:]
       NumTopBnc = NumTopBnc[:,0:maxnarr,:,:,:]
       NumBotBnc = NumBotBnc[:,0:maxnarr,:,:,:]
       Pos = {'freq':freq,'sourcex':sourcex,'sourcey':sourcey,'sourcez':sourcez,'receiver_depths':receiver_depths,'receiver_ranges':receiver_ranges,'receiver_thetas':receiver_thetas}
       Arr = {'Narr':Narr,'A':A,'delay':delay,'SrcDeclAngle':SrcDeclAngle,'SrcAzimAngle':SrcAzimAngle,'RcvrDeclAngle':RcvrDeclAngle,'RcvrAzimAngle':RcvrAzimAngle,'NumTopBnc':NumTopBnc,'NumBotBnc':NumBotBnc}

    fid.close()
    return Arr,Pos