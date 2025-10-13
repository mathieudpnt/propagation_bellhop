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
        fid.write("'CVWF'\n")  # See file: ./acoustic_toolbox/doc/EnvironmentalFile.html
        fid.write("19.3 35. 8. 50.\n")  # Temperature, Salinity, pH, Depth of bar (fixed values)
        #TODO rentrer dynamiquement les valeurs CROCO + commentaires
        fid.write(f"0 0.0 {zmax}\n")  # Depth range (0 to zmax)

        # Write sound speed profile
        for z, c in zip(zw, sound_speed):
            fid.write(f"{z} {c} /\n")
        #TODO erreur ici, on écrit mal le profil de celerité il semble, il faut le point le + profond de la radiale et en déduire le profil de c
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

    return rmax


def read_shd(filename=None, xs=None, ys=None):
    """
    Reads the .shd file resulting from using Bellhop and extracts the
    pressure field around the source and the geometry of the zone
    Only runs if the incoherent calculation type (I) is chosen

    Parameters
    ----------
    filename :
        name of the .shd file to read
    xs :
        latitude of the source
    ys :
        longitude of the source

    Returns
    -------
    pressure :
        NumPy array containing the values of the calculated acoustic
        pressure field at different positions.
    geometry :
        dictionary containing geometric information about the problem

    Notes
    -----
    # Arraial do Cabo, Qui Out 20 21:48:55 WEST 2016
    # Written by Tordar
    # Based on read_shd_bin.m by Michael Porter
    """
    fid = open(filename,'rb')
    recl  = int(np.fromfile(fid,np.int32,1))
    title = fid.read(80)
    fid.seek( 4*recl )
    PlotType = fid.read(10)
    fid.seek( 2*4*recl ) # reposition to end of second record
    freq   = float( np.fromfile( fid, np.float32, 1 ) )
    Ntheta = int(   np.fromfile( fid, np.int32  , 1 ) )
    Nsx    = int(   np.fromfile( fid, np.int32  , 1 ) )
    Nsy    = int(   np.fromfile( fid, np.int32  , 1 ) )
    Nsd    = int(   np.fromfile( fid, np.int32  , 1 ) )
    Nrd    = int(   np.fromfile( fid, np.int32  , 1 ) )
    Nrr    = int(   np.fromfile( fid, np.int32  , 1 ) )
    atten  = float( np.fromfile( fid, np.float32, 1 ) )
    fid.seek( 3 * 4 * recl ) # reposition to end of record 3
    thetas = np.fromfile( fid, np.float32, Ntheta )
    if  ( PlotType[ 0 : 1 ] != 'TL' ):
       fid.seek( 4 * 4 * recl ) # reposition to end of record 4
       Xs     = np.fromfile( fid, np.float32, Nsx )
       fid.seek( 5 * 4 * recl )  # reposition to end of record 5
       Ys     = np.fromfile( fid, np.float32, Nsy )
    else:   # compressed format for TL from FIELD3D
       fid.seek( 4 * 4 * recl ) # reposition to end of record 4
       Pos_S_x     = np.fromfile( fid, np.float32, 2 )
       Xs          = np.linspace( Pos_S_x[0], Pos_S_x[1], Nsx )
       fid.seek( 5 * 4 * recl ) # reposition to end of record 5
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


def plotray():
    """
    Reads the .ray file resulting from using Bellhop, extracts the
    information about eigenrays, the coordinates of a finite number of points on each beam,
    and creates a graph showing their propagation
    Only runs if the eigenvalues calculation type (E) is chosen

    Parameters
    ----------
    filename :
        name of the .ray file to read

    Returns
    -------
    None
        This function plots the eigenrays path on one graph

    Notes
    -----
    # Faro, Qua Dez  7 21:09:53 WET 2016
    # Written by Tordar
    # Based on plotray.m by Michael Porter
    """
    Nsxyz       = np.zeros(3) # source coordinates
    NBeamAngles = np.zeros(2) # number of beam angles

    # header reading
    fid = open(file,'r')
    title =        fid.readline() # title
    freq  = float( fid.readline() ) #frequency
    theline = str( fid.readline() )
    datai = theline.split()
    for i in range(len(datai)) :
        Nsxyz[i] = int( datai[i] )

    theline = str( fid.readline() )
    datai = theline.split()
    NBeamAngles[0] = int( datai[0] ) # number of elevation beams
    NBeamAngles[1] = int( datai[1] ) # number of azimutal beams

    deptht = float( fid.readline() ) # top depth
    depthb = float( fid.readline() ) # bottom depth
    Type   = fid.readline() # coordinate type (x,y,z for 3D cartesian and rz for 2D cylindrical)
    Nsx = int( Nsxyz[0] ) # x source coordinate
    Nsy = int( Nsxyz[1] ) # y source coordinate
    Nsz = int( Nsxyz[2] ) # z source depth
    Nalpha = int( NBeamAngles[0] ) # number of elevation beams
    Nbeta  = int( NBeamAngles[1] ) # number of azimuthal beams

    # axis limits
    rmin =  1.0e9
    rmax = -1.0e9
    zmin =  1.0e9
    zmax = -1.0e9

    # for each beam, the coordinates of nsteps points will be extracted from the initial file
    for ibeam in range(Nalpha):
        theline = str( fid.readline() ) # departure angle of the  beam
        l = len( theline )
        if l > 0: # loop until the end of the document
           theline = str( fid.readline() )
           datai = theline.split()
           nsteps    = int ( datai[0] ) # number of points on the beam
           NumTopBnc = int( datai[1] ) # number of top reflexions
           NumBotBnc = int( datai[2] ) # number of bottom reflexions
           r = np.zeros(nsteps) # initiation of range array
           z = np.zeros(nsteps) # initiation of depth array

           # extraction of the coordinates for each point
           for nj in range(nsteps):
               theline = str(fid.readline())
               rz = theline.split()
               r[nj] = float( rz[0] )
               z[nj] = float( rz[1] )

           # truncating the coordinates to r and z limits
           rmin = min( [ min(r), rmin ] )
           rmax = max( [ max(r), rmax ] )
           zmin = min( [ min(z), zmin ] )
           zmax = max( [ max(z), zmax ] )

           # creating the graph
           plt.plot( r, -z )
           plt.axis([rmin,rmax,-zmax,-zmin])

    fid.close()
    return Nalpha, NBeamAngles

def read_arrivals_asc(filename=None):
    """
    Reads the .asc file resulting from using Bellhop and extracts the ray's arrival times
    Only runs if the arrival calculation type (A) is chosen

    Parameters
    ----------
    filename :
        name of the .asc file to read

    Returns
    -------
    Arr :
        Array that contains information about the rays path
        (total number of studied rays, wave equation, complex delay, departure angle, arrival angle,
         number of top reflexions, number of bottom reflexions)
    Pos :
        Array that contains information about the problem
        (frequency, source depth, receiver depth, receiver range)

    Notes
    -----
    The structure of the .arr file is detailed in the file EXEMPLE.arr
    # Faro, Seg 11 Abr 2022 12:50:20 WEST
    # Written by Orlando Camargo Rodriguez
    # Based on read_arrivals_asc.m by Michael Porter
    """

    narrmx = 100
    fid = open(filename,'r')

    flag = str( fid.readline() ) # dimension
    freq  = float( fid.readline() ) # frequency

    # source depth
    theline = str( fid.readline() )
    datai = theline.split()
    nsz   = int( datai[0] ) # number of sources
    source_depth = float( datai[1] )

    # receiver depth
    theline = str( fid.readline() )
    datai = theline.split()
    nrz   = int( datai[0] ) #number of receivers depths
    receiver_depth = float( datai[1] )

    # receiver range
    theline = str( fid.readline() )
    datai = theline.split()
    nrr   = int( datai[0] ) # number of receivers ranges
    receiver_range = float( datai[1] )


    # Initiation of the components of Arr
    Narr      = int( fid.readline() ) # total number of studied rays
    A         = np.zeros( narrmx ) + 1j*np.zeros( narrmx ) # wave equation
    delay     = np.zeros( narrmx ) + 1j*np.zeros( narrmx ) #complex delay
    SrcAngle  = np.zeros( narrmx ) # departure angle
    RcvrAngle = np.zeros( narrmx ) # arrival angle
    NumTopBnc = np.zeros( narrmx, ) # number of top reflexions
    NumBotBnc = np.zeros( narrmx ) # number of bottom reflexions

    # reducing the size of the table to the desired size maxnarr
    Narrmx2 = int( fid.readline() ) # number of rays studied

    # Reading of the .arr file generated by bellhop
    for k in range(narrmx):
        theline = str( fid.readline() )
        datai = theline.split()
        amp   = float( datai[0] ) # amplitude
        phase = float( datai[1] ) # phase
        A[ k ] = amp*np.exp( 1j*phase*np.pi/180.0 ) # complex wave equation
        rtau = float( datai[2] ) # real part of delay
        itau = float( datai[3] ) # imaginary part of delay
        delay[ k ] = rtau + 1j*itau # complex delay
        SrcAngle[ k ] = float( datai[4] ) # departure angle
        RcvrAngle[ k ] = float( datai[5] ) # arrival angle
        NumTopBnc[ k ] = int( datai[6] ) # number of top reflexions
        NumBotBnc[ k ] = int( datai[7] ) # number of bottom reflexions

    # filling the output arrays
    Pos = {'freq':freq,'source_depth':source_depth,'receiver_depth':receiver_depth,'receiver_range':receiver_range}
    Arr = {'Narr':Narr,'A':A,'delay':delay,'SrcAngle':SrcAngle,'RcvrAngle':RcvrAngle,'NumTopBnc':NumTopBnc,'NumBotBnc':NumBotBnc}

    fid.close()
    return Arr,Pos