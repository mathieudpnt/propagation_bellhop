#!/usr/bin/env python3

"""Created on Mon Mar 25 15:38:01 2024.

@author: xdemoulin
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import dtype, float64, ndarray


def write_env_file(root : Path,  # noqa: PLR0913
                   env_name : str,
                   f_cen : float,
                   z_source : float,
                   z_reception : float,
                   sound_speed : np.array (Any),
                   zw : np.array(Any),
                   rmax : float,
                   nb_ray : int,
                   zmax : float,
                   param_seabed : pd.Series(),
                   opening_angle : float,
                   calc : str ) -> float:
    """Generate an environment file (.env) for Bellhop acoustic propagation modeling.

    Parameters
    ----------
    root : Path
        Directory where the environment file will be saved.
    env_name : str
        Name of the environment file (without extension).
    f_cen : float
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
    with Path.open(root / Path(f"{env_name}.env"), "w") as fid:

        # Write file header
        fid.write(f"'{env_name}'\n")  # Environment file name
        fid.write(f"{f_cen}\n")  # Central frequency
        fid.write("1\n")  # Number of layers (single layer for water column)
        fid.write("'CVWF'\n")  # See file: ./acoustic_toolbox/doc/EnvironmentalFile.html
        fid.write("19.3 35. 8. 50.\n")  # Temperature, Salinity, pH, Depth of bar
        # TODO(): rentrer dynamiquement les valeurs CROCO+commentaires noqa:FIX002,TD003
        fid.write(f"0 0.0 {zmax}\n")  # Depth range (0 to zmax)

        # Write sound speed profile
        for z, c in zip(zw, sound_speed, strict=False):
            fid.write(f"{z} {c} /\n")
        # TODO(): erreur ici, on écrit mal le profil de celerité il semble, il faut
        #  le point le + profond de la radiale et en déduire le profil de c  # noqa:
        #  FIX002, TD003
        # Ensure the profile extends to zmax
        if zmax > zw[-1]:
            fid.write(f"{zmax} {sound_speed[-1]} /\n")

        # Seabed properties
        fid.write("A* 0.0\n")  # Bottom condition
        fid.write(f"{zmax} {param_seabed['bulk_soundspeed']} 0.0 {param_seabed
        ['bulk_density']} {param_seabed['attenuation']} 0.0 /\n")  # Bottom properties

        # Source depth
        fid.write("1\n")
        fid.write(f"{z_source} /\n")

        # Receiver or Eigenray settings
        if calc == "I":
            n_rz = round(zmax / 1) + 1  # Depth step = 1m
            fid.write(f"{n_rz}\n0 {zmax} /\n")
        elif calc in ("E", "A"):
            fid.write("1\n")
            fid.write(f"{z_reception} /\n")

        # Range settings
        if calc == "I":
            fid.write(f"{nb_ray}\n0 {rmax} /\nI\n1001\n")
        elif calc in ("E", "A"):
            fid.write("1\n")
            fid.write(f"{rmax} /\n{calc}\n50001\n")

        # Beam angles
        fid.write(f"{-opening_angle} {opening_angle} /\n")

        # Grid resolution settings
        fid.write(f"0. {10 * round(1.05 * zmax / 10)} "
                  f"{round(1.05 * 100 * rmax) / 100}\n")

    return rmax


def read_shd(filename=Path) -> tuple[ndarray[float], ndarray[float]]:  # noqa: C901
    """Read the .shd file resulting from using Bellhop.

    and extracts the pressure field around the source and the geometry of the zone
    Only runs if the incoherent calculation type (I) is chosen

    Parameters
    ----------
    filename : Path
        Path and name of the .shd file to read

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
    fid = Path.open(filename, "rb")
    recl  = int(np.fromfile(fid,np.int32,1))
    next(fid)
    fid.seek( 4*recl )
    plot_type = fid.read(10)
    fid.seek( 2*4*recl ) # reposition to end of second record
    freq   = float( np.fromfile( fid, np.float32, 1 ) )
    ntheta = int(   np.fromfile( fid, np.int32  , 1 ) )
    nsx    = int(   np.fromfile( fid, np.int32  , 1 ) )
    nsy    = int(   np.fromfile( fid, np.int32  , 1 ) )
    nsd    = int(   np.fromfile( fid, np.int32  , 1 ) )
    nrd    = int(   np.fromfile( fid, np.int32  , 1 ) )
    nrr    = int(   np.fromfile( fid, np.int32  , 1 ) )
    fid.seek( 3 * 4 * recl ) # reposition to end of record 3
    thetas = np.fromfile( fid, np.float32, ntheta )
    if  plot_type[ 0 : 1 ] != "TL":
       fid.seek( 4 * 4 * recl ) # reposition to end of record 4
       xs     = np.fromfile( fid, np.float32, nsx )
       fid.seek( 5 * 4 * recl )  # reposition to end of record 5
       ys     = np.fromfile( fid, np.float32, nsy )
    else:   # compressed format for TL from FIELD3D
       fid.seek( 4 * 4 * recl ) # reposition to end of record 4
       pos_s_x     = np.fromfile( fid, np.float32, 2 )
       xs          = np.linspace( pos_s_x[0], pos_s_x[1], nsx )
       fid.seek( 5 * 4 * recl ) # reposition to end of record 5
       pos_s_y     = np.fromfile( fid, np.float32, 2 )
       ys          = np.linspace( pos_s_y[0], pos_s_y[1], nsy )
    fid.seek( 6 * 4 * recl ) # reposition to end of record 6
    zs = np.fromfile( fid, np.float32, nsd )
    fid.seek( 7 * 4 * recl ) # reposition to end of record 7
    zarray =  np.fromfile( fid, np.float32, nrd )
    fid.seek( 8 * 4 * recl ) # reposition to end of record 8
    rarray =  np.fromfile( fid, np.float32, nrr )
    if plot_type == "rectilin  ":
       nrcvrs_per_range = nrd
    elif plot_type == "irregular ":
       nrcvrs_per_range = 1
    else:
       nrcvrs_per_range = nrd
    pressure = (np.zeros( (ntheta,nsd,nrcvrs_per_range,nrr) )
                + 1j*np.zeros( (ntheta,nsd,nrcvrs_per_range,nrr) ))
    if np.isnan( xs ):
      for itheta in range(ntheta):
          for isd in range( nsd ):
              for ird in range( nrcvrs_per_range ):
                  recnum = (9 + itheta * nsd * nrcvrs_per_range + isd
                            * nrcvrs_per_range + ird)
                  status = fid.seek( recnum * 4 * recl ) #Move to end of previous record
                  if status == -1:
                     pass
                  temp = np.fromfile( fid, np.float32, 2 * nrr ) # Read complex data
                  for k in range(nrr):
                      pressure[ itheta, isd, ird, k ] = (temp[ 2 * k ] + 1j
                                                         * temp[ 2*k + 1 ])

    else:
       xdiff = abs( xs - xs * 1000.0 )
       idx_x  = xdiff.argmin(0)
       ydiff = abs( ys - ys * 1000.0 )
       idx_y  = ydiff.argmin(0)
       for itheta in range(ntheta):
           for isd in range(nsd):
               for ird in range( nrcvrs_per_range ):
                   recnum = (9 + idx_x * nsy * ntheta * nsd * nrcvrs_per_range + idx_y *
                             ntheta * nsd * nrcvrs_per_range + itheta * nsd *
                             nrcvrs_per_range + isd * nrcvrs_per_range + ird)
                   status = fid.seek( recnum * 4 * recl )#Move to end of previous record
                   if status == -1:
                      pass
                   temp = np.fromfile( fid, np.float32, 2 * nrr ) # Read complex data
                   for k in range(nrr):
                       pressure[ itheta, isd, ird, k ] = (temp[ 2 * k ] + 1j *
                                                          temp[ 2*k + 1 ])

       fid.close()
    geometry = {"zs":zs, "f":freq,"thetas":thetas,"rarray":rarray,"zarray":zarray}

    return pressure,geometry


def plotray(filename : Path) -> tuple[int, ndarray[tuple[int], dtype[float64]]]:
    """Read bellhop file and plot the rays.

    Read the .ray file resulting from using Bellhop, extract the
    information about eigenrays, the coordinates of a finite number
    of points on each beam,
    and create a graph showing their propagation

    Only run if the eigenvalues calculation type (E) is chosen

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
    n_beam_angles = np.zeros(2) # number of beam angles
    # header reading
    fid = Path.open( filename )
    next(fid), next(fid), next(fid)

    data = str(fid.readline()).split()
    n_beam_angles[0] = int( data[0] ) # number of elevation beams
    n_beam_angles[1] = int( data[1] ) # number of azimutal beams

    next(fid), next(fid), next(fid)
    nalpha = int( n_beam_angles[0] ) # number of elevation beams

    # for each beam the coordinates of nsteps points are extracted from the initial file
    for _ibeam in range(nalpha):
        len_ = len(str( fid.readline() )) # departure angle of the  beam
        if len_ > 0: # loop until the end of the document
           data = str(fid.readline()).split()
           nsteps = int ( data[0] ) # number of points on the beam
           r = np.zeros(nsteps) # initiation of range array
           z = np.zeros(nsteps) # initiation of depth array

           # extraction of the coordinates for each point
           for nj in range(nsteps):
               rz = str(fid.readline()).split()
               r[nj] = float( rz[0] )
               z[nj] = float( rz[1] )

           # truncating the coordinates to r and z limits
           rmin = float(min([min(r), 1.0e9]))
           rmax = float(max([max(r), -1.0e9]))
           zmin = float(min([min(z), 1.0e9]))
           zmax = float(max([max(z), -1.0e9]))

           # creating the graph
           plt.plot( r, -z )
           plt.axis((rmin,rmax,-zmax,-zmin))

    fid.close()
    return nalpha

def read_arrivals_asc(filename= Path) -> tuple[
    dict[str, int | ndarray[tuple[Any, ...], dtype[Any]] | ndarray[tuple[int],
    dtype[float64]] | Any], dict[str, float]]:
    """Read the .asc file resulting from using Bellhop.

    and extracts the ray's arrival times
    Only runs if the arrival calculation type (a) is chosen

    Parameters
    ----------
    filename : Path
        Path and name of the .asc file to read

    Returns
    -------
    arr : ndarray
        Array that contains information about the rays path
        (total number of studied rays, wave equation,
        complex delay, departure angle, arrival angle,
        number of top reflexions, number of bottom reflexions)
    pos : ndarray
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
    fid = Path.open(filename)
    next(fid)
    freq  = float( fid.readline() ) # frequency

    # source depth
    data = str(fid.readline()).split()
    source_depth = float( data[1] )

    # receiver depth
    data = str(fid.readline()).split()
    receiver_depth = float( data[1] )

    # receiver range
    data = str(fid.readline()).split()
    receiver_range = float( data[1] )


    # Initiation of the components of arr
    narr      = int( fid.readline() ) # total number of studied rays
    a         = np.zeros( narrmx ) + 1j*np.zeros( narrmx ) # wave equation
    delay     = np.zeros( narrmx ) + 1j*np.zeros( narrmx ) #complex delay
    src_angle  = np.zeros( narrmx ) # departure angle
    rcvr_angle = np.zeros( narrmx ) # arrival angle
    num_top_bnc = np.zeros( narrmx ) # number of top reflexions
    num_bot_bnc = np.zeros( narrmx ) # number of bottom reflexions
    next(fid)

    # Reading of the .arr file generated by bellhop
    for k in range(narrmx):
        data = str(fid.readline()).split()
        amp   = float( data[0] ) # amplitude
        phase = float( data[1] ) # phase
        a[ k ] = amp*np.exp( 1j*phase*np.pi/180.0 ) # complex wave equation
        rtau = float( data[2] ) # real part of delay
        itau = float( data[3] ) # imaginary part of delay
        delay[ k ] = rtau + 1j*itau # complex delay
        src_angle[ k ] = float( data[4] ) # departure angle
        rcvr_angle[ k ] = float( data[5] ) # arrival angle
        num_top_bnc[ k ] = int( data[6] ) # number of top reflexions
        num_bot_bnc[ k ] = int( data[7] ) # number of bottom reflexions

    # filling the output arrays
    pos = {"freq":freq,"source_depth":source_depth,"receiver_depth":receiver_depth,
           "receiver_range":receiver_range}
    arr = {"narr":narr,"a":a,"delay":delay,"src_angle":src_angle,
           "rcvr_angle":rcvr_angle,"num_top_bnc":num_top_bnc,"num_bot_bnc":num_bot_bnc}

    fid.close()
    return arr,pos
