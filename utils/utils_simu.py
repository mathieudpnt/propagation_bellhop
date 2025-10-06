"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
Created on Fri Jan  3 17:11:04 2025
@author: xdemoulin
"""

import numpy as np
from numpy.fft import ifft,fft
from scipy.signal import chirp
import os
from utils.utils_acoustic_toolbox import write_env_file,read_arrivals_asc
from utils.core_utils import FindPow2, Coef_Rbot2, Coef_Surf, Atten_FG, find_nearest, compute_sound_speed
from pathlib import Path
from netCDF4 import Dataset
from collections import namedtuple
import pandas as pd
from collections.abc import Iterable

def read_croco(file: Path):
    """
    Reads the output from the Croco model NetCDF file and extracts relevant variables.

    Parameters
    ----------
    file : Path
        The path to the Croco model output NetCDF file.

    Returns
    -------
    a nemed tuple containing the following variables
        temperature : ndarray
            Temperature data (time, depth, lat, lon).
        salinity : ndarray
            Salinity data (time, depth, lat, lon).
        depth : ndarray
            Depth values (depth, lat, lon).
        lon : ndarray
            Latitudes in a meshgrid format.
        lat : ndarray
            Longitudes in a meshgrid format.
    """
    with Dataset(file, 'r') as ncf:
        temperature = ncf.variables['T'][:] if 'T' in ncf.variables else None  # Temperature
        salinity = ncf.variables['S'][:] if 'S' in ncf.variables else None  # Salinity
        depth = ncf.variables['z_r'][:] if 'z_r' in ncf.variables else None  # Depth
        lon = ncf.variables['lon_r'][:] if 'lon_r' in ncf.variables else None  # Longitude grid
        lat = ncf.variables['lat_r'][:] if 'lat_r' in ncf.variables else None  # Latitude grid

        # Check if all required variables were found
        if any(var is None for var in [temperature, salinity, depth, lon, lat]):
            raise KeyError("One or more of the required variables are missing from the NetCDF file")

    # Optionally, squeeze the arrays to remove singleton dimensions
    temperature = np.squeeze(temperature)
    salinity = np.squeeze(salinity)
    lon = np.squeeze(lon)
    lat = np.squeeze(lat)
    depth = np.squeeze(depth)

    croco_data = namedtuple('croco_data', ['temperature', 'salinity', 'depth', 'lon', 'lat'])

    return croco_data(temperature=temperature, salinity=salinity, depth=depth, lon=lon, lat=lat)


def read_bathy(file: Path, lim_lat: list[float], lim_lon: list[float]):
    """
    Reads a bathymetric file and extracts the data that falls within the specified latitude and longitude limits.

    Parameters
    ----------
    file : Path
        The path to the bathymetric file (typically in ASCII format).
    lim_lat : list of float
        A list containing the minimum and maximum latitudes to extract from the bathymetric data, in the form [min_lat, max_lat].
    lim_lon : list of float
        A list containing the minimum and maximum longitudes to extract from the bathymetric data, in the form [min_lon, max_lon].

    Returns
    -------
    lat_extract : ndarray
        A 1D array of the latitudes that correspond to the extracted region based on `lim_lat`.
    lon_extract : ndarray
        A 1D array of the longitudes that correspond to the extracted region based on `lim_lon`.
    elev : ndarray
        A 2D array of the bathymetric elevation data for the extracted region.
    """

    # header infos
    with open(file, "r") as f:
        header = [next(f).strip() for _ in range(6)]
    nb_lon = int(header[0].split()[1])
    nb_lat = int(header[1].split()[1])
    lon_min = float(header[2].split()[1])
    lat_min = float(header[3].split()[1])
    grid_resolution = float(header[4].split()[1])
    lon_max = lon_min + (nb_lon * grid_resolution)
    lat_max = lat_min + (nb_lat * grid_resolution)

    data = np.loadtxt(file, skiprows=6)
    lat = np.linspace(lat_max, lat_min, nb_lat)
    lon = np.linspace(lon_min, lon_max, nb_lon)

    # extract data according to lim_lat/lim_lon
    data_extract = data[(lat >= lim_lat[0]) & (lat <= lim_lat[1])][:, (lon >= lim_lon[0]) & (lon <= lim_lon[1])]
    lat_extract = lat[(lat >= lim_lat[0]) & (lat <= lim_lat[1])]
    lon_extract = lon[(lon >= lim_lon[0]) & (lon <= lim_lon[1])]

    return lat_extract, lon_extract, data_extract


def extract_bathy(source: pd.Series, station: pd.Series, lat: np.ndarray, lon: np.ndarray, elev: np.ndarray):
    """
    Extracts bathymetric depth along a transect between a source and a station.

    This function samples bathymetric elevation along a straight-line transect
    from a source point to a station point, interpolating depth values at
    regular intervals. It also generates a corresponding distance vector and
    depth levels for compatibility with CROCO.

    Parameters
    ----------
    source : pd.Series
        A Pandas Series containing the latitude ('lat') and longitude ('lon')
        of the source point.
    station : pd.Series
        A Pandas Series containing the latitude ('lat'), longitude ('lon'), and
        distance ('distance', in km) of the destination point.
    lat : np.ndarray
        A 1D NumPy array of latitude values from the bathymetric dataset.
    lon : np.ndarray
        A 1D NumPy array of longitude values from the bathymetric dataset.
    elev : np.ndarray
        A 2D NumPy array representing the bathymetric elevation (depth) at each (lat, lon) point.

    Returns
    -------
    zb : list of float
        A list containing the extracted bathymetric depths along the transect.
    dist : np.ndarray
        A 1D NumPy array of distances (in km) from the source to the station along the transect.
    z_transect : np.ndarray
        A 1D NumPy array of 32 depth levels (for compatibility with CROCO), adjusted to the maximum bathymetric depth
        (rounded down to the nearest multiple of 5 meters).
    nb_layer : int
        The number of horizontal layers along the transect, assuming a 25-meter spacing.

    Notes
    -----
    The bathymetric depth is extracted from the nearest available latitude
    and longitude in the bathymetric dataset.
    """
    # at least 15 points or 10 points/km for the transect
    nb_p = max(15, int(10 * station['distance']))

    # generates latitudes and longitudes along the transect
    lat_i = np.linspace(source['lat'], station['lat'], nb_p)
    lon_i = np.linspace(source['lon'], station['lon'], nb_p)

    # extracts bathymetric depth along the transect from lat/lon grids
    zb =  []
    for i,j in zip(lat_i, lon_i):
        nearest_lat = find_nearest(lat, i)
        nearest_lon = find_nearest(lon, j)
        zb.append(abs(elev[nearest_lat, nearest_lon]))

    # generates a distance vector from 0 to nb_p between source and receptor
    dist = np.linspace(start=0, stop=station['distance'], num=nb_p)

    z_max = max(zb) - (max(zb) % 5)  # maximum depth with a 5-meter resolution
    z_transect = np.linspace(0, z_max,32) # for compatibility with CROCO

    # number of horizontal layers, assuming a 25-meter spacing
    # TODO, QUESTION MD: pas sûr de ma définition ici
    nb_layer = int((station['distance'] * 1000) // 25)
    
    return zb, dist, z_transect, nb_layer


def sound_speed_profile(method: str, yday: int, z, lon_ref: float, lat_ref: float, croco_data: namedtuple):
    """
    Computes the sound speed at specific depths based on environmental conditions from CROCO model output.

    This function calculates the sound speed profile at the given reference location (latitude, longitude) and
    depth (z), using temperature, salinity, and pressure data from the CROCO model. The sound speed is computed
    using a specified equation of state for seawater.

    Parameters
    ----------
    method : str
        The method to use for sound speed computation (e.g., 'mackenzie', 'chen', 'del_grosso', etc.).
    yday : int
        The day index in the model output (e.g., for a specific day in a time series).
    z : array-like
        A 1D array or list representing the vertical depth (z) coordinates, ordered from the surface to the bottom.
    lon_ref : float
        The reference longitude for the location where the sound speed will be calculated.
    lat_ref : float
        The reference latitude for the location where the sound speed will be calculated.
    croco_data : namedtuple
        A namedtuple containing the model data, including fields for 'salinity', 'temperature', and others at each
        time step, depth level, and grid location.

    Returns
    -------
    Cw : array-like
        An array of sound speed values at the given depths, computed using the specified equation of state.

    Notes
    -----
    - The depth levels (z) should be ordered from the surface to the bottom.
    - Pressure is calculated based on the depth values (z) in decibars.
    - The salinity and temperature data are extracted for the specified day and nearest grid location to the reference
      latitude and longitude.
    """
    # Find the nearest grid point to the given reference latitude and longitude
    idx_lat = find_nearest(croco_data.lat[:, 0], lat_ref)  # Index of the closest latitude in the CROCO grid
    idx_lon = find_nearest(croco_data.lon[0, :], lon_ref)  # Index of the closest longitude in the CROCO grid

    # Extract salinity, temperature, and calculate pressure at the selected depth
    # Salinity (S), temperature (T) at the specified day of the year (yday) and location (lat_c0, lon_c0)
    salinity = croco_data.salinity[yday, :, idx_lat, idx_lon]  # Salinity profile
    temperature = croco_data.temperature[yday, :, idx_lat, idx_lon]  # Temperature profile

    #TODO QUESTION MD: pourquoi ne pas utiliser les données profondeur CROCO ?
    z2 = abs(croco_data.depth[:, idx_lat, idx_lon])
    z_max = max(z2) - (max(z2) % 5)
    z_transect = np.linspace(0, z_max, 32)  # for compatibility with CROCO
    pressure2 = 10 + z_transect

    # Pressure (P) is calculated from depth (zw) in decibars (dBar)
    # p = p_ref + (rho * g * z) / 1e4, assuming rho=1000 kg/m3 and g=10m/s2 then,
    # P = 10 + z, where z is the depth in meters, assuming a reference pressure at 10 dBar for surface conditions
    pressure = 10 + z

    sound_speed = compute_sound_speed(salinity, temperature, pressure, equation=method)

    return sound_speed[::-1]


# TODO, Question MD: Manque explication sur Bellhop, qu'est ce qui est fait dans cette fonction ? à quoi correspondent les calc I/E/A ?
def run_bellhop(executable: Path, bellhop_dir: Path, filename: str, calc: str | list[str], z_max: int | float, source: pd.Series, station: pd.Series, dist: np.array, zb: Iterable, sound_speed: np.array, z_transect: np.array, param_seabed: pd.Series):
    """
    Runs the Bellhop acoustic model to compute underwater sound propagation.

    Parameters
    ----------
    executable : Path
        Path to the Bellhop executable file.
    bellhop_dir : Path
        Directory where Bellhop input and output files will be stored.
    filename : str
        Base name for the Bellhop files (without extension).
    calc : str or list of str
        Specifies the type of calculation (e.g., 'I' for incoherent, 'C' for coherent, etc.).
    z_max : int or float
        Maximum depth for the propagation model.
    source : pd.Series
        A Pandas Series containing source parameters.
    station : pd.Series
        A Pandas Series containing station parameters.
    dist : np.array
        A NumPy array representing distances along the propagation path.
    zb : list
        A list containing bathymetric depth values along the path.
    sound_speed : np.array
        A NumPy array containing the sound speed profile at different depths.
    z_transect : np.array
        A NumPy array representing the vertical depth (z) coordinates of the propagation path.
    param_seabed : pd.Series
        A Pandas Series containing seabed parameters.

    Returns
    -------
    None
        This function does not return anything but generates output files from the Bellhop model.

    Notes
    -----
    - The Bellhop model is used to compute acoustic transmission loss based on the environment.
    - The function writes necessary input files and executes Bellhop to perform the calculation.
    - The '.bty' bathymetry file is created based on the provided seabed profile.
    """

    # Ensure the Bellhop directory exists
    bellhop_dir.mkdir(parents=True, exist_ok=True)

    # Compute central frequency of the sound source
    f_cen = int((source['f_min'] + source['f_max']) / 2)

    # Define the number of bathymetry points (ensuring at least 15 points)
    nb_p = max(15, int(10 * station['distance']))  # Sampling at 100m intervals

    # Compute the number of rays based on distance (one ray per 25 meters)
    nb_ray = int((dist[-1] * 1000) // 25)

    # Ensure 'calc' is a list (even if a single string is provided)
    if isinstance(calc, str):
        calc = [calc]

    # Iterate over calculation types (e.g., incoherent, coherent, etc.)
    for c in calc:

        # Generate Bellhop environment file
        write_env_file(
            bellhop_dir, f"{filename}{c}", f_cen, source['depth'], station['depth'],
            sound_speed, z_transect, station['distance'], nb_ray, z_max,
            param_seabed, source['opening_angle'], f"{c}"
        )

        # Write bathymetry (.bty) file
        with open(bellhop_dir / f"{filename}{c}.bty", 'w') as fid:
            fid.write("L\n")
            fid.write(f"{nb_p}\n")
            for i in range(len(zb)):
                fid.write(f"{np.squeeze(dist[i])} {np.squeeze(zb[i])} /\n")

        # Run Bellhop model using system command
        os.system(f'{executable} {bellhop_dir / (filename + c)}')

    return


# TODO, Question MD: pas d'explications/commentaires sur cette fonction
def impulse_response(file, source, station, param_water, param_seabed, param_env):
    [Arr, _] = read_arrivals_asc(file)
    
    D = 1000 * station['distance']

    fmin = source['f_min']
    fmax = source['f_max']
    fe = source['fe']
    t0 = source['t0']
    ponder = source['ponder']
    opening_angle = source['opening_angle']
    Dir = source['dir']

    salinity = param_water['salinity']
    temperature = param_water['temperature']
    Ph = param_water['pH']
    W = param_env
    Ta = D / np.cos(np.pi/180*opening_angle) / 1500
    nbp = 2**FindPow2((Ta + 2*t0) * fe)
    Freq = np.arange(0, fe, fe/nbp)
    Temp = np.arange(0, nbp/fe, 1/fe)
    
    if source['type'] == 'click_M':
        f0 = (fmin + fmax) / 2
        u = np.where(Temp <= t0)
        se = np.zeros(len(Temp))
        se[u] = np.blackman(len(u)) * np.sin(2 * np.pi * f0 * Temp[u])
        Se = fft(se) / (nbp/2)
        z0 = se
    elif source['type'] == 'whistle_D':
        u = np.where(Temp <= t0)
        z0 = chirp(Temp[u], fmin, t0, fmax,'li')
        se = np.concatenate((z0, np.zeros(len(Temp)-len(z0))), axis=0)
        Se = fft(se) / (nbp/2)
    # elif source_type== 'FMH':
    #     z0 = chirp(Temp(u),fmin,T0,fmax,'q')
    #     y0 = np.concatenate(z0,np.zeros(1,len(Temp)-len(z0)))
    
    t = np.squeeze(Arr["delay"].real)  # temps d'arrivées
    a = np.squeeze(Arr["A"])  # amplitudes arrivées
    Tet = np.squeeze(1/2*(abs(Arr["SrcAngle"])+abs(Arr["RcvrAngle"])))  # angles aux interfaces (tant pis si c'est brutal)
    Dis = 1500 * t  # pareil, brutus approximus
    ns = np.squeeze(Arr["NumTopBnc"])
    nb = np.squeeze(Arr["NumBotBnc"])
    
    # Boucle en Freq
    n1 = np.where(Freq>=fmin)[0][0]
    n2 = np.where(Freq>=fmax)[0][0]
    n_win = n2-n1+1
    tmin0 = 0
    inc = 0
    y = np.zeros((len(Freq),len(t))) + 1j*np.zeros((len(Freq),len(t)))
    
    if ponder == 1:
        Se[n1:n2] = np.hamming(n_win-1) * Se[n1:n2]
    
    for ni in np.arange(n1, n2+1):
        inc = inc + 1
        fk = Freq[ni] / 1000
        Rs = -Coef_Surf(Tet,W,fk)
        Rb = Coef_Rbot2(Tet, [1500,1,0], param_seabed, Freq[ni]) # pour Iroise, ca ira ..
        Atv = Atten_FG(fk, salinity, temperature, 10, Ph)*Dis/1000 #(en dB)
        atten = np.exp(-Atv*np.log(10)/20) # en Neper
        y[ni,:] = Se[ni]*atten*(Rs**(ns))*(Rb**(nb))*np.exp(-1j*2*np.pi*Freq[ni]*(t-tmin0)) # *Se[ni]*
    
    spec = y.sum(axis=1)
    spec[0] = 0
    spec[n2] = 0
    mil = round(len(Freq)/2)
    toto = spec[:mil].conj()
    spec[mil:] = np.flipud(toto) 
    
    Ri_T = ifft(spec) #/(nbp/2)

    T_ri = tmin0 + Temp
    F_ri = Freq
    Ri_F = spec
    S_Emis = z0

    return S_Emis, Ri_T, T_ri, F_ri, Ri_F