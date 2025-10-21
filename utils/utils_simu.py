#!/usr/bin/env python3

"""Created on Mon Mar 25 15:38:01 2024.

@author: xdemoulin
"""
from __future__ import annotations

import os
from collections import namedtuple
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from netCDF4 import Dataset
from numpy import dtype, float64, floating, ndarray
from numpy.fft import fft, ifft
from scipy.signal import chirp

from utils.core_utils import (
    atten_fg,
    compute_sound_speed,
    find_nearest,
    find_pow2,
    ref_coeff_bot,
    ref_coeff_surf,
)
from utils.utils_acoustic_toolbox import read_arrivals_asc, write_env_file

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd


def read_croco(file: Path) -> namedtuple:  # noqa: PYI024
    """Read the output from the Croco model NetCDF file and extracts relevant variables.

    Parameters
    ----------
    file : Path
        The path to the Croco model output NetCDF file.

    Returns
    -------
    a named tuple containing the following variables
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
    CrocoData = namedtuple(  # noqa: PYI024
        "CrocoData", ["temperature", "salinity", "depth", "lon", "lat"])
    with Dataset(file, "r") as ncf:
        #temperature, salinity, depth, longitude grid, latitude grid
        temperature = ncf.variables["T"][:] if "T" in ncf.variables else None
        salinity = ncf.variables["S"][:] if "S" in ncf.variables else None
        depth = ncf.variables["z_r"][:] if "z_r" in ncf.variables else None
        lon = ncf.variables["lon_r"][:] if "lon_r" in ncf.variables else None
        lat = ncf.variables["lat_r"][:] if "lat_r" in ncf.variables else None

        # Check if all required variables were found
        if any(var is None for var in [temperature, salinity, depth, lon, lat]):
            error=("One or more of the required variables are"
                   " missing from the NetCDF file")
            raise KeyError(error)

    # Optionally, squeeze the arrays to remove singleton dimensions
    temperature = np.squeeze(temperature)
    salinity = np.squeeze(salinity)
    lon = np.squeeze(lon)
    lat = np.squeeze(lat)
    depth = np.squeeze(depth)

    return CrocoData((temperature, salinity, depth, lon, lat))

def read_bathy(file: Path, lim_lat: list[float], lim_lon: list[float]) -> tuple[
    ndarray[tuple[Any, ...], Any], ndarray[tuple[Any, ...], Any],
    ndarray[tuple[Any, ...], Any]]:
    """Read a bathymetric file and extracts information.

    Extracts the data that falls within the specified latitude and longitude limits.

    Parameters
    ----------
    file : Path
        The path to the bathymetric file (typically in ASCII format).
    lim_lat : list of float
        A list containing the minimum and maximum latitudes to extract from
        the bathymetric data, in the form [min_lat, max_lat].
    lim_lon : list of float
        A list containing the minimum and maximum longitudes to extract from
        the bathymetric data, in the form [min_lon, max_lon].

    Returns
    -------
    lat_extract : ndarray
        A 1D array of the latitudes that correspond to the extracted region
        based on `lim_lat`.
    lon_extract : ndarray
        A 1D array of the longitudes that correspond to the extracted region
        based on `lim_lon`.
    elev : ndarray
        A 2D array of the bathymetric elevation data for the extracted region.

    """
    # header infos
    with Path.open(file) as f:
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
    data_extract = data[(lat >= lim_lat[0]) & (lat <= lim_lat[1])][:,
                   (lon >= lim_lon[0]) & (lon <= lim_lon[1])]
    lat_extract = lat[(lat >= lim_lat[0]) & (lat <= lim_lat[1])]
    lon_extract = lon[(lon >= lim_lon[0]) & (lon <= lim_lon[1])]

    return lat_extract, lon_extract, data_extract


def extract_bty(source: pd.Series, station: pd.Series, lat: np.ndarray,
                lon: np.ndarray, elev: np.ndarray) -> tuple[list[ndarray[Any,
                dtype[floating[Any]]]], ndarray[tuple[Any, ...], dtype[float64]]
                | ndarray[tuple[Any, ...], dtype[floating]] | ndarray[tuple[Any, ...],
                dtype[np.complexfloating]] | ndarray[tuple[Any, ...], dtype[Any]],
                ndarray[tuple[Any, ...], dtype[float64]], int]:
    """Extract bathymetric depth along a transect between a source and a station.

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
        A 2D NumPy array representing the bathymetric elevation (depth)
        at each (lat, lon) point.

    Returns
    -------
    zb : list of float
        A list containing the extracted bathymetric depths along the transect.
    dist : np.ndarray
        A 1D NumPy array of distances (in km) from the source to the station
        along the transect.
    z_transect : np.ndarray
        A 1D NumPy array of 32 depth levels (for compatibility with CROCO),
        adjusted to the maximum bathymetric depth
        (rounded down to the nearest multiple of 5 meters).
    nb_layer : int
        The number of horizontal layers along the transect, assuming a 25-meter spacing.

    Notes
    -----
    The bathymetric depth is extracted from the nearest available latitude
    and longitude in the bathymetric dataset.

    """
    # at least 15 points or 10 points/km for the transect
    nb_p = max(15, int(10 * station["distance"]))

    # generates latitudes and longitudes along the transect
    lat_i = np.linspace(source["lat"], station["lat"], nb_p)
    lon_i = np.linspace(source["lon"], station["lon"], nb_p)

    # extracts bathymetric depth along the transect from lat/lon grids
    zb =  []
    for i,j in zip(lat_i, lon_i, strict=False):
        nearest_lat = find_nearest(lat, i)
        nearest_lon = find_nearest(lon, j)
        zb.append(abs(elev[nearest_lat, nearest_lon]))

    # generates a distance vector from 0 to nb_p between source and receptor
    dist = np.linspace(start=0, stop=station["distance"], num=nb_p)

    z_max = max(zb) - (max(zb) % 5)  # maximum depth with a 5-meter resolution
    z_transect = np.linspace(0, z_max,32) # for compatibility with CROCO

    # number of horizontal layers, assuming a 25-meter spacing
    # TODO (QUESTION MD): pas sûr de ma définition ici  # noqa: FIX002, TD003
    nb_layer = int((station["distance"] * 1000) // 25)

    return zb, dist, z_transect, nb_layer


def sound_speed_profile(method: str, yday: int, z : np.array, ref_coord : tuple
                        [float, float], croco_data: namedtuple) -> np.array:  # noqa: PYI024
    """Compute the sound speed at specific depths based on environmental conditions.

    This function calculates the sound speed profile at the given reference location
    (latitude, longitude) and depth (z), using temperature, salinity, and pressure data
     from the CROCO model. The sound speed is computed using a specified equation of
     state for seawater.

    Parameters
    ----------
    method : str
        The method to use for sound speed computation
        (e.g., 'mackenzie', 'chen', 'del_grosso', etc.).
    yday : int
        The day index in the model output (e.g., for a specific day in a time series).
    z : array-like
        A 1D array or list representing the vertical depth (z) coordinates,
        ordered from the surface to the bottom.
    ref_coord : tuple
        Reference location (lon, lat) coordinates where the sound speed
        will be calculated.
    croco_data : namedtuple
        A namedtuple containing the model data, including fields for 'salinity',
        'temperature', and others at each time step, depth level, and grid location.

    Returns
    -------
    Cw : array-like
        An array of sound speed values at the given depths,
        computed using the specified equation of state.

    Notes
    -----
    - The depth levels (z) should be ordered from the surface to the bottom.
    - Pressure is calculated based on the depth values (z) in decibars.
    - The salinity and temperature data are extracted for the specified day and
    nearest grid location to the reference latitude and longitude.

    """
    # Find the nearest grid point to the given reference latitude and longitude
    idx_lat = find_nearest(croco_data.lat[:, 0], ref_coord[1])  # Index of the closest
                                                           # latitude in the CROCO grid
    idx_lon = find_nearest(croco_data.lon[0, :], ref_coord[0])  # Index of the closest
                                                           # longitude in the CROCO grid

    # Extract salinity, temperature, and calculate pressure at the selected depth
    # Salinity (S), temperature (T) at the specified day of the year (yday)
    # and location (lat_c0, lon_c0)
    salinity = croco_data.salinity[yday, :, idx_lat, idx_lon]  # Salinity profile
    temperature = croco_data.temperature[yday, :, idx_lat, idx_lon] #Temperature profile
    latitude = croco_data.lat[idx_lat, idx_lon]

    return [compute_sound_speed(s=sal, t=temp, d=z_i, equation=method, lat=latitude)
            for sal, temp, z_i in zip(salinity, temperature, z, strict=False)]

def run_bellhop(executable: Path, bellhop_dir: Path, filename: str,  # noqa: PLR0913
                calc: str | list[str], z_max: float, source: pd.Series,
                station: pd.Series, dist: np.array, zb: Iterable, sound_speed: np.array,
                z_transect: np.array, param_seabed: pd.Series) -> None:
    """Run the Bellhop acoustic model to compute underwater sound propagation.

    Based on the given parameters, this code creates the necessary Bellhop input files
    (environment, bathymetry, etc.) then runs Bellhop
    to compute ray paths, eigenrays, and transmission loss.


    Parameters
    ----------
    executable : Path
        Path to the Bellhop executable file.
    bellhop_dir : Path
        Directory where Bellhop input and output files will be stored.
    filename : str
        Base name for the Bellhop files (without extension).
    calc : str or list of str
        Specifies the type of calculation
        (e.g., 'I' for incoherent, 'C' for coherent, etc.).
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
        A NumPy array representing the vertical depth (z) coordinates
        of the propagation path.
    param_seabed : pd.Series
        A Pandas Series containing seabed parameters.

    Returns
    -------
    None
        This function does not return anything but generates
        output files from the Bellhop model.
        Generates output files :
          'I' option : generates a .shd file, containing the pressure field.
               The data is extracted to calculate and plot the transmission loss diagram
          'E' option : generates a .ray file, containing the information about eigenrays
               (rays that connect the source and the receiver).
          'A' option : generates a .arr file containing the information about all
                the rays emitted (arrival delay, amplitude...)

    Notes
    -----
    - The Bellhop model is used to compute acoustic transmission loss based
     on the environment.
    - The function writes necessary input files and executes Bellhop to perform
     the calculation.
    - The '.bty' bathymetry file is created based on the provided seabed profile.

    """
    # Ensure the Bellhop directory exists
    bellhop_dir.mkdir(parents=True, exist_ok=True)

    # Compute central frequency of the sound source
    f_cen = int((source["f_min"] + source["f_max"]) / 2)

    # Define the number of bathymetry points (ensuring at least 15 points)
    nb_p = max(15, int(10 * station["distance"]))  # Sampling at 100m intervals

    # Compute the number of rays based on distance (one ray per 25 meters)
    nb_ray = int((dist[-1] * 1000) // 25)

    # Ensure 'calc' is a list (even if a single string is provided)
    if isinstance(calc, str):
        calc = [calc]

    # Iterate over calculation types (e.g., incoherent, coherent, etc.)
    for c in calc:

        # Generate Bellhop environment file
        write_env_file(
            bellhop_dir, f"{filename}{c}", f_cen, source["depth"], station["depth"],
            sound_speed, z_transect, station["distance"], nb_ray, z_max,
            param_seabed, source["opening_angle"], f"{c}",
        )

        # Write bathymetry (.bty) file
        with Path.open(bellhop_dir / f"{filename}{c}.bty", "w") as fid:
            fid.write("L\n")
            fid.write(f"{nb_p}\n")
            fid.writelines(f"{np.squeeze(dist[i])} {np.squeeze(zb[i])} /\n"
                           for i in range(len(zb)))

        # Run Bellhop model using system command
        os.system(f"{executable} {bellhop_dir / (filename + c)}")  # noqa: S605



# TODO(Question MD): pas d'explications/commentaires sur cette fonction
def impulse_response(file : Path, source : dict[str, float], station : dict[str, float],  # noqa: PLR0913, PLR0915
                     param_water : dict[str, float], param_seabed : dict[str, float],
                     param_env : float) -> tuple[np.ndarray[np.ndarray[np.float64],
                     np.complex128],np.ndarray[np.float64], np.ndarray[np.float64],
                     np.ndarray[np.complex128]]:
    """Reconstruct the received signal.

    This function reconstructs the received signal from environmental and signal data,
    written in the file produced by bellhop (.arr file)

    Parameters
    ----------
    file : Path
        The path to the .arr file containing the inforation about rays
        (time arrival, amplitude...)
    source : dict
        Dictionary containing information about the signal emitted by the source
    station : dict
        Dictionary containing information about the receivers (stations)
    param_water : dict
        Dictionary containing information about water parameters
    param_seabed : dict
        Dictionary containing seabed parameters
    param_env : float
        Wind speed

    Returns
    -------
    s_emis : ndarray [float]
        Emitted signal
    ri_t : ndarray [complex]
        Received signal, time response
    t_ri : ndarray [float]
        Time scale
    f_ri : ndarray [float]
        Frequency scale
    ri_f : ndarray [complex]
        Frequency response

    """
    [arr, _] = read_arrivals_asc(file)

    d = 1000 * station["distance"] # distance between the source and receiver

    fmin = source["f_min"] # minimum frequency
    fmax = source["f_max"] #maximum frequency
    samp_freq = source["samp_freq"] # sampling frequency
    t0 = source["t0"] # duration of the transmitted signal
    ponder = source["ponder"] #frequency weighting type
    opening_angle = source["opening_angle"]

    # environmental data
    salinity = param_water["salinity"]
    temperature = param_water["temperature"]
    ph = param_water["pH"]

    w = param_env

    ta = d / np.cos(np.pi/180*opening_angle) / 1500 # maximal duration of the wave
                                                    # path with a certain opening angle
    nbp = 2 ** find_pow2((ta + 2 * t0) * samp_freq) # number of points
    freq = np.arange(0, samp_freq, samp_freq/nbp) # frequency axis
    temp = np.arange(0, nbp/samp_freq, 1/samp_freq) # time axis

    # construction of the emitted signal
    # based on Blackman methode for a porpoise click
    if source["type"] == "click_M":
        f0 = (fmin + fmax) / 2
        u = np.where(temp <= t0)
        se = np.zeros(len(temp))
        se[u] = np.blackman(len(u)) * np.sin(2 * np.pi * f0 * temp[u])
        se = fft(se) / (nbp/2)
        z0 = se
    # and on a chirp signal if the signal is a dolphin whistle
    elif source["type"] == "whistle_D":
        u = np.where(temp <= t0)
        z0 = chirp(temp[u], fmin, t0, fmax,"li")
        se = np.concatenate((z0, np.zeros(len(temp)-len(z0))), axis=0)
        se = fft(se) / (nbp/2)

    t = np.squeeze(arr["delay"].real)  # arrival time
    tet = np.squeeze(1/2*(abs(arr["SrcAngle"])+abs(arr["RcvrAngle"]))) #average
                                                # ray angle at the interferences
    dis = 1500 * t  # distance traveled (m)
    ns = np.squeeze(arr["NumTopBnc"]) # number of top reflections
    nb = np.squeeze(arr["NumBotBnc"]) # number of bottom reflexions

    # loop on frequency
    n1 = np.where(freq>=fmin)[0][0]
    n2 = np.where(freq>=fmax)[0][0]
    n_win = n2-n1+1
    tmin0 = 0
    inc = 0
    y = np.zeros((len(freq),len(t))) + 1j*np.zeros((len(freq),len(t)))


    if ponder == 1:
        se[n1:n2] = np.hamming(n_win-1) * se[n1:n2]

        # construction of the frequencial response
    for ni in np.arange(n1, n2+1):
        inc = inc + 1
        fk = freq[ni] / 1000
        rs = -ref_coeff_surf(tet, w, fk) # surface reflexion
        rb = ref_coeff_bot(tet, [1500, 1, 0], param_seabed, freq[ni]) # bottom reflexion
        atv = atten_fg(fk, salinity, temperature,10, ph) * dis / 1000 # attenuation (dB)
        atten = np.exp(-atv*np.log(10)/20) # attenuation (Np)
        y[ni,:] = (se[ni]*atten*(rs**ns)*(rb**nb)*
                   np.exp(-1j*2*np.pi*freq[ni]*(t-tmin0))) # *se[ni]*

    spec = y.sum(axis=1)
    spec[0] = 0
    spec[n2] = 0
    mil = round(len(freq)/2)
    toto = spec[:mil].conj()
    spec[mil:] = np.flipud(toto)

    ri_t = ifft(spec) #/(nbp/2)

    t_ri = tmin0 + temp # time axis
    f_ri = freq # frequency axis
    ri_f = spec # received spectr
    s_emis = z0 # emitted signal

    return s_emis, ri_t, t_ri, f_ri, ri_f
