
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import _io
    from pathlib import Path

    from pandas import Series


def depth_to_pressure(z: float, lat: float) -> float:
    """Convert depth (m) to pressure (kPa) (Leroy & Parthiot, 1998).

    Parameters
    ----------
    z : float
        depth (m) to convert
    lat : float
        latitude of the point of depth z

    Returns
    -------
    float : pressure associated with depth (kPa)

    """
    g = 9.7803 * (1 + 5.3e-3 * math.sin(math.radians(lat))**2)

    kz = (g - 2e-5 * z) / (9.80612 - 2e-5 * z)

    hz = (1.00818e-2 * z
          + 2.465e-8 * z**2
          - 1.25e-13 * z**3
          + 2.8e-19 * z**4)

    return hz * kz * 1000


def compute_sound_speed(
        salinity: float,
        temperature: float,
        depth: float,
        equation: str,
        lat: float,
) -> float:
    """Compute the speed of sound in seawater using different empirical equations.

    Parameters
    ----------
    salinity : float
        Salinity in ppt.
    temperature : float
        Temperature in degrees Celsius.
    depth : float
        Depth in meters.
    equation : str
        The sound speed equation to use. Options are:
            - "mackenzie" : Mackenzie (1981)
            - "del_grosso" : Del Grosso (1974)
            - "chen" : Chen and Millero (1977)
        For further details on the sound speed equations please read the following page:
        https://resource.npl.co.uk/acoustics/techguides/soundseawater/underlying-phys.html
    lat: float | None
        Latitude in degrees.

    Returns
    -------
    float
        Speed of sound in seawater (m/s).

    """
    if equation.lower() == "mackenzie":
        return (1448.96
                + 4.591 * temperature
                - 5.304e-2 * (temperature ** 2)
                + 2.374e-4 * (temperature ** 3)
                + 1.340 * (salinity - 35)
                + 1.630e-2 * depth
                + 1.675e-7 * (depth ** 2)
                - 1.025e-2 * temperature * (salinity - 35)
                - 7.139e-13 * temperature * (depth ** 3)
                )

    if equation.lower() == "del_grosso":

        if not lat:
            msg = "`lat` must be provided."
            raise ValueError(msg)

        p = depth_to_pressure(depth, lat) * 0.010197162129779  # from dBar to kg.cm-2

        # Coefficients
        c000 = 1402.392

        c_t1 = 0.5012285E1
        c_t2 = -0.551184E-1
        c_t3 = 0.221649E-3

        c_s1 = 0.1329530E1
        c_s2 = 0.1288598E-3

        c_p1 = 0.1560592
        c_p2 = 0.2449993E-4
        c_p3 = -0.8833959E-8

        c_s_t = -0.1275936E-1
        c_t_p = 0.6353509E-2
        c_t2_p2 = 0.2656174E-7
        c_t_p2 = -0.1593895E-5
        c_t_p3 = 0.5222483E-9
        c_t3_p = -0.4383615E-6
        c_s2_p2 = -0.1616745E-8
        c_s_t2 = 0.9688441E-4
        c_s2_t_p = 0.4857614E-5
        c_s_t_p = -0.3406824E-3

        # Terms calculation
        delta_c_t = (c_t1 * temperature + c_t2 * temperature ** 2 + c_t3 *
                     temperature ** 3)
        delta_c_s = c_s1 * salinity + c_s2 * salinity ** 2
        delta_c_p = c_p1 * p + c_p2 * p**2 + c_p3 * p**3
        delta_c_s_t_p = (
                c_t_p * temperature * p
                + c_t3_p * temperature ** 3 * p
                + c_t_p2 * temperature * p ** 2
                + c_t2_p2 * temperature ** 2 * p ** 2
                + c_t_p3 * temperature * p ** 3
                + c_s_t * salinity * temperature
                + c_s_t2 * salinity * temperature ** 2
                + c_s_t_p * salinity * temperature * p
                + c_s2_t_p * salinity ** 2 * temperature * p
                + c_s2_p2 * salinity ** 2 * p ** 2
        )

        # Calculate the total speed of sound
        return c000 + delta_c_t + delta_c_s + delta_c_p + delta_c_s_t_p

    if equation.lower() == "chen":

        if not lat:
            msg = "`lat` must be provided."
            raise ValueError(msg)

        p = depth_to_pressure(depth, lat) / 100

        # Coefficients
        c_00 = 1402.388
        c_01 = 5.03830
        c_02 = -5.81090e-2
        c_03 = 3.3432e-4
        c_04 = -1.47797e-6
        c_05 = 3.1419e-9
        c_10 = 0.153563
        c_11 = 6.8999e-4
        c_12 = -8.1829e-6
        c_13 = 1.3632e-7
        c_14 = -6.1260e-10
        c_20 = 3.1260e-5
        c_21 = -1.7111e-6
        c_22 = 2.5986e-8
        c_23 = -2.5353e-10
        c_24 = 1.0415e-12
        c_30 = -9.7729e-9
        c_31 = 3.8513e-10
        c_32 = -2.3654e-12

        a_00 = 1.389
        a_01 = -1.262e-2
        a_02 = 7.166e-5
        a_03 = 2.008e-6
        a_04 = -3.21e-8
        a_10 = 9.4742e-5
        a_11 = -1.2583e-5
        a_12 = -6.4928e-8
        a_13 = 1.0515e-8
        a_14 = -2.0142e-10
        a_20 = -3.9064e-7
        a_21 = 9.1061e-9
        a_22 = -1.6009e-10
        a_23 = 7.994e-12
        a_30 = 1.100e-10
        a_31 = 6.651e-12
        a_32 = -3.391e-13

        b_00 = -1.922e-2
        b_01 = -4.42e-5
        b_10 = 7.3637e-5
        b_11 = 1.7950e-7

        d_00 = 1.727e-3
        d_10 = -7.9836e-6

        # Terms calculation
        cw = (
                (c_00 + c_01 * temperature + c_02 * temperature ** 2 + c_03 *
                 temperature ** 3 + c_04 * temperature ** 4 + c_05 * temperature ** 5)
                + (c_10 + c_11 * temperature + c_12 * temperature ** 2 + c_13 *
                   temperature ** 3 + c_14 * temperature ** 4) * p
                + (c_20 + c_21 * temperature + c_22 * temperature ** 2 + c_23 *
                   temperature ** 3 + c_24 * temperature ** 4) * p ** 2
                + (c_30 + c_31 * temperature + c_32 * temperature ** 2) * p ** 3
        )
        a = (
                (a_00 + a_01 * temperature + a_02 * temperature ** 2 + a_03 *
                 temperature ** 3 + a_04 * temperature ** 4)
                + (a_10 + a_11 * temperature + a_12 * temperature ** 2 + a_13 *
                   temperature ** 3 + a_14 * temperature ** 4) * p
                + (a_20 + a_21 * temperature + a_22 * temperature ** 2 + a_23 *
                   temperature ** 3) * p ** 2
                + (a_30 + a_31 * temperature + a_32 * temperature ** 2) * p ** 3
        )
        b = b_00 + b_01 * temperature + (b_10 + b_11 * temperature) * p
        depth = d_00 + d_10 * p

        return cw + a * salinity + b * salinity**1.5 + depth * salinity**2

    error = f"Unrecognized equation: {equation.lower()}"
    raise ValueError(error)


def find_nearest(array: np.ndarray, value: float) -> int:
    """Find index where array is closest to value.

    Parameters
    ----------
    array : np.array
            array containing the studied data
    value : float
            value to be the nearest

    """
    array = np.asarray(array)
    return int(np.abs(array - value).argmin())


def bottom_reflection_coefficient(
    theta: float,
    para_1: Series,
    para_2: Series,
    freq: float,
) -> float:
    """Calculate the reflection coefficient at the bottom.

    The interface is considered fluid-fluid at the bottom of the environment.

    Parameters
    ----------
    theta : float
        Angle of incidence of the wave with respect to the normal (degrees).
    para_1 : Series
        Parameters of the water [sound_speed, density, attenuation].
    para_2 : Series
        Parameters of the seabed [sound_speed, density, attenuation].
    freq : float
        Frequency in Hz.

    Returns
    -------
    float
        Reflection coefficient for a fluid-fluid interface.

    """
    c1 = para_1[0]  # soundspeed in the water
    rho1 = para_1[1]  # density of the water
    at1 = para_1[2]  # attenuation of the sound in the water (dB/lambda)

    c2 = para_2.iloc[0]  # soundspeed in seabed
    rho2 = para_2.iloc[1]  # density of seabed
    at2 = para_2.iloc[2]  # attenuation of the sound in seabed (dB/lambda)

    w = 2 * np.pi * freq  # omega
    atp1 = (at1 * freq) / (8.686 * c1)  # attenuation conversion to Np
    c1b = ((w**2 / c1) - 1j * atp1 * w) / (
        (w / c1) ** 2 + atp1**2
    )  # complex sound speed

    atp2 = (at2 * freq) / (8.686 * c2)  # attenuation conversion to Np
    c2b = ((w**2 / c2) - 1j * atp2 * w) / (
        (w / c2) ** 2 + atp2**2
    )  # complex sound speed

    sint1 = np.sin(np.pi / 180 * theta)
    cost1 = np.cos(np.pi / 180 * theta)
    sint2 = np.sqrt(1 - (c2b * cost1 / c1b)**2)
    # Calculation of the transmission angle in the ground (Snell's law)

    z1 = rho1 * c1b / sint1  # acoustic impedance calculation
    z2 = (rho2 * c2b) / sint2  # acoustic impedance calculation

    return (z2 - z1) / (z2 + z1)


def surface_reflection_coefficient(
    theta: float,
    wind_speed: float,
    freq: float,
) -> float:
    """Calculate the surface reflection coefficient.

    The calculation is based on the Beckmann equation.

    Parameters
    ----------
    theta : float
        Angle of incidence in degrees.
    wind_speed : float
        Wind speed in knots.
    freq : float
        Frequency in kHz.

    Returns
    -------
    float
        Reflection coefficient at the sea surface.

    """
    # Empirical term from Beckmann equation
    term = np.exp(
        -0.0381 * theta**2 / (3 + 2.6 * wind_speed),
    ) / np.sqrt(
        5 * np.pi / (3 + 2.6 * wind_speed),
    )

    # Correction factor
    k = np.minimum(
        0.707,
        np.sin(np.deg2rad(theta)) + 0.1 * term,
    )

    # Reflection coefficient
    return (
        0.3 + (0.7 / (1 + (0.0182 * wind_speed**2 * freq / 40) ** 2))
    ) * np.sqrt(1 - k)


def atten_fg(f: float, s: float, t: float, z: float, ph: float) -> float:
    """Calculate the attenuation coefficient.

    Calculations based on Francois & Garrison

    Parameters
    ----------
    f : float
        Frequency in kHz
    s : float
        Salinity in ppm (35 ppm typically)
    t : float
        Temperature in °C
    z : float
        Immersion in m
    ph : float
        ph (8 typically). Very sensitive at low frequency

    Returns
    -------
    Attenuation coefficient in dB/km : float

    """
    c = 1412 + 3.21 * t + 1.19 * s + 0.0167 * z

    # Bo(OH)3
    a1 = 8.68 / c * 10**(0.78 * ph - 5)
    p1 = 1
    f1 = 2.8 * np.sqrt(s / 35) * 10**(4 - (1245 / (273 + t)))

    # Mg(SO)4
    a2 = 21.44 * s / c * (1 + 0.025 * t)
    p2 = 1 - 1.37e-4 * z + 6.2e-9 * z**2
    f2 = (8.17 * 10**(8 - (1990 / (273 + t)))) / (1 + 0.0018 * (s - 35))

    # viscosity
    p3 = 1 - 3.83e-5 * z + 4.9e-10 * z**2
    if t <= 20:  # noqa: PLR2004
        a3 = 4.937e-4 - 2.59e-5 * t + 9.11e-7 * t**2 - 1.5e-8 * t**3
    else:
        a3 = 3.964e-4 - 1.146e-5 * t + 1.45e-7 * t**2 - 6.5e-10 * t**3

    return (a1 * p1 * (f1 * f**2) / (f**2 + f1**2) + a2 * p2 * (f2 * f**2) /
            (f**2 + f2**2) + a3 * p3 * f**2)


def find_pow2(x: float) -> int:
    """Nearest power of 2 by higher value.

    Parameters
    ----------
    x : float
        Value to find nearest power of 2

    Returns
    -------
    n : int
        Nearest power of 2

    """
    n = 0
    p = 1
    while p < x:
        p *= 2
        n += 1
    return n


def readline_1(fid: _io.TextIOWrapper, nb: int) -> float:
    """Read element nb of a line."""
    return float(fid.readline().split()[nb])


def zeros(size: int, flag: int) -> np.ndarray:
    """Create an array of zeros, could be real (0) or complex (1)."""
    if flag == 1:  # if complex
        return np.zeros(size) + 1j * np.zeros(size)
    return np.zeros(size)


def date_to_number(m: int, d: int) -> int:
    """Convert date to number in the year."""
    return (m - 1) * 30 + d


def check_file_exist(file: Path) -> None:
    """Check if the file exists."""
    if not file.exists():
        msg = f"{file} does not exist"
        raise FileNotFoundError(msg)


def check_suffix(file: Path, suffix: str) -> None:
    """Check if the suffix of the file is the one expected."""
    if file.suffix != suffix:
        msg = f"{file} is not a {suffix} file"
        raise ValueError(msg)


def check_empty_file(file: Path) -> None:
    """Check if the file is empty."""
    content = file.read_text(encoding="utf-8")
    if not content:
        msg = f"{file} is empty"
        raise ValueError(msg)


def check_elem_num(line: str, nb: int) -> bool:
    """Check number of element in a string."""
    line = line.split()
    return len(line) == nb


def check_len_list(list_: list, nb: int) -> bool:
    """Check if length of a list."""
    return len(list_) == nb
