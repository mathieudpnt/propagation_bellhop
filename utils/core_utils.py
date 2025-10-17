#!/usr/bin/env python3

"""Created on Tue Jun  6 07:45:28 2023.

- [R,Teta] = Coef_Rb(Para_1,Para_2,Fr)  / reflexion fond (fluide-fluide)
- R = Coef_Rbot2(Teta,Para_1,Para_2,Fr) / reflexion fond (fluide-fluide)
- r = Coef_Surf(t, v, f)        / reflexion surf (Beckmann)
- Atv = Atten_FG(f,s,t,z,ph)    / attenuation de Francois&Garrison
- La1,Lo1 = Proj(La0,Lo0,R0,Tet0) / calcul de P1 (La1,Lo1)
    situé à R0,Tet0, de P0 (Lo0,La0)

@author: xdemoulin
"""
from __future__ import annotations

import math

import numpy as np
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d


def depth_to_pressure(z:float, lat:float) -> float:
    """Convert depth (m) to pressure (kPa) (Leroy & Parthiot, 1998)."""
    g = 9.7803 * (1 + 5.3e-3 * math.sin(math.radians(lat))**2)

    kz = (g - 2e-5 * z) / (9.80612 - 2e-5 * z)

    hz = (1.00818e-2 * z
          + 2.465e-8 * z**2
          - 1.25e-13 * z**3
          + 2.8e-19 * z**4)

    return hz * kz * 1000

def compute_sound_speed(s:float, t:float, d:float, equation:str,
                        lat:float) -> float:
    """Compute the speed of sound in seawater using different empirical equations.

    Parameters
    ----------
    s : float
        Salinity in ppt.
    t : float
        Temperature in degrees Celsius.
    d : float
        Depth in meters.
    lat: float | None
        Latitude in degrees.
    equation : str
        The sound speed equation to use. Options are:
            - "mackenzie" : Mackenzie (1981)
            - "del_grosso" : Del Grosso (1974)
            - "chen" : Chen and Millero (1977)
        For further details on the sound speed equations please read the following page:
        https://resource.npl.co.uk/acoustics/techguides/soundseawater/underlying-phys.html

    Returns
    -------
    float
        Speed of sound in seawater (m/s).

    """
    if equation.lower() == "mackenzie":
        return (1448.96
         + 4.591 * t
         - 5.304e-2 * (t ** 2)
         + 2.374e-4 * (t ** 3)
         + 1.340 * (s - 35)
         + 1.630e-2 * d
         + 1.675e-7 * (d ** 2)
         - 1.025e-2 * t * (s - 35)
         - 7.139e-13 * t * (d ** 3)
                )

    if equation.lower() == "del_grosso":

        if not lat:
            msg = "`lat` must be provided."
            raise ValueError(msg)

        p = depth_to_pressure(d, lat) * 0.010197162129779  # convertion
                                                           # from dBar to kg.cm-2

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
        delta_c_t = c_t1 * t + c_t2 * t**2 + c_t3 * t**3
        delta_c_s = c_s1 * s + c_s2 * s**2
        delta_c_p = c_p1 * p + c_p2 * p**2 + c_p3 * p**3
        delta_c_s_t_p = (
            c_t_p * t * p
            + c_t3_p * t**3 * p
            + c_t_p2 * t * p**2
            + c_t2_p2 * t**2 * p**2
            + c_t_p3 * t * p**3
            + c_s_t * s * t
            + c_s_t2 * s * t**2
            + c_s_t_p * s * t * p
            + c_s2_t_p * s**2 * t * p
            + c_s2_p2 * s**2 * p**2
        )

        # Calculate the total speed of sound
        return c000 + delta_c_t + delta_c_s + delta_c_p + delta_c_s_t_p

    if equation.lower() == "chen":

        if not lat:
            msg = "`lat` must be provided."
            raise ValueError(msg)

        p = depth_to_pressure(d, lat) / 100

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
            (c_00 + c_01 * t + c_02 * t**2 + c_03 * t**3 + c_04 * t**4 + c_05 * t**5)
            + (c_10 + c_11 * t + c_12 * t**2 + c_13 * t**3 + c_14 * t**4) * p
            + (c_20 + c_21 * t + c_22 * t**2 + c_23 * t**3 + c_24 * t**4) * p**2
            + (c_30 + c_31 * t + c_32 * t**2) * p**3
        )
        a = (
            (a_00 + a_01 * t + a_02 * t**2 + a_03 * t**3 + a_04 * t**4)
            + (a_10 + a_11 * t + a_12 * t**2 + a_13 * t**3 + a_14 * t**4) * p
            + (a_20 + a_21 * t + a_22 * t**2 + a_23 * t**3) * p**2
            + (a_30 + a_31 * t + a_32 * t**2) * p**3
        )
        b = b_00 + b_01 * t + (b_10 + b_11 * t) * p
        d = d_00 + d_10 * p

        return cw + a * s + b * s**1.5 + d * s**2

    error=f"Unrecognized equation: {equation.lower()}"
    raise ValueError(error)


def find_nearest(array:np.array, value:float) -> float:
    """Find index where array is closest to value.

    Parameters
    ----------
    array : np.array
            array containing the studied data
    value : float
            value to be the nearest

    """
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()

def ref_coeff_bot(teta : float, para_1 : list[float, float, float], para_2 : list[float,
                    float, float], freq : float) -> float:
    """"Calculate the reflectcion coefficient at the bottom.

    The interface is considered fluid-fluid at the bottom of the environment

    Parameters
    ----------
        teta : float
            Angle d'incidence de l'onde par rapport à la normale
        para_1 : list[float, float, float]
            List of the parameters of the seabed (C1, rho1, At1)
        para_2 : list[float, float, float]
            List of parameters (C2, rho2, At2)
        freq : float
            Frequency (Hz)

    Returns
    -------
    Reflection coefficient for a fluid-fluid interface

    """
    c1 = para_1[0] # soundspeed in the seabed
    rho1 = para_1[1] # density of the seabed
    at1 = para_1[2] # attenuation of the sound in the seabed (dB/lambda)

    c2 = para_2.iloc[0] # soundspeed in water
    rho2 = para_2.iloc[1] # density of water
    at2 = para_2.iloc[2] # attenuation of the sound in water (dB/lambda)

    w = 2 * np.pi * freq # omega
    atp1 = (at1 * freq) / (8.686 * c1) # conversion de l'attenuation en Np
    c1b =	((w**2/c1)-1j*atp1*w)/((w/c1)**2+atp1**2) # vitesse de propa complexe
    atp2 = (at2 * freq) / (8.686 * c2) # conversion de l'attenuation en Np
    c2b =	((w**2/c2)-1j*atp2*w)/((w/c2)**2+atp2**2) # vitesse de propa complexe

    sint1 = np.sin(np.pi / 180 * teta)
    cost1 = np.cos(np.pi / 180 * teta)
    sint2 = np.sqrt(1-(c2b*cost1/c1b)**2) # calcul de l'angle de transmission dans
                                            # le sol (loi de Snell)

    z1=rho1*c1b/sint1 # calcul de l'impédance acoustique
    z2=(rho2*c2b)/sint2 # calcul de l'impédance acoustique

    return (z2-z1)/(z2+z1)



def ref_coeff_surf(teta : float, wind_speed : float, freq : float) -> float:
    """Calculate the surface reflection coefficient.

    Calculations based on Beckmann equation

    Parameters
    ----------
        teta : float
            angle in degree
        wind_speed : float
            wind speed in knots
        freq : float
            frequency in kHz

    Returns
    -------
        Reflection coefficient

    """
    term 	= np.exp(-0.0381 * teta ** 2 / (3 + 2.6 * wind_speed)) / np.sqrt(5 *
                                                        np.pi / (3 + 2.6 * wind_speed))
    k 	= np.minimum(0.707, (np.sin(np.pi / 180 * teta) + 0.1 * term))
    return ((0.3 + (0.7 / (1 + (0.0182 * wind_speed ** 2 * freq / 40) ** 2)))
            * np.sqrt(1 - k))


def atten_fg(f :float, s : float, t : float, z : float, ph : float) -> float:
    """Calculate the attenuation coefficient.

    Calculations based on Francois & Garrison

    Parameters
    ----------
        f : float
            frequency in kHz
        s : float
            salinity in ppm (35 ppm typically)
        t : float
            temperature in °C
        z : float
            immersion in m
        ph : float
            pH (8 typically). Very sensitive at low frequency

    Returns
    -------
        attenuation coefficient : float
            in dB/km

    """
    c = 1412+3.21*t+1.19*s+0.0167*z

    # Bo(OH)3
    a1 = 8.68/c*10**(0.78*ph-5)
    p1 = 1
    f1 = 2.8*np.sqrt(s/35)*10**(4-(1245/(273+t)))

    # Mg(SO)4
    a2 = 21.44*s/c*(1+0.025*t)
    p2 = 1-1.37e-4*z+6.2e-9*z**2
    f2 = (8.17*10**(8-(1990/(273+t))))/(1+0.0018*(s-35))

    # viscosity
    p3 = 1-3.83e-5*z+4.9e-10*z**2
    if t<=20:  # noqa: PLR2004
        a3 = 4.937e-4-2.59e-5*t+9.11e-7*t**2-1.5e-8*t**3
    else:
        a3 = 3.964e-4-1.146e-5*t+1.45e-7*t**2-6.5e-10*t**3

    return a1*p1*(f1*f**2)/(f**2+f1**2) + a2*p2*(f2*f**2)/(f**2+f2**2) + a3*p3*f**2


def find_pow2(x : float) -> int:
    """Nearest power of 2 by higher value.

    Parameters
    ----------
    x : float
        value to find nearest power of 2

    Returns
    -------
    n : int
        nearest power of 2

    """
    n=0
    p=1
    while p<x:
       p = p*2
       n = n+1
    return n


def synthetic_signal(fmin : float, fmax : float, deltaf : float, fe : float) -> float:
    """synthetic_signal.

    Parameters
    ----------
    fmin : float
    fmax : float
    deltaf : float
    fe : float

    Returns
    -------
    temp : float
    signal : float
    freq : float
    z : float

    """
    freq	= np.arange(deltaf,fe,deltaf) # generates an array of values
                                          # from deltaf to fe with a pas of deltaf

    if deltaf==0.1:  # noqa: PLR2004
        freq = np.around(freq,1)
    elif deltaf==0.01:  # noqa: PLR2004
        freq = np.around(freq,2)
    temp	= np.arange(1/fe,1/deltaf,1/fe)
    n1 	= list(freq).index(fmin)
    n2 	= list(freq).index(fmax)
    n3 =  list(freq).index(fe/2)

    y = np.zeros(len(freq))
    y0 = interp1d([n1, 2*n1, 4*n1, n2],[1, 2, 4, 2],kind="linear")(np.arange(n1,n2,1))
    y[n1:n2] = y0
    y[n3+2:-1] = np.flip(np.conj(y[2:n3]))

    bb = np.random.normal(0, 0.1, len(freq)) # val moy et std
    sbb = fft(bb)
    b = np.sum(sbb*np.conj(sbb))

    z = 1./np.sqrt(b)*(np.sqrt(y)*sbb)
    signal = np.sqrt(len(freq)/2)*np.real(ifft(z))

    return temp,signal,freq,z

