"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
Created on Tue Jun  6 07:45:28 2023

- [R,Teta] = Coef_Rb(Para_1,Para_2,Fr)  / reflexion fond (fluide-fluide)
- R = Coef_Rbot2(Teta,Para_1,Para_2,Fr) / reflexion fond (fluide-fluide)
- r = Coef_Surf(t, v, f)        / reflexion surf (Beckmann)
- Atv = Atten_FG(f,s,t,z,ph)    / attenuation de Francois&Garrison
- La1,Lo1 = Proj(La0,Lo0,R0,Tet0) / calcul de P1 (La1,Lo1) situé à R0,Tet0, de P0 (Lo0,La0)

@author: xdemoulin
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.fft import ifft,fft


def compute_soundspeed(salinity:float, temperature:float, depth:float, equation:str):
    """
    Computes the speed of sound in seawater using different empirical equations.

    Parameters
    ----------
    salinity : float
               Salinity in ppt (practical salinity units).
    temperature : float
                  Temperature in degrees Celsius.
    depth : float
            Depth in meters (for Mackenzie) or Pressure in dBar (for other equations).
    equation : str
               The sound speed equation to use. Options are:
               - "mackenzie" : Mackenzie (1981)
               - "del_grosso" : Del Grosso (1974)
               - "chen" : Chen and Millero (1977)

    Returns
    -------
    float
        Speed of sound in seawater (m/s).
    """
    if equation.lower() == "mackenzie":
        # Mackenzie equation coefficients
        sound_speed = (1402.392 + 4.591 * temperature - 5.304e-2 * temperature ** 2 + 2.374e-4 * temperature ** 3 +
               1.340 * (salinity - 35) + 1.630e-2 * depth + 1.675e-7 * depth ** 2 -
               1.025e-2 * temperature * (salinity - 35) - 7.139e-13 * temperature * depth ** 3)

    elif equation.lower() == "del_grosso":
        # Compute pressure in kg/cm²
        XX = np.sin(np.radians(45))  # Assume latitude = 45° for gravity calculation
        GR = 9.780318 * (1 + (5.2788e-3 + 2.36e-5 * XX) * XX) + 1.092e-6 * depth
        P = depth / GR

        # Del Grosso equation
        C000 = 1402.392
        DCT = (5.01109398873 - (5.50946843172e-1 - 2.21535969240e-3 * temperature) * temperature) * temperature
        DCS = (1.32952290781 + 1.28955756844e-3 * salinity) * salinity
        DCP = (0.156059257041 + (2.44998688441e-4 - 8.83392332513e-8 * P) * P) * P
        DCSTP = (-1.27562783426e-2 * temperature * salinity + 6.35191613389e-3 * temperature * P -
                 4.38031096213e-6 * temperature ** 3 * P - 1.61374495909e-8 * salinity ** 2 * P ** 2 +
                 9.68403156410e-4 * temperature ** 2 * salinity - 3.40597039004e-3 * temperature * salinity * P)

        sound_speed = C000 + DCT + DCS + DCP + DCSTP

    elif equation.lower() == "chen":
        # Chen and Millero equation
        P = depth / 10.0  # Convert pressure to bars
        SR = np.sqrt(np.abs(salinity))

        depth = 1.727e-3 - 7.9836e-6 * P
        B1 = 7.3637e-5 + 1.7945e-7 * temperature
        B0 = -1.922e-2 - 4.42e-5 * temperature
        B = B0 + B1 * P

        A3 = (-3.389e-13 * temperature + 6.649e-12) * temperature + 1.100e-10
        A2 = ((7.988e-12 * temperature - 1.6002e-10) * temperature + 9.1041e-9) * temperature - 3.9064e-7
        A1 = (((-2.0122e-10 * temperature + 1.0507e-8) * temperature - 6.4885e-8) * temperature - 1.2580e-5) * temperature + 9.4742e-5
        A0 = (((-3.21e-8 * temperature + 2.006e-6) * temperature + 7.164e-5) * temperature - 1.262e-2) * temperature + 1.389
        A = ((A3 * P + A2) * P + A1) * P + A0

        C3 = (-2.3643e-12 * temperature + 3.8504e-10) * temperature - 9.7729e-9
        C2 = (((1.0405e-12 * temperature - 2.5335e-10) * temperature + 2.5974e-8) * temperature - 1.7107e-6) * temperature + 3.1260e-5
        C1 = (((-6.1185e-10 * temperature + 1.3621e-7) * temperature - 8.1788e-6) * temperature + 6.8982e-4) * temperature + 0.153563
        C0 = ((((3.1464e-9 * temperature - 1.47800e-6) * temperature + 3.3420e-4) * temperature - 5.80852e-2) * temperature + 5.03711) * temperature + 1402.388
        C = ((C3 * P + C2) * P + C1) * P + C0

        sound_speed = C + (A + B * SR + depth * salinity) * salinity

    else:
        raise ValueError(f"Unrecognized equation: {equation.lower()}")

    return sound_speed


def find_nearest(array, value):
    """
    Finds index where array is closest to value

    Parameters
    ----------
    array
    value
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def Coef_Rbot(Para_1,Para_2,Fr):
      # Coef Ref fluide-fluide
      C1 = Para_1[0] ; rho1 = Para_1[1]; At1 = Para_1[2]
      C2 = Para_2[0] ; rho2 = Para_2[1]; At2 = Para_2[2]
      Teta = np.linspace(0,90,181)
      w = 2*np.pi*Fr
      atp1 =	(At1*Fr)/(8.686*C1)
      c1b =	((w**2/C1)-1j*atp1*w)/((w/C1)**2+atp1**2)
      atp2 =	(At2*Fr)/(8.686*C2)
      c2b =	((w**2/C2)-1j*atp2*w)/((w/C2)**2+atp2**2)
      
      #	calcul de (Z2-Z1)/(Z2+Z1) , Z = rhoi*ci/sin(thetai)
      sint1 = np.sin(np.pi/180*Teta[1:])
      cost1 = np.cos(np.pi/180*Teta[1:])
      sint2 = np.sqrt(1-(c2b*cost1/c1b)**2)
    
      z1=rho1*c1b/sint1
      z2=(rho2*c2b)/sint2
    
      # Sorties
      R = np.ones(181)+ 1j*np.zeros(181)
      R[1:] = (z2-z1)/(z2+z1)
      
      return R, Teta


def Coef_Rbot2(Teta,Para_1,Para_2,Fr):
      # Coef Ref fluide-fluide
      C1 = Para_1[0] ; rho1 = Para_1[1]; At1 = Para_1[2]
      C2 = Para_2.iloc[0] ; rho2 = Para_2.iloc[1]; At2 = Para_2.iloc[2]
      
      w = 2*np.pi*Fr
      atp1 =	(At1*Fr)/(8.686*C1)
      c1b =	((w**2/C1)-1j*atp1*w)/((w/C1)**2+atp1**2)
      atp2 =	(At2*Fr)/(8.686*C2)
      c2b =	((w**2/C2)-1j*atp2*w)/((w/C2)**2+atp2**2)
      
      #	calcul de (Z2-Z1)/(Z2+Z1) , Z = rhoi*ci/sin(thetai)
      sint1 = np.sin(np.pi/180*Teta)
      cost1 = np.cos(np.pi/180*Teta)
      sint2 = np.sqrt(1-(c2b*cost1/c1b)**2)
    
      z1=rho1*c1b/sint1
      z2=(rho2*c2b)/sint2
    
      # Sorties
      R = (z2-z1)/(z2+z1)
      
      return R   


def Coef_Surf(t, v, f):
    #   [R] = Coef_Surf(t, v, f)
    #	Reflexion de surface - formule de bekmann
    #	entres  : t, angle en degres - v, vent en noeuds - f, frequence en kHz
    #   rem     : t OU f peuvent etre des vecteurs

    term 	= np.exp(-0.0381*t**2/(3+2.6*v))/np.sqrt(5*np.pi/(3+2.6*v))
    k 	= np.minimum(0.707,(np.sin(np.pi/180*t)+0.1*term))
    r 	= (0.3+((0.7)/(1+(0.0182*v**2*f/40)**2)))*np.sqrt(1-k)
    return r


def Atten_FG(f,s,t,z,ph):
    # Coef d'atten de Francois & Garrison
    # input 	: f frequency(kHz), s salinity (ppm- 35 typical), t temp(°C), 
    #       z immersion(m), ph (8 typical - very sensitive at low frequency)		
    # output	: coef Atv (dB/km)  
    
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
    if t<=20:
        a3 = 4.937e-4-2.59e-5*t+9.11e-7*t**2-1.5e-8*t**3
    else:
        a3 = 3.964e-4-1.146e-5*t+1.45e-7*t**2-6.5e-10*t**3
    
    Atv = a1*p1*(f1*f**2)/(f**2+f1**2) + a2*p2*(f2*f**2)/(f**2+f2**2) + a3*p3*f**2
    return Atv


def FindPow2(x):
    """
    Puissance de 2 la plus proche par valeur supérieure
    """
    n=0
    p=1
    while(p<x):
       p = p*2
       n = n+1
    N = n
    return N


def SyntheticSignal(fmin,fmax,deltaf,fe):
    # Partie generation du signal synthetique 
    #fmin   = 20; fmax = 100; deltaf = 0.01;	fe 	= 250
    freq	= np.arange(deltaf,fe,deltaf) 
    if deltaf==0.1:
        freq = np.around(freq,1)
    elif deltaf==0.01:
        freq = np.around(freq,2)
    temp	= np.arange(1/fe,1/deltaf,1/fe)
    n1 	= list(freq).index(fmin)
    n2 	= list(freq).index(fmax)
    n3 =  list(freq).index(fe/2)
    
    y = np.zeros(len(freq))
    y0 = interp1d([n1, 2*n1, 4*n1, n2],[1, 2, 4, 2],kind='linear')(np.arange(n1,n2,1))
    y[n1:n2] = y0
    y[n3+2:-1] = np.flip(np.conj(y[2:n3]))
    
    a = np.sum(y*np.conj(y))
    bb = np.random.normal(0, 0.1, len(freq)) # val moy et std
    sbb = fft(bb)
    b = np.sum(sbb*np.conj(sbb))

    z = 1./np.sqrt(b)*(np.sqrt(y)*sbb)
    Signal = np.sqrt(len(freq)/2)*np.real(ifft(z)) # 
    
    return temp,Signal,freq,z

