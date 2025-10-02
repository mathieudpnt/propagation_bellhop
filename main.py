"""
Created on Mon Nov  4 11:16:24 2024 (Version Pour Bellhop)
Le simulateur - version "E" - effectue les opérations suivante:
    - 1) Définir les input : un point source (Lat1,Lon1), un type de signal, une géométrie, un environnement
    - 2) Charger les données environnement : la bathy (lons,lats,elev), la celerité (Cw,Zw)
    - 3) selectionner les seules stations qui se trouvent dans un perimètre R0 donné (R0 lié au signal)
    - 4) Boucle sur chaque couple Source-Station séléctionnée :
        -- extraction de la coupe bathy, du profil
        -- calcul de rayons (Bellhop *3: E-A-I, i;e. Eigen, Arrival time, Incoherent TL)
        -- Serie n°1 de 4 calculs:
            1/4) Calcul et visu du signal recu (en freq. et en temps) **
            2/4) Calcul et visu du Chp de pertes (à freq. centrale))
            3/4) Calcul et visu des rayons propres
            4/4) Calcul et visu du suivi du niveau selon distce et comparaison au bruit (plusieurs orientations)
            == Fig_NomSimu_1.png
        -- Série n°2 de calculs :
            1/3) Visu de la bathymétrie , des 7 stations et de la source
            2/3) Vue polaire des positionss des 7 stations relativement à la position de la source
            3/3) Vue polaire du champ de pertes autour de la source (à Zr donné)
            == Fig_NomSimu_2.png
    ** : le calcul des amplitudes a été réalisé par nos propres algos
        (disper geom, Atten,Coefs Ref ... )  car il semble que Bellhop ne gere pas cela correctement
"""
import numpy as np
import matplotlib.pyplot as plt
import pyproj
import cartopy.crs as ccrs
from pathlib import Path
from pandas import DataFrame, Series
from matplotlib import gridspec
from tqdm import tqdm
from geopy.distance import geodesic

from utils.utils_acoustic_toolbox import read_shd, plotray
from utils.utils_simu import read_croco, read_bathy, extract_bathy, compute_sound_speed, impulse_response, run_bellhop

# %% Simulation parameters

# signal emission position
source = Series({
    'lat': 48.48,
    'lon': -4.9,
    'depth': 5,
    'type': 'whistle_D',
})

# day of year (Y+1) # TODO, QUESTION MT : Comment ça Y+1 ?
day_number = 217+360
# TODO, QUESTION MT: Pourquoi on rentre ces paramètres en dur alors qu'on utilise les profils CROCO après ?
param_water = Series({
    'salinity': 35,
    'temperature': 12,
    'pH': 8,
})

# TODO, QUESTION MD: SOURCE ?
param_seabed = Series({
    'bulk_soundspeed': 1600,  #m/s
    'bulk_density': 1.75,  # kg/m3
    'attenuation': 1.05,  # dB/m ?
})
# TODO, QUESTION MT: Unité ? m/s ?
param_env = 5    # Vitesse du vent - bientot , ajouter bulles, vagues ...

z_max = 250

# TODO, QUESTION MD: à quoi cela correspond ?
Dz = 0.25  # param technique pour CETIROISE

ROOT = Path(r"L:\acoustock\Bioacoustique\PROPA")
root_f = ROOT / Path(r'.\Figures')  # figure path
bellhop_exe = ROOT / Path(r".\acoustic_toolbox\windows-bin-20201102\bellhop.exe")  # Bellhop executable path
root_bh = ROOT / Path(r'.\test')  # Bellhop output path
bathy = ROOT / Path(r'.\Data_Env\Bathy\MNT_FACADE_ATLANTIQUE_HOMONIM_PBMA\DONNEES\MNT_ATL100m_HOMONIM_WGS84_PBMA_ZNEG.asc')  # bathymetry data
ncdf = ROOT / Path(r'.\Data_Env\croco_out2.nc')  # CROCO model data

# %% Environment

# study area
lim_lat = [48.1, 48.7]
lim_lon = [-5.5, -4.5]

# bathymetry extraction
[lat, lon, elev] = read_bathy(file=bathy, lim_lat=lim_lat, lim_lon=lim_lon)

# croco extraction
# TODO, QUESTION MT: Préciser le format de la variable data_ncdf et ce qu'il y a dedans
data_ncdf = read_croco(file=ncdf)

# CETIROISE stations
lon_OBS = -([5,5,4,5,4,4,5] + np.array([13,7,55,12,52,48,21]) / 60 + np.array([51,4,1,0,59,20,3]) / 3600)
lat_OBS = 48 + np.array([31,31,29,23,27,29,27]) / 60 + np.array([11, 6, 6, 1, 19, 53, 3]) / 3600
# TODO, QUESTION MT : Variable pas réutilisée ? + wrong value
depth_OBS = [20] * len(lon_OBS)
stations = DataFrame(
    {
    'label': ['A','B','C','D','E','F','G'],
    'lat': lat_OBS,
    'lon': lon_OBS,
    'depth': depth_OBS,
    }
)


# distances and azimuths between source and stations
# TODO, QUESTION MT : unité distance (km?) et azimuth (rad?)?
geod = pyproj.Geod(ellps='WGS84')
azimuth = np.zeros(len(stations))
distance = np.zeros(len(stations))
az, _, d = geod.inv(stations['lon'], stations['lat'], [source['lon']]*len(stations), [source['lat']]*len(stations))
# TODO, QUESTION MT : Pourquoi -az -90 ?
stations['azimuth'] = np.radians(-az - 90)
stations['distance'] = [distance / 1000 for distance in d]

# %% Propagation simulation

if source['type'] =='whistle_D':
        # TODO, QUESTION MD: source ouv/opening_angle ?
        source['opening_angle'] = 20
        source['source_level'] = 155
        source['ambient_noise'] = 45 # TODO, QUESTION MT: On pourrait rendre cette variable dynamique par rapport aux mesures faites sur les données i situ ?
        source['f_min'] = 8_000
        source['f_max'] = 20_000
        source['fe'] = 92_000
        # TODO, QUESTION MT: Définir calirement les 4 valeurs suivantes
        source['t0'] = 1
        source['ponder'] = 1
        # TODO, QUESTION MD: dir=0 == chemin direct ?
        source['dir'] = 0
        source['r_max'] = 6 #Rayon max pour le calcul de propa, en km ?
# elif TypeS=='Click_D':
#         opening_angle = 20; SL = 170; BA = 30
#         P_Sig = [80000, 130000, 256000, 60e-6, 0, Ouv, 0, SL, BA]
elif source['type'] == 'click_M':
        source['opening_angle'] = 20
        source['source_level'] = 150
        source['ambient_noise'] = 30
        source['f_min'] = 90_000
        source['f_max'] = 110_000
        source['fe'] = 256_000
        # TODO, QUESTION MT: Pourquoi ici t0 = 80e-6 ? C'est la durée du signal ? Et pq ponder = 0 ?
        source['t0'] = 80e-6
        source['ponder'] = 0
        source['dir'] = 0
        source['r_max'] = 6
else:
    raise ValueError(
        f"TypeS '{source['type']}' is not a valid entry"
    )

# %% propa
def main():
    selected_station = stations[stations['distance'] < source['r_max']]

    for _, st in selected_station.iterrows():

        # bathymetric profile
        [zb, dist, z_transect, nb_layer] = extract_bathy(source=source, station=st, lat=lat, lon=lon, elev=elev)

        # sound speed profile
        sound_speed = compute_sound_speed(method="chen", yday=day_number, z=z_transect, croco_data=data_ncdf, lat_ref=source['lat'], lon_ref=source['lon'])
        #
        # TODO, QUESTION MD: je ne retrouve pas les meme profils de vitesse que la figure 3 du rapport
        # for d in [36, 95, 156, 217, 278]:
        #     sound_speed = compute_sound_speed(method="chen", yday=d, z=z_transect, croco_data=data_ncdf,
        #                                       lat_ref=48.25, lon_ref=-5.25)
        #
        #     plt.plot(sound_speed, z_transect, label=f"Day {d}")
        # plt.grid()
        # plt.legend()
        # plt.xlim(1480, 1530)
        # plt.ylim(150,0)
        # plt.xticks(np.arange(1480, 1531, 10))
        # plt.show()

        # Bellhop
        test_name = 'Test_01'

        run_bellhop(executable=bellhop_exe,
                    bellhop_dir=root_bh,
                    filename=test_name,
                    z_max=z_max,
                    source=source,
                    station=st,
                    dist=dist,
                    zb=zb,
                    sound_speed=sound_speed,
                    z_transect=z_transect,
                    param_seabed=param_seabed,
                    calc = ['I', 'E', 'A']
                    )
        # TODO Q MT : Signification I/E/A
        test_I = test_name + 'I'
        test_E = test_name + 'E'
        test_A = test_name + 'A'

        # (1/4) : Impulse response
        pathA = root_bh / f"{test_A}.arr"
        #TODO, QUESTION MD: pas d'explication sur la fonction utilisée ici
        [z0, Ri_T, T_ri, F_ri, Ri_F] = impulse_response(file=pathA.resolve(),
                                                        source=source,
                                                        station=st,
                                                        param_water=param_water,
                                                        param_seabed=param_seabed,
                                                        param_env=param_env
                                                        )

        # Visualisation (1/2)
        fig = plt.figure(figsize=(8, 14))
        gs = gridspec.GridSpec(4, 4, height_ratios=[1, 1, 1, 1])

        ax0_0 = fig.add_subplot(gs[0, 0:2])
        ax0_0.plot(F_ri, abs(Ri_F), 'k')
        ax0_0.set_title('Spectrum')
        ax0_0.set_xlim(0, source['fe'] / 2)
        ax0_0.set_xlabel('Freq (Hz)')
        ax0_0.grid()

        ax0_1 = fig.add_subplot(gs[0, 2:4])
        ax0_1.plot(T_ri, Ri_T.real, 'k')
        # Alternative si on veutr la RI if TypeS=='Whistl_GD': xx = (np.correlate(Ri_T,z0)).real
        transmission_loss = source['source_level'] + 20 * np.log10(max(abs(Ri_T)))
        ax0_1.set_title(f"Waveform")
        ax0_1.set_xlabel('Time (sec)')
        ax0_1.grid()
        # Alternative si on veutr la RI if TypeS=='Whistl_GD': xx = (np.correlate(Ri_T,z0)).real

        # (2/4) : Calcul de TL
        file_shd = root_bh / f"{test_I}.shd"
        pressure, geometry = read_shd(file_shd, 0, 0)

        p = np.squeeze(pressure, axis=(0,1))
        zi = zb[-1] // Dz
        # tl = 20 * np.log10(abs(p) + np.finfo(float).eps)
        tl = 20 * np.log10(abs(p))
        ax1_0 = fig.add_subplot(gs[1, :])
        pc = ax1_0.imshow(tl, extent=[0, st['distance'], -z_max, 0], aspect='auto', cmap='jet', vmin=-100, vmax=-30)
        fig.colorbar(pc, orientation="vertical", label="TL (dB)")
        ax1_0.set_title(f"source: {source['lat']:.2f}°N {source['lon']:.2f}°W - station: {st['label']} - Fr={0.5 * (source['f_max'] + source['f_min']):.0f} Hz")
        ax1_0.set_xlim(0, round(1.05 * 100 * st['distance']) / 100)
        ax1_0.set_ylim(-max(zb), -1)
        ax1_0.grid()
        ax1_0.set_xlabel('Range (km)')
        ax1_0.set_ylabel('Depth (m)')

        # (3/4) : Rayons propres / celerité
        ax2_0 = fig.add_subplot(gs[2, 0:3])
        file_ray = root_bh / f"{test_E}.ray"
        plotray(file_ray)
        ax2_0.set_title(f"Eigenrays - Zs={source['depth']}m / Zr={st['depth']}m")
        ax2_0.set_xlabel('Distance (m)')
        ax2_0.set_ylabel('Depth (m)')
        ax2_0.grid()

        ax2_1 = fig.add_subplot(gs[2, 3:4])
        ax2_1.plot(sound_speed, -np.array(z_transect), 'k')
        ax2_1.set_title(f"Speed of sound\nday {day_number}")
        ax2_1.set_xlabel('celerity (m/s)')
        ax2_1.grid()

        # (4/4) : Calcul du SNR (avec directivité)
        if source['type']=='whistle_D':
            Dir = [0,1.14,5.6,13.37,8.8,9.3,9.9]
        elif source['type']=='click_M':
            Dir = [0,10.7,14.7,17.4,18.25,18.8,20.5]

        ax3 = fig.add_subplot(gs[3, :])
        xx = np.linspace(0, dist[-1], len(tl.T))
        ind = 5 * np.where(z_transect == st['depth'])
        ax3.plot(xx, np.squeeze(source['source_level'] + tl[10, :]), 'b')
        ax3.plot(xx, np.squeeze(source['source_level'] - Dir[1] + tl[10, :]), 'r')
        ax3.plot(xx, np.squeeze(source['source_level'] - Dir[3] + tl[10, :]), 'm')
        ax3.plot(xx, np.squeeze(source['source_level'] - Dir[5] + tl[10, :]), 'k')
        ax3.plot(xx, source['ambient_noise'] * np.ones(len(xx)), 'r-.')
        ax3.legend(['0°','30°','60°','90°','Ambient noise'])
        ax3.grid()
        ax3.set_xlabel('Range (km)')
        ax3.set_ylabel('dB')

        plt.suptitle(f"Source: \"{source['type']}\" {source['lat']}N {abs(source['lon'])}W - Station: {st['label']}")
        plt.tight_layout()

        # save
        # plt.savefig(root_f / 'Fig_S1.png')
        plt.show()


    # %% Calcul supplémentaire avec Loc et vue polaire

    test_name = 'Test_02'
    nb_p = 36  # Number of azimuthal points (directions)
    nb_ray = int((source['r_max'] * 1000) // 25)  # Number of radial points (range resolution)

    # Generate azimuth (angle) and zenith (range) values
    azimuths = np.radians(np.linspace(0, 360, nb_p))  # Convert degrees to radians
    zeniths = np.linspace(0, source['r_max'], nb_ray)  # Distance steps from source
    r, theta = np.meshgrid(zeniths, azimuths)  # Create mesh grid for polar plots

    TTL = np.zeros((nb_p, nb_ray))
    azi2 = np.zeros(nb_p)

    # Iterate through azimuth angles
    for i in tqdm(range(nb_p), desc="Processing Azimuths"):
        # Compute geodesic endpoint at azimuth i
        [la1, lo1, _] = geodesic(kilometers=source['r_max']).destination((source['lat'], source['lon']), i * 360 / nb_p)
        _, _, d1 = geod.inv(source['lat'], source['lon'], la1, lo1)

        # Create a station at computed lat/lon/distance from source
        st1 = Series({
            'lat': la1,
            'lon': lo1,
            'distance': d1 / 1000,
            'depth': 20
            })

        # Extract bathymetry data
        [zb, dist, z_transect, nb_layer] = extract_bathy(source=source, station=st1, lat=lat, lon=lon, elev=elev)

        # Compute sound speed profile
        sound_speed = compute_sound_speed(method="chen",
                                          yday=day_number,
                                          z=z_transect,
                                          lat_ref=source['lat'],
                                          lon_ref=source['lon'],
                                          croco_data=data_ncdf
                                          )

        # Bellhop
        run_bellhop(executable=bellhop_exe,
                    bellhop_dir=root_bh,
                    filename=test_name,
                    z_max=z_max,
                    source=source,
                    station=st1,
                    dist=dist,
                    zb=zb,
                    sound_speed=sound_speed,
                    z_transect=z_transect,
                    param_seabed=param_seabed,
                    calc = 'I'
                    )

        file_shd = root_bh / f"{test_name}I.shd"
        pressure, geometry = read_shd(file_shd, 0, 0)
        p = np.squeeze(pressure, axis=(0, 1))
        TL = source['source_level'] + 20 * np.log10(abs(p) + np.finfo(float).eps)

        # Interpolate TL data to match zenith distances
        TL2 = np.interp(zeniths, np.linspace(0, st1['distance'], len(TL.T)), TL[int(st1['depth'] + 1), :])
        azi2[i] = (i * 360 / nb_p) / 180 * np.pi
        TTL[i, :] = TL2


    fig = plt.figure(figsize=(14, 8), dpi=200)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
    ax0 = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())

    # Plot bathymetry
    pc = ax0.pcolormesh(lon, lat, elev, vmin=-150, vmax=0, transform=ccrs.PlateCarree(), cmap='jet')
    # ax0.add_feature(cfeature.LAND, facecolor='lightgray', zorder=10)
    # ax0.add_feature(cfeature.COASTLINE, zorder=11)
    fig.colorbar(pc, orientation="vertical", label="Bathymetry (m)")
    gridlines = ax0.gridlines(draw_labels=True)
    gridlines.right_labels = False
    gridlines.bottom_labels = False

    # Plot stations and source location
    for _, st in stations.iterrows():
        ax0.scatter(st['lon'], st['lat'], marker='*', color='black', zorder=13)
        ax0.text(st['lon'], st['lat'], st['label'], color='black', ha='left', va='bottom', zorder=14)

    ax0.scatter(source['lon'], source['lat'], marker='o', color='r', zorder=14)
    # ax0.text(source['lon'], source['lat'],'Source', color='r',  ha='left', va='bottom', zorder=15)

    ax1_0 = fig.add_subplot(gs[1, 0], projection='polar')
    for _, st in stations.iterrows():
        ax1_0.scatter(st['azimuth'], st['distance'])
        ax1_0.annotate(st['label'], xy=(st['azimuth'], st['distance']))

    ax1_1 = fig.add_subplot(gs[1, 1], projection='polar')
    np.clip(TTL, 35, 130, out=TTL)
    pc2 = ax1_1.contourf(theta, r, TTL)
    fig.colorbar(pc2, orientation="vertical", label="Received level (dB)")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
