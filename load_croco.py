import matplotlib.pyplot as plt
from pathlib import Path
from Codes_Python.utils_simulation import read_croco
from utils import find_nearest

ncdf = Path(r'.\Data_Env\croco_out2.nc')
temperature, salinity, depth, lats, lons = read_croco(file=ncdf)

# surface temperature plot
plt.figure()
plt.pcolormesh(lats, lons, temperature[-1, -1, :, :], cmap='jet')
plt.title("Surface Temperature Distribution")
plt.xlabel("Latitude")
plt.ylabel("Longitude")
cbar = plt.colorbar()
cbar.set_label("Temperature (°C)")
plt.show()

# salinity plot along transect at given latitude
lat0 = find_nearest(lats[0, :], 48.12)
plt.figure()
plt.pcolormesh(lons[lat0, :], depth[:, lat0, :], salinity[-1, :, lat0, :], cmap ='jet')
plt.colorbar()
plt.show()
