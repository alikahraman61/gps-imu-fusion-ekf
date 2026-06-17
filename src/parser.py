import pykitti
import numpy as np
import matplotlib.pyplot as plt
from geographiclib.geodesic import Geodesic

# KITTI verisini yükle
basepath = '/home/ali/kitti_raw'
date = '2011_09_30'
drive = '0034'

data = pykitti.raw(basepath, date, drive)
print(f"Toplam frame: {len(data.oxts)}")

# Referans nokta (ilk frame = orijin)
lat0 = data.oxts[0].packet.lat
lon0 = data.oxts[0].packet.lon
alt0 = data.oxts[0].packet.alt

geod = Geodesic.WGS84

def wgs84_to_enu(lat, lon, alt, lat0, lon0, alt0):
    """GPS koordinatlarını ENU (metre) koordinatlarına çevirir."""
    result = geod.Inverse(lat0, lon0, lat, lon)
    dist = result['s12']
    azi  = np.radians(result['azi1'])
    e = dist * np.sin(azi)
    n = dist * np.cos(azi)
    u = alt - alt0
    return e, n, u

# Tüm frame'leri parse et
positions = []  # ENU
imu_data  = []  # [ax, ay, az, wx, wy, wz]
timestamps = []

for i, oxts in enumerate(data.oxts):
    p = oxts.packet
    e, n, u = wgs84_to_enu(p.lat, p.lon, p.alt, lat0, lon0, alt0)
    positions.append([e, n, u])
    imu_data.append([p.ax, p.ay, p.az, p.wx, p.wy, p.wz])
    timestamps.append(data.timestamps[i].timestamp())

positions  = np.array(positions)
imu_data   = np.array(imu_data)
timestamps = np.array(timestamps)
timestamps -= timestamps[0]  # sıfırdan başlat

print(f"Toplam mesafe: {np.max(np.linalg.norm(positions[:, :2], axis=1)):.1f} m")
print(f"Süre: {timestamps[-1]:.1f} saniye")

# GPS trajectory'yi çiz
plt.figure(figsize=(10, 6))
plt.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=1.5, label='Ham GPS (ENU)')
plt.scatter(positions[0, 0], positions[0, 1], c='green', s=100, zorder=5, label='Başlangıç')
plt.scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, zorder=5, label='Bitiş')
plt.xlabel('Doğu (m)')
plt.ylabel('Kuzey (m)')
plt.title('KITTI Drive 0034 — Ham GPS Trajectory (ENU)')
plt.legend()
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/ali/kitti-fusion/results/figures/01_gps_trajectory.png', dpi=150)
plt.show()
print("Grafik kaydedildi.")