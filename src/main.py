import numpy as np
import matplotlib.pyplot as plt
import pykitti
from geographiclib.geodesic import Geodesic
from ekf import EKF

# ── Koordinat dönüşümü ────────────────────────────────────
geod = Geodesic.WGS84

def wgs84_to_enu(lat, lon, alt, lat0, lon0, alt0):
    result = geod.Inverse(lat0, lon0, lat, lon)
    dist = result['s12']
    azi  = np.radians(result['azi1'])
    e = dist * np.sin(azi)
    n = dist * np.cos(azi)
    u = alt - alt0
    return e, n, u

# ── Veriyi yükle ──────────────────────────────────────────
basepath = '/home/ali/kitti_raw'
date     = '2011_09_30'
drive    = '0034'

data = pykitti.raw(basepath, date, drive)

lat0 = data.oxts[0].packet.lat
lon0 = data.oxts[0].packet.lon
alt0 = data.oxts[0].packet.alt

positions  = []
imu_data   = []
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
timestamps -= timestamps[0]

print(f"Veri yüklendi: {len(positions)} frame")

# ── GPS-denied penceresi ───────────────────────────────────
# Frame 300-600 arası GPS sinyali yok (tünel / bina senaryosu)
GPS_DENIED_START = 300
GPS_DENIED_END   = 600

def run_ekf(positions, imu_data, timestamps, gps_denied_start=None, gps_denied_end=None):
    ekf = EKF()
    ekf.x[0:3] = positions[0]

    ekf_positions = []
    ekf_stds      = []
    gps_used      = []
    gps_rate      = 10

    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i-1]
        if dt <= 0 or dt > 1.0:
            continue

        accel = imu_data[i, 0:3]
        gyro  = imu_data[i, 3:6]
        ekf.predict(accel, gyro, dt)

        # GPS-denied kontrolü
        in_denied_zone = (gps_denied_start is not None and
                          gps_denied_start <= i <= gps_denied_end)

        gps_update = False
        if i % gps_rate == 0 and not in_denied_zone:
            ekf.update_gps(positions[i])
            gps_update = True

        ekf_positions.append(ekf.get_position().copy())
        ekf_stds.append(ekf.get_position_std().copy())
        gps_used.append(gps_update)

    return (np.array(ekf_positions),
            np.array(ekf_stds),
            np.array(gps_used))

# Normal EKF (GPS her zaman var)
print("Normal EKF çalışıyor...")
ekf_normal, stds_normal, _ = run_ekf(positions, imu_data, timestamps)

# GPS-denied EKF
print("GPS-denied EKF çalışıyor...")
ekf_denied, stds_denied, gps_used = run_ekf(
    positions, imu_data, timestamps,
    gps_denied_start=GPS_DENIED_START,
    gps_denied_end=GPS_DENIED_END
)

print(f"Normal EKF  — son pozisyon hatası: "
      f"{np.linalg.norm(ekf_normal[-1] - positions[-1]):.2f} m")
print(f"GPS-denied  — son pozisyon hatası: "
      f"{np.linalg.norm(ekf_denied[-1] - positions[-1]):.2f} m")

# Denied bölgedeki max drift
denied_slice = ekf_denied[GPS_DENIED_START:GPS_DENIED_END]
gt_slice     = positions[GPS_DENIED_START:GPS_DENIED_END]
drifts       = np.linalg.norm(denied_slice - gt_slice, axis=1)
print(f"GPS-denied bölgede max drift: {np.max(drifts):.2f} m")

# ── Grafik ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Sol üst: Trajectory karşılaştırması
ax = axes[0, 0]
ax.plot(positions[:, 0],      positions[:, 1],      'b-',  lw=1,   alpha=0.5, label='Ham GPS')
ax.plot(ekf_normal[:, 0],     ekf_normal[:, 1],     'g-',  lw=1.5, label='EKF Normal')
ax.plot(ekf_denied[:, 0],     ekf_denied[:, 1],     'r--', lw=1.5, label='EKF GPS-denied')
# Denied bölgeyi işaretle
ax.axvspan(positions[GPS_DENIED_START, 0] - 5,
           positions[GPS_DENIED_END, 0] + 5,
           alpha=0.1, color='red', label='GPS yok')
ax.scatter(positions[0, 0],  positions[0, 1],  c='green', s=100, zorder=5)
ax.scatter(positions[-1, 0], positions[-1, 1], c='black', s=100, zorder=5)
ax.set_xlabel('Doğu (m)')
ax.set_ylabel('Kuzey (m)')
ax.set_title('Trajectory: Normal vs GPS-Denied')
ax.legend(fontsize=8)
ax.axis('equal')
ax.grid(True, alpha=0.3)

# Sağ üst: Pozisyon hatası zamanla
ax2 = axes[0, 1]
err_normal = np.linalg.norm(ekf_normal - positions[1:len(ekf_normal)+1], axis=1)
err_denied = np.linalg.norm(ekf_denied - positions[1:len(ekf_denied)+1], axis=1)
t = np.arange(len(err_normal))
ax2.plot(t, err_normal, 'g-', lw=1.5, label='Normal EKF hatası')
ax2.plot(t, err_denied, 'r-', lw=1.5, label='GPS-denied hatası')
ax2.axvspan(GPS_DENIED_START, GPS_DENIED_END, alpha=0.15, color='red', label='GPS yok')
ax2.set_xlabel('Frame')
ax2.set_ylabel('Pozisyon Hatası (m)')
ax2.set_title('Pozisyon Hatası — Normal vs GPS-Denied')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Sol alt: Covariance (belirsizlik) karşılaştırması
ax3 = axes[1, 0]
ax3.plot(np.arange(len(stds_normal)), stds_normal[:, 0], 'g-',  lw=1.5, label='Normal std-X')
ax3.plot(np.arange(len(stds_denied)), stds_denied[:, 0], 'r--', lw=1.5, label='Denied std-X')
ax3.axvspan(GPS_DENIED_START, GPS_DENIED_END, alpha=0.15, color='red', label='GPS yok')
ax3.set_xlabel('Frame')
ax3.set_ylabel('Standart Sapma (m)')
ax3.set_title('Belirsizlik Büyümesi — GPS Kesilince')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Sağ alt: GPS-denied drift detayı
ax4 = axes[1, 1]
denied_frames = np.arange(GPS_DENIED_START, min(GPS_DENIED_END, len(drifts) + GPS_DENIED_START))
ax4.plot(denied_frames, drifts[:len(denied_frames)], 'r-', lw=2)
ax4.fill_between(denied_frames, 0, drifts[:len(denied_frames)], alpha=0.3, color='red')
ax4.set_xlabel('Frame')
ax4.set_ylabel('Drift (m)')
ax4.set_title(f'GPS-Denied Bölgede IMU Drift\n(Frame {GPS_DENIED_START}–{GPS_DENIED_END})')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ali/kitti-fusion/results/figures/03_gps_denied.png', dpi=150)
plt.show()
print("Grafik kaydedildi.")