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

# ── EKF çalıştır ──────────────────────────────────────────
def run_ekf(positions, imu_data, timestamps,
            gps_denied_start=None, gps_denied_end=None):
    ekf = EKF()
    ekf.x[0:3] = positions[0]
    ekf_positions = []
    gps_rate = 10

    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i-1]
        if dt <= 0 or dt > 1.0:
            continue

        ekf.predict(imu_data[i, 0:3], imu_data[i, 3:6], dt)

        in_denied = (gps_denied_start is not None and
                     gps_denied_start <= i <= gps_denied_end)

        if i % gps_rate == 0 and not in_denied:
            ekf.update_gps(positions[i])

        ekf_positions.append(ekf.get_position().copy())

    return np.array(ekf_positions)

print("EKF çalışıyor...")
ekf_normal = run_ekf(positions, imu_data, timestamps)
ekf_denied = run_ekf(positions, imu_data, timestamps,
                     gps_denied_start=300, gps_denied_end=600)

# Ground truth (GPS'i referans alıyoruz)
gt = positions[1:len(ekf_normal)+1]

# ── Metrik fonksiyonları ──────────────────────────────────
def compute_ate(estimated, ground_truth):
    """Absolute Trajectory Error — ortalama mutlak pozisyon hatası."""
    errors = np.linalg.norm(estimated - ground_truth, axis=1)
    return {
        'rmse': np.sqrt(np.mean(errors**2)),
        'mean': np.mean(errors),
        'max':  np.max(errors),
        'std':  np.std(errors)
    }

def compute_rpe(estimated, ground_truth, delta=10):
    """Relative Pose Error — kısa mesafeli bağıl hata."""
    errors = []
    for i in range(len(estimated) - delta):
        gt_rel  = ground_truth[i+delta] - ground_truth[i]
        est_rel = estimated[i+delta]    - estimated[i]
        errors.append(np.linalg.norm(est_rel - gt_rel))
    errors = np.array(errors)
    return {
        'rmse': np.sqrt(np.mean(errors**2)),
        'mean': np.mean(errors),
        'max':  np.max(errors),
        'std':  np.std(errors)
    }

# Ham GPS metriği (baseline)
gps_ate = compute_ate(positions[1:len(ekf_normal)+1], gt)
gps_rpe = compute_rpe(positions[1:len(ekf_normal)+1], gt)

# Normal EKF
ekf_ate = compute_ate(ekf_normal, gt)
ekf_rpe = compute_rpe(ekf_normal, gt)

# GPS-denied EKF
den_ate = compute_ate(ekf_denied, gt)
den_rpe = compute_rpe(ekf_denied, gt)

# ── Sonuç tablosu ─────────────────────────────────────────
print("\n" + "="*55)
print(f"{'Metrik':<20} {'Ham GPS':>10} {'EKF Normal':>10} {'GPS-Denied':>10}")
print("="*55)
print(f"{'ATE RMSE (m)':<20} {gps_ate['rmse']:>10.3f} {ekf_ate['rmse']:>10.3f} {den_ate['rmse']:>10.3f}")
print(f"{'ATE Mean (m)':<20} {gps_ate['mean']:>10.3f} {ekf_ate['mean']:>10.3f} {den_ate['mean']:>10.3f}")
print(f"{'ATE Max (m)':<20} {gps_ate['max']:>10.3f} {ekf_ate['max']:>10.3f} {den_ate['max']:>10.3f}")
print(f"{'RPE RMSE (m)':<20} {gps_rpe['rmse']:>10.3f} {ekf_rpe['rmse']:>10.3f} {den_rpe['rmse']:>10.3f}")
print(f"{'RPE Mean (m)':<20} {gps_rpe['mean']:>10.3f} {ekf_rpe['mean']:>10.3f} {den_rpe['mean']:>10.3f}")
print("="*55)

# ── Grafik ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sol: ATE karşılaştırması bar chart
ax = axes[0]
methods = ['Ham GPS', 'EKF Normal', 'GPS-Denied']
ate_rmse = [gps_ate['rmse'], ekf_ate['rmse'], den_ate['rmse']]
colors   = ['#378ADD', '#1D9E75', '#E05252']
bars = ax.bar(methods, ate_rmse, color=colors, width=0.5, edgecolor='white')
for bar, val in zip(bars, ate_rmse):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.2f}m', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('ATE RMSE (m)')
ax.set_title('Absolute Trajectory Error Karşılaştırması')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(ate_rmse) * 1.2)

# Sağ: Hata zaman serisi
ax2 = axes[1]
err_gps    = np.linalg.norm(positions[1:len(ekf_normal)+1] - gt, axis=1)
err_normal = np.linalg.norm(ekf_normal - gt, axis=1)
err_denied = np.linalg.norm(ekf_denied - gt, axis=1)
t = np.arange(len(err_normal))
ax2.plot(t, err_gps,    'b-',  lw=1,   alpha=0.6, label=f'Ham GPS  (RMSE={gps_ate["rmse"]:.2f}m)')
ax2.plot(t, err_normal, 'g-',  lw=1.5, label=f'EKF Normal (RMSE={ekf_ate["rmse"]:.2f}m)')
ax2.plot(t, err_denied, 'r--', lw=1.5, label=f'GPS-Denied (RMSE={den_ate["rmse"]:.2f}m)')
ax2.axvspan(300, 600, alpha=0.1, color='red', label='GPS yok')
ax2.set_xlabel('Frame')
ax2.set_ylabel('Pozisyon Hatası (m)')
ax2.set_title('Hata Zaman Serisi')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ali/kitti-fusion/results/figures/04_evaluation.png', dpi=150)
plt.show()
print("\nGrafik kaydedildi.")