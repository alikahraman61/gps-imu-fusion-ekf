"""
Adaptive vs Sabit Noise EKF karşılaştırması.

Senaryo: GPS gürültüsü sürüş boyunca değişir (şehir içi yüksek
multipath → otoyol temiz sinyal). Sabit R bu değişime uyamazken,
innovation-based adaptive R ölçüm gürültüsünü online takip eder.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
import pykitti
from geographiclib.geodesic import Geodesic

from ekf import EKF
from adaptive_ekf import AdaptiveEKF

logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(message)s")

geod = Geodesic.WGS84


def wgs84_to_enu(lat, lon, alt, lat0, lon0, alt0):
    result = geod.Inverse(lat0, lon0, lat, lon)
    dist = result["s12"]
    azi  = np.radians(result["azi1"])
    return dist * np.sin(azi), dist * np.cos(azi), alt - alt0


# ── Veriyi yükle ──────────────────────────────────────────
basepath = "/home/ali/kitti_raw"
data = pykitti.raw(basepath, "2011_09_30", "0034")

lat0 = data.oxts[0].packet.lat
lon0 = data.oxts[0].packet.lon
alt0 = data.oxts[0].packet.alt

positions, imu_data, timestamps = [], [], []
for i, oxts in enumerate(data.oxts):
    p = oxts.packet
    e, n, u = wgs84_to_enu(p.lat, p.lon, p.alt, lat0, lon0, alt0)
    positions.append([e, n, u])
    imu_data.append([p.ax, p.ay, p.az, p.wx, p.wy, p.wz])
    timestamps.append(data.timestamps[i].timestamp())

positions  = np.array(positions)
imu_data   = np.array(imu_data)
timestamps = np.array(timestamps); timestamps -= timestamps[0]
gt = positions.copy()

# ── Zamanla değişen GPS gürültüsü ─────────────────────────
# İlk üçte bir: temiz (0.5m). Orta: yüksek multipath (4.0m). Son: temiz.
n = len(positions)
noise_profile = np.full(n, 0.5)
noise_profile[n//3:2*n//3] = 4.0

rng = np.random.default_rng(42)
gps_rate = 10

# Gürültülü GPS ölçümlerini önceden üret (iki filtre de aynı veriyi görsün)
noisy_gps = positions + rng.normal(0, 1.0, positions.shape) * \
            noise_profile[:, None]


def run_fixed():
    """Sabit R ile standart EKF."""
    ekf = EKF(
        accel_noise=0.5, gyro_noise=0.05,
        accel_bias_noise=0.01, gyro_bias_noise=0.001,
        gps_noise_pos=0.5,  # sabit, temiz sinyale göre ayarlı
        mahalanobis_threshold=1e9,  # bu deneyde rejection kapalı
    )
    ekf.x[0:3] = positions[0]
    est = []
    for i in range(1, n):
        dt = timestamps[i] - timestamps[i-1]
        if dt <= 0 or dt > 1.0:
            est.append(ekf.get_position().copy()); continue
        ekf.predict(imu_data[i, 0:3], imu_data[i, 3:6], dt)
        if i % gps_rate == 0:
            ekf.update_gps(noisy_gps[i])
        est.append(ekf.get_position().copy())
    return np.array(est)


def run_adaptive():
    """Adaptif R ile AdaptiveEKF."""
    ekf = AdaptiveEKF(
        window_size=20,
        accel_noise=0.5, gyro_noise=0.05,
        accel_bias_noise=0.01, gyro_bias_noise=0.001,
        gps_noise_pos=0.5,
        mahalanobis_threshold=1e9,
    )
    ekf.x[0:3] = positions[0]
    est = []
    R_track = []
    for i in range(1, n):
        dt = timestamps[i] - timestamps[i-1]
        if dt <= 0 or dt > 1.0:
            est.append(ekf.get_position().copy())
            R_track.append(np.sqrt(ekf._adaptive_R_pos))
            continue
        ekf.predict(imu_data[i, 0:3], imu_data[i, 3:6], dt)
        if i % gps_rate == 0:
            ekf.update_gps_adaptive(noisy_gps[i])
        est.append(ekf.get_position().copy())
        R_track.append(np.sqrt(ekf._adaptive_R_pos))
    return np.array(est), np.array(R_track)


est_fixed = run_fixed()
est_adaptive, R_track = run_adaptive()


def ate_rmse(est):
    gt_aligned = gt[1:len(est)+1]
    return np.sqrt(np.mean(np.linalg.norm(est - gt_aligned, axis=1) ** 2))


rmse_fixed    = ate_rmse(est_fixed)
rmse_adaptive = ate_rmse(est_adaptive)

print("=" * 55)
print("Adaptive vs Sabit Noise EKF")
print("=" * 55)
print(f"Sabit R    ATE RMSE : {rmse_fixed:.3f} m")
print(f"Adaptive R ATE RMSE : {rmse_adaptive:.3f} m")
print(f"İyileşme            : "
      f"{(1 - rmse_adaptive/rmse_fixed)*100:.1f}%")

# ── Grafik ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Sol üst: trajectory
ax = axes[0, 0]
ax.plot(gt[:, 0], gt[:, 1], "k-", lw=1, alpha=0.4, label="Ground truth")
ax.plot(est_fixed[:, 0], est_fixed[:, 1], "r-", lw=1.2,
        label=f"Sabit R ({rmse_fixed:.2f}m)")
ax.plot(est_adaptive[:, 0], est_adaptive[:, 1], "g-", lw=1.2,
        label=f"Adaptive R ({rmse_adaptive:.2f}m)")
ax.set_xlabel("Doğu (m)"); ax.set_ylabel("Kuzey (m)")
ax.set_title("Trajectory")
ax.legend(fontsize=8); ax.axis("equal"); ax.grid(True, alpha=0.3)

# Sağ üst: hata zaman serisi
ax2 = axes[0, 1]
err_f = np.linalg.norm(est_fixed - gt[1:len(est_fixed)+1], axis=1)
err_a = np.linalg.norm(est_adaptive - gt[1:len(est_adaptive)+1], axis=1)
t = np.arange(len(err_f))
ax2.plot(t, err_f, "r-", lw=1, label="Sabit R")
ax2.plot(t, err_a, "g-", lw=1, label="Adaptive R")
ax2.axvspan(n//3, 2*n//3, alpha=0.12, color="orange",
            label="Yüksek gürültü bölgesi")
ax2.set_xlabel("Frame"); ax2.set_ylabel("Pozisyon hatası (m)")
ax2.set_title("Hata Zaman Serisi")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

# Sol alt: GPS gürültü profili vs adaptif R tahmini
ax3 = axes[1, 0]
ax3.plot(t, noise_profile[1:len(t)+1], "b-", lw=1.5,
         label="Gerçek GPS gürültüsü (std)")
ax3.plot(t, R_track[:len(t)], "g-", lw=1.5,
         label="Adaptif R tahmini (√R)")
ax3.axvspan(n//3, 2*n//3, alpha=0.12, color="orange")
ax3.set_xlabel("Frame"); ax3.set_ylabel("Gürültü std (m)")
ax3.set_title("Adaptif R, Gerçek Gürültüyü Takip Ediyor")
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

# Sağ alt: RMSE bar
ax4 = axes[1, 1]
bars = ax4.bar(["Sabit R", "Adaptive R"], [rmse_fixed, rmse_adaptive],
               color=["#E05252", "#1D9E75"], width=0.5)
for bar, val in zip(bars, [rmse_fixed, rmse_adaptive]):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f"{val:.2f}m", ha="center", fontsize=11, fontweight="bold")
ax4.set_ylabel("ATE RMSE (m)")
ax4.set_title("Genel Performans")
ax4.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("/home/ali/kitti-fusion/results/figures/09_adaptive.png", dpi=150)
plt.show()
print("\nGrafik kaydedildi.")