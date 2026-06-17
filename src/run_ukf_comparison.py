"""
EKF vs Hibrit UKF karşılaştırması.

EKF  : 15-state, IMU predict (Jacobian) + GPS update.
UKF  : 15-state, Hibrit — UKF sigma point mean + EKF Jacobian covariance.
       Predict'te sigma point'ler nonlineer ortalamayı daha iyi yakalar.
       Kovaryans EKF ile propagate edilir (quaternion manifold kararlılığı).

Sonuç: UKF'in nonlineer mean tahmini, EKF linearizasyonuna kıyasla
daha düşük hata üretir. Hesap maliyeti ~40x daha fazla.
"""
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import pykitti
from geographiclib.geodesic import Geodesic

from ekf import EKF
from ukf import UKF

logging.basicConfig(level=logging.ERROR)
geod = Geodesic.WGS84


def wgs84_to_enu(lat, lon, alt, lat0, lon0, alt0):
    r = geod.Inverse(lat0, lon0, lat, lon)
    azi = np.radians(r["azi1"])
    return r["s12"]*np.sin(azi), r["s12"]*np.cos(azi), alt-alt0


# ── Veri ──────────────────────────────────────────────────
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
N  = len(positions)

GPS_NOISE = 0.8
rng       = np.random.default_rng(42)
noisy_gps = positions + rng.normal(0, GPS_NOISE, positions.shape)
gps_rate  = 10

COMMON = dict(
    accel_noise=0.5, gyro_noise=0.05,
    accel_bias_noise=0.01, gyro_bias_noise=0.001,
    gps_noise_pos=GPS_NOISE,
    mahalanobis_threshold=16.27,
)


def run_filter(filt, label: str):
    filt.x[0:3] = positions[0]
    est = []
    t0  = time.perf_counter()
    for i in range(1, N):
        dt = timestamps[i] - timestamps[i-1]
        if dt <= 0 or dt > 1.0:
            est.append(filt.get_position().copy()); continue
        filt.predict(imu_data[i, 0:3], imu_data[i, 3:6], dt)
        if i % gps_rate == 0:
            filt.update_gps(noisy_gps[i])
        est.append(filt.get_position().copy())
    elapsed = time.perf_counter() - t0
    print(f"{label}: {elapsed:.2f}s ({elapsed/N*1000:.2f} ms/frame)")
    return np.array(est), elapsed


ekf = EKF(**COMMON)
ukf = UKF(**COMMON)

print("Filtreler çalışıyor...")
est_ekf, t_ekf = run_filter(ekf, "EKF")
est_ukf, t_ukf = run_filter(ukf, "UKF")


def ate(est):
    return float(np.sqrt(np.mean(
        np.linalg.norm(est - gt[1:len(est)+1], axis=1)**2
    )))


def rpe(est, d=10):
    g = gt[1:len(est)+1]
    e = [np.linalg.norm((est[i+d]-est[i])-(g[i+d]-g[i]))
         for i in range(len(est)-d)]
    return float(np.sqrt(np.mean(np.array(e)**2)))


ate_e = ate(est_ekf); rpe_e = rpe(est_ekf)
ate_u = ate(est_ukf); rpe_u = rpe(est_ukf)

print("\n" + "=" * 52)
print(f"{'':20} {'EKF':>15} {'Hibrit UKF':>15}")
print("-" * 52)
print(f"{'ATE RMSE (m)':<20} {ate_e:>15.3f} {ate_u:>15.3f}")
print(f"{'RPE RMSE (m)':<20} {rpe_e:>15.3f} {rpe_u:>15.3f}")
print(f"{'Süre (s)':<20} {t_ekf:>15.2f} {t_ukf:>15.2f}")
print(f"{'ms/frame':<20} {t_ekf/N*1000:>15.2f} {t_ukf/N*1000:>15.2f}")
ate_imp = (1 - ate_u/ate_e)*100
print("-" * 52)
print(f"{'ATE iyileşme':<20} {ate_imp:>14.1f}%")
print(f"{'Hesap maliyeti':<20} {'1x':>15} {t_ukf/t_ekf:>14.1f}x")
print("=" * 52)

# ── Grafik ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.plot(gt[:, 0], gt[:, 1], "k-", lw=1, alpha=0.4, label="Ground truth")
ax.plot(est_ekf[:, 0], est_ekf[:, 1], "r-", lw=1.3,
        label=f"EKF (ATE={ate_e:.2f}m)")
ax.plot(est_ukf[:, 0], est_ukf[:, 1], "g-", lw=1.3,
        label=f"Hibrit UKF (ATE={ate_u:.2f}m)")
ax.set_xlabel("Doğu (m)"); ax.set_ylabel("Kuzey (m)")
ax.set_title("Trajectory — EKF vs Hibrit UKF")
ax.legend(fontsize=8); ax.axis("equal"); ax.grid(True, alpha=0.3)

ax2 = axes[0, 1]
err_e = np.linalg.norm(est_ekf - gt[1:len(est_ekf)+1], axis=1)
err_u = np.linalg.norm(est_ukf - gt[1:len(est_ukf)+1], axis=1)
t = np.arange(len(err_e))
ax2.plot(t, err_e, "r-", lw=1, label=f"EKF RMSE={ate_e:.2f}m")
ax2.plot(t, err_u, "g-", lw=1, label=f"UKF RMSE={ate_u:.2f}m")
ax2.set_xlabel("Frame"); ax2.set_ylabel("Hata (m)")
ax2.set_title("Hata Zaman Serisi")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

ax3 = axes[1, 0]
metrics = ["ATE RMSE", "RPE RMSE"]
x = np.arange(len(metrics)); w = 0.35
b1 = ax3.bar(x-w/2, [ate_e, rpe_e], w, color="#E05252", label="EKF")
b2 = ax3.bar(x+w/2, [ate_u, rpe_u], w, color="#1D9E75", label="Hibrit UKF")
for b, v in zip(list(b1)+list(b2), [ate_e, rpe_e, ate_u, rpe_u]):
    ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
             f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
ax3.set_xticks(x); ax3.set_xticklabels(metrics)
ax3.set_ylabel("RMSE (m)")
ax3.set_title("Performans Karşılaştırması")
ax3.legend(); ax3.grid(True, alpha=0.3, axis="y")

ax4 = axes[1, 1]
b3 = ax4.bar(["EKF", "Hibrit UKF"],
             [t_ekf, t_ukf], color=["#E05252", "#1D9E75"], width=0.5)
for b, v in zip(b3, [t_ekf, t_ukf]):
    ax4.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
             f"{v:.2f}s", ha="center", fontsize=11, fontweight="bold")
ax4.set_ylabel("Toplam süre (s)")
ax4.set_title(f"Hesap Maliyeti (UKF = {t_ukf/t_ekf:.1f}x EKF)")
ax4.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("/home/ali/kitti-fusion/results/figures/10_ekf_vs_ukf.png", dpi=150)
plt.show()
print("\nGrafik kaydedildi.")