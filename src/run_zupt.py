"""
ZUPT (Zero Velocity Update) etkisinin analizi.

Üç bölüm:
1. Kontrollü test — duran araçta ZUPT'un drift'i sıfırladığını gösterir.
2. Detector karşılaştırması — IMU-only vs GPS-aided hareketsizlik tespiti
   (precision / recall ile).
3. Tespit görselleştirmesi.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
import pykitti
from geographiclib.geodesic import Geodesic

from ekf import EKF
from motion_detector import detect_stationary_imu, detect_stationary_gps_aided

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

geod = Geodesic.WGS84


def wgs84_to_enu(lat, lon, alt, lat0, lon0, alt0):
    result = geod.Inverse(lat0, lon0, lat, lon)
    dist = result["s12"]
    azi  = np.radians(result["azi1"])
    return dist * np.sin(azi), dist * np.cos(azi), alt - alt0


# ══════════════════════════════════════════════════════════
# BÖLÜM 1 — KONTROLLÜ TEST
# ══════════════════════════════════════════════════════════
print("=" * 55)
print("BÖLÜM 1 — Kontrollü ZUPT Testi")
print("=" * 55)

fs = 100.0
dt = 1.0 / fs
n_steps = 1000

rng = np.random.default_rng(0)
accel_bias_true = np.array([0.05, 0.03, 0.0])
gyro_meas = np.zeros(3)


def run_controlled(use_zupt: bool):
    ekf = EKF()
    positions = []
    for i in range(n_steps):
        accel = np.array([0.0, 0.0, 9.81]) + accel_bias_true \
                + rng.normal(0, 0.01, 3)
        ekf.predict(accel, gyro_meas, dt)
        if use_zupt and i % 10 == 0:
            ekf.update_zupt(vel_noise=0.01)
        positions.append(ekf.get_position().copy())
    return np.array(positions)


pos_no_zupt   = run_controlled(use_zupt=False)
pos_with_zupt = run_controlled(use_zupt=True)

drift_no_zupt   = np.linalg.norm(pos_no_zupt[-1])
drift_with_zupt = np.linalg.norm(pos_with_zupt[-1])

print(f"ZUPT olmadan son drift : {drift_no_zupt:.2f} m")
print(f"ZUPT ile son drift     : {drift_with_zupt:.4f} m")
print(f"İyileşme               : {drift_no_zupt / max(drift_with_zupt, 1e-6):.0f}x")


# ══════════════════════════════════════════════════════════
# BÖLÜM 2 — DETECTOR KARŞILAŞTIRMASI
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("BÖLÜM 2 — IMU-only vs GPS-aided Detector")
print("=" * 55)

basepath = "/home/ali/kitti_raw"
data = pykitti.raw(basepath, "2011_09_30", "0034")

lat0 = data.oxts[0].packet.lat
lon0 = data.oxts[0].packet.lon
alt0 = data.oxts[0].packet.alt

positions, imu_data, timestamps, speeds = [], [], [], []
for i, oxts in enumerate(data.oxts):
    p = oxts.packet
    e, n, u = wgs84_to_enu(p.lat, p.lon, p.alt, lat0, lon0, alt0)
    positions.append([e, n, u])
    imu_data.append([p.ax, p.ay, p.az, p.wx, p.wy, p.wz])
    timestamps.append(data.timestamps[i].timestamp())
    speeds.append(np.sqrt(p.vf ** 2 + p.vl ** 2))

positions  = np.array(positions)
imu_data   = np.array(imu_data)
timestamps = np.array(timestamps); timestamps -= timestamps[0]
speeds     = np.array(speeds)

# Ground truth: gerçek durma (hız < 0.5 m/s)
true_stop = speeds < 0.5

window = 5
flags_imu = np.zeros(len(imu_data), dtype=bool)
flags_gps = np.zeros(len(imu_data), dtype=bool)

for i in range(window, len(imu_data)):
    acc_win = imu_data[i-window:i, 0:3]
    gyr_win = imu_data[i-window:i, 3:6]
    flags_imu[i] = detect_stationary_imu(
        acc_win, gyr_win,
        accel_var_thresh=0.15, gyro_thresh=0.03, gravity_tol=0.5
    )
    flags_gps[i] = detect_stationary_gps_aided(
        acc_win, gyr_win, speed=speeds[i],
        speed_thresh=0.5,
        accel_var_thresh=0.15, gyro_thresh=0.03, gravity_tol=0.5
    )


def metrics(flags, truth):
    tp = int(np.sum(flags & truth))
    fp = int(np.sum(flags & ~truth))
    fn = int(np.sum(~flags & truth))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) \
        if (precision + recall) > 0 else 0.0
    return tp, fp, fn, precision, recall, f1


tp_i, fp_i, fn_i, prec_i, rec_i, f1_i = metrics(flags_imu, true_stop)
tp_g, fp_g, fn_g, prec_g, rec_g, f1_g = metrics(flags_gps, true_stop)

print(f"\n{'Metrik':<14} {'IMU-only':>12} {'GPS-aided':>12}")
print("-" * 40)
print(f"{'True Pos':<14} {tp_i:>12} {tp_g:>12}")
print(f"{'False Pos':<14} {fp_i:>12} {fp_g:>12}")
print(f"{'False Neg':<14} {fn_i:>12} {fn_g:>12}")
print(f"{'Precision':<14} {prec_i:>11.1%} {prec_g:>11.1%}")
print(f"{'Recall':<14} {rec_i:>11.1%} {rec_g:>11.1%}")
print(f"{'F1 Score':<14} {f1_i:>11.3f} {f1_g:>11.3f}")


# ══════════════════════════════════════════════════════════
# GRAFİK
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Sol üst: Kontrollü test
ax = axes[0, 0]
t_ctrl = np.arange(n_steps) * dt
ax.plot(t_ctrl, np.linalg.norm(pos_no_zupt, axis=1),
        "r-", lw=1.5, label=f"ZUPT yok ({drift_no_zupt:.1f}m)")
ax.plot(t_ctrl, np.linalg.norm(pos_with_zupt, axis=1),
        "g-", lw=1.5, label=f"ZUPT var ({drift_with_zupt:.3f}m)")
ax.set_xlabel("Zaman (s)")
ax.set_ylabel("Pozisyon drift (m)")
ax.set_title("Kontrollü Test — Duran Araçta ZUPT")
ax.legend()
ax.grid(True, alpha=0.3)

# Sağ üst: IMU-only tespit
ax2 = axes[0, 1]
ax2.plot(timestamps, speeds, "b-", lw=1, label="Gerçek hız")
ax2.scatter(timestamps[flags_imu], speeds[flags_imu], c="orange", s=10,
            zorder=5, label=f"IMU-only (P={prec_i:.0%})")
ax2.axhline(0.5, color="gray", ls=":", alpha=0.6)
ax2.set_xlabel("Zaman (s)")
ax2.set_ylabel("Hız (m/s)")
ax2.set_title("IMU-only Detector — Yanlış Pozitifler")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Sol alt: GPS-aided tespit
ax3 = axes[1, 0]
ax3.plot(timestamps, speeds, "b-", lw=1, label="Gerçek hız")
ax3.scatter(timestamps[flags_gps], speeds[flags_gps], c="green", s=10,
            zorder=5, label=f"GPS-aided (P={prec_g:.0%})")
ax3.axhline(0.5, color="gray", ls=":", alpha=0.6)
ax3.set_xlabel("Zaman (s)")
ax3.set_ylabel("Hız (m/s)")
ax3.set_title("GPS-aided Detector — Temiz Tespit")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Sağ alt: Precision/Recall/F1 bar chart
ax4 = axes[1, 1]
labels = ["Precision", "Recall", "F1"]
imu_vals = [prec_i, rec_i, f1_i]
gps_vals = [prec_g, rec_g, f1_g]
x = np.arange(len(labels))
w = 0.35
ax4.bar(x - w/2, imu_vals, w, color="#EF9F27", label="IMU-only")
ax4.bar(x + w/2, gps_vals, w, color="#1D9E75", label="GPS-aided")
for i, (iv, gv) in enumerate(zip(imu_vals, gps_vals)):
    ax4.text(i - w/2, iv + 0.02, f"{iv:.2f}", ha="center", fontsize=9)
    ax4.text(i + w/2, gv + 0.02, f"{gv:.2f}", ha="center", fontsize=9)
ax4.set_xticks(x)
ax4.set_xticklabels(labels)
ax4.set_ylim(0, 1.15)
ax4.set_ylabel("Skor")
ax4.set_title("Detector Performans Karşılaştırması")
ax4.legend()
ax4.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("/home/ali/kitti-fusion/results/figures/07_zupt.png", dpi=150)
plt.show()
print("\nGrafik kaydedildi.")