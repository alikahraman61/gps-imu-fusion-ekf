"""
Multi-sequence KITTI benchmark.

Üç farklı sürüş senaryosunda EKF performansını karşılaştırır.
Hybrid UKF kısa sekanslar için aktif, uzun highway sekansında
kovaryans birikimi nedeniyle devre dışı (EKF daha kararlı).

Bu bulgu literatürle tutarlıdır: UKF kısa nonlineer manevralar için
avantajlı, uzun vadeli bias birikiminde EKF daha kararlıdır.
"""
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


def load_sequence(basepath, date, drive):
    try:
        data = pykitti.raw(basepath, date, drive)
    except Exception as e:
        print(f"  ⚠ Yüklenemedi: {e}")
        return None, None, None, None

    positions, imu_data, timestamps = [], [], []
    lat0 = lon0 = alt0 = None
    for i, oxts in enumerate(data.oxts):
        p = oxts.packet
        if i == 0:
            lat0, lon0, alt0 = p.lat, p.lon, p.alt
        e, n, u = wgs84_to_enu(p.lat, p.lon, p.alt, lat0, lon0, alt0)
        positions.append([e, n, u])
        imu_data.append([p.ax, p.ay, p.az, p.wx, p.wy, p.wz])
        timestamps.append(data.timestamps[i].timestamp())

    positions  = np.array(positions)
    imu_data   = np.array(imu_data)
    timestamps = np.array(timestamps); timestamps -= timestamps[0]
    dist = float(np.sum(
        np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)
    ))
    print(f"  Yüklendi: {len(positions)} frame, "
          f"{timestamps[-1]:.1f}s, {dist:.0f}m")
    return positions, imu_data, timestamps, dist


def run_filter(filt, positions, imu_data, timestamps,
               gps_rate=10, gps_noise=0.8, seed=42):
    rng    = np.random.default_rng(seed)
    noisy  = positions + rng.normal(0, gps_noise, positions.shape)
    filt.x[0:3] = positions[0]
    est = []
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i-1]
        if dt <= 0 or dt > 1.0:
            est.append(filt.get_position().copy()); continue
        filt.predict(imu_data[i, 0:3], imu_data[i, 3:6], dt)
        if i % gps_rate == 0:
            filt.update_gps(noisy[i])
        est.append(filt.get_position().copy())
    return np.array(est)


def compute_metrics(est, gt):
    gt_a  = gt[1:len(est)+1]
    ate   = float(np.sqrt(np.mean(np.linalg.norm(est - gt_a, axis=1)**2)))
    delta = 10
    rpe_e = [
        np.linalg.norm((est[i+delta]-est[i]) - (gt_a[i+delta]-gt_a[i]))
        for i in range(len(est)-delta)
    ]
    rpe = float(np.sqrt(np.mean(np.array(rpe_e)**2)))
    return ate, rpe


COMMON = dict(
    accel_noise=0.5, gyro_noise=0.05,
    accel_bias_noise=0.01, gyro_bias_noise=0.001,
    gps_noise_pos=0.8, mahalanobis_threshold=16.27,
)

# use_ukf=False: uzun sekanlarda UKF kovaryans birikimi nedeniyle kararsız
SEQUENCES = [
    ("2011_09_30", "0034", "Urban",       True),
    ("2011_09_30", "0027", "Residential", True),
    ("2011_09_30", "0028", "Highway",     False),
]

basepath = "/home/ali/kitti_raw"
results  = {}

print("=" * 60)
print("Multi-Sequence KITTI Benchmark")
print("=" * 60)

for date, drive, label, use_ukf in SEQUENCES:
    print(f"\n── {label} (drive_{drive}) ──")
    positions, imu_data, timestamps, dist = load_sequence(
        basepath, date, drive
    )
    if positions is None:
        continue

    # EKF
    ekf     = EKF(**COMMON)
    est_ekf = run_filter(ekf, positions, imu_data, timestamps)
    ate_e, rpe_e = compute_metrics(est_ekf, positions)
    print(f"  EKF — ATE: {ate_e:.3f}m  RPE: {rpe_e:.3f}m")

    # UKF (sadece kısa sekanslar)
    ate_u = rpe_u = None
    est_ukf = None
    if use_ukf:
        ukf     = UKF(**COMMON)
        est_ukf = run_filter(ukf, positions, imu_data, timestamps)
        ate_u, rpe_u = compute_metrics(est_ukf, positions)
        imp = (1 - ate_u/ate_e)*100
        print(f"  UKF — ATE: {ate_u:.3f}m  RPE: {rpe_u:.3f}m")
        print(f"  UKF iyileşme: {imp:.1f}%")
    else:
        print(f"  UKF — devre dışı (uzun sekans, kovaryans birikimi)")

    results[label] = {
        "drive": drive,
        "frames": len(positions),
        "duration": timestamps[-1],
        "distance": dist,
        "use_ukf": use_ukf,
        "ekf_ate": ate_e, "ekf_rpe": rpe_e,
        "ukf_ate": ate_u, "ukf_rpe": rpe_u,
        "positions": positions,
        "est_ekf": est_ekf,
        "est_ukf": est_ukf,
    }

# ── Özet tablo ────────────────────────────────────────────
print("\n" + "=" * 72)
print(f"{'Sekans':<14} {'Frame':>6} {'Süre':>7} {'Mesafe':>8} "
      f"{'EKF ATE':>9} {'UKF ATE':>10} {'İyileşme':>9}")
print("-" * 72)

ekf_ates_short, ukf_ates_short = [], []
for label, r in results.items():
    ukf_str = f"{r['ukf_ate']:>9.3f}m" if r["ukf_ate"] else "   N/A (uzun)"
    imp_str = ""
    if r["ukf_ate"]:
        imp     = (1 - r["ukf_ate"]/r["ekf_ate"])*100
        imp_str = f"{imp:>8.1f}%"
        ekf_ates_short.append(r["ekf_ate"])
        ukf_ates_short.append(r["ukf_ate"])
    print(f"{label:<14} {r['frames']:>6} {r['duration']:>6.1f}s "
          f"{r['distance']:>7.0f}m "
          f"{r['ekf_ate']:>9.3f}m {ukf_str} {imp_str}")

print("=" * 72)
if ekf_ates_short:
    avg_imp = (1 - np.mean(ukf_ates_short)/np.mean(ekf_ates_short))*100
    print(f"Kısa sekans ort. UKF iyileşmesi: {avg_imp:.1f}%")

# ── Grafik ────────────────────────────────────────────────
n_seq = len(results)
fig, axes = plt.subplots(2, n_seq, figsize=(6*n_seq, 10))
if n_seq == 1:
    axes = axes.reshape(2, 1)

for col, (label, r) in enumerate(results.items()):
    pos   = r["positions"]
    est_e = r["est_ekf"]
    est_u = r["est_ukf"]

    # Trajectory
    ax = axes[0, col]
    ax.plot(pos[:, 0], pos[:, 1], "b-", lw=1, alpha=0.3,
            label="Ground truth")
    ax.plot(est_e[:, 0], est_e[:, 1], "r-", lw=1.5,
            label=f"EKF ({r['ekf_ate']:.2f}m)")
    if est_u is not None:
        ax.plot(est_u[:, 0], est_u[:, 1], "g-", lw=1.5,
                label=f"UKF ({r['ukf_ate']:.2f}m)")
    ax.set_title(f"{label} — drive_{r['drive']}\n"
                 f"{r['frames']} frame, {r['duration']:.0f}s, "
                 f"{r['distance']:.0f}m")
    ax.set_xlabel("Doğu (m)"); ax.set_ylabel("Kuzey (m)")
    ax.legend(fontsize=7); ax.axis("equal"); ax.grid(True, alpha=0.3)

    # Hata zaman serisi
    ax2 = axes[1, col]
    gt_a  = pos[1:len(est_e)+1]
    err_e = np.linalg.norm(est_e - gt_a, axis=1)
    t = np.arange(len(err_e))
    ax2.plot(t, err_e, "r-", lw=1,
             label=f"EKF RMSE={r['ekf_ate']:.2f}m")
    if est_u is not None:
        err_u = np.linalg.norm(est_u - gt_a, axis=1)
        ax2.plot(t, err_u, "g-", lw=1,
                 label=f"UKF RMSE={r['ukf_ate']:.2f}m")
    else:
        ax2.text(0.5, 0.5, "UKF devre dışı\n(uzun sekans)",
                 transform=ax2.transAxes, ha="center", va="center",
                 fontsize=10, color="gray",
                 bbox=dict(boxstyle="round", fc="lightyellow"))
    ax2.set_xlabel("Frame"); ax2.set_ylabel("Hata (m)")
    ax2.set_title(f"{label} — Hata Zaman Serisi")
    ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/home/ali/kitti-fusion/results/figures/11_benchmark.png",
            dpi=150)
plt.show()
print("\nGrafik kaydedildi.")