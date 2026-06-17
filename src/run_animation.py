"""
Trajectory animasyonu — EKF füzyon sonucunu GIF olarak kaydeder.

Araç hareketi, EKF pozisyon tahmini ve covariance ellipse
gerçek zamanlı olarak animasyonlanır.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import pykitti
from geographiclib.geodesic import Geodesic

from ekf import EKF

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

# ── EKF çalıştır ──────────────────────────────────────────
ekf = EKF(
    accel_noise=0.5, gyro_noise=0.05,
    accel_bias_noise=0.01, gyro_bias_noise=0.001,
    gps_noise_pos=0.8, mahalanobis_threshold=16.27,
)
ekf.x[0:3] = positions[0]

ekf_positions = []
ekf_stds      = []
gps_rate      = 10

for i in range(1, len(timestamps)):
    dt = timestamps[i] - timestamps[i-1]
    if dt <= 0 or dt > 1.0:
        ekf_positions.append(ekf.get_position().copy())
        ekf_stds.append(ekf.get_position_std().copy())
        continue
    ekf.predict(imu_data[i, 0:3], imu_data[i, 3:6], dt)
    if i % gps_rate == 0:
        ekf.update_gps(positions[i])
    ekf_positions.append(ekf.get_position().copy())
    ekf_stds.append(ekf.get_position_std().copy())

ekf_positions = np.array(ekf_positions)
ekf_stds      = np.array(ekf_stds)

# Animasyon için her 5 frame'de bir örnekle (GIF boyutu için)
step   = 5
frames = range(0, len(ekf_positions), step)

# ── Animasyon ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(positions[:, 0].min() - 30, positions[:, 0].max() + 30)
ax.set_ylim(positions[:, 1].min() - 30, positions[:, 1].max() + 30)
ax.set_xlabel("Doğu (m)")
ax.set_ylabel("Kuzey (m)")
ax.set_title("GPS/IMU EKF Füzyon — Trajectory Animasyonu")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

# Statik ground truth
ax.plot(positions[:, 0], positions[:, 1], "b-",
        lw=1, alpha=0.3, label="Ham GPS")

# Dinamik elemanlar
ekf_trail,  = ax.plot([], [], "r-",  lw=1.5, label="EKF Trajectory")
ekf_dot,    = ax.plot([], [], "ro",  ms=6, zorder=5)
gps_dot,    = ax.plot([], [], "b^",  ms=5, zorder=4, alpha=0.7,
                       label="GPS fix")
time_text   = ax.text(0.02, 0.96, "", transform=ax.transAxes,
                       fontsize=10, va="top")
err_text    = ax.text(0.02, 0.91, "", transform=ax.transAxes,
                       fontsize=9, va="top", color="darkred")
ax.legend(loc="lower right", fontsize=8)

# Covariance ellipse
ellipse = patches.Ellipse(
    (0, 0), width=1, height=1,
    angle=0, fill=False, edgecolor="red", lw=1.5, alpha=0.6,
    label="1σ belirsizlik"
)
ax.add_patch(ellipse)


def init():
    ekf_trail.set_data([], [])
    ekf_dot.set_data([], [])
    gps_dot.set_data([], [])
    time_text.set_text("")
    err_text.set_text("")
    return ekf_trail, ekf_dot, gps_dot, time_text, err_text, ellipse


def update(frame):
    idx = frame

    # EKF trail
    ekf_trail.set_data(ekf_positions[:idx, 0], ekf_positions[:idx, 1])
    ekf_dot.set_data([ekf_positions[idx, 0]], [ekf_positions[idx, 1]])

    # GPS fix (her gps_rate frame'de)
    real_idx = idx + 1
    if real_idx % gps_rate == 0:
        gps_dot.set_data([positions[real_idx, 0]], [positions[real_idx, 1]])
    else:
        gps_dot.set_data([], [])

    # Covariance ellipse
    sx = ekf_stds[idx, 0]
    sy = ekf_stds[idx, 1]
    ellipse.set_center((ekf_positions[idx, 0], ekf_positions[idx, 1]))
    ellipse.width  = 2 * sx * 3  # 3-sigma
    ellipse.height = 2 * sy * 3

    # Zaman ve hata metni
    t = timestamps[idx + 1]
    err = np.linalg.norm(ekf_positions[idx] - positions[idx + 1])
    time_text.set_text(f"t = {t:.1f}s")
    err_text.set_text(f"hata = {err:.2f}m")

    return ekf_trail, ekf_dot, gps_dot, time_text, err_text, ellipse


print(f"Animasyon oluşturuluyor — {len(list(frames))} frame...")

anim = FuncAnimation(
    fig, update,
    frames=list(frames),
    init_func=init,
    blit=True,
    interval=50,
)

# GIF kaydet
gif_path = "/home/ali/kitti-fusion/results/figures/trajectory.gif"
writer = PillowWriter(fps=20)
anim.save(gif_path, writer=writer)
print(f"GIF kaydedildi: {gif_path}")

plt.show()