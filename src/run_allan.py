import logging
import numpy as np
import matplotlib.pyplot as plt
import pykitti

from allan_variance import allan_variance, identify_noise_params

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ── Veriyi yükle ──────────────────────────────────────────
basepath = "/home/ali/kitti_raw"
date     = "2011_09_30"
drive    = "0034"

data = pykitti.raw(basepath, date, drive)

# IMU verisini topla
accel = []  # [ax, ay, az]
gyro  = []  # [wx, wy, wz]
for oxts in data.oxts:
    p = oxts.packet
    accel.append([p.ax, p.ay, p.az])
    gyro.append([p.wx, p.wy, p.wz])

accel = np.array(accel)
gyro  = np.array(gyro)

fs = 100.0  # KITTI OXTS IMU rate (Hz)
print(f"IMU örnek sayısı: {len(accel)} ({len(accel)/fs:.1f} saniye)")

# NOT: KITTI drive 0034 sadece ~127 saniye.
# Allan Variance ideal olarak saatlerce sabit (durağan) veri ister.
# Burada metodolojiyi gösteriyoruz; gerçek karakterizasyon için
# uzun süreli sabit IMU kaydı (örn. IMU masada hareketsiz) gerekir.

# ── Her eksen için Allan Variance ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axis_labels = ["X", "Y", "Z"]
colors = ["#E05252", "#1D9E75", "#378ADD"]

# Gyro
ax = axes[0]
for i in range(3):
    taus, adev = allan_variance(gyro[:, i], fs)
    ax.loglog(taus, adev, color=colors[i], lw=1.5, label=f"Gyro {axis_labels[i]}")
    params = identify_noise_params(taus, adev)
    print(f"\nGyro {axis_labels[i]}:")
    print(f"  ARW (white noise)      : {params['white_noise']:.4e} rad/s")
    print(f"  Bias instability       : {params['bias_instability']:.4e} rad/s")

# Referans eğim çizgileri
tau_ref = np.array([taus[0], taus[-1]])
ax.loglog(tau_ref, adev[0] * (tau_ref / taus[0]) ** (-0.5),
          "k--", alpha=0.4, lw=1, label="eğim -1/2 (white noise)")
ax.set_xlabel("Averaging time τ (s)")
ax.set_ylabel("Allan deviation σ(τ) (rad/s)")
ax.set_title("Gyroscope Allan Deviation")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)

# Accelerometer
ax2 = axes[1]
for i in range(3):
    taus, adev = allan_variance(accel[:, i], fs)
    ax2.loglog(taus, adev, color=colors[i], lw=1.5, label=f"Accel {axis_labels[i]}")
    params = identify_noise_params(taus, adev)
    print(f"\nAccel {axis_labels[i]}:")
    print(f"  VRW (white noise)      : {params['white_noise']:.4e} m/s²")
    print(f"  Bias instability       : {params['bias_instability']:.4e} m/s²")

ax2.loglog(tau_ref, adev[0] * (tau_ref / taus[0]) ** (-0.5),
           "k--", alpha=0.4, lw=1, label="eğim -1/2 (white noise)")
ax2.set_xlabel("Averaging time τ (s)")
ax2.set_ylabel("Allan deviation σ(τ) (m/s²)")
ax2.set_title("Accelerometer Allan Deviation")
ax2.legend(fontsize=8)
ax2.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig("/home/ali/kitti-fusion/results/figures/05_allan_variance.png", dpi=150)
plt.show()
print("\nGrafik kaydedildi.")