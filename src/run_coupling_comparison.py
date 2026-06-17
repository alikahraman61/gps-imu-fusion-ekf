"""
Loosely-coupled vs Tightly-coupled füzyon karşılaştırması.

KITTI GPS'i çözülmüş konum (pseudorange değil) sağladığından, iki
gerçekçi varyant ADİL koşullarda karşılaştırılır:

  - Aynı sentetik GPS gürültüsü her iki yönteme uygulanır.
  - Process noise (Q) gerçekçi seviyede tutulur — düşük Q, EKF'i
    overconfident yapar, kovaryans yeterince şişmez ve Mahalanobis
    testi geçerli ölçümleri reddeder (ilk denemedeki bug'ın kökü).
    Yüksek Q ile kovaryans doğru büyür, geçerli GPS kabul edilir,
    ama gerçek sıçramalar hâlâ reddedilir — saha güvenliği korunur.
  - Outlier rejection AÇIK bırakılır; gerçek bir otonom araçta GPS
    multipath sıçramalarına karşı zorunludur.
  - Tek fark coupling mantığıdır:

  Loosely-coupled : GPS düşük oranda (yavaş fix, 0.5 s) gelir.
                    İki aşamalı yaklaşımı temsil eder.
  Tightly-coupled : GPS yüksek oranda (0.1 s) gelir ve EKF state'iyle
                    sıkı eşleşir. Daha sık ölçüm = daha iyi gözlemlenebilirlik.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
import pykitti
from geographiclib.geodesic import Geodesic

from ekf import EKF

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

# Ortak GPS gürültü seviyesi (her iki yönteme aynı uygulanır)
GPS_NOISE_STD = 0.8  # m — iyi bir GPS fix'in tipik değeri


def run_fusion(gps_rate: int, seed: int = 0):
    """
    Tek bir füzyon konfigürasyonu çalıştırır.

    Process noise gerçekçi seviyede; outlier rejection açık.

    Parameters
    ----------
    gps_rate : int
        Kaç IMU adımında bir GPS güncellemesi gelir.
        Düşük = sık (tightly), yüksek = seyrek (loosely).
    seed : int
        Tekrarlanabilirlik için rastgelelik tohumu.
    """
    rng = np.random.default_rng(seed)

    # Process noise gerçekçi seviyede — EKF'in overconfidence'ını önler.
    ekf = EKF(
        accel_noise=0.5,        # m/s² — artırılmış process noise
        gyro_noise=0.05,        # rad/s
        accel_bias_noise=0.01,
        gyro_bias_noise=0.001,
        gps_noise_pos=GPS_NOISE_STD,
        mahalanobis_threshold=16.27,  # 3-DOF, %99.9 güven (chi-sq)
    )
    ekf.x[0:3] = positions[0]

    est = []
    n_rejected = 0
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i-1]
        if dt <= 0 or dt > 1.0:
            est.append(ekf.get_position().copy())
            continue

        ekf.predict(imu_data[i, 0:3], imu_data[i, 3:6], dt)

        if i % gps_rate == 0:
            noisy_gps = positions[i] + rng.normal(0, GPS_NOISE_STD, 3)
            accepted = ekf.update_gps(noisy_gps)
            if not accepted:
                n_rejected += 1

        est.append(ekf.get_position().copy())

    return np.array(est), n_rejected


# ── İki konfigürasyon — TEK FARK: GPS oranı ───────────────
est_loosely, rej_l = run_fusion(gps_rate=50)   # seyrek fix (0.5 s)
est_tightly, rej_t = run_fusion(gps_rate=10)   # sık ölçüm (0.1 s)


# ── Metrikler ─────────────────────────────────────────────
def ate_rmse(est):
    gt_aligned = gt[1:len(est)+1]
    return np.sqrt(np.mean(np.linalg.norm(est - gt_aligned, axis=1) ** 2))


rmse_loosely = ate_rmse(est_loosely)
rmse_tightly = ate_rmse(est_tightly)

print("=" * 55)
print("Loosely vs Tightly Coupled (Adil Karşılaştırma)")
print("=" * 55)
print(f"Ortak GPS gürültüsü       : {GPS_NOISE_STD} m")
print(f"Loosely GPS oranı         : 0.5 s (seyrek fix)")
print(f"Tightly GPS oranı         : 0.1 s (sık ölçüm)")
print("-" * 55)
print(f"Loosely-coupled ATE RMSE  : {rmse_loosely:.3f} m  "
      f"(reddedilen: {rej_l})")
print(f"Tightly-coupled ATE RMSE  : {rmse_tightly:.3f} m  "
      f"(reddedilen: {rej_t})")
print(f"İyileşme                  : "
      f"{(1 - rmse_tightly/rmse_loosely)*100:.1f}%")

# ── Grafik ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.plot(gt[:, 0], gt[:, 1], "k-", lw=1, alpha=0.4, label="Ground truth")
ax.plot(est_loosely[:, 0], est_loosely[:, 1], "r-", lw=1.3,
        label=f"Loosely (RMSE={rmse_loosely:.2f}m)")
ax.plot(est_tightly[:, 0], est_tightly[:, 1], "g-", lw=1.3,
        label=f"Tightly (RMSE={rmse_tightly:.2f}m)")
ax.set_xlabel("Doğu (m)")
ax.set_ylabel("Kuzey (m)")
ax.set_title("Trajectory — Loosely vs Tightly Coupled")
ax.legend(fontsize=9)
ax.axis("equal")
ax.grid(True, alpha=0.3)

ax2 = axes[1]
err_l = np.linalg.norm(est_loosely - gt[1:len(est_loosely)+1], axis=1)
err_t = np.linalg.norm(est_tightly - gt[1:len(est_tightly)+1], axis=1)
t = np.arange(len(err_l))
ax2.plot(t, err_l, "r-", lw=1.2, label=f"Loosely (RMSE={rmse_loosely:.2f}m)")
ax2.plot(t, err_t, "g-", lw=1.2, label=f"Tightly (RMSE={rmse_tightly:.2f}m)")
ax2.set_xlabel("Frame")
ax2.set_ylabel("Pozisyon hatası (m)")
ax2.set_title("Hata Zaman Serisi")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/home/ali/kitti-fusion/results/figures/08_coupling.png", dpi=150)
plt.show()
print("\nGrafik kaydedildi.")