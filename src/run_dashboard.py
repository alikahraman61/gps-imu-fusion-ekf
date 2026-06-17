"""
İnteraktif Plotly dashboard — tüm analiz sonuçlarını tek sayfada gösterir.
HTML olarak export edilir, tarayıcıda açılır, demo linki paylaşılabilir.
"""
import numpy as np
import pykitti
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from geographiclib.geodesic import Geodesic

from ekf import EKF
from adaptive_ekf import AdaptiveEKF

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

positions, imu_data, timestamps, speeds = [], [], [], []
for i, oxts in enumerate(data.oxts):
    p = oxts.packet
    e, n, u = wgs84_to_enu(p.lat, p.lon, p.alt, lat0, lon0, alt0)
    positions.append([e, n, u])
    imu_data.append([p.ax, p.ay, p.az, p.wx, p.wy, p.wz])
    timestamps.append(data.timestamps[i].timestamp())
    speeds.append(np.sqrt(p.vf**2 + p.vl**2))

positions  = np.array(positions)
imu_data   = np.array(imu_data)
timestamps = np.array(timestamps); timestamps -= timestamps[0]
speeds     = np.array(speeds)
gt         = positions.copy()

COMMON = dict(
    accel_noise=0.5, gyro_noise=0.05,
    accel_bias_noise=0.01, gyro_bias_noise=0.001,
    gps_noise_pos=0.8, mahalanobis_threshold=16.27,
)
gps_rate = 10
rng = np.random.default_rng(42)


def run_ekf(denied_start=None, denied_end=None):
    ekf = EKF(**COMMON)
    ekf.x[0:3] = positions[0]
    est, stds = [], []
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i-1]
        if dt <= 0 or dt > 1.0:
            est.append(ekf.get_position().copy())
            stds.append(ekf.get_position_std().copy())
            continue
        ekf.predict(imu_data[i, 0:3], imu_data[i, 3:6], dt)
        in_denied = (denied_start is not None and
                     denied_start <= i <= denied_end)
        if i % gps_rate == 0 and not in_denied:
            ekf.update_gps(positions[i])
        est.append(ekf.get_position().copy())
        stds.append(ekf.get_position_std().copy())
    return np.array(est), np.array(stds)


def run_adaptive():
    noise_profile = np.full(len(positions), 0.5)
    n = len(positions)
    noise_profile[n//3:2*n//3] = 4.0
    noisy = positions + rng.normal(0, 1.0, positions.shape) * noise_profile[:, None]
    ekf = AdaptiveEKF(window_size=20, **COMMON)
    ekf.x[0:3] = positions[0]
    est, R_hist = [], []
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i-1]
        if dt <= 0 or dt > 1.0:
            est.append(ekf.get_position().copy())
            R_hist.append(np.sqrt(ekf._adaptive_R_pos))
            continue
        ekf.predict(imu_data[i, 0:3], imu_data[i, 3:6], dt)
        if i % gps_rate == 0:
            ekf.update_gps_adaptive(noisy[i])
        est.append(ekf.get_position().copy())
        R_hist.append(np.sqrt(ekf._adaptive_R_pos))
    return np.array(est), np.array(R_hist)


print("EKF çalışıyor...")
est_normal, stds_normal = run_ekf()
est_denied, stds_denied = run_ekf(denied_start=300, denied_end=600)
print("Adaptive EKF çalışıyor...")
est_adaptive, R_hist = run_adaptive()

gt_aligned = gt[1:len(est_normal)+1]
err_normal = np.linalg.norm(est_normal - gt_aligned, axis=1)
err_denied = np.linalg.norm(est_denied - gt_aligned, axis=1)
err_adapt  = np.linalg.norm(est_adaptive - gt_aligned, axis=1)
t = timestamps[1:len(est_normal)+1]

rmse_normal = float(np.sqrt(np.mean(err_normal**2)))
rmse_denied = float(np.sqrt(np.mean(err_denied**2)))
rmse_adapt  = float(np.sqrt(np.mean(err_adapt**2)))

# ── Dashboard ─────────────────────────────────────────────
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        "Trajectory — EKF vs Ground Truth",
        "GPS-Denied Trajectory",
        "Pozisyon Hatası Zaman Serisi",
        "EKF Pozisyon Belirsizliği (1σ)",
        "Hız Profili & GPS-Denied Bölge",
        "Adaptive R — GPS Gürültü Takibi",
    ),
    vertical_spacing=0.10,
    horizontal_spacing=0.08,
)

# 1. Trajectory
fig.add_trace(go.Scatter(
    x=gt[:, 0], y=gt[:, 1], mode="lines",
    line=dict(color="rgba(100,100,200,0.4)", width=1),
    name="Ham GPS"), row=1, col=1)
fig.add_trace(go.Scatter(
    x=est_normal[:, 0], y=est_normal[:, 1], mode="lines",
    line=dict(color="#E05252", width=2),
    name=f"EKF (RMSE={rmse_normal:.2f}m)"), row=1, col=1)

# 2. GPS-Denied
fig.add_trace(go.Scatter(
    x=gt[:, 0], y=gt[:, 1], mode="lines",
    line=dict(color="rgba(100,100,200,0.4)", width=1),
    name="Ham GPS", showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(
    x=est_normal[:, 0], y=est_normal[:, 1], mode="lines",
    line=dict(color="#1D9E75", width=1.5),
    name="Normal EKF", showlegend=True), row=1, col=2)
fig.add_trace(go.Scatter(
    x=est_denied[:, 0], y=est_denied[:, 1], mode="lines",
    line=dict(color="#E05252", width=1.5, dash="dash"),
    name=f"GPS-Denied (RMSE={rmse_denied:.0f}m)"), row=1, col=2)

# 3. Hata zaman serisi
fig.add_trace(go.Scatter(
    x=t, y=err_normal, mode="lines",
    line=dict(color="#1D9E75", width=1.5),
    name="Normal EKF hatası"), row=2, col=1)
fig.add_trace(go.Scatter(
    x=t, y=err_denied, mode="lines",
    line=dict(color="#E05252", width=1.5),
    name="GPS-Denied hatası"), row=2, col=1)
fig.add_vrect(
    x0=timestamps[300], x1=timestamps[600],
    fillcolor="red", opacity=0.1,
    row=2, col=1)

# 4. Belirsizlik
fig.add_trace(go.Scatter(
    x=t, y=stds_normal[:, 0], mode="lines",
    line=dict(color="#E05252", width=1.5),
    name="Normal std-X"), row=2, col=2)
fig.add_trace(go.Scatter(
    x=t, y=stds_denied[:, 0], mode="lines",
    line=dict(color="#E05252", width=1.5, dash="dash"),
    name="Denied std-X"), row=2, col=2)

# 5. Hız profili
fig.add_trace(go.Scatter(
    x=timestamps, y=speeds, mode="lines",
    line=dict(color="#378ADD", width=1.5),
    name="Gerçek hız"), row=3, col=1)
fig.add_vrect(
    x0=timestamps[300], x1=timestamps[600],
    fillcolor="red", opacity=0.1,
    annotation_text="GPS yok",
    row=3, col=1)

# 6. Adaptive R
n_pts = len(positions)
noise_prof = np.full(n_pts, 0.5)
noise_prof[n_pts//3:2*n_pts//3] = 4.0
fig.add_trace(go.Scatter(
    x=np.arange(len(R_hist)), y=R_hist, mode="lines",
    line=dict(color="#1D9E75", width=1.5),
    name="Adaptive R tahmini"), row=3, col=2)
fig.add_trace(go.Scatter(
    x=np.arange(n_pts), y=noise_prof, mode="lines",
    line=dict(color="#378ADD", width=1.5, dash="dot"),
    name="Gerçek gürültü"), row=3, col=2)

# Layout
fig.update_layout(
    title=dict(
        text="GPS/IMU Sensor Fusion — EKF Pipeline Dashboard",
        font=dict(size=20),
    ),
    height=1100,
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=-0.05),
)

# Eksen etiketleri
for col in [1, 2]:
    fig.update_xaxes(title_text="Doğu (m)", row=1, col=col)
    fig.update_yaxes(title_text="Kuzey (m)", row=1, col=col)
fig.update_xaxes(title_text="Zaman (s)", row=2, col=1)
fig.update_yaxes(title_text="Hata (m)", row=2, col=1)
fig.update_xaxes(title_text="Zaman (s)", row=2, col=2)
fig.update_yaxes(title_text="Std (m)", row=2, col=2)
fig.update_xaxes(title_text="Zaman (s)", row=3, col=1)
fig.update_yaxes(title_text="Hız (m/s)", row=3, col=1)
fig.update_xaxes(title_text="Frame", row=3, col=2)
fig.update_yaxes(title_text="Gürültü std (m)", row=3, col=2)

# HTML kaydet
html_path = "/home/ali/kitti-fusion/results/dashboard.html"
fig.write_html(html_path)
print(f"Dashboard kaydedildi: {html_path}")
fig.show()