import numpy as np
import sys
sys.path.insert(0, 'src')
from scipy.spatial.transform import Rotation
import pykitti
from geographiclib.geodesic import Geodesic

geod = Geodesic.WGS84


def wgs84_to_enu(lat, lon, alt, lat0, lon0, alt0):
    r = geod.Inverse(lat0, lon0, lat, lon)
    azi = np.radians(r["azi1"])
    return r["s12"] * np.sin(azi), r["s12"] * np.cos(azi), alt - alt0


data = pykitti.raw("/home/ali/kitti_raw", "2011_09_30", "0034")
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
timestamps = np.array(timestamps)
timestamps -= timestamps[0]

print("=" * 55)
print("Hibrit UKF mean + EKF covariance stabilitesi")
print("=" * 55)

n     = 15
alpha = 0.01
lam   = alpha ** 2 * n - n
Wm    = np.full(2 * n + 1, 0.5 / (n + lam))
Wc    = np.full(2 * n + 1, 0.5 / (n + lam))
Wm[0] = lam / (n + lam)
Wc[0] = lam / (n + lam) + (1 - alpha ** 2 + 2.0)

GPS_NOISE = 0.8
rng       = np.random.default_rng(42)
noisy_gps = positions + rng.normal(0, GPS_NOISE, positions.shape)
gps_rate  = 10

x = np.zeros(16)
x[6]   = 1.0
x[0:3] = positions[0]
P = np.eye(n) * 0.1


def skew(v):
    return np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],  0.0,  -v[0]],
        [-v[1],  v[0],  0.0 ],
    ])


def boxplus(x, dx):
    xn = x.copy()
    xn[0:3]   += dx[0:3]
    xn[3:6]   += dx[3:6]
    xn[10:13] += dx[9:12]
    xn[13:16] += dx[12:15]
    rv = dx[6:9]
    if np.linalg.norm(rv) > 1e-14:
        dq  = Rotation.from_rotvec(rv).as_quat()
        qc  = Rotation.from_quat([x[7], x[8], x[9], x[6]])
        qn  = (qc * Rotation.from_quat(dq)).as_quat()
        xn[6:10] = [qn[3], qn[0], qn[1], qn[2]]
    nq = np.linalg.norm(xn[6:10])
    xn[6:10] /= nq if nq > 1e-14 else 1.0
    return xn


def propagate(s, accel, gyro, dt):
    ba = s[10:13]
    bg = s[13:16]
    ac = accel - ba
    gc = gyro  - bg
    Rot = Rotation.from_quat([s[7], s[8], s[9], s[6]]).as_matrix()
    aw  = Rot @ ac + np.array([0.0, 0.0, -9.81])
    out = s.copy()
    out[0:3] = s[0:3] + s[3:6] * dt + 0.5 * aw * dt ** 2
    out[3:6] = s[3:6] + aw * dt
    ang = np.linalg.norm(gc) * dt
    if ang > 1e-14:
        dq  = Rotation.from_rotvec(gc / np.linalg.norm(gc) * ang).as_quat()
        qc  = Rotation.from_quat([s[7], s[8], s[9], s[6]])
        qn  = (qc * Rotation.from_quat(dq)).as_quat()
        nq  = np.linalg.norm(qn)
        out[6:10] = [qn[3]/nq, qn[0]/nq, qn[1]/nq, qn[2]/nq]
    return out


def sigma_points(x, P, n, lam):
    P = (P + P.T) / 2.0
    A = (n + lam) * P
    for eps in [1e-8, 1e-6, 1e-4, 1e-2]:
        try:
            L = np.linalg.cholesky(A + np.eye(n) * eps)
            break
        except np.linalg.LinAlgError:
            continue
    else:
        ev, evec = np.linalg.eigh(A)
        L = np.linalg.cholesky(
            evec @ np.diag(np.maximum(ev, 1e-8)) @ evec.T + np.eye(n) * 1e-8
        )
    sp = [x.copy()]
    for j in range(n):
        sp.append(boxplus(x,  L[:, j]))
        sp.append(boxplus(x, -L[:, j]))
    return np.array(sp)


est_positions = []

for i in range(1, len(timestamps)):
    dt = timestamps[i] - timestamps[i - 1]
    if dt <= 0 or dt > 1.0:
        est_positions.append(x[0:3].copy())
        continue

    # ── PREDICT: UKF mean + EKF covariance ────────────────
    sp      = sigma_points(x, P, n, lam)
    sp_prop = np.array([
        propagate(s, imu_data[i, 0:3], imu_data[i, 3:6], dt)
        for s in sp
    ])

    # UKF ağırlıklı ortalama (mean)
    x_mean = np.einsum("i,ij->j", Wm, sp_prop)
    nq = np.linalg.norm(x_mean[6:10])
    x_mean[6:10] /= nq if nq > 1e-14 else 1.0

    # EKF linearizasyonu ile kovaryans propagasyonu (kararlı)
    quat = x[6:10]
    ba   = x[10:13]
    ac   = imu_data[i, 0:3] - ba
    Rot  = Rotation.from_quat(
        [quat[1], quat[2], quat[3], quat[0]]
    ).as_matrix()

    F = np.eye(n)
    F[0:3, 3:6] = np.eye(3) * dt
    F[3:6, 6:9] = -Rot @ skew(ac) * dt

    Q = np.zeros((n, n))
    Q[3:6,   3:6]   = np.eye(3) * 0.5  ** 2 * dt ** 2
    Q[6:9,   6:9]   = np.eye(3) * 0.05 ** 2 * dt ** 2
    Q[9:12,  9:12]  = np.eye(3) * 0.01 ** 2 * dt
    Q[12:15, 12:15] = np.eye(3) * 0.001 ** 2 * dt

    x = x_mean
    P = F @ P @ F.T + Q

    # ── UPDATE: EKF tarzı (Joseph form) ───────────────────
    if i % gps_rate == 0:
        H = np.zeros((3, n))
        H[0:3, 0:3] = np.eye(3)

        innov = noisy_gps[i] - x[0:3]
        R_gps = np.eye(3) * GPS_NOISE ** 2
        S     = H @ P @ H.T + R_gps

        dist = float(innov @ np.linalg.inv(S) @ innov)
        if dist <= 16.27:
            K = P @ H.T @ np.linalg.inv(S)
            x = boxplus(x, K @ innov)

            IKH = np.eye(n) - K @ H
            P   = IKH @ P @ IKH.T + K @ R_gps @ K.T
            P   = (P + P.T) / 2.0

            ev = np.linalg.eigvalsh(P)
            if np.any(ev < 0):
                P += np.eye(n) * (abs(ev.min()) + 1e-8)

    est_positions.append(x[0:3].copy())

    if i % 200 == 0:
        err = np.linalg.norm(x[0:3] - positions[i])
        print(f"Frame {i:4d}: pos_err={err:8.3f}m  "
              f"P_trace={np.trace(P):.4f}  "
              f"min_ev={np.linalg.eigvalsh(P).min():.2e}")

est_positions = np.array(est_positions)
final_err = np.linalg.norm(
    est_positions - positions[1:len(est_positions) + 1], axis=1
)
ate = np.sqrt(np.mean(final_err ** 2))
print(f"\nATE RMSE : {ate:.3f} m")
print(f"EKF ref  : ~1.65 m")
print(f"Fark     : {ate/1.65:.1f}x")