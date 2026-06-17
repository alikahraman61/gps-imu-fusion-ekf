import logging
from numpy.typing import NDArray
import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


class UKF:
    """
    15-state Hybrid UKF for GPS/IMU fusion.

    Predict: UKF sigma points ile nonlineer mean propagasyonu +
             EKF Jacobian ile kovaryans propagasyonu (kararlılık).
    Update : EKF tarzı Joseph form (kovaryans pozitif tanımlılığı garantili).

    Bu yaklaşım "Sigma-Point Kalman Filter with linearized covariance"
    olarak bilinir. Tam UKF kovaryans propagasyonu quaternion manifold
    kısıtları nedeniyle sayısal kararsızlık ürettiğinden, EKF kovaryans
    propagasyonu tercih edilir.

    State vector:
        x[0:3]   — position (m) ENU
        x[3:6]   — velocity (m/s) ENU
        x[6:10]  — quaternion [w, x, y, z]
        x[10:13] — accel bias (m/s²)
        x[13:16] — gyro bias (rad/s)

    References
    ----------
    Wan, E.A. & Van der Merwe, R. (2000). "The Unscented Kalman Filter
    for Nonlinear Estimation." IEEE ASSPCC.
    """

    def __init__(
        self,
        accel_noise: float = 0.1,
        gyro_noise: float = 0.01,
        accel_bias_noise: float = 0.001,
        gyro_bias_noise: float = 0.0001,
        gps_noise_pos: float = 2.0,
        init_cov: float = 0.1,
        mahalanobis_threshold: float = 15.0,
        alpha: float = 0.01,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        self.x: NDArray[np.float64] = np.zeros(16)
        self.x[6] = 1.0

        self.n = 15
        self.P: NDArray[np.float64] = np.eye(self.n) * init_cov

        self.accel_noise = accel_noise
        self.gyro_noise = gyro_noise
        self.accel_bias_noise = accel_bias_noise
        self.gyro_bias_noise = gyro_bias_noise
        self.gps_noise_pos = gps_noise_pos
        self.mahalanobis_threshold = mahalanobis_threshold
        self.g = np.array([0.0, 0.0, -9.81])

        n = self.n
        self.lam = alpha ** 2 * (n + kappa) - n
        c = n + self.lam
        self.Wm = np.full(2 * n + 1, 0.5 / c)
        self.Wc = np.full(2 * n + 1, 0.5 / c)
        self.Wm[0] = self.lam / c
        self.Wc[0] = self.lam / c + (1 - alpha ** 2 + beta)

        logger.debug("Hybrid UKF initialized — n=%d, lambda=%.6f", n, self.lam)

    # ── Yardımcı ──────────────────────────────────────────
    @staticmethod
    def _skew(v: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([
            [ 0.0,  -v[2],  v[1]],
            [ v[2],  0.0,  -v[0]],
            [-v[1],  v[0],  0.0 ],
        ])

    def _chol(self, A: NDArray[np.float64]) -> NDArray[np.float64]:
        A = (A + A.T) / 2.0
        for eps in [1e-8, 1e-6, 1e-4, 1e-2]:
            try:
                return np.linalg.cholesky(A + np.eye(self.n) * eps)
            except np.linalg.LinAlgError:
                continue
        ev, evec = np.linalg.eigh(A)
        return np.linalg.cholesky(
            evec @ np.diag(np.maximum(ev, 1e-8)) @ evec.T + np.eye(self.n) * 1e-8
        )

    def _boxplus(
        self,
        x: NDArray[np.float64],
        dx: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """x ⊞ dx: 15D pertürbasyonu 16D state'e uygular."""
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

    def _sigma_points(self) -> NDArray[np.float64]:
        """2n+1 sigma noktası üretir."""
        self.P = (self.P + self.P.T) / 2.0
        L = self._chol((self.n + self.lam) * self.P)
        sp = [self.x.copy()]
        for j in range(self.n):
            sp.append(self._boxplus(self.x,  L[:, j]))
            sp.append(self._boxplus(self.x, -L[:, j]))
        return np.array(sp)

    def _propagate(
        self,
        s: NDArray[np.float64],
        accel: NDArray[np.float64],
        gyro: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        """Tek sigma noktasını IMU kinematikleriyle ilerletir."""
        ba = s[10:13]
        bg = s[13:16]
        ac = accel - ba
        gc = gyro  - bg
        R  = Rotation.from_quat([s[7], s[8], s[9], s[6]]).as_matrix()
        aw = R @ ac + self.g
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

    # ── Predict ────────────────────────────────────────────
    def predict(
        self,
        accel_meas: NDArray[np.float64],
        gyro_meas: NDArray[np.float64],
        dt: float,
    ) -> None:
        """
        Hibrit predict adımı.

        Mean: UKF sigma point propagasyonu (nonlineer).
        Covariance: EKF Jacobian propagasyonu (kararlı).

        Parameters
        ----------
        accel_meas : NDArray[np.float64]
            Accelerometer ölçümü [ax, ay, az] (m/s²).
        gyro_meas : NDArray[np.float64]
            Gyroscope ölçümü [wx, wy, wz] (rad/s).
        dt : float
            Zaman adımı (s).
        """
        # UKF mean
        sp      = self._sigma_points()
        sp_prop = np.array([
            self._propagate(s, accel_meas, gyro_meas, dt)
            for s in sp
        ])
        x_mean = np.einsum("i,ij->j", self.Wm, sp_prop)
        nq = np.linalg.norm(x_mean[6:10])
        x_mean[6:10] /= nq if nq > 1e-14 else 1.0

        # EKF kovaryans propagasyonu
        quat = self.x[6:10]
        ba   = self.x[10:13]
        ac   = accel_meas - ba
        Rot  = Rotation.from_quat(
            [quat[1], quat[2], quat[3], quat[0]]
        ).as_matrix()

        F = np.eye(self.n)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -Rot @ self._skew(ac) * dt

        Q = np.zeros((self.n, self.n))
        Q[3:6,   3:6]   = np.eye(3) * self.accel_noise      ** 2 * dt ** 2
        Q[6:9,   6:9]   = np.eye(3) * self.gyro_noise        ** 2 * dt ** 2
        Q[9:12,  9:12]  = np.eye(3) * self.accel_bias_noise  ** 2 * dt
        Q[12:15, 12:15] = np.eye(3) * self.gyro_bias_noise   ** 2 * dt

        self.x = x_mean
        self.P = F @ self.P @ F.T + Q

    # ── Update ─────────────────────────────────────────────
    def update_gps(
        self,
        gps_pos: NDArray[np.float64],
    ) -> bool:
        """
        EKF tarzı GPS pozisyon güncellemesi (Joseph form).

        Parameters
        ----------
        gps_pos : NDArray[np.float64]
            GPS pozisyon ölçümü [e, n, u] (m).

        Returns
        -------
        bool
            Kabul edildiyse True, outlier ise False.
        """
        H = np.zeros((3, self.n))
        H[0:3, 0:3] = np.eye(3)

        innov = gps_pos - self.x[0:3]
        R_gps = np.eye(3) * self.gps_noise_pos ** 2
        S     = H @ self.P @ H.T + R_gps

        dist = float(innov @ np.linalg.inv(S) @ innov)
        if dist > self.mahalanobis_threshold:
            logger.warning("UKF GPS outlier — dist: %.2f", dist)
            return False

        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self._boxplus(self.x, K @ innov)

        IKH    = np.eye(self.n) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_gps @ K.T
        self.P = (self.P + self.P.T) / 2.0

        ev = np.linalg.eigvalsh(self.P)
        if np.any(ev < 0):
            self.P += np.eye(self.n) * (abs(ev.min()) + 1e-8)

        return True

    def get_position(self) -> NDArray[np.float64]:
        """Returns current position estimate [e, n, u] in metres."""
        return self.x[0:3].copy()

    def get_position_std(self) -> NDArray[np.float64]:
        """Returns 1-sigma position uncertainty [se, sn, su] in metres."""
        return np.sqrt(np.diag(self.P)[0:3])