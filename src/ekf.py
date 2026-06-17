import logging
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


class EKF:
    """
    15-state Tightly-Coupled GPS/IMU Extended Kalman Filter.

    State vector:
        x[0:3]   — position (m) in ENU frame
        x[3:6]   — velocity (m/s) in ENU frame
        x[6:10]  — orientation quaternion [w, x, y, z]
        x[10:13] — accelerometer bias (m/s²)
        x[13:16] — gyroscope bias (rad/s)

    Parameters
    ----------
    accel_noise : float
        Accelerometer measurement noise std (m/s²).
    gyro_noise : float
        Gyroscope measurement noise std (rad/s).
    accel_bias_noise : float
        Accelerometer bias random walk std.
    gyro_bias_noise : float
        Gyroscope bias random walk std.
    gps_noise_pos : float
        GPS position measurement noise std (m).
    gps_noise_vel : float
        GPS velocity measurement noise std (m/s).
    init_cov : float
        Initial state covariance diagonal value.
    mahalanobis_threshold : float
        Chi-squared threshold for GPS outlier rejection.
    """

    def __init__(
        self,
        accel_noise: float = 0.1,
        gyro_noise: float = 0.01,
        accel_bias_noise: float = 0.001,
        gyro_bias_noise: float = 0.0001,
        gps_noise_pos: float = 2.0,
        gps_noise_vel: float = 0.1,
        init_cov: float = 0.1,
        mahalanobis_threshold: float = 15.0,
    ) -> None:
        # State vektörü
        self.x: NDArray[np.float64] = np.zeros(16)
        self.x[6] = 1.0  # quaternion w=1 (identity)

        # Kovaryans matrisi
        self.P: NDArray[np.float64] = np.eye(15) * init_cov

        # Noise parametreleri
        self.accel_noise = accel_noise
        self.gyro_noise = gyro_noise
        self.accel_bias_noise = accel_bias_noise
        self.gyro_bias_noise = gyro_bias_noise
        self.gps_noise_pos = gps_noise_pos
        self.gps_noise_vel = gps_noise_vel
        self.mahalanobis_threshold = mahalanobis_threshold

        self.g: NDArray[np.float64] = np.array([0.0, 0.0, -9.81])

        logger.debug("EKF initialized with %d states", 16)

    def predict(
        self,
        accel_meas: NDArray[np.float64],
        gyro_meas: NDArray[np.float64],
        dt: float,
    ) -> None:
        """
        IMU ölçümüyle state tahmini (predict adımı).

        Parameters
        ----------
        accel_meas : NDArray[np.float64]
            Accelerometer measurement [ax, ay, az] in body frame (m/s²).
        gyro_meas : NDArray[np.float64]
            Gyroscope measurement [wx, wy, wz] in body frame (rad/s).
        dt : float
            Time step (s).
        """
        pos  = self.x[0:3]
        vel  = self.x[3:6]
        quat = self.x[6:10]
        ba   = self.x[10:13]
        bg   = self.x[13:16]

        accel_corrected = accel_meas - ba
        gyro_corrected  = gyro_meas  - bg

        R = Rotation.from_quat(
            [quat[1], quat[2], quat[3], quat[0]]
        ).as_matrix()

        accel_world = R @ accel_corrected + self.g
        new_pos = pos + vel * dt + 0.5 * accel_world * dt ** 2
        new_vel = vel + accel_world * dt

        angle = np.linalg.norm(gyro_corrected) * dt
        if angle > 1e-10:
            axis  = gyro_corrected / np.linalg.norm(gyro_corrected)
            dq    = Rotation.from_rotvec(axis * angle).as_quat()
            q_cur = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
            q_new = q_cur * Rotation.from_quat(dq)
            q_arr = q_new.as_quat()
            new_quat = np.array([q_arr[3], q_arr[0], q_arr[1], q_arr[2]])
        else:
            new_quat = quat.copy()

        self.x[0:3]  = new_pos
        self.x[3:6]  = new_vel
        self.x[6:10] = new_quat

        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -R @ self._skew(accel_corrected) * dt

        Q = np.zeros((15, 15))
        Q[3:6,   3:6]   = np.eye(3) * self.accel_noise ** 2 * dt ** 2
        Q[6:9,   6:9]   = np.eye(3) * self.gyro_noise  ** 2 * dt ** 2
        Q[9:12,  9:12]  = np.eye(3) * self.accel_bias_noise ** 2 * dt
        Q[12:15, 12:15] = np.eye(3) * self.gyro_bias_noise  ** 2 * dt

        self.P = F @ self.P @ F.T + Q

    def update_gps(
        self,
        gps_pos: NDArray[np.float64],
        gps_vel: NDArray[np.float64] | None = None,
    ) -> bool:
        """
        GPS ölçümüyle state güncelleme (update adımı).

        Parameters
        ----------
        gps_pos : NDArray[np.float64]
            GPS position measurement [e, n, u] in ENU frame (m).
        gps_vel : NDArray[np.float64] | None
            GPS velocity measurement [ve, vn, vu] (m/s). Optional.

        Returns
        -------
        bool
            True if update accepted, False if rejected as outlier.
        """
        if gps_vel is not None:
            H = np.zeros((6, 15))
            H[0:3, 0:3] = np.eye(3)
            H[3:6, 3:6] = np.eye(3)
            z      = np.concatenate([gps_pos, gps_vel])
            z_pred = np.concatenate([self.x[0:3], self.x[3:6]])
            R_noise = np.eye(6)
            R_noise[0:3, 0:3] *= self.gps_noise_pos ** 2
            R_noise[3:6, 3:6] *= self.gps_noise_vel ** 2
        else:
            H = np.zeros((3, 15))
            H[0:3, 0:3] = np.eye(3)
            z      = gps_pos
            z_pred = self.x[0:3]
            R_noise = np.eye(3) * self.gps_noise_pos ** 2

        innovation = z - z_pred

        S    = H @ self.P @ H.T + R_noise
        dist = float(innovation @ np.linalg.inv(S) @ innovation)

        if dist > self.mahalanobis_threshold:
            logger.warning(
                "GPS outlier rejected — Mahalanobis distance: %.2f", dist
            )
            return False

        K  = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ innovation

        self.x[0:3]   += dx[0:3]
        self.x[3:6]   += dx[3:6]
        self.x[10:13] += dx[9:12]
        self.x[13:16] += dx[12:15]

        IKH    = np.eye(15) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_noise @ K.T

        logger.debug(
            "GPS update accepted — innovation norm: %.3f",
            float(np.linalg.norm(innovation)),
        )
        return True

    def update_zupt(self, vel_noise: float = 0.01) -> None:
        """
        Zero Velocity Update (ZUPT).

        Araç hareketsiz tespit edildiğinde hızın sıfır olduğunu bir
        ölçüm olarak EKF'e bildirir. Bu, IMU entegrasyonundan biriken
        hız hatasını sıfırlayarak GPS-denied senaryolarda drift'i
        ciddi şekilde azaltır.

        Parameters
        ----------
        vel_noise : float
            Zero-velocity pseudo-measurement noise std (m/s).
            Küçük değer = daha güçlü sıfırlama.
        """
        H = np.zeros((3, 15))
        H[0:3, 3:6] = np.eye(3)  # velocity state'lerini gözlemle

        z      = np.zeros(3)     # ölçülen hız sıfır
        z_pred = self.x[3:6]     # tahmin edilen hız
        innovation = z - z_pred

        R_noise = np.eye(3) * vel_noise ** 2

        S = H @ self.P @ H.T + R_noise
        K = self.P @ H.T @ np.linalg.inv(S)

        dx = K @ innovation
        self.x[0:3]   += dx[0:3]
        self.x[3:6]   += dx[3:6]
        self.x[10:13] += dx[9:12]
        self.x[13:16] += dx[12:15]

        IKH    = np.eye(15) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_noise @ K.T

        logger.debug("ZUPT applied — velocity reset toward zero")

    def get_position(self) -> NDArray[np.float64]:
        """Returns current position estimate [e, n, u] in metres."""
        return self.x[0:3].copy()

    def get_velocity(self) -> NDArray[np.float64]:
        """Returns current velocity estimate [ve, vn, vu] in m/s."""
        return self.x[3:6].copy()

    def get_position_std(self) -> NDArray[np.float64]:
        """Returns 1-sigma position uncertainty [se, sn, su] in metres."""
        return np.sqrt(np.diag(self.P)[0:3])

    def _skew(self, v: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        3x3 skew-symmetric matrix of vector v.

        Parameters
        ----------
        v : NDArray[np.float64]
            Input 3-vector.

        Returns
        -------
        NDArray[np.float64]
            Skew-symmetric matrix.
        """
        return np.array([
            [ 0.0,  -v[2],  v[1]],
            [ v[2],  0.0,  -v[0]],
            [-v[1],  v[0],  0.0 ],
        ])