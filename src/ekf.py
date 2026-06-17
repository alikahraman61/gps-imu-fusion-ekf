import numpy as np
from scipy.spatial.transform import Rotation

class EKF:
    """
    15-state Tightly-Coupled GPS/IMU Extended Kalman Filter
    State: [pos(3), vel(3), quat(4), accel_bias(3), gyro_bias(3)]
    """

    def __init__(self):
        # State vektörü [px, py, pz, vx, vy, vz, qw, qx, qy, qz, bax, bay, baz, bgx, bgy, bgz]
        self.x = np.zeros(16)
        self.x[6] = 1.0  # quaternion w=1 (identity rotation)

        # Kovaryans matrisi
        self.P = np.eye(15) * 0.1

        # IMU gürültü parametreleri (KITTI OXTS RT3003)
        self.accel_noise  = 0.1    # m/s²
        self.gyro_noise   = 0.01   # rad/s
        self.accel_bias_noise = 0.001
        self.gyro_bias_noise  = 0.0001

        # GPS gürültüsü
        self.gps_noise_pos = 2.0   # metre
        self.gps_noise_vel = 0.1   # m/s

        self.g = np.array([0, 0, -9.81])  # yerçekimi

    def predict(self, accel_meas, gyro_meas, dt):
        """IMU ölçümüyle state tahmini (predict adımı)."""
        # Mevcut state'i çıkar
        pos  = self.x[0:3]
        vel  = self.x[3:6]
        quat = self.x[6:10]   # [w, x, y, z]
        ba   = self.x[10:13]  # accel bias
        bg   = self.x[13:16]  # gyro bias

        # Bias korreksiyonu
        accel_corrected = accel_meas - ba
        gyro_corrected  = gyro_meas  - bg

        # Rotasyon matrisi (body → world)
        R = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()

        # Kinematik entegrasyon
        accel_world = R @ accel_corrected + self.g
        new_pos = pos + vel * dt + 0.5 * accel_world * dt**2
        new_vel = vel + accel_world * dt

        # Quaternion güncelleme
        angle = np.linalg.norm(gyro_corrected) * dt
        if angle > 1e-10:
            axis  = gyro_corrected / np.linalg.norm(gyro_corrected)
            dq    = Rotation.from_rotvec(axis * angle).as_quat()  # [x,y,z,w]
            q_cur = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
            q_new = q_cur * Rotation.from_quat(dq)
            q_arr = q_new.as_quat()  # [x,y,z,w]
            new_quat = np.array([q_arr[3], q_arr[0], q_arr[1], q_arr[2]])
        else:
            new_quat = quat

        # State güncelle
        self.x[0:3]   = new_pos
        self.x[3:6]   = new_vel
        self.x[6:10]  = new_quat
        self.x[10:13] = ba
        self.x[13:16] = bg

        # Jacobian (F matrisi) — linearizasyon
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -R @ self._skew(accel_corrected) * dt

        # Gürültü kovaryansı (Q matrisi)
        Q = np.zeros((15, 15))
        Q[3:6,   3:6]   = np.eye(3) * self.accel_noise**2 * dt**2
        Q[6:9,   6:9]   = np.eye(3) * self.gyro_noise**2  * dt**2
        Q[9:12,  9:12]  = np.eye(3) * self.accel_bias_noise**2 * dt
        Q[12:15, 12:15] = np.eye(3) * self.gyro_bias_noise**2  * dt

        # Kovaryans tahmini
        self.P = F @ self.P @ F.T + Q

    def update_gps(self, gps_pos, gps_vel=None):
        """GPS ölçümüyle state güncelleme (update adımı)."""
        # Ölçüm matrisi H
        if gps_vel is not None:
            H = np.zeros((6, 15))
            H[0:3, 0:3] = np.eye(3)
            H[3:6, 3:6] = np.eye(3)
            z = np.concatenate([gps_pos, gps_vel])
            z_pred = np.concatenate([self.x[0:3], self.x[3:6]])
            R_noise = np.eye(6)
            R_noise[0:3, 0:3] *= self.gps_noise_pos**2
            R_noise[3:6, 3:6] *= self.gps_noise_vel**2
        else:
            H = np.zeros((3, 15))
            H[0:3, 0:3] = np.eye(3)
            z = gps_pos
            z_pred = self.x[0:3]
            R_noise = np.eye(3) * self.gps_noise_pos**2

        # Innovation
        innovation = z - z_pred

        # Mahalanobis outlier rejection
        S = H @ self.P @ H.T + R_noise
        dist = innovation @ np.linalg.inv(S) @ innovation
        if dist > 15.0:  # chi-squared eşiği
            return False  # outlier, güncelleme yapma

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State güncelle
        dx = K @ innovation
        self.x[0:3]   += dx[0:3]
        self.x[3:6]   += dx[3:6]
        self.x[10:13] += dx[9:12]
        self.x[13:16] += dx[12:15]

        # Kovaryans güncelle (Joseph form — numerik kararlılık)
        IKH = np.eye(15) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_noise @ K.T

        return True

    def get_position(self):
        return self.x[0:3].copy()

    def get_velocity(self):
        return self.x[3:6].copy()

    def get_position_std(self):
        """Pozisyon belirsizliği (1-sigma, metre)."""
        return np.sqrt(np.diag(self.P)[0:3])

    def _skew(self, v):
        """3x3 skew-symmetric matris."""
        return np.array([
            [ 0,    -v[2],  v[1]],
            [ v[2],  0,    -v[0]],
            [-v[1],  v[0],  0   ]
        ])