import numpy as np
import pytest
from src.ekf import EKF


@pytest.fixture
def ekf() -> EKF:
    """Varsayılan parametrelerle EKF instance'ı."""
    return EKF()


class TestEKFInit:
    def test_initial_position_zero(self, ekf: EKF) -> None:
        """Başlangıç pozisyonu sıfır olmalı."""
        np.testing.assert_array_equal(ekf.get_position(), [0.0, 0.0, 0.0])

    def test_initial_quaternion_identity(self, ekf: EKF) -> None:
        """Başlangıç quaternion'u identity olmalı [w=1, x=0, y=0, z=0]."""
        np.testing.assert_array_almost_equal(ekf.x[6:10], [1.0, 0.0, 0.0, 0.0])

    def test_initial_covariance_positive_definite(self, ekf: EKF) -> None:
        """Başlangıç kovaryansı pozitif tanımlı olmalı."""
        eigenvalues = np.linalg.eigvalsh(ekf.P)
        assert np.all(eigenvalues > 0), "Kovaryans pozitif tanımlı değil"

    def test_initial_bias_zero(self, ekf: EKF) -> None:
        """Başlangıç bias'ları sıfır olmalı."""
        np.testing.assert_array_equal(ekf.x[10:16], np.zeros(6))


class TestEKFPredict:
    def test_predict_position_advances(self, ekf: EKF) -> None:
        """Sabit hız ile pozisyon ilerlemeli."""
        ekf.x[0:3] = [0.0, 0.0, 0.0]
        ekf.x[3:6] = [1.0, 0.0, 0.0]

        accel = np.array([0.0, 0.0, 9.81])
        gyro  = np.zeros(3)
        ekf.predict(accel, gyro, dt=1.0)

        assert ekf.get_position()[0] > 0.5, "Doğu yönünde ilerleme bekleniyor"

    def test_predict_covariance_grows(self, ekf: EKF) -> None:
        """Predict sonrası kovaryans büyümeli."""
        P_before = np.trace(ekf.P)
        accel = np.array([0.0, 0.0, 9.81])
        gyro  = np.zeros(3)
        ekf.predict(accel, gyro, dt=0.01)
        P_after = np.trace(ekf.P)
        assert P_after > P_before, "Predict sonrası kovaryans büyümeli"

    def test_predict_covariance_symmetric(self, ekf: EKF) -> None:
        """Kovaryans matrisi simetrik olmalı."""
        accel = np.array([0.1, 0.2, 9.81])
        gyro  = np.array([0.01, 0.02, 0.03])
        ekf.predict(accel, gyro, dt=0.01)
        np.testing.assert_array_almost_equal(ekf.P, ekf.P.T, decimal=10)

    def test_zero_dt_no_change(self, ekf: EKF) -> None:
        """dt=0 ile predict çağrısı pozisyonu değiştirmemeli."""
        pos_before = ekf.get_position().copy()
        ekf.predict(np.zeros(3), np.zeros(3), dt=0.0)
        np.testing.assert_array_almost_equal(
            ekf.get_position(), pos_before, decimal=5
        )

    def test_predict_multiple_steps_accumulates(self, ekf: EKF) -> None:
        """Çok adımlı predict birikimli pozisyon üretmeli."""
        ekf.x[3:6] = [1.0, 0.0, 0.0]
        accel = np.array([0.0, 0.0, 9.81])
        gyro  = np.zeros(3)
        for _ in range(10):
            ekf.predict(accel, gyro, dt=0.1)
        assert ekf.get_position()[0] > 0.5, "10 adım sonra pozisyon birikmiş olmalı"


class TestEKFUpdate:
    def test_gps_update_corrects_position(self, ekf: EKF) -> None:
        """GPS update pozisyonu GPS'e doğru çekmeli."""
        ekf.x[0:3] = [1.0, 1.0, 0.0]
        gps_pos    = np.array([0.0, 0.0, 0.0])

        accepted = ekf.update_gps(gps_pos)

        assert accepted, "Geçerli GPS ölçümü kabul edilmeli"
        pos = ekf.get_position()
        assert np.linalg.norm(pos) < np.sqrt(2.0), "GPS update pozisyonu düzeltmeli"

    def test_gps_update_reduces_covariance(self, ekf: EKF) -> None:
        """GPS update sonrası kovaryans küçülmeli."""
        P_before = np.trace(ekf.P)
        ekf.update_gps(np.array([0.0, 0.0, 0.0]))
        P_after = np.trace(ekf.P)
        assert P_after < P_before, "Update sonrası kovaryans küçülmeli"

    def test_gps_outlier_rejected(self, ekf: EKF) -> None:
        """Aşırı büyük GPS sıçraması Mahalanobis ile reddedilmeli."""
        ekf.x[0:3] = [0.0, 0.0, 0.0]
        gps_pos    = np.array([9999.0, 9999.0, 9999.0])

        accepted = ekf.update_gps(gps_pos)
        assert not accepted, "Outlier GPS ölçümü reddedilmeli"

    def test_gps_update_with_velocity(self, ekf: EKF) -> None:
        """GPS hız bilgisiyle birlikte update çalışmalı."""
        gps_pos = np.array([0.5, 0.5, 0.0])
        gps_vel = np.array([0.1, 0.1, 0.0])
        accepted = ekf.update_gps(gps_pos, gps_vel)
        assert accepted, "Geçerli GPS+vel update kabul edilmeli"

    def test_covariance_positive_definite_after_update(self, ekf: EKF) -> None:
        """Update sonrası kovaryans pozitif tanımlı kalmalı."""
        ekf.update_gps(np.array([0.5, 0.5, 0.0]))
        eigenvalues = np.linalg.eigvalsh(ekf.P)
        assert np.all(eigenvalues > 0), "Update sonrası kovaryans pozitif tanımlı olmalı"

    def test_repeated_updates_converge(self, ekf: EKF) -> None:
        """Tekrarlı GPS update kovaryansı azaltmalı."""
        std_before = ekf.get_position_std().copy()
        for _ in range(20):
            ekf.update_gps(np.array([0.0, 0.0, 0.0]))
        std_after = ekf.get_position_std()
        assert np.all(std_after < std_before), "Tekrarlı update belirsizliği azaltmalı"


class TestSkewMatrix:
    def test_skew_antisymmetric(self, ekf: EKF) -> None:
        """Skew-symmetric matris: S + S^T = 0 olmalı."""
        v = np.array([1.0, 2.0, 3.0])
        S = ekf._skew(v)
        np.testing.assert_array_almost_equal(S + S.T, np.zeros((3, 3)))

    def test_skew_cross_product(self, ekf: EKF) -> None:
        """S(a) @ b == a × b olmalı."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(
            ekf._skew(a) @ b, np.cross(a, b)
        )

    def test_skew_diagonal_zero(self, ekf: EKF) -> None:
        """Skew-symmetric matrisin diyagonali sıfır olmalı."""
        v = np.array([5.0, 3.0, 1.0])
        S = ekf._skew(v)
        np.testing.assert_array_equal(np.diag(S), [0.0, 0.0, 0.0])


class TestEKFIntegration:
    def test_predict_then_update_cycle(self, ekf: EKF) -> None:
        """Predict → update döngüsü kararlı çalışmalı."""
        ekf.x[0:3] = [0.0, 0.0, 0.0]
        accel = np.array([0.0, 0.0, 9.81])
        gyro  = np.zeros(3)

        for i in range(100):
            ekf.predict(accel, gyro, dt=0.01)
            if i % 10 == 0:
                ekf.update_gps(np.array([0.0, 0.0, 0.0]))

        assert np.isfinite(ekf.get_position()).all(), "Pozisyon sonlu olmalı"
        assert np.isfinite(ekf.P).all(), "Kovaryans sonlu olmalı"

    def test_state_remains_finite(self, ekf: EKF) -> None:
        """Uzun süre çalışmada state sonsuz olmamalı."""
        accel = np.array([0.1, 0.0, 9.81])
        gyro  = np.array([0.01, 0.0, 0.0])

        for _ in range(500):
            ekf.predict(accel, gyro, dt=0.01)

        assert np.isfinite(ekf.x).all(), "500 adım sonra state sonlu olmalı"