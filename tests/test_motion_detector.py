import numpy as np
import pytest
from src.motion_detector import (
    detect_stationary_imu,
    detect_stationary_gps_aided,
    GRAVITY,
)


@pytest.fixture
def stationary_window() -> tuple[np.ndarray, np.ndarray]:
    """Duran araç: accel ≈ g, düşük varyans, gyro ≈ 0."""
    rng = np.random.default_rng(0)
    accel = np.tile([0.0, 0.0, GRAVITY], (10, 1)) + rng.normal(0, 0.01, (10, 3))
    gyro  = rng.normal(0, 0.001, (10, 3))
    return accel, gyro


@pytest.fixture
def moving_window() -> tuple[np.ndarray, np.ndarray]:
    """Manevra yapan araç: yüksek varyans, dönüş var."""
    rng = np.random.default_rng(1)
    accel = np.tile([2.0, 1.0, GRAVITY], (10, 1)) + rng.normal(0, 1.0, (10, 3))
    gyro  = np.tile([0.0, 0.0, 0.3], (10, 1)) + rng.normal(0, 0.05, (10, 3))
    return accel, gyro


class TestIMUDetector:
    def test_detects_stationary(self, stationary_window) -> None:
        """Duran araç hareketsiz olarak tespit edilmeli."""
        accel, gyro = stationary_window
        assert detect_stationary_imu(accel, gyro) is True

    def test_detects_moving(self, moving_window) -> None:
        """Manevra yapan araç hareketli olarak tespit edilmeli."""
        accel, gyro = moving_window
        assert detect_stationary_imu(accel, gyro) is False

    def test_constant_velocity_false_positive(self) -> None:
        """KISIT: Sabit hız durma sanılır (bilinen IMU-only zaafı)."""
        rng = np.random.default_rng(2)
        # Sabit hızda düz gidiş: net ivme yok, accel ≈ g, gyro ≈ 0
        accel = np.tile([0.0, 0.0, GRAVITY], (10, 1)) + rng.normal(0, 0.01, (10, 3))
        gyro  = rng.normal(0, 0.001, (10, 3))
        # IMU-only bunu yanlışlıkla "durma" olarak işaretler
        assert detect_stationary_imu(accel, gyro) is True


class TestGPSAidedDetector:
    def test_detects_true_stop(self, stationary_window) -> None:
        """Gerçek durma (hız≈0) tespit edilmeli."""
        accel, gyro = stationary_window
        assert detect_stationary_gps_aided(accel, gyro, speed=0.1) is True

    def test_rejects_constant_velocity(self, stationary_window) -> None:
        """Sabit hız (IMU durgun ama hız yüksek) reddedilmeli."""
        accel, gyro = stationary_window
        # IMU durgun görünüyor ama GPS hızı 10 m/s
        assert detect_stationary_gps_aided(accel, gyro, speed=10.0) is False

    def test_speed_threshold_boundary(self, stationary_window) -> None:
        """Hız eşiği sınırında doğru karar vermeli."""
        accel, gyro = stationary_window
        assert detect_stationary_gps_aided(accel, gyro, speed=0.4,
                                           speed_thresh=0.5) is True
        assert detect_stationary_gps_aided(accel, gyro, speed=0.6,
                                           speed_thresh=0.5) is False