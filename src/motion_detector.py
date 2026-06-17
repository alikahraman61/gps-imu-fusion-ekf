import logging
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Yerçekimi büyüklüğü (m/s²)
GRAVITY = 9.81


def detect_stationary_imu(
    accel_window: NDArray[np.float64],
    gyro_window: NDArray[np.float64],
    accel_var_thresh: float = 0.15,
    gyro_thresh: float = 0.03,
    gravity_tol: float = 0.5,
) -> bool:
    """
    IMU-only hareketsizlik tespiti (varyans-tabanlı).

    Üç koşulu birlikte kontrol eder:
    1. Accelerometer büyüklük varyansı düşük (titreşim yok)
    2. Accelerometer ortalaması ≈ yerçekimi (net ivme yok)
    3. Gyroscope ortalaması düşük (dönüş yok)

    TEMEL KISIT: Accelerometer hızı değil ivmeyi ölçer. Sabit hızla
    giden bir araçta da net ivme sıfırdır, dolayısıyla bu yöntem
    "durma" ile "sabit hız"ı ayırt EDEMEZ. Bu, IMU-only ZUPT'un
    bilinen bir sınırlılığıdır.

    Parameters
    ----------
    accel_window : NDArray[np.float64]
        Accelerometer penceresi, şekil (W, 3).
    gyro_window : NDArray[np.float64]
        Gyroscope penceresi, şekil (W, 3).
    accel_var_thresh : float
        Accel norm varyans eşiği ((m/s²)²).
    gyro_thresh : float
        Gyro norm ortalama eşiği (rad/s).
    gravity_tol : float
        Accel ortalamasının g'den sapma toleransı (m/s²).

    Returns
    -------
    bool
        True if detected stationary, False otherwise.
    """
    accel_mag = np.linalg.norm(accel_window, axis=1)
    gyro_mag = np.linalg.norm(gyro_window, axis=1)

    cond_low_vibration = float(np.var(accel_mag)) < accel_var_thresh
    cond_gravity_only = abs(float(np.mean(accel_mag)) - GRAVITY) < gravity_tol
    cond_no_rotation = float(np.mean(gyro_mag)) < gyro_thresh

    return bool(cond_low_vibration and cond_gravity_only and cond_no_rotation)


def detect_stationary_gps_aided(
    accel_window: NDArray[np.float64],
    gyro_window: NDArray[np.float64],
    speed: float,
    speed_thresh: float = 0.5,
    accel_var_thresh: float = 0.15,
    gyro_thresh: float = 0.03,
    gravity_tol: float = 0.5,
) -> bool:
    """
    GPS-aided hareketsizlik tespiti.

    IMU-only koşullarına ek olarak bağımsız bir hız kaynağı (GPS,
    tekerlek odometresi veya CAN bus) kullanır. Hız eşiği, sabit hız
    ile gerçek durmayı ayırt eden kritik koşuldur ve IMU-only
    yöntemin temel kısıtını ortadan kaldırır.

    Parameters
    ----------
    accel_window : NDArray[np.float64]
        Accelerometer penceresi, şekil (W, 3).
    gyro_window : NDArray[np.float64]
        Gyroscope penceresi, şekil (W, 3).
    speed : float
        Bağımsız kaynaktan ölçülen hız büyüklüğü (m/s).
    speed_thresh : float
        Durma için hız eşiği (m/s).
    accel_var_thresh : float
        Accel norm varyans eşiği ((m/s²)²).
    gyro_thresh : float
        Gyro norm ortalama eşiği (rad/s).
    gravity_tol : float
        Accel ortalamasının g'den sapma toleransı (m/s²).

    Returns
    -------
    bool
        True if detected stationary, False otherwise.
    """
    imu_stationary = detect_stationary_imu(
        accel_window, gyro_window,
        accel_var_thresh, gyro_thresh, gravity_tol
    )
    speed_stationary = speed < speed_thresh
    return bool(imu_stationary and speed_stationary)