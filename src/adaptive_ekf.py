import logging
from collections import deque
import numpy as np
from numpy.typing import NDArray

try:
    from ekf import EKF
except ModuleNotFoundError:
    from src.ekf import EKF

logger = logging.getLogger(__name__)


class AdaptiveEKF(EKF):
    """
    Innovation-based Adaptive Extended Kalman Filter.

    Standart EKF'i genişleterek ölçüm gürültü kovaryansı R'yi online
    olarak günceller. Innovation sequence istatistiklerini kayan bir
    pencerede izler ve filtre tutarlılığını korumak için R'yi ayarlar.

    Yöntem
    ------
    Teorik innovation kovaryansı:  S = H P Hᵀ + R
    Gözlemlenen innovation kovaryansı (kayan pencere):
        Ĉ = (1/N) Σ νₖ νₖᵀ
    Eğer Ĉ teorik S'den sistematik olarak büyükse, filtre
    overconfident demektir; R aşağıdaki ilişkiyle güncellenir:
        R ← Ĉ - H P Hᵀ     (pozitif tanımlılık korunarak)

    Bu, Mehra (1970) ve Mohamed & Schwarz (1999) adaptif Kalman
    filtreleme yaklaşımlarının pratik bir uyarlamasıdır.

    Parameters
    ----------
    window_size : int
        Innovation istatistiklerinin hesaplandığı kayan pencere boyutu.
    **ekf_kwargs
        EKF taban sınıfına iletilen parametreler.

    References
    ----------
    Mehra, R. (1970). "On the identification of variances and adaptive
    Kalman filtering." IEEE Trans. Automatic Control.
    Mohamed, A.H. & Schwarz, K.P. (1999). "Adaptive Kalman filtering
    for INS/GPS." Journal of Geodesy.
    """

    def __init__(self, window_size: int = 30, **ekf_kwargs) -> None:
        super().__init__(**ekf_kwargs)
        self.window_size = window_size
        self._innovations: deque = deque(maxlen=window_size)
        # Adaptif R'nin başlangıç değeri (taban GPS gürültüsü)
        self._adaptive_R_pos = self.gps_noise_pos ** 2
        # İzleme için geçmiş
        self.R_history: list[float] = []

    def update_gps_adaptive(
        self,
        gps_pos: NDArray[np.float64],
    ) -> bool:
        """
        Adaptif R ile GPS pozisyon güncellemesi.

        Innovation izlenir, R kayan pencere istatistiğiyle güncellenir,
        ardından standart EKF update uygulanır.

        Parameters
        ----------
        gps_pos : NDArray[np.float64]
            GPS pozisyon ölçümü [e, n, u] (m).

        Returns
        -------
        bool
            Update kabul edildiyse True, outlier ise False.
        """
        H = np.zeros((3, 15))
        H[0:3, 0:3] = np.eye(3)

        z      = gps_pos
        z_pred = self.x[0:3]
        innovation = z - z_pred

        # Innovation'ı pencereye ekle
        self._innovations.append(innovation.copy())

        # Adaptif R güncellemesi (yeterli örnek varsa)
        if len(self._innovations) >= self.window_size:
            innov_arr = np.array(self._innovations)  # (N, 3)
            # Gözlemlenen innovation kovaryansı
            C_hat = (innov_arr.T @ innov_arr) / len(self._innovations)
            # Teorik tahmin kısmı H P Hᵀ
            HPHt = H @ self.P @ H.T
            # R ← Ĉ - H P Hᵀ (diyagonal, pozitif tutulur)
            R_diag = np.maximum(np.diag(C_hat - HPHt), 0.1)
            self._adaptive_R_pos = float(np.mean(R_diag))

        self.R_history.append(np.sqrt(self._adaptive_R_pos))

        R_noise = np.eye(3) * self._adaptive_R_pos

        # Mahalanobis outlier kontrolü (adaptif R ile)
        S    = H @ self.P @ H.T + R_noise
        dist = float(innovation @ np.linalg.inv(S) @ innovation)
        if dist > self.mahalanobis_threshold:
            logger.warning(
                "Adaptive GPS outlier rejected — distance: %.2f", dist
            )
            return False

        # Standart Kalman update
        K  = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ innovation
        self.x[0:3]   += dx[0:3]
        self.x[3:6]   += dx[3:6]
        self.x[10:13] += dx[9:12]
        self.x[13:16] += dx[12:15]

        IKH    = np.eye(15) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_noise @ K.T

        return True