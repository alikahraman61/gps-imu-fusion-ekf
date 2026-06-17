import numpy as np
import pytest
from src.allan_variance import allan_variance, identify_noise_params


@pytest.fixture
def white_noise_signal() -> tuple[np.ndarray, float, float]:
    """Bilinen white noise parametreli sentetik sinyal."""
    rng = np.random.default_rng(42)
    fs = 100.0
    n = int(fs * 3600)  # 1 saat
    t0 = 1.0 / fs
    true_wn = 0.01
    signal = rng.normal(0, true_wn / np.sqrt(t0), n)
    return signal, fs, true_wn


class TestAllanVariance:
    def test_output_shapes_match(self, white_noise_signal) -> None:
        """taus ve adev aynı uzunlukta olmalı."""
        signal, fs, _ = white_noise_signal
        taus, adev = allan_variance(signal, fs)
        assert len(taus) == len(adev)

    def test_taus_monotonic_increasing(self, white_noise_signal) -> None:
        """Averaging time'lar artan sırada olmalı."""
        signal, fs, _ = white_noise_signal
        taus, _ = allan_variance(signal, fs)
        assert np.all(np.diff(taus) > 0)

    def test_adev_positive(self, white_noise_signal) -> None:
        """Allan deviation her zaman pozitif olmalı."""
        signal, fs, _ = white_noise_signal
        _, adev = allan_variance(signal, fs)
        assert np.all(adev > 0)

    def test_white_noise_recovery(self, white_noise_signal) -> None:
        """White noise parametresi %15 hata payıyla geri çıkarılmalı."""
        signal, fs, true_wn = white_noise_signal
        taus, adev = allan_variance(signal, fs)
        tau1_idx = np.argmin(np.abs(taus - 1.0))
        measured = adev[tau1_idx]
        rel_err = abs(measured - true_wn) / true_wn
        assert rel_err < 0.15, f"White noise hatası çok yüksek: {rel_err:.1%}"

    def test_white_noise_slope(self, white_noise_signal) -> None:
        """White noise bölgesinde log-log eğim ≈ -0.5 olmalı."""
        signal, fs, _ = white_noise_signal
        taus, adev = allan_variance(signal, fs)
        # İlk yarıda (white noise hakimken) eğimi ölç
        half = len(taus) // 2
        log_tau = np.log10(taus[:half])
        log_adev = np.log10(adev[:half])
        slope = np.polyfit(log_tau, log_adev, 1)[0]
        assert -0.6 < slope < -0.4, f"Eğim -0.5 olmalı, ölçülen: {slope:.2f}"

    def test_identify_returns_all_params(self, white_noise_signal) -> None:
        """identify_noise_params tüm anahtarları döndürmeli."""
        signal, fs, _ = white_noise_signal
        taus, adev = allan_variance(signal, fs)
        params = identify_noise_params(taus, adev)
        assert "white_noise" in params
        assert "bias_instability" in params
        assert "rate_random_walk" in params