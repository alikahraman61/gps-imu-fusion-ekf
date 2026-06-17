import logging
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def allan_variance(
    data: NDArray[np.float64],
    fs: float,
    max_clusters: int = 100,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute the Allan variance of a time series.

    The Allan variance characterizes noise processes in inertial sensors
    by averaging the signal over clusters of increasing length and measuring
    the variance between adjacent cluster averages.

    Parameters
    ----------
    data : NDArray[np.float64]
        1D time series of sensor measurements (e.g. gyro rate or accel).
    fs : float
        Sampling frequency (Hz).
    max_clusters : int
        Number of averaging-time points (log-spaced) to evaluate.

    Returns
    -------
    taus : NDArray[np.float64]
        Averaging times (s).
    adev : NDArray[np.float64]
        Allan deviation at each averaging time.

    References
    ----------
    IEEE Std 952-1997, "IEEE Standard Specification Format Guide and Test
    Procedure for Single-Axis Interferometric Fiber Optic Gyros", Annex C.
    """
    n = len(data)
    t0 = 1.0 / fs

    # Maksimum cluster boyutu (en fazla N/2)
    max_m = int(np.floor(n / 2))
    # Log-spaced cluster boyutları
    m_values = np.unique(
        np.logspace(0, np.log10(max_m), max_clusters).astype(int)
    )
    m_values = m_values[m_values >= 1]

    taus = m_values * t0
    adev = np.zeros(len(m_values))

    # Kümülatif toplam — entegrasyon (açı/hız sinyaline dönüşüm)
    theta = np.cumsum(data) * t0

    for idx, m in enumerate(m_values):
        # Cluster ortalamaları arası fark
        # theta[i+2m] - 2*theta[i+m] + theta[i]
        if (n - 2 * m) < 1:
            adev[idx] = np.nan
            continue

        diffs = theta[2 * m:] - 2 * theta[m:-m] + theta[:-2 * m]
        avar = np.sum(diffs ** 2) / (2 * m ** 2 * t0 ** 2 * (n - 2 * m))
        adev[idx] = np.sqrt(avar)

    valid = ~np.isnan(adev)
    return taus[valid], adev[valid]


def identify_noise_params(
    taus: NDArray[np.float64],
    adev: NDArray[np.float64],
) -> dict[str, float]:
    """
    Identify noise parameters from an Allan deviation curve.

    Extracts the standard inertial-sensor noise coefficients by fitting
    the characteristic slopes of the log-log Allan deviation curve.

    Parameters
    ----------
    taus : NDArray[np.float64]
        Averaging times (s).
    adev : NDArray[np.float64]
        Allan deviation values.

    Returns
    -------
    dict[str, float]
        Dictionary with:
        - 'white_noise' : random walk coefficient (slope -1/2, read at tau=1s)
        - 'bias_instability' : bias instability (flat minimum / 0.664)
        - 'rate_random_walk' : rate random walk (slope +1/2, read at tau=3s)
    """
    log_tau = np.log10(taus)
    log_adev = np.log10(adev)

    # Lokal eğim (numerik türev)
    slopes = np.gradient(log_adev, log_tau)

    params: dict[str, float] = {}

    # ── White noise: slope ≈ -0.5 ──────────────────────────
    # tau=1s'deki Allan deviation değeri
    wn_idx = np.argmin(np.abs(slopes - (-0.5)))
    # tau=1s'e en yakın noktaya extrapolate
    tau1_idx = np.argmin(np.abs(taus - 1.0))
    params["white_noise"] = float(adev[tau1_idx])

    # ── Bias instability: slope ≈ 0 (minimum) ──────────────
    bi_idx = np.argmin(adev)
    # Scallop faktörü 0.664 (white noise + flicker için)
    params["bias_instability"] = float(adev[bi_idx] / 0.664)

    # ── Rate random walk: slope ≈ +0.5 ─────────────────────
    rrw_candidates = np.where(slopes > 0.3)[0]
    if len(rrw_candidates) > 0:
        rrw_idx = rrw_candidates[0]
        # tau=3s'e normalize: K = adev * sqrt(3/tau)
        params["rate_random_walk"] = float(
            adev[rrw_idx] * np.sqrt(3.0 / taus[rrw_idx])
        )
    else:
        params["rate_random_walk"] = 0.0

    logger.info(
        "Noise params — WN: %.2e, BI: %.2e, RRW: %.2e",
        params["white_noise"],
        params["bias_instability"],
        params["rate_random_walk"],
    )
    return params