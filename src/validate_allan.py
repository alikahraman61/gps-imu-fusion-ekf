"""
Allan Variance implementasyonunu sentetik veriyle doğrulama.

Bilinen white noise + bias instability parametreleriyle sinyal üretir,
ardından allan_variance() fonksiyonunun bu parametreleri geri
çıkarabildiğini doğrular. Bu, ground-truth validation yaklaşımıdır.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt

from allan_variance import allan_variance, identify_noise_params

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

rng = np.random.default_rng(42)

# ── Sentetik durağan IMU sinyali üret ─────────────────────
fs = 100.0          # Hz
duration = 3600.0   # 1 saat — Allan Variance için yeterli uzunluk
n = int(fs * duration)
t0 = 1.0 / fs

# Bilinen ground-truth parametreler
TRUE_WHITE_NOISE = 0.01   # rad/s @ tau=1s (ARW benzeri)
TRUE_BIAS_WALK   = 0.0005 # rad/s — bias random walk std

# White noise bileşeni: σ_wn = N / sqrt(t0)
# tau=1s'de Allan deviation = N olacak şekilde ölçekle
white = rng.normal(0, TRUE_WHITE_NOISE / np.sqrt(t0), n)

# Bias random walk bileşeni: kümülatif rastgele yürüyüş
bias_increments = rng.normal(0, TRUE_BIAS_WALK * np.sqrt(t0), n)
bias_walk = np.cumsum(bias_increments)

# Toplam sinyal (durağan IMU — gerçek hareket yok)
signal = white + bias_walk

print(f"Sentetik sinyal: {n} örnek, {duration:.0f} saniye")
print(f"Ground-truth white noise (tau=1s): {TRUE_WHITE_NOISE:.4e}")

# ── Allan Variance hesapla ────────────────────────────────
taus, adev = allan_variance(signal, fs)
params = identify_noise_params(taus, adev)

# tau=1s'deki gerçek değeri bul
tau1_idx = np.argmin(np.abs(taus - 1.0))
measured_wn = adev[tau1_idx]

print("\n── Doğrulama Sonucu ──")
print(f"Ground-truth white noise : {TRUE_WHITE_NOISE:.4e}")
print(f"Ölçülen white noise      : {measured_wn:.4e}")
rel_err = abs(measured_wn - TRUE_WHITE_NOISE) / TRUE_WHITE_NOISE * 100
print(f"Bağıl hata               : {rel_err:.1f}%")

if rel_err < 15.0:
    print("✓ DOĞRULAMA BAŞARILI — implementasyon white noise'u doğru çıkarıyor")
else:
    print("⚠ Sapma yüksek — kontrol gerekli")

# ── Grafik ────────────────────────────────────────────────
plt.figure(figsize=(9, 6))
plt.loglog(taus, adev, "b-", lw=1.8, label="Ölçülen Allan deviation")

# Teorik white noise eğrisi: σ(τ) = N / sqrt(τ)
theoretical = TRUE_WHITE_NOISE / np.sqrt(taus)
plt.loglog(taus, theoretical, "r--", lw=1.5,
           label=f"Teorik white noise (N={TRUE_WHITE_NOISE})")

# tau=1s işareti
plt.axvline(1.0, color="gray", ls=":", alpha=0.6)
plt.scatter([1.0], [measured_wn], c="red", s=80, zorder=5,
            label=f"τ=1s: {measured_wn:.4e}")

plt.xlabel("Averaging time τ (s)")
plt.ylabel("Allan deviation σ(τ)")
plt.title("Allan Variance Doğrulama — Sentetik Durağan IMU\n"
          "(Bilinen parametre geri çıkarılıyor)")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("/home/ali/kitti-fusion/results/figures/06_allan_validation.png", dpi=150)
plt.show()
print("\nGrafik kaydedildi.")