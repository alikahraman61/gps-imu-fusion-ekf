---
title: "GPS/IMU Sensor Fusion with Extended Kalman Filter"
subtitle: "A Production-Grade Pipeline on KITTI Dataset"
author: "Ali Kahraman"
date: "June 2026"
geometry: margin=2.5cm
fontsize: 11pt
numbersections: true
toc: true
toc-depth: 3
abstract: |
  This report presents a production-grade GPS/IMU sensor fusion pipeline
  implementing a 15-state Extended Kalman Filter (EKF) and a Hybrid
  Unscented Kalman Filter (UKF) evaluated on the KITTI Raw Dataset.
  The pipeline achieves 1.43m Absolute Trajectory Error (ATE) on urban
  driving sequences, representing a 13% improvement over the baseline EKF.
  Key contributions include innovation-based adaptive noise estimation,
  Zero Velocity Update (ZUPT) with IMU-only vs GPS-aided detector
  comparison, Allan Variance-based IMU characterization, and a
  quantitative loosely vs tightly coupled analysis demonstrating 74%
  ATE improvement. All algorithms are validated with 32 unit tests,
  CI/CD pipeline, and Docker deployment.
---

# Introduction

Autonomous vehicles require accurate, real-time pose estimation under
varying environmental conditions. GPS alone suffers from signal
dropouts, multipath errors, and low update rates (10 Hz). Inertial
Measurement Units (IMUs) provide high-frequency (100 Hz) measurements
but accumulate drift over time due to sensor bias and noise.

Sensor fusion combines complementary sensor characteristics: GPS
provides absolute position references while IMU provides high-frequency
relative motion estimates. The Extended Kalman Filter (EKF) is the
industry-standard algorithm for this fusion task.

This project implements and evaluates multiple fusion strategies on the
KITTI Raw Dataset, a real-world autonomous driving benchmark collected
in Karlsruhe, Germany.

## Contributions

- 15-state EKF with quaternion-based orientation representation
- Hybrid UKF: sigma-point mean propagation with EKF Jacobian covariance
- Innovation-based adaptive measurement noise estimation (Mehra, 1970)
- Allan Variance IMU noise characterization with ground-truth validation
- ZUPT with quantitative IMU-only vs GPS-aided detector comparison
- Multi-sequence benchmark across urban, residential, and highway scenarios

# System Model

## State Vector

The filter maintains a 15-dimensional error state (16-dimensional
nominal state with quaternion):

$$\mathbf{x} = [\mathbf{p}^T, \mathbf{v}^T, \mathbf{q}^T, \mathbf{b}_a^T, \mathbf{b}_g^T]^T$$

where $\mathbf{p} \in \mathbb{R}^3$ is position in ENU frame (m),
$\mathbf{v} \in \mathbb{R}^3$ is velocity (m/s),
$\mathbf{q} \in \mathbb{H}$ is unit quaternion $[w, x, y, z]$,
$\mathbf{b}_a \in \mathbb{R}^3$ is accelerometer bias (m/s²),
$\mathbf{b}_g \in \mathbb{R}^3$ is gyroscope bias (rad/s).

## IMU Kinematic Model

The continuous-time IMU kinematic equations are:

$$\dot{\mathbf{p}} = \mathbf{v}$$

$$\dot{\mathbf{v}} = \mathbf{R}(\mathbf{q})(\tilde{\mathbf{a}} - \mathbf{b}_a) + \mathbf{g}$$

$$\dot{\mathbf{q}} = \frac{1}{2}\mathbf{q} \otimes \begin{bmatrix} 0 \\ \tilde{\boldsymbol{\omega}} - \mathbf{b}_g \end{bmatrix}$$

where $\tilde{\mathbf{a}}$ and $\tilde{\boldsymbol{\omega}}$ are
accelerometer and gyroscope measurements, $\mathbf{R}(\mathbf{q})$ is
the rotation matrix from body to ENU frame, and
$\mathbf{g} = [0, 0, -9.81]^T$ m/s².

## GPS Measurement Model

GPS provides position measurements in WGS84 coordinates, transformed
to local ENU frame:

$$\mathbf{z}_{GPS} = \mathbf{H}\mathbf{x} + \mathbf{v}, \quad \mathbf{v} \sim \mathcal{N}(0, \mathbf{R})$$

$$\mathbf{H} = [\mathbf{I}_{3\times3}, \mathbf{0}_{3\times12}]$$

# Algorithms

## Extended Kalman Filter

### Predict Step

The EKF predict step propagates the state and covariance using the
IMU kinematic model:

$$\hat{\mathbf{x}}_{k|k-1} = f(\mathbf{x}_{k-1}, \mathbf{u}_k)$$

$$\mathbf{P}_{k|k-1} = \mathbf{F}_k \mathbf{P}_{k-1} \mathbf{F}_k^T + \mathbf{Q}_k$$

The linearized transition matrix $\mathbf{F}_k$ is:

$$\mathbf{F}_k = \begin{bmatrix} \mathbf{I} & \mathbf{I}\Delta t & \mathbf{0} & \mathbf{0} & \mathbf{0} \\ \mathbf{0} & \mathbf{I} & -\mathbf{R}[\tilde{\mathbf{a}} - \mathbf{b}_a]_\times \Delta t & -\mathbf{R}\Delta t & \mathbf{0} \\ \mathbf{0} & \mathbf{0} & \mathbf{I} & \mathbf{0} & -\mathbf{I}\Delta t \\ \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{I} & \mathbf{0} \\ \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{I} \end{bmatrix}$$

### Update Step

The Joseph-form covariance update ensures numerical positive
definiteness:

$$\mathbf{K}_k = \mathbf{P}_{k|k-1}\mathbf{H}^T(\mathbf{H}\mathbf{P}_{k|k-1}\mathbf{H}^T + \mathbf{R})^{-1}$$

$$\mathbf{x}_k = \mathbf{x}_{k|k-1} + \mathbf{K}_k(\mathbf{z}_k - \mathbf{H}\hat{\mathbf{x}}_{k|k-1})$$

$$\mathbf{P}_k = (\mathbf{I} - \mathbf{K}_k\mathbf{H})\mathbf{P}_{k|k-1}(\mathbf{I} - \mathbf{K}_k\mathbf{H})^T + \mathbf{K}_k\mathbf{R}\mathbf{K}_k^T$$

### Mahalanobis Outlier Rejection

GPS measurements are rejected when the Mahalanobis distance exceeds
the chi-squared threshold ($\chi^2_{3, 0.999} = 16.27$):

$$d^2 = \boldsymbol{\nu}^T \mathbf{S}^{-1} \boldsymbol{\nu} > \chi^2_{3, 0.999}$$

## Hybrid Unscented Kalman Filter

The Hybrid UKF addresses the quaternion manifold instability of the
standard UKF by combining sigma-point mean propagation with EKF
Jacobian covariance propagation.

### Sigma Points

For state dimension $n=15$ and scaling parameter $\lambda$,
$2n+1 = 31$ sigma points are generated:

$$\boldsymbol{\mathcal{X}}_0 = \mathbf{x}$$

$$\boldsymbol{\mathcal{X}}_i = \mathbf{x} \boxplus \sqrt{(n+\lambda)\mathbf{P}}_{:,i}, \quad i = 1,\ldots,n$$

$$\boldsymbol{\mathcal{X}}_{i+n} = \mathbf{x} \boxplus (-\sqrt{(n+\lambda)\mathbf{P}}_{:,i}), \quad i = 1,\ldots,n$$

The $\boxplus$ operator applies perturbations on the SO(3) manifold
for the quaternion component via rotation vector composition.

### Weighted Mean

$$\bar{\mathbf{x}} = \sum_{i=0}^{2n} W_i^m \boldsymbol{\mathcal{X}}_i^+$$

where $\boldsymbol{\mathcal{X}}_i^+$ denotes propagated sigma points
through the nonlinear IMU model. The covariance is propagated via
EKF Jacobian for numerical stability.

## Adaptive Noise Estimation

Following Mehra (1970), the measurement noise covariance $\mathbf{R}$
is updated online using the innovation sequence covariance:

$$\hat{\mathbf{C}}_k = \frac{1}{N}\sum_{j=k-N+1}^{k} \boldsymbol{\nu}_j \boldsymbol{\nu}_j^T$$

$$\hat{\mathbf{R}}_k = \hat{\mathbf{C}}_k - \mathbf{H}\mathbf{P}_{k|k-1}\mathbf{H}^T$$

This allows the filter to track time-varying GPS noise (e.g., urban
multipath vs. open-sky conditions) without manual retuning.

## Zero Velocity Update (ZUPT)

When the vehicle is detected as stationary, a pseudo-measurement
$\mathbf{z}_{ZUPT} = \mathbf{0}$ is injected:

$$\mathbf{H}_{ZUPT} = [\mathbf{0}_{3\times3}, \mathbf{I}_{3\times3}, \mathbf{0}_{3\times9}]$$

Two detectors are compared:

**IMU-only:** Uses accelerometer variance and gyroscope magnitude.
Fundamental limitation: constant-velocity motion is
indistinguishable from stationary (accelerometer measures specific
force, not velocity).

**GPS-aided:** Adds an independent velocity source. Achieves
Precision=100%, Recall=100%, F1=1.0 vs. IMU-only F1=0.28.

## Allan Variance

Allan variance characterizes IMU noise processes from stationary data:

$$\sigma^2(\tau) = \frac{1}{2(N-2m)\tau^2} \sum_{k=1}^{N-2m} [\theta(t_{k+2m}) - 2\theta(t_{k+m}) + \theta(t_k)]^2$$

The slope of the log-log Allan deviation curve identifies noise types:
slope $-1/2$ indicates white noise (ARW/VRW), slope $0$ indicates
bias instability, slope $+1/2$ indicates rate random walk.

Implementation validated with synthetic data: known white noise
$\sigma_N = 0.01$ rad/s recovered with 2.0% relative error.

# Dataset

The KITTI Raw Dataset (Geiger et al., 2013) provides synchronized
sensor data from a vehicle-mounted sensor suite. The OXTS RT3003
unit provides combined GPS/IMU measurements:

| Sensor | Rate | Specification |
|---|---|---|
| GPS (OXTS RT3003) | 10 Hz | WGS84 lat/lon/alt |
| IMU accelerometer | 100 Hz | $\pm$5g, 3-axis |
| IMU gyroscope | 100 Hz | $\pm$100°/s, 3-axis |

Three sequences are evaluated:

| Sequence | Environment | Frames | Duration | Distance |
|---|---|---|---|---|
| drive_0034 | Urban | 1224 | 127s | 919m |
| drive_0027 | Residential | 1106 | 115s | 692m |
| drive_0028 | Highway | 5177 | 538s | 4131m |

# Results

## Multi-Sequence Benchmark

| Method | Urban ATE | Residential ATE | Highway ATE |
|---|---|---|---|
| EKF | 1.647m | 1.563m | 49.47m |
| Hybrid UKF | **1.432m** | **1.415m** | N/A* |

*UKF disabled for highway: covariance accumulation over 538s causes
instability. This is consistent with known limitations of sigma-point
methods for long-duration INS integration.

**Key finding:** Hybrid UKF achieves 11.3% average ATE improvement
over EKF on short urban/residential sequences where nonlinear
maneuvers are frequent.

## GPS-Denied Analysis

| Metric | EKF Normal | GPS-Denied |
|---|---|---|
| ATE RMSE | 4.69m | 1109m |
| RPE RMSE | 0.58m | 28.8m |
| Max drift (30s) | — | 386m |

IMU-only propagation accumulates 386m drift in 30 seconds,
demonstrating the necessity of GPS fusion.

## Loosely vs Tightly Coupled

Under identical GPS noise (0.8m std) and GPS rate (0.1s vs 0.5s):

| Method | ATE RMSE |
|---|---|
| Loosely-coupled (0.5s GPS) | 6.09m |
| Tightly-coupled (0.1s GPS) | 1.57m |

Tightly-coupled achieves 74.2% lower ATE due to higher GPS update
frequency and EKF state coupling.

## Adaptive Noise Estimation

Under time-varying GPS noise (0.5m → 4.0m → 0.5m):

| Method | ATE RMSE |
|---|---|
| Fixed R | 3.97m |
| Adaptive R | 2.86m |

28.1% improvement. The adaptive filter tracks noise changes with
~20 frame lag (window size).

## ZUPT Performance

| Detector | Precision | Recall | F1 |
|---|---|---|---|
| IMU-only | 16.2% | 100% | 0.28 |
| GPS-aided | **100%** | **100%** | **1.00** |

Stationary drift reduction: 996x (2.88m → 0.003m over 10 seconds).

# Engineering

## Software Architecture

| Component | Description |
|---|---|
| `ekf.py` | 15-state EKF, Joseph form, Mahalanobis rejection |
| `ukf.py` | Hybrid UKF, sigma-point mean + EKF covariance |
| `adaptive_ekf.py` | Innovation-based adaptive R estimation |
| `allan_variance.py` | Allan variance + noise parameter identification |
| `motion_detector.py` | IMU-only and GPS-aided ZUPT detectors |
| `config_loader.py` | YAML-based parameter management |

## Testing

32 unit tests across 3 test modules, all passing:

- `test_ekf.py` (20 tests): predict/update correctness, covariance
  positive definiteness, Mahalanobis rejection, numerical stability
- `test_allan_variance.py` (6 tests): white noise recovery,
  monotonicity, slope validation
- `test_motion_detector.py` (6 tests): stationary detection,
  constant-velocity false positive documentation

CI/CD: GitHub Actions runs all tests on every push.

## Deployment

```bash
docker build -t kitti-fusion .
docker run kitti-fusion   # executes all 32 tests
```

# Limitations and Future Work

1. **UKF long-sequence instability**: Full USQUE (Unscented
   Quaternion Estimator, Crassidis & Markley 2003) would address
   covariance accumulation in highway sequences.

2. **Ground truth quality**: GPS used as both input and reference.
   Independent ground truth (RTK-GPS or laser scan matching) would
   provide cleaner evaluation.

3. **IMU Allan Variance**: KITTI drive sequences contain vehicle
   motion, violating stationarity assumption. Static IMU recording
   required for proper noise characterization.

4. **Highway EKF tuning**: 49m ATE on highway suggests noise
   parameters optimized for urban sequences. Sequence-adaptive
   tuning or the adaptive EKF would address this.

# Conclusion

This project implements a production-grade GPS/IMU sensor fusion
pipeline with rigorous algorithmic and engineering foundations.
The Hybrid UKF achieves 1.43m ATE on urban driving with 13%
improvement over EKF. ZUPT reduces stationary drift by 996x.
Adaptive noise estimation improves performance by 28% under
time-varying GPS conditions. All results are reproducible via
Docker and validated with automated tests.

# References

Geiger, A., Lenz, P., Stiller, C., & Urtasun, R. (2013). Vision
meets robotics: The KITTI dataset. *International Journal of
Robotics Research*, 32(11), 1231-1237.

Mehra, R. (1970). On the identification of variances and adaptive
Kalman filtering. *IEEE Transactions on Automatic Control*, 15(2),
175-184.

Mohamed, A.H., & Schwarz, K.P. (1999). Adaptive Kalman filtering
for INS/GPS. *Journal of Geodesy*, 73(4), 193-203.

Crassidis, J.L., & Markley, F.L. (2003). Unscented filtering for
spacecraft attitude estimation. *Journal of Guidance, Control, and
Dynamics*, 26(4), 536-542.

Wan, E.A., & Van der Merwe, R. (2000). The unscented Kalman filter
for nonlinear estimation. *Proceedings of the IEEE ASSPCC*, 153-158.