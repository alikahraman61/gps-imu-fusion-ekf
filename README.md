# GPS/IMU Sensor Fusion with Extended Kalman Filter

Tightly-coupled GPS + IMU fusion pipeline using a 15-state EKF.  
Evaluated on the KITTI Raw Dataset (drive 0034).

## Results

| Method     | ATE RMSE | RPE RMSE |
|------------|----------|----------|
| Raw GPS    | baseline | baseline |
| EKF Fusion | 4.69 m   | 0.58 m   |
| GPS-Denied | 1109 m   | 28.8 m   |

> EKF achieves **0.58m relative error** over 127 seconds of urban driving.  
> Without GPS, IMU-only drift reaches **386m in 30 seconds**.

## Key Features

- 15-state EKF: position, velocity, quaternion, accel bias, gyro bias
- WGS84 → ENU coordinate transformation
- Mahalanobis outlier rejection for GPS measurements
- GPS-denied scenario simulation with drift analysis
- ATE / RPE evaluation metrics

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python src/main.py        # EKF + GPS-denied comparison
python src/evaluation.py  # ATE / RPE metrics
```

## Dataset

[KITTI Raw Data](http://www.cvlibs.net/datasets/kitti/raw_data.php)  
Sequence: `2011_09_30_drive_0034`

## Tech Stack

Python · NumPy · SciPy · pykitti · GeographicLib · Matplotlib