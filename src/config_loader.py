from dataclasses import dataclass
import yaml
from pathlib import Path


@dataclass
class DataConfig:
    basepath: str
    date: str
    drive: str


@dataclass
class EKFConfig:
    accel_noise: float
    gyro_noise: float
    accel_bias_noise: float
    gyro_bias_noise: float
    gps_noise_pos: float
    gps_noise_vel: float
    init_cov: float
    mahalanobis_threshold: float


@dataclass
class SimulationConfig:
    gps_rate: int
    gps_denied_start: int
    gps_denied_end: int


@dataclass
class OutputConfig:
    figures_dir: str
    logs_dir: str


@dataclass
class Config:
    data: DataConfig
    ekf: EKFConfig
    simulation: SimulationConfig
    output: OutputConfig


def load_config(path: str = "config.yaml") -> Config:
    """YAML config dosyasını yükler ve Config dataclass'ına dönüştürür."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config dosyası bulunamadı: {path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    return Config(
        data=DataConfig(**raw["data"]),
        ekf=EKFConfig(**raw["ekf"]),
        simulation=SimulationConfig(**raw["simulation"]),
        output=OutputConfig(**raw["output"]),
    )