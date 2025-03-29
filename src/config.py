# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Dict, Tuple
from math import radians, degrees
import json
import os

# ======================
# Vehicle Configuration
# ======================
@dataclass
class VehicleConfig:
    # 기본 차량 설정 (2024 Model 3 Long Range AWD)
    LENGTH: float = 4.72           # 차량 길이 [m]
    WIDTH: float = 1.935           # 차량 너비 [m]
    HEIGHT: float = 1.44           # 차량 높이 [m]
    WHEELBASE: float = 2.875       # 축간거리 [m]
    TRACK: float = 1.585           # 트레드 [m]
    MASS: float = 1830.0           # 질량 [kg]
    INERTIA: float = 2500.0        # 관성 모멘트 [kg·m²]
    MAX_STEER: float = radians(35) # 최대 조향각 [rad]
    CG_HEIGHT: float = 0.55        # 무게중심 높이 [m]
    TIRE_WIDTH: float = 0.235      # 타이어 너비 [m]
    TIRE_HEIGHT: float = 0.669     # 타이어 높이 [m]
    TIRE_RADIUS: float = 0.334     # 타이어 반경 [m]

    # Powertrain Parameters
    MAX_ACCEL: float = 4.0         # 최대 가속도 [m/s²]
    MAX_BRAKE: float = 6.0         # 최대 제동 [m/s²]
    MAX_SPEED: float = 65.0        # 최대 속도 [m/s] (약 234 km/h)
    MIN_SPEED: float = -20.0       # 최소 속도 [m/s] (약 -72 km/h)

    # Tire Parameters (Pacejka Magic Formula)
    TIRE_B: float = 10.0           # Stiffness Factor
    TIRE_C: float = 1.9            # Shape Factor
    TIRE_D: float = 1.0            # Peak Factor
    TIRE_E: float = 0.97           # Curvature Factor

    @classmethod
    def from_json(cls, json_path: str) -> 'VehicleConfig':
        """JSON 파일에서 차량 설정 로드"""
        if not os.path.exists(json_path):
            print(f"설정 파일이 없습니다: {json_path}. 기본 설정을 사용합니다.")
            return cls()

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 라디안 변환이 필요한 필드
            radian_fields = ['MAX_STEER']
            for field in radian_fields:
                if field in data:
                    data[field] = radians(data[field])

            return cls(**data)
        except Exception as e:
            print(f"설정 파일 로드 오류: {e}. 기본 설정을 사용합니다.")
            return cls()

    def to_json(self, json_path: str) -> bool:
        """차량 설정을 JSON 파일로 저장"""
        try:
            data = self.__dict__.copy()

            # 도 단위로 변환이 필요한 필드
            degree_fields = ['MAX_STEER']
            for field in degree_fields:
                if field in data:
                    data[field] = degrees(data[field])

            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            return True
        except Exception as e:
            print(f"설정 파일 저장 오류: {e}")
            return False

# ======================
# Simulation Constants
# ======================
@dataclass
class SimConfig:
    FPS: int = 60
    SCALE: float = 5.0             # 1pixel당 미터 단위
    SIM_WIDTH: int = 1300
    SIM_HEIGHT: int = 800
    SIM_SIZE: tuple = field(default_factory=lambda: (1300, 800))  # 이제 명시적으로 계산

    # 색상 설정
    BACKGROUND_COLOR: tuple = (0, 0, 0)       # 배경 (검은색)
    GRID_COLOR: tuple = (100, 100, 100)       # 그리드 (회색)
    VEHICLE_COLOR: tuple = (255, 255, 255)    # 차량 몸체 (흰색)
    TIRE_COLOR: tuple = (255, 255, 255)       # 바퀴 (흰색)
    MARK_COLOR: tuple = (80, 80, 80)          # 타이어 자국 (회색)
    HUD_BG_COLOR: tuple = (50, 50, 50, 180)   # HUD 바탕 (반투명 회색)
    HUD_FG_COLOR: tuple = (255, 255, 0)       # HUD 글씨 (노란색)

    # 물리 상수
    GRAVITY: float = 9.81          # 중력가속도 [m/s²]
    AIR_DENSITY: float = 1.225     # 공기 밀도 [kg/m³]
    DRAG_COEFF: float = 0.30       # 공기저항 계수
    ROLL_RESIST: float = 0.015     # 구름저항 계수
    PHYSICS_SUBSTEPS: int = 4      # 물리 시뮬레이션 세부 단계

    # 시뮬레이션 환경 설정
    ENABLE_DEBUG_INFO: bool = True   # 디버그 정보 표시
    CAMERA_FOLLOW: bool = True       # 카메라가 차량 추적

    # 지형 설정
    TERRAIN_FRICTION: Dict[str, float] = field(default_factory=lambda: {
        "asphalt": 1.0,    # 아스팔트 (기준)
        "gravel": 0.8,     # 자갈
        "grass": 0.7,      # 잔디
        "dirt": 0.6,       # 흙
        "snow": 0.3,       # 눈
        "ice": 0.1         # 얼음
    })

    def __post_init__(self):
        """초기화 후 처리: SIM_SIZE가 튜플 형태로 제대로 설정되도록 함"""
        if not isinstance(self.SIM_SIZE, tuple) or len(self.SIM_SIZE) != 2:
            self.SIM_SIZE = (self.SIM_WIDTH, self.SIM_HEIGHT)

    @classmethod
    def from_json(cls, json_path: str) -> 'SimConfig':
        """JSON 파일에서 시뮬레이션 설정 로드"""
        if not os.path.exists(json_path):
            print(f"설정 파일이 없습니다: {json_path}. 기본 설정을 사용합니다.")
            return cls()

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 튜플로 변환이 필요한 필드
            tuple_fields = ['SIM_SIZE', 'BACKGROUND_COLOR', 'GRID_COLOR', 'VEHICLE_COLOR',
                            'TIRE_COLOR', 'MARK_COLOR', 'HUD_BG_COLOR', 'HUD_FG_COLOR', 'MAP_BORDER_COLOR']
            for field in tuple_fields:
                if field in data and isinstance(data[field], list):
                    data[field] = tuple(data[field])

            return cls(**data)
        except Exception as e:
            print(f"설정 파일 로드 오류: {e}. 기본 설정을 사용합니다.")
            return cls()

    def to_json(self, json_path: str) -> bool:
        """시뮬레이션 설정을 JSON 파일로 저장"""
        try:
            data = {}
            for key, value in self.__dict__.items():
                # 튜플을 리스트로 변환
                if isinstance(value, tuple):
                    data[key] = list(value)
                else:
                    data[key] = value

            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            return True
        except Exception as e:
            print(f"설정 파일 저장 오류: {e}")
            return False

    def get_terrain_friction(self, terrain_type: str) -> float:
        """지형 유형에 따른 마찰 계수 반환"""
        return self.TERRAIN_FRICTION.get(terrain_type, 1.0)  # 기본값은 아스팔트

# ======================
# 센서 설정
# ======================
@dataclass
class SensorConfig:
    """센서 기본 설정 클래스"""
    # 센서 식별 및 위치 설정
    SENSOR_TYPE: str = "BASE"               # 센서 유형
    RELATIVE_POS: Tuple[float, float] = (0.0, 0.0)  # 차량 중심 기준 상대 위치 [m]
    RELATIVE_ANGLE: float = 0.0             # 차량 방향 기준 상대 각도 [rad]

@dataclass
class LidarConfig(SensorConfig):
    """라이다 센서 설정 클래스"""
    SENSOR_TYPE: str = "LIDAR"
    NUM_SAMPLES: int = 360               # 샘플(레이) 수
    ANGLE_START: float = -radians(180)  # 시작 각도 [rad] (기본: -180도)
    ANGLE_END: float = radians(180)     # 종료 각도 [rad] (기본: 180도)
    MAX_RANGE: float = 50.0             # 최대 감지 거리 [m]
    MIN_RANGE: float = 0.1              # 최소 감지 거리 [m]
    SCAN_RATE: int = 10                 # 스캔 주기 [Hz]
    DRAW_RAYS: bool = False              # 레이 시각화 여부
    SCAN_HISTORY: int = 10              # 스캔 히스토리 개수

    # 노이즈 파라미터
    NOISE_GAUSSIAN_SIGMA: float = 0.05  # 가우시안 노이즈 표준편차 [m]
    NOISE_DISTANCE_FACTOR: float = 0.01 # 거리에 따른 노이즈 증가 계수 [m/m]

    def get_noise_params(self) -> Dict[str, float]:
        """노이즈 파라미터 딕셔너리 반환"""
        return {
            'gaussian_sigma': self.NOISE_GAUSSIAN_SIGMA,
            'distance_factor': self.NOISE_DISTANCE_FACTOR
        }

# ======================
# 센서 설정 관리
# ======================
class SensorsConfigManager:
    """센서 설정을 저장하고 불러오는 클래스"""

    @staticmethod
    def load_from_json(json_path: str) -> Dict[str, SensorConfig]:
        """JSON 파일에서 센서 설정 불러오기"""
        if not os.path.exists(json_path):
            print(f"센서 설정 파일이 없습니다: {json_path}. 빈 설정을 반환합니다.")
            return {}

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 센서 설정 딕셔너리 생성
            configs = {}

            for sensor_id, sensor_data in data.items():
                sensor_type = sensor_data.get('SENSOR_TYPE', '').lower()

                # 라디안 변환이 필요한 필드
                for field in ['RELATIVE_ANGLE', 'ANGLE_START', 'ANGLE_END']:
                    if field in sensor_data and isinstance(sensor_data[field], (int, float)):
                        sensor_data[field] = radians(sensor_data[field])

                # 튜플 변환이 필요한 필드
                for field in ['RELATIVE_POS']:
                    if field in sensor_data and isinstance(sensor_data[field], list):
                        sensor_data[field] = tuple(sensor_data[field])

                # 센서 타입에 따라 적절한 설정 객체 생성
                if sensor_type == 'LIDAR':
                    configs[sensor_id] = LidarConfig(**sensor_data)
                # 여기에 다른 센서 타입 추가 가능
                else:
                    # 알 수 없는 타입은 기본 SensorConfig 사용
                    configs[sensor_id] = SensorConfig(**sensor_data)

            return configs

        except Exception as e:
            print(f"센서 설정 파일 로드 오류: {e}. 빈 설정을 반환합니다.")
            return {}

    @staticmethod
    def save_to_json(configs: Dict[str, SensorConfig], json_path: str) -> bool:
        """센서 설정을 JSON 파일로 저장"""
        try:
            data = {}

            for sensor_id, config in configs.items():
                config_data = {}

                for key, value in config.__dict__.items():
                    # 튜플을 리스트로 변환
                    if isinstance(value, tuple):
                        config_data[key] = list(value)
                    # 라디안을 도로 변환
                    elif key in ['RELATIVE_ANGLE', 'ANGLE_START', 'ANGLE_END'] and isinstance(value, (int, float)):
                        config_data[key] = degrees(value)
                    else:
                        config_data[key] = value

                data[sensor_id] = config_data

            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)

            return True

        except Exception as e:
            print(f"센서 설정 파일 저장 오류: {e}")
            return False

    @staticmethod
    def create_default_configs() -> Dict[str, SensorConfig]:
        """기본 센서 설정 생성"""
        configs = {}

        # 기본 라이다
        lidar_config = LidarConfig()
        configs['0'] = lidar_config

        return configs

if __name__ == "__main__":
    # 차량 설정 저장
    vehicle_config = VehicleConfig()
    vehicle_config.to_json("config/vehicle_config.json")

    # 시뮬레이션 설정 저장
    sim_config = SimConfig()
    sim_config.to_json("config/simulation_config.json")

    # 센서 설정 저장
    sensor_config = SensorsConfigManager.create_default_configs()
    SensorsConfigManager.save_to_json(sensor_config, "config/sensor_config.json")
