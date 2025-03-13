# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Dict
from math import radians, degrees, pi, cos, sin

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
    MAX_SPEED: float = 65.0       # 최대 속도 [m/s] (약 234 km/h)
    MIN_SPEED: float = -20.0       # 최소 속도 [m/s] (약 72 km/h)

    # Tire Parameters (Pacejka Magic Formula)
    TIRE_B: float = 10.0           # Stiffness Factor
    TIRE_C: float = 1.9            # Shape Factor
    TIRE_D: float = 1.0            # Peak Factor
    TIRE_E: float = 0.97           # Curvature Factor

# ======================
# Simulation Constants
# ======================
@dataclass
class SimConfig:
    FPS: int = 60
    SCALE: float = 5.0             # 1pixel당 미터 단위
    SIM_WIDTH: int = 1300
    SIM_HEIGHT: int = 800
    SIM_SIZE: tuple = (SIM_WIDTH, SIM_HEIGHT)

    # 색상 설정
    BACKGROUND_COLOR: tuple = (0, 0, 0)       # 배경 (검은색)
    GRID_COLOR: tuple = (100, 100, 100)       # 그리드 (회색)
    VEHICLE_COLOR: tuple = (255, 255, 255)    # 차량 몸체 (흰색)
    TIRE_COLOR: tuple = (255, 255, 255)       # 바퀴 (흰색)
    MARK_COLOR: tuple = (80, 80, 80)           # 타이어 자국 (회색)
    HUD_BG_COLOR: tuple = (50, 50, 50, 180)   # HUD 바탕 (반투명 회색)
    HUD_FG_COLOR: tuple = (255, 255, 0)       # HUD 글씨 (노란색)
    MAP_BORDER_COLOR: tuple = (150, 150, 150) # 미니맵 테두리

    # 물리 상수
    GRAVITY: float = 9.81          # 중력가속도 [m/s²]
    AIR_DENSITY: float = 1.225     # 공기 밀도 [kg/m³]
    DRAG_COEFF: float = 0.30       # 공기저항 계수
    ROLL_RESIST: float = 0.015     # 구름저항 계수
    PHYSICS_SUBSTEPS: int = 4      # 물리 시뮬레이션 세부 단계

    # 시뮬레이션 환경 설정
    ENABLE_TRACK_MARKS: bool = True  # 타이어 자국 시각화
    ENABLE_DEBUG_INFO: bool = True  # 디버그 정보 표시
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
