# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import List
from collections import deque
import time
from math import radians, degrees, pi, cos, sin

# ======================
# State Management
# ======================
@dataclass
class VehicleState:
    """차량의 물리적 상태를 관리하는 클래스"""
    x: float = 0.0                 # 글로벌 X 좌표 [m]
    y: float = 0.0                 # 글로벌 Y 좌표 [m]
    yaw: float = pi/2              # 요각 [rad]
    vel: float = 0.0               # 종방향 속도 [m/s]
    vel_lateral: float = 0.0       # 횡방향 속도 [m/s] (신규)
    steer: float = 0.0             # 조향각 [rad]
    accel: float = 0.0             # 종방향 가속도 [m/s²]
    slip_angle: float = 0.0        # 차체 슬립각 [rad]
    drift_angle: float = 0.0       # 드리프트 각도 [rad]
    g_forces: List[float] = field(default_factory=lambda: [0.0, 0.0])  # [종방향, 횡방향] G-포스

    terrain_type: str = "asphalt"  # 현재 지형 유형

    # 궤적 기록 (최근 위치 추적, 성능 최적화를 위해 deque 사용)
    trajectory: deque = field(default_factory=lambda: deque(maxlen=3000))

    def normalize_angle(self, angle):
        """[-π, π] 범위로 각도 정규화"""
        return (angle + pi) % (2 * pi) - pi

    def encoding_angle(self, angle):
        """cos/sin 인코딩을 통한 각도 표현"""
        return cos(angle), sin(angle)

    def update_trajectory(self):
        """현재 위치를 궤적에 추가"""
        self.trajectory.append((self.x, self.y))

