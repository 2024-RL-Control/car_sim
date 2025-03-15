# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import List
from collections import deque
from math import radians, degrees, pi, cos, sin
import pygame
import numpy as np
from physics import DynamicModel
from object import RectangleObstacle

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

# ======================
# Vehicle Model
# ======================
class Vehicle:
    """차량 모델링, 제어 및 시각화를 담당하는 클래스"""

    def __init__(self, vehicle_config=None, sim_config=None):
        """차량 객체 초기화"""
        self.config = vehicle_config
        self.sim_config = sim_config
        self.state = VehicleState()
        self._track_marks = []  # 타이어 자국 [(x,y,width), ...]

        # 경계 생성
        self.collision_body = RectangleObstacle(
            x=self.state.x, y=self.state.y,
            width=self.config.LENGTH, height=self.config.WIDTH,
            yaw=self.state.yaw,
            obs_type="vehicle",
            color=self.sim_config.VEHICLE_COLOR
        )

        # 그래픽 리소스 초기화
        self._load_graphics()

    def _load_graphics(self):
        """차량 및 타이어 그래픽 리소스 생성"""
        # 차체
        car_length = self.config.LENGTH * self.sim_config.SCALE
        car_width = self.config.WIDTH * self.sim_config.SCALE
        self.car_surf = pygame.Surface((car_length, car_width), pygame.SRCALPHA)
        pygame.draw.rect(self.car_surf, self.sim_config.VEHICLE_COLOR,
                         (0, 0, car_length, car_width), 2, border_radius=int(car_width * 0.2))

        # 타이어
        tire_w = self.config.TIRE_WIDTH * self.sim_config.SCALE
        tire_h = self.config.TIRE_HEIGHT * self.sim_config.SCALE
        self.tire_surf = pygame.Surface((tire_h, tire_w), pygame.SRCALPHA)
        pygame.draw.rect(self.tire_surf, self.sim_config.TIRE_COLOR,
                         (0, 0, tire_h, tire_w), 2)

    def reset(self):
        """차량 상태 초기화"""
        self.state = VehicleState()
        self._track_marks = []

    def step(self, action, dt):
        """차량 상태 업데이트"""
        # 물리 모델 적용
        DynamicModel.apply_forces(self.state, action, dt, self.sim_config, self.config)

        # 충돌 바디 위치 및 방향 업데이트
        self._update_collision_body()

        # 타이어 자국 업데이트
        self._update_tire_marks()

        return self.state

    def _update_collision_body(self):
        """충돌 검사용 바디 업데이트"""
        self.collision_body.set(x=self.state.x, y=self.state.y, yaw=self.state.yaw)

    def get_outer_circles_world(self):
        """차량 외접원들의 월드 좌표와 반지름 반환 [(x, y, radius), ...]"""
        return self.collision_body.get_outer_circles_world()

    def get_middle_circles_world(self):
        """차량 중간원들의 월드 좌표와 반지름 반환 [(x, y, radius), ...]"""
        return self.collision_body.get_middle_circles_world()

    def get_inner_circles_world(self):
        """차량 내접원들의 월드 좌표와 반지름 반환 [(x, y, radius), ...]"""
        return self.collision_body.get_inner_circles_world()

    def check_collision(self, obstacle_manager):
        """
        장애물과의 충돌 검사 (3단계 검사)

        Args:
            obstacle_manager: ObstacleManager 객체

        Returns:
            충돌 여부 (Boolean)
        """
        # 위치 및 방향 업데이트 (만약 step 이외에서 호출될 경우 대비)
        self._update_collision_body()

        # 장애물이 없으면 충돌 없음
        if obstacle_manager.get_obstacle_count() == 0:
            return False

        # 광역 검사 (빠른 제외)
        obstacle_outer_circles = obstacle_manager.get_all_outer_circles()
        vehicle_outer_circles = self.get_outer_circles_world()
        if not self._circles_collision(vehicle_outer_circles, obstacle_outer_circles):
            return False

        # 중간 수준 검사
        obstacle_middle_circles = obstacle_manager.get_all_middle_circles()
        vehicle_middle_circles = self.get_middle_circles_world()
        if not self._circles_collision(vehicle_middle_circles, obstacle_middle_circles):
            return False

        # 정밀 검사
        obstacle_inner_circles = obstacle_manager.get_all_inner_circles()
        vehicle_inner_circles = self.get_inner_circles_world()
        return self._circles_collision(vehicle_inner_circles, obstacle_inner_circles)

    def _circles_collision(self, circles1, circles2):
        """
        두 원 집합 간의 충돌 검사

        Args:
            circles1: 첫 번째 원 집합 [(x, y, radius), ...]
            circles2: 두 번째 원 집합 [(x, y, radius), ...]

        Returns:
            충돌 여부 (Boolean)
        """
        for x1, y1, r1 in circles1:
            for x2, y2, r2 in circles2:
                # 두 원 중심 간 거리의 제곱 계산 (제곱근 계산 회피)
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                # 두 원 반지름 합의 제곱과 비교
                if dist_sq < (r1 + r2)**2:
                    return True

        return False

    def _update_tire_marks(self):
        """타이어 자국 업데이트 - 드리프트 중일 때만 추가"""
        if not self.sim_config.ENABLE_TRACK_MARKS:
            return

        # 드리프트 중일 때만 타이어 자국 추가
        if abs(self.state.drift_angle) > radians(5) and abs(self.state.vel) > 5.0:
            # 각 타이어 위치 계산
            wheelbase = self.config.WHEELBASE
            track = self.config.TRACK

            # 타이어 상대 위치 계산 (차량 좌표계 기준)
            tire_positions = [
                ( wheelbase/2,  track/2),  # Front Right
                ( wheelbase/2, -track/2),  # Front Left
                (-wheelbase/2,  track/2),  # Rear Right
                (-wheelbase/2, -track/2)   # Rear Left
            ]

            for dx, dy in tire_positions:
                # 차량 회전 변환
                rotated_x = dx * cos(self.state.yaw) - dy * sin(self.state.yaw)
                rotated_y = dx * sin(self.state.yaw) + dy * cos(self.state.yaw)

                # 월드 좌표 계산
                world_x = self.state.x + rotated_x
                world_y = self.state.y + rotated_y

                # 마크 너비는 드리프트 각도에 비례
                mark_width = max(1, min(5, abs(self.state.drift_angle) * 10))

                # 타이어 자국 추가
                self._track_marks.append((world_x, world_y, mark_width))

        # 타이어 자국 개수 제한 (너무 많으면 성능 저하)
        max_marks = 2000
        if len(self._track_marks) > max_marks:
            self._track_marks = self._track_marks[-max_marks:]

    def calculate_tire_positions(self):
        """각 타이어의 위치 및 각도 계산"""
        wheelbase = self.config.WHEELBASE
        track = self.config.TRACK
        steer = self.state.steer

        # 타이어 상대 위치 계산 (차량 좌표계 기준)
        tire_positions = [
            ( wheelbase/2,  track/2),  # Front Right
            ( wheelbase/2, -track/2),  # Front Left
            (-wheelbase/2,  track/2),  # Rear Right
            (-wheelbase/2, -track/2)   # Rear Left
        ]

        # pygame 시각화를 위한 아커만 조향각 계산 (좌/우 앞바퀴 각도 차이 계산)
        if abs(steer) > 0.001:  # 조향 중일 때만 아커만 계산
            # 회전 반경 계산 (자전거 모델 기준)
            R = wheelbase / np.tan(abs(steer))

            # 좌/우 조향각 계산 (아커만 공식)
            if steer < 0:  # 좌회전
                steer_inner = np.arctan(wheelbase / (R - track/2))  # 왼쪽 바퀴 (안쪽)
                steer_outer = np.arctan(wheelbase / (R + track/2))  # 오른쪽 바퀴 (바깥쪽)
            else:  # 우회전
                steer_inner = -np.arctan(wheelbase / (R - track/2))  # 오른쪽 바퀴 (안쪽)
                steer_outer = -np.arctan(wheelbase / (R + track/2))  # 왼쪽 바퀴 (바깥쪽)

            # 조향각 배열 [FR, FL, RR, RL]
            steer_angles = [
                steer_outer if steer > 0 else steer_inner,  # 우회전시 바깥쪽, 좌회전시 안쪽
                steer_inner if steer > 0 else steer_outer,  # 우회전시 안쪽, 좌회전시 바깥쪽
                0.0,  # 뒷바퀴는 조향 없음
                0.0   # 뒷바퀴는 조향 없음
            ]
        else:
            # 직진 시 모든 바퀴 조향각 0
            steer_angles = [0.0, 0.0, 0.0, 0.0]

        # 각 타이어의 위치와 각도 계산
        result = []
        for i, (dx, dy) in enumerate(tire_positions):
            # 차량 회전 변환
            rotated_x = dx * cos(self.state.yaw) - dy * sin(self.state.yaw)
            rotated_y = dx * sin(self.state.yaw) + dy * cos(self.state.yaw)

            # 월드 좌표 계산
            world_x = self.state.x + rotated_x
            world_y = self.state.y + rotated_y

            # 타이어 각도 계산 (앞바퀴는 아커만 스티어링 적용)
            tire_angle = steer_angles[i]
            total_angle = self.state.yaw + tire_angle

            result.append((world_x, world_y, total_angle))

        return result

    def draw(self, screen, world_to_screen_func):
        """차량 및 타이어 렌더링"""
        # 타이어 자국 그리기
        self._draw_tire_marks(screen, world_to_screen_func)

        # 궤적 그리기
        self._draw_trajectory(screen, world_to_screen_func)

        # 타이어 그리기
        self._draw_tires(screen, world_to_screen_func)

        # 차체 그리기
        self._draw_body(screen, world_to_screen_func)

        # 디버그 모드: 경계 원 표시
        if self.sim_config.ENABLE_DEBUG_INFO:
            # 충돌 바디의 경계 원 그리기 메서드 활용
            self.collision_body._draw_bounding_circles(screen, world_to_screen_func)

    def _draw_trajectory(self, screen, world_to_screen_func):
        """차량 궤적 그리기"""
        if len(self.state.trajectory) < 2:
            return

        # 지난 궤적을 점선으로 표시
        trajectory_points = [world_to_screen_func(x, y) for x, y in self.state.trajectory]

        # 부드러운 곡선 대신 선분으로 그림 (성능)
        if len(trajectory_points) > 1:
            trajectory_color = (0, 150, 255, 100)  # 반투명 파란색
            pygame.draw.lines(screen, trajectory_color, False, trajectory_points, 2)

    def _draw_tire_marks(self, screen, world_to_screen_func):
        """타이어 자국 그리기"""
        if not self.sim_config.ENABLE_TRACK_MARKS:
            return

        # 타이어 자국 렌더링 (모든 자국)
        for x, y, width in self._track_marks:
            pos = world_to_screen_func(x, y)
            pygame.draw.circle(screen, self.sim_config.MARK_COLOR, pos, width)

    def _draw_tires(self, screen, world_to_screen_func):
        """4개의 타이어 렌더링 (아커만 조향 적용)"""
        # 타이어 위치와 각도 가져오기
        tire_data = self.calculate_tire_positions()

        for world_x, world_y, total_angle in tire_data:
            # 타이어 회전 및 위치 변환
            rotated_tire = pygame.transform.rotate(self.tire_surf, degrees(total_angle))
            tire_rect = rotated_tire.get_rect(center=world_to_screen_func(world_x, world_y))
            screen.blit(rotated_tire, tire_rect.topleft)

    def _draw_body(self, screen, world_to_screen_func):
        """차체 렌더링"""
        rotated_surf = pygame.transform.rotate(self.car_surf, degrees(self.state.yaw))
        rect = rotated_surf.get_rect(center=world_to_screen_func(self.state.x, self.state.y))
        screen.blit(rotated_surf, rect.topleft)

    def get_track_marks(self):
        """타이어 자국 데이터 반환"""
        return self._track_marks

    def set_track_marks(self, track_marks):
        """타이어 자국 데이터 설정"""
        self._track_marks = track_marks

    def clear_track_marks(self):
        """모든 타이어 자국 삭제"""
        self._track_marks = []
