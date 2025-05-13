# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import List
from collections import deque
from math import radians, degrees, pi, cos, sin
import pygame
import numpy as np
import time
from .physics import PhysicsEngine
from .trajectory import TrajectoryPredictor, TrajectoryData
from .object import RectangleObstacle, GoalManager
from .sensor import SensorManager

# ======================
# State Management
# ======================
@dataclass
class VehicleState:
    """차량의 상태를 관리하는 클래스"""
    # 차량 속성
    x: float = 0.0                 # 글로벌 X 좌표 [m]
    y: float = 0.0                 # 글로벌 Y 좌표 [m]
    yaw: float = pi/2              # 요각 [rad]
    yaw_rate: float = 0.0          # 요 회전 속도 [rad/s]
    vel_long: float = 0.0          # 종방향 속도 [m/s]
    acc_long: float = 0.0          # 종방향 가속도 [m/s²]
    vel_lat: float = 0.0           # 횡방향 속도 [m/s]
    acc_lat: float = 0.0           # 횡방향 가속도 [m/s²]

    rear_axle_x: float = 0.0       # 뒷바퀴 축 X 좌표 [m]
    rear_axle_y: float = 0.0       # 뒷바퀴 축 Y 좌표 [m]
    half_wheelbase: float = None    # 차량 축간거리의 절반 [m]

    # 제어 상태
    steer: float = 0.0             # 조향각 [rad]
    throttle_engine: float = 0.0   # 파워 엔진의 요청 정도 [0,1]
    throttle_brake:  float = 0.0   # 브레이크 요청 정도[0,1]

    # 목적지 관련 속성
    target_x: float = 0.0         # 목표 X 좌표
    target_y: float = 0.0         # 목표 Y 좌표
    target_yaw: float = 0.0       # 목표 요각
    distance_to_target: float = 0.0  # 목표까지의 거리
    yaw_diff_to_target: float = 0.0  # 목표까지의 방향 차이

    # frenet 좌표
    frenet_d: float = None
    frenet_point: tuple = None
    target_vel_long: float = None

    # 라이다 센서 데이터, 거리 배열
    lidar_data: List = field(default_factory=list)

    # 차량 과거 궤적
    history_trajectory: deque = field(default_factory=lambda: deque(maxlen=3000))

    # 궤적 데이터
    polynomial_trajectory: List[TrajectoryData] = field(default_factory=list)
    physics_trajectory: List[TrajectoryData] = field(default_factory=list)

    # 상태 이력 (최근 N개 상태 기록)
    state_history: deque = field(default_factory=lambda: deque(maxlen=100))

    # 환경 속성
    terrain_type: str = "asphalt"  # 현재 지형 유형

    def reset(self):
        """차량 상태 초기화"""
        self.x = 0.0
        self.y = 0.0
        self.yaw = pi/2
        self.yaw_rate = 0.0
        self.vel_long = 0.0
        self.acc_long = 0.0
        self.vel_lat = 0.0
        self.acc_lat = 0.0

        self.rear_axle_x = 0.0
        self.rear_axle_y = 0.0
        self.update_rear_axle_position()

        self.steer = 0.0
        self.throttle_engine = 0.0
        self.throttle_brake = 0.0

        self.target_x = 0.0
        self.target_y = 0.0
        self.target_yaw = 0.0
        self.distance_to_target = 0.0
        self.yaw_diff_to_target = 0.0

        self.frenet_d = None
        self.frenet_point = None
        self.target_vel_long = None

        self.lidar_data.clear()

        self.history_trajectory.clear()
        self.polynomial_trajectory.clear()
        self.physics_trajectory.clear()

        self.state_history.clear()

        self.terrain_type = "asphalt"

    def normalize_angle(self, angle):
        """[-π, π] 범위로 각도 정규화"""
        return (angle + pi) % (2 * pi) - pi

    def encoding_angle(self, angle):
        """cos/sin 인코딩을 통한 각도 표현"""
        return cos(angle), sin(angle)

    def update_history_trajectory(self):
        """현재 위치를 궤적에 추가"""
        self.history_trajectory.append((self.x, self.y))

    def set_position(self, x, y, yaw=None):
        """차량 위치 및 방향 설정"""
        self.x = x
        self.y = y
        if yaw is not None:
            self.yaw = self.normalize_angle(yaw)
        # 뒷바퀴 축 위치 업데이트
        self.update_rear_axle_position()

    def update_rear_axle_position(self, half_wheelbase=None):
        """뒷바퀴 축 위치 업데이트"""
        if (half_wheelbase is not None) and (self.half_wheelbase != half_wheelbase):
            self.half_wheelbase = half_wheelbase

        self.rear_axle_x = self.x - self.half_wheelbase * cos(self.yaw)
        self.rear_axle_y = self.y - self.half_wheelbase * sin(self.yaw)

    def update_position_from_rear_axle(self, cos_yaw, sin_yaw):
        """뒷바퀴 축 위치에서 heading 방향으로 이동, 차량 중심점 위치 업데이트"""
        self.x = self.rear_axle_x + self.half_wheelbase * cos_yaw
        self.y = self.rear_axle_y + self.half_wheelbase * sin_yaw

    def update_road_data(self, road_manager):
        """
        도로 데이터 기반 차량 상태 업데이트
        """
        self.frenet_point, self.frenet_d, self.target_vel_long, outside_road = road_manager.get_vehicle_update_data((self.x, self.y, self.yaw))
        return outside_road

    def update_state_history(self):
        """현재 상태 복사본을 이력에 추가"""
        # 현재 상태의 중요 필드들을 딕셔너리로 복사
        state_snapshot = {
            'timestamp': time.time(),  # 타임스탬프
            'x': self.rear_axle_x,
            'y': self.rear_axle_y,
            'yaw': self.yaw,
            'vel_long': self.vel_long,
            'acc_long': self.acc_long,
            'vel_lat': self.vel_lat,
            'acc_lat': self.acc_lat,
            'steer': self.steer,
            'throttle_engine': self.throttle_engine,
            'throttle_brake': self.throttle_brake
        }
        self.state_history.append(state_snapshot)

    def update_trajectory(self, polynomial_trajectory, physics_trajectory):
        """궤적 업데이트"""
        self.polynomial_trajectory = polynomial_trajectory
        self.physics_trajectory = physics_trajectory

    def get_trajectory_data(self):
        """차량 궤적 데이터 반환, 각 지점의 상대적 위치로 변환

        각 궤적에서 시작, 중간, 마지막 데이터만 추출
        """
        polynomial_trajectory = self.polynomial_trajectory
        physics_trajectory = self.physics_trajectory

        data_list = []

        # 다항식 궤적에서 초반, 중반, 마지막 데이터 추출
        if polynomial_trajectory:
            traj_len = len(polynomial_trajectory)
            # 초반(0%), 중반(50%), 마지막(100%) 지점 인덱스 계산
            start_idx = 0
            mid_idx = traj_len // 2
            end_idx = traj_len - 1

            # 선택된 지점의 데이터만 추가
            if start_idx < traj_len:
                data = polynomial_trajectory[start_idx].get_data(self.x, self.y)
                data_list.append(data)

            if mid_idx < traj_len and mid_idx != start_idx:
                data = polynomial_trajectory[mid_idx].get_data(self.x, self.y)
                data_list.append(data)

            if end_idx < traj_len and end_idx != mid_idx and end_idx != start_idx:
                data = polynomial_trajectory[end_idx].get_data(self.x, self.y)
                data_list.append(data)

        # 물리 기반 궤적에서 초반, 중반, 마지막 데이터 추출
        if physics_trajectory:
            traj_len = len(physics_trajectory)
            # 초반(0%), 중반(50%), 마지막(100%) 지점 인덱스 계산
            start_idx = 0
            mid_idx = traj_len // 2
            end_idx = traj_len - 1

            # 선택된 지점의 데이터만 추가
            if start_idx < traj_len:
                data = physics_trajectory[start_idx].get_data(self.x, self.y)
                data_list.append(data)

            if mid_idx < traj_len and mid_idx != start_idx:
                data = physics_trajectory[mid_idx].get_data(self.x, self.y)
                data_list.append(data)

            if end_idx < traj_len and end_idx != mid_idx and end_idx != start_idx:
                data = physics_trajectory[end_idx].get_data(self.x, self.y)
                data_list.append(data)

        return np.array(data_list)

    def update_lidar_data(self, distances):
        """라이다 센서 데이터 업데이트"""
        self.lidar_data = distances

    def get_lidar_data(self):
        """라이다 센서 데이터 반환"""
        return np.array(self.lidar_data)

# ======================
# Vehicle Model
# ======================
class Vehicle:
    """차량 모델링, 제어 및 시각화를 담당하는 클래스"""

    def __init__(self, vehicle_id=None, vehicle_config=None, physics_config=None, visual_config=None):
        """차량 객체 초기화"""
        self.id = vehicle_id if vehicle_id is not None else id(self)
        self.vehicle_config = vehicle_config
        self.physics_config = physics_config
        self.visual_config = visual_config
        self.state = VehicleState()
        self.state.update_rear_axle_position(self.vehicle_config['wheelbase'] / 2.0)

        self.collision_body = RectangleObstacle(
            x=self.state.x, y=self.state.y,
            width=self.vehicle_config['length'], height=self.vehicle_config['width'],
            yaw=self.state.yaw,
            obs_type="vehicle",
            color=self.visual_config['vehicle_colors'],
            bounding_circle_colors=self.visual_config['bounding_circle_color']
        )

        self.goal_manager = GoalManager(bounding_circle_colors=self.visual_config['bounding_circle_color'])
        self.sensor_manager = SensorManager(self, self.vehicle_config['sensors'], self.visual_config)

        # 그래픽 리소스 초기화
        self._load_graphics()

    def get_state(self):
        """차량 상태 반환"""
        return self.state

    def get_id(self):
        """차량 ID 반환"""
        return self.id

    def set_position(self, x, y, yaw = None):
        """차량 위치 및 방향 설정"""
        self.state.set_position(x, y, yaw)
        self._update_collision_body()

    def get_position(self):
        """차량 위치 반환"""
        return (self.state.x, self.state.y, self.state.yaw)

    def get_current_goal(self):
        """현재 목표 반환"""
        return self.goal_manager.get_current_goal()

    def add_goal(self, x, y, yaw=0.0, radius=1.0, color=(0, 255, 0)):
        """
        목적지 추가 및 목표 정보 업데이트

        Args:
            x: 목적지 X 좌표
            y: 목적지 Y 좌표
            yaw: 목적지 방향각 [rad]
            radius: 목적지 반경
            color: 목적지 색상 (RGB)

        Returns:
            goal_id: 생성된 목적지 ID
        """
        goal_id = self.goal_manager.add_goal(x, y, yaw, radius, color)
        goal = self.goal_manager.get_current_goal()
        self.update_target(goal.x, goal.y, goal.yaw)
        return goal_id

    def remove_goal(self, index=None):
        """
        목적지 제거

        Args:
            index: 제거할 목적지 인덱스 (None이면 현재 목적지)

        Returns:
            성공 여부 (Boolean)
        """
        if index is None:
            bool = self.goal_manager.remove_current_goal(index)
            next_goal = self.goal_manager.get_current_goal()
            if next_goal:
                self.update_target(next_goal.x, next_goal.y, next_goal.yaw)
            return bool
        else:
            # 인덱스가 유효한지 확인
            if index < 0 or index >= len(self.goal_manager.goals):
                return False
            else:
                bool = self.goal_manager.remove_current_goal(index)
                next_goal = self.goal_manager.get_current_goal()
                if next_goal:
                    self.update_target(next_goal.x, next_goal.y, next_goal.yaw)
                return bool

    def clear_goals(self):
        """모든 목적지 제거"""
        self.goal_manager.clear_goals()

    def next_goal(self):
        """
        다음 목적지로 전환

        Returns:
            성공 여부 (Boolean)
        """
        result = self.goal_manager.next_goal()
        if result:
            # 현재 목표 업데이트
            goal = self.goal_manager.get_current_goal()
            if goal:
                self.update_target(goal.x, goal.y, goal.yaw)
        return result

    def reset(self):
        """차량 상태 초기화"""
        self.state.reset()
        self.goal_manager.clear_goals()
        self.sensor_manager.reset()
        self._load_graphics()
        self._update_collision_body()

    def step(self, action, dt, time_elapsed, road_manager, obstacles=[], vehicles=[]):
        """차량 상태 업데이트"""

        # 물리 모델 적용
        PhysicsEngine.update(self.state, action, dt, self.physics_config, self.vehicle_config)

        # 충돌 바디 위치 및 방향 업데이트
        self._update_collision_body()

        # 객체 목록 생성
        objects = []
        if obstacles:
            objects.extend(obstacles)
        if vehicles:
            for vehicle in vehicles:
                if vehicle.id != self.id:  # 자기 자신이 아닌 차량만 추가
                    objects.extend(vehicle.get_outer_circles_world())

        # 차량 센서 업데이트
        self.sensor_manager.update(dt, time_elapsed, objects)
        self._update_lidar_data()

        # 도로 정보 업데이트
        outside_road = self._update_road_data(road_manager)

        # 충돌 검사
        collision = self._check_collision(objects)

        # 목표 도달 여부 확인
        reached = False
        if self.goal_manager.has_goals():
            reached = self._check_target_reached()

        # 상태 이력 업데이트
        self._update_state_history()
        # 차량 궤적 업데이트
        self._predict_trajectory(self.physics_config['trajectory']['time_horizon'], self.physics_config['trajectory']['dt'])

        return self.state, collision, outside_road, reached

    def _update_road_data(self, road_manager):
        return self.state.update_road_data(road_manager)

    def _update_state_history(self):
        self.state.update_state_history()

    def _check_collision(self, objects):
        """
        장애물과의 충돌 검사 (1단계 검사)

        Args:
            objects: 객체 외접원

        Returns:
            충돌 여부 (Boolean)
        """
        # 위치 및 방향 업데이트 (만약 step 이외에서 호출될 경우 대비)
        self._update_collision_body()

        if len(objects) == 0:
            return False

        # 다른 객체들과의 외접원 수준의 충돌 검사
        return self._circles_collision(self.get_outer_circles_world(), objects)

    def get_outer_circles_world(self):
        """차량 외접원들의 월드 좌표와 반지름 반환 [(x, y, radius), ...]"""
        return self.collision_body.get_outer_circles_world()

    def _get_middle_circles_world(self):
        """차량 중간원들의 월드 좌표와 반지름 반환 [(x, y, radius), ...]"""
        return self.collision_body.get_middle_circles_world()

    def _get_inner_circles_world(self):
        """차량 내접원들의 월드 좌표와 반지름 반환 [(x, y, radius), ...]"""
        return self.collision_body.get_inner_circles_world()

    def _update_collision_body(self):
        """충돌 검사용 바디 업데이트"""
        self.collision_body.set(x=self.state.x, y=self.state.y, yaw=self.state.yaw)

    def _circles_collision(self, circles1, circles2):
        """
        두 원 집합 간의 충돌 검사

        Args:
            circles1: 첫 번째 원 집합 [(x, y, radius), ...]
            circles2: 두 번째 원 집합 [(x, y, radius), ...]

        Returns:
            충돌 여부 (Boolean)
        """
        # 원 집합이 비어있는 경우 충돌 없음
        if not circles1 or not circles2:
            return False

        array1 = np.array(circles1)
        array2 = np.array(circles2)

        # N1 x 1 배열
        x1 = array1[:, 0][:, np.newaxis]
        y1 = array1[:, 1][:, np.newaxis]
        r1 = array1[:, 2][:, np.newaxis]

        # 1 x N2 배열
        x2 = array2[:, 0]
        y2 = array2[:, 1]
        r2 = array2[:, 2]

        # 거리 제곱 계산 (모든 조합에 대해 한 번에 계산)
        dist_sq = (x1 - x2)**2 + (y1 - y2)**2

        # 반지름 합의 제곱 계산
        radii_sum_sq = (r1 + r2)**2

        # 충돌 여부 확인: 거리 제곱 < 반지름 합의 제곱
        collisions = dist_sq < radii_sum_sq

        return np.any(collisions)

    def update_target(self, x, y, yaw):
        """
        차량의 목표 업데이트

        Args:
            x: 목표 X 좌표
            y: 목표 Y 좌표
            yaw: 목표 요각
        """
        self.state.target_x = x
        self.state.target_y = y
        self.state.target_yaw = yaw
        self._update_target_info()

    def _check_target_reached(self, position_tolerance=None, yaw_tolerance=None):
        """
        목표 위치 도달 여부 확인

        Args:
            position_tolerance: 위치 도달 판정 거리 [m] (None이면 차량 길이의 절반 사용)
            yaw_tolerance: 방향 도달 판정 각도 [rad] (None이면 기본값 π/6 (30도) 사용)

        Returns:
            도달 여부 (Boolean)
        """
        if position_tolerance is None:
            position_tolerance = self.vehicle_config['length'] / 2

        if yaw_tolerance is None:
            yaw_tolerance = np.pi/6  # 30도

        # 목표 정보 업데이트
        self._update_target_info()

        # 거리 차이 계산
        distance = self.state.distance_to_target
        position_reached = distance <= position_tolerance

        # 방향 차이 계산, 목표 방향과 현재 방향의 차이 (절대값 -π ~ π 범위로)
        yaw_diff = abs(self.state.yaw_diff_to_target)
        direction_reached = yaw_diff <= yaw_tolerance

        reached = position_reached and direction_reached

        return reached

    def _update_target_info(self):
        """목표 위치까지의 거리, 각도도 계산 및 업데이트"""
        dx = self.state.x - self.state.target_x
        dy = self.state.y - self.state.target_y
        self.state.distance_to_target = (dx**2 + dy**2) ** 0.5
        self.state.yaw_diff_to_target = self.state.normalize_angle(self.state.target_yaw - self.state.yaw)

    def draw(self, screen, world_to_screen_func, is_active=False, debug=False):
        """차량 및 타이어 렌더링"""
        # 스케일 계산
        self._update_scale(self.visual_config['camera_zoom'])

        self.goal_manager.draw(screen, world_to_screen_func, debug)

        # 디버그 모드 - 활성 차량인 경우에만 센서 시각화
        if debug:
            if is_active:
                # 차량 센서 그리기
                self.sensor_manager.draw(screen, world_to_screen_func, debug)

                # frenet 좌표 시각화
                self._draw_frenet_point(screen, world_to_screen_func)

            # 충돌 바디의 경계 원 그리기 메서드 활용
            self.collision_body._draw_bounding_circles(screen, world_to_screen_func)

            # 궤적 그리기
            self._draw_history_trajectory(screen, world_to_screen_func)

            # 예측 궤적 그리기
            self._draw_predicted_trajectory(screen, world_to_screen_func)

        # 타이어 그리기
        self._draw_tires(screen, world_to_screen_func)

        # 차체 그리기
        self._draw_body(screen, world_to_screen_func)

    def _draw_frenet_point(self, screen, world_to_screen_func):
        """frenet 좌표 시각화"""
        if self.state.frenet_point is not None:
            radius = max(1, int(2 * self.visual_config['camera_zoom']))
            pygame.draw.circle(screen, (0, 255, 255), world_to_screen_func(self.state.frenet_point[0], self.state.frenet_point[1]), radius)

    def _load_graphics(self):
        """차량 및 타이어 그래픽 리소스 생성"""
        self._camera_zoom = self.visual_config['camera_zoom']

        # 성능 최적화를 위한 그리기 관련 캐시
        self._car_cache = {}
        self._tire_angle_cache = {}
        self._tire_positions = None
        self._last_yaw = None
        self._last_steer = None

        # 차체
        car_length = self.vehicle_config['length'] * self.visual_config['scale_factor']
        car_width = self.vehicle_config['width'] * self.visual_config['scale_factor']
        self.car_surf = pygame.Surface((car_length, car_width), pygame.SRCALPHA)
        pygame.draw.rect(self.car_surf, self.visual_config['vehicle_colors'],
                         (0, 0, car_length, car_width), 2, border_radius=int(car_width * 0.2))

        # 타이어
        tire_w = self.vehicle_config['tire_width'] * self.visual_config['scale_factor']
        tire_h = self.vehicle_config['tire_height'] * self.visual_config['scale_factor']
        self.tire_surf = pygame.Surface((tire_h, tire_w), pygame.SRCALPHA)
        pygame.draw.rect(self.tire_surf, self.visual_config['tire_color'],
                         (0, 0, tire_h, tire_w), 2)

        # 실제 렌더링에 사용될 스케일된 Surface 초기화
        self._update_scaled_surfaces(self.visual_config['scale_factor'] * self._camera_zoom)

    def _update_scale(self, camera_zoom):
        """카메라 줌 변경 시 스케일 및 Surface 업데이트"""
        if camera_zoom != self._camera_zoom:
            self._camera_zoom = camera_zoom
            # 현재 시뮬레이션 스케일과 카메라 줌을 곱한 유효 스케일 계산
            effective_scale = self.visual_config['scale_factor'] * camera_zoom
            # 스케일된 Surface 업데이트
            self._update_scaled_surfaces(effective_scale)

    def _update_scaled_surfaces(self, effective_scale):
        """현재 스케일에 맞게 차량과 타이어 Surface 생성"""
        # 이미 캐시에 있는 경우 재사용
        if effective_scale in self._car_cache:
            self.car_surf = self._car_cache[effective_scale]
        else:
            # 새 크기 계산
            new_length = self.vehicle_config['length'] * effective_scale
            new_width = self.vehicle_config['width'] * effective_scale

            # 차체 Surface 재생성
            self.car_surf = pygame.Surface((new_length, new_width), pygame.SRCALPHA)
            pygame.draw.rect(self.car_surf, self.visual_config['vehicle_colors'],
                            (0, 0, new_length, new_width), 2, border_radius=int(new_width * 0.2))

            # 캐시에 저장
            self._car_cache[effective_scale] = self.car_surf

            # 캐시 크기 제한 (10개 이상이면 가장 오래된 항목 제거)
            if len(self._car_cache) > 10:
                oldest_scale = next(iter(self._car_cache))
                if oldest_scale != effective_scale:
                    del self._car_cache[oldest_scale]

        # 타이어 Surface 스케일링 - 각도별 캐시는 무효화 (새 스케일에 맞게 다시 생성 필요)
        self._tire_angle_cache.clear()

        # 새 크기 계산
        new_tire_h = self.vehicle_config['tire_height'] * effective_scale
        new_tire_w = self.vehicle_config['tire_width'] * effective_scale

        # 타이어 Surface 재생성
        self.tire_surf = pygame.Surface((new_tire_h, new_tire_w), pygame.SRCALPHA)
        pygame.draw.rect(self.tire_surf, self.visual_config['tire_color'],
                         (0, 0, new_tire_h, new_tire_w), 2)

    def _draw_tires(self, screen, world_to_screen_func):
        """4개의 타이어 렌더링 (아커만 조향 적용, 스케일 적용)"""
        # 타이어 위치와 각도 가져오기
        tire_data = self._calculate_tire_positions()

        for world_x, world_y, total_angle in tire_data:
            # 회전 각도를 정수로 반올림 (캐싱용)
            angle_deg = int(degrees(total_angle))

            # 해당 각도의 회전된 타이어 이미지가 캐시에 없으면 생성
            if angle_deg not in self._tire_angle_cache:
                self._tire_angle_cache[angle_deg] = pygame.transform.rotate(self.tire_surf, angle_deg)

                # 캐시 크기 제한 (36개 각도 이상이면 랜덤하게 하나 제거)
                if len(self._tire_angle_cache) > 36:
                    random_key = next(iter(self._tire_angle_cache))
                    if random_key != angle_deg:
                        del self._tire_angle_cache[random_key]

            # 캐시된 회전 이미지 사용
            rotated_tire = self._tire_angle_cache[angle_deg]
            tire_rect = rotated_tire.get_rect(center=world_to_screen_func(world_x, world_y))
            screen.blit(rotated_tire, tire_rect.topleft)

    def _draw_body(self, screen, world_to_screen_func):
        """차체 렌더링"""
        rotated_surf = pygame.transform.rotate(self.car_surf, degrees(self.state.yaw))
        rect = rotated_surf.get_rect(center=world_to_screen_func(self.state.x, self.state.y))
        screen.blit(rotated_surf, rect.topleft)

    def _calculate_tire_positions(self):
        """각 타이어의 위치 및 각도 계산"""
        # 상대 위치 및 각도 캐싱 - yaw 또는 steer가 변하지 않으면 재사용
        recalculate_angles = (self._tire_positions is None or
                              self._last_yaw != self.state.yaw or
                              self._last_steer != self.state.steer)

        if recalculate_angles:
            # yaw 또는 steer가 변경되었을 때만 상대 위치 및 각도 재계산
            wheelbase = self.vehicle_config['wheelbase']
            track = self.vehicle_config['track']
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

            # 회전 및 각도 정보 캐싱
            self._cached_tire_data = []
            for i, (dx, dy) in enumerate(tire_positions):
                # 차량 회전 변환 (상대 좌표)
                rotated_x = dx * cos(self.state.yaw) - dy * sin(self.state.yaw)
                rotated_y = dx * sin(self.state.yaw) + dy * cos(self.state.yaw)

                # 타이어 각도 계산 (앞바퀴는 아커만 스티어링 적용)
                tire_angle = steer_angles[i]
                total_angle = self.state.yaw + tire_angle

                # 상대 위치와 각도 저장
                self._cached_tire_data.append((rotated_x, rotated_y, total_angle))

            # 캐싱 상태 업데이트
            self._last_yaw = self.state.yaw
            self._last_steer = self.state.steer

        # 월드 좌표 계산 (매 프레임 업데이트)
        result = []
        for rotated_x, rotated_y, total_angle in self._cached_tire_data:
            # 현재 차량 위치에 상대 좌표 더하기
            world_x = self.state.x + rotated_x
            world_y = self.state.y + rotated_y
            result.append((world_x, world_y, total_angle))

        # 최종 결과 저장 (월드 좌표)
        self._tire_positions = result

        return result

    def _update_lidar_data(self):
        """
        라이다 센서의 noisy_distances 값을 차량 상태에 업데이트
        """
        # 센서 매니저에서 모든 센서 가져오기
        sensors = self.sensor_manager.get_all_sensors()

        # 라이다 센서 찾기
        for sensor_id, sensor in sensors.items():
            if sensor.sensor_type == 'LidarSensor':
                # 라이다 데이터 가져오기
                lidar_data = sensor.get_data()
                if lidar_data:
                    # noisy_distances 추출하여 상태에 저장
                    self.state.update_lidar_data(lidar_data)

    def _draw_history_trajectory(self, screen, world_to_screen_func):
        """차량 궤적 그리기"""
        if len(self.state.history_trajectory) < 2:
            return

        # 지난 궤적을 점선으로 표시
        trajectory_points = [world_to_screen_func(x, y) for x, y in self.state.history_trajectory]

        # 줌 레벨에 맞게 선 너비 조정
        width = max(1, int(2 * self.visual_config['camera_zoom']))

        # 부드러운 곡선 대신 선분으로 그림 (성능)
        if len(trajectory_points) > 1:
            trajectory_color = (0, 150, 255, 100)  # 반투명 파란색
            pygame.draw.lines(screen, trajectory_color, False, trajectory_points, width)

    def _predict_trajectory(self, time_horizon=1.0, dt=0.05):
        """미래 궤적 예측 및 시각화
        Args:
            time_horizon: 궤적 예측 시간 범위 [s]
            dt: 시간 간격 [s]
        """
        predicted_polynomial_trajectory = []
        predicted_physics_trajectory = []

        if self.goal_manager.has_goals():
            # 목표 속도와 횡방향 위치 설정
            target_velocity = self.state.target_vel_long if self.state.target_vel_long is not None else self.state.vel_long
            target_d = 0.0 if self.state.frenet_d is not None else self.state.frenet_d

            # 궤적 예측
            predicted_polynomial_trajectory = TrajectoryPredictor.predict_polynomial_trajectory(
                state=self.state,
                target_velocity=target_velocity,
                target_d=target_d,
                time_horizon=time_horizon,
                dt=dt
            )

        predicted_physics_trajectory = TrajectoryPredictor.predict_physics_based_trajectory(
            state=self.state,
            time_horizon=time_horizon,
            dt=dt,
            physics_config=self.physics_config,
            vehicle_config=self.vehicle_config
        )

        self.state.update_trajectory(predicted_polynomial_trajectory, predicted_physics_trajectory)

    def _draw_predicted_trajectory(self, screen, world_to_screen):
        """예측 궤적 시각화

        Args:
            screen: Pygame 화면 객체
            world_to_screen: 월드 좌표를 화면 좌표로 변환하는 함수
            color: 궤적 색상 (RGB)
            width: 선 두께
        """
        width = max(1, int(2 * self.visual_config['camera_zoom']))

        # 궤적 포인트들을 화면 좌표로 변환
        polynomial_screen_points = [world_to_screen(point.x, point.y) for point in self.state.polynomial_trajectory]
        physics_screen_points = [world_to_screen(point.x, point.y) for point in self.state.physics_trajectory]

        # 라인으로 연결하여 궤적 그리기
        if len(polynomial_screen_points) > 0:
            pygame.draw.lines(screen, (0, 255, 255), False, polynomial_screen_points, width)
        if len(physics_screen_points) > 0:
            pygame.draw.lines(screen, (255, 0, 0), False, physics_screen_points, width)

    def get_serializable_state(self):
        """
        직렬화 가능한 상태 정보 반환

        Returns:
            직렬화 가능한 상태 정보 딕셔너리
        """
        state_dict = self.state.__dict__.copy()

        # 직렬화할 수 없는 trajectory 필드 처리
        state_dict['history_trajectory'] = list(self.state.history_trajectory)

        return {
            'id': self.id,
            'state': state_dict,
            'goals': self.goal_manager.get_serializable_goals()
        }

    def load_from_serialized(self, serialized_data):
        """
        직렬화된 데이터에서 상태 복원

        Args:
            serialized_data: 직렬화된 상태 정보
        """
        # 상태 복원
        if 'state' in serialized_data:
            state_dict = serialized_data['state']
            for key, value in state_dict.items():
                if hasattr(self.state, key):
                    if key == 'history_trajectory':
                        self.state.history_trajectory = deque(value, maxlen=3000)
                    else:
                        setattr(self.state, key, value)

        # 목적지 정보 복원
        if 'goals' in serialized_data:
            self.goal_manager.load_from_serialized(serialized_data['goals'])

            # 현재 목표 위치로 target 정보 업데이트
            current_goal = self.goal_manager.get_current_goal()
            if current_goal:
                self.update_target(current_goal.x, current_goal.y, current_goal.yaw)

# ======================
# Vehicle Manager
# ======================
class VehicleManager:
    """차량들을 관리하는 클래스"""

    def __init__(self, road_manager, vehicle_config=None, physics_config=None, visual_config=None):
        """
        차량 관리자 초기화

        Args:
            vehicle_config: 차량 설정
            physics_config: 물리 설정
            visual_config: 시각화 설정
        """
        self.vehicles = []  # 차량 목록
        self.vehicle_map = {}  # {vehicle_id: vehicle} 매핑
        self.active_vehicle_idx = 0  # 현재 활성화된 차량 인덱스
        self.next_vehicle_id = 0  # 다음 차량 ID (자동 생성용)

        self.road_manager = road_manager  # 도로 관리자

        # 설정 저장
        self.vehicle_config = vehicle_config
        self.physics_config = physics_config
        self.visual_config = visual_config

    def create_vehicle(self, x=0.0, y=0.0, yaw=None, vehicle_id=None, vehicle_config=None,physics_config=None, visual_config=None):
        """
        새 차량 생성 및 추가

        Args:
            x: 초기 X 좌표
            y: 초기 Y 좌표
            yaw: 초기 방향각
            vehicle_id: 차량 ID (None이면 자동 생성)
            vehicle_config: 차량 설정 (None이면 기본값 사용)
            physics_config: 물리 설정 (None이면 기본값 사용)
            visual_config: 시각화 설정 (None이면 기본값 사용)

        Returns:
            생성된 차량 객체
        """
        # ID 할당 (지정되지 않은 경우 자동 생성)
        if vehicle_id is None:
            self.next_vehicle_id = len(self.vehicles)
            vehicle_id = self.next_vehicle_id
            self.next_vehicle_id += 1

        # 설정 사용 (지정되지 않은 경우 기본값 사용)
        v_config = vehicle_config or self.vehicle_config
        p_config = physics_config or self.physics_config
        vis_config = visual_config or self.visual_config

        # 차량 생성
        vehicle = Vehicle(
            vehicle_id=vehicle_id,
            vehicle_config=v_config,
            physics_config=p_config,
            visual_config=vis_config
        )

        # 위치 설정
        vehicle.set_position(x, y, yaw)

        # 차량 목록과 맵에 추가
        self.vehicles.append(vehicle)
        self.vehicle_map[vehicle_id] = vehicle

        # 첫 번째 차량이면 활성화
        if len(self.vehicles) == 1:
            self.active_vehicle_idx = 0

        return vehicle

    def remove_vehicle(self, vehicle_id):
        """
        특정 ID의 차량 제거

        Args:
            vehicle_id: 제거할 차량 ID

        Returns:
            성공 여부 (Boolean)
        """
        # 차량이 존재하는지 확인
        if vehicle_id not in self.vehicle_map:
            return False

        vehicle = self.vehicle_map[vehicle_id]

        # 차량 목록과 맵에서 제거
        self.vehicles.remove(vehicle)
        del self.vehicle_map[vehicle_id]

        # 활성 차량 인덱스 조정
        if len(self.vehicles) > 0:
            self.active_vehicle_idx = min(self.active_vehicle_idx, len(self.vehicles) - 1)
        else:
            self.active_vehicle_idx = 0

        return True

    def reset_vehicle(self, vehicle_id=None):
        """
        특정 ID 또는 모든 차량 초기화

        Args:
            vehicle_id: 초기화할 차량 ID (None이면 모든 차량 초기화)

        Returns:
            성공 여부 (Boolean)
        """
        if vehicle_id is not None:
            # 특정 차량만 초기화
            if vehicle_id in self.vehicle_map:
                self.vehicle_map[vehicle_id].reset()
                return True
            return False
        else:
            # 모든 차량 초기화
            for vehicle in self.vehicles:
                vehicle.reset()
            return True

    def get_vehicle_by_id(self, vehicle_id):
        """
        ID로 차량 찾기

        Args:
            vehicle_id: 찾을 차량 ID

        Returns:
            차량 객체 또는 None
        """
        return self.vehicle_map.get(vehicle_id)

    def get_vehicle_by_index(self, index):
        """
        인덱스로 차량 찾기

        Args:
            index: 찾을 차량 인덱스

        Returns:
            차량 객체 또는 None
        """
        if 0 <= index < len(self.vehicles):
            return self.vehicles[index]
        return None

    def get_all_vehicles(self):
        """
        모든 차량 목록 반환

        Returns:
            차량 객체 리스트
        """
        return self.vehicles

    def get_vehicle_count(self):
        """
        등록된 차량 수 반환

        Returns:
            차량 수
        """
        return len(self.vehicles)

    def set_active_vehicle_by_id(self, vehicle_id):
        """
        ID로 활성 차량 설정

        Args:
            vehicle_id: 활성화할 차량 ID

        Returns:
            성공 여부 (Boolean)
        """
        if vehicle_id not in self.vehicle_map:
            return False

        # ID에 해당하는 차량의 인덱스 찾기
        for i, vehicle in enumerate(self.vehicles):
            if vehicle.id == vehicle_id:
                self.active_vehicle_idx = i
                return True

        return False

    def set_active_vehicle_by_index(self, index):
        """
        인덱스로 활성 차량 설정

        Args:
            index: 활성화할 차량 인덱스

        Returns:
            성공 여부 (Boolean)
        """
        if 0 <= index < len(self.vehicles):
            self.active_vehicle_idx = index
            return True
        return False

    def cycle_active_vehicle(self):
        """
        다음 차량으로 활성 차량 전환

        Returns:
            새로운 활성 차량 인덱스
        """
        if len(self.vehicles) == 0:
            return 0

        self.active_vehicle_idx = (self.active_vehicle_idx + 1) % len(self.vehicles)
        return self.active_vehicle_idx

    def get_active_vehicle(self):
        """
        현재 활성화된 차량 반환

        Returns:
            활성 차량 객체 또는 None
        """
        if len(self.vehicles) == 0:
            return None
        return self.vehicles[self.active_vehicle_idx]

    def get_active_vehicle_index(self):
        """
        현재 활성화된 차량 인덱스 반환

        Returns:
            활성 차량 인덱스
        """
        return self.active_vehicle_idx

    def draw_vehicles(self, screen, world_to_screen_func, debug=False):
        """
        모든 차량 렌더링

        Args:
            screen: pygame 화면 객체
            world_to_screen_func: 월드 좌표를 스크린 좌표로 변환하는 함수
            debug: 디버그 모드 여부
        """
        # 활성 차량 판별
        active_vehicle = self.get_active_vehicle()

        # 모든 차량 그리기 (활성 차량은 표시)
        for vehicle in self.vehicles:
            is_active = (vehicle == active_vehicle)
            vehicle.draw(screen, world_to_screen_func, is_active, debug)

    def step(self, actions, dt, time_elapsed, obstacles=None):
        """
        모든 차량 상태 업데이트

        Args:
            actions: 차량별 액션 (단일 차량이면 단일 액션, 다중 차량이면 액션 리스트)
            dt: 시간 간격
            time_elapsed: 총 경과 시간
            obstacles: 장애물 목록

        Returns:
            collisions: 차량별 충돌 여부 {vehicle_id: bool}
            reached_targets: 차량별 목표 도달 여부 {vehicle_id: bool}
        """
        # 빈 장애물 리스트 초기화
        if obstacles is None:
            obstacles = []

        collisions = {}
        outside_roads = {}
        reached_targets = {}

        # 단일 액션인 경우 (단일 차량 모드)
        if not isinstance(actions, list):
            if len(self.vehicles) > 0:
                vehicle = self.vehicles[0]
                _, collision, outside_road, reached = vehicle.step(actions, dt, time_elapsed, self.road_manager, obstacles, self.vehicles)
                collisions[vehicle.id] = collision
                outside_roads[vehicle.id] = outside_road
                reached_targets[vehicle.id] = reached
        else:
            # 다중 차량 모드
            for i, vehicle in enumerate(self.vehicles):
                if i < len(actions):
                    vehicle_action = actions[i]
                    _, collision, outside_road, reached = vehicle.step(vehicle_action, dt, time_elapsed, self.road_manager, obstacles, self.vehicles)
                    collisions[vehicle.id] = collision
                    outside_roads[vehicle.id] = outside_road
                    reached_targets[vehicle.id] = reached
        return collisions, outside_roads, reached_targets

    def add_goal_for_vehicle(self, vehicle_id, x, y, yaw=0.0, radius=1.0, color=(0, 255, 0)):
        """
        차량에 목적지 추가

        Args:
            vehicle_id: 차량 ID
            x: 목적지 X 좌표
            y: 목적지 Y 좌표
            yaw: 목적지 방향
            radius: 목적지 반경
            color: 목적지 색상 (RGB)

        Returns:
            추가된 목적지 ID 또는 None
        """
        # 차량이 존재하는지 확인
        vehicle = self.get_vehicle_by_id(vehicle_id)
        if not vehicle:
            return None

        # 차량의 목적지 관리자에 목적지 추가
        goal_id = vehicle.add_goal(x, y, yaw, radius, color)
        return goal_id

    def get_serializable_state(self):
        """
        직렬화 가능한 상태 정보 반환

        Returns:
            직렬화 가능한 상태 정보
        """
        # 차량 상태 및 설정 저장
        vehicles_data = []
        for vehicle in self.vehicles:
            # 차량 ID와 상태 저장
            vehicle_data = vehicle.get_serializable_state()
            vehicle_data.update({
                'vehicle_config': vehicle.vehicle_config,
                'physics_config': vehicle.physics_config,
                'visual_config': vehicle.visual_config
            })
            vehicles_data.append(vehicle_data)

        # 직렬화 가능한 데이터
        serialized_data = {
            'vehicles': vehicles_data,
            'active_vehicle_idx': self.active_vehicle_idx,
            'next_vehicle_id': self.next_vehicle_id
        }

        return serialized_data

    def load_from_serialized(self, serialized_data):
        """
        직렬화된 데이터에서 상태 복원

        Args:
            serialized_data: 직렬화된 상태 정보
        """
        # 기존 차량 초기화
        self.vehicles = []
        self.vehicle_map = {}

        # 차량 정보 복원
        if 'vehicles' in serialized_data:
            for vehicle_data in serialized_data['vehicles']:
                vehicle_id = vehicle_data.get('id')
                vehicle_config = vehicle_data.get('vehicle_config', self.vehicle_config)
                physics_config = vehicle_data.get('physics_config', self.physics_config)
                visual_config = vehicle_data.get('visual_config', self.visual_config)

                # 차량 생성
                vehicle = Vehicle(
                    vehicle_id=vehicle_id,
                    vehicle_config=vehicle_config,
                    physics_config=physics_config,
                    visual_config=visual_config
                )

                # 상태 복원 및 목적지 정보 복원
                vehicle.load_from_serialized(vehicle_data)

                # 차량 목록 및 맵에 추가
                self.vehicles.append(vehicle)
                self.vehicle_map[vehicle_id] = vehicle

        # 활성 차량 인덱스 복원
        if 'active_vehicle_idx' in serialized_data:
            self.active_vehicle_idx = serialized_data['active_vehicle_idx']
            # 유효 범위 확인
            if len(self.vehicles) > 0:
                self.active_vehicle_idx = min(self.active_vehicle_idx, len(self.vehicles) - 1)

        # 다음 차량 ID 복원
        if 'next_vehicle_id' in serialized_data:
            self.next_vehicle_id = serialized_data['next_vehicle_id']
