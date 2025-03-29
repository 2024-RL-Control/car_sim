# -*- coding: utf-8 -*-
import pygame
import numpy as np
from math import radians, degrees, pi, cos, sin, atan2, sqrt
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Union

# ======================
# 센서 인터페이스 정의
# ======================
class BaseSensor:
    """
    모든 센서의 기본 클래스
    센서의 공통 속성 및 메서드 정의
    """
    def __init__(self, vehicle, relative_pos=(0, 0), relative_angle=0):
        """
        기본 센서 초기화

        Args:
            vehicle: 센서가 부착된 차량 객체
            relative_pos: 차량 중심으로부터의 상대적 위치 (x, y) [m]
            relative_angle: 차량 방향에 대한 상대적 각도 [rad]
        """
        self.vehicle = vehicle
        self.relative_pos = relative_pos
        self.relative_angle = relative_angle

        # 캐싱된 절대 위치 및 방향
        self._cached_pos = None
        self._cached_dir = None
        self._cached_state_sig = None  # 차량 상태 시그니처 (캐싱용)

    def get_pose(self):
        """
        센서의 현재 절대 위치 및 방향 반환
        캐싱을 통한 반복 계산 최소화

        Returns:
            tuple: (x, y, yaw) - 센서의 월드 좌표 및 방향 [m, m, rad]
        """
        # 차량 상태의 시그니처 계산 (position + orientation)
        v_state = self.vehicle.state
        current_sig = (v_state.x, v_state.y, v_state.yaw)

        # 상태가 변경되지 않았으면 캐시된 값 반환
        if self._cached_state_sig == current_sig and self._cached_pos is not None:
            return self._cached_pos

        # 상대 위치를 차량 기준 좌표계에서 글로벌 좌표계로 변환
        rel_x, rel_y = self.relative_pos
        v_yaw = v_state.yaw

        # 회전 변환 적용
        rot_x = rel_x * cos(v_yaw) - rel_y * sin(v_yaw)
        rot_y = rel_x * sin(v_yaw) + rel_y * cos(v_yaw)

        # 센서의 글로벌 좌표 및 방향
        sensor_x = v_state.x + rot_x
        sensor_y = v_state.y + rot_y
        sensor_yaw = v_state.yaw + self.relative_angle

        # 결과 캐싱
        self._cached_pos = (sensor_x, sensor_y, sensor_yaw)
        self._cached_state_sig = current_sig

        return self._cached_pos

    def update(self, dt):
        """
        센서 상태 업데이트 (오버라이드용)

        Args:
            dt: 시간 간격 [s]
        """
        pass

    def draw(self, screen, world_to_screen_func, debug=False):
        """
        센서 시각화 (오버라이드용)

        Args:
            screen: pygame 화면 객체
            world_to_screen_func: 월드 좌표를 화면 좌표로 변환하는 함수
            debug: 디버그 정보 표시 여부
        """
        pass

# ======================
# 라이다 센서 구현
# ======================
@dataclass
class LidarMeasurement:
    """
    단일 라이다 측정값 저장 클래스
    """
    distance: float          # 거리 [m]
    angle: float             # 글로벌 각도 [rad]
    local_angle: float       # 센서 기준 각도 [rad]
    hit_x: float             # 히트 지점 X 좌표 [m]
    hit_y: float             # 히트 지점 Y 좌표 [m]
    raw_distance: float      # 노이즈 적용 전 거리 [m]

@dataclass
class LidarData:
    """
    라이다 스캔 데이터 저장 클래스
    """
    timestamp: float                           # 측정 시간 [s]
    sensor_pose: Tuple[float, float, float]    # 센서 위치 및 방향 (x, y, yaw)
    ranges: List[LidarMeasurement]             # 거리 측정값 리스트

    def get_points(self) -> List[Tuple[float, float]]:
        """측정 포인트(히트 지점) 목록 반환"""
        return [(m.hit_x, m.hit_y) for m in self.ranges]

    def get_distances(self) -> List[float]:
        """거리 측정값 목록 반환"""
        return [m.distance for m in self.ranges]

class LidarSensor(BaseSensor):
    """
    2D 라이다 센서 클래스
    레이캐스팅을 통한 거리 측정 구현
    """
    def __init__(self, vehicle,
                 num_samples=180,         # 샘플(레이) 수
                 angle_start=-np.pi/2,    # 시작 각도 [rad]
                 angle_end=np.pi/2,       # 종료 각도 [rad]
                 max_range=50.0,          # 최대 감지 거리 [m]
                 min_range=0.1,           # 최소 감지 거리 [m]
                 scan_rate=10,            # 스캔 주기 [Hz]
                 relative_pos=(0, 0),     # 차량 기준 상대 위치 [m]
                 relative_angle=0,        # 차량 기준 상대 각도 [rad]
                 noise_params=None,       # 노이즈 파라미터
                 draw_rays=True,          # 레이 시각화 여부
                 scan_point_history=0):   # 스캔 히스토리 개수
        """
        라이다 센서 초기화

        Args:
            vehicle: 센서가 부착된 차량 객체
            num_samples: 샘플(레이) 수
            angle_start: 시작 각도 [rad] (센서 기준 왼쪽)
            angle_end: 종료 각도 [rad] (센서 기준 오른쪽)
            max_range: 최대 감지 거리 [m]
            min_range: 최소 감지 거리 [m]
            scan_rate: 스캔 주기 [Hz]
            relative_pos: 차량 중심으로부터의 상대적 위치 (x, y) [m]
            relative_angle: 차량 방향에 대한 상대적 각도 [rad]
            noise_params: 노이즈 파라미터 (dict)
                - gaussian_sigma: 가우시안 노이즈 표준편차 [m]
                - distance_factor: 거리에 따른 노이즈 증가 계수 [m/m]
            draw_rays: 레이 시각화 여부
            scan_point_history: 저장할 스캔 히스토리 개수
        """
        super().__init__(vehicle, relative_pos, relative_angle)

        self.num_samples = num_samples
        self.angle_start = angle_start
        self.angle_end = angle_end
        self.max_range = max_range
        self.min_range = min_range
        self.scan_rate = scan_rate
        self.scan_interval = 1.0 / scan_rate if scan_rate > 0 else 0
        self.draw_rays = draw_rays

        # 노이즈 파라미터 설정
        self.noise_params = noise_params or {
            'gaussian_sigma': 0.05,     # 기본 가우시안 노이즈 표준편차 [m]
            'distance_factor': 0.01,    # 거리에 따른 노이즈 증가 계수 [m/m]
        }

        # 센서 상태 및 데이터
        self.last_scan_time = 0
        self.current_data = None
        self.scan_history = deque(maxlen=scan_point_history) if scan_point_history > 0 else None

        # 레이 방향 벡터 미리 계산 (최적화)
        self._precompute_ray_directions()

        # 시각화 리소스
        self._visualization_cache = {}

    def _precompute_ray_directions(self):
        """
        레이 방향 벡터 미리 계산하여 저장 (최적화)
        """
        self.ray_angles = np.linspace(self.angle_start, self.angle_end, self.num_samples)
        self.ray_directions = np.array([(np.cos(angle), np.sin(angle)) for angle in self.ray_angles])

    def update(self, dt, obstacle_manager=None, time_elapsed=0):
        """
        라이다 센서 업데이트 및 스캔 수행

        Args:
            dt: 시간 간격 [s]
            obstacle_manager: 장애물 관리 객체
            time_elapsed: 시뮬레이션 누적 시간 [s]

        Returns:
            측정이 수행되었는지 여부 (Boolean)
        """
        # 스캔 주기에 따른 업데이트 수행
        if time_elapsed - self.last_scan_time >= self.scan_interval:
            if obstacle_manager:
                # 스캔 수행
                self.current_data = self._perform_scan(obstacle_manager, time_elapsed)

                # 스캔 결과 히스토리 저장
                if self.scan_history is not None:
                    self.scan_history.append(self.current_data)

                self.last_scan_time = time_elapsed
                return True

        return False

    def _perform_scan(self, obstacle_manager, timestamp):
        """
        라이다 스캔 수행

        Args:
            obstacle_manager: 장애물 관리 객체
            timestamp: 현재 시간 [s]

        Returns:
            LidarData: 라이다 스캔 결과
        """
        # 센서의 현재 위치 및 방향 얻기
        sensor_x, sensor_y, sensor_yaw = self.get_pose()

        # 모든 레이에 대한 측정 수행
        measurements = []

        # 장애물 목록 가져오기
        obstacles = obstacle_manager.get_all_outer_circles()

        # 각 레이에 대해 가장 가까운 충돌 지점 찾기
        for i, (local_angle, ray_dir) in enumerate(zip(self.ray_angles, self.ray_directions)):
            # 센서 좌표계의 레이 방향을 글로벌 좌표계로 변환
            global_angle = sensor_yaw + local_angle
            global_dir = (np.cos(global_angle), np.sin(global_angle))

            # 레이캐스팅 수행
            closest_hit = self._raycast(sensor_x, sensor_y, global_dir, obstacles)

            # 측정값 생성 및 추가
            measurements.append(closest_hit)

        # 라이다 데이터 생성
        return LidarData(
            timestamp=timestamp,
            sensor_pose=(sensor_x, sensor_y, sensor_yaw),
            ranges=measurements
        )

    def _raycast(self, origin_x, origin_y, direction, obstacle):
        """
        단일 레이에 대한 레이캐스팅 수행 (라인-원 교차 검출)

        Args:
            origin_x, origin_y: 레이 시작점 [m]
            direction: 레이 방향 벡터 (dx, dy)
            obstacle: 장애물물 객체

        Returns:
            LidarMeasurement: 측정 결과
        """
        # 방향 벡터 정규화
        dir_x, dir_y = direction
        dir_norm = np.sqrt(dir_x**2 + dir_y**2)
        dir_x, dir_y = dir_x / dir_norm, dir_y / dir_norm

        # 레이의 글로벌 각도
        global_angle = np.arctan2(dir_y, dir_x)

        # 기본값: 최대 범위까지 감지 실패
        min_dist = self.max_range

        # 각 원에 대해 레이-원 충돌 검사 (라인-원 교차 감지)
        for circle_x, circle_y, circle_radius in obstacle:
            # 원의 중심에서 레이 시작점까지의 벡터
            to_circle_x = circle_x - origin_x
            to_circle_y = circle_y - origin_y

            # 레이 방향으로의 투영 거리 (레이 위의 가장 가까운 점까지의 거리)
            proj_dist = to_circle_x * dir_x + to_circle_y * dir_y

            # 원이 레이 뒤에 있으면 무시
            if proj_dist < 0:
                continue

            # 레이와 원 중심 사이의 수직 거리 제곱
            perp_dist_sq = (to_circle_x**2 + to_circle_y**2) - proj_dist**2

            # 수직 거리가 원 반지름보다 크면 충돌 없음
            if perp_dist_sq > circle_radius**2:
                continue

            # 레이-원 교차점까지의 거리 계산 (라인-원 교차 공식)
            dist_to_intersection = proj_dist - np.sqrt(circle_radius**2 - perp_dist_sq)

            # 최소 범위 밖이면 무시
            if dist_to_intersection < self.min_range:
                continue

            # 최대 범위 안이고, 현재까지의 최소 거리보다 가까우면 업데이트
            if dist_to_intersection < min_dist:
                min_dist = dist_to_intersection

        # 노이즈 적용
        noisy_distance = self._apply_noise(min_dist)

        # 최종 히트 지점 계산 (노이즈 적용 후)
        noisy_hit_point = (
            origin_x + dir_x * noisy_distance,
            origin_y + dir_y * noisy_distance
        )

        # 측정 결과 생성
        return LidarMeasurement(
            distance=noisy_distance,
            angle=global_angle,
            local_angle=global_angle - (self.get_pose()[2] - self.relative_angle),
            hit_x=noisy_hit_point[0],
            hit_y=noisy_hit_point[1],
            raw_distance=min_dist
        )

    def _apply_noise(self, distance):
        """
        측정 거리에 노이즈 적용

        Args:
            distance: 원본 측정 거리 [m]

        Returns:
            float: 노이즈가 적용된 거리 [m]
        """
        # 최대 거리인 경우 노이즈 적용하지 않음 (감지 실패)
        if distance >= self.max_range:
            return distance

        # 기본 가우시안 노이즈
        gaussian_sigma = self.noise_params['gaussian_sigma']
        noise = np.random.normal(0, gaussian_sigma)

        # 거리에 따른 노이즈 증가
        distance_factor = self.noise_params['distance_factor']
        distance_noise = np.random.normal(0, distance * distance_factor)

        # 최종 노이즈 적용
        noisy_dist = distance + noise + distance_noise

        # 범위 제한
        return np.clip(noisy_dist, self.min_range, self.max_range)

    def get_data(self):
        """
        최신 스캔 데이터 반환

        Returns:
            LidarData: 최신 라이다 스캔 데이터 또는 None
        """
        return self.current_data

    def get_point_cloud(self):
        """
        최신 스캔의 포인트 클라우드 반환

        Returns:
            List[Tuple[float, float]]: 포인트 클라우드 좌표 목록 또는 빈 리스트
        """
        if self.current_data:
            return self.current_data.get_points()
        return []

    def draw(self, screen, world_to_screen_func, debug=False):
        """
        라이다 센서 및 측정 결과 시각화

        Args:
            screen: pygame 화면 객체
            world_to_screen_func: 월드 좌표를 화면 좌표로 변환하는 함수
            debug: 디버그 정보 표시 여부
        """
        if not self.current_data:
            return

        # 센서 위치
        sensor_x, sensor_y, _ = self.current_data.sensor_pose
        sensor_screen_pos = world_to_screen_func(sensor_x, sensor_y)

        # 센서 본체 그리기
        pygame.draw.circle(screen, (0, 200, 255), sensor_screen_pos, 3)

        # 히스토리 그리기 (있는 경우)
        if self.scan_history:
            for scan in self.scan_history:
                for hit in scan.ranges:
                    hit_screen_pos = world_to_screen_func(hit.hit_x, hit.hit_y)
                    # 작은 반투명 점으로 히스토리 표시
                    pygame.draw.circle(screen, (0, 100, 200, 100), hit_screen_pos, 1)

        # 현재 스캔 결과 그리기
        for i, hit in enumerate(self.current_data.ranges):
            hit_screen_pos = world_to_screen_func(hit.hit_x, hit.hit_y)

            # 레이 그리기 (옵션)
            if self.draw_rays:
                # 감지 실패(최대 거리)인 경우와 성공한 경우 색상 다르게
                ray_color = (255, 50, 50) if hit.distance < self.max_range else (150, 150, 150)
                pygame.draw.line(screen, ray_color, sensor_screen_pos, hit_screen_pos, 1)

            # 히트 포인트 그리기
            if hit.distance < self.max_range:
                pygame.draw.circle(screen, (255, 0, 0), hit_screen_pos, 2)

        # 디버그 모드: 추가 정보 표시
        if debug:
            # 스캔 각도 범위 표시
            start_angle = self.current_data.sensor_pose[2] + self.angle_start
            end_angle = self.current_data.sensor_pose[2] + self.angle_end
            radius = 10  # 각도 표시 아크의 반지름

            # 시작 방향 표시
            start_dir_x = sensor_screen_pos[0] + radius * np.cos(start_angle)
            start_dir_y = sensor_screen_pos[1] - radius * np.sin(start_angle)
            pygame.draw.line(screen, (0, 255, 0), sensor_screen_pos, (start_dir_x, start_dir_y), 2)

            # 종료 방향 표시
            end_dir_x = sensor_screen_pos[0] + radius * np.cos(end_angle)
            end_dir_y = sensor_screen_pos[1] - radius * np.sin(end_angle)
            pygame.draw.line(screen, (0, 255, 0), sensor_screen_pos, (end_dir_x, end_dir_y), 2)

# ======================
# 센서 매니저 구현
# ======================
class SensorManager:
    """
    차량에 부착된 센서들을 관리하는 클래스
    """
    def __init__(self, vehicle):
        """
        센서 매니저 초기화

        Args:
            vehicle: 센서가 부착될 차량 객체
        """
        self.vehicle = vehicle
        self.sensors = {}
        self.sensor_counter = 0

    def add_sensor(self, sensor_type, *args, **kwargs):
        """
        센서 추가

        Args:
            sensor_type: 센서 클래스 (예: LidarSensor)
            *args, **kwargs: 센서 초기화 인자

        Returns:
            int: 추가된 센서의 ID
        """
        # 센서 ID 생성
        sensor_id = self.sensor_counter
        self.sensor_counter += 1

        # 센서 객체 생성
        sensor = sensor_type(self.vehicle, *args, **kwargs)

        # 센서 목록에 추가
        self.sensors[sensor_id] = sensor

        return sensor_id

    def remove_sensor(self, sensor_id):
        """
        센서 제거

        Args:
            sensor_id: 제거할 센서의 ID

        Returns:
            bool: 성공 여부
        """
        if sensor_id in self.sensors:
            del self.sensors[sensor_id]
            return True
        return False

    def get_sensor_data(self, sensor_id):
        """
        센서 데이터 가져오기
        Args:
            sensor_id: 가져올 센서의 ID

        Returns:
            List: 센서 데이터 (예: LidarData) 또는 None
        """
        sensor = self.get_sensor(sensor_id)
        if sensor:
            return sensor.get_data()
        return None

    def get_sensor(self, sensor_id):
        """
        센서 객체 가져오기

        Args:
            sensor_id: 가져올 센서의 ID

        Returns:
            BaseSensor: 센서 객체 또는 None
        """
        return self.sensors.get(sensor_id)

    def update(self, dt, obstacle_manager=None, time_elapsed=0):
        """
        모든 센서 업데이트

        Args:
            dt: 시간 간격 [s]
            obstacle_manager: 장애물 관리 객체
            time_elapsed: 시뮬레이션 누적 시간 [s]
        """
        for sensor in self.sensors.values():
            sensor.update(dt, obstacle_manager, time_elapsed)

    def draw(self, screen, world_to_screen_func, debug=False):
        """
        모든 센서 시각화

        Args:
            screen: pygame 화면 객체
            world_to_screen_func: 월드 좌표를 화면 좌표로 변환하는 함수
            debug: 디버그 정보 표시 여부
        """
        for sensor in self.sensors.values():
            sensor.draw(screen, world_to_screen_func, debug)
