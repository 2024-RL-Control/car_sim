﻿# -*- coding: utf-8 -*-
import pygame
import numpy as np
from math import radians, degrees, pi, cos, sin, atan2, sqrt
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Union
from abc import ABC, abstractmethod

# ======================
# 센서 데이터 기본 클래스
# ======================
@dataclass
class SensorData(ABC):
    """모든 센서 데이터의 기본 클래스"""
    timestamp: float  # 측정 시간 [s]
    sensor_pose: Tuple[float, float, float]  # 센서 위치 및 방향 (x, y, yaw)

# ======================
# 센서 인터페이스 정의
# ======================
class BaseSensor(ABC):
    """
    모든 센서의 기본 클래스
    센서의 공통 속성 및 메서드 정의
    """
    def __init__(self, vehicle, relative_pos=(0, 0), relative_angle=0, sensor_id=None):
        """
        기본 센서 초기화

        Args:
            vehicle: 센서가 부착된 차량 객체
            relative_pos: 차량 중심으로부터의 상대적 위치 (x, y) [m]
            relative_angle: 차량 방향에 대한 상대적 각도 [rad]
            sensor_id: 센서 고유 ID (None인 경우 자동 생성)
        """
        self.sensor_id = sensor_id if sensor_id is not None else id(self)
        self.sensor_type = self.__class__.__name__
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

    @abstractmethod
    def update(self, dt, time_elapsed=0, obstacles=None, vehicles=None):
        """
        센서 상태 업데이트 (반드시 구현 필요)

        Args:
            dt: 시간 간격 [s]
            time_elapsed: 시뮬레이션 누적 시간 [s]
            obstacles: 장애물 외접원
            vehicles: 차량 리스트

        Returns:
            측정이 수행되었는지 여부 (Boolean)
        """
        pass

    @abstractmethod
    def get_data(self):
        """현재 센서 데이터 반환 (반드시 구현 필요)"""
        pass

    def draw(self, screen, world_to_screen_func, debug=False):
        """
        센서 시각화 (필요시 오버라이드)

        Args:
            screen: pygame 화면 객체
            world_to_screen_func: 월드 좌표를 화면 좌표로 변환하는 함수
            debug: 디버그 정보 표시 여부
        """
        pass

    def reset(self):
        """센서 상태 및 데이터 초기화 (필요시 오버라이드)"""
        pass

    def get_config(self):
        """센서 설정 반환 (필요시 오버라이드)"""
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type,
            "relative_pos": self.relative_pos,
            "relative_angle": self.relative_angle
        }

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
class LidarData(SensorData):
    """
    라이다 스캔 데이터 저장 클래스
    """
    ranges: List[LidarMeasurement] = field(default_factory=list)  # 거리 측정값 리스트

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
    def __init__(self, vehicle, sensor_config=None, visual_config=None):
        """
        라이다 센서 초기화

        Args:
            vehicle: 센서가 부착된 차량 객체
            sensor_config: 센서 설정 딕셔너리
            visual_config: 시각화 설정 딕셔너리
        """
        # config에서 필요한 값 추출
        sensor_id = sensor_config.get('sensor_id', None)
        relative_pos = tuple(sensor_config.get('relative_pos', (0, 0)))
        relative_angle = radians(sensor_config.get('relative_angle', 0.0))

        # 기본 센서 초기화
        super().__init__(vehicle, relative_pos, relative_angle, sensor_id)

        # 라이다 속성 설정
        self.num_samples = int(sensor_config.get('num_samples', 360))
        self.angle_start = radians(sensor_config.get('angle_start', -180.0))
        self.angle_end = radians(sensor_config.get('angle_end', 180.0))
        self.max_range = float(sensor_config.get('max_range', 50.0))
        self.min_range = float(sensor_config.get('min_range', 0.1))
        self.scan_rate = int(sensor_config.get('scan_rate', 10))
        self.scan_interval = 1.0 / self.scan_rate if self.scan_rate > 0 else 0

        # 노이즈 파라미터 설정
        self.noise_params = {
            'gaussian_sigma': float(sensor_config.get('noise_gaussian_sigma', 0.05)),
            'distance_factor': float(sensor_config.get('noise_distance_factor', 0.01))
        }

        # 센서 상태 및 데이터
        self.last_scan_time = 0
        self.current_data = None
        scan_history_len = int(sensor_config.get('scan_history', 10))
        self.scan_history = deque(maxlen=scan_history_len) if scan_history_len > 0 else None

        # 레이 방향 벡터 미리 계산 (최적화)
        self._precompute_ray_directions()

        # 시각화 리소스
        self._visualization_cache = {}
        self.sensor_color = tuple(visual_config.get('lidar_sensor_color', (0, 200, 255)))
        self.ray_success_color = tuple(visual_config.get('lidar_ray_success_color', (255, 50, 50)))
        self.ray_failure_color = tuple(visual_config.get('lidar_ray_failure_color', (150, 150, 150)))
        self.sensor_data_color = tuple(visual_config.get('lidar_sensor_data_color', (255, 0, 0)))
        self.sensor_history_data_color = tuple(visual_config.get('lidar_sensor_history_data_color', (0, 100, 200, 100)))

    def get_config(self):
        """라이다 센서 설정 반환"""
        config = super().get_config()
        config.update({
            "num_samples": self.num_samples,
            "angle_start": degrees(self.angle_start),
            "angle_end": degrees(self.angle_end),
            "max_range": self.max_range,
            "min_range": self.min_range,
            "scan_rate": self.scan_rate,
            "noise_gaussian_sigma": self.noise_params['gaussian_sigma'],
            "noise_distance_factor": self.noise_params['distance_factor'],
            "scan_history": self.scan_history.maxlen if self.scan_history else 0
        })
        return config

    def reset(self):
        """센서 상태 및 데이터 초기화"""
        self.last_scan_time = 0
        self.current_data = None
        if self.scan_history:
            self.scan_history.clear()

    def _precompute_ray_directions(self):
        """
        레이 방향 벡터 미리 계산하여 저장 (최적화)
        """
        self.ray_angles = np.linspace(self.angle_start, self.angle_end, self.num_samples)
        self.ray_directions = np.array([(np.cos(angle), np.sin(angle)) for angle in self.ray_angles])

    def update(self, dt, time_elapsed=0, objects=None):
        """
        라이다 센서 업데이트 및 스캔 수행

        Args:
            dt: 시간 간격 [s]
            time_elapsed: 시뮬레이션 누적 시간 [s]
            objects: 객체 외접원

        Returns:
            측정이 수행되었는지 여부 (Boolean)
        """
        # 스캔 주기에 따른 업데이트 수행
        if time_elapsed - self.last_scan_time >= self.scan_interval:
            if len(objects) > 0:
                # 스캔 수행
                self.current_data = self._perform_scan(time_elapsed, objects)

                # 스캔 결과 히스토리 저장
                if self.scan_history is not None:
                    self.scan_history.append(self.current_data)

                self.last_scan_time = time_elapsed
                return True

        return False

    def _perform_scan(self, timestamp, objects):
        """
        라이다 스캔 수행 - 벡터화된 연산으로 구현

        Args:
            timestamp: 현재 시간 [s]
            objects: 객체 외접원
        Returns:
            LidarData: 라이다 스캔 결과
        """
        # 센서의 현재 위치 및 방향 얻기
        sensor_x, sensor_y, sensor_yaw = self.get_pose()

        # 모든 레이에 대한 글로벌 각도 벡터화 계산
        global_angles = sensor_yaw + self.ray_angles
        global_dirs_x = np.cos(global_angles)
        global_dirs_y = np.sin(global_angles)

        # 벡터화된 레이캐스팅 수행
        closest_hits = self._raycast_vectorized(
            sensor_x, sensor_y,
            global_dirs_x, global_dirs_y,
            global_angles,
            objects
        )

        # 라이다 데이터 생성
        return LidarData(
            timestamp=timestamp,
            sensor_pose=(sensor_x, sensor_y, sensor_yaw),
            ranges=closest_hits
        )

    def _raycast_vectorized(self, origin_x, origin_y, dirs_x, dirs_y, global_angles, objects):
        """
        벡터화된 레이캐스팅 수행

        Args:
            origin_x, origin_y: 레이 시작점 [m]
            dirs_x, dirs_y: 레이 방향 벡터 배열 (정규화됨)
            global_angles: 레이의 글로벌 각도 배열
            objects: 객체 외접원 리스트

        Returns:
            List[LidarMeasurement]: 모든 레이에 대한 측정 결과
        """
        # 결과를 저장할 배열 초기화
        num_rays = len(dirs_x)
        min_distances = np.ones(num_rays) * self.max_range

        # 장애물이 없는 경우
        if len(objects) == 0:
            # 기본 최대 거리 배열 반환
            return self._create_measurements(
                min_distances, global_angles, origin_x, origin_y,
                dirs_x, dirs_y, min_distances
            )

        # 장애물 데이터를 NumPy 배열로 변환
        object_array = np.array(objects)

        # 각 레이와 모든 장애물에 대해 한 번에 계산
        for i in range(len(object_array)):
            circle_x, circle_y, circle_radius = object_array[i]

            # 원의 중심에서 레이 시작점까지의 벡터 (모든 레이에 대해 동일)
            to_circle_x = circle_x - origin_x
            to_circle_y = circle_y - origin_y

            # 레이 방향으로의 투영 거리 벡터화 계산
            proj_dists = to_circle_x * dirs_x + to_circle_y * dirs_y

            # 원이 레이 뒤에 있는 경우 마스크
            valid_rays = proj_dists >= 0

            if not np.any(valid_rays):
                continue

            # 레이와 원 중심 사이의 수직 거리 제곱 벡터화 계산
            perp_dist_sq = np.zeros_like(proj_dists)
            perp_dist_sq[valid_rays] = (to_circle_x**2 + to_circle_y**2) - proj_dists[valid_rays]**2

            # 수직 거리가 원 반지름보다 큰 경우 마스크
            valid_rays &= (perp_dist_sq <= circle_radius**2)

            if not np.any(valid_rays):
                continue

            # 레이-원 교차점까지의 거리 계산
            dist_to_intersections = np.zeros_like(proj_dists)
            dist_to_intersections[valid_rays] = proj_dists[valid_rays] - np.sqrt(
                circle_radius**2 - perp_dist_sq[valid_rays]
            )

            # 최소 범위 밖인 경우 마스크
            valid_rays &= (dist_to_intersections >= self.min_range)

            if not np.any(valid_rays):
                continue

            # 최소 거리 업데이트
            min_distances = np.where(
                (valid_rays) & (dist_to_intersections < min_distances),
                dist_to_intersections,
                min_distances
            )

        # 노이즈 적용 및 히트 지점 계산 (벡터화)
        noisy_distances = self._apply_noise_vectorized(min_distances)

        # 측정 결과 생성
        return self._create_measurements(
            noisy_distances, global_angles, origin_x, origin_y,
            dirs_x, dirs_y, min_distances
        )

    def _apply_noise_vectorized(self, distances):
        """
        측정 거리 배열에 노이즈 벡터화 적용

        Args:
            distances: 원본 측정 거리 배열 [m]

        Returns:
            ndarray: 노이즈가 적용된 거리 배열 [m]
        """
        # 최대 거리 마스크 (감지 실패)
        max_dist_mask = distances >= self.max_range

        # 노이즈 계산 (감지 성공한 경우만)
        valid_mask = ~max_dist_mask
        valid_count = np.sum(valid_mask)

        if valid_count == 0:
            return distances

        # 결과 배열 초기화
        noisy_distances = distances.copy()

        if valid_count > 0:
            # 기본 가우시안 노이즈
            gaussian_sigma = self.noise_params['gaussian_sigma']
            noise = np.random.normal(0, gaussian_sigma, valid_count)

            # 거리에 따른 노이즈 증가
            distance_factor = self.noise_params['distance_factor']
            distance_noise = np.random.normal(0, distances[valid_mask] * distance_factor, valid_count)

            # 노이즈 적용
            noisy_distances[valid_mask] = distances[valid_mask] + noise + distance_noise

            # 범위 제한
            noisy_distances[valid_mask] = np.clip(
                noisy_distances[valid_mask],
                self.min_range,
                self.max_range
            )

        return noisy_distances

    def _create_measurements(self, distances, angles, origin_x, origin_y, dirs_x, dirs_y, raw_distances):
        """
        LidarMeasurement 객체 리스트 생성

        Args:
            distances: 노이즈가 적용된 거리 배열
            angles: 글로벌 각도 배열
            origin_x, origin_y: 레이 시작점
            dirs_x, dirs_y: 방향 벡터 배열
            raw_distances: 원본 거리 배열

        Returns:
            List[LidarMeasurement]: 측정 결과 리스트
        """
        # 센서 기준 로컬 각도 계산
        local_angles = angles - (self.get_pose()[2] - self.relative_angle)

        # 히트 지점 계산
        hit_x = origin_x + dirs_x * distances
        hit_y = origin_y + dirs_y * distances

        # 측정 결과 생성
        measurements = []
        for i in range(len(distances)):
            measurements.append(LidarMeasurement(
                distance=distances[i],
                angle=angles[i],
                local_angle=local_angles[i],
                hit_x=hit_x[i],
                hit_y=hit_y[i],
                raw_distance=raw_distances[i]
            ))

        return measurements

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
        pygame.draw.circle(screen, self.sensor_color, sensor_screen_pos, 3)

        # 현재 스캔 결과 그리기
        for i, hit in enumerate(self.current_data.ranges):
            hit_screen_pos = world_to_screen_func(hit.hit_x, hit.hit_y)


            # 감지 실패(최대 거리)인 경우와 성공한 경우 색상 다르게
            ray_color = self.ray_success_color if hit.distance < self.max_range else self.ray_failure_color
            pygame.draw.line(screen, ray_color, sensor_screen_pos, hit_screen_pos, 1)

            # 히트 포인트 그리기
            if hit.distance < self.max_range:
                pygame.draw.circle(screen, self.sensor_data_color, hit_screen_pos, 2)

        # 디버그 모드: 추가 정보 표시
        if debug:
            # 히스토리 그리기 (있는 경우)
            if self.scan_history:
                for scan in self.scan_history:
                    for hit in scan.ranges:
                        hit_screen_pos = world_to_screen_func(hit.hit_x, hit.hit_y)
                        # 작은 반투명 점으로 히스토리 표시
                        pygame.draw.circle(screen, self.sensor_history_data_color, hit_screen_pos, 1)

    def _apply_noise(self, distance):
        """
        측정 거리에 노이즈 적용 (호환성 유지)

        Args:
            distance: 원본 측정 거리 [m]

        Returns:
            float: 노이즈가 적용된 거리 [m]
        """
        # 단일 거리 값을 배열로 변환
        distance_array = np.array([distance])

        # 벡터화된 노이즈 함수 호출
        noisy_distance_array = self._apply_noise_vectorized(distance_array)

        # 단일 값 반환
        return noisy_distance_array[0]

    def _raycast(self, origin_x, origin_y, direction, objects):
        """
        단일 레이에 대한 레이캐스팅 수행 (라인-원 교차 검출)

        Args:
            origin_x, origin_y: 레이 시작점 [m]
            direction: 레이 방향 벡터 (dx, dy)
            objects: 객체 외접원 리스트

        Returns:
            LidarMeasurement: 측정 결과
        """
        # 방향 벡터 정규화
        dir_x, dir_y = direction
        dir_norm = np.sqrt(dir_x**2 + dir_y**2)
        dir_x, dir_y = dir_x / dir_norm, dir_y / dir_norm

        # 레이의 글로벌 각도
        global_angle = np.arctan2(dir_y, dir_x)

        # 벡터화된 함수를 사용하여 계산
        dirs_x = np.array([dir_x])
        dirs_y = np.array([dir_y])
        global_angles = np.array([global_angle])

        # 벡터화된 레이캐스팅 호출
        measurements = self._raycast_vectorized(
            origin_x, origin_y, dirs_x, dirs_y, global_angles, objects
        )

        # 첫 번째 (유일한) 측정값 반환
        return measurements[0]

# ======================
# 센서 매니저 구현
# ======================
class SensorManager:
    """
    차량 내부에서 센서들을 관리하는 클래스
    """
    def __init__(self, vehicle, sensor_config, visual_config):
        """
        센서 매니저 초기화

        Args:
            vehicle: 센서가 부착될 차량 객체
            sensor_config: 센서 설정 딕셔너리
            visual_config: 시각화 설정 딕셔너리
        """
        self.vehicle = vehicle
        self.sensor_config = sensor_config
        self.visual_config = visual_config
        self.sensors = {}
        self.sensor_counter = 0

        self._initialize_sensors()

    def _initialize_sensors(self):
        """
        config에서 센서 설정을 로드하여 초기화
        """

        for sensor_id_str, sensor_config in self.sensor_config.items():
            # 센서 타입에 따라 다른 센서 클래스 생성
            sensor_type = sensor_config.get('sensor_type', '').upper()

            if sensor_type == 'LIDAR':
                # 라이다 센서 생성
                self.add_sensor(LidarSensor, sensor_config, self.visual_config)

    def reset(self):
        """센서 상태 초기화"""
        for sensor in self.sensors.values():
            sensor.reset()

    def add_sensor(self, sensor_type, sensor_config=None, *args, **kwargs):
        """
        센서 추가

        Args:
            sensor_type: 센서 클래스 (예: LidarSensor)
            sensor_config: 센서 설정 딕셔너리
            *args, **kwargs: 추가 센서 초기화 인자

        Returns:
            int: 추가된 센서의 ID
        """
        # 센서 ID 생성
        sensor_id = self.sensor_counter
        self.sensor_counter += 1

        # 센서 객체 생성 - 설정 객체 전달
        sensor = sensor_type(self.vehicle, sensor_config, *args, **kwargs)

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

    def get_all_sensors(self):
        """
        모든 센서 객체 목록 반환

        Returns:
            dict: 센서 ID를 키로 하는 센서 객체 사전
        """
        return self.sensors

    def update(self, dt, time_elapsed=0, objects=[]):
        """
        모든 센서 업데이트

        Args:
            dt: 시간 간격 [s]
            time_elapsed: 시뮬레이션 누적 시간 [s]
            objects: 객체 외접원 리스트
        """
        for sensor in self.sensors.values():
            sensor.update(dt, time_elapsed, objects)

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
