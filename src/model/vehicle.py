# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from collections import deque, OrderedDict
from math import degrees, pi, cos, sin, log, e, exp, sqrt
import pygame
import numpy as np
import time
import weakref
import threading
from .physics import PhysicsEngine
from .trajectory import TrajectoryPredictor, TrajectoryData
from .object import RectangleObstacle, GoalManager
from .sensor import SensorManager

# ======================
# Subsystem Management System
# ======================
class SubsystemManager:
    """차량 서브시스템의 업데이트 빈도를 통합 관리하는 클래스"""

    def __init__(self, vehicle_instance=None, simulation_config=None):
        """서브 시스템 관리자 초기화

        Args:
            vehicle_instance: Vehicle 인스턴스 참조 (콜백에서 사용)
            simulation_config: 시뮬레이션 설정 (주파수 등)
        """
        # 순환 참조 방지를 위한 약한 참조 사용
        self._vehicle_ref = weakref.ref(vehicle_instance) if vehicle_instance else None

        self._update_intervals = {}  # {subsystem_name: interval}
        self._last_update_times = {}  # {subsystem_name: last_time}
        self._adaptive_configs = {}  # {subsystem_name: adaptive_config}

        # 콜백 및 캐시 관리
        self._update_callbacks = {}  # {subsystem_name: callback_function}
        self._initialization_callbacks = {}  # {subsystem_name: init_callback}

        # 메모리 관리 설정 (기본값 설정 후 config에서 로드)
        self._cache_config = {
            'max_cache_size': 1000,
            'cleanup_interval': 60.0,  # seconds
            'memory_threshold_mb': 100,
            'max_cache_age': 3600.0
        }

        # 설정 파일에서 메모리 관리 설정 로드
        if simulation_config and 'memory_management' in simulation_config:
            memory_config = simulation_config['memory_management'].get('subsystem_cache', {})
            self._cache_config.update(memory_config)

        # LRU 캐시 사용 (OrderedDict 기반)
        self._cached_data = OrderedDict()  # {subsystem_name: cached_result}
        self._interpolation_data = OrderedDict()  # {subsystem_name: interpolation_info}

        # 캐시 액세스 시간 추적 (시뮬레이션 시간 기준)
        self._cache_access_times = {}  # {subsystem_name: last_access_time}
        self._last_cleanup_time = 0.0  # 시뮬레이션 시간으로 초기화

        # Thread safety를 위한 락
        self._cache_lock = threading.RLock()

        # 적응적 업데이트를 위한 추가 정보
        self._last_positions = {}  # {subsystem_name: (x, y)}
        self._context_data = {}  # {subsystem_name: context_specific_data}

        # 성능 모니터링
        self._update_counts = {}  # {subsystem_name: count}
        self._skip_counts = {}  # {subsystem_name: skip_count}

        # 설정이 제공되면 서브시스템 설정 및 콜백 등록 (지연 초기화)
        self._simulation_config = simulation_config
        self._callbacks_registered = False

    def _get_vehicle(self):
        """약한 참조로부터 vehicle 인스턴스를 안전하게 가져오기

        Returns:
            Vehicle 인스턴스 또는 None (참조가 만료된 경우)
        """
        if self._vehicle_ref is None:
            return None
        return self._vehicle_ref()

    def _ensure_callbacks_registered(self):
        """콜백이 등록되지 않았다면 지연 등록 수행"""
        vehicle = self._get_vehicle()
        if not self._callbacks_registered and vehicle and self._simulation_config:
            self._setup_subsystem_manager(self._simulation_config)
            self._callbacks_registered = True

    def _manage_cache_memory(self, current_simulation_time=None):
        """캐시 메모리 관리 - LRU 정책 및 크기 제한 적용 (시뮬레이션 시간 기준)"""
        if current_simulation_time is None:
            return  # 시뮬레이션 시간이 제공되지 않으면 스킵

        with self._cache_lock:
            # 주기적 정리 체크 (시뮬레이션 시간 기준)
            if current_simulation_time - self._last_cleanup_time > self._cache_config['cleanup_interval']:
                self._cleanup_old_cache_entries(current_simulation_time)
                self._last_cleanup_time = current_simulation_time

            # 캐시 크기 제한 적용
            max_size = self._cache_config['max_cache_size']

            # cached_data 크기 제한
            while len(self._cached_data) > max_size:
                # 가장 오래된 항목 제거 (LRU)
                oldest_key = next(iter(self._cached_data))
                del self._cached_data[oldest_key]
                if oldest_key in self._cache_access_times:
                    del self._cache_access_times[oldest_key]

            # interpolation_data 크기 제한
            while len(self._interpolation_data) > max_size:
                oldest_key = next(iter(self._interpolation_data))
                del self._interpolation_data[oldest_key]

    def _cleanup_old_cache_entries(self, current_simulation_time: float):
        """오래된 캐시 항목 정리 (시뮬레이션 시간 기준)"""
        max_age = self._cache_config.get('max_cache_age', 3600.0)

        expired_keys = []
        for key, last_access in self._cache_access_times.items():
            if current_simulation_time - last_access > max_age:
                expired_keys.append(key)

        for key in expired_keys:
            if key in self._cached_data:
                del self._cached_data[key]
            if key in self._interpolation_data:
                del self._interpolation_data[key]
            if key in self._cache_access_times:
                del self._cache_access_times[key]

    def _access_cache(self, subsystem_name: str, cache_dict: OrderedDict, current_simulation_time: float = None):
        """캐시 액세스 시 LRU 업데이트 (시뮬레이션 시간 기준)"""
        if subsystem_name in cache_dict:
            # LRU 업데이트 - 항목을 맨 뒤로 이동
            value = cache_dict.pop(subsystem_name)
            cache_dict[subsystem_name] = value
            if current_simulation_time is not None:
                self._cache_access_times[subsystem_name] = current_simulation_time
            return value
        return None

    def _setup_subsystem_manager(self, simulation_config):
        """서브 시스템 관리자 설정 및 콜백 등록"""
        vehicle = self._get_vehicle()
        if not vehicle or not simulation_config or 'hz' not in simulation_config:
            return

        hz_config = simulation_config['hz']
        adaptive_config = {
            'velocity_thresholds': [2.0, 10.0],
            'interval_factors': [2.0, 0.5],
            'min_distance': 0.05
        }

        try:
            # Frenet 업데이트 설정 (적응적 업데이트)
            if 'frenet_update' in hz_config:
                frenet_hz = hz_config['frenet_update']
                self.configure_subsystem('frenet', frenet_hz, adaptive_config)
                # 안전한 콜백 등록 (weakref 사용)
                self._register_safe_callback('frenet', '_update_frenet_callback', '_initialize_frenet_callback')

            # 라이다 센서 업데이트 설정 (적응적 업데이트)
            if 'lidar_update' in hz_config:
                lidar_hz = hz_config['lidar_update']
                self.configure_subsystem('sensor', lidar_hz, adaptive_config)
                # 안전한 콜백 등록
                self._register_safe_callback('sensor', '_update_sensor_callback', '_initialize_sensor_callback')

            # 궤적 예측 업데이트 설정 (적응적 업데이트)
            if 'trajectory_update' in hz_config:
                trajectory_hz = hz_config['trajectory_update']
                self.configure_subsystem('trajectory', trajectory_hz, adaptive_config)
                # 안전한 콜백 등록
                self._register_safe_callback('trajectory', '_update_trajectory_callback', '_initialize_trajectory_callback')

            # 충돌 검사 서브시스템 설정
            if 'collision_check' in hz_config:
                collision_hz = hz_config['collision_check']
                collision_adaptive_config = {
                    'velocity_thresholds': [5.0, 15.0],
                    'interval_factors': [3.0, 0.5],  # 저속에서 간격 늘리기, 고속에서 빠르게
                    'min_distance': 0.1,
                    'priority_mode': 'velocity'
                }
                self.configure_subsystem('collision_check', collision_hz, collision_adaptive_config)
                self._register_safe_callback('collision_check', '_update_collision_check_callback', '_initialize_collision_check_callback')

            # 목적지 도달 확인 서브시스템 설정
            if 'goal_check' in hz_config:
                goal_hz = hz_config['goal_check']
                goal_adaptive_config = {
                    'velocity_thresholds': [2.0, 8.0],
                    'interval_factors': [2.0, 1.0],
                    'min_distance': 0.05,
                    'distance_thresholds': [3.0, 15.0],  # 목적지 기반 적응적 업데이트
                    'priority_mode': 'hybrid'
                }
                self.configure_subsystem('goal_check', goal_hz, goal_adaptive_config)
                self._register_safe_callback('goal_check', '_update_goal_check_callback', '_initialize_goal_check_callback')

            # 상태 이력 서브시스템 설정
            if 'state_history' in hz_config:
                history_hz = hz_config['state_history']
                history_adaptive_config = {
                    'velocity_thresholds': [1.0, 10.0],
                    'interval_factors': [5.0, 0.8],  # 정지 시 이력 간격 늘리기
                    'min_distance': 0.2,
                    'priority_mode': 'velocity'
                }
                self.configure_subsystem('state_history', history_hz, history_adaptive_config)
                self._register_safe_callback('state_history', '_update_state_history_callback', '_initialize_state_history_callback')

        except Exception as e:
            print(f"Warning: Failed to setup subsystem manager: {e}")

    def _register_safe_callback(self, subsystem_name: str, update_method_name: str, init_method_name: str):
        """안전한 콜백 등록 - weakref를 사용하여 순환 참조 방지"""
        def safe_update_callback(*args, **kwargs):
            vehicle = self._get_vehicle()
            if vehicle and hasattr(vehicle, update_method_name):
                try:
                    method = getattr(vehicle, update_method_name)
                    return method(*args, **kwargs)
                except Exception as e:
                    print(f"Warning: Error in {subsystem_name} update callback: {e}")
                    return None
            return None

        def safe_init_callback(*args, **kwargs):
            vehicle = self._get_vehicle()
            if vehicle and hasattr(vehicle, init_method_name):
                try:
                    method = getattr(vehicle, init_method_name)
                    return method(*args, **kwargs)
                except Exception as e:
                    print(f"Warning: Error in {subsystem_name} init callback: {e}")
                    return None
            return None

        self.register_update_callback(subsystem_name, safe_update_callback)
        self.register_initialization_callback(subsystem_name, safe_init_callback)

    def configure_subsystem(self, subsystem_name: str, hz: float, adaptive_config=None):
        """서브시스템의 업데이트 빈도 설정

        Args:
            subsystem_name: 서브시스템 이름
            hz: 업데이트 주파수 [Hz]
            adaptive_config: 적응적 업데이트 설정
                - velocity_thresholds: [low_vel, high_vel] 속도 임계값
                - interval_factors: [low_factor, high_factor] 간격 조정 인수
                - min_distance: 최소 거리 변화 임계값
                - distance_thresholds: [close_dist, far_dist] 거리 임계값 (목적지용)
                - priority_mode: 'velocity', 'distance', 'hybrid' 중 선택
        """
        interval = 1.0 / hz if hz > 0 else float('inf')
        self._update_intervals[subsystem_name] = interval
        self._last_update_times[subsystem_name] = 0.0
        self._update_counts[subsystem_name] = 0
        self._skip_counts[subsystem_name] = 0

        if adaptive_config:
            self._adaptive_configs[subsystem_name] = adaptive_config

        # 초기 위치 설정
        self._last_positions[subsystem_name] = None
        self._context_data[subsystem_name] = {}

    def register_update_callback(self, subsystem_name: str, callback_function):
        """서브시스템의 업데이트 콜백 함수 등록"""
        self._update_callbacks[subsystem_name] = callback_function

    def register_initialization_callback(self, subsystem_name: str, init_callback):
        """서브시스템의 초기화 콜백 함수 등록"""
        self._initialization_callbacks[subsystem_name] = init_callback

    def initialize_subsystem(self, subsystem_name: str, current_simulation_time: float = 0.0, *args, **kwargs):
        """서브시스템 초기화 실행"""
        # 콜백이 등록되지 않았다면 지연 등록 시도
        self._ensure_callbacks_registered()

        if subsystem_name in self._initialization_callbacks:
            try:
                result = self._initialization_callbacks[subsystem_name](*args, **kwargs)
                with self._cache_lock:
                    self._cached_data[subsystem_name] = result
                    self._cache_access_times[subsystem_name] = current_simulation_time
                return result
            except Exception as e:
                print(f"Warning: Error initializing {subsystem_name}: {e}")
                return None
        return None

    def initialize_all_subsystems(self, *args, **kwargs):
        """모든 등록된 서브시스템 초기화"""
        self._ensure_callbacks_registered()
        for subsystem_name in self._initialization_callbacks:
            self.initialize_subsystem(subsystem_name, *args, **kwargs)

    def should_update(self, subsystem_name: str, current_time: float,
                     velocity: float = 0.0, position: tuple = None,
                     context: dict = None) -> bool:
        """향상된 서브시스템 업데이트 여부 판단

        Args:
            subsystem_name: 서브시스템 이름
            current_time: 현재 시간
            velocity: 현재 속도
            position: 현재 위치 (x, y)
            context: 추가 컨텍스트 정보 (목적지 거리, 장애물 거리 등)
        """
        # 설정되지 않은 서브시스템은 항상 업데이트
        if subsystem_name not in self._update_intervals:
            return True

        last_update_time = self._last_update_times[subsystem_name]
        base_interval = self._update_intervals[subsystem_name]

        # 무한 간격(비활성화)인 경우 업데이트 안함
        if base_interval == float('inf'):
            return False

        # 적응적 업데이트 설정 적용
        if subsystem_name in self._adaptive_configs:
            adaptive_config = self._adaptive_configs[subsystem_name]
            interval = self._calculate_adaptive_interval(
                base_interval, velocity, adaptive_config, context
            )

            # 시간 간격 확인
            time_elapsed = current_time - last_update_time
            if time_elapsed < interval:
                self._skip_counts[subsystem_name] += 1
                return False

            # 위치/거리 기반 확인
            if not self._check_positional_update_criteria(
                subsystem_name, position, adaptive_config, context
            ):
                self._skip_counts[subsystem_name] += 1
                return False
        else:
            # 단순 시간 기반 업데이트
            time_elapsed = current_time - last_update_time
            if time_elapsed < base_interval:
                self._skip_counts[subsystem_name] += 1
                return False

        # 업데이트 승인
        self._last_update_times[subsystem_name] = current_time
        if position:
            self._last_positions[subsystem_name] = position
        if context:
            self._context_data[subsystem_name].update(context)
        self._update_counts[subsystem_name] += 1

        return True

    def _check_positional_update_criteria(self, subsystem_name: str, position: tuple,
                                        adaptive_config: dict, context: dict = None) -> bool:
        """위치/거리 기반 업데이트 기준 확인"""
        if position is None:
            return True

        last_position = self._last_positions.get(subsystem_name)
        if last_position is None:
            return True

        # 이동 거리 계산
        dx = position[0] - last_position[0]
        dy = position[1] - last_position[1]
        distance_moved = (dx*dx + dy*dy)**0.5

        min_distance = adaptive_config.get('min_distance', 0.05)
        if distance_moved < min_distance:
            return False

        # 컨텍스트 기반 추가 검사
        if context and 'goal_distance' in context:
            goal_distance = context['goal_distance']
            distance_thresholds = adaptive_config.get('distance_thresholds', [5.0, 20.0])

            # 목적지에 가까울수록 더 자주 업데이트
            if goal_distance < distance_thresholds[0]:  # 매우 가까움
                return distance_moved > min_distance * 0.5
            elif goal_distance > distance_thresholds[1]:  # 매우 멀음
                return distance_moved > min_distance * 2.0

        return True

    def execute_update(self, subsystem_name: str, current_time: float,
                      velocity: float = 0.0, position: tuple = None,
                      context: dict = None, *args, **kwargs):
        """향상된 서브시스템 업데이트 실행"""
        # 콜백이 등록되지 않았다면 지연 등록 시도
        self._ensure_callbacks_registered()

        # 메모리 관리 수행 (시뮬레이션 시간 기준)
        self._manage_cache_memory(current_time)

        should_update = self.should_update(subsystem_name, current_time, velocity, position, context)

        if should_update and subsystem_name in self._update_callbacks:
            try:
                # 새로운 업데이트 실행
                result = self._update_callbacks[subsystem_name](*args, **kwargs)

                # 캐시에 안전하게 저장
                with self._cache_lock:
                    self._cached_data[subsystem_name] = result
                    self._cache_access_times[subsystem_name] = current_time

                return result
            except Exception as e:
                print(f"Warning: Error executing update for {subsystem_name}: {e}")
                # 오류 발생 시 캐시된 데이터 반환 시도
                return self.get_cached_data(subsystem_name, current_time)
        else:
            # 캐시된 데이터 반환 (보간 가능)
            return self._get_cached_or_interpolated_data(
                subsystem_name, current_time, position, context
            )

    def get_cached_data(self, subsystem_name: str, current_simulation_time: float = None):
        """캐시된 데이터 반환"""
        with self._cache_lock:
            return self._access_cache(subsystem_name, self._cached_data, current_simulation_time)

    def _get_cached_or_interpolated_data(self, subsystem_name: str, current_time: float,
                                       position: tuple = None, context: dict = None):
        """캐시된 데이터 또는 보간된 데이터 반환"""
        with self._cache_lock:
            cached_data = self._access_cache(subsystem_name, self._cached_data, current_time)

        if cached_data is None:
            return None

        try:
            # 서브시스템별 특별한 보간 처리
            if subsystem_name == 'frenet' and position:
                return self._interpolate_frenet_data(cached_data, position)
            elif subsystem_name == 'collision_check':
                return self._interpolate_collision_data(cached_data, position, context)
            elif subsystem_name == 'goal_check':
                return self._interpolate_goal_data(cached_data, position, context)
        except Exception as e:
            print(f"Warning: Error interpolating data for {subsystem_name}: {e}")

        return cached_data

    def _interpolate_frenet_data(self, cached_data, position: tuple):
        """Frenet 데이터 보간"""
        vehicle = self._get_vehicle()
        if not vehicle:
            return cached_data

        try:
            # 보간을 사용한 부드러운 전환
            state = vehicle.state
            if (state.frenet_d is not None and state.frenet_point is not None and
                'frenet' in self._last_positions and
                self._last_positions['frenet'] is not None):

                # 이전 위치에서 현재 위치로의 변화량 계산
                last_pos = self._last_positions['frenet']
                dx = position[0] - last_pos[0]
                dy = position[1] - last_pos[1]

                # 도로 방향으로의 투영 (간단한 보간)
                if state.frenet_point and len(state.frenet_point) >= 3:
                    road_yaw = state.frenet_point[2]

                    # 도로 방향과 수직 방향의 단위 벡터
                    perp_dir_x = -sin(road_yaw)
                    perp_dir_y = cos(road_yaw)

                    # 현재 위치에서의 d 값 보간 계산
                    ref_x, ref_y = state.frenet_point[0], state.frenet_point[1]
                    to_vehicle_x = position[0] - ref_x
                    to_vehicle_y = position[1] - ref_y

                    # 수직 거리 계산 (좌측이 양수)
                    interpolated_d = to_vehicle_x * perp_dir_x + to_vehicle_y * perp_dir_y

                    # 보간된 값으로 업데이트
                    state.frenet_d = interpolated_d

                road_width = 8.0  # 기본 도로 폭
                outside_road = abs(state.frenet_d) > (road_width / 2)
                return outside_road

        except Exception as e:
            print(f"Warning: Error in frenet data interpolation: {e}")

        return cached_data if cached_data is not None else False

    def _interpolate_collision_data(self, cached_data, position: tuple, context: dict):
        """충돌 데이터 보간 - 빠른 움직임 시 더 보수적으로"""
        if context and 'velocity' in context:
            velocity = context['velocity']
            # 고속일 때는 충돌 위험을 더 보수적으로 평가
            if velocity > 10.0 and cached_data:  # 충돌이 감지된 상태에서 고속
                return True  # 더 보수적으로 충돌 상태 유지
        return cached_data

    def _interpolate_goal_data(self, cached_data, position: tuple, context: dict):
        """목적지 도달 데이터 보간"""
        # 목적지 거리가 빠르게 변하는 경우 캐시 무효화
        if context and 'goal_distance' in context:
            goal_distance = context['goal_distance']
            last_context = self._context_data.get('goal_check', {})

            if 'last_goal_distance' in last_context:
                distance_change = abs(goal_distance - last_context['last_goal_distance'])
                if distance_change > 2.0:  # 2m 이상 변화시 재평가 필요
                    return None  # 캐시 무효화

        return cached_data

    def execute_all_updates(self, current_time: float, current_velocity: float = 0.0,
                           current_position: tuple = None, context: dict = None,
                           **subsystem_args):
        """모든 등록된 서브시스템을 한번에 업데이트"""
        results = {}
        base_context = context or {}

        for subsystem_name in self._update_intervals.keys():
            # 각 서브시스템별 인수 가져오기
            args_key = f"{subsystem_name}_args"
            args = subsystem_args.get(args_key, ())

            # 서브시스템별 컨텍스트 정보 추가
            subsystem_context = base_context.copy()
            subsystem_context['velocity'] = current_velocity

            if not isinstance(args, tuple):
                args = (args,) if args is not None else ()

            result = self.execute_update(
                subsystem_name,
                current_time,
                current_velocity,
                current_position,
                subsystem_context,
                *args
            )

            results[subsystem_name] = result

        return results

    def _calculate_adaptive_interval(self, base_interval: float, velocity: float,
                                   config: dict, context: dict = None) -> float:
        """향상된 적응적 업데이트 간격 계산"""
        priority_mode = config.get('priority_mode', 'velocity')

        if priority_mode == 'velocity':
            return self._calculate_velocity_based_interval(base_interval, velocity, config)
        elif priority_mode == 'distance' and context and 'goal_distance' in context:
            return self._calculate_distance_based_interval(base_interval, context['goal_distance'], config)
        elif priority_mode == 'hybrid':
            velocity_interval = self._calculate_velocity_based_interval(base_interval, velocity, config)
            if context and 'goal_distance' in context:
                distance_interval = self._calculate_distance_based_interval(base_interval, context['goal_distance'], config)
                return min(velocity_interval, distance_interval)  # 더 짧은 간격 선택
            return velocity_interval
        else:
            return base_interval

    def _calculate_velocity_based_interval(self, base_interval: float, velocity: float, config: dict) -> float:
        """속도 기반 간격 계산"""
        velocity_thresholds = config.get('velocity_thresholds', [2.0, 10.0])
        interval_factors = config.get('interval_factors', [2.0, 0.5])

        low_vel, high_vel = velocity_thresholds
        low_factor, high_factor = interval_factors

        if velocity < low_vel:
            return base_interval * low_factor
        elif velocity > high_vel:
            return base_interval * high_factor
        else:
            # 선형 보간
            t = (velocity - low_vel) / (high_vel - low_vel)
            factor = low_factor + t * (high_factor - low_factor)
            return base_interval * factor

    def _calculate_distance_based_interval(self, base_interval: float, distance: float, config: dict) -> float:
        """거리 기반 간격 계산"""
        distance_thresholds = config.get('distance_thresholds', [5.0, 20.0])
        interval_factors = config.get('interval_factors', [0.5, 2.0])  # 가까울수록 자주

        close_dist, far_dist = distance_thresholds
        close_factor, far_factor = interval_factors

        if distance < close_dist:
            return base_interval * close_factor
        elif distance > far_dist:
            return base_interval * far_factor
        else:
            # 선형 보간
            t = (distance - close_dist) / (far_dist - close_dist)
            factor = close_factor + t * (far_factor - close_factor)
            return base_interval * factor

    def get_performance_stats(self) -> dict:
        """성능 통계 반환"""
        stats = {}
        for subsystem_name in self._update_intervals.keys():
            total_attempts = self._update_counts[subsystem_name] + self._skip_counts[subsystem_name]
            if total_attempts > 0:
                update_rate = self._update_counts[subsystem_name] / total_attempts
            else:
                update_rate = 0.0

            stats[subsystem_name] = {
                'updates': self._update_counts[subsystem_name],
                'skips': self._skip_counts[subsystem_name],
                'update_rate': update_rate,
                'configured_hz': 1.0 / self._update_intervals[subsystem_name] if self._update_intervals[subsystem_name] > 0 else 0
            }
        return stats

    def reset(self):
        """모든 서브시스템의 상태 초기화"""
        with self._cache_lock:
            for subsystem_name in self._last_update_times:
                self._last_update_times[subsystem_name] = 0.0
                self._last_positions[subsystem_name] = None
                self._context_data[subsystem_name] = {}
                self._update_counts[subsystem_name] = 0
                self._skip_counts[subsystem_name] = 0

            # 캐시 완전 초기화
            self._cached_data.clear()
            self._interpolation_data.clear()
            self._cache_access_times.clear()
            self._last_cleanup_time = 0.0  # 시뮬레이션 시간으로 초기화

    def reset_and_reinitialize(self, *args, **kwargs):
        """모든 서브시스템 초기화 후 재초기화 실행"""
        self.reset()
        self.initialize_all_subsystems(*args, **kwargs)

    def set_cache_config(self, config: Dict[str, Any]):
        """캐시 설정 업데이트"""
        with self._cache_lock:
            self._cache_config.update(config)

    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 사용 통계 반환"""
        with self._cache_lock:
            return {
                'cached_data_size': len(self._cached_data),
                'interpolation_data_size': len(self._interpolation_data),
                'access_times_size': len(self._cache_access_times),
                'max_cache_size': self._cache_config['max_cache_size'],
                'cleanup_interval': self._cache_config['cleanup_interval'],
                'last_cleanup_time': self._last_cleanup_time
            }

    def validate_timing_configuration(self, dt: float, hz_config: Dict[str, float]) -> Dict[str, Any]:
        """
        시뮬레이션 타이밍 설정의 일관성 검증

        Args:
            dt: 고정 시뮬레이션 시간 간격 (초)
            hz_config: 서브시스템별 Hz 설정 딕셔너리

        Returns:
            검증 결과 딕셔너리
        """
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }

        simulation_hz = 1.0 / dt if dt > 0 else float('inf')

        # Hz 설정 검증
        for subsystem_name, subsystem_hz in hz_config.items():
            # 1. 시뮬레이션 주파수보다 높은 서브시스템 Hz 확인
            if subsystem_hz > simulation_hz:
                validation_result['warnings'].append(
                    f"{subsystem_name}: {subsystem_hz}Hz는 시뮬레이션 주파수 {simulation_hz:.1f}Hz보다 높습니다. "
                    f"실제로는 최대 {simulation_hz:.1f}Hz로 제한됩니다."
                )
                validation_result['recommendations'].append(
                    f"{subsystem_name}: Hz를 {simulation_hz:.1f} 이하로 설정하는 것을 권장합니다."
                )

            # 2. 비효율적인 Hz 설정 확인 (시뮬레이션 Hz의 배수가 아닌 경우)
            if simulation_hz % subsystem_hz != 0 and subsystem_hz < simulation_hz:
                closest_efficient_hz = simulation_hz / round(simulation_hz / subsystem_hz)
                validation_result['recommendations'].append(
                    f"{subsystem_name}: 현재 {subsystem_hz}Hz보다 {closest_efficient_hz:.1f}Hz가 더 효율적입니다."
                )

            # 3. 너무 낮은 Hz 설정 경고 (안전 관련 서브시스템)
            safety_critical_systems = ['collision_check', 'lidar_update']
            if subsystem_name in safety_critical_systems and subsystem_hz < 10:
                validation_result['warnings'].append(
                    f"{subsystem_name}: {subsystem_hz}Hz는 안전 관련 시스템으로는 너무 낮을 수 있습니다."
                )

        # 4. 전체적인 성능 영향 평가
        total_operations_per_sim_step = sum(
            min(hz / simulation_hz, 1.0) for hz in hz_config.values()
        )

        if total_operations_per_sim_step > 0.8:
            validation_result['warnings'].append(
                f"서브시스템들의 총 연산 부하({total_operations_per_sim_step:.2f})가 높습니다. "
                "성능 저하가 발생할 수 있습니다."
            )

        # 검증 결과 종합
        if validation_result['warnings'] or validation_result['errors']:
            validation_result['is_valid'] = len(validation_result['errors']) == 0

        return validation_result

    def print_timing_validation_report(self, dt: float, hz_config: Dict[str, float]):
        """타이밍 검증 결과를 보기 좋게 출력"""
        result = self.validate_timing_configuration(dt, hz_config)

        print("=== 시뮬레이션 타이밍 설정 검증 결과 ===")
        print(f"시뮬레이션 dt: {dt:.4f}초 ({1/dt:.1f}Hz)")
        print(f"전체 검증 상태: {'양호' if result['is_valid'] else '문제 발견'}")
        print()

        if result['errors']:
            print("오류:")
            for error in result['errors']:
                print(f"  - {error}")
            print()

        if result['warnings']:
            print("경고:")
            for warning in result['warnings']:
                print(f"  - {warning}")
            print()

        if result['recommendations']:
            print("권장사항:")
            for recommendation in result['recommendations']:
                print(f"  - {recommendation}")
            print()

        print("서브시스템별 Hz 설정:")
        for name, hz in hz_config.items():
            efficiency = "효율적" if (1/dt) % hz == 0 else "비효율적"
            print(f"  - {name}: {hz}Hz ({efficiency})")

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
    initial_distance_to_target: float = 0.0  # 초기 목표까지의 거리
    prev_distance_to_target: float = 0.0  # 이전 목표까지의 거리
    curr_distance_to_target: float = 0.0  # 남은 목표까지의 거리
    yaw_diff_to_target: float = 0.0  # 목표까지의 방향 차이

    # frenet 좌표
    frenet_d: float = 0.0
    frenet_point: tuple = None
    target_vel_long: float = 0.0

    # 라이다 센서 데이터, 거리 배열
    lidar_data: List = field(default_factory=list)

    # 차량 과거 궤적
    history_trajectory: deque = field(default_factory=lambda: deque(maxlen=500))

    # 궤적 데이터
    physics_trajectory: List[TrajectoryData] = field(default_factory=list)

    # 상태 이력 (최근 N개 상태 기록)
    state_history: deque = field(default_factory=lambda: deque(maxlen=20))

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
        self.initial_distance_to_target = 0.0
        self.prev_distance_to_target = 0.0
        self.curr_distance_to_target = 0.0
        self.yaw_diff_to_target = 0.0

        self.frenet_d = 0.0
        self.frenet_point = None
        self.target_vel_long = 0.0

        self.lidar_data.clear()

        self.history_trajectory.clear()
        self.physics_trajectory.clear()

        self.state_history.clear()

        self.terrain_type = "asphalt"

    def clone(self):
        """차량 상태 복제 - 핵심 상태 값만 복제하고 큰 데이터 구조는 제외"""
        new_state = VehicleState()

        # 위치 및 방향 관련 속성 복사
        new_state.x = self.x
        new_state.y = self.y
        new_state.yaw = self.yaw
        new_state.rear_axle_x = self.rear_axle_x
        new_state.rear_axle_y = self.rear_axle_y
        new_state.half_wheelbase = self.half_wheelbase

        # 속도 및 가속도 관련 속성 복사
        new_state.vel_long = self.vel_long
        new_state.vel_lat = self.vel_lat
        new_state.acc_long = self.acc_long
        new_state.acc_lat = self.acc_lat
        new_state.yaw_rate = self.yaw_rate

        # 제어 입력 관련 속성 복사
        new_state.throttle_engine = self.throttle_engine
        new_state.throttle_brake = self.throttle_brake
        new_state.steer = self.steer

        # 환경 속성 복사
        new_state.terrain_type = self.terrain_type
        return new_state

    def normalize_angle(self, angle):
        """[-π, π] 범위로 각도 정규화"""
        return (angle + pi) % (2 * pi) - pi

    def encoding_angle(self, angle):
        """cos/sin 인코딩을 통한 각도 표현"""
        return cos(angle), sin(angle)

    def get_progress(self):
        """거리 정규화, -1.0 ~ 1.0 범위로 목표 도달 진행률 반환"""
        if self.initial_distance_to_target == 0:
            return 0.0
        raw = (self.initial_distance_to_target - self.curr_distance_to_target) / self.initial_distance_to_target
        progress = max(-1.0, min(1.0, raw))  # -1.0 ~ 1.0 범위로 제한
        return progress

    def get_delta_progress(self):
        """목표 도달 진행률 변화량 계산"""
        return self.prev_distance_to_target - self.curr_distance_to_target

    def scale_frenet_d(self, d, road_width=6.0):
        """frenet_d 값의 스케일링 (도로 폭에 따라 -1.0 ~ 1.0 범위로)"""
        if d is None:
            return 0.0
        return d / (road_width / 2.0)  # 도로 폭에 따라 -1.0 ~ 1.0 범위로 스케일링

    def scale_long(self, vel_long, acc_long, max_vel, min_vel, max_acc, min_acc):
        """종방향 속도 정규화"""
        vel_long = min(max_vel, max(vel_long, min_vel))
        acc_long = min(max_acc, max(acc_long, min_acc))
        scale_vel = 0
        scale_acc = 0
        if vel_long > 0:
            scale_vel = vel_long / max_vel
        else:
            scale_vel = vel_long / min_vel
        if acc_long > 0:
            scale_acc = acc_long / max_acc
        else:
            scale_acc = acc_long / min_acc
        return scale_vel, scale_acc

    def scale_lat(self, vel_lat, acc_lat, max_vel_lat, max_acc_lat):
        """횡방향 속도 정규화"""
        vel_lat = min(max_vel_lat, max(vel_lat, -max_vel_lat))
        acc_lat = min(max_acc_lat, max(acc_lat, -max_acc_lat))
        scale_vel = vel_lat / max_vel_lat
        scale_acc = acc_lat / max_acc_lat
        return scale_vel, scale_acc

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

    def update_state_history(self, simulation_time: float = None):
        """현재 상태 복사본을 이력에 추가"""
        # 현재 상태의 중요 필드들을 딕셔너리로 복사
        state_snapshot = {
            'timestamp': simulation_time if simulation_time is not None else 0.0,  # 시뮬레이션 시간 타임스탬프
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

    def update_trajectory(self, physics_trajectory):
        """궤적 업데이트"""
        self.physics_trajectory = physics_trajectory

    def get_position(self):
        """차량 위치 반환"""
        return (self.x, self.y, self.yaw)

    def get_rear_axle_position(self):
        """차량 뒷바퀴 위치 반환"""
        return (self.rear_axle_x, self.rear_axle_y, self.yaw)

    def get_trajectory_data(self):
        """차량 궤적 데이터 반환, 각 지점의 상대적 위치로 변환

        각 궤적에서 시작, 중간, 마지막 데이터만 추출
        """
        physics_trajectory = self.physics_trajectory
        data_list = []

        # 물리 기반 궤적에서 중반, 마지막 데이터 추출
        if physics_trajectory:
            traj_len = len(physics_trajectory)
            x, y, yaw = self.get_rear_axle_position()
            # 중반(50%), 마지막(100%) 지점 인덱스 계산
            start_idx = 0
            mid_idx = traj_len // 2
            end_idx = traj_len - 1

            # 선택된 지점의 데이터만 추가
            if mid_idx < traj_len and mid_idx != start_idx:
                data = physics_trajectory[mid_idx].get_data(x, y, yaw)
                data_list.append(data)

            if end_idx < traj_len and end_idx != mid_idx and end_idx != start_idx:
                data = physics_trajectory[end_idx].get_data(x, y, yaw)
                data_list.append(data)

        return np.array(data_list)

    def update_lidar_data(self, distances):
        """라이다 센서 데이터 업데이트"""
        self.lidar_data = distances

    def get_lidar_data(self):
        """라이다 센서 데이터 반환"""
        return np.array(self.lidar_data, dtype=np.float32)

# ======================
# Vehicle Model
# ======================
class Vehicle:
    """차량 모델링, 제어 및 시각화를 담당하는 클래스"""

    def __init__(self, vehicle_id=None, vehicle_config=None, physics_config=None, visual_config=None, simulation_config=None):
        """차량 객체 초기화"""
        self.id = vehicle_id if vehicle_id is not None else id(self)
        self.vehicle_config = vehicle_config
        self.physics_config = physics_config
        self.visual_config = visual_config
        self.simulation_config = simulation_config
        self.state = VehicleState()
        self.state.update_rear_axle_position(self.vehicle_config['wheelbase'] / 2.0)

        # Subsystem 관리자 초기화
        self.subsystem_manager = SubsystemManager(vehicle_instance=self, simulation_config=self.simulation_config)

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

        # 서브 시스템 관리자를 통한 초기화
        self._initialize_all_subsystems()

        # 그래픽 리소스 초기화
        self._load_graphics()

    def reset(self):
        """차량 상태 초기화"""
        self.state.reset()
        self.goal_manager.clear_goals()
        self.sensor_manager.reset()
        self._load_graphics()
        self._update_collision_body()
        # 서브 시스템 관리자를 통한 재초기화
        self.subsystem_manager.reset_and_reinitialize()

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

        # 서브 시스템 관리자를 통한 통합 업데이트
        current_velocity = sqrt(self.state.vel_long**2 + self.state.vel_lat**2)
        current_position = (self.state.x, self.state.y)

        # 모든 서브시스템을 한번에 업데이트
        update_results = self.subsystem_manager.execute_all_updates(
            current_time=time_elapsed,
            current_velocity=current_velocity,
            current_position=current_position,
            context={},
            sensor_args=(dt, time_elapsed, objects),
            frenet_args=(road_manager, time_elapsed),
            trajectory_args=(),
            collision_check_args=(objects,),
            goal_check_args=(),
            state_history_args=()
        )

        # 서브시스템 결과 추출
        outside_road = update_results.get('frenet', False)
        collision = update_results.get('collision_check', False)
        reached = update_results.get('goal_check', False)

        return self.state, collision, outside_road, reached

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
        return self.state.get_position()

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

    def _initialize_all_subsystems(self):
        """서브 시스템 관리자를 통한 모든 서브시스템 초기화"""
        self.subsystem_manager.initialize_all_subsystems()

    def _initialize_frenet_callback(self):
        """Frenet 초기화 콜백"""
        return False

    def _initialize_sensor_callback(self):
        """센서 초기화 콜백"""
        self._update_sensor_data()
        return self.state.get_lidar_data()

    def _initialize_trajectory_callback(self):
        """궤적 초기화 콜백"""
        self._predict_trajectory(self.physics_config['trajectory']['time_horizon'],
                               self.physics_config['trajectory']['dt'])
        return self.state.physics_trajectory

    def _update_frenet_callback(self, road_manager, time_elapsed):
        """Frenet 업데이트 콜백 - road_manager를 통해 계산 및 보간 처리"""
        # 기본 frenet 업데이트
        outside_road = self._update_road_data_internal(road_manager, time_elapsed)

        return outside_road

    def _update_road_data_internal(self, road_manager, current_time: float = None):
        """내부용 road data 업데이트 - 서브 시스템 관리자에서 호출"""
        if current_time is None:
            current_time = 0.0  # 시뮬레이션 시간 기본값

        # road_manager를 통해 Frenet 좌표 계산
        frenet_point, frenet_d, target_vel_long, outside_road = road_manager.get_vehicle_update_data((self.state.x, self.state.y, self.state.yaw))

        # 상태 업데이트
        self.state.frenet_point = frenet_point
        self.state.frenet_d = frenet_d
        self.state.target_vel_long = target_vel_long

        return outside_road

    def _update_sensor_callback(self, dt, time_elapsed, objects):
        """센서 업데이트 콜백"""
        self.sensor_manager.update(dt, time_elapsed, objects)
        self._update_sensor_data()
        return self.state.get_lidar_data()

    def _update_trajectory_callback(self):
        """궤적 업데이트 콜백"""
        self._predict_trajectory(self.physics_config['trajectory']['time_horizon'],
                               self.physics_config['trajectory']['dt'])
        return self.state.physics_trajectory

    def _initialize_collision_check_callback(self):
        """충돌 검사 초기화 콜백"""
        return False

    def _update_collision_check_callback(self, objects):
        """충돌 검사 업데이트 콜백

        Args:
            objects: 충돌 검사할 객체들

        Returns:
            충돌 여부 (Boolean)
        """
        if len(objects) == 0:
            return False

        # 다른 객체들과의 외접원 수준의 충돌 검사
        return self._check_collision(objects)

    def _initialize_goal_check_callback(self):
        """목적지 도달 확인 초기화 콜백"""
        return False

    def _update_goal_check_callback(self, position_tolerance=None, yaw_tolerance=None):
        """목적지 도달 확인 업데이트 콜백

        Args:
            position_tolerance: 위치 도달 판정 거리 [m]
            yaw_tolerance: 방향 도달 판정 각도 [rad]

        Returns:
            목적지 도달 여부 (Boolean)
        """
        if not self.goal_manager.has_goals():
            return False

        return self._check_target_reached(position_tolerance, yaw_tolerance)

    def _initialize_state_history_callback(self):
        """상태 이력 초기화 콜백"""
        self.state.state_history.clear()
        return None

    def _update_state_history_callback(self):
        """상태 이력 업데이트 콜백

        Returns:
            업데이트된 상태 이력 크기
        """
        self.state.update_state_history()
        return len(self.state.state_history)

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
        self.state.initial_distance_to_target = self.state.curr_distance_to_target

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
        distance = self.state.curr_distance_to_target
        position_reached = distance <= position_tolerance

        # 방향 차이 계산, 목표 방향과 현재 방향의 차이 (절대값 -π ~ π 범위로)
        yaw_diff = abs(self.state.yaw_diff_to_target)
        direction_reached = yaw_diff <= yaw_tolerance

        reached = position_reached and direction_reached

        return reached

    def _update_target_info(self):
        """목표 위치까지의 거리, 각도도 계산 및 업데이트"""
        self.state.prev_distance_to_target = self.state.curr_distance_to_target
        dx = self.state.x - self.state.target_x
        dy = self.state.y - self.state.target_y
        self.state.curr_distance_to_target = (dx**2 + dy**2) ** 0.5
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
            pygame.draw.circle(screen, (0, 255, 255), world_to_screen_func((self.state.frenet_point[0], self.state.frenet_point[1])), radius)

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
            tire_rect = rotated_tire.get_rect(center=world_to_screen_func((world_x, world_y)))
            screen.blit(rotated_tire, tire_rect.topleft)

    def _draw_body(self, screen, world_to_screen_func):
        """차체 렌더링"""
        rotated_surf = pygame.transform.rotate(self.car_surf, degrees(self.state.yaw))
        rect = rotated_surf.get_rect(center=world_to_screen_func((self.state.x, self.state.y)))
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

    def _update_sensor_data(self):
        """
        센서의 값을 차량 상태에 업데이트
        """
        # 센서 매니저에서 모든 센서 가져오기
        sensors = self.sensor_manager.get_all_sensors()

        # 라이다 센서 찾기
        for sensor_id, sensor in sensors.items():
            if sensor.sensor_type == 'LidarSensor':
                # 라이다 데이터 가져오기 (기본값 포함)
                lidar_data = sensor.get_data()
                # 이제 get_data()는 항상 데이터 반환 (기본값 포함)
                self.state.update_lidar_data(lidar_data)
                break  # 첫 번째 라이다 센서만 사용

    def _draw_history_trajectory(self, screen, world_to_screen_func):
        """차량 궤적 그리기"""
        if len(self.state.history_trajectory) < 2:
            return

        # 지난 궤적을 점선으로 표시
        trajectory_points = world_to_screen_func(self.state.history_trajectory)

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
        predicted_physics_trajectory = []

        predicted_physics_trajectory = TrajectoryPredictor.predict_physics_based_trajectory(
            state=self.state,
            time_horizon=time_horizon,
            dt=dt,
            physics_config=self.physics_config,
            vehicle_config=self.vehicle_config
        )

        self.state.update_trajectory(predicted_physics_trajectory)

    def _draw_predicted_trajectory(self, screen, world_to_screen_func):
        """예측 궤적 시각화

        Args:
            screen: Pygame 화면 객체
            world_to_screen_func: 월드 좌표를 화면 좌표로 변환하는 함수
            color: 궤적 색상 (RGB)
            width: 선 두께
        """
        width = max(1, int(2 * self.visual_config['camera_zoom']))

        # 궤적 포인트들을 화면 좌표로 변환
        physics_screen_points = [world_to_screen_func((point.x, point.y)) for point in self.state.physics_trajectory]

        # 라인으로 연결하여 궤적 그리기
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
                        self.state.history_trajectory = deque(value, maxlen=500)
                    else:
                        setattr(self.state, key, value)

        # 목적지 정보 복원
        if 'goals' in serialized_data:
            self.goal_manager.load_from_serialized(serialized_data['goals'])

            # 현재 목표 위치로 target 정보 업데이트
            current_goal = self.goal_manager.get_current_goal()
            if current_goal:
                self.update_target(current_goal.x, current_goal.y, current_goal.yaw)

    def get_subsystem_manager_stats(self) -> dict:
        """
        서브 시스템 관리자 성능 통계 반환

        Returns:
            서브시스템별 업데이트 통계 데이터
        """
        return self.subsystem_manager.get_performance_stats()

    def reset_subsystem_manager_stats(self):
        """
        서브 시스템 관리자 통계 초기화
        """
        for subsystem_name in self.subsystem_manager._update_intervals.keys():
            self.subsystem_manager._update_counts[subsystem_name] = 0
            self.subsystem_manager._skip_counts[subsystem_name] = 0

    def print_subsystem_manager_stats(self, verbose=False):
        """
        서브 시스템 관리자 성능 통계 출력

        Args:
            verbose: 상세 정보 출력 여부
        """
        stats = self.get_subsystem_manager_stats()
        print("\n=== Hz Manager Performance Stats ===")
        for subsystem, data in stats.items():
            print(f"{subsystem}:")
            print(f"  Configured Hz: {data['configured_hz']:.1f}")
            print(f"  Update Rate: {data['update_rate']:.2f}")
            print(f"  Updates: {data['updates']}, Skips: {data['skips']}")
            if verbose:
                total = data['updates'] + data['skips']
                if total > 0:
                    actual_hz = data['updates'] / (total / 60.0) if total > 0 else 0  # 60 FPS 가정
                    print(f"  Estimated Actual Hz: {actual_hz:.1f}")
        print("==================================\n")

# ======================
# Vehicle Manager
# ======================
class VehicleManager:
    """차량들을 관리하는 클래스"""

    def __init__(self, road_manager, vehicle_config=None, physics_config=None, visual_config=None, simulation_config=None):
        """
        차량 관리자 초기화

        Args:
            vehicle_config: 차량 설정
            physics_config: 물리 설정
            visual_config: 시각화 설정
            simulation_config: 시뮬레이션 설정
        """
        self.vehicles = []  # 차량 목록
        self.vehicle_map = {}  # {vehicle_id: vehicle} 매핑
        self.active_vehicle_idx = 0  # 현재 활성화된 차량 인덱스
        self.next_vehicle_id = 0  # 다음 차량 ID (자동 생성용)

        self.road_manager = road_manager  # 도로 관리자

        self.collisions = {}
        self.outside_roads = {}
        self.reached_targets = {}
        self.truncated = {}
        self.terminated = {}

        # 설정 저장
        self.vehicle_config = vehicle_config
        self.physics_config = physics_config
        self.visual_config = visual_config
        self.simulation_config = simulation_config

    def create_vehicle(self, x=0.0, y=0.0, yaw=None, vehicle_id=None, vehicle_config=None, physics_config=None, visual_config=None, simulation_config=None):
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
            simulation_config: 시뮬레이션 설정 (None이면 기본값 사용)

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
        sim_config = simulation_config or self.simulation_config

        # 차량 생성
        vehicle = Vehicle(
            vehicle_id=vehicle_id,
            vehicle_config=v_config,
            physics_config=p_config,
            visual_config=vis_config,
            simulation_config=sim_config
        )

        # 위치 설정
        vehicle.set_position(x, y, yaw)

        # 차량 목록과 맵에 추가
        self.vehicles.append(vehicle)
        self.vehicle_map[vehicle_id] = vehicle

        self.collisions[vehicle_id] = False
        self.outside_roads[vehicle_id] = False
        self.reached_targets[vehicle_id] = False
        self.truncated[vehicle_id] = False
        self.terminated[vehicle_id] = False

        # 첫 번째 차량이면 활성화
        if len(self.vehicles) == 1:
            self.active_vehicle_idx = 0

        return vehicle

    def clear_all_vehicles(self):
        """
        모든 차량 제거
        """
        self.vehicles.clear()
        self.vehicle_map.clear()
        self.collisions = {}
        self.outside_roads = {}
        self.reached_targets = {}
        self.truncated = {}
        self.terminated = {}
        self.active_vehicle_idx = 0
        self.next_vehicle_id = 0

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
        del self.collisions[vehicle_id]
        del self.outside_roads[vehicle_id]
        del self.reached_targets[vehicle_id]
        del self.truncated[vehicle_id]
        del self.terminated[vehicle_id]

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
                self.collisions[vehicle_id] = False
                self.outside_roads[vehicle_id] = False
                self.reached_targets[vehicle_id] = False
                self.truncated[vehicle_id] = False
                self.terminated[vehicle_id] = False
                return True
            return False
        else:
            # 모든 차량 초기화
            for vehicle in self.vehicles:
                vehicle.reset()
                self.collisions[vehicle.id] = False
                self.outside_roads[vehicle.id] = False
                self.reached_targets[vehicle.id] = False
                self.truncated[vehicle.id] = False
                self.terminated[vehicle.id] = False
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
            actions: 차량별 액션 (액션 리스트, 2차원 배열)
            dt: 시간 간격
            time_elapsed: 총 경과 시간
            obstacles: 장애물 목록

        Returns:
            collisions: 차량별 충돌 여부 {vehicle_id: bool}
            outside_roads: 차량별 도로 외부 여부 {vehicle_id: bool}
            reached_targets: 차량별 목표 도달 여부 {vehicle_id: bool}
            terminated: 차량별 정상 종료 여부 {vehicle_id: bool}
            truncated: 차량별 비정상 중단 여부 {vehicle_id: bool}
        """
        # 빈 장애물 리스트 초기화
        if obstacles is None:
            obstacles = []

        for i, vehicle in enumerate(self.vehicles):
            if i < len(actions):
                vehicle_action = actions[i]
                _, collision, outside_road, reached = vehicle.step(vehicle_action, dt, time_elapsed, self.road_manager, obstacles, self.vehicles)
                if collision:
                    if not self.collisions[vehicle.id]:
                        self.collisions[vehicle.id] = True
                if outside_road:
                    if not self.outside_roads[vehicle.id]:
                        self.outside_roads[vehicle.id] = True
                if collision or outside_road:
                    if not self.truncated[vehicle.id]:
                        self.truncated[vehicle.id] = True
                if reached:
                    if not self.reached_targets[vehicle.id]:
                        self.reached_targets[vehicle.id] = True
                    if not self.terminated[vehicle.id]:
                        self.terminated[vehicle.id] = True

        return self.collisions, self.outside_roads, self.reached_targets, self.terminated, self.truncated

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
                'visual_config': vehicle.visual_config,
                'collision': self.collisions.get(vehicle.id, False),
                'outside_road': self.outside_roads.get(vehicle.id, False),
                'reached_target': self.reached_targets.get(vehicle.id, False),
                'truncated': self.truncated.get(vehicle.id, False),
                'terminated': self.terminated.get(vehicle.id, False),
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

        # 충돌, 도로 이탈, 목표 도달 여부 초기화
        self.collisions = {}
        self.outside_roads = {}
        self.reached_targets = {}
        self.truncated = {}
        self.terminated = {}

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
                    visual_config=visual_config,
                    simulation_config=self.simulation_config
                )

                # 상태 복원 및 목적지 정보 복원
                vehicle.load_from_serialized(vehicle_data)

                # 차량 목록 및 맵에 추가
                self.vehicles.append(vehicle)
                self.vehicle_map[vehicle_id] = vehicle

                # 충돌, 도로 이탈, 목표 도달 여부 복원
                self.collisions[vehicle_id] = vehicle_data.get('collision', False)
                self.outside_roads[vehicle_id] = vehicle_data.get('outside_road', False)
                self.reached_targets[vehicle_id] = vehicle_data.get('reached_target', False)
                self.truncated[vehicle_id] = vehicle_data.get('truncated', False)
                self.terminated[vehicle_id] = vehicle_data.get('terminated', False)

        # 활성 차량 인덱스 복원
        if 'active_vehicle_idx' in serialized_data:
            self.active_vehicle_idx = serialized_data['active_vehicle_idx']
            # 유효 범위 확인
            if len(self.vehicles) > 0:
                self.active_vehicle_idx = min(self.active_vehicle_idx, len(self.vehicles) - 1)

        # 다음 차량 ID 복원
        if 'next_vehicle_id' in serialized_data:
            self.next_vehicle_id = serialized_data['next_vehicle_id']
