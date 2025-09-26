# -*- coding: utf-8 -*-
"""
Modern Road Management System
완전히 재구현된 도로 관리 시스템 - 모든 기능을 단일 파일에 통합

주요 기능:
- B-Spline 기반 정확한 곡선 표현
- Newton-Raphson 투영을 통한 정밀한 Frenet 좌표 계산
- YAML 기반 선언적 도로 네트워크 정의
- RRT-Dubins 경로 계획 통합
- 사용자 친화적 통합 API

기존 시스템의 문제점 해결:
- 차량이 도로에 수직일 때의 부정확한 거리 계산 → 연속적 B-Spline 투영으로 해결
- 이산적 샘플링 한계 → 연속적 곡선 표현으로 해결
"""

import pygame
import numpy as np
import yaml
import uuid
import math
import random
from collections import deque
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path

# =============================================================================
# 1. 기본 유틸리티 함수들
# =============================================================================

def ortho(vect2d):
    """Computes an orthogonal vector to the one given"""
    return np.array((-vect2d[1], vect2d[0]))

def dist(pt_a, pt_b):
    """Euclidian distance between two (x, y) points"""
    return ((pt_a[0]-pt_b[0])**2 + (pt_a[1]-pt_b[1])**2)**.5

def normalize_angle(angle):
    """각도를 -pi~pi 범위로 정규화"""
    return (angle + math.pi) % (2 * math.pi) - math.pi

# =============================================================================
# 2. 기하학적 클래스들 (B-Spline, Frenet 계산)
# =============================================================================

@dataclass
class ProjectionInfo:
    """점의 곡선 투영 정보"""
    point: Tuple[float, float, float]  # 투영된 점 (x, y, yaw)
    distance: float                    # 원점으로부터의 거리
    parameter: float                   # 곡선의 매개변수 t [0,1]
    arc_length: float                 # 호장 길이


@dataclass
class FrenetState:
    """Frenet 좌표계 상태"""
    s: float                          # 호장 길이 좌표
    d: float                          # 횡방향 거리 (좌측 양수)
    s_dot: float = 0.0               # 종방향 속도
    d_dot: float = 0.0               # 횡방향 속도
    s_ddot: float = 0.0              # 종방향 가속도
    d_ddot: float = 0.0              # 횡방향 가속도

    # 참조 경로 정보
    x_ref: float = 0.0               # 참조점 x좌표
    y_ref: float = 0.0               # 참조점 y좌표
    yaw_ref: float = 0.0             # 참조점 yaw 각도
    curvature_ref: float = 0.0       # 참조점 곡률

    segment_id: Optional[str] = None # 소속 세그먼트 ID


class BSplineCurve:
    """3차 B-Spline 곡선 클래스 - Newton-Raphson 투영 지원"""

    def __init__(self, waypoints: List[Tuple[float, float, float]], degree: int = 3):
        self.waypoints = waypoints
        self.degree = min(degree, len(waypoints) - 1)
        self.n = len(waypoints) - 1
        self.control_points = self._compute_control_points()
        self.knot_vector = self._compute_knot_vector()
        self._cached_length = None

        # 성능 최적화를 위한 캐싱 시스템
        self._basis_cache = {}           # 기저함수 결과 캐싱
        self._evaluation_cache = {}      # 곡선 평가 결과 캐싱
        self._projection_cache = {}      # 투영 결과 캐싱
        self._cache_precision = 4        # 캐시 키 정밀도 (소수점 4자리)
        self._max_cache_size = 1000      # 최대 캐시 크기

    def _compute_control_points(self) -> np.ndarray:
        """웨이포인트로부터 B-Spline 제어점 계산"""
        if len(self.waypoints) <= 2:
            return np.array([[wp[0], wp[1]] for wp in self.waypoints])

        # 단순화: 웨이포인트를 제어점으로 직접 사용
        return np.array([[wp[0], wp[1]] for wp in self.waypoints])

    def _compute_knot_vector(self) -> np.ndarray:
        """균등한 knot vector 생성"""
        if self.degree >= len(self.control_points):
            self.degree = len(self.control_points) - 1

        m = len(self.control_points) + self.degree + 1
        knots = np.zeros(m)

        # Clamped knot vector
        for i in range(self.degree + 1):
            knots[i] = 0.0
            knots[m - 1 - i] = 1.0

        for i in range(1, len(self.control_points) - self.degree):
            knots[self.degree + i] = i / (len(self.control_points) - self.degree)

        return knots

    def _manage_cache_size(self, cache_dict):
        """캐시 크기 관리 - 크기 초과시 가장 오래된 항목 제거"""
        if len(cache_dict) > self._max_cache_size:
            # 가장 오래된 항목들을 절반까지 제거
            keys_to_remove = list(cache_dict.keys())[:len(cache_dict) // 2]
            for key in keys_to_remove:
                del cache_dict[key]

    def _get_cache_key(self, *args) -> tuple:
        """캐시 키 생성 (정밀도 제한)"""
        return tuple(round(arg, self._cache_precision) if isinstance(arg, float) else arg
                    for arg in args)

    def _basis_function(self, i: int, k: int, t: float) -> float:
        """B-Spline 기저함수 계산 (de Boor 알고리즘) - 캐싱 적용"""
        # 캐시 키 생성
        cache_key = self._get_cache_key(i, k, t)
        if cache_key in self._basis_cache:
            return self._basis_cache[cache_key]

        # 기존 계산 로직
        if k == 0:
            result = 1.0 if self.knot_vector[i] <= t < self.knot_vector[i + 1] else 0.0
        elif i + k >= len(self.knot_vector) or i < 0:
            result = 0.0
        else:
            left_denom = self.knot_vector[i + k] - self.knot_vector[i]
            right_denom = self.knot_vector[i + k + 1] - self.knot_vector[i + 1]

            left_term = 0.0
            if left_denom > 1e-10:
                left_term = (t - self.knot_vector[i]) / left_denom * self._basis_function(i, k - 1, t)

            right_term = 0.0
            if right_denom > 1e-10:
                right_term = (self.knot_vector[i + k + 1] - t) / right_denom * self._basis_function(i + 1, k - 1, t)

            result = left_term + right_term

        # 캐시에 저장
        self._basis_cache[cache_key] = result
        self._manage_cache_size(self._basis_cache)

        return result

    def evaluate(self, t: float) -> Tuple[float, float, float]:
        """매개변수 t에서 곡선 위의 점 계산 (x, y, yaw) - 캐싱 적용"""
        t = max(0.0, min(1.0, t))

        # 캐시 확인
        cache_key = self._get_cache_key(t)
        if cache_key in self._evaluation_cache:
            return self._evaluation_cache[cache_key]

        if len(self.control_points) == 1:
            x, y = self.control_points[0]
            result = (x, y, 0.0)
            self._evaluation_cache[cache_key] = result
            return result

        if len(self.control_points) == 2:
            # 직선 보간
            x = (1 - t) * self.control_points[0][0] + t * self.control_points[1][0]
            y = (1 - t) * self.control_points[0][1] + t * self.control_points[1][1]
            dx = self.control_points[1][0] - self.control_points[0][0]
            dy = self.control_points[1][1] - self.control_points[0][1]
            yaw = math.atan2(dy, dx)
            result = (x, y, yaw)
            self._evaluation_cache[cache_key] = result
            return result

        # B-Spline 계산
        x = y = 0.0
        for i in range(len(self.control_points)):
            basis = self._basis_function(i, self.degree, t)
            x += basis * self.control_points[i][0]
            y += basis * self.control_points[i][1]

        # 접선 방향 계산 (yaw)
        dt = 0.001
        t_next = min(1.0, t + dt)

        x_next = y_next = 0.0
        for i in range(len(self.control_points)):
            basis = self._basis_function(i, self.degree, t_next)
            x_next += basis * self.control_points[i][0]
            y_next += basis * self.control_points[i][1]

        dx = x_next - x
        dy = y_next - y
        yaw = math.atan2(dy, dx) if abs(dx) > 1e-10 or abs(dy) > 1e-10 else 0.0

        result = (x, y, yaw)

        # 캐시에 저장
        self._evaluation_cache[cache_key] = result
        self._manage_cache_size(self._evaluation_cache)

        return result

    def project_point(self, point: Tuple[float, float]) -> ProjectionInfo:
        """Newton-Raphson 방법으로 점을 곡선에 투영 (최적화됨)"""
        px, py = point

        # 캐싱 확인
        cache_key = self._get_cache_key(px, py)
        if cache_key in self._projection_cache:
            return self._projection_cache[cache_key]

        best_t = 0.0
        best_dist = float('inf')
        best_proj_point = None

        # 초기 추정: 초기점 수를 20개에서 8개로 감소
        for initial_t in np.linspace(0, 1, 8):
            t = initial_t

            # Newton-Raphson 반복: 최대 반복 횟수를 10회에서 5회로 감소
            for _ in range(5):
                curve_point = self.evaluate(t)

                # 현재 점에서 목표 점까지의 벡터
                dx = curve_point[0] - px
                dy = curve_point[1] - py

                # 1차, 2차 미분 계산 (수치 미분)
                dt = 0.001
                t_plus = min(1.0, t + dt)
                t_minus = max(0.0, t - dt)

                point_plus = self.evaluate(t_plus)
                point_minus = self.evaluate(t_minus)

                # 1차 미분 (속도 벡터)
                dx_dt = (point_plus[0] - point_minus[0]) / (2 * dt)
                dy_dt = (point_plus[1] - point_minus[1]) / (2 * dt)

                # 2차 미분 (가속도 벡터) - 간략화
                d2x_dt2 = (point_plus[0] - 2 * curve_point[0] + point_minus[0]) / (dt * dt)
                d2y_dt2 = (point_plus[1] - 2 * curve_point[1] + point_minus[1]) / (dt * dt)

                # f(t) = (C(t) - P) · C'(t) = 0 이 되는 t 찾기
                f = dx * dx_dt + dy * dy_dt
                f_prime = dx_dt * dx_dt + dy_dt * dy_dt + dx * d2x_dt2 + dy * d2y_dt2

                if abs(f) < 1e-4:  # 수렴 조건 완화: 1e-6 → 1e-4
                    break

                if abs(f_prime) < 1e-8:  # 발산 방지 조건 완화: 1e-10 → 1e-8
                    break

                # Newton step
                t_new = t - f / f_prime
                t_new = max(0.0, min(1.0, t_new))

                if abs(t_new - t) < 1e-4:  # 수렴 조건 완화: 1e-6 → 1e-4
                    break

                t = t_new

            # 최종 거리 계산
            final_point = self.evaluate(t)
            distance = math.sqrt((final_point[0] - px)**2 + (final_point[1] - py)**2)

            if distance < best_dist:
                best_dist = distance
                best_t = t
                best_proj_point = final_point

        # 호장 길이 계산
        arc_length = self.get_arc_length_at_parameter(best_t)

        result = ProjectionInfo(
            point=best_proj_point,
            distance=best_dist,
            parameter=best_t,
            arc_length=arc_length
        )

        # 캐시에 저장
        self._projection_cache[cache_key] = result
        self._manage_cache_size(self._projection_cache)

        return result

    def get_arc_length_at_parameter(self, t: float) -> float:
        """매개변수 t에서의 호장 길이 계산"""
        if t <= 0:
            return 0.0

        t = min(t, 1.0)

        # 수치적 적분으로 호장 길이 계산
        num_segments = max(10, int(t * 100))
        total_length = 0.0

        for i in range(num_segments):
            t1 = (i / num_segments) * t
            t2 = ((i + 1) / num_segments) * t

            point1 = self.evaluate(t1)
            point2 = self.evaluate(t2)

            segment_length = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
            total_length += segment_length

        return total_length

    def get_total_length(self) -> float:
        """곡선의 총 길이 계산 (캐시됨)"""
        if self._cached_length is None:
            self._cached_length = self.get_arc_length_at_parameter(1.0)
        return self._cached_length

    def sample_uniform(self, num_points: int) -> List[Tuple[float, float, float]]:
        """균등한 간격으로 곡선 샘플링"""
        if num_points < 2:
            return []

        points = []
        for i in range(num_points):
            t = i / num_points  # 0부터 (num_points-1)/num_points까지
            points.append(self.evaluate(t))

        return points

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """곡선의 bounding box 계산"""
        points = self.sample_uniform(50)  # 50개 점으로 근사

        if not points:
            return (0, 0, 0, 0)

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        return (min(xs), min(ys), max(xs), max(ys))


class FrenetCoordinateCalculator:
    """Frenet 좌표계 계산 클래스 - 캐싱 및 최적화 적용"""

    def __init__(self):
        # 성능 최적화를 위한 캐싱 시스템
        self._frenet_cache = {}              # Frenet 좌표 결과 캐싱
        self._last_vehicle_pos = None        # 마지막 차량 위치
        self._last_frenet_result = None      # 마지막 계산 결과
        self._cache_valid_distance = 0.1     # 캐시 유효 거리 (0.1m)
        self._cache_precision = 3            # 캐시 키 정밀도
        self._max_cache_size = 500           # 최대 캐시 크기

    def _get_cache_key(self, *args) -> tuple:
        """캐시 키 생성"""
        return tuple(round(arg, self._cache_precision) if isinstance(arg, float) else arg
                    for arg in args)

    def _manage_cache_size(self):
        """캐시 크기 관리"""
        if len(self._frenet_cache) > self._max_cache_size:
            # 가장 오래된 항목들을 절반까지 제거
            keys_to_remove = list(self._frenet_cache.keys())[:len(self._frenet_cache) // 2]
            for key in keys_to_remove:
                del self._frenet_cache[key]

    def _should_use_cache(self, vehicle_pos: Tuple[float, float, float]) -> bool:
        """캐시 사용 여부 판단"""
        if self._last_vehicle_pos is None or self._last_frenet_result is None:
            return False

        # 이전 위치와의 거리 계산
        dx = vehicle_pos[0] - self._last_vehicle_pos[0]
        dy = vehicle_pos[1] - self._last_vehicle_pos[1]
        distance = math.sqrt(dx*dx + dy*dy)

        return distance < self._cache_valid_distance

    def calculate_frenet_from_curve(self,
                                  vehicle_pos: Tuple[float, float, float],
                                  curve: BSplineCurve,
                                  road_width: float,
                                  segment_id: str = None) -> FrenetState:
        """곡선으로부터 Frenet 좌표 계산 - 캐싱 최적화"""
        # 캐시 사용 가능 여부 확인
        if self._should_use_cache(vehicle_pos):
            return self._last_frenet_result

        # 캐시 키 생성
        cache_key = self._get_cache_key(vehicle_pos[0], vehicle_pos[1], vehicle_pos[2],
                                       segment_id if segment_id else "none")

        if cache_key in self._frenet_cache:
            result = self._frenet_cache[cache_key]
            self._last_vehicle_pos = vehicle_pos
            self._last_frenet_result = result
            return result

        x, y, yaw = vehicle_pos

        # 곡선에 투영
        projection = curve.project_point((x, y))

        # 횡방향 거리 계산 (부호 있음)
        ref_point = projection.point

        # 차량에서 참조점으로의 벡터
        dx = x - ref_point[0]
        dy = y - ref_point[1]

        # 참조점에서의 경로 방향 (오른쪽이 음수)
        path_yaw = ref_point[2]

        # 경로 방향에 수직인 벡터 (왼쪽 방향)
        perp_x = -math.sin(path_yaw)
        perp_y = math.cos(path_yaw)

        # 횡방향 거리 (왼쪽이 양수)
        d = dx * perp_x + dy * perp_y

        result = FrenetState(
            s=projection.arc_length,
            d=d,
            x_ref=ref_point[0],
            y_ref=ref_point[1],
            yaw_ref=ref_point[2],
            segment_id=segment_id
        )

        # 캐시에 저장
        self._frenet_cache[cache_key] = result
        self._manage_cache_size()
        self._last_vehicle_pos = vehicle_pos
        self._last_frenet_result = result

        return result

    def calculate_frenet_with_velocity(self,
                                     vehicle_pos: Tuple[float, float, float],
                                     vehicle_vel: Tuple[float, float, float],
                                     curve: BSplineCurve,
                                     road_width: float,
                                     segment_id: str = None) -> FrenetState:
        """속도 정보를 포함한 Frenet 좌표 계산"""
        frenet = self.calculate_frenet_from_curve(vehicle_pos, curve, road_width, segment_id)

        vx, vy, yaw_rate = vehicle_vel

        # 속도를 Frenet 좌표계로 변환
        path_yaw = frenet.yaw_ref

        # 회전 행렬
        cos_theta = math.cos(path_yaw)
        sin_theta = math.sin(path_yaw)

        # 종방향/횡방향 속도 분해
        frenet.s_dot = vx * cos_theta + vy * sin_theta
        frenet.d_dot = -vx * sin_theta + vy * cos_theta

        return frenet

    def calculate_multiple_curves(self,
                                vehicle_pos: Tuple[float, float, float],
                                curves_data: List[Tuple[BSplineCurve, float, str]]) -> Optional[FrenetState]:
        """여러 곡선 중 가장 적합한 곡선에서 Frenet 좌표 계산 - 최적화"""
        if not curves_data:
            return None

        # 캐시 사용 가능 여부 확인
        if self._should_use_cache(vehicle_pos):
            return self._last_frenet_result

        best_frenet = None
        best_score = float('inf')

        x, y, vehicle_yaw = vehicle_pos

        for curve, road_width, segment_id in curves_data:
            frenet = self.calculate_frenet_from_curve(vehicle_pos, curve, road_width, segment_id)

            # 점수 계산: 거리 + 방향 차이
            distance_score = abs(frenet.d)
            yaw_diff = abs(normalize_angle(vehicle_yaw - frenet.yaw_ref))
            direction_score = yaw_diff * 2.0  # 방향 차이에 가중치

            total_score = distance_score + direction_score

            if total_score < best_score:
                best_score = total_score
                best_frenet = frenet

        # 결과를 캐시에 저장
        if best_frenet:
            self._last_vehicle_pos = vehicle_pos
            self._last_frenet_result = best_frenet

        return best_frenet

    def get_road_bounds(self, curve: BSplineCurve, width: float,
                       num_samples: int = 50) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """도로 경계선 계산"""
        center_points = curve.sample_uniform(num_samples)
        half_width = width / 2

        left_boundary = []
        right_boundary = []

        for x, y, yaw in center_points:
            # 수직 방향 벡터
            perp_x = -math.sin(yaw)
            perp_y = math.cos(yaw)

            # 좌우 경계점
            left_x = x + perp_x * half_width
            left_y = y + perp_y * half_width
            right_x = x - perp_x * half_width
            right_y = y - perp_y * half_width

            left_boundary.append((left_x, left_y))
            right_boundary.append((right_x, right_y))

        return left_boundary, right_boundary


# =============================================================================
# 3. 도로 네트워크 클래스들
# =============================================================================

@dataclass
class Lane:
    """개별 차선 정보"""
    id: str
    lane_index: int                    # 차선 번호 (0부터 시작)
    direction: int = 1                 # 1: 정방향, -1: 역방향
    width: float = 3.0                 # 차선 폭
    speed_limit: Optional[float] = None # 차선별 제한속도
    lane_type: str = "driving"         # driving, parking, shoulder


class RoadSegment:
    """도로의 기본 단위 - B-Spline 곡선 기반"""

    def __init__(self,
                 waypoints: List[Tuple[float, float, float]],
                 width: float = 6.0,
                 speed_limit: float = 15.0,
                 lane_count: int = 1,
                 surface_type: str = "asphalt",
                 segment_id: Optional[str] = None):
        self.id = segment_id if segment_id else str(uuid.uuid4())
        self.waypoints = waypoints
        self.width = width
        self.speed_limit = speed_limit
        self.lane_count = lane_count
        self.surface_type = surface_type

        # B-Spline 곡선 생성
        self.curve = BSplineCurve(waypoints)

        # 차선 정보 초기화
        self.lanes = self._create_lanes()

        # Frenet 계산기 (최적화됨)
        self._frenet_calculator = FrenetCoordinateCalculator()

        # 캐시된 데이터
        self._cached_boundary_lines = None
        self._cached_length = None

    def _create_lanes(self) -> List[Lane]:
        """차선 정보 생성"""
        lanes = []
        lane_width = self.width / self.lane_count

        for i in range(self.lane_count):
            lane = Lane(
                id=f"{self.id}_lane_{i}",
                lane_index=i,
                direction=1,
                width=lane_width,
                speed_limit=self.speed_limit,
                lane_type="driving"
            )
            lanes.append(lane)

        return lanes

    def get_closest_point(self, vehicle_pos: Tuple[float, float, float]) -> FrenetState:
        """차량 위치에서 가장 가까운 도로상의 점 및 Frenet 좌표 계산"""
        return self._frenet_calculator.calculate_frenet_from_curve(
            vehicle_pos, self.curve, self.width, self.id
        )

    def get_frenet_with_velocity(self, vehicle_pos: Tuple[float, float, float],
                               vehicle_vel: Tuple[float, float, float]) -> FrenetState:
        """차량 속도를 고려한 완전한 Frenet 좌표 계산"""
        return self._frenet_calculator.calculate_frenet_with_velocity(
            vehicle_pos, vehicle_vel, self.curve, self.width, self.id
        )

    def sample_centerline(self, interval: float = 0.5) -> List[Tuple[float, float, float]]:
        """중심선을 지정된 간격으로 샘플링"""
        total_length = self.get_length()
        num_points = max(2, int(total_length / interval))
        return self.curve.sample_uniform(num_points)

    def get_boundary_lines(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """도로 경계선 계산 (캐싱됨)"""
        if self._cached_boundary_lines is None:
            self._cached_boundary_lines = self._frenet_calculator.get_road_bounds(
                self.curve, self.width, num_samples=50
            )
        return self._cached_boundary_lines

    def get_length(self) -> float:
        """도로 세그먼트의 총 길이 반환"""
        if self._cached_length is None:
            self._cached_length = self.curve.get_total_length()
        return self._cached_length

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """도로 세그먼트의 bounding box 계산"""
        curve_bbox = self.curve.get_bounding_box()
        half_width = self.width / 2

        return (
            curve_bbox[0] - half_width,  # min_x
            curve_bbox[1] - half_width,  # min_y
            curve_bbox[2] + half_width,  # max_x
            curve_bbox[3] + half_width   # max_y
        )

    def is_point_on_road(self, point: Tuple[float, float]) -> bool:
        """점이 도로 위에 있는지 확인"""
        projection = self.curve.project_point(point)
        return abs(projection.distance) <= (self.width / 2)

    def draw(self, screen, world_to_screen_func, debug=False):
        """pygame 시각화"""
        if len(self.waypoints) < 2:
            return

        road_color = (100, 100, 100)  # 회색
        line_color = (255, 255, 0)   # 노란색

        # 경계선 그리기
        left_boundary, right_boundary = self.get_boundary_lines()

        if left_boundary and right_boundary:
            left_screen = [world_to_screen_func(pt) for pt in left_boundary]
            right_screen = [world_to_screen_func(pt) for pt in right_boundary]

            # 중심선
            center_points = self.curve.sample_uniform(50)
            center_screen = [world_to_screen_func((pt[0], pt[1])) for pt in center_points]

            pygame.draw.lines(screen, road_color, False, center_screen, 2)
            pygame.draw.lines(screen, line_color, False, left_screen, 2)
            pygame.draw.lines(screen, line_color, False, right_screen, 2)

        if debug:
            font = pygame.font.SysFont(None, 12)
            text = font.render(f"Seg:{self.id[:8]}", True, (255, 255, 255))
            if len(self.waypoints) > 0:
                mid_point = self.waypoints[len(self.waypoints)//2]
                mid_screen = world_to_screen_func((mid_point[0], mid_point[1]))
                screen.blit(text, (int(mid_screen[0]), int(mid_screen[1]) - 20))


@dataclass
class RoadConnection:
    """도로 연결 정보"""
    from_segment_id: str
    to_segment_id: str
    connection_point: Tuple[float, float]
    connection_type: str = "simple"  # simple, merge, split, intersection


class SpatialRoadIndex:
    """간단한 공간 인덱싱 - 최적화됨"""

    def __init__(self):
        self.segments = {}  # {segment_id: RoadSegment}
        self.bounding_boxes = {}  # {segment_id: (min_x, min_y, max_x, max_y)}

        # 검색 최적화를 위한 캐싱
        self._last_search_point = None
        self._last_search_result = []
        self._search_cache_radius = 5.0  # 5m 이내면 이전 결과 재사용

    def add_segment(self, segment: RoadSegment):
        """도로 세그먼트 추가 및 인덱싱"""
        self.segments[segment.id] = segment
        self.bounding_boxes[segment.id] = segment.get_bounding_box()

    def remove_segment(self, segment_id: str):
        """도로 세그먼트 제거"""
        if segment_id in self.segments:
            del self.segments[segment_id]
            del self.bounding_boxes[segment_id]

    def find_nearby_segments(self, point: Tuple[float, float],
                           radius: float) -> List[RoadSegment]:
        """반경 내 도로 세그먼트 검색 - 최적화됨"""
        # 캐시 사용 가능 여부 확인
        if self._last_search_point is not None:
            dx = point[0] - self._last_search_point[0]
            dy = point[1] - self._last_search_point[1]
            distance = math.sqrt(dx*dx + dy*dy)

            if distance < self._search_cache_radius:
                # 캐시된 결과에서 반경 내 세그먼트만 필터링
                nearby_segments = []
                for segment in self._last_search_result:
                    if segment.id in self.segments:  # 세그먼트가 아직 존재하는지 확인
                        projection = segment.curve.project_point(point)
                        if projection.distance <= radius:
                            nearby_segments.append(segment)

                if nearby_segments:  # 캐시 결과가 유효한 경우
                    nearby_segments.sort(key=lambda seg: seg.curve.project_point(point).distance)
                    return nearby_segments

        # 새로운 검색 수행
        x, y = point
        search_box = (x - radius, y - radius, x + radius, y + radius)

        nearby_segments = []

        for segment_id, bbox in self.bounding_boxes.items():
            # Bounding box 교집합 검사
            if self._boxes_intersect(search_box, bbox):
                segment = self.segments[segment_id]

                # 더 정확한 거리 계산
                projection = segment.curve.project_point(point)
                if projection.distance <= radius:
                    nearby_segments.append(segment)

        # 거리순 정렬
        nearby_segments.sort(key=lambda seg: seg.curve.project_point(point).distance)

        # 검색 결과 캐싱 (더 넓은 범위로)
        self._last_search_point = point
        extended_radius = radius * 2  # 확장된 반경으로 캐시
        extended_box = (x - extended_radius, y - extended_radius,
                       x + extended_radius, y + extended_radius)

        extended_segments = []
        for segment_id, bbox in self.bounding_boxes.items():
            if self._boxes_intersect(extended_box, bbox):
                extended_segments.append(self.segments[segment_id])

        self._last_search_result = extended_segments

        return nearby_segments

    def _boxes_intersect(self, box1: Tuple[float, float, float, float],
                        box2: Tuple[float, float, float, float]) -> bool:
        """두 bounding box가 교집합을 가지는지 확인"""
        return not (box1[2] < box2[0] or box2[2] < box1[0] or
                   box1[3] < box2[1] or box2[3] < box1[1])

    def get_all_segments(self) -> List[RoadSegment]:
        """모든 세그먼트 반환"""
        return list(self.segments.values())

    def clear(self):
        """모든 인덱스 클리어"""
        self.segments.clear()
        self.bounding_boxes.clear()
        # 검색 캐시도 클리어
        self._last_search_point = None
        self._last_search_result = []


class RoadNetwork:
    """도로 네트워크 관리 시스템"""

    def __init__(self):
        self.segments = {}  # {segment_id: RoadSegment}
        self.connections = []  # List[RoadConnection]
        self.spatial_index = SpatialRoadIndex()

        # Frenet 계산기 (최적화됨)
        self.frenet_calculator = FrenetCoordinateCalculator()

        # 네트워크 메타데이터
        self.name = "Road Network"
        self.version = "1.0"
        self.metadata = {}

    def add_segment(self, segment: RoadSegment) -> bool:
        """도로 세그먼트 추가"""
        if segment.id in self.segments:
            return False

        self.segments[segment.id] = segment
        self.spatial_index.add_segment(segment)
        return True

    def remove_segment(self, segment_id: str) -> bool:
        """도로 세그먼트 제거"""
        if segment_id not in self.segments:
            return False

        # 연결 정보도 함께 제거
        self.connections = [conn for conn in self.connections
                          if conn.from_segment_id != segment_id and
                             conn.to_segment_id != segment_id]

        del self.segments[segment_id]
        self.spatial_index.remove_segment(segment_id)
        return True

    def get_segment(self, segment_id: str) -> Optional[RoadSegment]:
        """세그먼트 ID로 도로 세그먼트 조회"""
        return self.segments.get(segment_id)

    def find_closest_segment(self, point: Tuple[float, float]) -> Optional[RoadSegment]:
        """점에서 가장 가까운 도로 세그먼트 찾기"""
        if not self.segments:
            return None

        # 공간 인덱스를 이용한 후보 검색
        nearby_segments = self.spatial_index.find_nearby_segments(point, radius=100.0)

        if not nearby_segments:
            # 후보가 없으면 전체 검색
            nearby_segments = list(self.segments.values())

        # 가장 가까운 세그먼트 찾기
        min_distance = float('inf')
        closest_segment = None

        for segment in nearby_segments:
            projection = segment.curve.project_point(point)
            if projection.distance < min_distance:
                min_distance = projection.distance
                closest_segment = segment

        return closest_segment

    def calculate_frenet(self, vehicle_pos: Tuple[float, float, float]) -> Optional[FrenetState]:
        """차량 위치에 대한 정확한 Frenet 좌표 계산"""
        # 1단계: 가장 가까운 세그먼트 찾기
        closest_segment = self.find_closest_segment(vehicle_pos[:2])

        if not closest_segment:
            return None

        # 2단계: 여러 후보 세그먼트에서 Frenet 계산
        nearby_segments = self.spatial_index.find_nearby_segments(
            vehicle_pos[:2], radius=20.0)

        curves_data = [(seg.curve, seg.width, seg.id) for seg in nearby_segments[:5]]

        if curves_data:
            return self.frenet_calculator.calculate_multiple_curves(vehicle_pos, curves_data)
        else:
            # 가장 가까운 세그먼트만 사용
            return closest_segment.get_closest_point(vehicle_pos)

    def is_point_on_road(self, point: Tuple[float, float]) -> bool:
        """점이 도로 네트워크 위에 있는지 확인"""
        closest_segment = self.find_closest_segment(point)
        if not closest_segment:
            return False
        return closest_segment.is_point_on_road(point)

    def connect_segments(self, connection: RoadConnection):
        """두 도로 세그먼트를 연결"""
        if (connection.from_segment_id in self.segments and
            connection.to_segment_id in self.segments):
            self.connections.append(connection)

    def get_connected_segments(self, segment_id: str) -> List[str]:
        """특정 세그먼트에 연결된 모든 세그먼트 ID 반환"""
        connected = []

        for conn in self.connections:
            if conn.from_segment_id == segment_id:
                connected.append(conn.to_segment_id)
            elif conn.to_segment_id == segment_id:
                connected.append(conn.from_segment_id)

        return connected

    def get_network_bounds(self) -> Tuple[float, float, float, float]:
        """네트워크 전체의 bounding box 계산"""
        if not self.segments:
            return (0, 0, 0, 0)

        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')

        for segment in self.segments.values():
            bbox = segment.get_bounding_box()
            min_x = min(min_x, bbox[0])
            min_y = min(min_y, bbox[1])
            max_x = max(max_x, bbox[2])
            max_y = max(max_y, bbox[3])

        return (min_x, min_y, max_x, max_y)

    def get_total_length(self) -> float:
        """네트워크의 총 도로 길이 계산"""
        return sum(segment.get_length() for segment in self.segments.values())

    def clear(self):
        """네트워크 초기화"""
        self.segments.clear()
        self.connections.clear()
        self.spatial_index.clear()

    def get_all_segments(self) -> List[RoadSegment]:
        """모든 세그먼트 반환"""
        return list(self.segments.values())

    def get_segment_count(self) -> int:
        """세그먼트 개수 반환"""
        return len(self.segments)

    def draw(self, screen, world_to_screen_func, debug=False):
        """네트워크 시각화"""
        for segment in self.segments.values():
            segment.draw(screen, world_to_screen_func, debug)


# =============================================================================
# 4. 경로 계획 시스템 (기존 RRT-Dubins 통합)
# =============================================================================

class Dubins:
    """Dubins 경로 계획 클래스 (기존 코드 유지)"""

    def __init__(self, radius, point_separation):
        assert radius > 0 and point_separation > 0
        self.radius = radius
        self.point_separation = point_separation

    def all_options(self, start, end, sort=False):
        center_0_left = self.find_center(start, 'L')
        center_0_right = self.find_center(start, 'R')
        center_2_left = self.find_center(end, 'L')
        center_2_right = self.find_center(end, 'R')
        options = [self.lsl(start, end, center_0_left, center_2_left),
                   self.rsr(start, end, center_0_right, center_2_right),
                   self.rsl(start, end, center_0_right, center_2_left),
                   self.lsr(start, end, center_0_left, center_2_right),
                   self.rlr(start, end, center_0_right, center_2_right),
                   self.lrl(start, end, center_0_left, center_2_left)]
        if sort:
            options.sort(key=lambda x: x[0])
        return options

    def dubins_path(self, start, end):
        options = self.all_options(start, end)
        dubins_path, straight = min(options, key=lambda x: x[0])[1:]
        return self.generate_points(start, end, dubins_path, straight)

    def generate_points(self, start, end, dubins_path, straight):
        if straight:
            return self.generate_points_straight(start, end, dubins_path)
        return self.generate_points_curve(start, end, dubins_path)

    def lsl(self, start, end, center_0, center_2):
        straight_dist = dist(center_0, center_2)
        alpha = np.arctan2((center_2-center_0)[1], (center_2-center_0)[0])
        beta_2 = (end[2]-alpha)%(2*np.pi)
        beta_0 = (alpha-start[2])%(2*np.pi)
        total_len = self.radius*(beta_2+beta_0)+straight_dist
        return (total_len, (beta_0, beta_2, straight_dist), True)

    def rsr(self, start, end, center_0, center_2):
        straight_dist = dist(center_0, center_2)
        alpha = np.arctan2((center_2-center_0)[1], (center_2-center_0)[0])
        beta_2 = (-end[2]+alpha)%(2*np.pi)
        beta_0 = (-alpha+start[2])%(2*np.pi)
        total_len = self.radius*(beta_2+beta_0)+straight_dist
        return (total_len, (-beta_0, -beta_2, straight_dist), True)

    def rsl(self, start, end, center_0, center_2):
        median_point = (center_2 - center_0)/2
        psia = np.arctan2(median_point[1], median_point[0])
        half_intercenter = np.linalg.norm(median_point)
        if half_intercenter < self.radius:
            return (float('inf'), (0, 0, 0), True)
        alpha = np.arccos(self.radius/half_intercenter)
        beta_0 = -(psia+alpha-start[2]-np.pi/2)%(2*np.pi)
        beta_2 = (np.pi+end[2]-np.pi/2-alpha-psia)%(2*np.pi)
        straight_dist = 2*(half_intercenter**2-self.radius**2)**.5
        total_len = self.radius*(beta_2+beta_0)+straight_dist
        return (total_len, (-beta_0, beta_2, straight_dist), True)

    def lsr(self, start, end, center_0, center_2):
        median_point = (center_2 - center_0)/2
        psia = np.arctan2(median_point[1], median_point[0])
        half_intercenter = np.linalg.norm(median_point)
        if half_intercenter < self.radius:
            return (float('inf'), (0, 0, 0), True)
        alpha = np.arccos(self.radius/half_intercenter)
        beta_0 = (psia-alpha-start[2]+np.pi/2)%(2*np.pi)
        beta_2 = (.5*np.pi-end[2]-alpha+psia)%(2*np.pi)
        straight_dist = 2*(half_intercenter**2-self.radius**2)**.5
        total_len = self.radius*(beta_2+beta_0)+straight_dist
        return (total_len, (beta_0, -beta_2, straight_dist), True)

    def lrl(self, start, end, center_0, center_2):
        dist_intercenter = dist(center_0, center_2)
        intercenter = (center_2 - center_0)/2
        psia = np.arctan2(intercenter[1], intercenter[0])
        if dist_intercenter < 2*self.radius or dist_intercenter > 4*self.radius:
            return (float('inf'), (0, 0, 0), False)
        gamma = 2*np.arcsin(dist_intercenter/(4*self.radius))
        beta_0 = (psia-start[2]+np.pi/2+(np.pi-gamma)/2)%(2*np.pi)
        beta_1 = (-psia+np.pi/2+end[2]+(np.pi-gamma)/2)%(2*np.pi)
        total_len = (2*np.pi-gamma+abs(beta_0)+abs(beta_1))*self.radius
        return (total_len,
                (beta_0, beta_1, 2*np.pi-gamma),
                False)

    def rlr(self, start, end, center_0, center_2):
        dist_intercenter = dist(center_0, center_2)
        intercenter = (center_2 - center_0)/2
        psia = np.arctan2(intercenter[1], intercenter[0])
        if dist_intercenter < 2*self.radius or dist_intercenter > 4*self.radius:
            return (float('inf'), (0, 0, 0), False)
        gamma = 2*np.arcsin(dist_intercenter/(4*self.radius))
        beta_0 = -((-psia+(start[2]+np.pi/2)+(np.pi-gamma)/2)%(2*np.pi))
        beta_1 = -((psia+np.pi/2-end[2]+(np.pi-gamma)/2)%(2*np.pi))
        total_len = (2*np.pi-gamma+abs(beta_0)+abs(beta_1))*self.radius
        return (total_len,
                (beta_0, beta_1, 2*np.pi-gamma),
                False)

    def find_center(self, point, side):
        assert side in 'LR'
        angle = point[2] + (np.pi/2 if side == 'L' else -np.pi/2)
        return np.array((point[0] + np.cos(angle)*self.radius,
                         point[1] + np.sin(angle)*self.radius))

    def generate_points_straight(self, start, end, path):
        total = self.radius*(abs(path[1])+abs(path[0]))+path[2]
        center_0 = self.find_center(start, 'L' if path[0] > 0 else 'R')
        center_2 = self.find_center(end, 'L' if path[1] > 0 else 'R')

        if abs(path[0]) > 0:
            angle = start[2]+(abs(path[0])-np.pi/2)*np.sign(path[0])
            ini = center_0+self.radius*np.array([np.cos(angle), np.sin(angle)])
        else: ini = np.array(start[:2])
        if abs(path[1]) > 0:
            angle = end[2]+(-abs(path[1])-np.pi/2)*np.sign(path[1])
            fin = center_2+self.radius*np.array([np.cos(angle), np.sin(angle)])
        else: fin = np.array(end[:2])
        dist_straight = dist(ini, fin)

        points = []
        for x in np.arange(0, total, self.point_separation):
            if x < abs(path[0])*self.radius:
                points.append(self.circle_arc(start, path[0], center_0, x))
            elif x > total - abs(path[1])*self.radius:
                points.append(self.circle_arc(end, path[1], center_2, x-total))
            else:
                coeff = (x-abs(path[0])*self.radius)/dist_straight
                points.append(coeff*fin + (1-coeff)*ini)
        points.append(end[:2])
        return np.array(points)

    def generate_points_curve(self, start, end, path):
        total = self.radius*(abs(path[1])+abs(path[0])+abs(path[2]))
        center_0 = self.find_center(start, 'L' if path[0] > 0 else 'R')
        center_2 = self.find_center(end, 'L' if path[1] > 0 else 'R')
        intercenter = dist(center_0, center_2)
        center_1 = (center_0 + center_2)/2 +\
                   np.sign(path[0])*ortho((center_2-center_0)/intercenter)\
                    *(4*self.radius**2-(intercenter/2)**2)**.5
        psi_0 = np.arctan2((center_1 - center_0)[1],
                           (center_1 - center_0)[0])-np.pi

        points = []
        for x in np.arange(0, total, self.point_separation):
            if x < abs(path[0])*self.radius:
                points.append(self.circle_arc(start, path[0], center_0, x))
            elif x > total - abs(path[1])*self.radius:
                points.append(self.circle_arc(end, path[1], center_2, x-total))
            else:
                angle = psi_0-np.sign(path[0])*(x/self.radius-abs(path[0]))
                vect = np.array([np.cos(angle), np.sin(angle)])
                points.append(center_1+self.radius*vect)
        points.append(end[:2])
        return np.array(points)

    def circle_arc(self, reference, beta, center, x):
        angle = reference[2]+((x/self.radius)-np.pi/2)*np.sign(beta)
        vect = np.array([np.cos(angle), np.sin(angle)])
        return center+self.radius*vect


class RRTNode:
    """RRT 노드 클래스"""
    def __init__(self, position, time, cost, parent_pos=None):
        self.destination_list = []
        self.position = position
        self.time = time
        self.cost = cost
        self.parent_pos = parent_pos


class RRTEdge:
    """RRT 엣지 클래스"""
    def __init__(self, node_from, node_to, path, cost):
        self.node_from = node_from
        self.node_to = node_to
        self.path = deque(path)
        self.cost = cost


@dataclass
class PlannedPath:
    """계획된 경로 정보"""
    waypoints: List[Tuple[float, float, float]]
    total_length: float
    max_curvature: float
    recommended_speed: float
    planning_mode: str


class PathPlanner:
    """경로 계획 클래스 - 기존 RRT-Dubins 통합"""

    def __init__(self, config):
        self.local_planner = Dubins(radius=config["min_radius"],
                                  point_separation=config["point_separation"])
        self.precision = (5, 5, 1)
        self.width = config["road_width"]
        self.config = config
        self.default_speed = self.config.get('default_speed', 15.0)
        self.min_speed = self.config.get('min_speed', 5.0)
        self.max_speed = self.config.get('max_speed', 25.0)

    def plan_path(self, start: Tuple[float, float, float],
                 end: Tuple[float, float, float],
                 obstacles: List = [],
                 mode: str = 'auto') -> Optional[PlannedPath]:
        """경로 계획 메인 메서드"""

        if mode == 'auto':
            # 로컬좌표계 변환
            dx_world = end[0] - start[0]
            dy_world = end[1] - start[1]
            cos_yaw = math.cos(-start[2])
            sin_yaw = math.sin(-start[2])
            dx_local = cos_yaw * dx_world - sin_yaw * dy_world
            dy_local = sin_yaw * dx_world + cos_yaw * dy_world

            # 장애물이 있으면 무조건 RRT 사용
            if len(obstacles) > 0:
                mode = 'rrt'
            elif dy_local < 0:
                mode = 'rrt'  # 목표가 뒤에 있으면 RRT
            else:
                target_angle_local = math.atan2(dy_local, dx_local)
                angle_diff = abs(normalize_angle(target_angle_local))

                if angle_diff < math.radians(2.5):  # ±2.5도
                    mode = 'straight'
                elif angle_diff < math.radians(45):  # ±10도
                    mode = 'curve'
                else:
                    mode = 'rrt'

        if mode == 'rrt':
            waypoints, total_length = self._rrt_with_dubins(start, end, obstacles)
        elif mode == 'curve':
            waypoints, total_length = self._quadratic_bezier_curve(start, end)
        else:
            waypoints, total_length = self._straight_line(start, end)

        if not waypoints:
            return None

        # 곡률과 권장 속도 계산
        max_curvature = self._calculate_max_curvature(waypoints)
        recommended_speed = self._calculate_recommended_speed(waypoints)

        return PlannedPath(
            waypoints=waypoints,
            total_length=total_length,
            max_curvature=max_curvature,
            recommended_speed=recommended_speed,
            planning_mode=mode
        )

    def _straight_line(self, start, end):
        """직선 경로"""
        waypoints = []
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.sqrt(dx**2 + dy**2)

        if distance < 0.1:
            return [start, end], distance

        yaw = math.atan2(dy, dx)

        # 5개 점으로 직선 생성
        for i in range(5):
            t = i / 4.0
            x = start[0] + t * dx
            y = start[1] + t * dy
            waypoints.append((x, y, yaw))

        return waypoints, distance

    def _quadratic_bezier_curve(self, start, end):
        """베지어 곡선 경로 - 개선된 제어점 계산"""
        start_pos = np.array([start[0], start[1]])
        end_pos = np.array([end[0], end[1]])
        start_dir = np.array([math.cos(start[2]), math.sin(start[2])])
        end_dir = np.array([math.cos(end[2]), math.sin(end[2])])

        distance = np.linalg.norm(end_pos - start_pos)

        if distance < 0.1:
            return [start, end], distance

        # 제어점 계산 개선
        # 1. 시작점과 끝점에서의 방향선이 만나는 점 찾기
        # 방향선: start + t1 * start_dir, end + t2 * end_dir

        # 두 직선이 평행한지 확인
        cross_product = np.cross(start_dir, end_dir)

        if abs(cross_product) < 1e-6:
            # 평행한 경우: 중점에서 약간 벗어난 제어점 사용
            mid_point = (start_pos + end_pos) / 2
            # 수직 방향으로 오프셋 추가
            perpendicular = np.array([-start_dir[1], start_dir[0]])
            control_point = mid_point + perpendicular * distance * 0.2
        else:
            # 교차점 계산
            # start_pos + t1 * start_dir = end_pos + t2 * end_dir
            # 연립방정식 해결
            diff = end_pos - start_pos

            # 2x2 행렬 방정식: [start_dir, -end_dir] * [t1, t2]^T = diff
            matrix = np.column_stack([start_dir, -end_dir])
            try:
                t_vals = np.linalg.solve(matrix, diff)
                t1 = t_vals[0]

                # 제어점이 너무 멀리 있으면 제한
                max_distance = distance * 1.5
                if abs(t1) > max_distance:
                    t1 = np.sign(t1) * max_distance

                control_point = start_pos + t1 * start_dir

                # 제어점이 시작점과 끝점 사이의 적절한 범위에 있는지 확인
                # 너무 극단적인 곡선 방지
                start_to_control = np.linalg.norm(control_point - start_pos)
                end_to_control = np.linalg.norm(control_point - end_pos)

                if start_to_control > distance * 2 or end_to_control > distance * 2:
                    # 극단적인 경우: 중점 근처로 제한
                    control_point = (start_pos + end_pos) / 2

            except np.linalg.LinAlgError:
                # 수치적 문제가 있는 경우 중점 사용
                control_point = (start_pos + end_pos) / 2

        # 베지어 곡선 생성
        waypoints = []
        total_length = 0.0

        for i in range(20):
            t = i / 19.0

            # 이차 베지어 곡선 공식: B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
            pos = ((1-t)**2 * start_pos +
                   2*(1-t)*t * control_point +
                   t**2 * end_pos)

            # 베지어 곡선의 1차 미분으로 접선 방향 계산
            # B'(t) = 2(1-t)(P₁-P₀) + 2t(P₂-P₁)
            tangent = (2*(1-t) * (control_point - start_pos) +
                      2*t * (end_pos - control_point))

            # yaw 계산 - 첫 번째와 마지막 점은 원래 yaw 사용
            if i == 0:
                # 첫 번째 점: 시작 yaw 사용
                yaw = start[2]
            elif i == 19:
                # 마지막 점: 끝 yaw 사용
                yaw = end[2]
            elif np.linalg.norm(tangent) > 1e-6:
                # 중간 점들: 접선 방향 사용
                yaw = math.atan2(tangent[1], tangent[0])
            else:
                # 특이점인 경우 시작/끝 yaw 사용
                yaw = start[2] if t < 0.5 else end[2]

            waypoints.append((pos[0], pos[1], yaw))

            # 길이 계산
            if i > 0:
                prev_pos = np.array([waypoints[i-1][0], waypoints[i-1][1]])
                segment_length = np.linalg.norm(pos - prev_pos)
                total_length += segment_length

        return waypoints, total_length

    def _rrt_with_dubins(self, start, end, obstacles):
        """RRT-Dubins 경로 계획"""
        nodes = {}
        edges = {}
        goal = end
        collision_dist = self.width / 2.0
        obstacles_array = np.array(obstacles) if obstacles else np.array([])

        nodes[start] = RRTNode(start, 0, 0, parent_pos=None)

        for i_iter in range(self.config.get("max_iterations", 1000)):
            current_step_size = self.config.get("base_step_size", 10.0)

            if random.random() < self.config.get("goal_sampling_rate", 0.1):
                sample = goal
            else:
                rand_x = random.uniform(min(start[0], goal[0]) - current_step_size,
                                      max(start[0], goal[0]) + current_step_size)
                rand_y = random.uniform(min(start[1], goal[1]) - current_step_size,
                                      max(start[1], goal[1]) + current_step_size)
                rand_yaw = random.uniform(-math.pi, math.pi)
                sample = (rand_x, rand_y, rand_yaw)

            options = self._select_options(nodes, sample, 10)
            for node_from_option, opt_data in options:
                if opt_data[0] == float('inf'):
                    break

                path_points = self.local_planner.generate_points(node_from_option, sample,
                                                               opt_data[1], opt_data[2])

                if len(obstacles_array) > 0 and self._is_collision_path(path_points, obstacles_array, collision_dist):
                    continue

                parent_node = nodes[node_from_option]
                new_cost = parent_node.cost + opt_data[0]

                if sample not in nodes or new_cost < nodes[sample].cost:
                    nodes[sample] = RRTNode(sample, parent_node.time + opt_data[0],
                                          new_cost, parent_pos=node_from_option)
                    edges[(node_from_option, sample)] = RRTEdge(node_from_option, sample,
                                                              path_points, opt_data[0])

                    if self._is_goal(sample, goal):
                        return self._reconstruct_path(start, sample, nodes, edges)
                break

        # RRT 실패시 직선 경로로 fallback
        print("Warning: RRT failed, falling back to straight line")
        return self._straight_line(start, goal)

    def _select_options(self, nodes, sample, nb_options):
        """Dubins 옵션 선택"""
        options = []
        for node in nodes:
            options.extend([(node, opt) for opt in self.local_planner.all_options(node, sample)])
        options.sort(key=lambda x: x[1][0])  # 거리순 정렬
        return options[:nb_options]

    def _is_collision_path(self, path_points, obstacles_array, collision_dist):
        """경로와 장애물 충돌 검사"""
        if path_points is None or len(path_points) == 0 or len(obstacles_array) == 0:
            return False

        path_coords = path_points
        obstacles_center = obstacles_array[:, :2]
        obstacles_radii_sq = (obstacles_array[:, 2] + collision_dist)**2

        diff = path_coords[:, np.newaxis, :] - obstacles_center[np.newaxis, :, :]
        dist_sq_matrix = np.sum(diff**2, axis=2)
        collision_matrix = dist_sq_matrix < obstacles_radii_sq[np.newaxis, :]

        return np.any(collision_matrix)

    def _is_goal(self, sample, goal):
        """목표 도달 확인"""
        for i, (val, target) in enumerate(zip(sample, goal)):
            if abs(target - val) > self.precision[i]:
                return False
        return True

    def _reconstruct_path(self, root_pos, goal_pos, nodes, edges):
        """경로 재구성"""
        if goal_pos not in nodes:
            return [], 0.0

        waypoints = []
        total_length = 0.0
        current_pos = goal_pos

        # 역방향으로 경로 추적
        path_segments = []
        while current_pos != root_pos:
            node = nodes[current_pos]
            if node.parent_pos is None:
                break

            edge = edges.get((node.parent_pos, current_pos))
            if edge:
                path_segments.append(list(edge.path))
                total_length += edge.cost

            current_pos = node.parent_pos

        # 정방향으로 경로 결합
        path_segments.reverse()
        for segment in path_segments:
            for point in segment:
                if len(point) == 2:
                    # (x, y)만 있는 경우 yaw 추가
                    if len(waypoints) > 0:
                        prev = waypoints[-1]
                        yaw = math.atan2(point[1] - prev[1], point[0] - prev[0])
                    else:
                        yaw = 0.0
                    waypoints.append((point[0], point[1], yaw))
                else:
                    waypoints.append((point[0], point[1], point[2] if len(point) > 2 else 0.0))

        return waypoints, total_length

    def _calculate_max_curvature(self, waypoints):
        """최대 곡률 계산"""
        if len(waypoints) < 3:
            return 0.0

        max_curvature = 0.0

        for i in range(1, len(waypoints) - 1):
            p1 = waypoints[i-1]
            p2 = waypoints[i]
            p3 = waypoints[i+1]

            # 곡률 계산
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])

            len_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
            len_v2 = math.sqrt(v2[0]**2 + v2[1]**2)

            if len_v1 > 0.01 and len_v2 > 0.01:
                cross_product = abs(v1[0] * v2[1] - v1[1] * v2[0])
                curvature = 2 * cross_product / (len_v1 * len_v2 * (len_v1 + len_v2))
                max_curvature = max(max_curvature, curvature)

        return max_curvature

    def _calculate_recommended_speed(self, waypoints):
        """권장 속도 계산"""
        max_curvature = self._calculate_max_curvature(waypoints)

        # 곡률에 따른 속도 조정
        max_safe_curvature = 0.2
        normalized_curvature = min(max_curvature / max_safe_curvature, 1.0)

        speed = self.max_speed - normalized_curvature * (self.max_speed - self.min_speed)
        return max(self.min_speed, min(self.max_speed, speed))


# =============================================================================
# 5. YAML 설정 시스템
# =============================================================================

@dataclass
class RoadTemplate:
    """도로 템플릿 정의"""
    name: str
    waypoints: List[Tuple[float, float, float]]
    width: float = 6.0
    speed_limit: float = 15.0
    lane_count: int = 1
    surface_type: str = "asphalt"


@dataclass
class NetworkConfig:
    """네트워크 전체 설정"""
    name: str
    version: str = "1.0"
    coordinate_system: str = "cartesian"
    units: str = "meters"
    metadata: Dict[str, Any] = None


class YAMLRoadLoader:
    """YAML 파일로부터 도로 네트워크 로드"""

    def __init__(self):
        self.templates = {}
        self.path_planner = None

    def set_path_planner(self, path_planner: PathPlanner):
        """경로 계획기 설정"""
        self.path_planner = path_planner

    def load_network_from_file(self, file_path: str) -> RoadNetwork:
        """YAML 파일로부터 네트워크 로드"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"YAML file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        return self.load_network_from_dict(data)

    def load_network_from_dict(self, data: Dict[str, Any]) -> RoadNetwork:
        """딕셔너리 데이터로부터 네트워크 생성"""
        # 네트워크 설정
        config = self._parse_network_config(data.get('config', {}))

        # 템플릿 로드
        self._load_templates(data.get('templates', {}))

        # 네트워크 생성
        network = RoadNetwork()
        network.name = config.name
        network.version = config.version
        network.metadata = config.metadata or {}

        # 세그먼트 생성
        for seg_data in data.get('segments', []):
            segment = self._create_segment_from_data(seg_data)
            if segment:
                network.add_segment(segment)

        # 연결 정보 생성
        for conn_data in data.get('connections', []):
            connection = self._create_connection_from_data(conn_data)
            if connection:
                network.connect_segments(connection)

        # 자동 생성 도로
        for auto_data in data.get('auto_roads', []):
            segments = self._create_auto_road_segments(auto_data)
            for segment in segments:
                network.add_segment(segment)

        return network

    def _parse_network_config(self, config_data: Dict[str, Any]) -> NetworkConfig:
        """네트워크 설정 파싱"""
        return NetworkConfig(
            name=config_data.get('name', 'Road Network'),
            version=config_data.get('version', '1.0'),
            coordinate_system=config_data.get('coordinate_system', 'cartesian'),
            units=config_data.get('units', 'meters'),
            metadata=config_data.get('metadata', {})
        )

    def _load_templates(self, templates_data: Dict[str, Any]):
        """템플릿 로드"""
        self.templates.clear()

        for name, template_data in templates_data.items():
            waypoints = []
            for wp_data in template_data.get('waypoints', []):
                if len(wp_data) >= 2:
                    x, y = wp_data[0], wp_data[1]
                    yaw = wp_data[2] if len(wp_data) > 2 else 0.0
                    waypoints.append((x, y, yaw))

            if len(waypoints) >= 2:
                template = RoadTemplate(
                    name=name,
                    waypoints=waypoints,
                    width=template_data.get('width', 6.0),
                    speed_limit=template_data.get('speed_limit', 15.0),
                    lane_count=template_data.get('lane_count', 1),
                    surface_type=template_data.get('surface_type', 'asphalt')
                )
                self.templates[name] = template

    def _create_segment_from_data(self, seg_data: Dict[str, Any]) -> Optional[RoadSegment]:
        """세그먼트 데이터로부터 세그먼트 생성"""
        segment_id = seg_data.get('id')
        if not segment_id:
            return None

        # 템플릿 사용
        template_name = seg_data.get('template')
        if template_name and template_name in self.templates:
            template = self.templates[template_name]
            waypoints = template.waypoints.copy()
            width = seg_data.get('width', template.width)
            speed_limit = seg_data.get('speed_limit', template.speed_limit)
            lane_count = seg_data.get('lane_count', template.lane_count)
            surface_type = seg_data.get('surface_type', template.surface_type)
        else:
            # 직접 정의
            waypoints = []
            for wp_data in seg_data.get('waypoints', []):
                if len(wp_data) >= 2:
                    x, y = wp_data[0], wp_data[1]
                    yaw = wp_data[2] if len(wp_data) > 2 else 0.0
                    waypoints.append((x, y, yaw))

            width = seg_data.get('width', 6.0)
            speed_limit = seg_data.get('speed_limit', 15.0)
            lane_count = seg_data.get('lane_count', 1)
            surface_type = seg_data.get('surface_type', 'asphalt')

        if len(waypoints) < 2:
            return None

        # 변형 적용
        transform = seg_data.get('transform', {})
        if transform:
            waypoints = self._apply_transforms(waypoints, transform)

        return RoadSegment(
            waypoints=waypoints,
            width=width,
            speed_limit=speed_limit,
            lane_count=lane_count,
            surface_type=surface_type,
            segment_id=segment_id
        )

    def _create_connection_from_data(self, conn_data: Dict[str, Any]) -> Optional[RoadConnection]:
        """연결 데이터 생성"""
        from_id = conn_data.get('from')
        to_id = conn_data.get('to')

        if not from_id or not to_id:
            return None

        connection_point = tuple(conn_data.get('point', [0.0, 0.0]))
        connection_type = conn_data.get('type', 'simple')

        return RoadConnection(
            from_segment_id=from_id,
            to_segment_id=to_id,
            connection_point=connection_point,
            connection_type=connection_type
        )

    def _create_auto_road_segments(self, auto_data: Dict[str, Any]) -> List[RoadSegment]:
        """자동 생성 도로 세그먼트"""
        if not self.path_planner:
            return []

        auto_type = auto_data.get('type', 'point_to_point')

        if auto_type == 'point_to_point':
            return self._create_point_to_point_road(auto_data)

        return []

    def _create_point_to_point_road(self, auto_data: Dict[str, Any]) -> List[RoadSegment]:
        """두 점 사이 자동 도로 생성"""
        start_point = auto_data.get('start')
        end_point = auto_data.get('end')

        if not start_point or not end_point or len(start_point) < 2 or len(end_point) < 2:
            return []

        start = (start_point[0], start_point[1], start_point[2] if len(start_point) > 2 else 0.0)
        end = (end_point[0], end_point[1], end_point[2] if len(end_point) > 2 else 0.0)

        planning_mode = auto_data.get('planning_mode', 'auto')
        width = auto_data.get('width', 6.0)
        speed_limit = auto_data.get('speed_limit', 15.0)
        lane_count = auto_data.get('lane_count', 1)
        surface_type = auto_data.get('surface_type', 'asphalt')

        planned_path = self.path_planner.plan_path(start, end, obstacles=[], mode=planning_mode)

        if not planned_path or not planned_path.waypoints:
            return []

        segment_id = auto_data.get('id', f'auto_road_{len(planned_path.waypoints)}')

        segment = RoadSegment(
            waypoints=planned_path.waypoints,
            width=width,
            speed_limit=speed_limit,
            lane_count=lane_count,
            surface_type=surface_type,
            segment_id=segment_id
        )

        return [segment]

    def _apply_transforms(self, waypoints: List[Tuple[float, float, float]],
                         transform: Dict[str, Any]) -> List[Tuple[float, float, float]]:
        """변형 적용"""
        if not transform:
            return waypoints

        # 이동
        translation = transform.get('translation', [0.0, 0.0])
        tx, ty = translation[0], translation[1]

        # 회전
        rotation = transform.get('rotation', 0.0)
        cos_r, sin_r = math.cos(rotation), math.sin(rotation)

        # 스케일링
        scale = transform.get('scale', [1.0, 1.0])
        sx, sy = scale[0], scale[1]

        # 회전 중심점
        center = transform.get('center', [0.0, 0.0])
        cx, cy = center[0], center[1]

        transformed = []

        for x, y, yaw in waypoints:
            # 중심점 기준 이동 후 스케일링
            x_centered = (x - cx) * sx
            y_centered = (y - cy) * sy

            # 회전
            x_rotated = x_centered * cos_r - y_centered * sin_r
            y_rotated = x_centered * sin_r + y_centered * cos_r

            # 최종 위치
            x_final = x_rotated + cx + tx
            y_final = y_rotated + cy + ty
            yaw_final = yaw + rotation

            transformed.append((x_final, y_final, yaw_final))

        return transformed

    def save_network_to_file(self, network: RoadNetwork, file_path: str):
        """네트워크를 YAML 파일로 저장"""
        data = {
            'config': {
                'name': network.name,
                'version': network.version,
                'coordinate_system': 'cartesian',
                'units': 'meters',
                'metadata': network.metadata
            },
            'segments': [
                {
                    'id': segment.id,
                    'waypoints': segment.waypoints,
                    'width': segment.width,
                    'speed_limit': segment.speed_limit,
                    'lane_count': segment.lane_count,
                    'surface_type': segment.surface_type
                }
                for segment in network.get_all_segments()
            ],
            'connections': [
                {
                    'from': conn.from_segment_id,
                    'to': conn.to_segment_id,
                    'point': list(conn.connection_point),
                    'type': conn.connection_type
                }
                for conn in network.connections
            ]
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, indent=2)


# =============================================================================
# 6. 도로 패턴 생성기
# =============================================================================

class RoadPatternGenerator:
    """일반적인 도로 패턴 생성기"""

    @staticmethod
    def create_straight_road(start: Tuple[float, float, float],
                           length: float,
                           width: float = 6.0,
                           speed_limit: float = 15.0,
                           segment_id: Optional[str] = None) -> RoadSegment:
        """직선 도로 생성"""
        x, y, yaw = start

        end_x = x + length * math.cos(yaw)
        end_y = y + length * math.sin(yaw)

        waypoints = [(x, y, yaw), (end_x, end_y, yaw)]

        return RoadSegment(
            waypoints=waypoints,
            width=width,
            speed_limit=speed_limit,
            segment_id=segment_id or "straight_road"
        )

    @staticmethod
    def create_circular_arc(center: Tuple[float, float],
                           radius: float,
                           start_angle: float,
                           arc_angle: float,
                           width: float = 6.0,
                           speed_limit: float = 12.0,
                           num_points: int = 20,
                           segment_id: Optional[str] = None) -> RoadSegment:
        """원호 도로 생성"""
        cx, cy = center
        waypoints = []

        for i in range(num_points):
            t = i / (num_points - 1)
            angle = start_angle + arc_angle * t

            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)

            tangent_angle = angle + math.pi / 2 if arc_angle > 0 else angle - math.pi / 2

            waypoints.append((x, y, tangent_angle))

        return RoadSegment(
            waypoints=waypoints,
            width=width,
            speed_limit=speed_limit,
            segment_id=segment_id or "circular_arc"
        )


class NetworkPatternGenerator:
    """네트워크 패턴 생성기"""

    @staticmethod
    def create_grid_network(origin: Tuple[float, float] = (0.0, 0.0),
                           grid_size: Tuple[int, int] = (3, 3),
                           cell_size: float = 100.0,
                           road_width: float = 6.0,
                           speed_limit: float = 15.0) -> RoadNetwork:
        """격자형 도로 네트워크 생성"""
        network = RoadNetwork()
        network.name = "Grid Network"

        ox, oy = origin
        rows, cols = grid_size

        # 수평 도로들
        for row in range(rows):
            y = oy + row * cell_size
            waypoints = []
            for col in range(cols):
                x = ox + col * cell_size
                waypoints.append((x, y, 0.0))

            if len(waypoints) >= 2:
                segment = RoadSegment(
                    waypoints=waypoints,
                    width=road_width,
                    speed_limit=speed_limit,
                    segment_id=f"h_road_{row}"
                )
                network.add_segment(segment)

        # 수직 도로들
        for col in range(cols):
            x = ox + col * cell_size
            waypoints = []
            for row in range(rows):
                y = oy + row * cell_size
                waypoints.append((x, y, math.pi / 2))

            if len(waypoints) >= 2:
                segment = RoadSegment(
                    waypoints=waypoints,
                    width=road_width,
                    speed_limit=speed_limit,
                    segment_id=f"v_road_{col}"
                )
                network.add_segment(segment)

        return network

    @staticmethod
    def create_roundabout(center: Tuple[float, float],
                         radius: float = 20.0,
                         num_entries: int = 4,
                         entry_length: float = 30.0,
                         road_width: float = 6.0,
                         speed_limit: float = 8.0) -> RoadNetwork:
        """로터리 네트워크 생성"""
        network = RoadNetwork()
        network.name = "Roundabout Network"

        cx, cy = center

        # 중앙 원형 도로
        roundabout_segment = RoadPatternGenerator.create_circular_arc(
            center=(cx, cy),
            radius=radius,
            start_angle=0.0,
            arc_angle=2 * math.pi,
            width=road_width,
            speed_limit=speed_limit,
            num_points=40,
            segment_id="roundabout_circle"
        )
        network.add_segment(roundabout_segment)

        # 진입로들
        for i in range(num_entries):
            angle = 2 * math.pi * i / num_entries

            entry_x = cx + radius * math.cos(angle)
            entry_y = cy + radius * math.sin(angle)

            start_x = entry_x + entry_length * math.cos(angle)
            start_y = entry_y + entry_length * math.sin(angle)

            entry_segment = RoadPatternGenerator.create_straight_road(
                start=(start_x, start_y, angle + math.pi),
                length=entry_length,
                width=road_width * 0.8,
                speed_limit=speed_limit,
                segment_id=f"entry_{i}"
            )
            network.add_segment(entry_segment)

            # 연결
            connection = RoadConnection(
                from_segment_id=f"entry_{i}",
                to_segment_id="roundabout_circle",
                connection_point=(entry_x, entry_y),
                connection_type="merge"
            )
            network.connect_segments(connection)

        return network


# =============================================================================
# 7. 통합 API (RoadSystemAPI)
# =============================================================================

class RoadSystemAPI:
    """도로 시스템의 통합 API - 모든 기능을 하나의 인터페이스로 제공"""

    def __init__(self, config: Optional[Dict] = None):
        self.network = RoadNetwork()

        # 기본 설정
        self.config = config or {
            'road_width': 6.0,
            'min_radius': 5.0,
            'point_separation': 0.5,
            'default_speed': 15.0,
            'min_speed': 5.0,
            'max_speed': 25.0,
            'max_iterations': 1000,
            'base_step_size': 10.0,
            'goal_sampling_rate': 0.1
        }

        self.path_planner = PathPlanner(self.config)
        self.yaml_loader = YAMLRoadLoader()
        self.yaml_loader.set_path_planner(self.path_planner)

        self._builder = None

    # === 네트워크 생성 및 로딩 ===

    def load_from_yaml(self, file_path: str) -> 'RoadSystemAPI':
        """YAML 파일로부터 도로 네트워크 로드"""
        self.network = self.yaml_loader.load_network_from_file(file_path)
        return self

    def save_to_yaml(self, file_path: str):
        """현재 네트워크를 YAML 파일로 저장"""
        self.yaml_loader.save_network_to_file(self.network, file_path)

    def create_empty_network(self, name: str = "Road Network") -> 'RoadSystemAPI':
        """빈 네트워크 생성"""
        self.network.clear()
        self.network.name = name
        return self

    # === 간편한 도로 생성 메서드 ===

    def add_straight_road(self,
                         start: Tuple[float, float, float],
                         length: float,
                         width: float = 6.0,
                         speed_limit: float = 15.0,
                         road_id: Optional[str] = None) -> 'RoadSystemAPI':
        """직선 도로 추가"""
        segment = RoadPatternGenerator.create_straight_road(
            start, length, width, speed_limit, road_id
        )
        self.network.add_segment(segment)
        return self

    def add_point_to_point_road(self,
                               start: Tuple[float, float, float],
                               end: Tuple[float, float, float],
                               width: float = 6.0,
                               speed_limit: float = 15.0,
                               road_id: Optional[str] = None,
                               planning_mode: str = 'auto',
                               obstacles: List = []) -> 'bool':
        """두 점 사이 도로 추가 (경로 계획 사용)"""
        planned_path = self.path_planner.plan_path(start, end, obstacles, planning_mode)

        if planned_path and planned_path.waypoints:
            if not road_id:
                road_id = f"road_{self.network.get_segment_count()}"

            segment = RoadSegment(
                waypoints=planned_path.waypoints,
                width=width,
                speed_limit=speed_limit,
                segment_id=road_id
            )
            self.network.add_segment(segment)
            return True
        return False

    def add_waypoint_road(self,
                         waypoints: List[Tuple[float, float, float]],
                         width: float = 6.0,
                         speed_limit: float = 15.0,
                         lane_count: int = 1,
                         road_id: Optional[str] = None) -> 'RoadSystemAPI':
        """웨이포인트로 도로 추가"""
        if len(waypoints) < 2:
            return self

        if not road_id:
            road_id = f"road_{self.network.get_segment_count()}"

        segment = RoadSegment(
            waypoints=waypoints,
            width=width,
            speed_limit=speed_limit,
            lane_count=lane_count,
            segment_id=road_id
        )
        self.network.add_segment(segment)
        return self

    def add_circular_road(self,
                         center: Tuple[float, float],
                         radius: float,
                         start_angle: float = 0.0,
                         arc_angle: float = 6.28,  # 2*pi
                         width: float = 6.0,
                         speed_limit: float = 12.0,
                         road_id: Optional[str] = None) -> 'RoadSystemAPI':
        """원호 도로 추가"""
        segment = RoadPatternGenerator.create_circular_arc(
            center, radius, start_angle, arc_angle, width, speed_limit,
            segment_id=road_id
        )
        self.network.add_segment(segment)
        return self

    # === 네트워크 패턴 생성 ===

    def create_grid_network(self,
                           origin: Tuple[float, float] = (0.0, 0.0),
                           grid_size: Tuple[int, int] = (3, 3),
                           cell_size: float = 100.0,
                           road_width: float = 6.0) -> 'RoadSystemAPI':
        """격자형 네트워크 생성"""
        self.network = NetworkPatternGenerator.create_grid_network(
            origin, grid_size, cell_size, road_width
        )
        return self

    def create_roundabout_network(self,
                                 center: Tuple[float, float],
                                 radius: float = 20.0,
                                 num_entries: int = 4) -> 'RoadSystemAPI':
        """로터리 네트워크 생성"""
        self.network = NetworkPatternGenerator.create_roundabout(
            center, radius, num_entries
        )
        return self

    # === 차량 위치 계산 (핵심 API) ===

    def get_vehicle_road_info(self, vehicle_pos: Tuple[float, float, float]) -> Optional[Dict[str, Any]]:
        """
        차량의 도로 정보 계산 (기존 get_vehicle_update_data 대체)

        기존 문제점 해결:
        - 이산적 샘플링 → B-Spline 연속 투영
        - 차량 방향 무관하게 정확한 거리 계산

        Returns:
            dict with keys: 'frenet_state', 'segment_id', 'distance_to_center',
                           'is_on_road', 's', 'd', 's_dot', 'd_dot', 'road_center_point', 'road_yaw'
        """
        frenet_state = self.network.calculate_frenet(vehicle_pos)

        if not frenet_state:
            return None

        return {
            'frenet_state': frenet_state,
            'segment_id': frenet_state.segment_id,
            'distance_to_center': abs(frenet_state.d),
            'is_on_road': self.network.is_point_on_road(vehicle_pos[:2]),
            's': frenet_state.s,
            'd': frenet_state.d,
            's_dot': frenet_state.s_dot,
            'd_dot': frenet_state.d_dot,
            'road_center_point': (frenet_state.x_ref, frenet_state.y_ref),
            'road_yaw': frenet_state.yaw_ref
        }

    def get_vehicle_update_data(self, vehicle_position: Tuple[float, float, float]):
        """
        기존 API 호환성을 위한 메서드

        Returns:
            (road_center_point, distance_to_center, road_yaw, is_outside_road)
        """
        info = self.get_vehicle_road_info(vehicle_position)

        if not info:
            return None, None, None, True

        return (
            info['road_center_point'],
            info['distance_to_center'],
            info['road_yaw'],
            not info['is_on_road']
        )

    def is_vehicle_on_road(self, vehicle_pos: Tuple[float, float]) -> bool:
        """차량이 도로 위에 있는지 확인"""
        return self.network.is_point_on_road(vehicle_pos)

    def get_distance_to_road(self, point: Tuple[float, float]) -> float:
        """점에서 가장 가까운 도로까지의 거리"""
        closest_segment = self.network.find_closest_segment(point)
        if not closest_segment:
            return float('inf')

        projection = closest_segment.curve.project_point(point)
        return projection.distance

    # === 경로 계획 ===

    def plan_path(self,
                  start: Tuple[float, float, float],
                  end: Tuple[float, float, float],
                  obstacles: List = [],
                  mode: str = 'auto') -> Optional[PlannedPath]:
        """두 점 사이의 경로 계획"""
        return self.path_planner.plan_path(start, end, obstacles, mode)

    # === 정보 조회 ===

    def get_network_info(self) -> Dict[str, Any]:
        """네트워크 정보 반환"""
        return {
            'name': self.network.name,
            'version': self.network.version,
            'segment_count': self.network.get_segment_count(),
            'total_length': self.network.get_total_length(),
            'bounds': self.network.get_network_bounds(),
            'metadata': self.network.metadata
        }

    def get_all_segment_ids(self) -> List[str]:
        """모든 세그먼트 ID 반환"""
        return list(self.network.segments.keys())

    def get_segment_info(self, segment_id: str) -> Optional[Dict[str, Any]]:
        """특정 세그먼트 정보 반환"""
        segment = self.network.get_segment(segment_id)
        if not segment:
            return None

        return {
            'id': segment.id,
            'length': segment.get_length(),
            'width': segment.width,
            'speed_limit': segment.speed_limit,
            'lane_count': segment.lane_count,
            'surface_type': segment.surface_type,
            'waypoint_count': len(segment.waypoints),
            'bounds': segment.get_bounding_box()
        }

    # === 네트워크 접근 ===

    def get_network(self) -> RoadNetwork:
        """내부 네트워크 객체 반환"""
        return self.network

    def get_segment(self, segment_id: str) -> Optional[RoadSegment]:
        """세그먼트 객체 반환"""
        return self.network.get_segment(segment_id)

    def clear_network(self):
        """네트워크 초기화"""
        self.network.clear()

    def draw(self, screen, world_to_screen_func, debug=False):
        """네트워크 시각화 (기존 API 호환)"""
        self.network.draw(screen, world_to_screen_func, debug)

    def __str__(self) -> str:
        return f"RoadSystemAPI({self.network})"


# =============================================================================
# 8. 편의 함수들
# =============================================================================

def load_road_network(yaml_file: str, config: Optional[Dict] = None) -> RoadSystemAPI:
    """YAML 파일로부터 도로 네트워크 로드"""
    return RoadSystemAPI(config).load_from_yaml(yaml_file)

def create_simple_road(start: Tuple[float, float, float],
                      end: Tuple[float, float, float],
                      width: float = 6.0,
                      config: Optional[Dict] = None) -> RoadSystemAPI:
    """간단한 점대점 도로 생성"""
    return RoadSystemAPI(config).add_point_to_point_road(start, end, width)

def create_test_track(size: float = 100.0, config: Optional[Dict] = None) -> RoadSystemAPI:
    """테스트용 원형 트랙 생성"""
    return RoadSystemAPI(config).add_circular_road(
        center=(0.0, 0.0),
        radius=size,
        width=8.0,
        speed_limit=20.0,
        road_id="test_track"
    )

def create_grid_city(size: int = 3, block_size: float = 50.0,
                    config: Optional[Dict] = None) -> RoadSystemAPI:
    """격자형 도시 도로 생성"""
    return RoadSystemAPI(config).create_grid_network(
        grid_size=(size, size),
        cell_size=block_size
    )
