# -*- coding: utf-8 -*-
import pygame
import numpy as np
import math
import random
import json
import yaml
import time
from collections import deque, OrderedDict
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path
from scipy.spatial import KDTree


# =============================================================================
# 1. 기본 유틸리티 함수
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
# 2. 데이터 구조 정의
# =============================================================================

@dataclass
class FrenetState:
    """Frenet 좌표계 상태"""
    s: float = 0.0                # 호장 길이 좌표
    d: float = 0.0                # 횡방향 거리 (좌측 양수)
    s_dot: float = 0.0            # 종방향 속도
    d_dot: float = 0.0            # 횡방향 속도

    # 참조 경로 정보
    x_ref: float = 0.0            # 참조점 x좌표
    y_ref: float = 0.0            # 참조점 y좌표
    yaw_ref: float = 0.0          # 참조점 yaw 각도
    curvature_ref: float = 0.0    # 참조점 곡률
    segment_id: Optional[str] = None


@dataclass
class PlannedPath:
    """계획된 경로 정보"""
    waypoints: List[Tuple[float, float, float]]
    total_length: float
    planning_mode: str


@dataclass
class RoadConnection:
    """도로 연결 정보"""
    from_segment_id: str
    to_segment_id: str
    connection_point: Tuple[float, float]
    connection_type: str = "simple"


# =============================================================================
# 3. 선형 보간 기반 도로 세그먼트
# =============================================================================

class LinearRoadSegment:
    """선형 보간 기반 경량화된 도로 세그먼트"""

    def __init__(self, waypoints: List[Tuple[float, float, float]],
                 width: float = 6.0, speed_limit: float = 15.0,
                 segment_id: Optional[str] = None, cache_config: Optional[Dict] = None):
        self.id = segment_id or f"seg_{id(self)}"
        self.waypoints = waypoints
        self.width = width
        self.speed_limit = speed_limit

        # 캐시된 데이터
        self._cached_total_length = None
        self._cached_segment_lengths = None
        self._cached_cumulative_lengths = None
        self._cached_boundary_lines = None
        self._cached_curvatures = None  # 각 waypoint에서의 곡률

        # 메모리 관리 설정 (기본값 설정 후 config에서 로드)
        self._cache_config = {
            'max_cache_size': 1000,
            'cleanup_interval': 300.0,
            'max_cache_age': 1800.0
        }

        # 설정 파일에서 캐시 설정 로드
        if cache_config:
            self._cache_config.update(cache_config)

        # 메모리 관리가 개선된 캐시 시스템
        self._projection_cache = OrderedDict()
        self._cache_access_times = {}
        self._last_cleanup_time = time.time()

        self._precompute_geometry()

    def _precompute_geometry(self):
        """기하학적 데이터 사전 계산"""
        if len(self.waypoints) < 2:
            self._cached_segment_lengths = []
            self._cached_cumulative_lengths = [0.0]
            self._cached_total_length = 0.0
            self._cached_curvatures = [0.0]
            return

        # 세그먼트 길이 계산
        lengths = []
        cumulative = [0.0]

        for i in range(len(self.waypoints) - 1):
            p1 = self.waypoints[i]
            p2 = self.waypoints[i + 1]
            length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            lengths.append(length)
            cumulative.append(cumulative[-1] + length)

        self._cached_segment_lengths = lengths
        self._cached_cumulative_lengths = cumulative
        self._cached_total_length = cumulative[-1]

        # 곡률 계산 (Menger curvature)
        self._cached_curvatures = self._compute_curvatures()

    def get_length(self) -> float:
        """총 길이 반환"""
        return self._cached_total_length or 0.0

    def _compute_curvatures(self) -> List[float]:
        """각 waypoint에서의 곡률 계산 (Menger curvature)"""
        n = len(self.waypoints)

        if n < 3:
            return [0.0] * n

        curvatures = []

        for i in range(n):
            if i == 0:
                # 첫 번째 점: 다음 세그먼트의 곡률 사용
                curvatures.append(self._calculate_curvature_at_index(1))
            elif i == n - 1:
                # 마지막 점: 이전 세그먼트의 곡률 사용
                curvatures.append(self._calculate_curvature_at_index(n - 2))
            else:
                # 중간 점: i-1, i, i+1 세 점의 곡률 계산
                curvatures.append(self._calculate_curvature_at_index(i))

        return curvatures

    def _calculate_curvature_at_index(self, i: int) -> float:
        """인덱스 i를 중심으로 한 세 점의 곡률 계산"""
        if i <= 0 or i >= len(self.waypoints) - 1:
            return 0.0

        p1 = self.waypoints[i - 1]
        p2 = self.waypoints[i]
        p3 = self.waypoints[i + 1]

        # 세 변의 길이 계산
        a = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)  # p1-p2
        b = math.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)  # p2-p3
        c = math.sqrt((p3[0] - p1[0])**2 + (p3[1] - p1[1])**2)  # p1-p3

        # 너무 짧은 세그먼트는 0 반환
        if a < 0.01 or b < 0.01 or c < 0.01:
            return 0.0

        # 삼각형 면적 계산 (외적 사용)
        cross_product = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) -
                          (p2[1] - p1[1]) * (p3[0] - p1[0]))
        area = 0.5 * cross_product

        # Menger curvature: k = 4*Area / (a*b*c)
        denominator = a * b * c
        if denominator > 1e-10:
            return 4.0 * area / denominator

        return 0.0

    def sample_centerline(self, interval: float = 0.5) -> List[Tuple[float, float, float]]:
        """중심선을 지정된 간격으로 샘플링"""
        if self._cached_total_length <= 0:
            return self.waypoints.copy()

        num_points = max(2, int(self._cached_total_length / interval))
        sampled = []

        for i in range(num_points):
            s = (i / (num_points - 1)) * self._cached_total_length
            point = self.evaluate_at_arc_length(s)
            sampled.append(point)

        return sampled

    def evaluate_at_arc_length(self, s: float) -> Tuple[float, float, float]:
        """호장 길이 s에서의 점 계산"""
        if not self.waypoints:
            return (0.0, 0.0, 0.0)

        if len(self.waypoints) == 1:
            wp = self.waypoints[0]
            return (wp[0], wp[1], wp[2])

        # s를 범위 내로 제한
        s = max(0.0, min(s, self._cached_total_length))

        # 해당하는 세그먼트 찾기
        for i in range(len(self._cached_cumulative_lengths) - 1):
            if s <= self._cached_cumulative_lengths[i + 1]:
                # 세그먼트 내에서의 비율 계산
                segment_start_s = self._cached_cumulative_lengths[i]
                segment_length = self._cached_segment_lengths[i]

                if segment_length > 0:
                    t = (s - segment_start_s) / segment_length
                else:
                    t = 0.0

                # 선형 보간
                p1 = self.waypoints[i]
                p2 = self.waypoints[i + 1]

                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])

                # yaw 계산 (세그먼트 방향)
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                yaw = math.atan2(dy, dx) if abs(dx) > 1e-6 or abs(dy) > 1e-6 else p1[2]

                return (x, y, yaw)

        # 끝점 반환
        wp = self.waypoints[-1]
        return (wp[0], wp[1], wp[2])

    def project_point(self, point: Tuple[float, float]) -> Tuple[Tuple[float, float, float], float, float]:
        """점을 도로에 투영 (투영점, 거리, 호장길이 반환)"""
        px, py = point

        # 캐시 확인 및 LRU 업데이트
        cache_key = (round(px, 2), round(py, 2))
        cached_result = self._access_cache(cache_key)
        if cached_result is not None:
            return cached_result

        if not self.waypoints:
            return (0.0, 0.0, 0.0), float('inf'), 0.0

        if len(self.waypoints) == 1:
            wp = self.waypoints[0]
            dist = math.sqrt((px - wp[0])**2 + (py - wp[1])**2)
            result = ((wp[0], wp[1], wp[2]), dist, 0.0)

            # 캐시 관리 후 저장
            self._manage_cache()
            self._projection_cache[cache_key] = result
            self._cache_access_times[cache_key] = time.time()
            return result

        min_dist = float('inf')
        best_projection = None
        best_arc_length = 0.0

        try:
            # 각 세그먼트에 대해 투영 계산
            for i in range(len(self.waypoints) - 1):
                p1 = self.waypoints[i]
                p2 = self.waypoints[i + 1]

                # 세그먼트 벡터
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length_sq = dx*dx + dy*dy

                if length_sq < 1e-10:
                    continue

                # 투영 매개변수 t 계산
                t = max(0.0, min(1.0, ((px - p1[0]) * dx + (py - p1[1]) * dy) / length_sq))

                # 투영점
                proj_x = p1[0] + t * dx
                proj_y = p1[1] + t * dy
                proj_yaw = math.atan2(dy, dx)

                # 거리 계산
                dist = math.sqrt((px - proj_x)**2 + (py - proj_y)**2)

                if dist < min_dist:
                    min_dist = dist
                    best_projection = (proj_x, proj_y, proj_yaw)
                    best_arc_length = self._cached_cumulative_lengths[i] + t * self._cached_segment_lengths[i]

            result = (best_projection, min_dist, best_arc_length)

        except Exception as e:
            print(f"Warning: Error in point projection: {e}")
            # 오류 발생 시 기본값 반환
            result = ((px, py, 0.0), 0.0, 0.0)

        # 캐시 관리 후 저장
        self._manage_cache()
        self._projection_cache[cache_key] = result
        self._cache_access_times[cache_key] = time.time()

        return result

    def _manage_cache(self):
        """개선된 캐시 크기 및 메모리 관리"""
        current_time = time.time()

        # 주기적 정리
        cleanup_interval = self._cache_config.get('cleanup_interval', 300.0)
        if current_time - self._last_cleanup_time > cleanup_interval:
            self._cleanup_old_cache_entries(current_time)
            self._last_cleanup_time = current_time

        # LRU 기반 크기 제한
        max_cache_size = self._cache_config.get('max_cache_size', 1000)
        while len(self._projection_cache) > max_cache_size:
            # 가장 오래된 항목 제거
            oldest_key = next(iter(self._projection_cache))
            del self._projection_cache[oldest_key]
            if oldest_key in self._cache_access_times:
                del self._cache_access_times[oldest_key]

    def _cleanup_old_cache_entries(self, current_time: float):
        """오래된 캐시 항목 정리"""
        max_age = self._cache_config.get('max_cache_age', 1800.0)

        expired_keys = []
        for key, last_access in self._cache_access_times.items():
            if current_time - last_access > max_age:
                expired_keys.append(key)

        for key in expired_keys:
            if key in self._projection_cache:
                del self._projection_cache[key]
            if key in self._cache_access_times:
                del self._cache_access_times[key]

    def _access_cache(self, cache_key):
        """캐시 액세스 시 LRU 업데이트"""
        if cache_key in self._projection_cache:
            # LRU 업데이트 - 항목을 맨 뒤로 이동
            value = self._projection_cache.pop(cache_key)
            self._projection_cache[cache_key] = value
            self._cache_access_times[cache_key] = time.time()
            return value
        return None

    def get_boundary_lines(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """도로 경계선 계산"""
        if self._cached_boundary_lines:
            return self._cached_boundary_lines

        if len(self.waypoints) < 2:
            return [], []

        half_width = self.width / 2
        left_boundary = []
        right_boundary = []

        for i in range(len(self.waypoints) - 1):
            p1 = self.waypoints[i]
            p2 = self.waypoints[i + 1]

            # 방향 벡터
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx*dx + dy*dy)

            if length < 1e-6:
                continue

            # 정규화된 수직 벡터
            perp_x = -dy / length
            perp_y = dx / length

            # 경계점 계산
            left_x = p1[0] + perp_x * half_width
            left_y = p1[1] + perp_y * half_width
            right_x = p1[0] - perp_x * half_width
            right_y = p1[1] - perp_y * half_width

            left_boundary.append((left_x, left_y))
            right_boundary.append((right_x, right_y))

        # 마지막 점 추가
        if len(self.waypoints) >= 2:
            p1 = self.waypoints[-2]
            p2 = self.waypoints[-1]

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx*dx + dy*dy)

            if length > 1e-6:
                perp_x = -dy / length
                perp_y = dx / length

                left_x = p2[0] + perp_x * half_width
                left_y = p2[1] + perp_y * half_width
                right_x = p2[0] - perp_x * half_width
                right_y = p2[1] - perp_y * half_width

                left_boundary.append((left_x, left_y))
                right_boundary.append((right_x, right_y))

        self._cached_boundary_lines = (left_boundary, right_boundary)
        return left_boundary, right_boundary

    def is_point_on_road(self, point: Tuple[float, float]) -> bool:
        """점이 도로 위에 있는지 확인"""
        _, distance, _ = self.project_point(point)
        return distance <= self.width / 2

    def get_curvature_at_s(self, s: float) -> float:
        """호장 길이 s에서의 곡률 반환 (선형 보간)"""
        if not self._cached_curvatures or not self._cached_cumulative_lengths:
            return 0.0

        # s를 범위 내로 제한
        s = max(0.0, min(s, self._cached_total_length))

        # s에 해당하는 세그먼트 찾기
        for i in range(len(self._cached_cumulative_lengths) - 1):
            s_start = self._cached_cumulative_lengths[i]
            s_end = self._cached_cumulative_lengths[i + 1]

            if s_start <= s <= s_end:
                # 세그먼트 내에서의 비율 계산
                segment_length = s_end - s_start
                if segment_length < 1e-10:
                    return self._cached_curvatures[i]

                # 선형 보간
                t = (s - s_start) / segment_length
                curvature_start = self._cached_curvatures[i]
                curvature_end = self._cached_curvatures[i + 1]

                return curvature_start + t * (curvature_end - curvature_start)

        # 끝점인 경우
        return self._cached_curvatures[-1] if self._cached_curvatures else 0.0

    def get_max_curvature_in_range(self, s_start: float, s_end: float) -> float:
        """호장 길이 범위 [s_start, s_end]에서의 최대 곡률 반환 (lookahead)

        Args:
            s_start: 시작 호장 길이 [m]
            s_end: 종료 호장 길이 [m]

        Returns:
            범위 내 최대 곡률 [1/m]
        """
        if not self._cached_curvatures or not self._cached_cumulative_lengths:
            return 0.0

        # 범위를 도로 길이 내로 제한
        s_start = max(0.0, min(s_start, self._cached_total_length))
        s_end = max(0.0, min(s_end, self._cached_total_length))

        if s_start >= s_end:
            return self.get_curvature_at_s(s_start)

        max_curvature = 0.0

        # 범위에 포함되는 waypoint들의 곡률 확인
        for i in range(len(self._cached_cumulative_lengths)):
            s_i = self._cached_cumulative_lengths[i]

            # waypoint가 범위 내에 있으면 곡률 확인
            if s_start <= s_i <= s_end:
                max_curvature = max(max_curvature, self._cached_curvatures[i])

        # 시작점과 끝점의 보간된 곡률도 확인
        curvature_start = self.get_curvature_at_s(s_start)
        curvature_end = self.get_curvature_at_s(s_end)
        max_curvature = max(max_curvature, curvature_start, curvature_end)

        return max_curvature

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """바운딩 박스 계산"""
        if not self.waypoints:
            return (0, 0, 0, 0)

        xs = [wp[0] for wp in self.waypoints]
        ys = [wp[1] for wp in self.waypoints]

        half_width = self.width / 2

        return (
            min(xs) - half_width,
            min(ys) - half_width,
            max(xs) + half_width,
            max(ys) + half_width
        )

    def draw(self, screen, world_to_screen_func, debug=False):
        """pygame 시각화"""
        if len(self.waypoints) < 2:
            return

        road_color = (100, 100, 100)
        line_color = (255, 255, 0)

        # 경계선 그리기
        left_boundary, right_boundary = self.get_boundary_lines()

        if left_boundary and right_boundary:
            left_screen = [world_to_screen_func(pt) for pt in left_boundary]
            right_screen = [world_to_screen_func(pt) for pt in right_boundary]

            # 중심선
            center_points = [(wp[0], wp[1]) for wp in self.waypoints]
            center_screen = [world_to_screen_func(pt) for pt in center_points]

            pygame.draw.lines(screen, road_color, False, center_screen, 2)
            pygame.draw.lines(screen, line_color, False, left_screen, 2)
            pygame.draw.lines(screen, line_color, False, right_screen, 2)

        if debug:
            font = pygame.font.SysFont(None, 12)
            text = font.render(f"Seg:{self.id[:8]}", True, (255, 255, 255))
            if self.waypoints:
                mid_idx = len(self.waypoints) // 2
                mid_point = self.waypoints[mid_idx]
                mid_screen = world_to_screen_func((mid_point[0], mid_point[1]))
                screen.blit(text, (int(mid_screen[0]), int(mid_screen[1]) - 20))

    def get_serializable_state(self) -> Dict[str, Any]:
        """직렬화 가능한 상태 반환"""
        return {
            "id": self.id,
            "waypoints": self.waypoints,
            "width": self.width,
            "speed_limit": self.speed_limit
        }

    def load_from_serialized(self, data: Dict[str, Any]):
        """직렬화된 데이터로부터 복원"""
        self.id = data["id"]
        self.waypoints = data["waypoints"]
        self.width = data["width"]
        self.speed_limit = data["speed_limit"]
        self._precompute_geometry()


# =============================================================================
# 4. 최적화된 Frenet 좌표계 계산
# =============================================================================

class FrenetCalculator:
    """최적화된 Frenet 좌표계 계산기"""

    def __init__(self):
        self._cache = {}
        self._cache_max_size = 200
        self._last_vehicle_pos = None
        self._last_result = None
        self._cache_threshold = 0.1  # 0.1m 이내면 캐시 사용

    def calculate_frenet(self, vehicle_pos: Tuple[float, float, float],
                        segment: LinearRoadSegment) -> FrenetState:
        """차량 위치에 대한 Frenet 좌표 계산"""
        try:
            x, y, yaw = vehicle_pos

            # 빠른 캐시 확인
            if self._should_use_last_cache(vehicle_pos):
                return self._last_result

            # 투영 계산
            projected_point, distance, arc_length = segment.project_point((x, y))

            # 횡방향 거리 부호 계산
            ref_x, ref_y, ref_yaw = projected_point

            # 차량에서 참조점으로의 벡터
            dx = x - ref_x
            dy = y - ref_y

            # 경로 방향에 수직인 벡터 (왼쪽 방향)
            perp_x = -math.sin(ref_yaw)
            perp_y = math.cos(ref_yaw)

            # 횡방향 거리 (왼쪽이 양수)
            d = dx * perp_x + dy * perp_y

            result = FrenetState(
                s=arc_length,
                d=d,
                x_ref=ref_x,
                y_ref=ref_y,
                yaw_ref=ref_yaw,
                segment_id=segment.id
            )

            # 캐시 저장
            self._last_vehicle_pos = vehicle_pos
            self._last_result = result

            return result

        except Exception as e:
            print(f"Warning: Error calculating Frenet coordinates: {e}")
            # 오류 발생 시 기본값 반환
            x, y, yaw = vehicle_pos
            return FrenetState(
                s=0.0,
                d=0.0,
                x_ref=x,
                y_ref=y,
                yaw_ref=yaw,
                segment_id=segment.id if segment else "unknown"
            )

    def _should_use_last_cache(self, vehicle_pos: Tuple[float, float, float]) -> bool:
        """마지막 캐시 사용 여부 판단"""
        if self._last_vehicle_pos is None or self._last_result is None:
            return False

        dx = vehicle_pos[0] - self._last_vehicle_pos[0]
        dy = vehicle_pos[1] - self._last_vehicle_pos[1]
        distance = math.sqrt(dx*dx + dy*dy)

        return distance < self._cache_threshold


# =============================================================================
# 5. 경로 계획 시스템 (기존 Dubins 유지)
# =============================================================================

class Dubins:
    """Dubins 경로 계획 클래스"""

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
    def __init__(self, position, time, cost, parent_pos=None):
        self.destination_list = []
        self.position = position
        self.time = time
        self.cost = cost
        self.parent_pos = parent_pos


class RRTEdge:
    def __init__(self, node_from, node_to, path, cost):
        self.node_from = node_from
        self.node_to = node_to
        self.path = deque(path)
        self.cost = cost


class PathPlanner:
    """경로 계획 클래스"""

    def __init__(self, config):
        self.local_planner = Dubins(radius=config["min_radius"],
                                  point_separation=config["point_separation"])
        self.precision = (5, 5, 1)
        self.width = config["road_width"]
        self.config = config
        self.default_speed = self.config.get('default_speed', 15.0)
        self.min_speed = self.config.get('min_speed', 5.0)
        self.max_speed = self.config.get('max_speed', 25.0)
        self.max_lateral_accel = self.config.get('max_lateral_acceleration', 4.0)

    def plan_path(self, start: Tuple[float, float, float],
                 end: Tuple[float, float, float],
                 obstacles: List = [],
                 mode: str = 'auto') -> Optional[PlannedPath]:
        """경로 계획 메인 메서드"""

        if mode == 'auto':
            # 자동 모드 선택 (로컬 좌표계 기반 4사분면)
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            distance = math.sqrt(dx*dx + dy*dy)

            # 로컬 좌표계 변환 (시작점 기준)
            cos_yaw = math.cos(-start[2])
            sin_yaw = math.sin(-start[2])
            dx_local = dx * cos_yaw - dy * sin_yaw
            dy_local = dx * sin_yaw + dy * cos_yaw

            # 로컬 좌표계에서 목적지 방향 (차량 진행 방향 기준)
            local_target_angle = math.atan2(dy_local, dx_local)

            # 출발지와 목적지의 yaw 차이
            yaw_diff = abs(normalize_angle(end[2] - start[2]))

            if len(obstacles) > 0:
                mode = 'rrt'
            elif dy_local < 0:  # 3,4사분면 (뒤쪽)
                mode = 'rrt'
            elif abs(local_target_angle) <= math.radians(5) and yaw_diff <= math.radians(5):
                # 목적지가 앞쪽 -5도~5도 & yaw 차이 5도 이내
                mode = 'straight'
            elif abs(local_target_angle) <= math.radians(60) and yaw_diff <= math.radians(90):
                # 목적지가 앞쪽 -60도~60도 & yaw 차이 90도 이내
                mode = 'curve'
            else:  # 1,2사분면 & 그 외
                mode = 'rrt'

        if mode == 'rrt':
            waypoints, total_length = self._rrt_with_dubins(start, end, obstacles)
        elif mode == 'curve':
            waypoints, total_length = self._quadratic_bezier_curve(start, end)
        else:
            waypoints, total_length = self._straight_line(start, end)

        if not waypoints:
            return None

        return PlannedPath(
            waypoints=waypoints,
            total_length=total_length,
            planning_mode=mode
        )

    def _straight_line(self, start, end):
        """직선 경로"""
        waypoints = []
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.sqrt(dx*dx + dy*dy)

        if distance < 0.1:
            return [start, end], distance

        yaw = math.atan2(dy, dx)

        for i in range(5):
            t = i / 4.0
            x = start[0] + t * dx
            y = start[1] + t * dy
            waypoints.append((x, y, yaw))

        return waypoints, distance

    def _quadratic_bezier_curve(self, start, end):
        """베지어 곡선 경로"""
        start_pos = np.array([start[0], start[1]])
        end_pos = np.array([end[0], end[1]])
        start_dir = np.array([math.cos(start[2]), math.sin(start[2])])
        end_dir = np.array([math.cos(end[2]), math.sin(end[2])])

        distance = np.linalg.norm(end_pos - start_pos)

        if distance < 0.1:
            return [start, end], distance

        # 제어점 계산
        control_distance = distance * 0.5
        control_point = (start_pos + end_pos) / 2

        # 시작점과 끝점의 방향을 고려한 제어점 조정
        avg_dir = (start_dir + end_dir) / 2
        if np.linalg.norm(avg_dir) > 0.1:
            perp_dir = np.array([-avg_dir[1], avg_dir[0]])
            control_point += perp_dir * control_distance * 0.3

        # 베지어 곡선 생성
        waypoints = []
        total_length = 0.0

        for i in range(20):
            t = i / 19.0

            pos = ((1-t)**2 * start_pos +
                   2*(1-t)*t * control_point +
                   t**2 * end_pos)

            # 접선 방향 계산
            if i == 0:
                yaw = start[2]
            elif i == 19:
                yaw = end[2]
            else:
                tangent = (2*(1-t) * (control_point - start_pos) +
                          2*t * (end_pos - control_point))
                if np.linalg.norm(tangent) > 1e-6:
                    yaw = math.atan2(tangent[1], tangent[0])
                else:
                    yaw = start[2] if t < 0.5 else end[2]

            waypoints.append((pos[0], pos[1], yaw))

            if i > 0:
                prev_pos = np.array([waypoints[i-1][0], waypoints[i-1][1]])
                segment_length = np.linalg.norm(pos - prev_pos)
                total_length += segment_length

        return waypoints, total_length

    def _rrt_with_dubins(self, start, end, obstacles):
        """RRT-Dubins 경로 계획 (성능 최적화)"""
        nodes = {}
        edges = {}
        goal = end
        collision_dist = self.width / 2.0
        obstacles_array = np.array(obstacles) if obstacles else np.array([])

        # 기존 경로 선분 캐시 [x1, y1, x2, y2] 형태
        cached_segments = []

        # 공간 인덱싱을 위한 그리드 (셀 크기 = max_distance)
        segment_grid = {}  # {(grid_x, grid_y): [segment_indices]}
        grid_cell_size = self.config.get("max_distance", 50.0)

        nodes[start] = RRTNode(start, 0, 0, parent_pos=None)

        # 노드 위치 관리용 리스트
        node_keys = [start]
        node_positions = [[start[0], start[1]]]

        base_step = self.config.get("base_step_size", 1.0)
        min_factor = self.config.get("min_step_factor", 1.0)
        max_factor = self.config.get("max_step_factor", 10.0)
        max_distance = self.config.get("max_distance", 50.0)  # 기준 거리 [m]
        k_nearest = self.config.get("nearest_neighbor_count", 15)  # KD-Tree 최근접 이웃 수

        # KD-Tree 재구성 주기 설정
        kdtree_rebuild_interval = self.config.get("kdtree_rebuild_interval", 15)  # 15회마다 재구성
        kdtree = None
        distance_to_goal = float('inf')
        current_step_size = base_step * max_factor

        for i_iter in range(self.config.get("max_iterations", 1000)):
            # KD-Tree 주기적 재구성 (성능 최적화)
            if i_iter % kdtree_rebuild_interval == 0:
                node_positions_array = np.array(node_positions)
                kdtree = KDTree(node_positions_array)

                # Goal 거리도 재구성 시에만 계산
                _, closest_idx = kdtree.query([goal[0], goal[1]])
                closest_node = node_keys[closest_idx]
                distance_to_goal = math.sqrt((closest_node[0]-goal[0])**2 + (closest_node[1]-goal[1])**2)

                # 거리 비율 계산 (0~1)
                distance_ratio = min(distance_to_goal / max_distance, 1.0)

                # step_factor 계산 (가까울수록 작게, 멀수록 크게)
                step_factor = min_factor + (max_factor - min_factor) * distance_ratio
                current_step_size = base_step * step_factor

            # 샘플링
            if random.random() < self.config.get("goal_sampling_rate", 0.1):
                sample = goal
            else:
                rand_x = random.uniform(min(start[0], goal[0]) - current_step_size,
                                      max(start[0], goal[0]) + current_step_size)
                rand_y = random.uniform(min(start[1], goal[1]) - current_step_size,
                                      max(start[1], goal[1]) + current_step_size)
                rand_yaw = random.uniform(-math.pi, math.pi)
                sample = (rand_x, rand_y, rand_yaw)

            # KD-Tree 기반 최적화된 옵션 선택
            options = self._select_options_optimized(nodes, node_positions_array, kdtree, sample, 10, k_nearest)
            for node_from_option, opt_data in options:
                if opt_data[0] == float('inf'):
                    break

                path_points = self.local_planner.generate_points(node_from_option, sample,
                                                               opt_data[1], opt_data[2])

                if len(obstacles_array) > 0 and self._is_collision_path(path_points, obstacles_array, collision_dist):
                    continue

                # 공간 인덱싱 기반 교차 검사
                if self._is_path_crossing_spatial(path_points, cached_segments, segment_grid, grid_cell_size):
                    continue

                parent_node = nodes[node_from_option]
                new_cost = parent_node.cost + opt_data[0]

                if sample not in nodes or new_cost < nodes[sample].cost:
                    nodes[sample] = RRTNode(sample, parent_node.time + opt_data[0],
                                          new_cost, parent_pos=node_from_option)
                    edges[(node_from_option, sample)] = RRTEdge(node_from_option, sample,
                                                              path_points, opt_data[0])

                    # 노드 위치 추가
                    node_keys.append(sample)
                    node_positions.append([sample[0], sample[1]])

                    # 새 경로의 선분들을 캐시 및 그리드에 추가
                    if len(path_points) >= 2:
                        segments = np.column_stack([
                            path_points[:-1, :2],  # 시작점 (x1, y1)
                            path_points[1:, :2]    # 끝점 (x2, y2)
                        ])  # shape: (N-1, 4)
                        segment_idx = len(cached_segments)
                        cached_segments.append(segments)

                        # 그리드에 등록
                        self._add_segments_to_grid(segments, segment_idx, segment_grid, grid_cell_size)

                    if self._is_goal(sample, goal):
                        return self._reconstruct_path(start, sample, nodes, edges)
                break

        # RRT 실패시 예외 발생
        raise RuntimeError("RRT path planning failed")

    def _select_options(self, nodes, sample, nb_options):
        """Dubins 옵션 선택"""
        options = []
        for node in nodes:
            options.extend([(node, opt) for opt in self.local_planner.all_options(node, sample)])
        options.sort(key=lambda x: x[1][0])
        return options[:nb_options]

    def _select_options_optimized(self, nodes, node_positions, kdtree, sample, nb_options, k_nearest):
        """KD-Tree 기반 최적화된 Dubins 옵션 선택

        Args:
            nodes: 노드 딕셔너리
            node_positions: 노드 위치 배열 (N, 2)
            kdtree: KDTree 객체
            sample: 샘플 위치 (x, y, yaw)
            nb_options: 반환할 최대 옵션 수
            k_nearest: 검색할 최근접 이웃 수

        Returns:
            상위 nb_options 개의 (node, dubins_option) 튜플 리스트
        """
        # sample 근처 k개 노드 찾기
        k_search = min(k_nearest, len(node_positions))
        distances, indices = kdtree.query([sample[0], sample[1]], k=k_search)

        # 근접 노드들에 대해서만 Dubins 경로 계산
        options = []
        node_keys = list(nodes.keys())

        # indices가 스칼라인 경우 리스트로 변환
        if k_search == 1:
            indices = [indices]

        for idx in indices:
            node = node_keys[idx]
            options.extend([(node, opt) for opt in self.local_planner.all_options(node, sample)])

        # 비용 기준 정렬
        options.sort(key=lambda x: x[1][0])
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

    def _vectorized_segments_intersect(self, seg1, seg2):
        """벡터화된 선분 교차 검사 (CCW 알고리즘)

        Args:
            seg1: 새 경로 선분들 (shape: (N, 4)) - [x1, y1, x2, y2]
            seg2: 기존 경로 선분들 (shape: (M, 4)) - [x1, y1, x2, y2]

        Returns:
            True if any pair intersects, False otherwise
        """
        # Broadcasting: (N, 1, 2) vs (1, M, 2) → (N, M, 2)
        p1 = seg1[:, np.newaxis, :2]  # (N, 1, 2) - seg1의 시작점
        p2 = seg1[:, np.newaxis, 2:]  # (N, 1, 2) - seg1의 끝점
        p3 = seg2[np.newaxis, :, :2]  # (1, M, 2) - seg2의 시작점
        p4 = seg2[np.newaxis, :, 2:]  # (1, M, 2) - seg2의 끝점

        # CCW 계산 (Counter-clockwise)
        def ccw(A, B, C):
            return ((C[..., 1] - A[..., 1]) * (B[..., 0] - A[..., 0]) >
                    (B[..., 1] - A[..., 1]) * (C[..., 0] - A[..., 0]))

        # 교차 조건: p1-p2 기준 p3, p4가 반대편 AND p3-p4 기준 p1, p2가 반대편
        # 결과: (N, M) boolean matrix
        intersections = ((ccw(p1, p3, p4) != ccw(p2, p3, p4)) &
                         (ccw(p1, p2, p3) != ccw(p1, p2, p4)))

        return np.any(intersections)

    def _add_segments_to_grid(self, segments, segment_idx, segment_grid, grid_cell_size):
        """세그먼트를 공간 그리드에 추가

        Args:
            segments: 선분 배열 (shape: (N, 4)) - [x1, y1, x2, y2]
            segment_idx: 캐시된 세그먼트 인덱스
            segment_grid: 그리드 딕셔너리 {(grid_x, grid_y): [indices]}
            grid_cell_size: 그리드 셀 크기 [m]
        """
        for seg in segments:
            x1, y1, x2, y2 = seg
            # 선분이 걸치는 모든 그리드 셀 계산
            min_x, max_x = min(x1, x2), max(x1, x2)
            min_y, max_y = min(y1, y2), max(y1, y2)

            grid_x_min = int(min_x // grid_cell_size)
            grid_x_max = int(max_x // grid_cell_size)
            grid_y_min = int(min_y // grid_cell_size)
            grid_y_max = int(max_y // grid_cell_size)

            for gx in range(grid_x_min, grid_x_max + 1):
                for gy in range(grid_y_min, grid_y_max + 1):
                    grid_key = (gx, gy)
                    if grid_key not in segment_grid:
                        segment_grid[grid_key] = []
                    if segment_idx not in segment_grid[grid_key]:
                        segment_grid[grid_key].append(segment_idx)

    def _is_path_crossing_spatial(self, new_path, cached_segments, segment_grid, grid_cell_size):
        """공간 인덱싱 기반 경로 교차 검사

        Args:
            new_path: 새로 생성된 경로 (numpy array, shape: (N, 2))
            cached_segments: 기존 선분 리스트 [array(M1, 4), array(M2, 4), ...]
            segment_grid: 그리드 딕셔너리 {(grid_x, grid_y): [indices]}
            grid_cell_size: 그리드 셀 크기 [m]

        Returns:
            True if paths cross, False otherwise
        """
        if not cached_segments or len(new_path) < 2:
            return False

        # 1. 새 경로의 선분들을 [x1, y1, x2, y2] 형태로 변환
        new_segments = np.column_stack([
            new_path[:-1, :2],  # (N-1, 2)
            new_path[1:, :2]    # (N-1, 2)
        ])  # (N-1, 4)

        # 2. 새 경로가 걸치는 그리드 셀들 찾기
        relevant_segment_indices = set()
        for seg in new_segments:
            x1, y1, x2, y2 = seg
            min_x, max_x = min(x1, x2), max(x1, x2)
            min_y, max_y = min(y1, y2), max(y1, y2)

            grid_x_min = int(min_x // grid_cell_size)
            grid_x_max = int(max_x // grid_cell_size)
            grid_y_min = int(min_y // grid_cell_size)
            grid_y_max = int(max_y // grid_cell_size)

            for gx in range(grid_x_min, grid_x_max + 1):
                for gy in range(grid_y_min, grid_y_max + 1):
                    grid_key = (gx, gy)
                    if grid_key in segment_grid:
                        relevant_segment_indices.update(segment_grid[grid_key])

        # 3. 관련 세그먼트만 교차 검사
        if not relevant_segment_indices:
            return False

        relevant_segments = [cached_segments[i] for i in relevant_segment_indices]
        existing_segments = np.vstack(relevant_segments)  # (M, 4)

        # 4. 벡터화된 교차 검사
        return self._vectorized_segments_intersect(new_segments, existing_segments)

    def _is_path_crossing_cached(self, new_path, cached_segments):
        """새 경로가 캐시된 기존 경로 선분들과 교차하는지 검사

        Args:
            new_path: 새로 생성된 경로 (numpy array, shape: (N, 2))
            cached_segments: 기존 선분 리스트 [array(M1, 4), array(M2, 4), ...]
                            각 행은 [x1, y1, x2, y2]

        Returns:
            True if paths cross, False otherwise
        """
        if not cached_segments or len(new_path) < 2:
            return False

        # 1. 새 경로의 선분들을 [x1, y1, x2, y2] 형태로 변환
        new_segments = np.column_stack([
            new_path[:-1, :2],  # (N-1, 2)
            new_path[1:, :2]    # (N-1, 2)
        ])  # (N-1, 4)

        # 2. 캐시된 선분들을 하나의 배열로 통합
        existing_segments = np.vstack(cached_segments)  # (M, 4)

        # 3. 벡터화된 교차 검사
        return self._vectorized_segments_intersect(new_segments, existing_segments)

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

        path_segments.reverse()
        for segment in path_segments:
            for point in segment:
                if len(point) == 2:
                    if len(waypoints) > 0:
                        prev = waypoints[-1]
                        yaw = math.atan2(point[1] - prev[1], point[0] - prev[0])
                    else:
                        yaw = 0.0
                    waypoints.append((point[0], point[1], yaw))
                else:
                    waypoints.append((point[0], point[1], point[2] if len(point) > 2 else 0.0))

        return waypoints, total_length

# =============================================================================
# 6. 네트워크 및 공간 인덱싱
# =============================================================================

class SimpleGridIndex:
    """간단한 그리드 기반 공간 인덱스"""

    def __init__(self, cell_size: float = 50.0):
        self.cell_size = cell_size
        self.grid = {}  # {(grid_x, grid_y): [segment_ids]}
        self.segments = {}  # {segment_id: segment}

    def _get_grid_coords(self, x: float, y: float) -> Tuple[int, int]:
        """월드 좌표를 그리드 좌표로 변환"""
        return (int(x // self.cell_size), int(y // self.cell_size))

    def add_segment(self, segment: LinearRoadSegment):
        """세그먼트 추가"""
        self.segments[segment.id] = segment

        # 바운딩 박스 계산
        min_x, min_y, max_x, max_y = segment.get_bounding_box()

        # 해당하는 모든 그리드 셀에 추가
        min_grid_x, min_grid_y = self._get_grid_coords(min_x, min_y)
        max_grid_x, max_grid_y = self._get_grid_coords(max_x, max_y)

        for gx in range(min_grid_x, max_grid_x + 1):
            for gy in range(min_grid_y, max_grid_y + 1):
                grid_key = (gx, gy)
                if grid_key not in self.grid:
                    self.grid[grid_key] = []
                if segment.id not in self.grid[grid_key]:
                    self.grid[grid_key].append(segment.id)

    def remove_segment(self, segment_id: str):
        """세그먼트 제거"""
        if segment_id in self.segments:
            segment = self.segments[segment_id]

            # 모든 그리드에서 제거
            for grid_key, segment_ids in self.grid.items():
                if segment_id in segment_ids:
                    segment_ids.remove(segment_id)

            del self.segments[segment_id]

    def find_nearby_segments(self, point: Tuple[float, float],
                           radius: float) -> List[LinearRoadSegment]:
        """반경 내 세그먼트 검색"""
        x, y = point

        # 검색 영역의 그리드 범위 계산
        min_grid_x, min_grid_y = self._get_grid_coords(x - radius, y - radius)
        max_grid_x, max_grid_y = self._get_grid_coords(x + radius, y + radius)

        candidate_ids = set()
        for gx in range(min_grid_x, max_grid_x + 1):
            for gy in range(min_grid_y, max_grid_y + 1):
                grid_key = (gx, gy)
                if grid_key in self.grid:
                    candidate_ids.update(self.grid[grid_key])

        # 실제 거리 확인
        nearby_segments = []
        for segment_id in candidate_ids:
            segment = self.segments[segment_id]
            _, distance, _ = segment.project_point(point)
            if distance <= radius:
                nearby_segments.append(segment)

        # 거리순 정렬
        nearby_segments.sort(key=lambda seg: seg.project_point(point)[1])

        return nearby_segments

    def get_all_segments(self) -> List[LinearRoadSegment]:
        """모든 세그먼트 반환"""
        return list(self.segments.values())

    def clear(self):
        """모든 데이터 클리어"""
        self.grid.clear()
        self.segments.clear()


class RoadNetwork:
    """도로 네트워크 관리 시스템"""

    def __init__(self, config: Optional[Dict] = None):
        self.segments = {}  # {segment_id: LinearRoadSegment}
        self.connections = []  # List[RoadConnection]
        self.spatial_index = SimpleGridIndex()
        self.frenet_calculator = FrenetCalculator()

        # 캐시 설정 추출
        self._road_cache_config = None
        if config and 'memory_management' in config:
            self._road_cache_config = config['memory_management'].get('road_cache', {})

        # 메타데이터
        self.name = "Road Network"
        self.version = "1.0"
        self.metadata = {}

    def add_segment(self, segment: LinearRoadSegment) -> bool:
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

    def get_segment(self, segment_id: str) -> Optional[LinearRoadSegment]:
        """세그먼트 ID로 도로 세그먼트 조회"""
        return self.segments.get(segment_id)

    def find_closest_segment(self, point: Tuple[float, float]) -> Optional[LinearRoadSegment]:
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
            _, distance, _ = segment.project_point(point)
            if distance < min_distance:
                min_distance = distance
                closest_segment = segment

        return closest_segment

    def calculate_frenet(self, vehicle_pos: Tuple[float, float, float]) -> Optional[Tuple[FrenetState, LinearRoadSegment]]:
        """차량 위치에 대한 Frenet 좌표 계산"""
        try:
            closest_segment = self.find_closest_segment(vehicle_pos[:2])

            if not closest_segment:
                return None, None

            frenet_state = self.frenet_calculator.calculate_frenet(vehicle_pos, closest_segment)
            return frenet_state, closest_segment

        except Exception as e:
            print(f"Warning: Error in RoadNetwork calculate_frenet: {e}")
            return None, None

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

    def get_all_segments(self) -> List[LinearRoadSegment]:
        """모든 세그먼트 반환"""
        return list(self.segments.values())

    def get_segment_count(self) -> int:
        """세그먼트 개수 반환"""
        return len(self.segments)

    def draw(self, screen, world_to_screen_func, debug=False):
        """네트워크 시각화"""
        for segment in self.segments.values():
            segment.draw(screen, world_to_screen_func, debug)

    def get_serializable_state(self) -> Dict[str, Any]:
        """네트워크 직렬화"""
        return {
            "name": self.name,
            "version": self.version,
            "metadata": self.metadata,
            "segments": [segment.get_serializable_state() for segment in self.segments.values()],
            "connections": [
                {
                    "from": conn.from_segment_id,
                    "to": conn.to_segment_id,
                    "point": list(conn.connection_point),
                    "type": conn.connection_type
                }
                for conn in self.connections
            ]
        }

    def load_from_serialized(self, data: Dict[str, Any]):
        """직렬화된 데이터로부터 복원"""
        self.clear()

        self.name = data.get("name", "Road Network")
        self.version = data.get("version", "1.0")
        self.metadata = data.get("metadata", {})

        # 세그먼트 복원
        for seg_data in data.get("segments", []):
            segment = LinearRoadSegment([], 6.0, 15.0)
            segment.load_from_serialized(seg_data)
            self.add_segment(segment)

        # 연결 복원
        for conn_data in data.get("connections", []):
            connection = RoadConnection(
                from_segment_id=conn_data["from"],
                to_segment_id=conn_data["to"],
                connection_point=tuple(conn_data["point"]),
                connection_type=conn_data.get("type", "simple")
            )
            self.connect_segments(connection)


# =============================================================================
# 7. YAML 설정 및 직렬화 시스템
# =============================================================================

class RoadConfigLoader:
    """YAML 설정 로더"""

    def __init__(self):
        self.path_planner = None

    def set_path_planner(self, path_planner: PathPlanner):
        """경로 계획기 설정"""
        self.path_planner = path_planner

    def load_config_from_yaml(self, config_file: str) -> Dict[str, Any]:
        """YAML 설정 파일 로드"""
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # path_planning 설정 추출
        return config.get('simulation', {}).get('path_planning', {})

    def create_network_from_config(self, config: Dict[str, Any],
                                  road_definitions: List[Dict[str, Any]]) -> RoadNetwork:
        """설정으로부터 네트워크 생성"""
        network = RoadNetwork(config)

        # 캐시 설정 추출
        road_cache_config = None
        if 'memory_management' in config:
            road_cache_config = config['memory_management'].get('road_cache', {})

        for road_def in road_definitions:
            waypoints = road_def.get('waypoints', [])
            if len(waypoints) < 2:
                continue

            # 웨이포인트 변환
            converted_waypoints = []
            for wp in waypoints:
                if len(wp) >= 2:
                    x, y = wp[0], wp[1]
                    yaw = wp[2] if len(wp) > 2 else 0.0
                    converted_waypoints.append((x, y, yaw))

            if len(converted_waypoints) >= 2:
                segment = LinearRoadSegment(
                    waypoints=converted_waypoints,
                    width=road_def.get('width', config.get('road_width', 6.0)),
                    speed_limit=road_def.get('speed_limit', config.get('default_speed', 15.0)),
                    segment_id=road_def.get('id', f"road_{len(network.segments)}"),
                    cache_config=road_cache_config
                )
                network.add_segment(segment)

        return network

    def save_network_to_file(self, network: RoadNetwork, file_path: str):
        """네트워크를 JSON 파일로 저장"""
        data = network.get_serializable_state()

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_network_from_file(self, file_path: str) -> RoadNetwork:
        """JSON 파일로부터 네트워크 로드"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        network = RoadNetwork()
        network.load_from_serialized(data)

        return network


# =============================================================================
# 8. 통합 API (RoadSystemAPI)
# =============================================================================

class RoadSystemAPI:
    """도로 시스템의 통합 API"""

    def __init__(self, config: Optional[Dict] = None):
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

        self.network = RoadNetwork(config)
        self.path_planner = PathPlanner(self.config)
        self.config_loader = RoadConfigLoader()
        self.config_loader.set_path_planner(self.path_planner)

        # 속도 계산을 위한 lookahead 거리 설정 [m]
        self.lookahead_distance = self.config.get('lookahead_distance', 15.0)

    # === 네트워크 생성 및 로딩 ===

    def load_config_from_yaml(self, config_file: str) -> 'RoadSystemAPI':
        """YAML 설정 파일 로드"""
        config = self.config_loader.load_config_from_yaml(config_file)
        self.config.update(config)
        self.path_planner = PathPlanner(self.config)
        return self

    def load_network_from_file(self, file_path: str) -> 'RoadSystemAPI':
        """JSON 파일로부터 네트워크 로드"""
        self.network = self.config_loader.load_network_from_file(file_path)
        return self

    def save_network_to_file(self, file_path: str):
        """네트워크를 파일로 저장"""
        self.config_loader.save_network_to_file(self.network, file_path)

    def create_empty_network(self, name: str = "Road Network") -> 'RoadSystemAPI':
        """빈 네트워크 생성"""
        self.network.clear()
        self.network.name = name
        return self

    # === 도로 생성 메서드 ===

    def add_straight_road(self, start: Tuple[float, float, float], length: float,
                         width: float = None, speed_limit: float = None,
                         road_id: Optional[str] = None) -> 'RoadSystemAPI':
        """직선 도로 추가"""
        if width is None:
            width = self.config['road_width']
        if speed_limit is None:
            speed_limit = self.config['default_speed']

        x, y, yaw = start
        end_x = x + length * math.cos(yaw)
        end_y = y + length * math.sin(yaw)

        waypoints = [(x, y, yaw), (end_x, end_y, yaw)]

        segment = LinearRoadSegment(
            waypoints=waypoints,
            width=width,
            speed_limit=speed_limit,
            segment_id=road_id or f"straight_{len(self.network.segments)}",
            cache_config=self.network._road_cache_config
        )
        self.network.add_segment(segment)
        return self

    def add_point_to_point_road(self, start: Tuple[float, float, float],
                               end: Tuple[float, float, float],
                               width: float = None, speed_limit: float = None,
                               road_id: Optional[str] = None,
                               planning_mode: str = 'auto',
                               obstacles: List = []) -> bool:
        """두 점 사이 도로 추가"""
        if width is None:
            width = self.config['road_width']
        if speed_limit is None:
            speed_limit = self.config['default_speed']

        planned_path = self.path_planner.plan_path(start, end, obstacles, planning_mode)

        if planned_path and planned_path.waypoints:
            if not road_id:
                road_id = f"road_{len(self.network.segments)}"

            segment = LinearRoadSegment(
                waypoints=planned_path.waypoints,
                width=width,
                speed_limit=speed_limit,
                segment_id=road_id,
                cache_config=self.network._road_cache_config
            )
            self.network.add_segment(segment)
            return True
        return False

    def add_waypoint_road(self, waypoints: List[Tuple[float, float, float]],
                         width: float = None, speed_limit: float = None,
                         road_id: Optional[str] = None) -> 'RoadSystemAPI':
        """웨이포인트로 도로 추가"""
        if len(waypoints) < 2:
            return self

        if width is None:
            width = self.config['road_width']
        if speed_limit is None:
            speed_limit = self.config['default_speed']

        if not road_id:
            road_id = f"road_{len(self.network.segments)}"

        segment = LinearRoadSegment(
            waypoints=waypoints,
            width=width,
            speed_limit=speed_limit,
            segment_id=road_id,
            cache_config=self.network._road_cache_config
        )
        self.network.add_segment(segment)
        return self

    # === 차량 위치 계산 (핵심 API) ===

    def get_vehicle_road_info(self, vehicle_pos: Tuple[float, float, float]) -> Optional[Dict[str, Any]]:
        """차량의 도로 정보 계산 (Frenet 좌표, 권장 속도, heading error 포함)"""
        frenet_state, closest_segment = self.network.calculate_frenet(vehicle_pos)

        if not frenet_state:
            return None

        # 권장 속도 계산 (Lookahead 범위 최대 곡률 기반)
        if closest_segment:
            s_current = frenet_state.s
            s_lookahead = s_current + self.lookahead_distance
            max_curvature_ahead = closest_segment.get_max_curvature_in_range(
                s_current, s_lookahead
            )
            recommended_speed = self._calculate_speed_from_curvature(max_curvature_ahead)
        else:
            recommended_speed = self.config.get('default_speed', 15.0)

        # Heading error 계산 (차량 기준 상대 각도)
        vehicle_yaw = vehicle_pos[2]
        road_yaw = frenet_state.yaw_ref
        heading_error = normalize_angle(road_yaw - vehicle_yaw)

        # 세그먼트 전체 길이
        segment_length = closest_segment.get_length() if closest_segment else 0.0

        return {
            'closest_segment': closest_segment,
            'frenet_state': frenet_state,
            'segment_id': frenet_state.segment_id,
            'distance_to_center': abs(frenet_state.d),
            'is_on_road': self.network.is_point_on_road(vehicle_pos[:2]),
            's': frenet_state.s,
            'd': frenet_state.d,
            's_dot': frenet_state.s_dot,
            'd_dot': frenet_state.d_dot,
            'road_center_point': (frenet_state.x_ref, frenet_state.y_ref),
            'road_yaw': frenet_state.yaw_ref,
            'recommended_speed': recommended_speed,
            'heading_error': heading_error,
            'segment_length': segment_length
        }

    def _calculate_speed_from_curvature(self, curvature: float) -> float:
        """곡률로부터 권장 속도 계산 (물리 기반 선형)

        Args:
            curvature: 경로의 곡률 [1/m]

        Returns:
            권장 속도 [m/s]
        """
        min_speed = self.config.get('min_speed', 5.0)
        max_speed = self.config.get('max_speed', 25.0)
        max_lateral_accel = self.config.get('max_lateral_acceleration', 4.0)

        # 곡률이 거의 0이면 (직선 경로) 최대 속도 사용
        if curvature < 1e-6:
            return max_speed

        # 물리 기반 속도 계산: v = sqrt(a_lat_max / curvature)
        safe_speed = math.sqrt(max_lateral_accel / curvature)

        # min_speed와 max_speed 사이로 제한
        return max(min_speed, min(max_speed, safe_speed))

    def get_vehicle_update_data(self, vehicle_position: Tuple[float, float, float]):
        """차량 업데이트 데이터 계산 (곡률 기반 동적 권장 속도)

        Returns:
            Tuple[closest_segment, road_center_point, d, recommended_speed, is_outside_road, heading_error, frenet_s, segment_length]
            - closest_segment: 가장 가까운 도로 세그먼트 객체 (없으면 None)
            - road_center_point: 도로 중심점 (x, y)
            - d: 횡방향 거리 [m] (왼쪽 양수)
            - recommended_speed: 곡률 기반 권장 종방향 속도 [m/s]
            - is_outside_road: 도로 이탈 여부
            - heading_error: 차량 기준 heading error [rad] (road_yaw - vehicle_yaw, -π~π)
                            > 0: 도로가 왼쪽 → 좌회전 필요
                            < 0: 도로가 오른쪽 → 우회전 필요
            - frenet_s: Frenet 호장 길이 [m]
            - segment_length: 현재 세그먼트 전체 길이 [m]
        """
        info = self.get_vehicle_road_info(vehicle_position)

        if not info:
            return None, None, None, None, True, None, None, None

        return (
            info['closest_segment'],
            info['road_center_point'],
            info['d'],
            info['recommended_speed'],
            not info['is_on_road'],
            info['heading_error'],
            info['s'],
            info['segment_length']
        )

    def get_road_wdith(self, segment_id: str) -> Optional[float]:
        """도로 폭 반환"""
        segment = self.network.get_segment(segment_id)
        if segment:
            return segment.width
        return None

    # === 네트워크 정보 ===

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
            'waypoint_count': len(segment.waypoints),
            'bounds': segment.get_bounding_box()
        }

    # === 기존 API 호환 ===

    def connect(self, point1: Tuple[float, float, float] = (0, 0, 0),
               point2: Tuple[float, float, float] = (100, 100, 0),
               obstacles: List = [], mode: str = 'plan') -> bool:
        """기존 connect API 호환"""
        return self.add_point_to_point_road(point1, point2, obstacles=obstacles, planning_mode=mode)

    def draw(self, screen, world_to_screen_func, debug=False):
        """네트워크 시각화"""
        self.network.draw(screen, world_to_screen_func, debug)

    def reset(self):
        """네트워크 초기화"""
        self.network.clear()

    # === 네트워크 접근 ===

    def get_network(self) -> RoadNetwork:
        """내부 네트워크 객체 반환"""
        return self.network


# =============================================================================
# 9. 편의 함수들 (기존 API 호환성)
# =============================================================================

def create_road_system_from_config(config_file: str) -> RoadSystemAPI:
    """설정 파일로부터 도로 시스템 생성"""
    return RoadSystemAPI().load_config_from_yaml(config_file)
