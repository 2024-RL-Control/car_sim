# -*- coding: utf-8 -*-
import pygame
import numpy as np
from math import radians, degrees, pi, cos, sin
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

# ======================
# Bounding Management
# ======================
@dataclass
class BoundingCircle:
    """
    객체의 경계를 나타내는 원
    객체 중심 기준 상대 좌표와 반지름으로 정의
    """
    rel_x: float = 0.0  # 객체 중심으로부터의 상대 X 좌표
    rel_y: float = 0.0  # 객체 중심으로부터의 상대 Y 좌표
    radius: float = 0.0      # 반지름

    def get_world_position(self, obj_x: float, obj_y: float, obj_yaw: float) -> Tuple[float, float]:
        """객체 위치와 회전에 따른 월드 좌표 계산"""
        # 회전 변환 적용
        rotated_x = self.rel_x * cos(obj_yaw) - self.rel_y * sin(obj_yaw)
        rotated_y = self.rel_x * sin(obj_yaw) + self.rel_y * cos(obj_yaw)

        # 객체 위치에 더하기
        world_x = obj_x + rotated_x
        world_y = obj_y + rotated_y

        return (world_x, world_y)

# ======================
# Base Obstacle
# ======================
class BaseObstacle:
    """모든 장애물의 기본 클래스"""
    def __init__(self, obstacle_id: int = None, x: float = 0.0, y: float = 0.0, yaw: float = 0.0,
                 yaw_rate: float = 0.0, vel: float = 0.0, obs_type: str = "base",
                 color: Tuple[int, int, int] = (255, 0, 0),
                 bounding_circle_colors: List[Tuple[int, int, int]] = None):
        self.id = obstacle_id if obstacle_id is not None else id(self)  # 각 장애물의 고유 ID
        self.x = x                  # 중심 X 좌표
        self.y = y                  # 중심 Y 좌표
        self.yaw = yaw              # 방향 각도 [rad]
        self.yaw_rate = yaw_rate    # 각속도 [rad/s]
        self.vel = vel              # 이동 속도 [m/s]
        self.type = obs_type        # 장애물 유형
        self.color = color          # 색상 (RGB)

        # 경계 원 색상 설정
        self.bounding_circle_colors = bounding_circle_colors or [(255, 50, 50), (50, 255, 50), (50, 50, 255)]

        # 경계 원 리스트
        self.outer_circles: List[BoundingCircle] = []  # 외접원 리스트
        self.middle_circles: List[BoundingCircle] = []  # 중간원 리스트
        self.inner_circles: List[BoundingCircle] = []  # 내접원 리스트

    def set(self, x: float, y: float, yaw: float = 0.0, yaw_rate: float = 0.0, vel: float = 0.0):
        """객체 위치 및 상태 설정"""
        self.x = x
        self.y = y
        self.yaw = yaw
        self.yaw_rate = yaw_rate
        self.vel = vel

    def _normalize_angle(self, angle):
        """-pi ~ pi 범위로 정규화 (numpy 지원)"""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def update(self, dt: float):
        """객체 상태 업데이트: 이동 및 회전"""
        # 회전 업데이트
        if abs(self.yaw_rate) > 0.001:
            self.yaw += self.yaw_rate * dt
            self.yaw = self._normalize_angle(self.yaw)

        # 이동 업데이트 (현재 방향으로)
        if abs(self.vel) > 0.001:  # 속도가 있는 경우만
            self.x += self.vel * cos(self.yaw) * dt
            self.y += self.vel * sin(self.yaw) * dt

    def draw(self, screen, world_to_screen_func, debug: bool = False):
        """객체 렌더링 (구현 클래스에서 오버라이드)"""
        pass

    def _draw_bounding_circles(self, screen, world_to_screen_func):
        """디버그용: 경계 원 시각화"""
        # 외접원 (색상: bounding_circle_colors[0])
        for circle in self.outer_circles:
            world_pos = circle.get_world_position(self.x, self.y, self.yaw)
            circle_pos = world_to_screen_func((world_pos[0], world_pos[1]))

            # 반지름을 화면 스케일로 변환
            x2 = world_to_screen_func((world_pos[0] + circle.radius, world_pos[1]))
            screen_radius = max(1, int(np.linalg.norm(np.array(x2) - np.array(circle_pos))))

            pygame.draw.circle(screen, self.bounding_circle_colors[0], circle_pos, screen_radius, 1)

        # 중간원 (색상: bounding_circle_colors[1])
        for circle in self.middle_circles:
            world_pos = circle.get_world_position(self.x, self.y, self.yaw)
            circle_pos = world_to_screen_func((world_pos[0], world_pos[1]))

            # 반지름을 화면 스케일로 변환
            x2 = world_to_screen_func((world_pos[0] + circle.radius, world_pos[1]))
            screen_radius = max(1, int(np.linalg.norm(np.array(x2) - np.array(circle_pos))))

            pygame.draw.circle(screen, self.bounding_circle_colors[1], circle_pos, screen_radius, 1)

        # 내접원 (색상: bounding_circle_colors[2])
        for circle in self.inner_circles:
            world_pos = circle.get_world_position(self.x, self.y, self.yaw)
            circle_pos = world_to_screen_func((world_pos[0], world_pos[1]))

            # 반지름을 화면 스케일로 변환
            x2 = world_to_screen_func((world_pos[0] + circle.radius, world_pos[1]))
            screen_radius = max(1, int(np.linalg.norm(np.array(x2) - np.array(circle_pos))))

            pygame.draw.circle(screen, self.bounding_circle_colors[2], circle_pos, screen_radius, 1)

    def get_outer_circles_world(self) -> List[Tuple[float, float, float]]:
        """외접원들의 월드 좌표와 반지름 반환 [(x, y, radius), ...]"""
        circles = []
        for circle in self.outer_circles:
            world_pos = circle.get_world_position(self.x, self.y, self.yaw)
            circles.append((world_pos[0], world_pos[1], circle.radius))
        return circles

    def get_middle_circles_world(self) -> List[Tuple[float, float, float]]:
        """중간원들의 월드 좌표와 반지름 반환 [(x, y, radius), ...]"""
        circles = []
        for circle in self.middle_circles:
            world_pos = circle.get_world_position(self.x, self.y, self.yaw)
            circles.append((world_pos[0], world_pos[1], circle.radius))
        return circles

    def get_inner_circles_world(self) -> List[Tuple[float, float, float]]:
        """내접원들의 월드 좌표와 반지름 반환 [(x, y, radius), ...]"""
        circles = []
        for circle in self.inner_circles:
            world_pos = circle.get_world_position(self.x, self.y, self.yaw)
            circles.append((world_pos[0], world_pos[1], circle.radius))
        return circles

    def get_id(self) -> int:
        """장애물 ID 반환"""
        return self.id

# ======================
# Circle Obstacle
# ======================
class CircleObstacle(BaseObstacle):
    """원형 장애물 클래스"""
    def __init__(self, obstacle_id: int = None, x: float = 0.0, y: float = 0.0, yaw: float = 0.0,
                 yaw_rate: float = 0.0, vel: float = 0.0, obs_type: str = "circle",
                 radius: float = 1.0, color: Tuple[int, int, int] = (255, 0, 0),
                 bounding_circle_colors: List[Tuple[int, int, int]] = None):
        super().__init__(obstacle_id, x, y, yaw, yaw_rate, vel, obs_type, color, bounding_circle_colors)
        self.radius = radius

        # 원형 객체는 외접원, 중간원, 내접원이 모두 동일
        self._init_bounding_circles()

    def _init_bounding_circles(self):
        """경계 원 초기화 (원형은 모두 동일)"""
        circle = BoundingCircle(0, 0, self.radius)
        self.outer_circles = [circle]
        self.middle_circles = [circle]
        self.inner_circles = [circle]

    def draw(self, screen, world_to_screen_func, debug: bool = False):
        """원형 객체 렌더링"""
        # 객체 중심 위치를 화면 좌표로 변환
        screen_pos = world_to_screen_func((self.x, self.y))

        # 반지름을 화면 스케일로 변환
        x2 = world_to_screen_func((self.x + self.radius, self.y))
        screen_radius = max(1, int(np.linalg.norm(np.array(x2) - np.array(screen_pos))))

        # 원형 객체 그리기
        pygame.draw.circle(screen, self.color, screen_pos, screen_radius, 2)

        # 방향 표시 (작은 선으로)
        dir_end_x = screen_pos[0] + screen_radius * cos(self.yaw)
        dir_end_y = screen_pos[1] - screen_radius * sin(self.yaw)  # 화면 좌표계는 y축이 반전됨
        pygame.draw.line(screen, self.color, screen_pos, (dir_end_x, dir_end_y), 2)

        # 디버그 모드: 경계 원 표시
        if debug:
            self._draw_bounding_circles(screen, world_to_screen_func)

# ======================
# Square Obstacle
# ======================
class SquareObstacle(BaseObstacle):
    """정사각형 장애물 클래스"""
    def __init__(self, obstacle_id: int = None, x: float = 0.0, y: float = 0.0, yaw: float = 0.0,
                 yaw_rate: float = 0.0, vel: float = 0.0, obs_type: str = "square",
                 length: float = 1.0, color: Tuple[int, int, int] = (255, 0, 0),
                 bounding_circle_colors: List[Tuple[int, int, int]] = None):
        super().__init__(obstacle_id, x, y, yaw, yaw_rate, vel, obs_type, color, bounding_circle_colors)
        self.length = length  # 한 변의 길이

        # 경계 원 초기화
        self._init_bounding_circles()

    def _init_bounding_circles(self):
        """정사각형에 대한 경계 원 초기화"""
        half_length = self.length / 2

        # 외접원 (정사각형 꼭지점까지의 거리)
        outer_radius = np.sqrt(2 * half_length**2)
        self.outer_circles = [BoundingCircle(0, 0, outer_radius)]

        # 중간원 (꼭지점과 변의 중점 사이의 거리)
        middle_radius = half_length * np.sqrt(1.25)  # 근사값
        self.middle_circles = [BoundingCircle(0, 0, middle_radius)]

        # 내접원 (정사각형 중심에서 변까지의 거리)
        inner_radius = half_length
        self.inner_circles = [BoundingCircle(0, 0, inner_radius)]

    def draw(self, screen, world_to_screen_func, debug: bool = False):
        """정사각형 객체 렌더링"""
        # 꼭지점 계산 (객체 기준 좌표)
        half_length = self.length / 2
        corners = [
            (-half_length, -half_length),
            (half_length, -half_length),
            (half_length, half_length),
            (-half_length, half_length)
        ]

        # 꼭지점 변환 (월드 좌표 → 화면 좌표)
        world_corners = []
        for dx, dy in corners:
            # 회전 변환
            rotated_x = dx * cos(self.yaw) - dy * sin(self.yaw)
            rotated_y = dx * sin(self.yaw) + dy * cos(self.yaw)

            # 월드 좌표
            world_x = self.x + rotated_x
            world_y = self.y + rotated_y

            # 화면 좌표
            world_corners.append((world_x, world_y))

        # 정사각형 그리기
        screen_corners = world_to_screen_func(world_corners)
        pygame.draw.polygon(screen, self.color, screen_corners, 2)

        # 디버그 모드: 경계 원 표시
        if debug:
            self._draw_bounding_circles(screen, world_to_screen_func)

# ======================
# Rectangle Obstacle
# ======================
class RectangleObstacle(BaseObstacle):
    """직사각형 장애물 클래스"""
    def __init__(self, obstacle_id: int = None, x: float = 0.0, y: float = 0.0, yaw: float = 0.0,
                 yaw_rate: float = 0.0, vel: float = 0.0, obs_type: str = "rectangle",
                 width: float = 2.0, height: float = 1.0, color: Tuple[int, int, int] = (255, 0, 0),
                 bounding_circle_colors: List[Tuple[int, int, int]] = None):
        super().__init__(obstacle_id, x, y, yaw, yaw_rate, vel, obs_type, color, bounding_circle_colors)
        self.width = width    # 너비 (X축 방향)
        self.height = height  # 높이 (Y축 방향)

        # 분할 계산 및 경계 원 초기화
        self._init_bounding_circles()

    def _init_bounding_circles(self):
        """직사각형을 분할하여 경계 원 초기화"""
        # 직사각형 분할 (작은 변의 길이를 기준으로)
        min_side = min(self.width, self.height)
        n_w = self.width / min_side
        n_h = self.height / min_side
        num_squares_x = int(np.ceil(n_w))
        num_squares_y = int(np.ceil(n_h))

        # 정사각형 크기
        square_size = min_side

        # 중첩 비율 계산
        rx = 0 if num_squares_x <= 1 else (num_squares_x * square_size - self.width) / ((num_squares_x - 1) * square_size)
        ry = 0 if num_squares_y <= 1 else (num_squares_y * square_size - self.height) / ((num_squares_y - 1) * square_size)
        overlap_ratio = min(0.5, max(rx, ry))

        # 중첩을 고려한 실효 간격 계산
        effective_step = square_size * (1 - overlap_ratio)

        # 중첩을 고려하여 각 방향의 총 길이 계산
        total_width = effective_step * (num_squares_x - 1) + square_size
        total_height = effective_step * (num_squares_y - 1) + square_size

        # 초기 위치 계산
        start_x = -self.width / 2 + (self.width - total_width) / 2 + square_size / 2
        start_y = -self.height / 2 + (self.height - total_height) / 2 + square_size / 2

        # 각 정사각형 위치에 대해 경계 원 생성
        for i in range(num_squares_x):
            for j in range(num_squares_y):
                # 정사각형 중심 좌표 (객체 기준)
                sq_x = start_x + i * effective_step
                sq_y = start_y + j * effective_step

                # 분할 정사각형이 직사각형 내부에 있는지 확인
                if abs(sq_x) > self.width/2 - 0.01 or abs(sq_y) > self.height/2 - 0.01:
                    continue

                # 외접원 (정사각형 꼭지점까지의 거리)
                outer_radius = np.sqrt(2) * square_size / 2
                self.outer_circles.append(BoundingCircle(sq_x, sq_y, outer_radius))

                # 중간원 (꼭지점과 변의 중점 사이의 거리)
                middle_radius = square_size / 2 * np.sqrt(1.25)  # 근사값
                self.middle_circles.append(BoundingCircle(sq_x, sq_y, middle_radius))

                # 내접원 (정사각형 중심에서 변까지의 거리)
                inner_radius = square_size / 2
                self.inner_circles.append(BoundingCircle(sq_x, sq_y, inner_radius))

    def draw(self, screen, world_to_screen_func, debug: bool = False):
        """직사각형 객체 렌더링"""
        # 꼭지점 계산 (객체 기준 좌표)
        half_width = self.width / 2
        half_height = self.height / 2
        corners = [
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height)
        ]

        # 꼭지점 변환 (월드 좌표 → 화면 좌표)
        world_corners = []
        for dx, dy in corners:
            # 회전 변환
            rotated_x = dx * cos(self.yaw) - dy * sin(self.yaw)
            rotated_y = dx * sin(self.yaw) + dy * cos(self.yaw)

            # 월드 좌표
            world_x = self.x + rotated_x
            world_y = self.y + rotated_y

            # 화면 좌표
            world_corners.append((world_x, world_y))

        # 직사각형 그리기
        screen_corners = world_to_screen_func(world_corners)
        pygame.draw.polygon(screen, self.color, screen_corners, 2)

        # 디버그 모드: 경계 원 표시
        if debug:
            self._draw_bounding_circles(screen, world_to_screen_func)

# ======================
# 목적지 클래스
# ======================
class Goal(BaseObstacle):
    """목적지 클래스"""
    def __init__(self, goal_id: int = None, x: float = 0.0, y: float = 0.0, yaw: float = 0.0,
                 radius: float = 1.0, color: Tuple[int, int, int] = (0, 255, 0),
                 bounding_circle_colors: List[Tuple[int, int, int]] = None):
        # 기본적으로 정적 목적지
        super().__init__(goal_id, x, y, yaw, 0, 0, "goal", color, bounding_circle_colors)
        self.radius = radius

        # 목적지는 원형으로 표시
        self._init_bounding_circles()

    def _init_bounding_circles(self):
        """목적지 경계 원 초기화"""
        # 목적지는 단일 원형
        circle = BoundingCircle(0, 0, self.radius)
        self.outer_circles = [circle]
        self.middle_circles = [circle]
        self.inner_circles = [circle]

    def draw(self, screen, world_to_screen_func, is_current: bool = False, debug: bool = False):
        """목적지 렌더링"""
        # 객체 중심 위치를 화면 좌표로 변환
        screen_pos = world_to_screen_func((self.x, self.y))

        # 반지름을 화면 스케일로 변환
        x2 = world_to_screen_func((self.x + self.radius, self.y))
        screen_radius = max(1, int(np.linalg.norm(np.array(x2) - np.array(screen_pos))))

        original_color = self.color
        if not is_current:
            # 반투명(밝기 70% + 흰 30%)
            draw_color = tuple(min(255, int(c * 0.7 + 255 * 0.3)) for c in original_color)
        else:
            draw_color = original_color

        if is_current:
            # 실선 원
            pygame.draw.circle(screen, draw_color, screen_pos, screen_radius, 2)
        else:
            # 목적지 원 그리기 - 점선 효과를 위한 대시 패턴
            num_segments = 36
            for i in range(num_segments):
                if i % 2 == 0:  # 짝수 세그먼트만 그림
                    angle_start = i * 2 * pi / num_segments
                    angle_end = (i + 1) * 2 * pi / num_segments

                    # 시작점
                    x_start = screen_pos[0] + screen_radius * cos(angle_start)
                    y_start = screen_pos[1] - screen_radius * sin(angle_start)

                    # 끝점
                    x_end = screen_pos[0] + screen_radius * cos(angle_end)
                    y_end = screen_pos[1] - screen_radius * sin(angle_end)

                    pygame.draw.line(screen, self.color, (int(x_start), int(y_start)),
                                    (int(x_end), int(y_end)), 2)


        # 방향 표시 (화살표)
        arrow_length = screen_radius * 0.8
        arrow_end_x = screen_pos[0] + arrow_length * cos(self.yaw)
        arrow_end_y = screen_pos[1] - arrow_length * sin(self.yaw)

        # 화살표 몸통
        pygame.draw.line(screen, self.color, screen_pos, (int(arrow_end_x), int(arrow_end_y)), 2)

        # 화살표 머리
        head_size = screen_radius * 0.2
        angle = self.yaw
        arrow_head1_x = arrow_end_x - head_size * cos(angle - radians(25))
        arrow_head1_y = arrow_end_y + head_size * sin(angle - radians(25))
        arrow_head2_x = arrow_end_x - head_size * cos(angle + radians(25))
        arrow_head2_y = arrow_end_y + head_size * sin(angle + radians(25))

        pygame.draw.line(screen, self.color, (int(arrow_end_x), int(arrow_end_y)),
                        (int(arrow_head1_x), int(arrow_head1_y)), 2)
        pygame.draw.line(screen, self.color, (int(arrow_end_x), int(arrow_end_y)),
                        (int(arrow_head2_x), int(arrow_head2_y)), 2)

        # 디버그 모드: 추가 시각화
        if debug:
            pass
            # self._draw_bounding_circles(screen, world_to_screen_func)

# ======================
# Obstacle Manager
# ======================
class ObstacleManager:
    """장애물 관리 클래스"""
    def __init__(self, bounding_circle_colors: List[Tuple[int, int, int]] = None):
        self.obstacles: Dict[int, BaseObstacle] = {}  # {obstacle_id: BaseObstacle 객체}
        self.next_obstacle_id = 0  # 자동 ID 생성용
        self.bounding_circle_colors = bounding_circle_colors or [(255, 50, 50), (50, 255, 50), (50, 50, 255)]

    def _add_obstacle(self, obstacle: BaseObstacle):
        """
        장애물 추가

        Args:
            obstacle: 추가할 장애물 객체

        Returns:
            obstacle_id: 추가된 장애물의 ID
        """
        obstacle_id = obstacle.get_id()
        self.obstacles[obstacle_id] = obstacle

        # 다음 자동 ID 업데이트
        if isinstance(obstacle_id, int) and self.next_obstacle_id <= obstacle_id:
            self.next_obstacle_id = obstacle_id + 1

        return obstacle_id

    def remove_obstacle(self, obstacle_id: int):
        """
        ID로 장애물 제거

        Args:
            obstacle_id: 제거할 장애물 ID

        Returns:
            성공 여부 (Boolean)
        """
        if obstacle_id in self.obstacles:
            del self.obstacles[obstacle_id]
            return True
        return False

    def clear_obstacles(self):
        """모든 장애물 제거"""
        self.obstacles.clear()
        self.next_obstacle_id = 0

    def update(self, dt: float):
        """모든 장애물 상태 업데이트 후 외접원 반환"""
        for obstacle in self.obstacles.values():
            obstacle.update(dt)

        return self.get_all_outer_circles()

    def draw(self, screen, world_to_screen_func, debug: bool = False):
        """모든 장애물 렌더링"""
        for obstacle in self.obstacles.values():
            obstacle.draw(screen, world_to_screen_func, debug)

    def get_all_outer_circles(self) -> List[Tuple[float, float, float]]:
        """모든 장애물의 외접원 반환 [(x, y, radius), ...]"""
        circles = []
        for obstacle in self.obstacles.values():
            circles.extend(obstacle.get_outer_circles_world())
        return circles

    def get_all_middle_circles(self) -> List[Tuple[float, float, float]]:
        """모든 장애물의 중간원 반환 [(x, y, radius), ...]"""
        circles = []
        for obstacle in self.obstacles.values():
            circles.extend(obstacle.get_middle_circles_world())
        return circles

    def get_all_inner_circles(self) -> List[Tuple[float, float, float]]:
        """모든 장애물의 내접원 반환 [(x, y, radius), ...]"""
        circles = []
        for obstacle in self.obstacles.values():
            circles.extend(obstacle.get_inner_circles_world())
        return circles

    def get_obstacle_count(self) -> int:
        """장애물 수 반환"""
        return len(self.obstacles)

    def get_obstacle(self, obstacle_id: int) -> Optional[BaseObstacle]:
        """
        ID로 장애물 객체 반환

        Args:
            obstacle_id: 장애물 ID

        Returns:
            BaseObstacle 객체 또는 None
        """
        return self.obstacles.get(obstacle_id)

    def get_obstacles(self) -> List[BaseObstacle]:
        """
        모든 장애물 객체 목록 반환

        Returns:
            장애물 객체 리스트
        """
        return list(self.obstacles.values())

    def add_circle_obstacle(self, obstacle_id: int = None, x: float = 0.0, y: float = 0.0,
                            yaw: float = 0.0, yaw_rate: float = 0.0, vel: float = 0.0,
                            radius: float = 1.0, color: Tuple[int, int, int] = (255, 0, 0)):
        """
        원형 장애물 추가

        Args:
            obstacle_id: 장애물 ID (None이면 자동 생성)
            x: 장애물 중심 X 좌표
            y: 장애물 중심 Y 좌표
            yaw: 초기 방향각 [rad]
            yaw_rate: 각속도 [rad/s]
            vel: 이동 속도 [m/s]
            radius: 원 반지름
            color: RGB 색상

        Returns:
            장애물 ID
        """
        if obstacle_id is None:
            obstacle_id = self.next_obstacle_id
            self.next_obstacle_id += 1

        obstacle = CircleObstacle(obstacle_id, x, y, yaw, yaw_rate, vel, "circle", radius, color, self.bounding_circle_colors)
        return self._add_obstacle(obstacle)

    def add_square_obstacle(self, obstacle_id: int = None, x: float = 0.0, y: float = 0.0,
                            yaw: float = 0.0, yaw_rate: float = 0.0, vel: float = 0.0,
                            length: float = 1.0, color: Tuple[int, int, int] = (255, 0, 0)):
        """
        정사각형 장애물 추가

        Args:
            obstacle_id: 장애물 ID (None이면 자동 생성)
            x: 장애물 중심 X 좌표
            y: 장애물 중심 Y 좌표
            yaw: 초기 방향각 [rad]
            yaw_rate: 각속도 [rad/s]
            vel: 이동 속도 [m/s]
            length: 한 변의 길이
            color: RGB 색상

        Returns:
            장애물 ID
        """
        if obstacle_id is None:
            obstacle_id = self.next_obstacle_id
            self.next_obstacle_id += 1

        obstacle = SquareObstacle(obstacle_id, x, y, yaw, yaw_rate, vel, "square", length, color, self.bounding_circle_colors)
        return self._add_obstacle(obstacle)

    def add_rectangle_obstacle(self, obstacle_id: int = None, x: float = 0.0, y: float = 0.0,
                                yaw: float = 0.0, yaw_rate: float = 0.0, vel: float = 0.0,
                                width: float = 2.0, height: float = 1.0, color: Tuple[int, int, int] = (255, 0, 0)):
        """
        직사각형 장애물 추가

        Args:
            obstacle_id: 장애물 ID (None이면 자동 생성)
            x: 장애물 중심 X 좌표
            y: 장애물 중심 Y 좌표
            yaw: 초기 방향각 [rad]
            yaw_rate: 각속도 [rad/s]
            vel: 이동 속도 [m/s]
            width: 너비
            height: 높이
            color: RGB 색상

        Returns:
            장애물 ID
        """
        if obstacle_id is None:
            obstacle_id = self.next_obstacle_id
            self.next_obstacle_id += 1

        obstacle = RectangleObstacle(obstacle_id, x, y, yaw, yaw_rate, vel, "rectangle", width, height, color, self.bounding_circle_colors)
        return self._add_obstacle(obstacle)

    def get_serializable_obstacles(self):
        """
        장애물 정보를 직렬화 가능한 형태로 반환 (저장용)

        Returns:
            직렬화 가능한 장애물 정보 딕셔너리
        """
        serialized_data = {
            'obstacles': {},
            'next_obstacle_id': self.next_obstacle_id
        }

        for obstacle_id, obstacle in self.obstacles.items():
            obstacle_data = {
                'id': obstacle_id,
                'type': obstacle.type,
                'x': obstacle.x,
                'y': obstacle.y,
                'yaw': obstacle.yaw,
                'yaw_rate': obstacle.yaw_rate,
                'vel': obstacle.vel,
                'color': obstacle.color,
                'bounding_circle_colors': obstacle.bounding_circle_colors
            }

            # 장애물 유형별 추가 속성
            if obstacle.type == "circle":
                obstacle_data['radius'] = obstacle.radius
            elif obstacle.type == "square":
                obstacle_data['length'] = obstacle.length
            elif obstacle.type == "rectangle":
                obstacle_data['width'] = obstacle.width
                obstacle_data['height'] = obstacle.height

            serialized_data['obstacles'][str(obstacle_id)] = obstacle_data

        return serialized_data

    def load_obstacles_from_serialized(self, serialized_data):
        """
        직렬화된 장애물 정보로부터 장애물 복원 (로드용)

        Args:
            serialized_data: 직렬화된 장애물 정보 딕셔너리
        """
        # 기존 장애물 제거
        self.clear_obstacles()

        # 이전 형식과의 호환성 유지
        if isinstance(serialized_data, list):
            # 이전 형식: 리스트 형태의 장애물 데이터
            for obstacle_data in serialized_data:
                obs_type = obstacle_data.get('type', 'base')
                x = obstacle_data.get('x', 0.0)
                y = obstacle_data.get('y', 0.0)
                yaw = obstacle_data.get('yaw', 0.0)
                yaw_rate = obstacle_data.get('yaw_rate', 0.0)
                vel = obstacle_data.get('vel', 0.0)
                color = obstacle_data.get('color', (255, 0, 0))

                if obs_type == "circle":
                    radius = obstacle_data.get('radius', 1.0)
                    self.add_circle_obstacle(None, x, y, yaw, yaw_rate, vel, radius, color)
                elif obs_type == "square":
                    length = obstacle_data.get('length', 1.0)
                    self.add_square_obstacle(None, x, y, yaw, yaw_rate, vel, length, color)
                elif obs_type == "rectangle":
                    width = obstacle_data.get('width', 2.0)
                    height = obstacle_data.get('height', 1.0)
                    self.add_rectangle_obstacle(None, x, y, yaw, yaw_rate, vel, width, height, color)
            return

        # 새 형식: 딕셔너리 형태의 구조화된 장애물 데이터
        obstacles_data = serialized_data.get('obstacles', {})
        for obstacle_id_str, obstacle_data in obstacles_data.items():
            # 문자열 ID를 정수로 변환
            obstacle_id = int(obstacle_id_str) if isinstance(obstacle_id_str, str) else obstacle_id_str

            obs_type = obstacle_data.get('type', 'base')
            x = obstacle_data.get('x', 0.0)
            y = obstacle_data.get('y', 0.0)
            yaw = obstacle_data.get('yaw', 0.0)
            yaw_rate = obstacle_data.get('yaw_rate', 0.0)
            vel = obstacle_data.get('vel', 0.0)
            color = obstacle_data.get('color', (255, 0, 0))

            if obs_type == "circle":
                radius = obstacle_data.get('radius', 1.0)
                self.add_circle_obstacle(obstacle_id, x, y, yaw, yaw_rate, vel, radius, color)
            elif obs_type == "square":
                length = obstacle_data.get('length', 1.0)
                self.add_square_obstacle(obstacle_id, x, y, yaw, yaw_rate, vel, length, color)
            elif obs_type == "rectangle":
                width = obstacle_data.get('width', 2.0)
                height = obstacle_data.get('height', 1.0)
                self.add_rectangle_obstacle(obstacle_id, x, y, yaw, yaw_rate, vel, width, height, color)

        # 다음 ID 설정
        if 'next_obstacle_id' in serialized_data:
            self.next_obstacle_id = serialized_data.get('next_obstacle_id')

        # 현재 설정된 bounding_circle_colors를 유지하기 위해 모든 장애물에 적용
        for obstacle in self.obstacles.values():
            obstacle.bounding_circle_colors = self.bounding_circle_colors

# ======================
# Goal Manager
# ======================
class GoalManager:
    """개별 차량의 순차적 목적지를 관리하는 클래스"""
    def __init__(self, bounding_circle_colors: List[Tuple[int, int, int]] = None):
        self.goals = []  # 순차적 목적지 목록 [Goal, Goal, ...]
        self.current_goal_index = -1  # 현재 목표의 인덱스 (-1은 없음)
        self.next_goal_id = 0  # 목적지 ID 생성용
        self.bounding_circle_colors = bounding_circle_colors or [(255, 50, 50), (50, 255, 50), (50, 50, 255)]

    def add_goal(self, x, y, yaw=0.0, radius=1.0, color=(0, 255, 0)):
        """
        목적지 추가 (순차적 목표 리스트에 추가)

        Args:
            x: 목적지 X 좌표
            y: 목적지 Y 좌표
            yaw: 방향각 [rad]
            radius: 목적지 반경
            color: 색상 (RGB)

        Returns:
            goal_id: 생성된 목적지 ID
        """
        goal_id = self.next_goal_id
        self.next_goal_id += 1

        goal = Goal(goal_id, x, y, yaw, radius, color, self.bounding_circle_colors)
        self.goals.append(goal)

        # 첫 번째 목표를 추가하는 경우, 현재 목표로 설정
        if len(self.goals) == 1:
            self.current_goal_index = 0

        return goal_id

    def remove_goal(self, index=None):
        """
        목적지 제거

        Args:
            index: 제거할 목적지 인덱스 (None이면 현재 목적지)

        Returns:
            성공 여부 (Boolean)
        """
        if len(self.goals) == 0:
            return False

        if index is None:
            index = self.current_goal_index

        if 0 <= index < len(self.goals):
            self.goals.pop(index)

            # 현재 목표가 제거된 경우 조정
            if self.current_goal_index >= len(self.goals):
                self.current_goal_index = len(self.goals) - 1 if len(self.goals) > 0 else -1

            return True
        return False

    def get_current_goal(self):
        """
        현재 목적지 객체 반환

        Returns:
            Goal 객체 또는 None
        """
        if 0 <= self.current_goal_index < len(self.goals):
            return self.goals[self.current_goal_index]
        return None

    def has_goals(self):
        """
        목적지가 있는지 확인

        Returns:
            목적지 존재 여부 (Boolean)
        """
        return len(self.goals) > 0

    def next_goal(self):
        """
        다음 목적지로 전환

        Returns:
            성공 여부 (Boolean)
        """
        if self.current_goal_index < len(self.goals) - 1:
            self.current_goal_index += 1
            return True
        return False

    def get_goal_count(self):
        """
        목적지 개수 반환

        Returns:
            목적지 개수
        """
        return len(self.goals)

    def get_remaining_goals(self):
        """
        남은 목적지 개수 반환 (현재 포함)

        Returns:
            남은 목적지 개수
        """
        if self.current_goal_index < 0:
            return 0
        return len(self.goals) - self.current_goal_index

    def draw(self, screen, world_to_screen_func, debug=False):
        """
        모든 목적지 렌더링 (현재 목적지는 실선, 나머지는 점선)

        Args:
            screen: Pygame 화면 객체
            world_to_screen_func: 좌표 변환 함수
            debug: 디버그 정보 표시 여부
        """
        for i, goal in enumerate(self.goals):
            # 현재 목적지는 기본 색상, 나머지는 반투명하게 표시
            is_current = (i == self.current_goal_index)

            # 목적지 그리기
            goal.draw(screen, world_to_screen_func, is_current, debug)

    def clear_goals(self):
        """모든 목적지 제거"""
        self.goals = []
        self.current_goal_index = -1
        self.next_goal_id = 0

    def get_serializable_goals(self):
        """
        목적지 정보를 직렬화 가능한 형태로 반환 (저장용)

        Returns:
            직렬화 가능한 목적지 정보 딕셔너리
        """
        goals_data = []
        for goal in self.goals:
            goals_data.append({
                'id': goal.id,
                'x': goal.x,
                'y': goal.y,
                'yaw': goal.yaw,
                'radius': goal.radius,
                'color': goal.color,
            })

        return {
            'goals': goals_data,
            'current_goal_index': self.current_goal_index,
            'next_goal_id': self.next_goal_id
        }

    def load_from_serialized(self, serialized_data):
        """
        직렬화된 목적지 정보로부터 목적지 복원 (로드용)

        Args:
            serialized_data: 직렬화된 목적지 정보 딕셔너리
        """
        # 기존 목적지 초기화
        self.clear_goals()

        # 목적지 데이터 복원
        goals_data = serialized_data.get('goals', [])
        for goal_data in goals_data:
            goal_id = goal_data.get('id', self.next_goal_id)
            self.next_goal_id = max(self.next_goal_id, goal_id + 1)

            goal = Goal(
                goal_id,
                x=goal_data.get('x', 0),
                y=goal_data.get('y', 0),
                yaw=goal_data.get('yaw', 0),
                radius=goal_data.get('radius', 1.0),
                color=goal_data.get('color', (0, 255, 0)),
                bounding_circle_colors=self.bounding_circle_colors
            )
            self.goals.append(goal)

        # 현재 목적지 인덱스 설정
        self.current_goal_index = serialized_data.get('current_goal_index', -1)
        if self.current_goal_index >= len(self.goals):
            self.current_goal_index = len(self.goals) - 1 if len(self.goals) > 0 else -1

        # 다음 목적지 ID 설정
        if 'next_goal_id' in serialized_data:
            self.next_goal_id = serialized_data.get('next_goal_id')
