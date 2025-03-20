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
    def __init__(self, x: float, y: float, yaw: float = 0.0, yaw_rate: float = 0.0, vel: float = 0.0,
                 obs_type: str = "base", color: Tuple[int, int, int] = (255, 0, 0)):
        self.x = x                  # 중심 X 좌표
        self.y = y                  # 중심 Y 좌표
        self.yaw = yaw              # 방향 각도 [rad]
        self.yaw_rate = yaw_rate    # 각속도 [rad/s]
        self.vel = vel              # 이동 속도 [m/s]
        self.type = obs_type        # 장애물 유형
        self.color = color          # 색상 (RGB)

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

    def update(self, dt: float):
        """객체 상태 업데이트: 이동 및 회전"""
        # 회전 업데이트
        self.yaw += self.yaw_rate * dt
        self.yaw = (self.yaw + pi) % (2 * pi) - pi  # [-π, π] 범위로 정규화

        # 이동 업데이트 (현재 방향으로)
        if abs(self.vel) > 0.001:  # 속도가 있는 경우만
            self.x += self.vel * cos(self.yaw) * dt
            self.y += self.vel * sin(self.yaw) * dt

    def draw(self, screen, world_to_screen_func, debug: bool = False):
        """객체 렌더링 (구현 클래스에서 오버라이드)"""
        pass

    def _draw_bounding_circles(self, screen, world_to_screen_func):
        """디버그용: 경계 원 시각화"""
        # 외접원 (빨간색)
        for circle in self.outer_circles:
            world_pos = circle.get_world_position(self.x, self.y, self.yaw)
            circle_pos = world_to_screen_func(world_pos[0], world_pos[1])

            # 반지름을 화면 스케일로 변환
            x2 = world_to_screen_func(world_pos[0] + circle.radius, world_pos[1])
            screen_radius = max(1, int(np.linalg.norm(np.array(x2) - np.array(circle_pos))))

            pygame.draw.circle(screen, (255, 50, 50), circle_pos, screen_radius, 1)

        # 중간원 (녹색)
        for circle in self.middle_circles:
            world_pos = circle.get_world_position(self.x, self.y, self.yaw)
            circle_pos = world_to_screen_func(world_pos[0], world_pos[1])

            # 반지름을 화면 스케일로 변환
            x2 = world_to_screen_func(world_pos[0] + circle.radius, world_pos[1])
            screen_radius = max(1, int(np.linalg.norm(np.array(x2) - np.array(circle_pos))))

            pygame.draw.circle(screen, (50, 255, 50), circle_pos, screen_radius, 1)

        # 내접원 (파란색)
        for circle in self.inner_circles:
            world_pos = circle.get_world_position(self.x, self.y, self.yaw)
            circle_pos = world_to_screen_func(world_pos[0], world_pos[1])

            # 반지름을 화면 스케일로 변환
            x2 = world_to_screen_func(world_pos[0] + circle.radius, world_pos[1])
            screen_radius = max(1, int(np.linalg.norm(np.array(x2) - np.array(circle_pos))))

            pygame.draw.circle(screen, (50, 50, 255), circle_pos, screen_radius, 1)

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

# ======================
# Circle Obstacle
# ======================
class CircleObstacle(BaseObstacle):
    """원형 장애물 클래스"""
    def __init__(self, x: float, y: float, yaw: float = 0.0, yaw_rate: float = 0.0, vel: float = 0.0,
                 obs_type: str = "circle",color: Tuple[int, int, int] = (255, 0, 0),
                 radius: float = 1.0):
        super().__init__(x, y, yaw, yaw_rate, vel, obs_type, color)
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
        screen_pos = world_to_screen_func(self.x, self.y)

        # 반지름을 화면 스케일로 변환
        x2 = world_to_screen_func(self.x + self.radius, self.y)
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
    def __init__(self, x: float, y: float, yaw: float = 0.0, yaw_rate: float = 0.0, vel: float = 0.0,
                 obs_type: str = "square", color: Tuple[int, int, int] = (255, 0, 0),
                 length: float = 1.0):
        super().__init__(x, y, yaw, yaw_rate, vel, obs_type, color)
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
        screen_corners = []
        for dx, dy in corners:
            # 회전 변환
            rotated_x = dx * cos(self.yaw) - dy * sin(self.yaw)
            rotated_y = dx * sin(self.yaw) + dy * cos(self.yaw)

            # 월드 좌표
            world_x = self.x + rotated_x
            world_y = self.y + rotated_y

            # 화면 좌표
            screen_corners.append(world_to_screen_func(world_x, world_y))

        # 정사각형 그리기
        pygame.draw.polygon(screen, self.color, screen_corners, 2)

        # 디버그 모드: 경계 원 표시
        if debug:
            self._draw_bounding_circles(screen, world_to_screen_func)

# ======================
# Rectangle Obstacle
# ======================
class RectangleObstacle(BaseObstacle):
    """직사각형 장애물 클래스"""
    def __init__(self, x: float, y: float, yaw: float = 0.0, yaw_rate: float = 0.0, vel: float = 0.0,
                 obs_type: str = "rectangle", color: Tuple[int, int, int] = (255, 0, 0),
                 width: float = 2.0, height: float = 1.0):
        super().__init__(x, y, yaw, yaw_rate, vel, obs_type, color)
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
        screen_corners = []
        for dx, dy in corners:
            # 회전 변환
            rotated_x = dx * cos(self.yaw) - dy * sin(self.yaw)
            rotated_y = dx * sin(self.yaw) + dy * cos(self.yaw)

            # 월드 좌표
            world_x = self.x + rotated_x
            world_y = self.y + rotated_y

            # 화면 좌표
            screen_corners.append(world_to_screen_func(world_x, world_y))

        # 직사각형 그리기
        pygame.draw.polygon(screen, self.color, screen_corners, 2)

        # 디버그 모드: 경계 원 표시
        if debug:
            self._draw_bounding_circles(screen, world_to_screen_func)

# ======================
# 목적지 클래스
# ======================
class Goal(BaseObstacle):
    """목적지 클래스"""
    def __init__(self, x: float, y: float, yaw: float = 0.0,
                 color: Tuple[int, int, int] = (0, 255, 0),
                 radius: float = 1.0):
        # 기본적으로 정적 목적지
        super().__init__(x, y, yaw, 0, 0, "goal", color)
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

    def draw(self, screen, world_to_screen_func, debug: bool = False):
        """목적지 렌더링"""
        # 객체 중심 위치를 화면 좌표로 변환
        screen_pos = world_to_screen_func(self.x, self.y)

        # 반지름을 화면 스케일로 변환
        x2 = world_to_screen_func(self.x + self.radius, self.y)
        screen_radius = max(1, int(np.linalg.norm(np.array(x2) - np.array(screen_pos))))

        # 목적지 원 그리기 - 점선 효과를 위한 대시 패턴
        points = []
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
    def __init__(self):
        self.obstacles: List[BaseObstacle] = []

    def add_obstacle(self, obstacle: BaseObstacle):
        """장애물 추가"""
        self.obstacles.append(obstacle)
        return len(self.obstacles) - 1  # 인덱스 반환

    def remove_obstacle(self, index: int):
        """인덱스로 장애물 제거"""
        if 0 <= index < len(self.obstacles):
            del self.obstacles[index]
            return True
        return False

    def clear_obstacles(self):
        """모든 장애물 제거"""
        self.obstacles = []

    def update(self, dt: float):
        """모든 장애물 상태 업데이트"""
        for obstacle in self.obstacles:
            obstacle.update(dt)

    def draw(self, screen, world_to_screen_func, debug: bool = False):
        """모든 장애물 렌더링"""
        for obstacle in self.obstacles:
            obstacle.draw(screen, world_to_screen_func, debug)

    def get_all_outer_circles(self) -> List[Tuple[float, float, float]]:
        """모든 장애물의 외접원 반환 [(x, y, radius), ...]"""
        circles = []
        for obstacle in self.obstacles:
            circles.extend(obstacle.get_outer_circles_world())
        return circles

    def get_all_middle_circles(self) -> List[Tuple[float, float, float]]:
        """모든 장애물의 중간원 반환 [(x, y, radius), ...]"""
        circles = []
        for obstacle in self.obstacles:
            circles.extend(obstacle.get_middle_circles_world())
        return circles

    def get_all_inner_circles(self) -> List[Tuple[float, float, float]]:
        """모든 장애물의 내접원 반환 [(x, y, radius), ...]"""
        circles = []
        for obstacle in self.obstacles:
            circles.extend(obstacle.get_inner_circles_world())
        return circles

    def get_obstacle_count(self) -> int:
        """장애물 수 반환"""
        return len(self.obstacles)

    def add_circle_obstacle(self, x: float, y: float, radius: float,
                            yaw: float = 0.0, yaw_rate: float = 0.0,
                            vel: float = 0.0, color: Tuple[int, int, int] = (255, 0, 0)):
        """
        원형 장애물 추가

        Args:
            x: 장애물 중심 X 좌표
            y: 장애물 중심 Y 좌표
            radius: 원 반지름
            yaw: 초기 방향각 [rad]
            yaw_rate: 각속도 [rad/s]
            vel: 이동 속도 [m/s]
            color: RGB 색상

        Returns:
            장애물 인덱스
        """
        obstacle = CircleObstacle(x, y, yaw, yaw_rate, vel, "circle", color, radius)
        return self.add_obstacle(obstacle)

    def add_square_obstacle(self, x: float, y: float, length: float,
                            yaw: float = 0.0, yaw_rate: float = 0.0,
                            vel: float = 0.0, color: Tuple[int, int, int] = (255, 0, 0)):
        """
        정사각형 장애물 추가

        Args:
            x: 장애물 중심 X 좌표
            y: 장애물 중심 Y 좌표
            length: 한 변의 길이
            yaw: 초기 방향각 [rad]
            yaw_rate: 각속도 [rad/s]
            vel: 이동 속도 [m/s]
            color: RGB 색상

        Returns:
            장애물 인덱스
        """
        obstacle = SquareObstacle(x, y, yaw, yaw_rate, vel, "square", color, length)
        return self.add_obstacle(obstacle)

    def add_rectangle_obstacle(self, x: float, y: float, width: float, height: float,
                                yaw: float = 0.0, yaw_rate: float = 0.0,
                                vel: float = 0.0, color: Tuple[int, int, int] = (255, 0, 0)):
        """
        직사각형 장애물 추가

        Args:
            x: 장애물 중심 X 좌표
            y: 장애물 중심 Y 좌표
            width: 너비
            height: 높이
            yaw: 초기 방향각 [rad]
            yaw_rate: 각속도 [rad/s]
            vel: 이동 속도 [m/s]
            color: RGB 색상

        Returns:
            장애물 인덱스
        """
        obstacle = RectangleObstacle(x, y, yaw, yaw_rate, vel, "rectangle", color, width, height)
        return self.add_obstacle(obstacle)

# ======================
# Goal Manager
# ======================
class GoalManager:
    """목적지들을 관리하고 차량에 할당하는 클래스"""
    def __init__(self):
        self.goals = {}  # {goal_id: Goal 객체}
        self.vehicle_goals = {}  # {vehicle_id: goal_id}
        self.next_goal_id = 0  # 목적지 ID 자동 생성용

    def add_goal(self, x, y, yaw=0.0, radius=1.0, color=(0, 255, 0), is_static=True):
        """
        새 목적지 추가

        Args:
            x: 목적지 X 좌표
            y: 목적지 Y 좌표
            yaw: 방향각 [rad]
            radius: 목적지 반경
            color: 색상 (RGB)
            is_static: 정적 목적지 여부 (True: 고정, False: 이동 가능)

        Returns:
            goal_id: 생성된 목적지 ID
        """
        goal_id = self.next_goal_id
        self.next_goal_id += 1

        goal = Goal(x, y, yaw, color=color, radius=radius, is_static=is_static)
        self.goals[goal_id] = goal

        return goal_id

    def add_goal_with_id(self, goal_id, x, y, yaw=0.0, radius=1.0, color=(0, 255, 0)):
        """
        특정 ID로 목적지 추가 (상태 로드용)

        Args:
            goal_id: 목적지 ID (저장된 상태에서 복원된 ID)
            x: 목적지 X 좌표
            y: 목적지 Y 좌표
            yaw: 방향각 [rad]
            radius: 목적지 반경
            color: 색상 (RGB)

        Returns:
            goal_id: 동일한 입력 목적지 ID
        """
        goal = Goal(x, y, yaw, color=color, radius=radius)
        self.goals[goal_id] = goal

        # 다음 ID가 현재 ID보다 작으면 업데이트
        if self.next_goal_id <= goal_id:
            self.next_goal_id = goal_id + 1

        return goal_id

    def remove_goal(self, goal_id):
        """
        목적지 제거

        Args:
            goal_id: 제거할 목적지 ID

        Returns:
            성공 여부 (Boolean)
        """
        if goal_id in self.goals:
            del self.goals[goal_id]

            # 차량-목적지 매핑에서도 제거
            for vehicle_id, assigned_goal_id in list(self.vehicle_goals.items()):
                if assigned_goal_id == goal_id:
                    del self.vehicle_goals[vehicle_id]

            return True
        return False

    def assign_goal_to_vehicle(self, vehicle_id, goal_id):
        """
        차량에 목적지 할당

        Args:
            vehicle_id: 차량 ID
            goal_id: 목적지 ID

        Returns:
            성공 여부 (Boolean)
        """
        if goal_id in self.goals:
            self.vehicle_goals[vehicle_id] = goal_id
            return True
        return False

    def get_vehicle_goal(self, vehicle_id):
        """
        차량의 할당된 목적지 반환

        Args:
            vehicle_id: 차량 ID

        Returns:
            Goal 객체 또는 None
        """
        goal_id = self.vehicle_goals.get(vehicle_id)
        if goal_id is not None:
            return self.goals.get(goal_id)
        return None

    def get_vehicle_goal_id(self, vehicle_id):
        """
        차량의 할당된 목적지 ID 반환

        Args:
            vehicle_id: 차량 ID

        Returns:
            goal_id 또는 None
        """
        return self.vehicle_goals.get(vehicle_id)

    def draw(self, screen, world_to_screen_func, debug=False):
        """
        모든 목적지 렌더링

        Args:
            screen: Pygame 화면 객체
            world_to_screen_func: 좌표 변환 함수
            debug: 디버그 정보 표시 여부
        """
        for goal in self.goals.values():
            goal.draw(screen, world_to_screen_func, debug)

    def get_all_goals(self):
        """모든 목적지 객체 반환"""
        return list(self.goals.values())

    def get_goal_count(self):
        """목적지 수 반환"""
        return len(self.goals)

    def clear_goals(self):
        """모든 목적지 제거"""
        self.goals.clear()
        self.vehicle_goals.clear()
