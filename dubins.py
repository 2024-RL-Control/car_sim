# -*- coding: utf-8 -*-
import numpy as np

class DubinsPath:
    """Dubins 곡선을 직접 구현하여 차량 경로 생성"""
    def __init__(self, min_turn_radius=1.0, step_size=0.5):
        self.min_turn_radius = min_turn_radius
        self.step_size = step_size

    def dubins_segment(self, start, theta, length, direction):
        """
        단일 Dubins 세그먼트 (회전 or 직선)
        Args:
            start: 시작점 (x, y)
            theta: 진행 방향 (rad)
            length: 이동 거리
            direction: 회전 방향 (1: 좌회전, -1: 우회전, 0: 직선)
        Returns:
            path: [(x1, y1), (x2, y2), ...]
        """
        x, y = start
        points = [(x, y)]
        step_size = self.step_size  # 샘플링 간격

        for _ in np.arange(0, length, step_size):
            if direction == 0:  # 직선 이동
                x += step_size * np.cos(theta)
                y += step_size * np.sin(theta)
            else:  # 회전 (좌회전 or 우회전)
                theta += direction * (step_size / self.min_turn_radius)
                x += self.min_turn_radius * (np.sin(theta) - np.sin(theta - direction * (step_size / self.min_turn_radius)))
                y -= self.min_turn_radius * (np.cos(theta) - np.cos(theta - direction * (step_size / self.min_turn_radius)))

            points.append((x, y))

        return points

    def generate_path(self, start, start_angle, goal, goal_angle):
        """
        Dubins 경로 생성 (LSL, RSR, LSR, RSL, RLR, LRL 중 최적 경로 선택)
        Args:
            start: 시작점 (x, y)
            start_angle: 시작 각도 (rad)
            goal: 목표점 (x, y)
            goal_angle: 목표 각도 (rad)
        Returns:
            path: [(x1, y1), (x2, y2), ...]
        """
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        distance = np.sqrt(dx**2 + dy**2)

        if distance < 2 * self.min_turn_radius:
            distance = 2 * self.min_turn_radius

        # 직선 이동 거리 계산
        straight_length = distance - 2 * self.min_turn_radius

        # LSL (좌회전 - 직선 - 좌회전) 경로
        turn1 = self.dubins_segment(start, start_angle, self.min_turn_radius, 1)
        straight_start = turn1[-1]
        straight = self.dubins_segment(straight_start, start_angle, straight_length, 0)
        turn2_start = straight[-1]
        turn2 = self.dubins_segment(turn2_start, goal_angle, self.min_turn_radius, 1)

        # 최종 경로 반환
        path = turn1 + straight + turn2
        return path

if __name__ == "__main__":
    dubins_path = DubinsPath(min_turn_radius=2.0)

    start = (0, 0)
    start_angle = np.radians(45)  # 45도 방향
    goal = (10, 10)
    goal_angle = np.radians(90)  # 90도 방향

    path = dubins_path.generate_path(start, start_angle, goal, goal_angle)

    print("Dubins Path:")
    for point in path:
        print(point)
