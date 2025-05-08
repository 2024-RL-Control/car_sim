# -*- coding: utf-8 -*-
import pygame
import numpy as np
import math
import random
from collections import deque

class Node:
    _id_iter = 0
    def __init__(self, x, y, yaw=0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.id = Node._id_iter
        Node._id_iter += 1

    def draw(self, screen, debug=False, color=(0, 0, 255), radius=5):
        """
        pygame 시각화 메소드
        debug 모드 시 노드 id 같이 출력
        """
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), radius)
        if debug:
            font = pygame.font.SysFont(None, 12)
            text = font.render(f"{self.id}", True, (255, 255, 255))
            screen.blit(text, (int(self.x)-2, int(self.y)-4))

    def distance(self, other):
        """두 노드 사이의 유클리드 거리 계산"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def __eq__(self, other):
        if isinstance(other, Node):
            # 위치가 매우 가까우면 같은 노드로 간주 (부동소수점 오차 고려)
            return abs(self.x - other.x) < 0.01 and abs(self.y - other.y) < 0.01 and abs(self.yaw - other.yaw) < 0.01
        return False

    def get_point(self):
        """
        (x, y, yaw) 반환
        """
        return (self.x, self.y, self.yaw)


class Link:
    def __init__(self, start_node, end_node, obstacles=[], mode='plan', config=None):
        """
        start_node, end_node: Node 객체 (x, y, yaw 포함)
        mode: 'plan', 'straight', 'curve' 중 하나
        """
        self.start = start_node
        self.end = end_node
        self.mode = mode
        self.width = 6  # 6m
        self.path = PathPlanner.plan(
            start_node,
            end_node,
            obstacles,
            self.width,
            mode,
            config
        )
        self.target_vel = PathPlanner.calculate_target_vel(self.path, config["default_speed"], config["min_speed"], config["max_speed"])
        self._sampled_points = None
        self._sample_step = None

    def sample(self, step_size=0.5):
        """
        step_size 간격으로 (x,y,yaw) 리스트 반환
        """
        if self._sampled_points is not None and self._sample_step == step_size:
            return self._sampled_points

        result = []
        if not self.path or len(self.path) < 2:
            return result

        # 경로의 각 점에 대해 처리
        for i in range(len(self.path) - 1):
            p1 = self.path[i]
            p2 = self.path[i + 1]

            # 두 점 사이의 거리 계산
            dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

            # 필요한 샘플 수 계산
            num_samples = max(2, int(dist / step_size))

            # 두 점 사이를 균등하게 나누어 샘플링
            for j in range(num_samples):
                t = j / (num_samples - 1)
                x = p1[0] * (1 - t) + p2[0] * t
                y = p1[1] * (1 - t) + p2[1] * t

                # yaw 계산 (두 점 사이의 각도)
                if i < len(self.path) - 2 and j == num_samples - 1:
                    # 다음 세그먼트의 방향을 고려한 보간
                    p3 = self.path[i + 2]
                    yaw = math.atan2(p3[1] - p1[1], p3[0] - p1[0])
                else:
                    yaw = math.atan2(p2[1] - p1[1], p2[0] - p1[0])

                result.append((x, y, yaw))

        self._sampled_points = result
        self._sample_step = step_size
        return result

    def draw(self, screen, debug=False):
        """
        pygame 시각화 메소드
        mode에 따라 더 정확한 pygame 시각화 메소드 사용
        path를 중심선으로 고정 너비(self.width)의 도로 시각화
        debug 모드 시 중심선과 연결된 노드 id 출력
        """
        if not self.path or len(self.path) < 2:
            return

        # 도로 색상 설정 (모드에 따라 다른 색상 사용)
        if self.mode == 'plan':
            road_color = (100, 100, 100)  # 회색
        else:
            road_color = (80, 80, 80)     # 어두운 회색

        # 도로 그리기
        for i in range(len(self.path) - 1):
            p1 = self.path[i]
            p2 = self.path[i + 1]

            # 선분의 방향 벡터
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx*dx + dy*dy)
            if length < 0.01:
                continue

            # 방향 벡터를 정규화
            dx, dy = dx/length, dy/length

            # 수직 방향 벡터 (시계 방향으로 90도 회전)
            perpx, perpy = -dy, dx

            # 도로 너비의 절반
            half_width = self.width / 2

            # 도로 모서리 4개 점
            points = [
                (p1[0] + perpx * half_width, p1[1] + perpy * half_width),
                (p2[0] + perpx * half_width, p2[1] + perpy * half_width),
                (p2[0] - perpx * half_width, p2[1] - perpy * half_width),
                (p1[0] - perpx * half_width, p1[1] - perpy * half_width)
            ]

            # 사각형 그리기
            pygame.draw.polygon(screen, road_color, points)

            # 중앙선 그리기 (디버그 모드)
            if debug:
                pygame.draw.line(screen, (255, 255, 0), (p1[0], p1[1]), (p2[0], p2[1]), 1)

        # 노드 그리기
        self.start.draw(screen, debug)
        self.end.draw(screen, debug)

        # 디버그 정보 출력
        if debug:
            font = pygame.font.SysFont(None, 12)
            text = font.render(f"{self.start.id}->{self.end.id}", True, (255, 255, 255))
            mid_point = self.path[len(self.path)//2]
            screen.blit(text, (int(mid_point[0]), int(mid_point[1]) - 20))


class PathPlanner:
    """
    경로 계획 클래스
    경로 계획시 Dubins steer 적용된 RRT 알고리즘 사용
        1. 장애물 회피 경로 계획
        2. 비홀로노믹 제약 반영한 경로 계획
    경로 지정(직진, 좌회전, 우회전)시 각각 맞는 함수 사용
    """

    @classmethod
    def plan(cls, node1, node2, obstacles, width, mode, config):
        """
        경로 계획 메소드
        """
        if mode == 'plan':
            return cls._rrt_with_dubins(node1, node2, obstacles, width, config)
        elif mode == 'straight':
            return cls._straight(node1, node2)
        elif mode == 'curve':
            return cls._quadratic_bezier_curve(node1, node2)
        else:
            return cls._straight(node1, node2)

    @classmethod
    def calculate_target_vel(cls, path, default_speed=30, min_speed=5, max_speed=50):
        """
        경로 곡률에 따른 목표 속도 계산
        최소 속도 5, 최대 속도 50
        """
        if not path or len(path) < 3:
            return default_speed  # 기본 속도

        # 전체 경로의 평균 곡률 계산
        total_curvature = 0
        count = 0

        for i in range(1, len(path) - 1):
            p1 = path[i-1]
            p2 = path[i]
            p3 = path[i+1]

            # 세 점을 이용한 곡률 계산
            try:
                # 두 벡터 사이의 각도 변화로 곡률 근사
                v1 = (p2[0] - p1[0], p2[1] - p1[1])
                v2 = (p3[0] - p2[0], p3[1] - p2[1])

                # 벡터의 길이
                len_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
                len_v2 = math.sqrt(v2[0]**2 + v2[1]**2)

                if len_v1 < 0.01 or len_v2 < 0.01:
                    continue

                # 정규화된 벡터
                v1_norm = (v1[0]/len_v1, v1[1]/len_v1)
                v2_norm = (v2[0]/len_v2, v2[1]/len_v2)

                # 내적 계산
                dot_product = v1_norm[0]*v2_norm[0] + v1_norm[1]*v2_norm[1]
                dot_product = max(-1.0, min(1.0, dot_product))  # 도메인 제한 (-1 ~ 1)

                # 각도 변화 (라디안)
                angle_change = math.acos(dot_product)

                # 이동 거리
                dist = (len_v1 + len_v2) / 2

                # 곡률 = 각도 변화 / 이동 거리
                curvature = angle_change / dist

                total_curvature += curvature
                count += 1
            except:
                # 곡률 계산 오류 시 무시
                pass

        if count == 0:
            return default_speed  # 기본 속도

        # 평균 곡률
        avg_curvature = total_curvature / count

        # 곡률에 따른 속도 조정 (곡률이 높을수록 속도 감소)
        # 일반적으로 곡률은 [0, 0.3] 범위의 값을 가짐
        max_curvature = 0.2  # 임의의 최대 곡률
        normalized_curvature = min(avg_curvature / max_curvature, 1.0)

        # 속도 범위: 5 ~ 50
        velocity = max_speed - normalized_curvature * (max_speed - min_speed)

        return max(min_speed, min(max_speed, velocity))

    @classmethod
    def _rrt_with_dubins(cls, start_node, end_node, obstacles, width, config):
        """
        RRT 알고리즘을 통해 목적지까지의 도로 너비 고려한 장애물 회피 경로 생성
        RRT 알고리즘에 Dubins steer 적용, 비홀로노믹 제약 고려
        RRT 알고리즘의 트리 노드들은 Node 클래스 사용하지 않고 내부적으로 노드 처리
        """
        # 에러 상황 처리
        if not obstacles:
            return cls._straight(start_node, end_node)

        class RRTNode:
            def __init__(self, x, y, yaw, parent=None):
                self.x = x
                self.y = y
                self.yaw = yaw
                self.parent = parent
                # 비교를 위한 반올림된 값
                self.key = (round(x, 4), round(y, 4), round(yaw, 4))

            def __eq__(self, other):
                if isinstance(other, RRTNode):
                    return self.key == other.key
                return False

            def __hash__(self):
                return hash(self.key)

            def get_tuple(self):
                return (self.x, self.y, self.yaw)

            def distance_to(self, other):
                """다른 노드와의 거리 계산"""
                if isinstance(other, RRTNode):
                    return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
                else:  # 튜플이나 다른 형태의 좌표
                    return math.sqrt((self.x - other[0])**2 + (self.y - other[1])**2)

        # 시작, 목표 노드
        start = RRTNode(start_node.x, start_node.y, start_node.yaw)
        goal = RRTNode(end_node.x, end_node.y, end_node.yaw)

        # RRT 트리
        nodes = [start]

        # 장애물 충돌 검사 거리 (도로 너비의 절반 + 여유)
        collision_dist = width / 2 + 2.0

        for i in range(config["max_iterations"]):
            # 랜덤 위치치 샘플링 (일정 확률로 목표 지점 선택)
            if random.random() < config["goal_sampling_rate"]:
                rand_point = (goal.x, goal.y, None)  # yaw는 경로 생성 후 결정
            else:
                rand_x = random.uniform(min(start.x, goal.x) - 100, max(start.x, goal.x) + 100)
                rand_y = random.uniform(min(start.y, goal.y) - 100, max(start.y, goal.y) + 100)
                rand_point = (rand_x, rand_y, None)

            # 가장 가까운 노드 찾기
            nearest_node = min(nodes, key=lambda n: n.distance_to(rand_point))
            nearest_point = nearest_node.get_tuple()

            # 스티어링 (Dubins Curve 적용)
            new_point = cls._steer(nearest_point, rand_point, config["step_size"], config["min_radius"])

            # 장애물과 충돌 확인
            if cls._is_collision_free(nearest_point, new_point, obstacles, collision_dist):
                new_node = RRTNode(new_point[0], new_point[1], new_point[2], parent=nearest_node)
                nodes.append(new_node)

                # 목표 지점 근처에 도달했는지 확인
                if new_node.distance_to(goal) < config["step_size"]:
                    # 목표 노드에 연결 시도
                    if cls._is_collision_free(new_node.get_tuple(), goal.get_tuple(), obstacles, collision_dist):
                        # 목표 노드의 부모를 마지막 노드로 설정정
                        goal.parent = new_node
                        nodes.append(goal)
                        break

        # 경로 생성
        path = []
        current = goal

        # 목표 노드가 트리에 연결되었는지 확인
        if goal.parent is not None:
            # 역추적하여 경로 생성
            while current is not None:
                path.append(current.get_tuple())
                current = current.parent
            path.reverse()  # 시작 -> 목표 순서로 반전

            # 경로 스무딩
            path = cls._smooth_path(path, obstacles, collision_dist)
        else:
            # 경로를 찾지 못한 경우
            print("Path not found after maximum iterations")

            # 최대한 목표 지점에 가까운 경로 생성
            closest_to_goal = min(nodes, key=lambda n: n.distance_to(goal))

            current = closest_to_goal
            while current is not None:
                path.append(current.get_tuple())
                current = current.parent
            path.reverse()

        return path

    @classmethod
    def _distance(cls, node1, node2):
        """두 노드 사이의 유클리드 거리"""
        return math.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)

    @classmethod
    def _steer(cls, from_node, to_node, step_size, min_radius):
        """
        from_node에서 to_node 방향으로 step_size만큼 이동
        Dubins 제약 (최소 회전 반경)을 고려
        """
        # 방향 벡터
        dx = to_node[0] - from_node[0]
        dy = to_node[1] - from_node[1]

        # 거리
        dist = math.sqrt(dx*dx + dy*dy)

        if dist < 0.01:
            return from_node  # 거리가 너무 가까우면 원래 노드 반환

        # 정규화
        dx, dy = dx/dist, dy/dist

        # from_node의 현재 방향과 목표 방향 사이의 각도 차이 계산
        target_yaw = math.atan2(dy, dx)
        current_yaw = from_node[2]

        # 각도 차이 (-pi ~ pi 범위로 조정)
        delta_yaw = target_yaw - current_yaw
        while delta_yaw > math.pi: delta_yaw -= 2 * math.pi
        while delta_yaw < -math.pi: delta_yaw += 2 * math.pi

        # 최대 각도 변화 제한 (비홀로노믹 제약)
        max_yaw_change = step_size / min_radius  # 회전 반경에 따른 최대 각도 변화

        if abs(delta_yaw) > max_yaw_change:
            # 최대 각도 변화로 제한
            new_yaw = current_yaw + (max_yaw_change if delta_yaw > 0 else -max_yaw_change)

            # 수정된 방향으로 이동
            new_x = from_node[0] + step_size * math.cos(new_yaw)
            new_y = from_node[1] + step_size * math.sin(new_yaw)
        else:
            # 목표 방향으로 직접 이동
            step = min(step_size, dist)  # 목표까지의 거리가 step_size보다 작으면 그 거리만큼만 이동
            new_x = from_node[0] + step * dx
            new_y = from_node[1] + step * dy
            new_yaw = target_yaw

        return (new_x, new_y, new_yaw)

    @classmethod
    def _is_collision_free(cls, point1, point2, obstacles, collision_dist):
        """
        point1과 point2 사이의 경로가 장애물과 충돌하지 않는지 확인
        선분-원형 교차 수학 공식 사용
        충돌하지 않으면 True, 충돌하면 False 반환
        """
        # 선분의 시작점과 끝점
        p1 = np.array([point1[0], point1[1]])
        p2 = np.array([point2[0], point2[1]])

        # 선분 벡터
        d = p2 - p1

        # 선분 길이의 제곱
        len_squared = np.dot(d, d)

        if len_squared < 1e-10:  # 선분 길이가 0에 가까운 경우
            # 시작점만 검사
            for obstacle in obstacles:
                obs_center = np.array([obstacle[0], obstacle[1]])
                obs_radius = obstacle[2] + collision_dist

                # 시작점과 장애물 거리 계산
                dist = np.linalg.norm(p1 - obs_center)
                if dist <= obs_radius:
                    return False
            return True

        # 각 장애물에 대해 검사
        for obstacle in obstacles:
            obs_center = np.array([obstacle[0], obstacle[1]])
            obs_radius = obstacle[2] + collision_dist

            # 장애물 중심에서 선분까지의 최단 거리 계산
            # 선분 매개변수 t 계산 (0 <= t <= 1이면 선분 위의 점)
            t = max(0, min(1, np.dot(obs_center - p1, d) / len_squared))

            # 선분 위의 최단 거리 지점
            projection = p1 + t * d

            # 장애물 중심에서 투영점까지의 거리
            dist = np.linalg.norm(obs_center - projection)

            # 거리가 장애물 반경 + 충돌 거리보다 작으면 충돌
            if dist <= obs_radius:
                return False

        return True

    @classmethod
    def _smooth_path(cls, path, obstacles, collision_dist):
        """경로 스무딩 - 불필요한 노드 제거"""
        if len(path) <= 2:
            return path

        smooth_path = [path[0]]
        i = 0

        while i < len(path) - 1:
            current = path[i]

            # 현재 노드에서 직접 연결 가능한 가장 먼 노드 찾기
            for j in range(len(path) - 1, i, -1):
                if cls._is_collision_free(current, path[j], obstacles, collision_dist):
                    smooth_path.append(path[j])
                    i = j
                    break
            else:
                # 직접 연결 가능한 노드를 찾지 못한 경우
                smooth_path.append(path[i + 1])
                i += 1

        # yaw 값 업데이트 - 각 노드의 yaw는 다음 노드를 향하는 방향
        for i in range(len(smooth_path) - 1):
            dx = smooth_path[i+1][0] - smooth_path[i][0]
            dy = smooth_path[i+1][1] - smooth_path[i][1]
            yaw = math.atan2(dy, dx)
            smooth_path[i] = (smooth_path[i][0], smooth_path[i][1], yaw)

        # 마지막 노드의 yaw 값은 원래 목표의 yaw를 유지
        if len(smooth_path) > 1:
            last_index = len(smooth_path) - 1
            smooth_path[last_index] = (
                smooth_path[last_index][0],
                smooth_path[last_index][1],
                path[-1][2]  # 원래 목표의 yaw
            )

        return smooth_path

    @classmethod
    def _straight(cls, node1, node2):
        """
        두 노드간 직선 경로 계획
        """
        # 시작점과 끝점
        start = (node1.x, node1.y, node1.yaw)
        end = (node2.x, node2.y, node2.yaw)

        # 방향 계산
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        yaw = math.atan2(dy, dx)

        # 5개 지점으로 경로 생성 (부드러운 시각화를 위해)
        path = []
        for i in range(5):
            t = i / 4  # 0, 0.25, 0.5, 0.75, 1.0
            x = start[0] + t * dx
            y = start[1] + t * dy

            # 시작과 끝의 yaw는 원래 값 유지, 중간은 보간
            if i == 0:
                path_yaw = start[2]
            elif i == 4:
                path_yaw = end[2]
            else:
                # 시작과 끝의 yaw 사이를 보간
                # yaw 보간 시 각도의 차이를 고려해야 함
                delta_yaw = end[2] - start[2]
                # -pi ~ pi 범위로 조정
                while delta_yaw > math.pi: delta_yaw -= 2 * math.pi
                while delta_yaw < -math.pi: delta_yaw += 2 * math.pi

                path_yaw = start[2] + t * delta_yaw

            path.append((x, y, path_yaw))

        return path

    @classmethod
    def _quadratic_bezier_curve(cls, node1, node2):
        """
        두 노드를 yaw 방향으로 직선을 앞 뒤로 확장했을 때, 수직으로 만나는 지점을 제어점으로 사용하여 베지에 곡선 경로 계획
        """
        # 시작점과 끝점
        start = (node1.x, node1.y, node1.yaw)
        end = (node2.x, node2.y, node2.yaw)

        # 시작 방향 및 끝 방향의 단위 벡터
        start_dir = (math.cos(start[2]), math.sin(start[2]))
        end_dir = (math.cos(end[2]), math.sin(end[2]))

        # 두 직선의 방정식 계수 계산
        # 직선 방정식: a*x + b*y + c = 0
        a1 = -start_dir[1]  # -sin(yaw1)
        b1 = start_dir[0]   # cos(yaw1)
        c1 = -(a1 * start[0] + b1 * start[1])

        a2 = -end_dir[1]    # -sin(yaw2)
        b2 = end_dir[0]     # cos(yaw2)
        c2 = -(a2 * end[0] + b2 * end[1])

        # 두 직선의 교차점 계산 (크래머 법칙)
        det = a1 * b2 - a2 * b1

        # 두 직선이 평행하거나 거의 평행한 경우
        if abs(det) < 1e-10:
            # 시작점과 끝점 사이의 거리의 절반을 사용하여 제어점 계산 (기존 방식 사용)
            dist = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            control_dist = dist * 0.5

            # 시작점과 끝점 사이의 방향 벡터
            dir_vector = ((end[0] - start[0]), (end[1] - start[1]))
            dir_length = math.sqrt(dir_vector[0]**2 + dir_vector[1]**2)

            if dir_length > 1e-10:
                normalized_dir = (dir_vector[0] / dir_length, dir_vector[1] / dir_length)

                # 수직 방향으로 제어점 배치
                perp_dir = (-normalized_dir[1], normalized_dir[0])
                mid_x = (start[0] + end[0]) / 2
                mid_y = (start[1] + end[1]) / 2

                control_x = mid_x + perp_dir[0] * control_dist
                control_y = mid_y + perp_dir[1] * control_dist
            else:
                # 두 점이 너무 가까운 경우 시작점을 그대로 사용
                control_x = start[0]
                control_y = start[1]
        else:
            # 두 직선의 교차점 계산
            control_x = (b1 * c2 - b2 * c1) / det
            control_y = (a2 * c1 - a1 * c2) / det

            # 교차점이 너무 멀리 있는지 확인
            dist_to_start = math.sqrt((control_x - start[0])**2 + (control_y - start[1])**2)
            dist_to_end = math.sqrt((control_x - end[0])**2 + (control_y - end[1])**2)
            max_allowed_dist = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2) * 5

            if dist_to_start > max_allowed_dist or dist_to_end > max_allowed_dist:
                # 제어점이 너무 멀리 있는 경우 중간점 사용
                mid_x = (start[0] + end[0]) / 2
                mid_y = (start[1] + end[1]) / 2

                # 시작점과 끝점을 연결하는 벡터에 수직인 방향으로 제어점 설정
                dir_vector = ((end[0] - start[0]), (end[1] - start[1]))
                dir_length = math.sqrt(dir_vector[0]**2 + dir_vector[1]**2)

                if dir_length > 1e-10:
                    normalized_dir = (dir_vector[0] / dir_length, dir_vector[1] / dir_length)
                    perp_dir = (-normalized_dir[1], normalized_dir[0])

                    # 중간점에서 수직 방향으로 적절한 거리만큼 이동
                    control_dist = dir_length * 0.3
                    control_x = mid_x + perp_dir[0] * control_dist
                    control_y = mid_y + perp_dir[1] * control_dist
                else:
                    control_x = mid_x
                    control_y = mid_y

        # 베지에 곡선 경로 생성 (20개 지점)
        path = []
        num_points = 20

        for i in range(num_points):
            t = i / (num_points - 1)  # 0 ~ 1

            # 이차 베지에 곡선 공식: B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
            x = (1-t)**2 * start[0] + 2*(1-t)*t * control_x + t**2 * end[0]
            y = (1-t)**2 * start[1] + 2*(1-t)*t * control_y + t**2 * end[1]

            # yaw 계산 (접선의 방향)
            # 접선 벡터 = dB/dt = 2(1-t)(P₁-P₀) + 2t(P₂-P₁)
            dx = 2*(1-t)*(control_x - start[0]) + 2*t*(end[0] - control_x)
            dy = 2*(1-t)*(control_y - start[1]) + 2*t*(end[1] - control_y)

            yaw = math.atan2(dy, dx)

            path.append((x, y, yaw))

        return path


class RoadNetworkManager:
    """
    노드와 링크 통합 관리

    차량 별 초기 목적지에 대한 도로 생성 or 초기 노드들에 대한 도로 생성

    장애물 정보: ObstacleManager.get_all_outer_circles()
    """
    def __init__(self, config=None):
        self.nodes = []
        self.links = []
        self.closest_point = None
        self.config = config

    def add_node(self, point):
        """노드 추가"""
        for node in self.nodes:
            if node.get_point() == point:
                return node

        new_node = Node(point[0], point[1], point[2])
        self.nodes.append(new_node)
        return new_node

    def add_link(self, node1, node2, obstacles=[], mode='plan'):
        """링크 추가"""
        # 링크 생성 및 추가
        new_link = Link(node1, node2, obstacles, mode, self.config)
        self.links.append(new_link)

        return new_link

    def connect(self, point1=(0,0,0), point2=(100,100,0), obstacles=[], mode='plan'):
        """
        1. 두 좌표(x,y,yaw)를 Node 클래스로 변환 후 nodes에 추가
            1. 같은 노드 있으면 재활용
        2. Link 클래스 활용해 두 노드간 연결 후 links에 추가
            1. 경로 계획 모드: 장애물과 충돌하지 않는 경로 생성
            2. 직진 모드: 직선 경로 생성
            3. 곡선 모드: 베지에 곡선으로 좌회전, 우회전 경로 생성
        """
        node1 = self.add_node(point1)
        node2 = self.add_node(point2)

        new_link = self.add_link(node1, node2, obstacles, mode)
        return new_link

    def _get_closest_node(self, x, y):
        """특정 위치와 가장 가까운 노드 찾기"""
        if not self.nodes:
            return None

        min_dist = float('inf')
        closest_node = None

        for node in self.nodes:
            dist = math.sqrt((node.x - x)**2 + (node.y - y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_node = node

        return closest_node

    def _find_link_by_node(self, node):
        """노드가 포함된 모든 링크 찾기"""
        if not node:
            return []

        result = []
        for link in self.links:
            if link.start == node or link.end == node:
                result.append(link)

        return result

    def _get_closest_point(self, x, y, link):
        """링크의 샘플링 지점중 가장 가까운 포인트 찾기"""
        if not link:
            return None, float('inf')

        # 링크의 경로 샘플링
        sampled_points = link.sample(0.5)

        if not sampled_points:
            return None, float('inf')

        # 가장 가까운 포인트 찾기
        min_dist = float('inf')
        closest_point = None

        for point in sampled_points:
            dist = math.sqrt((point[0] - x)**2 + (point[1] - y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_point = point

        return closest_point, min_dist

    def get_frenet_d(self, vehicle):
        """
        가장 가까운 포인트와의 거리 계산(Frenet d), 차량의 상태로 활용
        """
        if not self.links:
            return float('inf')

        # 차량 위치
        x, y, yaw = vehicle.get_position()

        # 가장 가까운 노드 찾기
        closest_node = self._get_closest_node(x, y)

        # 해당 노드와 연결된 링크 찾기
        links = self._find_link_by_node(closest_node)

        # 가장 가까운 링크와 포인트 찾기
        min_dist = float('inf')
        closest_point = None
        closest_link = None

        for link in links:
            point, dist = self._get_closest_point(x, y, link)
            if dist < min_dist:
                min_dist = dist
                closest_point = point
                closest_link = link

        # 추가적으로 근처의 다른 링크도 확인
        for link in self.links:
            if link in links:
                continue

            point, dist = self._get_closest_point(x, y, link)
            if dist < min_dist:
                min_dist = dist
                closest_point = point
                closest_link = link

        if not closest_point:
            return float('inf')

        self.closest_point = closest_point  # 가장 가까운 포인트 저장

        # Frenet 좌표계의 d 계산 (차량이 경로에서 좌/우로 얼마나 떨어져 있는지)
        # 차량과 경로 포인트 사이의 벡터
        dx = x - closest_point[0]
        dy = y - closest_point[1]

        # 경로의 방향 벡터 (yaw 기준)
        path_dir_x = math.cos(closest_point[2])
        path_dir_y = math.sin(closest_point[2])

        # 경로 방향에 수직인 벡터 (왼쪽 방향)
        perp_dir_x = -path_dir_y
        perp_dir_y = path_dir_x

        # d 계산 (수직 벡터와의 내적)
        d = dx * perp_dir_x + dy * perp_dir_y

        return d

    def reset(self):
        """노드와 링크 초기화"""
        self.nodes = []
        self.links = []
        Node._id_iter = 0  # 노드 ID 초기화
        self.closest_point = None

    def draw(self, screen, debug=False):
        """네트워크 시각화"""
        # 모든 링크 그리기
        for link in self.links:
            link.draw(screen, debug)

        if debug:
            # 모든 노드 그리기 (링크가 그리지 않은 노드만)
            drawn_nodes = set()
            for link in self.links:
                drawn_nodes.add(link.start)
                drawn_nodes.add(link.end)

            # 링크에 포함되지 않은 노드 그리기
            for node in self.nodes:
                if node not in drawn_nodes:
                    node.draw(screen, debug)

            if self.closest_point:
                # 가장 가까운 포인트 시각화
                pygame.draw.circle(screen, (255, 0, 0), (int(self.closest_point[0]), int(self.closest_point[1])), 2)


# 메인 테스트 함수
def main():
    class Vehicle:
        """테스트용 더미 차량 클래스"""
        def __init__(self, x=0, y=0, yaw=0):
            self.x = x
            self.y = y
            self.yaw = yaw
            self.speed = 0
            self.width = 2.0
            self.length = 4.5

        def move(self, dx, dy):
            """위치 이동"""
            self.x += dx
            self.y += dy
            # 이동 방향에 따라 yaw 업데이트
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                self.yaw = math.atan2(dy, dx)

        def draw(self, screen, color=(0, 150, 255)):
            """차량 시각화"""
            # 차량 중심점
            cx, cy = int(self.x), int(self.y)

            # 차량 모양 (방향에 따라 회전)
            corners = [
                (-self.length/2, -self.width/2),
                (self.length/2, -self.width/2),
                (self.length/2, self.width/2),
                (-self.length/2, self.width/2)
            ]

            # 회전 변환
            rotated_corners = []
            for x, y in corners:
                # 회전 행렬 적용
                rx = x * math.cos(self.yaw) - y * math.sin(self.yaw)
                ry = x * math.sin(self.yaw) + y * math.cos(self.yaw)
                rotated_corners.append((int(cx + rx), int(cy + ry)))

            # 차량 본체 그리기
            pygame.draw.polygon(screen, color, rotated_corners)

            # 차량 전면 표시 (방향 표시)
            front_x = cx + math.cos(self.yaw) * self.length/2
            front_y = cy + math.sin(self.yaw) * self.length/2
            pygame.draw.circle(screen, (255, 0, 0), (int(front_x), int(front_y)), 3)

    # Pygame 초기화
    pygame.init()
    width, height = 1000, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Road Network System Test")
    clock = pygame.time.Clock()

    # 도로 네트워크 관리자 생성
    road_manager = RoadNetworkManager()

    # 더미 차량 생성
    vehicle = Vehicle(400, 300)

    # 테스트용 변수
    nodes = []  # 마우스 클릭으로 추가된 노드 (최대 2개)
    obstacles = []  # 마우스 우클릭으로 추가된 장애물
    debug_mode = False  # 디버그 모드

    # 교차로 생성 함수
    def create_intersection(center_x, center_y):
        # 초기화
        road_manager.reset()

        # 교차로 중심
        center = (center_x, center_y)

        # 도로 길이
        road_length = 200
        road_width = 20

        # 우측 통행을 위한 각 방향의 입구/출구 노드 (동, 서, 남, 북)
        # 각 방향별로 두 개의 노드 생성 (입구, 출구)

        # 동쪽 (오른쪽)
        east_in = (center_x + road_length, center_y - road_width, math.pi)  # 서쪽 방향을 바라봄
        east_out = (center_x + road_length, center_y + road_width, 0)       # 동쪽 방향을 바라봄

        # 서쪽 (왼쪽)
        west_in = (center_x - road_length, center_y + road_width, 0)        # 동쪽 방향을 바라봄
        west_out = (center_x - road_length, center_y - road_width, math.pi) # 서쪽 방향을 바라봄

        # 남쪽 (아래)
        south_in = (center_x + road_width, center_y + road_length, -math.pi/2)  # 북쪽 방향을 바라봄
        south_out = (center_x - road_width, center_y + road_length, math.pi/2)  # 남쪽 방향을 바라봄

        # 북쪽 (위)
        north_in = (center_x - road_width, center_y - road_length, math.pi/2)   # 남쪽 방향을 바라봄
        north_out = (center_x + road_width, center_y - road_length, -math.pi/2) # 북쪽 방향을 바라봄

        # 교차로 중심 근처 노드 (각 방향에서 교차로로 들어가는 지점, 사분면면)
        center_1 = (center_x + road_width, center_y - road_width, math.pi)
        center_2 = (center_x - road_width, center_y - road_width, math.pi/2)
        center_3 = (center_x - road_width, center_y + road_width, 0)
        center_4 = (center_x + road_width, center_y + road_width, -math.pi/2)

        # 직진 경로 생성
        # 동쪽 진입 -> 서쪽 진출
        road_manager.connect(east_in, center_1, obstacles, 'straight')
        road_manager.connect(center_1, center_2, obstacles, 'straight')
        road_manager.connect(center_2, west_out, obstacles, 'straight')

        # 서쪽 진입 -> 동쪽 진출
        road_manager.connect(west_in, center_3, obstacles, 'straight')
        road_manager.connect(center_3, center_4, obstacles, 'straight')
        road_manager.connect(center_4, east_out, obstacles, 'straight')

        # 남쪽 진입 -> 북쪽 진출
        road_manager.connect(south_in, center_4, obstacles, 'straight')
        road_manager.connect(center_4, center_1, obstacles, 'straight')
        road_manager.connect(center_1, north_out, obstacles, 'straight')

        # 북쪽 진입 -> 남쪽 진출
        road_manager.connect(north_in, center_2, obstacles, 'straight')
        road_manager.connect(center_2, center_3, obstacles, 'straight')
        road_manager.connect(center_3, south_out, obstacles, 'straight')

        # 좌회전 경로 생성
        # 동쪽 진입 -> 남쪽 진출
        road_manager.connect(east_in, center_1, obstacles, 'straight')
        road_manager.connect(center_1, south_out, obstacles, 'curve')

        # 서쪽 진입 -> 북쪽 진출
        road_manager.connect(west_in, center_3, obstacles, 'straight')
        road_manager.connect(center_3, north_out, obstacles, 'curve')

        # 남쪽 진입 -> 서쪽 진출
        road_manager.connect(south_in, center_4, obstacles, 'straight')
        road_manager.connect(center_4, west_out, obstacles, 'curve')

        # 북쪽 진입 -> 동쪽 진출
        road_manager.connect(north_in, center_2, obstacles, 'straight')
        road_manager.connect(center_2, east_out, obstacles, 'curve')

        # 우회전 경로 생성
        # 동쪽 진입 -> 북쪽 진출
        road_manager.connect(east_in, north_out, obstacles, 'curve')

        # 서쪽 진입 -> 남쪽 진출
        road_manager.connect(west_in, south_out, obstacles, 'curve')

        # 남쪽 진입 -> 동쪽 진출
        road_manager.connect(south_in, east_out, obstacles, 'curve')

        # 북쪽 진입 -> 서쪽 진출
        road_manager.connect(north_in, west_out, obstacles, 'curve')

    # 기본 교차로 생성
    create_intersection(width/2, height/2)

    # 게임 루프
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # 마우스 클릭 위치
                x, y = pygame.mouse.get_pos()
                if event.button == 1:  # 좌클릭: 노드 추가
                    if len(nodes) < 2:
                        nodes.append((x, y, 0))  # 기본 yaw = 0
                        # 두 번째 노드가 추가되면 경로 계획
                        if len(nodes) == 2:
                            road_manager.connect(nodes[0], nodes[1], obstacles, 'plan')
                            nodes = []  # 노드 리스트 초기화
                elif event.button == 3:  # 우클릭: 장애물 추가
                    obstacles.append((x, y, 10))  # 반경 10의 원형 장애물
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F1:  # F1: 디버그 모드 토글
                    debug_mode = not debug_mode
                elif event.key == pygame.K_r:  # R: 초기화
                    road_manager.reset()
                    nodes = []
                    obstacles = []
                elif event.key == pygame.K_c:  # C: 교차로 생성
                    create_intersection(width/2, height/2)

        # 차량 이동 제어 (W,A,S,D)
        keys = pygame.key.get_pressed()
        move_speed = 3
        dx, dy = 0, 0

        if keys[pygame.K_w]:  # 위
            dy -= move_speed
        if keys[pygame.K_s]:  # 아래
            dy += move_speed
        if keys[pygame.K_a]:  # 왼쪽
            dx -= move_speed
        if keys[pygame.K_d]:  # 오른쪽
            dx += move_speed

        vehicle.move(dx, dy)

        # 화면 그리기
        screen.fill((0, 0, 0))  # 검은색 배경

        # 교차로 그리기
        road_manager.draw(screen, debug_mode)

        # 더미 차량 그리기
        vehicle.draw(screen)

        # 장애물 그리기
        for obs in obstacles:
            pygame.draw.circle(screen, (255, 0, 0), (int(obs[0]), int(obs[1])), obs[2])

        # Frenet 좌표계 d 값 출력 (차량과 경로의 거리)
        frenet_d = road_manager.get_distance(vehicle)
        frenet_text = f"Frenet d: {frenet_d:.2f}m"
        font = pygame.font.SysFont(None, 16)
        text = font.render(frenet_text, True, (255, 255, 255))
        screen.blit(text, (10, 10))

        # 컨트롤 도움말
        if debug_mode:
            help_texts = [
                "Controls:",
                "Left Click: Add node (2 clicks to create path)",
                "Right Click: Add obstacle",
                "W,A,S,D: Move vehicle",
                "F1: Toggle debug mode",
                "R: Reset",
                "C: Create intersection"
            ]

            y_pos = 40
            for line in help_texts:
                text = font.render(line, True, (200, 200, 200))
                screen.blit(text, (10, y_pos))
                y_pos += 20

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
