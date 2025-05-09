# -*- coding: utf-8 -*-
import pygame
import numpy as np
import math
import random
from collections import deque

def ortho(vect2d):
    """Computes an orthogonal vector to the one given"""
    return np.array((-vect2d[1], vect2d[0]))

def dist(pt_a, pt_b):
    """Euclidian distance between two (x, y) points"""
    return ((pt_a[0]-pt_b[0])**2 + (pt_a[1]-pt_b[1])**2)**.5

class Node:
    _id_iter = 0
    def __init__(self, x, y, yaw=0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.id = Node._id_iter
        Node._id_iter += 1

    def draw(self, screen, world_to_screen_func, debug=False, color=(0, 0, 255), radius=5):
        """
        pygame 시각화 메소드
        debug 모드 시 노드 id 같이 출력
        """
        screen_pos = world_to_screen_func(self.x, self.y)
        pygame.draw.circle(screen, color, (int(screen_pos[0]), int(screen_pos[1])), radius)
        if debug:
            font = pygame.font.SysFont(None, 12)
            text = font.render(f"{self.id}", True, (255, 255, 255))
            screen.blit(text, (int(screen_pos[0])-2, int(screen_pos[1])-4))

    def distance(self, other):
        """두 노드 사이의 유클리드 거리 계산"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def __eq__(self, other):
        if isinstance(other, Node):
            # 위치가 매우 가까우면 같은 노드로 간주 (부동소수점 오차 고려)
            return abs(self.x - other.x) < 0.01 and abs(self.y - other.y) < 0.01 and abs(self.yaw - other.yaw) < 0.01
        return False

    def __hash__(self):
        # ID 기반으로 해시값 생성 (set, dict 사용 위함)
        return hash(self.id)

    def get_point(self):
        """
        (x, y, yaw) 반환
        """
        return (self.x, self.y, self.yaw)

    def get_serializable_state(self):
        return {
            "x": self.x,
            "y": self.y,
            "yaw": self.yaw,
            "id": self.id
        }

    def load_from_serialized(self, data):
        self.x = data["x"]
        self.y = data["y"]
        self.yaw = data["yaw"]
        self.id = data["id"]


class Link:
    def __init__(self, start_node, end_node, obstacles=[], mode='plan', config=None):
        """
        start_node, end_node: Node 객체 (x, y, yaw 포함)
        mode: 'plan', 'straight', 'curve' 중 하나
        """
        self.start = start_node
        self.end = end_node
        self.mode = mode
        self.width = config['road_width']
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

    def draw(self, screen, world_to_screen_func, debug=False):
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
            p1_world = self.path[i]
            p2_world = self.path[i+1]

            # 선분의 방향 벡터
            dx = p2_world[0] - p1_world[0]
            dy = p2_world[1] - p1_world[1]
            length = math.sqrt(dx*dx + dy*dy)
            if length < 0.01:
                continue

            # 방향 벡터를 정규화
            dx_norm, dy_norm = dx/length, dy/length

            # 수직 방향 벡터 (시계 방향으로 90도 회전)
            perpx, perpy = -dy_norm, dx_norm

            # 도로 너비의 절반
            half_width = self.width / 2

            # 도로 모서리 4개 점 (월드 좌표)
            world_points = [
                (p1_world[0] + perpx * half_width, p1_world[1] + perpy * half_width),
                (p2_world[0] + perpx * half_width, p2_world[1] + perpy * half_width),
                (p2_world[0] - perpx * half_width, p2_world[1] - perpy * half_width),
                (p1_world[0] - perpx * half_width, p1_world[1] - perpy * half_width)
            ]

            # 스크린 좌표로 변환
            screen_points = [world_to_screen_func(*wp[:2]) for wp in world_points]

            # 사각형 그리기
            pygame.draw.polygon(screen, road_color, screen_points)

            # 중앙선 그리기 (디버그 모드)
            if debug:
                screen_p1 = world_to_screen_func(p1_world[0], p1_world[1])
                screen_p2 = world_to_screen_func(p2_world[0], p2_world[1])
                pygame.draw.line(screen, (255, 255, 0), screen_p1, screen_p2, 1)

        # 노드 그리기 (world_to_screen_func 전달)
        self.start.draw(screen, world_to_screen_func, debug)
        self.end.draw(screen, world_to_screen_func, debug)

        # 디버그 정보 출력
        if debug:
            font = pygame.font.SysFont(None, 12)
            text = font.render(f"{self.start.id}->{self.end.id}", True, (255, 255, 255))
            mid_point_world = self.path[len(self.path)//2]
            mid_point_screen = world_to_screen_func(mid_point_world[0], mid_point_world[1])
            screen.blit(text, (int(mid_point_screen[0]), int(mid_point_screen[1]) - 20))

    def get_serializable_state(self):
        """
        Link 객체의 직렬화 상태 반환
        """
        return {
            "start_id": self.start.id,
            "end_id": self.end.id,
            "mode": self.mode,
            "width": self.width
        }

class Dubins:
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
        if 2*self.radius < dist_intercenter > 4*self.radius:
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
        if 2*self.radius < dist_intercenter > 4*self.radius:
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
        total = self.radius*(abs(path[1])+abs(path[0]))+path[2] # Path length
        center_0 = self.find_center(start, 'L' if path[0] > 0 else 'R')
        center_2 = self.find_center(end, 'L' if path[1] > 0 else 'R')

        # We first need to find the points where the straight segment starts
        if abs(path[0]) > 0:
            angle = start[2]+(abs(path[0])-np.pi/2)*np.sign(path[0])
            ini = center_0+self.radius*np.array([np.cos(angle), np.sin(angle)])
        else: ini = np.array(start[:2])
        # We then identify its end
        if abs(path[1]) > 0:
            angle = end[2]+(-abs(path[1])-np.pi/2)*np.sign(path[1])
            fin = center_2+self.radius*np.array([np.cos(angle), np.sin(angle)])
        else: fin = np.array(end[:2])
        dist_straight = dist(ini, fin)

        # We can now generate all the points with the desired precision
        points = []
        for x in np.arange(0, total, self.point_separation):
            if x < abs(path[0])*self.radius: # First turn
                points.append(self.circle_arc(start, path[0], center_0, x))
            elif x > total - abs(path[1])*self.radius: # Last turn
                points.append(self.circle_arc(end, path[1], center_2, x-total))
            else: # Straight segment
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
            if x < abs(path[0])*self.radius: # First turn
                points.append(self.circle_arc(start, path[0], center_0, x))
            elif x > total - abs(path[1])*self.radius: # Last turn
                points.append(self.circle_arc(end, path[1], center_2, x-total))
            else: # Middle Turn
                angle = psi_0-np.sign(path[0])*(x/self.radius-abs(path[0]))
                vect = np.array([np.cos(angle), np.sin(angle)])
                points.append(center_1+self.radius*vect)
        points.append(end[:2])
        return np.array(points)

    def circle_arc(self, reference, beta, center, x):
        angle = reference[2]+((x/self.radius)-np.pi/2)*np.sign(beta)
        vect = np.array([np.cos(angle), np.sin(angle)])
        return center+self.radius*vect

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
    def _is_collision_path(cls, path_points, obstacles, collision_dist):
        """Checks if any point in path_points collides with obstacles."""
        # Handle None, empty list, or empty NumPy array
        if path_points is None or len(path_points) == 0:
            return False  # Empty path is not in collision
        for point in path_points: # Assuming point is (x, y, optional_yaw)
            point_coords = (point[0], point[1]) # Take only x, y for distance calculation
            for obstacle in obstacles: # obstacle format: (x_center, y_center, radius)
                obs_center = np.array([obstacle[0], obstacle[1]])
                # collision_dist already includes half width + buffer
                obs_effective_radius = obstacle[2] + collision_dist

                dist_sq = (point_coords[0] - obs_center[0])**2 + (point_coords[1] - obs_center[1])**2
                if dist_sq < obs_effective_radius**2:
                    return True  # Collision detected
        return False # No collision

    @classmethod
    def _reconstruct_path(cls, root_pos, goal_reached_pos, rrt_nodes_map, rrt_edges_map):
        """
        Reconstructs the path from start to goal using parent pointers.
        Returns a list of (x, y, yaw) waypoints.
        """
        if goal_reached_pos is None or goal_reached_pos not in rrt_nodes_map:
            return []

        # Handle trivial case: goal is the root
        if goal_reached_pos == root_pos:
            edge_for_start_goal = rrt_edges_map.get((root_pos, root_pos))
            if edge_for_start_goal and edge_for_start_goal.path:
                return list(edge_for_start_goal.path)
            return [root_pos]

        final_path_waypoints_segments = deque()
        current_pos = goal_reached_pos

        while current_pos != root_pos:
            node_data = rrt_nodes_map.get(current_pos)
            if node_data is None or node_data.parent_pos is None:
                # This case should ideally not be reached if tree is consistent
                # and current_pos is not root_pos yet.
                print("Path reconstruction error: Inconsistent tree or orphaned node.")
                return []

            parent_pos = node_data.parent_pos
            edge = rrt_edges_map.get((parent_pos, current_pos))

            if edge is None or not edge.path:
                print(f"Path reconstruction error: Missing edge or path segment from {parent_pos} to {current_pos}.")
                return []

            final_path_waypoints_segments.appendleft(list(edge.path))
            current_pos = parent_pos

        # Concatenate segments
        full_path = []
        for i, segment in enumerate(final_path_waypoints_segments):
            if not segment: # Should not happen if edge.path was checked
                continue
            if i == 0:
                full_path.extend(segment)
            else:
                if full_path and np.array_equal(full_path[-1], segment[0]):
                    full_path.extend(segment[1:])
                else:
                    full_path.extend(segment)
        return full_path

    @classmethod
    def _rrt_with_dubins(cls, start_node, end_node, obstacles, width, config, precision=(5, 5, 1)):
        """
        RRT 알고리즘을 통해 목적지까지의 도로 너비 고려한 장애물 회피 경로 생성
        RRT 알고리즘에 Dubins steer 적용, 비홀로노믹 제약 고려
        RRT 알고리즘의 트리 노드들은 Node 클래스 사용하지 않고 내부적으로 노드 처리
        """

        nodes = {}
        edges = {}
        local_planner = Dubins(radius=config["min_radius"], point_separation=config["point_separation"])
        goal = (0, 0, 0)
        root = (0, 0, 0)

        class RRTNode:
            def __init__(self, position, time, cost, parent_pos=None):
                self.destination_list = []
                self.position = position
                self.time = time
                self.cost = cost
                self.parent_pos = parent_pos

        class Edge:
            def __init__(self, node_from, node_to, path, cost):
                self.node_from = node_from
                self.node_to = node_to
                self.path = deque(path)
                self.cost = cost

        start = (start_node.x, start_node.y, start_node.yaw)
        goal = (end_node.x, end_node.y, end_node.yaw)
        root = start
        collision_dist = width / 2.0

        is_already_at_goal = True
        for i_dim in range(len(goal)):
            if abs(goal[i_dim] - start[i_dim]) > precision[i_dim] / 2.0:
                is_already_at_goal = False
                break
        if is_already_at_goal:
            direct_options = local_planner.all_options(start, goal)
            if direct_options:
                valid_options = [opt for opt in direct_options if opt[0] != float('inf')]
                if valid_options:
                    best_option_cost, best_option_type, best_option_params = min(valid_options, key=lambda opt: opt[0])
                    direct_path_points = local_planner.generate_points(start, goal, best_option_type, best_option_params)
                    if not cls._is_collision_path(direct_path_points, obstacles, collision_dist):
                        nodes[start] = RRTNode(start, 0, 0, parent_pos=None)
                        if start != goal:
                             nodes[goal] = RRTNode(goal, best_option_cost, best_option_cost, parent_pos=start)
                             edges[(start, goal)] = Edge(start, goal, direct_path_points, best_option_cost)
                             return cls._reconstruct_path(root, goal, nodes, edges)
                        else:
                             return [start]
            return [start]

        nodes[start] = RRTNode(start, 0, 0, parent_pos=None)

        for i_iter in range(config["max_iterations"]):
            current_iteration_step_size = config["base_step_size"] * random.uniform(config["min_step_factor"], config["max_step_factor"])

            if random.random() < config["goal_sampling_rate"]:
                sample = goal
            else:
                rand_x = random.uniform(min(start[0], goal[0]) - current_iteration_step_size, max(start[0], goal[0]) + current_iteration_step_size)
                rand_y = random.uniform(min(start[1], goal[1]) - current_iteration_step_size, max(start[1], goal[1]) + current_iteration_step_size)
                rand_yaw = random.uniform(-math.pi, math.pi)
                sample = (rand_x, rand_y, rand_yaw)

            nearest_node_pos = None
            min_dist_sq = float('inf')
            for existing_node_pos in nodes:
                dist_sq = (sample[0] - existing_node_pos[0])**2 + (sample[1] - existing_node_pos[1])**2 # Simple Euclidean for nearest
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    nearest_node_pos = existing_node_pos

            if nearest_node_pos is None: continue

            dubins_options_from_nearest = local_planner.all_options(nearest_node_pos, sample)

            for dubins_opt_data in dubins_options_from_nearest:
                if dubins_opt_data[0] == float('inf'):
                    continue

                path_segment_points = local_planner.generate_points(nearest_node_pos,
                                                                    sample,
                                                                    dubins_opt_data[1],
                                                                    dubins_opt_data[2])

                if cls._is_collision_path(path_segment_points, obstacles, collision_dist):
                    continue

                parent_rrt_node = nodes[nearest_node_pos]
                new_node_time = parent_rrt_node.time + dubins_opt_data[0]
                new_node_cost = parent_rrt_node.cost + dubins_opt_data[0]

                if sample not in nodes or new_node_cost < nodes[sample].cost:
                    nodes[sample] = RRTNode(sample, new_node_time, new_node_cost, parent_pos=nearest_node_pos)
                    edges[(nearest_node_pos, sample)] = Edge(nearest_node_pos, sample, path_segment_points, dubins_opt_data[0])

                    is_goal_reached = True
                    for i_dim, val_dim in enumerate(sample):
                        if abs(goal[i_dim] - val_dim) > precision[i_dim]:
                            is_goal_reached = False
                            break
                    if is_goal_reached:
                        return cls._reconstruct_path(root, sample, nodes, edges)
                break

        return []

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
            t = i / 4  # 0 ~ 1

            # 이차 베지에 곡선 공식: B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
            x = (1-t)**2 * start[0] + 2*(1-t)*t * ((start[0] + end[0]) / 2) + t**2 * end[0]
            y = (1-t)**2 * start[1] + 2*(1-t)*t * ((start[1] + end[1]) / 2) + t**2 * end[1]

            # yaw 계산 (접선의 방향)
            # 접선 벡터 = dB/dt = 2(1-t)(P₁-P₀) + 2t(P₂-P₁)
            dx = 2*(1-t)*(end[0] - start[0]) + 2*t*(end[0] - ((start[0] + end[0]) / 2))
            dy = 2*(1-t)*(end[1] - start[1]) + 2*t*(end[1] - ((start[1] + end[1]) / 2))

            yaw = math.atan2(dy, dx)

            path.append((x, y, yaw))

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

    def get_frenet_d(self, vehicle_position=(0,0,0)):
        """
        가장 가까운 포인트와의 거리 계산(Frenet d), 차량의 상태로 활용
        """
        if not self.links:
            return None, None, False

        # 차량 위치
        x, y, yaw = vehicle_position

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
            return None, None, False

        closest_point  # 가장 가까운 포인트 저장

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

        bool_outside_road = abs(d) > self.config['road_width'] / 2.0

        return d, closest_point, bool_outside_road

    def get_serializable_state(self):
        """
        RoadNetworkManager 객체의 직렬화 상태 반환
        """
        nodes_data = {}
        for node in self.nodes:
            nodes_data[node.id] = node.get_serializable_state()

        links_data = []
        for link in self.links:
            links_data.append(link.get_serializable_state())

        return {
            "nodes": nodes_data,
            "links": links_data,
            "node_id_iter": Node._id_iter
        }

    def load_from_serialized(self, data, obstacles):
        """
        직렬화된 데이터로부터 RoadNetworkManager 객체 상태 복원

        Args:
            data: 직렬화된 데이터 딕셔너리
        """
        # 초기화
        self.reset()

        # 노드 복원
        nodes_dict = {}
        for node_id, node_data in data["nodes"].items():
            node = Node(0, 0)  # 임시 값으로 생성
            node.load_from_serialized(node_data)
            self.nodes.append(node)
            nodes_dict[node.id] = node

        # 노드 ID 이터레이터 복원
        Node._id_iter = data.get("node_id_iter", 0)

        # 링크 복원
        for link_data in data["links"]:
            # 링크의 시작 및 끝 노드 가져오기
            start_id = link_data["start_id"]
            end_id = link_data["end_id"]

            # 이미 복원된 노드들에서 해당 노드들 찾기
            start_node = nodes_dict[start_id]
            end_node = nodes_dict[end_id]

            # 새 링크 생성
            link = Link(start_node, end_node, obstacles, link_data["mode"], self.config)

            self.links.append(link)

    def reset(self):
        """노드와 링크 초기화"""
        self.nodes = []
        self.links = []
        Node._id_iter = 0  # 노드 ID 초기화

    def draw(self, screen, world_to_screen_func, debug=False):
        """네트워크 시각화"""
        # 모든 링크 그리기
        for link in self.links:
            link.draw(screen, world_to_screen_func, debug)

        if debug:
            # 모든 노드 그리기 (링크가 그리지 않은 노드만)
            drawn_nodes = set()
            for link in self.links:
                drawn_nodes.add(link.start)
                drawn_nodes.add(link.end)

            # 링크에 포함되지 않은 노드 그리기
            for node in self.nodes:
                if node not in drawn_nodes:
                    node.draw(screen, world_to_screen_func, debug)
