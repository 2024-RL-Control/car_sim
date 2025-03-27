import numpy as np
import random

class Node:
    """RRT 노드 클래스 (방향 추가)"""
    def __init__(self, x, y, theta=0):
        self.x = x
        self.y = y
        self.theta = theta  # 차량의 방향 (라디안 단위)
        self.parent = None

class RRT:
    """RRT 알고리즘을 통한 경로 탐색"""
    def __init__(self, start, goal, obstacles, x_range, y_range, step_size=1.0, max_iter=1000):
        self.start = Node(*start, theta=0)  # 시작점의 방향을 0으로 초기화
        self.goal = Node(*goal, theta=0)    # 목표점도 기본적으로 0도 (필요시 조정)
        self.obstacles = obstacles
        self.x_range = x_range
        self.y_range = y_range
        self.step_size = step_size
        self.max_iter = max_iter
        self.nodes = [self.start]

    def get_random_node(self):
        """랜덤 노드 생성 (목표 지점을 샘플링할 확률 포함)"""
        goal_sample_rate = 10  # 목표 샘플링 확률 (5% 확률로 목표 지점 선택)

        if random.randint(0, 100) < goal_sample_rate:
            node = Node(self.goal.x, self.goal.y, self.goal.theta)
        else:
            x = random.uniform(self.x_range[0], self.x_range[1])
            y = random.uniform(self.y_range[0], self.y_range[1])
            theta = random.uniform(-np.pi, np.pi)  # -π ~ π 사이의 랜덤 방향 추가
            node = Node(x, y, theta)


        return node

    def get_nearest_node(self, random_node):
        """가장 가까운 노드 찾기 (유클리드 거리 기준)"""
        return min(self.nodes, key=lambda node: np.linalg.norm([node.x - random_node.x, node.y - random_node.y]))

    def is_collision(self, node):
        """노드가 장애물과 충돌하는지 검사 (디버깅 기능 추가)"""
        for ox, oy, r in self.obstacles:
            dist = np.sqrt((node.x - ox) ** 2 + (node.y - oy) ** 2)

            if dist < (r + 1.0):  # 안전 마진 3.0 추가
                print(f"🚨 충돌 발생! 노드 위치: ({node.x:.2f}, {node.y:.2f}) 장애물: ({ox:.2f}, {oy:.2f}) 거리: {dist:.2f}")

                # 🔥 충돌 노드 시각화 (빨간색 점)
                if hasattr(self, 'screen'):
                    screen_x, screen_y = self._world_to_screen(node.x, node.y)
                    pygame.draw.circle(self.screen, (255, 0, 0), (screen_x, screen_y), 5)  # 빨간색

                return True  # 충돌 감지

        return False  # 충돌 없음

    def get_new_node(self, nearest_node, random_node):
        """새로운 노드 생성 (목표 방향으로 이동)"""
        theta = np.arctan2(random_node.y - nearest_node.y, random_node.x - nearest_node.x)  # 이동 방향 계산
        new_x = nearest_node.x + self.step_size * np.cos(theta)
        new_y = nearest_node.y + self.step_size * np.sin(theta)
        return Node(new_x, new_y, theta)  # theta 포함

    def planning(self):
        """RRT 알고리즘 실행"""
        for _ in range(self.max_iter):
            rand_node = self.get_random_node()
            nearest_node = self.get_nearest_node(rand_node)
            new_node = self.get_new_node(nearest_node, rand_node)

            if self.is_collision(new_node):
                continue

            new_node.parent = nearest_node
            self.nodes.append(new_node)

            # 목표 근처에 도달하면 종료 (목표 반경 확장)
            if np.linalg.norm([new_node.x - self.goal.x, self.goal.y - new_node.y]) < self.step_size * 500:
                print(f"🎯 목표 도착 성공! ({new_node.x:.2f}, {new_node.y:.2f})")
                return self.reconstruct_path(new_node)


        return []  # 경로 찾기 실패

    def reconstruct_path(self, node):
        """경로 재구성"""
        path = []
        while node is not None:
            path.append((node.x, node.y, node.theta))  # 방향 정보까지 포함
            node = node.parent
        return path[::-1]
