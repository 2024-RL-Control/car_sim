import matplotlib.pyplot as plt
import numpy as np
import random
import time

WIDTH, HEIGHT = 1600, 1200

class Node:
    def __init__(self, x, y, yaw=0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.parent = None

class RRTConnect:
    def __init__(self, start, goal, obstacles, x_range, y_range, step_size=12, max_iter=15000):
        self.start = Node(*start)
        self.goal = Node(*goal)
        self.obstacles = obstacles
        self.x_range = x_range
        self.y_range = y_range
        self.step_size = step_size
        self.max_iter = max_iter
        self.tree_start = [self.start]
        self.tree_goal = [self.goal]

    def get_random_node(self):
        if random.randint(0, 100) < 30:
            return Node(self.goal.x, self.goal.y)
        x = random.uniform(*self.x_range)
        y = random.uniform(*self.y_range)
        return Node(x, y)

    def get_nearest_node(self, tree, target):
        return min(tree, key=lambda node: np.hypot(node.x - target.x, node.y - target.y))

    def is_collision(self, node):
        car_radius = 6
        for ox, oy, r in self.obstacles:
            if np.hypot(node.x - ox, node.y - oy) < r + car_radius:
                return True
        return False

    def check_path_collision(self, node1, node2):
        distance = np.hypot(node2.x - node1.x, node2.y - node1.y)
        steps = int(distance / 2)
        if steps == 0:
            return False
        for i in range(steps + 1):
            t = i / steps
            x = node1.x + t * (node2.x - node1.x)
            y = node1.y + t * (node2.y - node1.y)
            temp_node = Node(x, y)
            if self.is_collision(temp_node):
                return True
        return False

    def extend(self, tree, target):
        nearest = self.get_nearest_node(tree, target)
        angle = np.arctan2(target.y - nearest.y, target.x - nearest.x)
        new_x = nearest.x + self.step_size * np.cos(angle)
        new_y = nearest.y + self.step_size * np.sin(angle)
        new_node = Node(new_x, new_y)
        if self.is_collision(new_node):
            return None
        if self.check_path_collision(nearest, new_node):
            return None
        new_node.parent = nearest
        tree.append(new_node)
        return new_node

    def build_path(self, node_start, node_goal):
        path_start = []
        node = node_start
        while node:
            path_start.append((node.x, node.y, node.yaw))
            node = node.parent
        path_goal = []
        node = node_goal
        while node:
            path_goal.append((node.x, node.y, node.yaw))
            node = node.parent
        return path_start[::-1] + path_goal

    def planning(self):
        for _ in range(self.max_iter):
            rand_node = self.get_random_node()
            node_new_start = self.extend(self.tree_start, rand_node)
            if node_new_start is None:
                continue
            node_new_goal = self.extend(self.tree_goal, node_new_start)
            if node_new_goal is not None:
                if not self.check_path_collision(node_new_start, node_new_goal):
                    return self.build_path(node_new_start, node_new_goal)
            self.tree_start, self.tree_goal = self.tree_goal, self.tree_start
        return []

class DubinsPath:
    def __init__(self, min_turn_radius=10, step_size=0.5):
        self.min_turn_radius = min_turn_radius
        self.step_size = step_size

    def generate_dubins_path(self, path):
        smooth_path = []
        for i in range(len(path) - 1):
            smooth_path.extend(self.dubins_segment(path[i], path[i + 1]))
        return smooth_path

    def dubins_segment(self, start, end):
        x1, y1, _ = start
        x2, y2, _ = end
        points = [(x1, y1)]
        dx = x2 - x1
        dy = y2 - y1
        distance = np.hypot(dx, dy)
        steps = max(1, int(distance / self.step_size))
        for i in range(1, steps):
            x = x1 + dx * i / steps
            y = y1 + dy * i / steps
            points.append((x, y))
        return points

def prune_path(path, obstacles):
    def is_collision_free(p1, p2):
        x1, y1 = p1[:2]
        x2, y2 = p2[:2]
        steps = int(np.hypot(x2 - x1, y2 - y1) / 2)
        for i in range(steps + 1):
            t = i / steps
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            for ox, oy, r in obstacles:
                if np.hypot(x - ox, y - oy) < r + 10:
                    return False
        return True

    if not path:
        return []

    pruned = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if is_collision_free(path[i], path[j]):
                break
            j -= 1
        pruned.append(path[j])
        i = j
    return pruned

def visualize_path(start, goal, obstacle_list):
    obstacles = [tuple(o) for o in obstacle_list]

    rrt = RRTConnect(
        start=start,
        goal=goal,
        obstacles=obstacles,
        x_range=(0, WIDTH),
        y_range=(0, HEIGHT),
        step_size=12,
        max_iter=15000
    )
    path = rrt.planning()
    if path:
        pruned = prune_path(path, obstacles)
        dubins = DubinsPath(step_size=0.5)
        smooth = dubins.generate_dubins_path(pruned)

        plt.figure(figsize=(12, 9))
        ax = plt.gca()
        ax.set_xlim(0, WIDTH)
        ax.set_ylim(0, HEIGHT)

        for ox, oy, r in obstacles:
            ax.add_patch(plt.Circle((ox, oy), r, color='red', alpha=0.5))

        plt.plot(start[0], start[1], 'bo', markersize=10, label="Start")
        plt.plot(goal[0], goal[1], 'go', markersize=10, label="Goal")

        xs, ys = zip(*smooth)
        plt.plot(xs, ys, 'g-', linewidth=2, label="Dubins Path")

        plt.legend()
        plt.grid(True)
        plt.title("RRT-Connect with Dubins Path")
        plt.show()
    else:
        print("❌ 경로를 찾지 못했습니다.")

if __name__ == "__main__":
    # 사용 예시
    start = (100, 1100, 0)
    goal = (1500, 50, 0)
    obstacles = [
        [400, 800, 30],
        [800, 600, 30],
        [1200, 400, 30],
        [900, 900, 40],
        [600, 200, 30]
    ]
    visualize_path(start, goal, obstacles)