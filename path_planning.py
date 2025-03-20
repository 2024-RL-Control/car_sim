import numpy as np
from dubins import DubinsPath  # 위에서 만든 DubinsPath 사용
from rrt import RRT  # 기존 RRT 알고리즘 사용

class PathPlanner:
    """RRT + Dubins 기반 글로벌 경로 생성"""
    def __init__(self, start, goal, obstacles, x_range, y_range, min_turn_radius=2.0):
        self.start = start  # (x, y, theta)
        self.goal = goal  # (x, y, theta)
        self.obstacles = obstacles
        self.x_range = x_range
        self.y_range = y_range
        self.min_turn_radius = min_turn_radius

    def smooth_path(self, rrt_path):
        """RRT 경로를 Dubins 곡선으로 변환"""
        if len(rrt_path) < 2:
            return []

        dubins_path = DubinsPath(min_turn_radius=self.min_turn_radius)
        smooth_path = []

        for i in range(len(rrt_path) - 1):
            start = (rrt_path[i][0], rrt_path[i][1])  # (x, y)
            goal = (rrt_path[i + 1][0], rrt_path[i + 1][1])  # (x, y)

            # 시작 각도 및 목표 각도 설정 (연속성을 위해 보정)
            start_theta = np.arctan2(goal[1] - start[1], goal[0] - start[0])
            goal_theta = start_theta  # 직전과 같은 방향 유지

            # Dubins 곡선 생성
            segment = dubins_path.generate_path(start, start_theta, goal, goal_theta)

            # 생성된 Dubins 경로 추가
            smooth_path.extend(segment[:-1])  # 중복 방지

        smooth_path.append(rrt_path[-1])  # 마지막 목표점 추가
        return smooth_path

    def plan(self):
        """RRT 경로 생성 후 Dubins 경로 변환"""
        rrt = RRT(self.start, self.goal, self.obstacles, self.x_range, self.y_range)
        path = rrt.planning()

        if not path:
            print("❌ RRT 경로 탐색 실패")
            return []

        # RRT 경로를 Dubins 경로로 변환
        smoothed_path = self.smooth_path(path)

        print("✅ Dubins 적용 완료! 최적화된 경로 반환")
        return smoothed_path