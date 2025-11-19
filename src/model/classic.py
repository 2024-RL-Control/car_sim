# -*- coding: utf-8 -*-
import numpy as np
import math
from math import pi, cos, sin, atan2, sqrt
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

from .vehicle import VehicleState
from .road import RoadSystemAPI, FrenetState, LinearRoadSegment

def normalize_angle(angle: float) -> float:
    """-pi ~ pi 범위로 정규화 (numpy 지원)"""
    return np.arctan2(np.sin(angle), np.cos(angle))

@dataclass
class Waypoint:
    """궤적 상의 한 점을 나타내는 데이터 클래스"""
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v: float = 0.0
    kappa: float = 0.0  # 곡률

@dataclass
class Trajectory:
    """하나의 후보 궤적을 저장하는 데이터 클래스"""
    global_path: List[Waypoint] = field(default_factory=list)
    frenet_path: List[FrenetState] = field(default_factory=list)
    cost: float = float('inf')
    t_horizon: float = 0.0  # 이 궤적의 총 시간

class QuinticPolynomial:
    """
    5차 다항식 궤적 생성기
    시작 (x, v, a) 와 끝 (x, v, a) 상태를 T 시간 내에 연결합니다.
    """
    def __init__(self, x0, v0, a0, x1, v1, a1, T):
        self.a0 = x0
        self.a1 = v0
        self.a2 = a0 / 2.0

        A = np.array([
            [T**3, T**4, T**5],
            [3 * T**2, 4 * T**3, 5 * T**4],
            [6 * T, 12 * T**2, 20 * T**3]
        ])

        b = np.array([
            x1 - (self.a0 + self.a1 * T + self.a2 * T**2),
            v1 - (self.a1 + 2 * self.a2 * T),
            a1 - (2 * self.a2)
        ])

        try:
            x = np.linalg.solve(A, b)
            self.a3 = x[0]
            self.a4 = x[1]
            self.a5 = x[2]
            self.valid = True
        except np.linalg.LinAlgError:
            print("Warning: Quintic Polynomial solver failed. Using linear ramp.")
            # 비상: 선형 보간 (Jerk 발생)
            self.valid = False
            self.a3 = 0.0
            self.a4 = 0.0
            self.a5 = 0.0

    def calc(self, t):
        """t초에서의 위치 (x) 계산"""
        return self.a0 + self.a1 * t + self.a2 * t**2 + \
               self.a3 * t**3 + self.a4 * t**4 + self.a5 * t**5

    def calc_d(self, t):
        """t초에서의 속도 (v) 계산"""
        return self.a1 + 2 * self.a2 * t + 3 * self.a3 * t**2 + \
               4 * self.a4 * t**3 + 5 * self.a5 * t**4

    def calc_dd(self, t):
        """t초에서의 가속도 (a) 계산"""
        return 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2 + \
               20 * self.a5 * t**3

    def calc_ddd(self, t):
        """t초에서의 저크 (jerk) 계산"""
        return 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t**2

class LatticePlanner:
    """
    Frenet 좌표계에서 다수의 후보 궤적을 생성하는 Lattice Planner (Vectorized)
    """
    def __init__(self, planning_config: Dict):
        # config.yaml에서 로드할 파라미터
        self.T_HORIZONS = planning_config.get('t_horizons', [1.0, 2.0, 3.0])
        self.D_OFFSETS = planning_config.get('d_offsets', [-1.0, 0.0, 1.0])
        self.PATH_RESOLUTION = planning_config.get('path_resolution_dt', 0.2)

    def plan(self, current_frenet: FrenetState, current_segment: LinearRoadSegment, target_vel: float) -> List[Trajectory]:
        """
        후보 궤적들을 계획합니다.
        """
        candidate_trajectories = []

        # 시작 상태 (Frenet)
        s0, s_dot0, s_ddot0 = current_frenet.s, current_frenet.s_dot, 0.0
        d0, d_dot0, d_ddot0 = current_frenet.d, current_frenet.d_dot, 0.0

        for T in self.T_HORIZONS:
            for D in self.D_OFFSETS:
                # 1. 종방향(s) 궤적 생성
                s_end_state = (s0 + target_vel * T, target_vel, 0.0)
                s_poly = QuinticPolynomial(s0, s_dot0, s_ddot0, s_end_state[0], s_end_state[1], s_end_state[2], T)

                # 2. 횡방향(d) 궤적 생성
                d_end_state = (D, 0.0, 0.0)
                d_poly = QuinticPolynomial(d0, d_dot0, d_ddot0, d_end_state[0], d_end_state[1], d_end_state[2], T)

                if not s_poly.valid or not d_poly.valid:
                    continue

                # 3. 궤적 샘플링 및 전역 변환 (벡터화 적용)
                global_path, frenet_path = self._generate_path_vectorized(s_poly, d_poly, T, current_segment)

                # 경로 생성에 성공했을 경우에만 추가
                if global_path is not None:
                    candidate_trajectories.append(Trajectory(
                        global_path=global_path,
                        frenet_path=frenet_path,
                        t_horizon=T
                    ))

        return candidate_trajectories

    def _generate_path_vectorized(self, s_poly, d_poly, T, current_segment):
        """
        NumPy를 이용하여 궤적 포인트들을 한 번에 계산합니다. (최적화됨)
        """
        # 시간 배열 생성 (예: [0.0, 0.2, 0.4 ... T])
        t_samples = np.arange(0.0, T + self.PATH_RESOLUTION, self.PATH_RESOLUTION)

        # 1. 다항식 평가 (NumPy Broadcasting)
        # 입력이 배열이면 출력도 배열로 나옵니다.
        s_values = s_poly.calc(t_samples)
        s_dot_values = s_poly.calc_d(t_samples)
        d_values = d_poly.calc(t_samples)
        d_dot_values = d_poly.calc_d(t_samples)

        # 분모 0 방지 (최소 속도 클리핑)
        s_dot_values = np.maximum(s_dot_values, 0.1)

        # 2. 도로 참조점 조회 (List Comprehension)
        # RoadSystemAPI가 스칼라 입력만 받는다고 가정할 때, 이곳은 Python 루프를 돕니다.
        # (만약 evaluate_at_arc_length가 numpy array를 지원한다면 이 부분도 더 빨라질 수 있습니다)
        try:
            ref_points = np.array([current_segment.evaluate_at_arc_length(s) for s in s_values])
            ref_kappas = np.array([current_segment.get_curvature_at_s(s) for s in s_values])

            # ref_points shape: (N, 3) -> x, y, yaw
            ref_x = ref_points[:, 0]
            ref_y = ref_points[:, 1]
            ref_yaw = ref_points[:, 2]
        except Exception as e:
            # 도로 범위를 벗어나는 등 에러 발생 시 해당 궤적 폐기
            return None, None

        # 3. Frenet -> Global 좌표 변환 (전체 배열 연산)
        # math.sin 대신 np.sin을 사용해야 배열 연산이 가능합니다.
        x = ref_x - d_values * np.sin(ref_yaw)
        y = ref_y + d_values * np.cos(ref_yaw)

        # Yaw 및 속도 계산을 위한 중간 변수
        d_prime = d_dot_values / s_dot_values
        one_minus_kappa_d = 1.0 - ref_kappas * d_values

        # 유효성 검사: 분모가 0에 가까운지 확인 (Singularity check)
        if np.any(np.abs(one_minus_kappa_d) < 1e-6):
            return None, None

        # Path Yaw 계산 (vectorized)
        # np.arctan2 사용
        yaw_diff = np.arctan2(d_prime, one_minus_kappa_d)
        path_yaw = ref_yaw + yaw_diff

        # 정규화 (-pi ~ pi)
        path_yaw = np.arctan2(np.sin(path_yaw), np.cos(path_yaw))

        # Path Velocity 계산
        path_vel = s_dot_values * (one_minus_kappa_d / np.cos(path_yaw - ref_yaw))

        # 4. 결과 패키징 (Waypoint 객체 리스트로 변환)
        # 최종 결과는 다른 모듈과의 호환성을 위해 객체 리스트로 반환합니다.
        global_path = [
            Waypoint(x=xi, y=yi, yaw=yawi, v=vi, kappa=ki)
            for xi, yi, yawi, vi, ki in zip(x, y, path_yaw, path_vel, ref_kappas)
        ]

        frenet_path = [
            FrenetState(s=si, d=di, s_dot=sdi, d_dot=ddi)
            for si, di, sdi, ddi in zip(s_values, d_values, s_dot_values, d_dot_values)
        ]

        return global_path, frenet_path

class CostFunction:
    """
    Lattice 궤적을 평가하여 최적의 궤적을 선택합니다.
    """
    def __init__(self, planning_config: Dict, vehicle_config: Dict):
        # 비용 가중치
        self.W_SAFETY = planning_config.get('w_safety', 100.0)
        self.W_COMFORT = planning_config.get('w_comfort', 1.0)
        self.W_EFFICIENCY = planning_config.get('w_efficiency', 5.0)
        self.W_LANE = planning_config.get('w_lane', 2.0)

        # 안전 거리
        self.VEHICLE_RADIUS = vehicle_config.get('length', 1.935) / 2.0 + 0.1 # 0.1m 여유

    def evaluate(self, trajectories: List[Trajectory], target_vel: float, obstacles: np.ndarray) -> Optional[Trajectory]:
        """
        후보 궤적 리스트를 평가하고 최적의 궤적을 반환합니다.

        Args:
            trajectories: 평가할 궤적 리스트
            target_vel: 목표 속도
            obstacles: 장애물 리스트 [(x, y, radius), ...]

        Returns:
            Optional[Trajectory]: 선택된 최적 궤적 (안전한 궤적이 없으면 None)
        """
        best_traj = None
        min_cost = float('inf')

        for traj in trajectories:
            # 1. 안전 비용 (충돌)
            cost_safety = self._cost_safety(traj, obstacles)
            if cost_safety == float('inf'):
                traj.cost = float('inf')
                continue # 이 궤적은 즉시 폐기

            # 2. 효율성 비용 (목표 속도)
            cost_efficiency = self._cost_efficiency(traj, target_vel)

            # 3. 차선 유지 비용 (횡방향 오프셋)
            cost_lane = self._cost_lane(traj)

            # 4. 편안함 비용 (Jerk) - (여기서는 단순화를 위해 속도 변화로 대체)
            cost_comfort = self._cost_comfort(traj)

            # 5. 총 비용 계산
            total_cost = (self.W_SAFETY * cost_safety +
                          self.W_EFFICIENCY * cost_efficiency +
                          self.W_LANE * cost_lane +
                          self.W_COMFORT * cost_comfort)

            traj.cost = total_cost

            if total_cost < min_cost:
                min_cost = total_cost
                best_traj = traj

        return best_traj

    def _cost_safety(self, traj: Trajectory, obstacles: np.ndarray) -> float:
        """안전 비용 (장애물 충돌) 계산"""
        if obstacles.size == 0:
            return 0.0

        waypoints_xy = np.array([[wp.x, wp.y] for wp in traj.global_path])
        if waypoints_xy.size == 0:
            return 0.0

        obstacles_xy = obstacles[:, :2]
        obstacles_r = obstacles[:, 2]

        dist_sq = np.sum((waypoints_xy[:, np.newaxis, :] - obstacles_xy[np.newaxis, :, :])**2, axis=2)
        safe_dist_sq = (self.VEHICLE_RADIUS + obstacles_r[np.newaxis, :])**2

        if np.any(dist_sq < safe_dist_sq):
            return float('inf')

        return 0.0

    def _cost_efficiency(self, traj: Trajectory, target_vel: float) -> float:
        """효율성 비용 (목표 속도와의 차이) 계산"""
        final_vel = traj.global_path[-1].v
        cost = (target_vel - final_vel)**2
        return cost

    def _cost_lane(self, traj: Trajectory) -> float:
        """차선 유지 비용 (최종 횡방향 오프셋) 계산"""
        final_d = traj.frenet_path[-1].d
        cost = final_d**2
        return cost

    def _cost_comfort(self, traj: Trajectory) -> float:
        """편안함 비용 (가속도/Jerk) 계산 - 여기서는 속도 변화량으로 근사"""
        vel_changes = np.diff([wp.v for wp in traj.global_path])
        accel_sq_sum = np.sum(vel_changes**2) # 가속도 제곱의 합 (Jerk 근사)
        return accel_sq_sum

class PIDController:
    """종방향 속도 제어를 위한 PID 제어기"""
    def __init__(self, Kp: float, Ki: float, Kd: float, dt: float):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0
        self.integral_max = 5.0 # Anti-windup

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_v = 0.0

    def compute_throttle(self, current_v: float, target_v: float) -> float:
        """
        목표 속도를 추종하기 위한 가속/제동 값을 계산합니다.

        Args:
            current_v: 현재 종방향 속도
            target_v: 목표 종방향 속도

        Returns:
            float: [-1.0 (제동), 1.0 (가속)] 사이의 제어 값
        """
        if self.dt <= 0:
            return 0.0

        error = target_v - current_v

        # 비례 (Proportional)
        p_term = self.Kp * error

        # 적분 (Integral) with Anti-windup
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -self.integral_max, self.integral_max)
        i_term = self.Ki * self.integral

        # 미분 (Derivative)
        # derivative = (error - self.prev_error) / self.dt
        derivative = - (current_v - self.prev_v) / self.dt  # 노이즈 감소를 위해 속도 변화로 대체
        d_term = self.Kd * derivative

        # Update
        self.prev_error = error
        self.prev_v = current_v

        # 최종 출력
        output = p_term + i_term + d_term
        return np.clip(output, -1.0, 1.0)


class StanleyController:
    """횡방향 궤적 추종을 위한 Stanley 제어기"""
    def __init__(self, k_e: float, k_s: float):
        self.k_e = k_e  # 횡방향 오차 게인
        self.k_s = k_s  # 저속 보정 (Softening) 게인 (0 방지)

    def compute_steer(self, vehicle_state: VehicleState, path: List[Waypoint]) -> float:
        """
        목표 궤적을 추종하기 위한 조향각(radian)을 계산합니다.

        Args:
            vehicle_state: 현재 차량 상태
            path: 추종할 궤적 (Waypoint 리스트)

        Returns:
            float: 조향각 [rad]
        """
        if not path:
            return 0.0

        yaw = vehicle_state.yaw
        v = vehicle_state.vel_long
        half_wb = vehicle_state.half_wheelbase

        # 1. 전방 차축 위치 계산
        x_front = vehicle_state.x + half_wb * cos(yaw)
        y_front = vehicle_state.y + half_wb * sin(yaw)

        # 2. 궤적 상에서 가장 가까운 목표점 탐색 (전방 차축 기준)
        min_dist = float('inf')
        target_idx = 0
        for i, wp in enumerate(path):
            dist_sq = (wp.x - x_front)**2 + (wp.y - y_front)**2
            if dist_sq < min_dist:
                min_dist = dist_sq
                target_idx = i

        target_wp = path[target_idx]

        # 3. 횡방향 오차 (Crosstrack Error, e_fa) 계산
        # 경로 법선 벡터(Normal)와 전방 차축 위치 벡터의 내적
        path_normal_x = -sin(target_wp.yaw)
        path_normal_y = cos(target_wp.yaw)
        vec_axle_to_path_x = target_wp.x - x_front
        vec_axle_to_path_y = target_wp.y - y_front

        # e_fa: 경로 기준 차량이 왼쪽에 있다면 -, 오른쪽이라면 +
        e_fa = (vec_axle_to_path_x * path_normal_x + vec_axle_to_path_y * path_normal_y)

        # 4. 방향 오차 (Heading Error, theta_e) 계산
        theta_e = normalize_angle(target_wp.yaw - yaw)

        # 5. Stanley 제어 법칙
        # 횡방향 오차 보정항
        theta_crosstrack = atan2(self.k_e * e_fa, v + self.k_s)

        # 최종 조향각
        steer_angle_rad = normalize_angle(theta_e + theta_crosstrack)

        return -steer_angle_rad


class VehicleController:
    """Stanley(횡) 및 PID(종) 제어기를 통합 관리합니다."""

    def __init__(self, vehicle_config: Dict, control_config: Dict, dt: float):
        self.max_steer_rad = np.radians(vehicle_config.get('max_steer', 35.0))

        self.pid = PIDController(
            Kp=control_config.get('pid_kp', 1.0),
            Ki=control_config.get('pid_ki', 0.1),
            Kd=control_config.get('pid_kd', 0.05),
            dt=dt
        )
        self.stanley = StanleyController(
            k_e=control_config.get('stanley_k_e', 0.5),
            k_s=control_config.get('stanley_k_s', 0.1),
        )

    def compute_action(self, vehicle_state: VehicleState, best_trajectory: Trajectory, target_velocity: float) -> List[float]:
        """
        최적 궤적을 기반으로 최종 [throttle, steer] 액션을 계산합니다.

        Args:
            vehicle_state: 현재 차량 상태
            best_trajectory: Lattice Planner가 선택한 최적 궤적
            target_velocity: PID가 추종할 목표 속도

        Returns:
            List[float]: [throttle_brake_cmd, steer_cmd] (각각 -1.0 ~ 1.0)
        """
        path_waypoints = best_trajectory.global_path
        if not path_waypoints:
            return [-1.0, 0.0] # 비상 정지

        # 1. 횡방향(Stanley) 제어
        steer_rad = self.stanley.compute_steer(vehicle_state, path_waypoints)
        steer_cmd = np.clip(steer_rad / self.max_steer_rad, -1.0, 1.0)

        # 2. 종방향(PID) 제어
        throttle_cmd = self.pid.compute_throttle(vehicle_state.vel_long, target_velocity)

        return [throttle_cmd, steer_cmd]


    def reset(self):
        """제어기 상태 초기화 (PID 누적값 등)"""
        self.pid.reset()

class ClassicController:
    """
    클래식 제어 시스템 전체를 총괄하는 메인 클래스.
    RL 에이전트의 `predict()` 메서드를 대체합니다.
    """
    def __init__(self, road_api: RoadSystemAPI, vehicle_config: Dict, control_config: Dict, planning_config: Dict, dt: float, verbose: int = 1):
        self.road_api = road_api
        self.vehicle_config = vehicle_config
        self.dt = dt
        self.verbose = verbose

        # 1. 지역 경로 계획기
        self.lattice_planner = LatticePlanner(planning_config)

        # 2. 궤적 평가기
        self.cost_function = CostFunction(planning_config, vehicle_config)

        # 3. 차량 제어기 (Stanley + PID)
        self.vehicle_controller = VehicleController(vehicle_config, control_config, self.dt)

        self.candidate_trajectory: Optional[List[Trajectory]] = None
        self.best_trajectory: Optional[Trajectory] = None

    def plan_and_control(self, vehicle_state: VehicleState, obstacles: np.ndarray) -> List[float]:
        """
        전체 계획 및 제어 사이클을 실행합니다.

        Args:
            vehicle_state: `vehicle.py`의 `VehicleState` 객체
            obstacles: 장애물 리스트 [(x, y, radius), ...]

        Returns:
            List[float]: [throttle_cmd, steer_cmd]
        """

        # 1. 현재 상태 (Frenet) 및 목표 속도 가져오기
        current_frenet_state = vehicle_state.frenet_state
        current_segment = vehicle_state.closest_segment
        target_vel = vehicle_state.smoothed_target_vel_long

        # 2. Lattice Planner로 후보 궤적 생성
        self.candidate_trajectory = self.lattice_planner.plan(
            current_frenet_state,
            current_segment,
            target_vel
        )

        if not self.candidate_trajectory:
            if self.verbose > 0:
                print("    경고: Lattice planner가 경로 생성을 실패했습니다. 비상 정지.")
            return [-1.0, 0.0] # 경로 생성 실패 시 비상 정지

        # 3. Cost Function으로 최적 궤적 선택
        self.best_trajectory = self.cost_function.evaluate(
            self.candidate_trajectory,
            target_vel,
            obstacles
        )

        if self.best_trajectory is None:
            if self.verbose > 0:
                print("    경고: 안전한 경로를 찾을 수 없습니다. 비상 정지.")
            return [-1.0, 0.0] # 안전한 경로 없음 -> 비상 정지

        # 4. Vehicle Controller로 궤적 추종
        action = self.vehicle_controller.compute_action(
            vehicle_state,
            self.best_trajectory,
            target_vel
        )

        return action

    def get_trajectory(self) -> Optional[Tuple[Trajectory, List[Trajectory]]]:
        """디버깅 및 시각화를 위해 최적 궤적을 반환합니다."""
        return self.best_trajectory, self.candidate_trajectory

    def reset(self):
        """제어기 상태 리셋 (PID 등)"""
        self.vehicle_controller.reset()
