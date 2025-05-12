# -*- coding: utf-8 -*-
import numpy as np
from math import cos, sin, tan, atan, pi
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ======================
# Trajectory Predictor
# ======================

@dataclass
class TrajectoryPoint:
    """경로 점 데이터 클래스"""
    t: float = 0.0       # 시간 [s]
    x: float = 0.0       # x 좌표 [m]
    y: float = 0.0       # y 좌표 [m]
    yaw: float = 0.0     # 요각 [rad]
    v_long: float = 0.0  # 종방향 속도 [m/s]
    a_long: float = 0.0  # 종방향 가속도 [m/s²]
    v_lat: float = 0.0   # 횡방향 속도 [m/s]
    a_lat: float = 0.0   # 횡방향 가속도 [m/s²]
    s: float = 0.0       # Frenet s 좌표 [m]
    d: float = 0.0       # Frenet d 좌표 [m]


class TrajectoryPredictor:
    """물리 모델 기반 차량 궤적 예측기"""

    @classmethod
    def predict_polynomial_trajectory(cls,
                                        state,
                                        target_velocity: float,
                                        target_d: float,
                                        time_horizon: float = 5.0,
                                        dt: float = 0.1,
                                        jerk_long: float = 1.0,
                                        jerk_lat: float = 1.0) -> List[TrajectoryPoint]:
        """종/횡 방향 궤적 예측을 결합

        Args:
            state: 차량 상태 객체
            target_velocity: 목표 종방향 속도 [m/s]
            target_d: 목표 횡방향 위치(Frenet d 좌표) [m]
            time_horizon: 궤적 예측 시간 범위 [s]
            dt: 시간 간격 [s]
            jerk_long: 종방향 저크(가속도 변화율) 제한 [m/s³]
            jerk_lat: 횡방향 저크(가속도 변화율) 제한 [m/s³]

        Returns:
            trajectory_points: 예측된 궤적 포인트 리스트
        """
        # 종방향 궤적 예측
        s_trajectory = cls._predict_longitudinal_polynomial_trajectory(
            init_s=0.0,  # 시작점을 0으로 설정
            init_v=state.vel_long,
            init_a=state.acc_long,
            target_v=target_velocity,
            time_horizon=time_horizon,
            dt=dt,
            max_jerk=jerk_long
        )

        # 횡방향 궤적 예측
        d_trajectory = cls._predict_lateral_polynomial_trajectory(
            init_d=state.frenet_d if state.frenet_d is not None else 0.0,
            init_d_dot=state.vel_lat,
            init_d_ddot=state.acc_lat,
            target_d=target_d,
            time_horizon=time_horizon,
            dt=dt,
            max_jerk=jerk_lat
        )

        # 결합된 궤적 생성
        trajectory_points = []

        # 각 시간 단계에 대해 종/횡 방향 궤적을 결합
        for i in range(len(s_trajectory)):
            t = i * dt
            s = s_trajectory[i][0]
            s_dot = s_trajectory[i][1]
            s_ddot = s_trajectory[i][2]
            d = d_trajectory[i][0]
            d_dot = d_trajectory[i][1]
            d_ddot = d_trajectory[i][2]

            # Frenet 좌표를 Cartesian 좌표로 변환 (근사)
            # 시작 위치를 현재 차량 위치로 설정
            if state.frenet_point is not None:
                ref_x, ref_y, ref_yaw = state.frenet_point

                # 현재 reference_path의 방향을 기준으로 s방향 이동
                x = ref_x + s * cos(ref_yaw) - d * sin(ref_yaw)
                y = ref_y + s * sin(ref_yaw) + d * cos(ref_yaw)

                # 경로에 따른 요각 조정 (간단한 근사)
                yaw = ref_yaw + atan(d_dot / max(s_dot, 0.1))  # 0으로 나누기 방지
            else:
                # reference_path가 없는 경우 현재 위치에서 직선 경로 가정
                yaw = state.yaw
                x = state.x + s * cos(yaw) - d * sin(yaw)
                y = state.y + s * sin(yaw) + d * cos(yaw)

            # 종/횡방향 속도를 글로벌 속도로 변환
            v_long = s_dot
            v_lat = d_dot

            # 궤적 포인트 추가
            point = TrajectoryPoint(
                t=t,
                x=x,
                y=y,
                yaw=yaw,
                v_long=v_long,
                a_long=s_ddot,
                v_lat=v_lat,
                a_lat=d_ddot,
                s=s,
                d=d
            )
            trajectory_points.append(point)

        return trajectory_points

    @classmethod
    def _predict_longitudinal_polynomial_trajectory(cls,
                                                    init_s: float,
                                                    init_v: float,
                                                    init_a: float,
                                                    target_v: float,
                                                    time_horizon: float,
                                                    dt: float,
                                                    max_jerk: float = 1.0) -> List[Tuple[float, float, float]]:
        """종방향 궤적 예측

        Args:
            init_s: 초기 종방향 위치 [m]
            init_v: 초기 종방향 속도 [m/s]
            init_a: 초기 종방향 가속도 [m/s²]
            target_v: 목표 종방향 속도 [m/s]
            time_horizon: 궤적 예측 시간 범위 [s]
            dt: 시간 간격 [s]
            max_jerk: 최대 저크(가속도 변화율) [m/s³]

        Returns:
            종방향 궤적 [(s, s_dot, s_ddot), ...]
        """
        # 목표 가속도는 0 (일정 속도에 도달)
        target_a = 0.0

        # 최적 Jerk 프로파일 계산
        jerk_profile = cls._calculate_optimal_jerk(
            init_v, init_a, target_v, target_a, max_jerk, time_horizon
        )

        # 궤적 생성
        trajectory = []
        s = init_s
        v = init_v
        a = init_a
        t = 0.0

        while t < time_horizon:
            trajectory.append((s, v, a))

            # 현재 시점의 jerk 계산
            jerk = cls._get_jerk_at_time(t, jerk_profile)

            # 상태 업데이트 (운동 방정식 적용)
            s = s + v * dt + 0.5 * a * dt**2 + (1/6) * jerk * dt**3
            v = v + a * dt + 0.5 * jerk * dt**2
            a = a + jerk * dt
            t += dt

        return trajectory

    @classmethod
    def _predict_lateral_polynomial_trajectory(cls,
                                                init_d: float,
                                                init_d_dot: float,
                                                init_d_ddot: float,
                                                target_d: float,
                                                time_horizon: float,
                                                dt: float,
                                                max_jerk: float = 1.0) -> List[Tuple[float, float, float]]:
        """횡방향 궤적 예측

        Args:
            init_d: 초기 횡방향 위치 (Frenet d) [m]
            init_d_dot: 초기 횡방향 속도 [m/s]
            init_d_ddot: 초기 횡방향 가속도 [m/s²]
            target_d: 목표 횡방향 위치 [m]
            time_horizon: 궤적 예측 시간 범위 [s]
            dt: 시간 간격 [s]
            max_jerk: 최대 저크(가속도 변화율) [m/s³]

        Returns:
            횡방향 궤적 [(d, d_dot, d_ddot), ...]
        """
        # 목표 횡방향 속도와 가속도는 0 (정지 상태)
        target_d_dot = 0.0
        target_d_ddot = 0.0

        # 경계 조건
        boundary_conditions = {
            'initial': [init_d, init_d_dot, init_d_ddot],
            'final': [target_d, target_d_dot, target_d_ddot]
        }

        # 5차 다항식 계수 생성
        T = time_horizon
        quintic_coeffs = cls._solve_quintic_polynomial(boundary_conditions, T)

        # 궤적 생성
        trajectory = []
        t = 0.0

        while t < time_horizon:
            # 5차 다항식 평가
            d = cls._evaluate_polynomial(quintic_coeffs, t, 0)
            d_dot = cls._evaluate_polynomial(quintic_coeffs, t, 1)
            d_ddot = cls._evaluate_polynomial(quintic_coeffs, t, 2)

            trajectory.append((d, d_dot, d_ddot))
            t += dt

        return trajectory

    @staticmethod
    def _calculate_optimal_jerk(init_v, init_a, target_v, target_a, max_jerk, T):
        """최적의 저크 프로파일 계산

        Args:
            init_v: 초기 속도 [m/s]
            init_a: 초기 가속도 [m/s²]
            target_v: 목표 속도 [m/s]
            target_a: 목표 가속도 [m/s²]
            max_jerk: 최대 저크 [m/s³]
            T: 시간 범위 [s]

        Returns:
            저크 프로파일 [(시작 시간, 끝 시간, 저크 값), ...]
        """
        # 속도 차이
        dv = target_v - init_v

        # 가속도 차이
        da = target_a - init_a

        # 속도 변화를 위한 시간 계산
        if abs(dv) < 0.1 and abs(da) < 0.1:  # 이미 목표에 가까운 경우
            return [(0, T, 0.0)]  # 저크 없음

        # 가속도를 0으로 만들기 위한 시간 계산 (da = jerk * t)
        t_needed = abs(da) / max_jerk

        # 속도 변화를 만들기 위한 시간 계산 (dv = a * t + 0.5 * jerk * t^2)
        # 2차 방정식 해: a*t + 0.5*j*t^2 = dv
        # 여기서 j는 저크, t는 시간

        # 간단한 근사: 가속도를 줄이는 시간과 속도를 조정하는 시간을 분리
        jerk_sign = -1 if da > 0 else 1  # 초기 가속도를 줄이는 방향

        # 초기 jerk 프로파일 (가속도를 0으로 만든다)
        jerk_profile = [(0, min(t_needed, T/3), jerk_sign * max_jerk)]

        # 남은 시간
        remaining_time = T - min(t_needed, T/3)

        if remaining_time > 0 and abs(dv) > 0.1:
            # 속도 조정을 위한 저크 (반대 방향)
            velocity_jerk_sign = 1 if dv > 0 else -1
            jerk_profile.append((T - remaining_time, T, velocity_jerk_sign * max_jerk))

        return jerk_profile

    @staticmethod
    def _get_jerk_at_time(t, jerk_profile):
        """주어진 시간에서의 저크 값을 반환"""
        for start_time, end_time, jerk_value in jerk_profile:
            if start_time <= t < end_time:
                return jerk_value
        return 0.0  # 프로파일 외의 시간에는 저크가 0

    @staticmethod
    def _solve_quintic_polynomial(boundary_conditions, T):
        """경계 조건을 만족하는 5차 다항식 계수 계산

        Args:
            boundary_conditions: 초기 및 최종 위치, 속도, 가속도
            T: 종료 시간

        Returns:
            5차 다항식 계수 [a0, a1, a2, a3, a4, a5]
        """
        # 경계 조건 추출
        p0, v0, a0 = boundary_conditions['initial']
        pT, vT, aT = boundary_conditions['final']

        # 5차 다항식: p(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        # 경계 조건 설정:
        # p(0) = p0, p'(0) = v0, p''(0) = a0
        # p(T) = pT, p'(T) = vT, p''(T) = aT

        # 계수 행렬 설정
        A = np.array([
            [1, 0, 0, 0, 0, 0],             # p(0) = a0
            [0, 1, 0, 0, 0, 0],             # p'(0) = a1
            [0, 0, 2, 0, 0, 0],             # p''(0) = 2*a2
            [1, T, T**2, T**3, T**4, T**5], # p(T)
            [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4], # p'(T)
            [0, 0, 2, 6*T, 12*T**2, 20*T**3]  # p''(T)
        ])

        # 경계값 벡터
        b = np.array([p0, v0, a0, pT, vT, aT])

        # 선형 방정식 해결
        try:
            coeffs = np.linalg.solve(A, b)
            return coeffs
        except np.linalg.LinAlgError:
            # 역행렬이 존재하지 않는 경우에 대한 처리
            # 시간이 너무 짧거나 경계 조건이 모순되는 경우 발생 가능
            # 간단한 대안으로 최소제곱 해를 제공
            coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            return coeffs

    @staticmethod
    def _evaluate_polynomial(coeffs, t, derivative=0):
        """다항식 및 그 도함수 평가

        Args:
            coeffs: 다항식 계수 [a0, a1, a2, a3, a4, a5]
            t: 평가할 시간
            derivative: 도함수 차수 (0=위치, 1=속도, 2=가속도)

        Returns:
            다항식 평가 결과
        """
        if derivative == 0:  # 위치
            return coeffs[0] + coeffs[1]*t + coeffs[2]*t**2 + coeffs[3]*t**3 + coeffs[4]*t**4 + coeffs[5]*t**5
        elif derivative == 1:  # 속도
            return coeffs[1] + 2*coeffs[2]*t + 3*coeffs[3]*t**2 + 4*coeffs[4]*t**3 + 5*coeffs[5]*t**4
        elif derivative == 2:  # 가속도
            return 2*coeffs[2] + 6*coeffs[3]*t + 12*coeffs[4]*t**2 + 20*coeffs[5]*t**3
        else:
            raise ValueError(f"도함수 차수 {derivative}는 지원되지 않습니다.")

    @classmethod
    def predict_physics_based_trajectory(cls,
                                        state,
                                        target_velocity: float,
                                        target_d: float,
                                        time_horizon: float = 5.0,
                                        dt: float = 0.1,
                                        physics_config=None,
                                        vehicle_config=None) -> List[TrajectoryPoint]:
        """
        물리 모델 기반 차량 궤적 예측
        state_history의 패턴을 분석하여 물리 모델을 통한 궤적 예측 수행

        Args:
            state: 차량 상태 객체 (state_history 포함)
            target_velocity: 목표 종방향 속도 [m/s]
            target_d: 목표 횡방향 위치(Frenet d 좌표) [m]
            time_horizon: 궤적 예측 시간 범위 [s]
            dt: 시간 간격 [s]
            physics_config: 물리 설정, None이면 state_history 기반으로 추정
            vehicle_config: 차량 설정, None이면 state_history 기반으로 추정

        Returns:
            trajectory_points: 예측된 궤적 포인트 리스트
        """
        from copy import deepcopy
        from .physics import PhysicsEngine

        # 결과를 저장할 궤적 리스트
        trajectory_points = []

        # state_history가 충분히 있는지 확인
        if not hasattr(state, 'state_history') or len(state.state_history) < 2:
            # 이력이 부족한 경우 다항식 기반 궤적으로 대체
            return cls.predict_polynomial_trajectory(
                state, target_velocity, target_d, time_horizon, dt
            )

        # 물리 파라미터 추정 또는 기본값 사용
        if physics_config is None:
            physics_config = cls._estimate_physics_params(state.state_history)

        if vehicle_config is None:
            vehicle_config = cls._estimate_vehicle_params(state.state_history)

        # 초기 상태 복사 및 시뮬레이션용 임시 상태 생성
        sim_state = deepcopy(state)

        # 시뮬레이션을 위한 제어 입력 패턴 추정 (최근 state_history 사용)
        control_patterns = cls._estimate_control_pattern(
            state.state_history,
            target_velocity,
            target_d,
            time_horizon,
            dt
        )

        # 첫 번째 포인트는 현재 상태
        init_point = TrajectoryPoint(
            t=0.0,
            x=state.x,
            y=state.y,
            yaw=state.yaw,
            v_long=state.vel_long,
            a_long=state.acc_long,
            v_lat=state.vel_lat,
            a_lat=state.acc_lat,
            s=0.0,  # 시작점 기준
            d=state.frenet_d if state.frenet_d is not None else 0.0
        )
        trajectory_points.append(init_point)

        # 각 시간 스텝마다 물리 엔진으로 시뮬레이션
        num_steps = int(time_horizon / dt)
        for i in range(num_steps):
            t = (i + 1) * dt

            # 현재 시간 스텝에 대한 제어 입력 가져오기
            action = control_patterns[min(i, len(control_patterns) - 1)]

            # 물리 엔진으로 상태 업데이트 (deepcopy를 피하기 위해 직접 업데이트)
            PhysicsEngine.update(
                sim_state, action, dt, physics_config, vehicle_config
            )

            # 업데이트된 상태에서 궤적 포인트 생성
            s = 0.0  # Frenet s 좌표는 현재 위치에서의 상대 거리
            if len(trajectory_points) > 0:
                prev_point = trajectory_points[-1]
                dx = sim_state.x - prev_point.x
                dy = sim_state.y - prev_point.y
                s = prev_point.s + np.sqrt(dx**2 + dy**2)  # 이전 포인트로부터의 거리 누적

            # 새 궤적 포인트 생성 및 추가
            new_point = TrajectoryPoint(
                t=t,
                x=sim_state.x,
                y=sim_state.y,
                yaw=sim_state.yaw,
                v_long=sim_state.vel_long,
                a_long=sim_state.acc_long,
                v_lat=sim_state.vel_lat,
                a_lat=sim_state.acc_lat,
                s=s,
                d=sim_state.frenet_d if sim_state.frenet_d is not None else 0.0
            )
            trajectory_points.append(new_point)

        return trajectory_points

    @classmethod
    def _estimate_control_pattern(cls, state_history, target_velocity, target_d, time_horizon, dt):
        """
        state_history를 분석하여 제어 입력 패턴 추정

        Args:
            state_history: 차량 상태 이력
            target_velocity: 목표 속도
            target_d: 목표 횡방향 위치
            time_horizon: 시간 범위
            dt: 시간 간격

        Returns:
            제어 입력 패턴의 리스트 [action1, action2, ...]
            각 action은 [throttle_engine, throttle_brake, steering] 형태
        """
        # 상태 이력이 없으면 빈 패턴 반환
        if len(state_history) < 2:
            num_steps = int(time_horizon / dt)
            return [[0.0, 0.0, 0.0]] * num_steps

        # 패턴을 저장할 리스트
        control_patterns = []
        num_steps = int(time_horizon / dt)

        # 최근 상태 이력에서 제어 입력 추출
        recent_history = list(state_history)[-min(10, len(state_history)):]

        # 목표 속도에 도달하기 위한 스로틀/브레이크 패턴 추정
        last_state = recent_history[-1]
        current_vel = last_state['vel_long']

        # 목표 속도와 현재 속도를 비교하여 제어 패턴 결정
        if target_velocity > current_vel + 0.5:  # 가속 필요
            throttle_engine = min(1.0, 0.5 + (target_velocity - current_vel) * 0.1)
            throttle_brake = 0.0
        elif target_velocity < current_vel - 0.5:  # 감속 필요
            throttle_engine = 0.0
            throttle_brake = min(1.0, 0.3 + (current_vel - target_velocity) * 0.1)
        else:  # 현재 속도 유지
            throttle_engine = 0.3  # 가벼운 가속으로 속도 유지
            throttle_brake = 0.0

        # 최근 조향 패턴 분석
        recent_steers = [state['steer'] for state in recent_history]
        avg_steer_rate = 0.0

        # 조향각 변화율 계산
        if len(recent_steers) >= 2:
            steer_diffs = [recent_steers[i] - recent_steers[i-1] for i in range(1, len(recent_steers))]
            avg_steer_rate = sum(steer_diffs) / len(steer_diffs) if steer_diffs else 0.0

        # 목표 횡방향 위치(d)로 가기 위한 조향각 추정
        current_d = last_state['frenet_d'] if last_state['frenet_d'] is not None else 0.0
        d_diff = target_d - current_d

        # 횡방향 차이에 따른 조향각 추정 (단순 PD 제어)
        steering_p = d_diff * 0.3  # 비례 항
        steering_d = avg_steer_rate * 0.5  # 미분 항
        target_steer = np.clip(steering_p + steering_d, -1.0, 1.0)

        # 기본 제어 입력 패턴 생성
        base_action = [throttle_engine, throttle_brake, target_steer]

        # 제어 입력 패턴을 시간 범위 동안 확장
        for _ in range(num_steps):
            # 여기서는 간단하게 동일한 제어 입력 패턴을 사용
            # 더 정교한 모델은 시간에 따라 패턴을 변화시킬 수 있음
            control_patterns.append(base_action)

        return control_patterns

    @classmethod
    def _estimate_physics_params(cls, state_history):
        """
        상태 이력을 기반으로 물리 파라미터 추정

        Args:
            state_history: 차량 상태 이력

        Returns:
            추정된 물리 파라미터 딕셔너리
        """
        # 기본 물리 설정
        physics_config = {
            'physics_substeps': 5,
            'throttle_response_rate': 3.0,
            'steering_response_rate': 3.0,
            'min_speed_threshold': 0.5,
            'steer_speed_threshold': 10.0,
            'min_steer_speed_factor': 1.0,
            'max_steer_speed_factor': 0.3,
            'air_drag_coefficient': 0.3,
            'rolling_resistance_coefficient': 0.01,
            'cornering_stiffness_front': 7.0,
            'cornering_stiffness_rear': 7.2
        }

        # state_history로부터 차량 반응성 추정
        if len(state_history) >= 3:
            # 최근 데이터 사용
            recent_history = list(state_history)[-min(10, len(state_history)):]

            # 종방향 반응성 추정 (throttle과 가속도 관계)
            throttle_diffs = []
            acc_diffs = []

            for i in range(1, len(recent_history)):
                throttle_diff = recent_history[i]['throttle_engine'] - recent_history[i-1]['throttle_engine']
                acc_diff = recent_history[i]['acc_long'] - recent_history[i-1]['acc_long']

                if abs(throttle_diff) > 0.01:  # 충분한 변화가 있는 경우만 고려
                    throttle_diffs.append(throttle_diff)
                    acc_diffs.append(acc_diff)

            # 충분한 데이터가 있으면 response_rate 조정
            if throttle_diffs and acc_diffs:
                avg_response = sum(acc_diffs) / sum(throttle_diffs) if sum(throttle_diffs) != 0 else 3.0
                # 비정상적인 값 필터링
                if 1.0 <= avg_response <= 10.0:
                    physics_config['throttle_response_rate'] = avg_response

            # 조향 반응성 추정 (steer 변화와 yaw_rate 변화 관계)
            steer_changes = []
            yaw_changes = []

            # 조향 반응성 계산에 필요한 데이터 수집
            for i in range(1, len(recent_history)):
                if 'yaw' in recent_history[i] and 'yaw' in recent_history[i-1]:
                    yaw_change = recent_history[i]['yaw'] - recent_history[i-1]['yaw']
                    steer_change = recent_history[i]['steer'] - recent_history[i-1]['steer']

                    if abs(steer_change) > 0.01:  # 충분한 변화가 있는 경우만 고려
                        steer_changes.append(steer_change)
                        yaw_changes.append(yaw_change)

            # 충분한 데이터가 있으면 cornering_stiffness 조정
            if steer_changes and yaw_changes:
                avg_cornering = sum(yaw_changes) / sum(steer_changes) if sum(steer_changes) != 0 else 7.0
                # 비정상적인 값 필터링
                if 3.0 <= avg_cornering <= 15.0:
                    physics_config['cornering_stiffness_front'] = avg_cornering
                    physics_config['cornering_stiffness_rear'] = avg_cornering + 0.2  # 뒷바퀴 약간 더 크게

        return physics_config

    @classmethod
    def _estimate_vehicle_params(cls, state_history):
        """
        상태 이력을 기반으로 차량 파라미터 추정

        Args:
            state_history: 차량 상태 이력

        Returns:
            추정된 차량 파라미터 딕셔너리
        """
        # 기본 차량 설정
        vehicle_config = {
            'width': 1.8,
            'length': 4.5,
            'wheelbase': 2.8,
            'mass': 1500.0,
            'max_speed': 40.0,
            'min_speed': -20.0,
            'max_steer': 30.0,
            'engine_power': 200.0,
            'brake_power': 100.0
        }

        # state_history로부터 차량 성능 추정
        if len(state_history) >= 3:
            # 최근 데이터 사용
            recent_history = list(state_history)[-min(20, len(state_history)):]

            # 최대 종방향 속도/가속도 분석
            max_vel = max([state['vel_long'] for state in recent_history])
            max_acc = max([state['acc_long'] for state in recent_history])
            min_acc = min([state['acc_long'] for state in recent_history])  # 감속도(제동)

            # 차량 성능 파라미터 조정
            if max_vel > 5.0:
                # 최대 속도 추정 (여유 추가)
                vehicle_config['max_speed'] = max(vehicle_config['max_speed'], max_vel * 1.2)

            if max_acc > 0.5:
                # 엔진 출력 추정
                estimated_power = vehicle_config['mass'] * max_acc
                vehicle_config['engine_power'] = max(vehicle_config['engine_power'], estimated_power * 1.3)

            if min_acc < -0.5:
                # 제동 출력 추정
                estimated_brake = vehicle_config['mass'] * abs(min_acc)
                vehicle_config['brake_power'] = max(vehicle_config['brake_power'], estimated_brake * 1.3)

        return vehicle_config

    @classmethod
    def visualize_trajectory(cls, screen, world_to_screen, trajectory_points, color=(255, 0, 0), width=2):
        """예측 궤적 시각화

        Args:
            screen: Pygame 화면 객체
            world_to_screen: 월드 좌표를 화면 좌표로 변환하는 함수
            trajectory_points: 예측된 궤적 포인트 리스트
            color: 궤적 색상 (RGB)
            width: 선 두께
        """
        import pygame

        if len(trajectory_points) < 2:
            return

        # 궤적 포인트들을 화면 좌표로 변환
        screen_points = [world_to_screen(point.x, point.y) for point in trajectory_points]

        # 라인으로 연결하여 궤적 그리기
        pygame.draw.lines(screen, color, False, screen_points, width)

        # 일부 키 포인트에 마커 표시 (가시성을 위해)
        for i in range(0, len(trajectory_points), 5):  # 5점마다 마커 표시
            if i < len(screen_points):
                pygame.draw.circle(screen, color, screen_points[i], 3)
