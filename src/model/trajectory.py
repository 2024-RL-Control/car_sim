# -*- coding: utf-8 -*-
import numpy as np
from math import cos, sin, tan, atan, pi
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ======================
# Trajectory Predictor
# ======================

@dataclass
class TrajectoryData:
    """궤적 데이터 클래스"""
    t: float = 0.0       # 시간 [s]
    x: float = 0.0       # x 좌표 [m]
    y: float = 0.0       # y 좌표 [m]
    yaw: float = 0.0     # 요각 [rad]
    v_long: float = 0.0  # 종방향 속도 [m/s]
    a_long: float = 0.0  # 종방향 가속도 [m/s²]
    v_lat: float = 0.0   # 횡방향 속도 [m/s]
    a_lat: float = 0.0   # 횡방향 가속도 [m/s²]
    s: Optional[float] = None       # Frenet s 좌표 [m]
    d: Optional[float] = None       # Frenet d 좌표 [m]

class TrajectoryPredictor:
    """차량 궤적 예측기"""

    @classmethod
    def predict_polynomial_trajectory(cls,
                                        state,
                                        target_velocity: float,
                                        target_d: float,
                                        time_horizon: float = 5.0,
                                        dt: float = 0.1,
                                        jerk_long: float = 1.0,
                                        jerk_lat: float = 1.0) -> List[TrajectoryData]:
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
            trajectory_data_list: 예측된 궤적 데이터 리스트
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
        trajectory_data_list = []

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
            trajectory_data = TrajectoryData(
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
            trajectory_data_list.append(trajectory_data)

        return trajectory_data_list

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
                                        time_horizon: float = 5.0,
                                        dt: float = 0.1,
                                        physics_config=None,
                                        vehicle_config=None) -> List[TrajectoryData]:
        """
        물리 모델 기반 차량 궤적 예측
        state_history만을 사용하여 물리 모델을 통한 궤적 예측 수행

        Args:
            state: 차량 상태 객체 (state_history 포함)
            time_horizon: 궤적 예측 시간 범위 [s]
            dt: 시간 간격 [s]
            physics_config: 물리 설정, None이면 state_history 기반으로 추정
            vehicle_config: 차량 설정, None이면 state_history 기반으로 추정

        Returns:
            trajectory_data_list: 예측된 궤적 데이터 리스트
        """
        from copy import deepcopy
        from .physics import PhysicsEngine

        # 결과를 저장할 궤적 리스트
        trajectory_data_list = []

        # state_history가 충분히 있는지 확인
        if not hasattr(state, 'state_history') or len(state.state_history) < 2:
            # 이력이 부족한 경우 간단한 유지 궤적으로 대체
            return cls._create_simple_continuation_trajectory(state, time_horizon, dt)

        # 초기 상태 복사 및 시뮬레이션용 임시 상태 생성
        sim_state = deepcopy(state)

        # 시뮬레이션을 위한 제어 입력 패턴 추정 (최근 state_history 사용)
        control_patterns = cls._estimate_control_pattern_from_history(
            state.state_history,
            time_horizon,
            dt
        )

        # 첫 번째 포인트는 현재 상태
        init_trajectory_data = TrajectoryData(
            t=0.0,
            x=state.x,
            y=state.y,
            yaw=state.yaw,
            v_long=state.vel_long,
            a_long=state.acc_long,
            v_lat=state.vel_lat,
            a_lat=state.acc_lat
        )
        trajectory_data_list.append(init_trajectory_data)

        # 각 시간 스텝마다 물리 엔진으로 시뮬레이션
        num_steps = int(time_horizon / dt)
        for i in range(num_steps):
            t = (i + 1) * dt

            # 현재 시간 스텝에 대한 제어 입력 가져오기
            action = control_patterns[min(i, len(control_patterns) - 1)]

            # 물리 엔진으로 상태 업데이트, deepcopy된 상태 사용
            PhysicsEngine.update(
                sim_state, action, dt, physics_config, vehicle_config
            )

            # 새 궤적 포인트 생성 및 추가
            new_trajectory_data = TrajectoryData(
                t=t,
                x=sim_state.x,
                y=sim_state.y,
                yaw=sim_state.yaw,
                v_long=sim_state.vel_long,
                a_long=sim_state.acc_long,
                v_lat=sim_state.vel_lat,
                a_lat=sim_state.acc_lat
            )
            trajectory_data_list.append(new_trajectory_data)

        return trajectory_data_list

    @classmethod
    def _create_simple_continuation_trajectory(cls, state, time_horizon, dt):
        """
        state_history가 부족할 때 간단히 현재 속도와 방향을 유지하는 궤적 생성

        Args:
            state: 차량 상태 객체
            time_horizon: 궤적 예측 시간 범위 [s]
            dt: 시간 간격 [s]

        Returns:
            간단한 궤적 데이터 리스트
        """
        trajectory_data_list = []

        # 첫 번째 포인트는 현재 상태
        trajectory_data_list.append(TrajectoryData(
            t=0.0,
            x=state.x,
            y=state.y,
            yaw=state.yaw,
            v_long=state.vel_long,
            a_long=0.0,  # 가속도는 0으로 가정
            v_lat=state.vel_lat,
            a_lat=0.0
        ))

        # 현재 속도와 방향으로 직진하는 궤적 생성
        num_steps = int(time_horizon / dt)
        for i in range(num_steps):
            t = (i + 1) * dt

            # 이전 포인트
            prev_trajectory_data = trajectory_data_list[-1]

            # 단순 직선 이동
            distance = prev_trajectory_data.v_long * dt
            dx = distance * np.cos(prev_trajectory_data.yaw)
            dy = distance * np.sin(prev_trajectory_data.yaw)

            new_trajectory_data = TrajectoryData(
                t=t,
                x=prev_trajectory_data.x + dx,
                y=prev_trajectory_data.y + dy,
                yaw=prev_trajectory_data.yaw,
                v_long=prev_trajectory_data.v_long,
                a_long=0.0,
                v_lat=0.0,
                a_lat=0.0
            )
            trajectory_data_list.append(new_trajectory_data)

        return trajectory_data_list

    @classmethod
    def _estimate_control_pattern_from_history(cls, state_history, time_horizon, dt):
        """
        state_history만을 사용하여 제어 입력 패턴 추정

        Args:
            state_history: 차량 상태 이력
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

        # 최근 상태 이력에서 제어 입력 추출 (가장 최근 30개 데이터 사용)
        recent_history = list(state_history)[-min(30, len(state_history)):]

        # 최근 스로틀, 브레이크, 조향 값 추출
        recent_throttle_engine = [state['throttle_engine'] for state in recent_history]
        recent_throttle_brake = [state['throttle_brake'] for state in recent_history]
        recent_steer = [state['steer'] for state in recent_history]

        # 평균 제어 입력 계산
        avg_throttle_engine = sum(recent_throttle_engine) / len(recent_throttle_engine)
        avg_throttle_brake = sum(recent_throttle_brake) / len(recent_throttle_brake)
        avg_steer = sum(recent_steer) / len(recent_steer)

        # 변화 추세 계산 (선형 추세)
        throttle_engine_trend = 0.0
        throttle_brake_trend = 0.0
        steer_trend = 0.0

        if len(recent_history) >= 3:
            # 간단한 추세 계산 (최근 3개 포인트 사용)
            throttle_engine_trend = (recent_throttle_engine[-1] - recent_throttle_engine[-3]) / 2
            throttle_brake_trend = (recent_throttle_brake[-1] - recent_throttle_brake[-3]) / 2
            steer_trend = (recent_steer[-1] - recent_steer[-3]) / 2

        # 제어 입력 패턴을 시간 범위 동안 확장
        for i in range(num_steps):
            # 시간에 따른 추세 반영
            trend_factor = min(i * dt, 1.0)  # 최대 1초까지만 추세 반영

            # 추세를 반영한 제어 입력 계산
            throttle_engine = np.clip(avg_throttle_engine + throttle_engine_trend * trend_factor, 0.0, 1.0)
            throttle_brake = np.clip(avg_throttle_brake + throttle_brake_trend * trend_factor, 0.0, 1.0)
            steer = np.clip(avg_steer + steer_trend * trend_factor, -1.0, 1.0)

            # 동시에 가속과 제동은 불가능
            if throttle_engine > 0.1 and throttle_brake > 0.1:
                if throttle_engine > throttle_brake:
                    throttle_brake = 0.0
                else:
                    throttle_engine = 0.0

            control_patterns.append([throttle_engine, throttle_brake, steer])

        return control_patterns
