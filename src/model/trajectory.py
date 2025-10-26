# -*- coding: utf-8 -*-
import numpy as np
from math import cos, sin, atan, pi, log, e, exp
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

    def _encoding_angle(self, angle):
        """cos/sin 인코딩을 통한 각도 표현"""
        return cos(angle), sin(angle)

    def _normalize_angle(self, angle):
        """[-π, π] 범위로 각도 정규화"""
        return (angle + pi) % (2 * pi) - pi

    def _scale_distance(self, distance):
        """거리 정규화"""
        # return 1.0 / log(e + distance/5)
        return 1 - exp(-(distance / 30.0))

    def get_data(self, x, y, yaw):
        dx = x - self.x
        dy = y - self.y
        distance = (dx**2 + dy**2) ** 0.5
        yaw_diff = self._normalize_angle(yaw - self.yaw)

        # norm_distance = self._scale_distance(distance)
        cos_diff, sin_diff = self._encoding_angle(yaw_diff)
        data = np.array([
            cos_diff,
            sin_diff
        ])
        return data

class TrajectoryPredictor:
    """차량 궤적 예측기"""

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
        from .physics import PhysicsEngine

        # 결과를 저장할 궤적 리스트
        trajectory_data_list = []

        # state_history가 충분히 있는지 확인
        if len(state.state_history) < 2:
            # 이력이 부족한 경우 간단한 유지 궤적으로 대체
            return cls._create_simple_continuation_trajectory(state, time_horizon, dt)

        # 초기 상태 복사 및 시뮬레이션용 임시 상태 복제
        sim_state = state.clone()

        # 시뮬레이션을 위한 제어 입력 패턴 추정 (최근 state_history 사용)
        control_patterns = cls._estimate_control_pattern_from_history(
            state.state_history,
            time_horizon,
            dt
        )

        # 첫 번째 포인트는 현재 상태
        init_trajectory_data = TrajectoryData(
            t=0.0,
            x=state.rear_axle_x,
            y=state.rear_axle_y,
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

            # 물리 엔진으로 상태 업데이트, clone된 상태 사용
            PhysicsEngine.update_single_step(
                sim_state, action, dt, physics_config, vehicle_config, predict=True
            )

            # 새 궤적 포인트 생성 및 추가
            new_trajectory_data = TrajectoryData(
                t=t,
                x=sim_state.rear_axle_x,
                y=sim_state.rear_axle_y,
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
            x=state.rear_axle_x,
            y=state.rear_axle_y,
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
            return [[0.0, 0.0]] * num_steps

        num_steps = int(time_horizon / dt)
        if num_steps == 0:
            return []

        # 최근 상태 이력에서 제어 입력 추출 (가장 최근 30개 데이터 사용)
        recent_history_slice = list(state_history)[-min(30, len(state_history)):]

        # NumPy arrays for easier computation
        recent_throttle_engines = np.array([state['throttle_engine'] for state in recent_history_slice])
        recent_throttle_brakes = np.array([state['throttle_brake'] for state in recent_history_slice])
        recent_steers = np.array([state['steer'] for state in recent_history_slice])

        # 평균 제어 입력 계산
        avg_throttle_engine = np.mean(recent_throttle_engines)
        avg_throttle_brake = np.mean(recent_throttle_brakes)
        avg_steer = np.mean(recent_steers)

        # 변화 추세 계산 (선형 추세)
        throttle_engine_trend = 0.0
        throttle_brake_trend = 0.0
        steer_trend = 0.0

        if len(recent_history_slice) >= 3:
            # 간단한 추세 계산 (최근 3개 포인트 사용)
            throttle_engine_trend = (recent_throttle_engines[-1] - recent_throttle_engines[-3]) / 2.0
            throttle_brake_trend = (recent_throttle_brakes[-1] - recent_throttle_brakes[-3]) / 2.0
            steer_trend = (recent_steers[-1] - recent_steers[-3]) / 2.0

        # 제어 입력 패턴을 시간 범위 동안 확장 (Vectorized)
        time_indices = np.arange(num_steps)  # 0, 1, ..., num_steps-1
        trend_factors = np.minimum(time_indices * dt, 1.0)  # 시간에 따른 추세 반영 비율

        # 추세를 반영한 제어 입력 계산 (raw values)
        throttle_engines_out = avg_throttle_engine + throttle_engine_trend * trend_factors
        throttle_brakes_out = avg_throttle_brake + throttle_brake_trend * trend_factors
        steers_out = avg_steer + steer_trend * trend_factors

        # np.clip 적용
        throttle_engines_out = np.clip(throttle_engines_out, 0.0, 1.0)
        throttle_brakes_out = np.clip(throttle_brakes_out, 0.0, 1.0)
        steers_out = np.clip(steers_out, -1.0, 1.0)

        # 동시에 가속과 제동은 불가능 조건 적용
        # 조건 로직에서 이미 수정된 값을 사용하지 않도록 원본 클립된 값의 복사본을 사용합니다.
        temp_engines = throttle_engines_out.copy()
        temp_brakes = throttle_brakes_out.copy()

        conflict_condition = (temp_engines > 0.1) & (temp_brakes > 0.1)

        # 충돌 조건이고 엔진 > 브레이크이면, 브레이크 = 0
        throttle_brakes_out = np.where(
            conflict_condition & (temp_engines > temp_brakes),
            0.0,
            temp_brakes  # 그렇지 않으면 현재 브레이크 값 유지
        )

        # 충돌 조건이고 엔진 <= 브레이크이면, 엔진 = 0
        throttle_engines_out = np.where(
            conflict_condition & (temp_engines <= temp_brakes),
            0.0,
            temp_engines  # 그렇지 않으면 현재 엔진 값 유지
        )

        acceleration_out = np.where(
            throttle_brakes_out > throttle_engines_out,
            -throttle_brakes_out,
            throttle_engines_out
        )

        # 결과 조합: num_steps x 2 배열을 만들고 리스트로 변환
        control_patterns = np.stack((acceleration_out, steers_out), axis=-1).tolist()

        return control_patterns
