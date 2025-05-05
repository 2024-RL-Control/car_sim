# -*- coding: utf-8 -*-
import numpy as np
from math import radians, degrees, pi, cos, sin, tan, atan

# ======================
# Physics Engine Components
# ======================

class InputProcessor:
    """
    차량 제어 입력 처리 클래스
    스로틀, 브레이크, 조향 입력을 처리하고 차량 상태 값을 업데이트
    """

    @classmethod
    def process_inputs(cls, state, action, dt, config, physics_config):
        """
        제어 입력 처리 (가속 및 조향)

        Args:
            state: 차량 상태 객체
            action: 제어 입력 배열 [throttle_engine, throttle_brake, steering] ([0, 1], [0, 1], [-1, 1]) 범위
            dt: 시간 간격 [s]
            config: 차량 설정
            physics_config: 물리 설정
        """
        # 액션 배열에서 입력값 추출
        engine_input = np.clip(action[0], 0.0, 1.0)     # 엔진 입력 (0: 정지, 1: 최대 가속)
        brake_input = np.clip(action[1], 0.0, 1.0)      # 브레이크 입력 (0: 정지, 1: 최대 제동)
        steer_input = np.clip(action[2], -1.0, 1.0)     # 조향 입력 (-1: 좌회전, 1: 우회전)

        # 엔진/브레이크 요청의 정도
        target_throttle_engine = engine_input
        target_throttle_brake  = brake_input

        # 목표 조향각 계산 (입력 * 최대 조향각)
        target_steer = steer_input * np.radians(config['max_steer'])

        # 속도에 따른 조향 응답 조정 (고속에서 조향이 느려지도록)
        steer_speed_factor = cls._calculate_steer_speed_factor(state.vel_long, physics_config)

        # 스로틀 응답 지연 (점진적 변화)
        throttle_response_rate = physics_config['throttle_response_rate']
        state.throttle_engine += (target_throttle_engine - state.throttle_engine) * throttle_response_rate * dt
        state.throttle_brake  += (target_throttle_brake  - state.throttle_brake ) * throttle_response_rate * dt

        # 조향 응답 지연 (점진적 변화, 속도에 따라 조정)
        steering_response_rate = physics_config['steering_response_rate'] * steer_speed_factor
        state.steer += (target_steer - state.steer) * steering_response_rate * dt

        # 조향각 제한
        state.steer = np.clip(state.steer, -np.radians(config['max_steer']), np.radians(config['max_steer']))

    @classmethod
    def _calculate_steer_speed_factor(cls, speed, physics_config):
        """
        속도에 따른 조향 응답 계수 계산
        고속에서는 조향이 느려지고, 저속에서는 빨라짐

        Args:
            speed: 차량 속도 [m/s]
            physics_config: 물리 설정

        Returns:
            조향 응답 계수 (0 ~ 1)
        """
        # 절대 속도 사용 (후진 시에도 동일하게 적용)
        abs_speed = abs(speed)

        if abs_speed < physics_config['min_speed_threshold']:
            # 정지 상태에 가까우면 최대 응답률
            return physics_config['min_steer_speed_factor']
        elif abs_speed >= physics_config['steer_speed_threshold']:
            # 고속에서는 최소 응답률
            return physics_config['max_steer_speed_factor']
        else:
            # 중간 속도에서는 선형 보간
            t = abs_speed / physics_config['steer_speed_threshold']
            return physics_config['min_steer_speed_factor'] * (1-t) + physics_config['max_steer_speed_factor'] * t


class ForceCalculator:
    """
    차량에 작용하는 힘을 계산하는 클래스
    추진력, 저항력, 횡방향 힘 등을 계산
    """

    @classmethod
    def calculate_forces(cls, state, dt, physics_config, vehicle_config):
        """
        차량에 작용하는 힘과 가속도 계산

        Args:
            state: 차량 상태 객체
            dt: 시간 간격 [s]
            physics_config: 물리 설정
            vehicle_config: 차량 설정

        Returns:
            (acc_long, acc_lat, yaw_rate): 종방향 가속도, 횡방향 가속도, 각속도
        """
        # 현재 속도 및 가속도 값
        vel_long = state.vel_long  # 종방향 속도 [m/s]

        # 지형 마찰 계수 적용 (기본값은 1.0)
        friction_coeff = physics_config['terrain_friction'][state.terrain_type]

        # 가속도 계산 (스로틀 기반)
        acc_engine = state.throttle_engine  * vehicle_config['max_accel'] * friction_coeff
        acc_brake = state.throttle_brake * vehicle_config['max_brake'] * friction_coeff
        acc_long = acc_engine - acc_brake

        # 저항력 계산
        rolling_acc = cls._calculate_rolling_resistance(vel_long, state.throttle_engine, vehicle_config, physics_config, dt)
        drag_acc = cls._calculate_drag(vel_long, vehicle_config, physics_config)

        # 최종 종방향 가속도
        acc_long = acc_long + rolling_acc + drag_acc

        # 회전 반경 및 각속도 계산
        yaw_rate, acc_lat = cls._calculate_lateral_dynamics(state.steer, vel_long, physics_config, vehicle_config)

        return acc_long, acc_lat, yaw_rate

    @classmethod
    def _calculate_rolling_resistance(cls, vel_long, throttle_engine, vehicle_config, physics_config, dt):
        """구름 저항 계산"""
        # 구름 저항 (Cr * m * g)
        rolling_resistance = physics_config['roll_resist'] * vehicle_config['mass'] * physics_config['gravity']

        # 방향에 따라 저항 방향 결정
        if abs(vel_long) > physics_config['min_speed_threshold']:
            rolling_acc = (rolling_resistance / vehicle_config['mass']) * (-1 if vel_long > 0 else 1)
        else:
            # 정지 상태에 가까우면 속도를 0으로 설정
            if throttle_engine < rolling_resistance / vehicle_config['mass']:
                rolling_acc = -vel_long / dt  # 완전히 정지하도록
            else:
                rolling_acc = 0

        return rolling_acc

    @classmethod
    def _calculate_drag(cls, vel_long, vehicle_config, physics_config):
        """공기 저항 계산"""
        # 공기 저항 (0.5 * Cd * A * ρ * v^2)
        # 정면 단면적 (height * width) 대략 계산
        frontal_area = vehicle_config['height'] * vehicle_config['width'] * 0.5  # 50% 적용 (단순화)
        drag_force = 0.5 * physics_config['drag_coeff'] * frontal_area * physics_config['air_density'] * vel_long**2

        # 방향에 따라 저항 방향 결정
        if abs(vel_long) > physics_config['min_speed_threshold']:
            drag_acc = (drag_force / vehicle_config['mass']) * (-1 if vel_long > 0 else 1)
        else:
            drag_acc = 0

        return drag_acc

    @classmethod
    def _calculate_lateral_dynamics(cls, steer, vel_long, physics_config, vehicle_config):
        """횡방향 동역학 계산 (회전율 및 횡방향 가속도)"""
        yaw_rate = 0.0
        acc_lat = 0.0

        # 조향각이 있는 경우에만 회전 계산
        if abs(steer) > 0.001 and abs(vel_long) > physics_config['min_speed_threshold']:
            # Ackermann 조향 모델: R = L / tan(δ)
            turn_radius = max(physics_config['min_turning_radius'], vehicle_config['wheelbase'] / tan(abs(steer)))

            # 각속도 = 선속도 / 회전반경
            yaw_rate = vel_long / turn_radius

            # 조향 방향에 따라 각속도 방향 결정
            if steer > 0:
                yaw_rate = -yaw_rate

            # 횡방향 가속도 = 원심력 / 질량 = v^2 / r
            acc_lat = (vel_long**2) / turn_radius

            # 각속도 방향에 따라 횡방향 가속도 방향 결정
            if yaw_rate > 0:
                acc_lat = -acc_lat

        return yaw_rate, acc_lat


class StateUpdater:
    """
    차량 상태 업데이트 클래스
    위치, 속도, 방향 등의 상태 변수를 물리적으로 업데이트
    """

    @classmethod
    def update_vehicle_state(cls, state, acc_long, acc_lat, yaw_rate, dt, vehicle_config, physics_config):
        """
        차량 상태 업데이트

        Args:
            state: 차량 상태 객체
            acc_long: 종방향 가속도 [m/s²]
            acc_lat: 횡방향 가속도 [m/s²]
            yaw_rate: 각속도 [rad/s]
            dt: 시간 간격 [s]
            vehicle_config: 차량 설정
            physics_config: 물리 설정
        """
        # 속도 업데이트
        vel_long_new = np.clip(state.vel_long + acc_long * dt, vehicle_config['min_speed'], vehicle_config['max_speed'])

        # 요각 업데이트, -π ~ π 범위로 정규화
        yaw_new = state.normalize_angle(state.yaw + yaw_rate * dt)

        # 위치 업데이트 (속도가 충분한 경우) - 뒷바퀴 축 기준 자전거 모델
        if abs(vel_long_new) > physics_config['min_speed_threshold']:
            # 뒷바퀴 축은 차량 헤딩 방향으로만 이동 (자전거 모델)
            cos_yaw = cos(yaw_new)
            sin_yaw = sin(yaw_new)
            rear_vel_x = vel_long_new * cos_yaw
            rear_vel_y = vel_long_new * sin_yaw

            # 뒷바퀴 축 위치 업데이트
            new_rear_x = state.rear_axle_x + rear_vel_x * dt
            new_rear_y = state.rear_axle_y + rear_vel_y * dt

            # 뒷바퀴 축 위치 저장
            state.rear_axle_x = new_rear_x
            state.rear_axle_y = new_rear_y

            # 차량 중심점 위치 업데이트 (뒷바퀴 축 기준)
            state.update_position_from_rear_axle(cos_yaw, sin_yaw)

            # 횡방향 속도 계산 (차량 좌표계 기준)
            vel_lat = vel_long_new * tan(state.steer) / 2
            state.vel_lat = vel_lat
        else:
            # 정지 상태 또는 매우 저속일 경우 횡방향 속도를 0으로 설정
            state.vel_lat = 0.0

        # 나머지 상태 업데이트
        state.yaw = yaw_new
        state.vel_long = vel_long_new
        state.acc_long = acc_long
        state.acc_lat = acc_lat
        state.yaw_rate = yaw_rate


# ======================
# Main Physics Engine
# ======================
class PhysicsEngine:
    """
    차량 물리 시뮬레이션 엔진

    Ackermann 조향 모델과 기본적인 차량 동역학을 구현한
    2D Top-Down 시점의 물리 엔진
    """

    @classmethod
    def update(cls, state, action, dt, physics_config, vehicle_config):
        """
        차량 상태 업데이트

        Args:
            state: 차량 상태 객체 (VehicleState)
            action: 제어 입력 배열 [throttle_engine, throttle_brake, steering] ([0, 1], [0, 1], [-1, 1]) 범위
            dt: 시간 간격 [s]
            physics_config: 물리 설정
            vehicle_config: 차량 설정
        """
        # 서브스텝 시간 간격 계산
        substep_dt = dt / physics_config['physics_substeps']

        # 각 서브스텝마다 물리 업데이트 수행
        for _ in range(physics_config['physics_substeps']):
            # 입력 처리
            InputProcessor.process_inputs(state, action, substep_dt, vehicle_config, physics_config)

            # 물리 업데이트 (힘 계산)
            acc_long, acc_lat, yaw_rate = ForceCalculator.calculate_forces(
                state, substep_dt, physics_config, vehicle_config
            )

            # 상태 업데이트
            StateUpdater.update_vehicle_state(
                state, acc_long, acc_lat, yaw_rate, substep_dt, vehicle_config, physics_config
            )

        # 궤적 업데이트
        state.update_trajectory()
