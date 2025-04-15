# -*- coding: utf-8 -*-
import numpy as np
from math import radians, degrees, pi, cos, sin, tan, atan

# ======================
# Physics Engine
# ======================
class PhysicsEngine:
    """
    차량 물리 시뮬레이션 엔진

    Ackermann 조향 모델과 기본적인 차량 동역학을 구현한
    2D Top-Down 시점의 물리 엔진
    """

    # 내부 상수
    _MIN_SPEED_THRESHOLD = 0.1  # 정지 상태 판단 기준 속도 [m/s]
    _MIN_TURNING_RADIUS = 6.0   # 최소 회전 반경 [m]
    _THROTTLE_RESPONSE_RATE = 2.0  # 가속 페달 응답률 [1/s]
    _STEERING_RESPONSE_RATE = 2.5  # 조향 응답률 [1/s]
    _MIN_STEER_SPEED_FACTOR = 0.5  # 저속 조향 속도 계수
    _MAX_STEER_SPEED_FACTOR = 0.2  # 고속 조향 속도 계수
    _STEER_SPEED_THRESHOLD = 15.0  # 조향 속도 계수 변화 임계값 [m/s]

    @classmethod
    def update(self, state, action, dt, sim_config, vehicle_config):
        """
        차량 상태 업데이트

        Args:
            state: 차량 상태 객체 (VehicleState)
            action: 제어 입력 배열 [throttle, steering] (-1 ~ 1 범위)
            dt: 시간 간격 [s]
            sim_config: 시뮬레이션 설정 객체 (SimConfig)
            vehicle_config: 차량 설정 객체 (VehicleConfig)
        """
        # 서브스텝 시간 간격 계산
        substep_dt = dt / sim_config.PHYSICS_SUBSTEPS

        # 각 서브스텝마다 물리 업데이트 수행
        for _ in range(sim_config.PHYSICS_SUBSTEPS):
            # 입력 처리
            self._process_inputs(state, action, substep_dt, vehicle_config)

            # 물리 업데이트
            self._update_physics(state, substep_dt, sim_config, vehicle_config)

            # 상태 업데이트
            self._update_vehicle_state(state, substep_dt)

        # 궤적 업데이트
        state.update_trajectory()

    @classmethod
    def _process_inputs(self, state, action, dt, config):
        """
        제어 입력 처리 (가속 및 조향)

        Args:
            state: 차량 상태 객체
            action: 제어 입력 배열 [throttle, steering] (-1 ~ 1 범위)
            dt: 시간 간격 [s]
            config: 차량 설정 객체
        """
        # 액션 배열에서 입력값 추출
        throttle_input = np.clip(action[0], -1.0, 1.0)  # 가속 입력 (-1: 최대 제동, 1: 최대 가속)
        steer_input = np.clip(action[1], -1.0, 1.0)     # 조향 입력 (-1: 좌회전, 1: 우회전)

        # 목표 스로틀 계산 (입력 * 최대 가속도/최대 제동)
        target_throttle = 0.0
        if throttle_input >= 0:
            # 가속 요청
            target_throttle = throttle_input * config.MAX_ACCEL
        else:
            # 제동 요청
            target_throttle = throttle_input * config.MAX_BRAKE

        # 목표 조향각 계산 (입력 * 최대 조향각)
        target_steer = steer_input * config.MAX_STEER

        # 속도에 따른 조향 응답 조정 (고속에서 조향이 느려지도록)
        steer_speed_factor = self._calculate_steer_speed_factor(state.vel_long)

        # 스로틀 응답 지연 (점진적 변화)
        throttle_response_rate = self._THROTTLE_RESPONSE_RATE
        state.throttle += (target_throttle - state.throttle) * throttle_response_rate * dt

        # 조향 응답 지연 (점진적 변화, 속도에 따라 조정)
        steering_response_rate = self._STEERING_RESPONSE_RATE * steer_speed_factor
        state.steer += (target_steer - state.steer) * steering_response_rate * dt

        # 조향각 제한
        state.steer = np.clip(state.steer, -config.MAX_STEER, config.MAX_STEER)

    @classmethod
    def _calculate_steer_speed_factor(self, speed):
        """
        속도에 따른 조향 응답 계수 계산
        고속에서는 조향이 느려지고, 저속에서는 빨라짐

        Args:
            speed: 차량 속도 [m/s]

        Returns:
            조향 응답 계수 (0 ~ 1)
        """
        # 절대 속도 사용 (후진 시에도 동일하게 적용)
        abs_speed = abs(speed)

        if abs_speed < self._MIN_SPEED_THRESHOLD:
            # 정지 상태에 가까우면 최대 응답률
            return self._MIN_STEER_SPEED_FACTOR
        elif abs_speed >= self._STEER_SPEED_THRESHOLD:
            # 고속에서는 최소 응답률
            return self._MAX_STEER_SPEED_FACTOR
        else:
            # 중간 속도에서는 선형 보간
            t = abs_speed / self._STEER_SPEED_THRESHOLD
            return self._MIN_STEER_SPEED_FACTOR * (1-t) + self._MAX_STEER_SPEED_FACTOR * t

    @classmethod
    def _update_physics(self, state, dt, sim_config, vehicle_config):
        """
        차량 물리 업데이트

        Args:
            state: 차량 상태 객체
            dt: 시간 간격 [s]
            sim_config: 시뮬레이션 설정 객체
            vehicle_config: 차량 설정 객체
        """
        # 현재 속도 및 가속도 값
        vel_long = state.vel_long  # 종방향 속도 [m/s]

        # 지형 마찰 계수 적용 (기본값은 1.0)
        friction_coeff = sim_config.get_terrain_friction(state.terrain_type)

        # 가속도 계산 (스로틀 기반)
        acc_long = state.throttle * friction_coeff

        # 저항력 계산
        # 1. 구름 저항 (Cr * m * g)
        rolling_resistance = sim_config.ROLL_RESIST * vehicle_config.MASS * sim_config.GRAVITY
        # 방향에 따라 저항 방향 결정
        if abs(vel_long) > self._MIN_SPEED_THRESHOLD:
            rolling_acc = (rolling_resistance / vehicle_config.MASS) * (-1 if vel_long > 0 else 1)
        else:
            # 정지 상태에 가까우면 속도를 0으로 설정
            if abs(acc_long) < rolling_resistance / vehicle_config.MASS:
                rolling_acc = -vel_long / dt  # 완전히 정지하도록
            else:
                rolling_acc = 0

        # 2. 공기 저항 (0.5 * Cd * A * ρ * v^2)
        # 정면 단면적 (height * width) 대략 계산
        frontal_area = vehicle_config.HEIGHT * vehicle_config.WIDTH * 0.5  # 50% 적용 (단순화)
        drag_force = 0.5 * sim_config.DRAG_COEFF * frontal_area * sim_config.AIR_DENSITY * vel_long**2
        # 방향에 따라 저항 방향 결정
        if abs(vel_long) > self._MIN_SPEED_THRESHOLD:
            drag_acc = (drag_force / vehicle_config.MASS) * (-1 if vel_long > 0 else 1)
        else:
            drag_acc = 0

        # 최종 종방향 가속도
        acc_long = acc_long + rolling_acc + drag_acc

        # 속도 업데이트
        vel_long_new = vel_long + acc_long * dt

        # 속도 제한
        vel_long_new = np.clip(vel_long_new, vehicle_config.MIN_SPEED, vehicle_config.MAX_SPEED)

        # 회전 반경 및 각속도 계산
        yaw_rate = 0.0
        acc_lat = 0.0

        # 조향각이 있는 경우에만 회전 계산
        if abs(state.steer) > 0.001 and abs(vel_long_new) > self._MIN_SPEED_THRESHOLD:
            # Ackermann 조향 모델: R = L / tan(δ)
            turn_radius = max(self._MIN_TURNING_RADIUS, vehicle_config.WHEELBASE / tan(abs(state.steer)))

            # 각속도 = 선속도 / 회전반경
            yaw_rate = vel_long_new / turn_radius

            # 조향 방향에 따라 각속도 방향 결정
            if state.steer > 0:
                yaw_rate = -yaw_rate

            # 횡방향 가속도 = 원심력 / 질량 = v^2 / r
            acc_lat = (vel_long_new**2) / turn_radius

            # 각속도 방향에 따라 횡방향 가속도 방향 결정
            if yaw_rate > 0:
                acc_lat = -acc_lat

        # G-force 계산 (가속도를 g 단위로 변환)
        g_forces = [
            -acc_long / sim_config.GRAVITY,  # 종방향 G-force
            -acc_lat / sim_config.GRAVITY    # 횡방향 G-force
        ]

        # 상태 업데이트
        state.acc_long = acc_long
        state.acc_lat = acc_lat
        state.vel_long = vel_long_new
        state.yaw_rate = yaw_rate
        state.g_forces = g_forces

    @classmethod
    def _update_vehicle_state(self, state, dt):
        """
        차량 위치 및 방향 업데이트

        Args:
            state: 차량 상태 객체
            dt: 시간 간격 [s]
        """
        # 요각 업데이트, -π ~ π 범위로 정규화
        state.yaw = state.normalize_angle(state.yaw + state.yaw_rate * dt)

        # 현재 차량 속도가 충분히 있는 경우에만 위치 업데이트
        if abs(state.vel_long) > self._MIN_SPEED_THRESHOLD:
            # 차량 헤딩 방향으로 속도 벡터 계산
            vel_x = state.vel_long * cos(state.yaw)
            vel_y = state.vel_long * sin(state.yaw)

            # 횡방향 속도 계산 (차량 좌표계 기준)
            vel_lat = state.vel_long * tan(state.steer) / 2

            # 위치 업데이트
            state.x = state.x + vel_x * dt
            state.y = state.y + vel_y * dt

            # 횡방향 속도 상태 업데이트
            state.vel_lat = vel_lat
        else:
            # 정지 상태 또는 매우 저속일 경우 횡방향 속도를 0으로 설정
            state.vel_lat = 0.0
