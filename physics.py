# -*- coding: utf-8 -*-
import numpy as np

# ========================
# Physics Model Components
# ========================
class KinematicModel:
    """Ackermann steering geometry 기반 운동학 모델"""

    @staticmethod
    def update(state, dt, accel, steer, config):
        """
        향상된 동역학 모델을 사용한 차량 상태 업데이트
        측면 미끄러짐과 드리프트를 차량 움직임에 반영

        Args:
            state: The current vehicle state
            dt: Time step in seconds
            accel: Acceleration command [-1, 1]
            steer: Steering command [-1, 1]
            config: Vehicle configuration
        """
        # 입력 정규화
        max_steer = config.MAX_STEER
        steer = np.clip(steer, -max_steer, max_steer)

        # 종방향 속도 업데이트
        state.vel += state.accel * dt
        state.vel = np.clip(state.vel, config.MIN_SPEED, config.MAX_SPEED)

        # 드리프트 각도 계산
        if abs(state.vel) > 1.0 or abs(state.vel_lateral) > 0.5:
            state.drift_angle = np.arctan2(state.vel_lateral, state.vel)
        else:
            state.drift_angle = 0.0

        # 속도 벡터 계산 (종방향 및 횡방향 속도 벡터 합성)
        local_vx = state.vel
        local_vy = state.vel_lateral

        # 속도 벡터의 크기
        velocity_magnitude = np.sqrt(local_vx**2 + local_vy**2)

        # 속도 벡터의 전역 좌표계 방향
        velocity_direction = state.yaw + state.drift_angle

        # 전역 좌표계에서의 차량 이동
        state.x += velocity_magnitude * np.cos(velocity_direction) * dt
        state.y += velocity_magnitude * np.sin(velocity_direction) * dt

        # 조향 각도에 따른 요 레이트 계산
        if abs(state.vel) > 0.1:  # 속도가 너무 낮을 때 수치적 불안정성 방지
            # 기본 조향에 의한 요 레이트
            steering_yaw_rate = (state.vel / config.WHEELBASE) * np.tan(steer)

            # 슬립에 의한 요 레이트는 속도에 따라 다름
            # 저속: 슬립이 안정화 효과 (반대 방향 회전)
            # 고속: 슬립이 과조향 효과 (같은 방향 회전)
            critical_speed = 15.0  # 요 영향이 반전되는 임계 속도 [m/s]
            slip_factor_base = 0.6  # 기본 슬립 영향 계수

            # 속도에 따른 슬립 영향 계수 조정
            if abs(state.vel) < critical_speed:
                # 저속: 슬립이 안정화 효과 (반대 방향으로 회전)
                # 저속에서는 슬립과 반대 방향으로 회전하므로 부호 반전
                speed_ratio = abs(state.vel) / critical_speed  # 0~1 사이 값
                slip_factor = -slip_factor_base * (1 - speed_ratio)
            else:
                # 고속: 슬립이 과조향 효과 (같은 방향으로 회전)
                # 속도가 높을수록 과조향 효과 증가
                speed_ratio = min(1.0, (abs(state.vel) - critical_speed) / 15.0)  # 0~1 사이 값
                slip_factor = slip_factor_base * speed_ratio

            # 슬립에 의한 요 레이트 계산
            slip_yaw_rate = slip_factor * state.vel_lateral / config.WHEELBASE

            # 총 요 레이트 계산
            total_yaw_rate = steering_yaw_rate + slip_yaw_rate

            # 방향 업데이트
            state.yaw += total_yaw_rate * dt

        state.yaw = state.normalize_angle(state.yaw)

        # 입력 저장
        state.steer = steer
        state.accel = accel

        # 궤적 업데이트
        state.update_trajectory()


class DynamicModel:
    """차량 동역학 모델 (Magic Formula 기반)"""

    @staticmethod
    def magic_formula(slip_angle, config, terrain_friction=1.0):
        """
        Pacejka Magic Formula 타이어 모델

        Args:
            slip_angle: 타이어 슬립각 [rad]
            config: 차량 설정 (VehicleConfig)
            terrain_friction: 지형 마찰 계수 (기본값 1.0 = 아스팔트)

        Returns:
            타이어 측면력 계수
        """
        B = config.TIRE_B
        C = config.TIRE_C
        D = config.TIRE_D * terrain_friction  # 지형에 따른 최대 마찰력 조정
        E = config.TIRE_E

        return D * np.sin(C * np.arctan(B * slip_angle - E * (B * slip_angle - np.arctan(B * slip_angle))))

    @staticmethod
    def calculate_forces(state, sim_config, vehicle_config):
        """
        차량에 작용하는 힘 계산

        Args:
            state: 차량 상태 (VehicleState)
            sim_config: 시뮬레이션 설정 (SimConfig)
            vehicle_config: 차량 설정 (VehicleConfig)

        Returns:
            aero_drag: 공기 저항 [N]
            rolling_resist: 구름 저항 [N]
            lateral_forces: 각 타이어의 횡방향 힘 [N]
            longitudinal_forces: 각 타이어의 종방향 힘 [N]
        """
        # 지형에 따른 마찰 계수 적용
        terrain_friction = sim_config.TERRAIN_FRICTION.get(state.terrain_type, 1.0)

        # 공기 저항 (차량 전면적 고려)
        frontal_area = vehicle_config.WIDTH * vehicle_config.HEIGHT
        aero_drag = 0.5 * sim_config.AIR_DENSITY * sim_config.DRAG_COEFF * frontal_area * state.vel**2

        # 구름 저항
        rolling_resist = vehicle_config.MASS * sim_config.GRAVITY * sim_config.ROLL_RESIST * np.sign(state.vel)

        # 각 타이어의 수직 하중 계산 (무게 배분, 하중 이동 고려)
        # 간단한 모델: 50:50 정적 무게 배분 가정
        static_load = vehicle_config.MASS * sim_config.GRAVITY / 4

        # 요 레이트 및 횡방향 가속도 계산
        yaw_rate = (state.vel / vehicle_config.WHEELBASE) * np.tan(state.steer) if abs(state.vel) > 0.1 else 0
        lateral_accel = state.vel * yaw_rate

        # 하중 이동 계산
        load_transfer_lateral = (vehicle_config.MASS * lateral_accel * vehicle_config.CG_HEIGHT) / vehicle_config.TRACK
        load_transfer_longitudinal = (vehicle_config.MASS * state.accel * vehicle_config.CG_HEIGHT) / vehicle_config.WHEELBASE

        # 각 타이어 수직 하중 [FR, FL, RR, RL]
        vertical_loads = [
            static_load + load_transfer_longitudinal/2 + load_transfer_lateral/2,  # FR (우측 앞)
            static_load + load_transfer_longitudinal/2 - load_transfer_lateral/2,  # FL (좌측 앞)
            static_load - load_transfer_longitudinal/2 + load_transfer_lateral/2,  # RR (우측 뒤)
            static_load - load_transfer_longitudinal/2 - load_transfer_lateral/2   # RL (좌측 뒤)
        ]

        # 각 타이어 슬립각 계산 (아커만 지오메트리 기반)
        wheelbase = vehicle_config.WHEELBASE
        track = vehicle_config.TRACK
        steer = state.steer

        if abs(steer) > 0.001:
            # 회전 반경 계산
            R = wheelbase / np.tan(abs(steer))

            # 부호 규약: 양수=우회전, 음수=좌회전
            if steer > 0:  # 우회전
                alpha_fr = np.arctan(wheelbase / (R - track/2)) - steer  # 안쪽 (우측 앞)
                alpha_fl = np.arctan(wheelbase / (R + track/2)) - steer  # 바깥쪽 (좌측 앞)
            else:  # 좌회전
                alpha_fl = -np.arctan(wheelbase / (R - track/2)) - steer  # 안쪽 (좌측 앞)
                alpha_fr = -np.arctan(wheelbase / (R + track/2)) - steer  # 바깥쪽 (우측 앞)
        else:
            alpha_fl = alpha_fr = 0.0

        # 후륜 슬립각 (후륜 조향이 없을 경우)
        alpha_rr = alpha_rl = state.slip_angle

        # 각 타이어 슬립각 [FR, FL, RR, RL]
        slip_angles = [alpha_fr, alpha_fl, alpha_rr, alpha_rl]

        # 타이어별 횡방향 힘 계산
        lateral_forces = [DynamicModel.magic_formula(alpha, vehicle_config, terrain_friction) * load
                        for alpha, load in zip(slip_angles, vertical_loads)]

        # 종방향 힘은 단순화 (4륜 구동 가정, 구동력 균등 분배)
        drive_force = state.accel * vehicle_config.MASS
        longitudinal_forces = [drive_force/4 for _ in range(4)]

        return aero_drag, rolling_resist, lateral_forces, longitudinal_forces

    @staticmethod
    def update_g_forces(state, accel_cmd, dt, sim_config, vehicle_config):
        """
        물리 기반 G-force 계산 및 상태 업데이트
        원심력 기반 계산으로 현실적인 G-force 제공

        Args:
            state: 차량 상태 (VehicleState)
            accel_cmd: 가속 명령 (정규화된 가속/제동 입력)
            dt: 시간 간격 [s]
            sim_config: 시뮬레이션 설정 (SimConfig)
            vehicle_config: 차량 설정 (VehicleConfig)
        """
        # 종방향 G-force 계산
        # 실제 가속도에 기반한 계산 (경미한 증폭만 적용)
        long_g = -state.accel * 1.1 / sim_config.GRAVITY

        # 가속 및 제동 명령에 따른 가중치 부여
        max_accel = vehicle_config.MAX_ACCEL if accel_cmd > 0 else vehicle_config.MAX_BRAKE
        command_intensity = abs(accel_cmd) * max_accel / sim_config.GRAVITY

        # 실제 가속도와 명령 가중치 혼합 (체감 반응성 향상)
        effective_long_g = long_g * 0.4 - np.sign(accel_cmd) * command_intensity * 0.6

        # 물리적 한계 적용 (현실적인 범위로 제한)
        max_long_g = 1.5  # 고성능 전기차 기준
        effective_long_g = np.clip(effective_long_g, -max_long_g, max_long_g)

        # 횡방향 G-force 계산 (원심력만 고려)
        if abs(state.vel) > 1.0 and abs(state.steer) > 0.001:
            # 회전 반경 계산
            R = vehicle_config.WHEELBASE / np.tan(abs(state.steer))

            # 원심 가속도: a = v²/r
            centripetal_accel = (state.vel ** 2) / R
            lat_g = np.sign(state.steer) * centripetal_accel / sim_config.GRAVITY

            # 물리적 한계 적용 (현실적인 범위로 제한)
            max_lat_g = 1.3  # 고성능 스포츠카 기준
            effective_lat_g = min(abs(lat_g), max_lat_g) * np.sign(lat_g)

        else:
            effective_lat_g = 0

        # 필터링 적용 (부드러운 변화 및 반응성)
        # 기본 필터 계수 계산
        base_filter = min(1.0, dt * 2.0)

        # 종방향 필터: 급격한 가속/제동 시 반응성 향상
        long_filter = base_filter
        if abs(accel_cmd) > 0.4:
            accel_factor = 1.0 + min(1.5, abs(accel_cmd) * 3.0)
            long_filter = min(1.0, base_filter * accel_factor)

        # 횡방향 필터: 급격한 조향 시 반응성 향상
        lat_filter = base_filter
        steer_intensity = abs(state.steer) / vehicle_config.MAX_STEER
        lateral_vel_intensity = min(1.0, abs(state.vel_lateral) / 15.0)

        # 가장 강한 효과 선택
        max_lat_intensity = max(steer_intensity, lateral_vel_intensity)
        if max_lat_intensity > 0.3:
            lat_factor = 1.0 + min(0.8, max_lat_intensity * 0.4)
            lat_filter = min(1.0, base_filter * lat_factor)

        # G-force 업데이트 (필터링 적용)
        state.g_forces[0] = state.g_forces[0] * (1.0 - long_filter) + effective_long_g * long_filter
        state.g_forces[1] = state.g_forces[1] * (1.0 - lat_filter) + effective_lat_g * lat_filter

    @staticmethod
    def calculate_slip(state, steer_cmd, dt, sim_config, lateral_accel):
        """
        # 슬립 동역학 모델링
        # 조향 입력과 속도에 따른 슬립 생성 및 회복
        # 종방향 속도 감소

        Args:
            state: 차량 상태 (VehicleState)
            steer_cmd: 조향 명령 [-1, 1]
            dt: 시간 간격 [s]
            sim_config: 시뮬레이션 설정 (SimConfig)
            lateral_accel: 횡방향 가속도 [m/s²]
        """

        # 기존 슬립 자연 감소
        terrain_slip_recovery = sim_config.TERRAIN_FRICTION.get(state.terrain_type, 1.0)  # 지형에 따른 회복 계수
        recovery_rate = 1.2 * terrain_slip_recovery  # 기본 회복률에 지형 효과 적용
        state.vel_lateral *= (1.0 - recovery_rate * dt)

        # 슬립 생성 (조향, 속도에 기반)
        if abs(state.vel) > 3.0:  # 최소 속도 이상에서만 슬립 계산
            # 슬립 계수 (속도와 조향에 비례)
            slip_generation_base = 0.8

            # 속도 효과 (고속일수록 더 많은 슬립)
            speed_ratio = min(1.0, abs(state.vel) / 40.0)

            # 조향 변화율 효과 (급격한 조향일수록 더 많은 슬립)
            steer_change_factor = min(0.7, abs(steer_cmd - state.steer) * 5.0)

            # 슬립 생성 강도
            slip_intensity = (
                slip_generation_base *
                abs(steer_cmd) *
                speed_ratio *
                abs(state.vel) *
                (1.0 + steer_change_factor) *
                dt
            )

            # 타이어 횡방향 힘 효과 (실제 타이어가 발생시키는 힘 기반)
            # 이 값에 기반한 추가 슬립은 조향 방향과 반대로 작용
            lateral_force_effect = np.sign(steer_cmd) * slip_intensity

            # 횡방향 가속도와 타이어 힘 기반 슬립 조합
            # 횡방향 가속도는 현재 상태의 실제 물리적 반응
            # 타이어 힘 효과는 조향 입력에 따른 예측 반응
            state.vel_lateral += lateral_accel * dt - lateral_force_effect
        else:
            # 저속에서는 단순히 횡방향 가속도만 적용 (안정적인 거동)
            state.vel_lateral += lateral_accel * dt

        # 슬립 비율에 따른 종방향 속도 감소 효과
        if abs(state.vel) > 0.5:  # 최소 속도 체크
            # 슬립 비율 계산 (횡방향 속도 / 종방향 속도)
            slip_magnitude = abs(state.vel_lateral) / max(abs(state.vel), 0.001)  # 0 나누기 방지

            # 슬립 크기에 따른 에너지 손실 계수 (슬립이 클수록 더 많은 에너지 손실)
            speed_loss_factor = min(0.3, slip_magnitude * 0.8)

            # 종방향 속도 직접 감소 (마찰로 인한 에너지 손실)
            state.vel *= (1.0 - speed_loss_factor * dt)

    @staticmethod
    def apply_forces(state, action, dt, sim_config, vehicle_config):
        """
        동역학 모델 기반 힘 적용

        Args:
            state: 차량 상태 (VehicleState)
            action: 제어 입력 [accel, steer]
            dt: 시간 간격 [s]
            sim_config: 시뮬레이션 설정 (SimConfig)
            vehicle_config: 차량 설정 (VehicleConfig)
        """
        # 서브스텝 적용 (수치적 안정성)
        substep_dt = dt / sim_config.PHYSICS_SUBSTEPS

        for _ in range(sim_config.PHYSICS_SUBSTEPS):
            # 입력 정규화
            accel_cmd = np.clip(action[0], -1, 1) * (
                vehicle_config.MAX_BRAKE if action[0] < 0 else vehicle_config.MAX_ACCEL
            )
            steer_cmd = np.clip(action[1], -1, 1) * vehicle_config.MAX_STEER

            # 현재 값에서 목표 조향각으로 점진적 변화 (급격한 변화 방지)
            max_steer_change = vehicle_config.MAX_STEER * 2.0 * substep_dt  # 초당 최대 조향각 변화율
            current_steer = state.steer
            target_steer = steer_cmd
            steer_diff = target_steer - current_steer
            steer_change = np.clip(steer_diff, -max_steer_change, max_steer_change)
            steer_cmd = current_steer + steer_change

            # 힘 계산
            aero_drag, rolling_resist, lateral_forces, longitudinal_forces = DynamicModel.calculate_forces(state, sim_config, vehicle_config)

            # 순 가속도 계산
            net_accel = accel_cmd - (aero_drag + rolling_resist)/vehicle_config.MASS

            # 타이어 횡방향 힘으로 인한 차량의 횡방향 가속도 계산
            lateral_accel = sum(lateral_forces) / vehicle_config.MASS

            # 슬립 모델링 (횡방향 가속도, 조향 입력, 시간 간격)
            DynamicModel.calculate_slip(state, steer_cmd, substep_dt, sim_config, lateral_accel)

            # 운동학 모델 업데이트 (위치, 방향 등 업데이트)
            KinematicModel.update(state, substep_dt, net_accel, steer_cmd, vehicle_config)

            # G-force 계산 및 업데이트
            DynamicModel.update_g_forces(state, action[0], substep_dt, sim_config, vehicle_config)
